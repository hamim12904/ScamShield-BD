"""Stage 4: baseline + first calibrated submission.   python -m src.training.train_baseline"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from src import config
from src.augment import corrupt as augment_corrupt
from src.data_loader import (load_sample, load_submission_template, load_test,
                             multihot_to_techniques, techniques_to_multihot)
from src.evaluation.calibration import apply as apply_confidence
from src.evaluation.calibration import fit_confidence
from src.preprocessing import BANGLA, URL_RE, build_views, preprocess
from src.training.cross_validation import assign_groups, cross_validate, leave_cluster_out_cv

EXPERIMENTS_PATH = config.VALIDATION_DIR / "experiments.csv"

# Bilingual (English + Banglish + Bangla Unicode) semantic cues
_PHONE_RE = re.compile(r"(?:\+?880|01[3-9])\s*[\dXx-]{6,}")
_USSD_RE = re.compile(r"\*\d+(?:\*\d+)*#")
_MONEY_RE = re.compile(
    r"(?:\btk\.?|\btaka\b|\bbdt\b|৳|\b\d+(?:,\d+)*\s*k\b|টাকা|লক্ষ|হাজার|ক্যাশ\s*আউট)",
    re.I,
)
_CRED_RE = re.compile(
    r"(?:\b(?:pin|otp|cvv|password|passcode|nid|kyc|verification\s*code)\b"
    r"|(?:current|bkash|nagad|rocket|mcash)(?:pin|otp|kyc)"
    r"|পিন|ওটিপি|পাসওয়ার্ড|পাসওয়ার্ড|ভেরিফিকেশন|কোড|এনআইডি)",
    re.I,
)
_URGENCY_RE = re.compile(
    r"(?:\b(?:urgent|immediately|expire|blocked|suspend|within\s*\d+|today|sesh|ekhoni|joruri)\b"
    r"|জরুরি|জরুরী|ব্লক|বন্ধ|আজই|এখনই|আটকে|বাতিল|শেষ)",
    re.I,
)
_REWARD_RE = re.compile(
    r"(?:\b(?:congrat|won|winner|prize|lottery|lucky\s*draw|bonus|jitech|jitun|cashback|free)\b"
    r"|অভিনন্দন|জিতেছেন|জিতুন|ক্যাশব্যাক|পুরস্কার|বোনাস|ফ্রি|লটারি|অফার|ছাড়|ছাড়)",
    re.I,
)
_BANGLA_RE = re.compile(rf"[{BANGLA}]")


def _extract_cues(norm_texts, n_messages_arr, emoji_counts, noise_ratios, bangla_ratios):
    """Domain + structure cues (including emoji_count and noise_ratio) complementing TF-IDF."""
    n = len(norm_texts)
    feats = np.zeros((n, 12), dtype=np.float32)
    for i, t in enumerate(norm_texts):
        if t:
            feats[i, 0] = 1.0 if (("." in t or "://" in t) and URL_RE.search(t)) else 0.0
            feats[i, 1] = 1.0 if _PHONE_RE.search(t) else 0.0
            feats[i, 2] = 1.0 if _USSD_RE.search(t) else 0.0
            feats[i, 3] = 1.0 if _MONEY_RE.search(t) else 0.0
            feats[i, 4] = 1.0 if _CRED_RE.search(t) else 0.0
            feats[i, 5] = 1.0 if _URGENCY_RE.search(t) else 0.0
            feats[i, 6] = 1.0 if _REWARD_RE.search(t) else 0.0
            feats[i, 7] = 1.0 if _BANGLA_RE.search(t) else 0.0
        feats[i, 8] = 1.0 if n_messages_arr[i] > 1 else 0.0
        feats[i, 9] = min(float(emoji_counts[i]), 5.0) / 5.0
        feats[i, 10] = float(noise_ratios[i])
        feats[i, 11] = float(bangla_ratios[i])
    return csr_matrix(feats * 0.35)


class _ConstBinaryClf:
    """Fallback when a Task B label has only one class in a training split."""
    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, X):
        p1 = np.full(X.shape[0], self.value, dtype=np.float64)
        return np.column_stack([1.0 - p1, p1])


class Baseline:
    def __init__(self, n_aug=3, c_a=3.0, c_b=2.0, c_c=2.0, tech_threshold=0.50, seed=config.SEED):
        self.n_aug = n_aug
        self.c_a = c_a
        self.c_b = c_b
        self.c_c = c_c
        self.tech_threshold = tech_threshold
        self.seed = seed

    def _transform_first_last(self, firsts, lasts):
        """Reuse X_first on single-turn rows (first == last) to avoid redundant TF-IDF transforms."""
        X_first = self.v_turn.transform(firsts)
        diff_idx = [i for i, (f, l) in enumerate(zip(firsts, lasts)) if f != l]
        if not diff_idx:
            return X_first, X_first
        X_diff = self.v_turn.transform([lasts[i] for i in diff_idx])
        row_map = np.arange(len(firsts), dtype=np.int64)
        for pos, orig_i in enumerate(diff_idx):
            row_map[orig_i] = len(firsts) + pos
        X_last = vstack([X_first, X_diff], format="csr")[row_map]
        return X_first, X_last

    def _build_text_features(self, skel, norm, firsts, lasts, n_msgs, emojis, noise_ratios, bangla_ratios):
        X_first, X_last = self._transform_first_last(firsts, lasts)
        return hstack([
            self.vc.transform(skel),
            self.vw.transform([s.lower() for s in norm]),
            self.vp.transform(norm),
            X_first,
            X_last,
            _extract_cues(norm, n_msgs, emojis, noise_ratios, bangla_ratios),
        ]).tocsr()

    def _fit_task_a(self, X, y):
        clf = LogisticRegression(C=self.c_a, max_iter=3000, class_weight="balanced")
        clf.fit(X, y)
        return clf

    def _proba_full(self, clf, X, n_classes):
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        proba[:, clf.classes_] = clf.predict_proba(X)
        return proba

    def _fit_task_b(self, X, Y_tech):
        clfs = []
        for j in range(Y_tech.shape[1]):
            col = Y_tech[:, j]
            if col.sum() == 0:
                clfs.append(_ConstBinaryClf(0.0))
            elif col.sum() == len(col):
                clfs.append(_ConstBinaryClf(1.0))
            else:
                clf = LogisticRegression(C=self.c_b, max_iter=2000, class_weight="balanced")
                clf.fit(X, col)
                clfs.append(clf)
        return clfs

    def _predict_task_b_proba(self, clfs, X):
        return np.column_stack([c.predict_proba(X)[:, 1] for c in clfs])

    def fit(self, train):
        rng = random.Random(self.seed)
        train = train.reset_index(drop=True)
        n_orig = len(train)

        y_a_real = train["label"].to_numpy(dtype=int)
        y_c_real = train["action"].to_numpy(dtype=int)
        y_b_real = techniques_to_multihot(train["techniques"]).astype(int)
        self.n_tech_ = y_b_real.shape[1]

        self.action_by_label = train.groupby("label")["action"].agg(lambda s: int(s.mode().iloc[0])).to_dict()
        self.default_action_ = int(train["action"].mode().iloc[0])

        groups_orig = (
            train["group"].to_numpy()
            if "group" in train.columns
            else assign_groups(train).to_numpy()
        )

        # Training augmentation uses Set A ONLY (held-out Set B is reserved for evaluation)
        texts = list(train["text"])
        parent_idx = list(range(n_orig))
        for i, t in enumerate(train["text"]):
            for _ in range(self.n_aug):
                texts.append(augment_corrupt(t, rng, level="medium", ops="A"))
                parent_idx.append(i)
        parent_idx = np.array(parent_idx, dtype=int)

        prep = [preprocess(t) for t in texts]
        skel = [p["skel"] for p in prep]
        norm = [p["preserved"] for p in prep]
        firsts = [p["first_msg"].lower() for p in prep]
        lasts = [p["last_msg"].lower() for p in prep]
        n_msgs = np.array([p["n_messages"] for p in prep], dtype=np.int32)
        emojis = np.array([p["emoji_count"] for p in prep], dtype=np.int32)
        noise_ratios = np.array([p["noise_ratio"] for p in prep], dtype=np.float32)
        bangla_ratios = np.array([p["script_bangla_ratio"] for p in prep], dtype=np.float32)

        min_df = 2 if self.n_aug >= 1 else 1
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=min_df)
        self.vw = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            token_pattern=r"(?u)[\w\u0980-\u09FF]+", min_df=min_df,
        )
        self.vp = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 3), lowercase=False,
            sublinear_tf=True, min_df=min_df, max_features=15_000,
        )
        self.v_turn = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
            min_df=min_df, max_features=15_000,
        )

        self.vc.fit(skel)
        self.vw.fit([s.lower() for s in norm])
        self.vp.fit(norm)
        self.v_turn.fit(firsts + [l for f, l in zip(firsts, lasts) if f != l])

        X_all = self._build_text_features(skel, norm, firsts, lasts, n_msgs, emojis, noise_ratios, bangla_ratios)
        y_a_all = y_a_real[parent_idx]
        y_b_all = y_b_real[parent_idx]
        y_c_all = y_c_real[parent_idx]

        # 1) Task A classifier
        self.clf_a = self._fit_task_a(X_all, y_a_all)

        # 2) Leak-free inner-OOF grouping by source group
        min_cls_grp = int(
            pd.DataFrame({"y": y_a_real, "g": groups_orig}).groupby("y")["g"].nunique().min()
        )
        if min_cls_grp >= 2:
            splitter = StratifiedGroupKFold(n_splits=min(5, min_cls_grp), shuffle=True, random_state=self.seed)
            splits = list(splitter.split(np.zeros(n_orig), y_a_real, groups_orig))
        else:
            min_cnt = int(pd.Series(y_a_real).value_counts().min())
            if min_cnt >= 2:
                splitter = StratifiedKFold(n_splits=min(5, min_cnt), shuffle=True, random_state=self.seed)
                splits = list(splitter.split(np.zeros(n_orig), y_a_real))
            else:
                splitter = KFold(n_splits=min(3, n_orig), shuffle=True, random_state=self.seed)
                splits = list(splitter.split(np.zeros(n_orig)))

        oof_p_a = np.zeros((len(texts), config.NUM_CLASSES), dtype=np.float64)
        oof_p_b = np.zeros((len(texts), self.n_tech_), dtype=np.float64)

        for tr_orig, va_orig in splits:
            tr_mask = np.isin(parent_idx, tr_orig)
            va_mask = np.isin(parent_idx, va_orig)
            clf_a_fold = self._fit_task_a(X_all[tr_mask], y_a_all[tr_mask])
            oof_p_a[va_mask] = self._proba_full(clf_a_fold, X_all[va_mask], config.NUM_CLASSES)

        X_for_b = hstack([X_all, csr_matrix(oof_p_a)]).tocsr()
        self.clf_b = self._fit_task_b(X_for_b, y_b_all)

        for tr_orig, va_orig in splits:
            tr_mask = np.isin(parent_idx, tr_orig)
            va_mask = np.isin(parent_idx, va_orig)
            clf_b_fold = self._fit_task_b(X_for_b[tr_mask], y_b_all[tr_mask])
            oof_p_b[va_mask] = self._predict_task_b_proba(clf_b_fold, X_for_b[va_mask])

        X_for_c = hstack([X_all, csr_matrix(oof_p_a), csr_matrix(oof_p_b)]).tocsr()
        self.clf_c = LogisticRegression(C=self.c_c, max_iter=3000, class_weight="balanced")
        self.clf_c.fit(X_for_c, y_c_all)
        return self

    def _predict_batch(self, skel, norm, firsts, lasts, n_msgs, emojis, noise_ratios, bangla_ratios, noise):
        X_text = self._build_text_features(skel, norm, firsts, lasts, n_msgs, emojis, noise_ratios, bangla_ratios)
        proba_a = self._proba_full(self.clf_a, X_text, config.NUM_CLASSES)

        X_for_b = hstack([X_text, csr_matrix(proba_a)]).tocsr()
        proba_b = self._predict_task_b_proba(self.clf_b, X_for_b)
        tech = (proba_b >= self.tech_threshold).astype(int)

        X_for_c = hstack([X_text, csr_matrix(proba_a), csr_matrix(proba_b)]).tocsr()
        proba_c = self._proba_full(self.clf_c, X_for_c, config.NUM_ACTIONS)
        action = proba_c.argmax(1)

        if noise.any():
            other_p = 0.40 / max(config.NUM_CLASSES - 1, 1)
            noise_vec = np.full(config.NUM_CLASSES, other_p, dtype=np.float64)
            noise_vec[0] = 0.60
            proba_a[noise] = noise_vec
            proba_b[noise] = 0.0
            tech[noise] = 0
            action[noise] = self.action_by_label.get(0, self.default_action_)

        label = proba_a.argmax(1)
        return label, proba_a, action, tech, proba_b, proba_c

    def predict(self, df, preprocessed=False, batch_size=50_000):
        n = len(df)
        req = {
            "skel", "preserved", "first_msg", "last_msg",
            "n_messages", "emoji_count", "noise_ratio", "script_bangla_ratio", "noise_only",
        }
        if preprocessed and req.issubset(df.columns):
            skel_all = df["skel"].fillna("").tolist()
            norm_all = df["preserved"].fillna("").tolist()
            first_all = df["first_msg"].fillna("").str.lower().tolist()
            last_all = df["last_msg"].fillna("").str.lower().tolist()
            n_msgs_all = df["n_messages"].to_numpy(dtype=np.int32)
            emoji_all = df["emoji_count"].to_numpy(dtype=np.int32)
            nratio_all = df["noise_ratio"].to_numpy(dtype=np.float32)
            bnratio_all = df["script_bangla_ratio"].to_numpy(dtype=np.float32)
            noise_all = df["noise_only"].to_numpy(dtype=bool)
        else:
            prep = [preprocess(t) for t in df["text"]]
            skel_all = [p["skel"] for p in prep]
            norm_all = [p["preserved"] for p in prep]
            first_all = [p["first_msg"].lower() for p in prep]
            last_all = [p["last_msg"].lower() for p in prep]
            n_msgs_all = np.array([p["n_messages"] for p in prep], dtype=np.int32)
            emoji_all = np.array([p["emoji_count"] for p in prep], dtype=np.int32)
            nratio_all = np.array([p["noise_ratio"] for p in prep], dtype=np.float32)
            bnratio_all = np.array([p["script_bangla_ratio"] for p in prep], dtype=np.float32)
            noise_all = np.array([p["noise_only"] for p in prep], dtype=bool)

        if n <= batch_size:
            lb, pr, ac, tc, pb, pc = self._predict_batch(
                skel_all, norm_all, first_all, last_all, n_msgs_all, emoji_all, nratio_all, bnratio_all, noise_all
            )
            return {"label": lb, "proba": pr, "action": ac, "tech": tc, "proba_b": pb, "proba_c": pc}

        uniq_map = {}
        u_idx = []
        inv_idx = np.empty(n, dtype=np.int64)
        for i, key in enumerate(zip(skel_all, norm_all, n_msgs_all.tolist(), noise_all.tolist())):
            idx = uniq_map.get(key)
            if idx is None:
                idx = len(u_idx)
                uniq_map[key] = idx
                u_idx.append(i)
            inv_idx[i] = idx

        u_n = len(u_idx)
        print(f"  predicting {u_n:,} unique cleaned rows (from {n:,} total)...", flush=True)
        u_skel = [skel_all[i] for i in u_idx]
        u_norm = [norm_all[i] for i in u_idx]
        u_first = [first_all[i] for i in u_idx]
        u_last = [last_all[i] for i in u_idx]
        u_msgs = n_msgs_all[u_idx]
        u_emoji = emoji_all[u_idx]
        u_nratio = nratio_all[u_idx]
        u_bnratio = bnratio_all[u_idx]
        u_noise = noise_all[u_idx]

        u_labels = np.empty(u_n, dtype=np.int64)
        u_probas = np.empty((u_n, config.NUM_CLASSES), dtype=np.float64)
        u_actions = np.empty(u_n, dtype=np.int64)
        u_techs = np.empty((u_n, self.n_tech_), dtype=np.int64)
        u_pb = np.empty((u_n, self.n_tech_), dtype=np.float64)
        u_pc = np.empty((u_n, config.NUM_ACTIONS), dtype=np.float64)

        for start in range(0, u_n, batch_size):
            end = min(start + batch_size, u_n)
            lb, pr, ac, tc, pb, pc = self._predict_batch(
                u_skel[start:end], u_norm[start:end], u_first[start:end], u_last[start:end],
                u_msgs[start:end], u_emoji[start:end], u_nratio[start:end], u_bnratio[start:end],
                u_noise[start:end],
            )
            u_labels[start:end] = lb
            u_probas[start:end] = pr
            u_actions[start:end] = ac
            u_techs[start:end] = tc
            u_pb[start:end] = pb
            u_pc[start:end] = pc
            print(f"  predicted {end:,}/{u_n:,} unique rows...", flush=True)

        return {
            "label": u_labels[inv_idx],
            "proba": u_probas[inv_idx],
            "action": u_actions[inv_idx],
            "tech": u_techs[inv_idx],
            "proba_b": u_pb[inv_idx],
            "proba_c": u_pc[inv_idx],
        }


def _preprocessing_fingerprint():
    src_file = Path(__file__).resolve().parents[1] / "preprocessing.py"
    return hashlib.md5(src_file.read_bytes()).hexdigest()[:10]


def load_or_build_preprocessed_test(test_df):
    cache_dir = config.VALIDATION_DIR.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"test_prep_{_preprocessing_fingerprint()}.pkl"

    if cache_path.exists():
        print(f"Loading cached preprocessed test set: {cache_path.name}", flush=True)
        return pd.read_pickle(cache_path)

    t0 = time.perf_counter()
    print(f"Preprocessing {len(test_df):,} test rows (cached once per preprocessing.py version)...", flush=True)
    test_prep = build_views(test_df[["id", "text"]], log_every=100_000)
    keep_cols = [
        "id", "clean", "skel", "preserved", "first_msg", "last_msg",
        "n_messages", "emoji_count", "noise_ratio", "script_bangla_ratio",
        "noise_only", "n_noise_tokens", "template_key",
    ]
    test_prep = test_prep[keep_cols]
    test_prep.to_pickle(cache_path)
    print(f"Done in {time.perf_counter() - t0:.1f}s -> saved {cache_path.name}", flush=True)
    return test_prep


def _append_experiment_row(row, path=EXPERIMENTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def main(exp_name="e001"):
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    res, summary, extra = cross_validate(
        df, Baseline, n_repeats=config.N_REPEATS, exp_name=exp_name, model_tag="m1", return_oof=True
    )
    print(summary)
    slice_means = extra["slices"].groupby("view").mean(numeric_only=True).drop(columns=["rep"])
    print("\nSlice F_A (mean across repeats):")
    print(slice_means.round(3))

    lco_scores, _ = leave_cluster_out_cv(df, Baseline, saved_folds=extra["folds"])
    print("\nLeave-cluster-out proxy:", {k: round(v, 3) for k, v in lco_scores.items()})

    res.to_csv(config.VALIDATION_DIR / "baseline_cv.csv", index=False)

    # Fit cross-fitted confidence calibrator on repeat-averaged OOF Task A probabilities
    sample_views = build_views(df)
    calibrator = fit_confidence(
        extra["clean"]["proba"],
        df["label"].to_numpy(),
        extra_feats={"noise_ratio": sample_views["noise_ratio"], "noise_only": sample_views["noise_only"]},
        folds=extra["folds"]["fold_r0"].to_numpy(),
    )
    print(
        f"\nCalibration (OOF): brier={calibrator['oof_brier']:.4f} "
        f"(const baseline={calibrator['const_brier']:.4f}, 1-brier={1.0 - calibrator['oof_brier']:.4f})"
    )

    clean_m = res[res["view"] == "clean"][["F_A", "F_B", "F_C", "brier"]].mean()
    corr_m = res[res["view"] == "corrupted"][["F_A", "F_B", "F_C", "brier"]].mean()
    exp_row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exp_name": exp_name,
        "model_tag": "m1",
        "F_A": round(float(clean_m["F_A"]), 4),
        "F_B": round(float(clean_m["F_B"]), 4),
        "F_C": round(float(clean_m["F_C"]), 4),
        "brier": round(float(clean_m["brier"]), 4),
        "cal_oof_brier": round(float(calibrator["oof_brier"]), 4),
        "corrupted_F_A": round(float(corr_m["F_A"]), 4),
        "lco_F_A": round(float(lco_scores["F_A"]), 4),
    }
    if "clean" in slice_means.index:
        for col, val in slice_means.loc["clean"].items():
            exp_row[col] = round(float(val), 4)
    _append_experiment_row(exp_row)
    print(f"Logged experiment row to {EXPERIMENTS_PATH}")

    model = Baseline().fit(df.assign(group=extra["folds"]["group"].values))
    test, template = load_test(), load_submission_template()
    test_prep = load_or_build_preprocessed_test(test)

    t0 = time.perf_counter()
    pred = model.predict(test_prep, preprocessed=True, batch_size=50_000)
    print(f"Inference finished in {time.perf_counter() - t0:.1f}s", flush=True)

    conf = apply_confidence(
        calibrator,
        pred["proba"],
        extra_feats={"noise_ratio": test_prep["noise_ratio"], "noise_only": test_prep["noise_only"]},
    )

    sub = pd.DataFrame({
        "id": test["id"],
        "label": pred["label"].astype(int),
        "techniques": [multihot_to_techniques(r) for r in pred["tech"]],
        "action": pred["action"].astype(int),
        "confidence": np.round(conf, 4),
    })

    # Stage 9 sanity checks
    assert (sub["id"].values == template["id"].values).all() and len(sub) == len(template)
    assert not sub.isna().any().any()
    assert sub["label"].between(0, config.NUM_CLASSES - 1).all()
    assert sub["action"].between(0, config.NUM_ACTIONS - 1).all()
    assert sub["confidence"].between(0.0, 1.0).all()

    sub.to_csv(config.PREDICTION_DIR / "submission_full.csv", index=False)
    sub[["id", "label"]].to_csv(config.PREDICTION_DIR / "submission.csv", index=False)
    print("\npredicted class mix:\n", sub["label"].value_counts(normalize=True).sort_index().round(3))
    print("\npredicted action mix:\n", sub["action"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    main()