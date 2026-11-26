"""Stage 4: baseline + first submission.   python -m src.training.train_baseline"""
import hashlib
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from src import config
from src.data_loader import (load_sample, load_submission_template, load_test,
                             multihot_to_techniques, techniques_to_multihot)
from src.preprocessing import BANGLA, URL_RE, preprocess, preprocess_frame
from src.training.cross_validation import corrupt_text, cross_validate

# Lightweight content-semantic cues (intentionally excludes corruption flags to avoid synthetic bias)
_PHONE_RE = re.compile(r"(?:\+?880|01[3-9])\s*[\dXx-]{6,}")
_USSD_RE = re.compile(r"\*\d+(?:\*\d+)*#")
_MONEY_RE = re.compile(r"(?:\btk\.?|\btaka\b|\bbdt\b|৳|\b\d+(?:,\d+)*\s*k\b)", re.I)
_CRED_RE = re.compile(r"\b(?:pin|otp|cvv|password|passcode|nid|kyc|verification\s*code)\b", re.I)
_URGENCY_RE = re.compile(r"\b(?:urgent|immediately|expire|blocked|suspend|within\s*\d+|today|sesh|ekhoni|joruri)\b", re.I)
_REWARD_RE = re.compile(r"\b(?:congrat|won|winner|prize|lottery|lucky\s*draw|bonus|jitech|jitun|cashback|free)\b", re.I)
_BANGLA_RE = re.compile(rf"[{BANGLA}]")


def _extract_cues(clean_texts, n_messages_arr):
    """Domain cues complementing TF-IDF for Tasks A, B, and C."""
    n = len(clean_texts)
    feats = np.zeros((n, 9), dtype=np.float32)
    for i, t in enumerate(clean_texts):
        if not t:
            continue
        feats[i, 0] = 1.0 if (("." in t or "://" in t) and URL_RE.search(t)) else 0.0
        feats[i, 1] = 1.0 if _PHONE_RE.search(t) else 0.0
        feats[i, 2] = 1.0 if _USSD_RE.search(t) else 0.0
        feats[i, 3] = 1.0 if _MONEY_RE.search(t) else 0.0
        feats[i, 4] = 1.0 if _CRED_RE.search(t) else 0.0
        feats[i, 5] = 1.0 if _URGENCY_RE.search(t) else 0.0
        feats[i, 6] = 1.0 if _REWARD_RE.search(t) else 0.0
        feats[i, 7] = 1.0 if _BANGLA_RE.search(t) else 0.0
        feats[i, 8] = 1.0 if n_messages_arr[i] > 1 else 0.0
    # Modest weight so sparse TF-IDF stays primary
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

    def _build_text_features(self, clean_lower, n_messages_arr):
        return hstack([
            self.vc.transform(clean_lower),
            self.vw.transform(clean_lower),
            _extract_cues(clean_lower, n_messages_arr),
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

        # Fallbacks for noise-only rows or unseen classes
        self.action_by_label = train.groupby("label")["action"].agg(lambda s: int(s.mode().iloc[0])).to_dict()
        self.default_action_ = int(train["action"].mode().iloc[0])

        # Build augmented training corpus (clean + corrupted copies, keeping parent index for leak-free stacking)
        texts = list(train["text"])
        parent_idx = list(range(n_orig))
        for i, t in enumerate(train["text"]):
            for _ in range(self.n_aug):
                texts.append(corrupt_text(t, rng))
                parent_idx.append(i)
        parent_idx = np.array(parent_idx, dtype=int)

        prep = [preprocess(t) for t in texts]
        clean_lower = [p["clean"].lower() for p in prep]
        n_msgs = np.array([p["n_messages"] for p in prep], dtype=np.int32)

        # min_df=2 filters single-occurrence random typos while retaining every term from real rows (each appears 1+n_aug times)
        min_df = 2 if self.n_aug >= 1 else 1
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=min_df)
        self.vw = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b", min_df=min_df,
        )
        self.vc.fit(clean_lower)
        self.vw.fit(clean_lower)

        X_all = self._build_text_features(clean_lower, n_msgs)
        y_a_all = y_a_real[parent_idx]
        y_b_all = y_b_real[parent_idx]
        y_c_all = y_c_real[parent_idx]

        # 1) Task A classifier on all training rows
        self.clf_a = self._fit_task_a(X_all, y_a_all)

        # 2) Leak-free inner-OOF Task A & Task B probabilities on original rows for stacking into Tasks B and C
        min_cnt = int(pd.Series(y_a_real).value_counts().min())
        if min_cnt >= 2:
            splitter = StratifiedKFold(n_splits=min(5, min_cnt), shuffle=True, random_state=self.seed)
            splits = list(splitter.split(np.zeros(n_orig), y_a_real))
        else:
            splitter = KFold(n_splits=min(3, n_orig), shuffle=True, random_state=self.seed)
            splits = list(splitter.split(np.zeros(n_orig)))

        oof_p_a = np.zeros((n_orig, config.NUM_CLASSES), dtype=np.float64)
        oof_p_b = np.zeros((n_orig, self.n_tech_), dtype=np.float64)

        for tr_orig, va_orig in splits:
            tr_mask = np.isin(parent_idx, tr_orig)
            va_orig_arr = np.asarray(va_orig)

            clf_a_fold = self._fit_task_a(X_all[tr_mask], y_a_all[tr_mask])
            oof_p_a[va_orig_arr] = self._proba_full(clf_a_fold, X_all[va_orig_arr], config.NUM_CLASSES)

        # Stack OOF P(A) onto every row (augmented copies inherit their parent's OOF P(A))
        X_for_b = hstack([X_all, csr_matrix(oof_p_a[parent_idx])]).tocsr()
        self.clf_b = self._fit_task_b(X_for_b, y_b_all)

        for tr_orig, va_orig in splits:
            tr_mask = np.isin(parent_idx, tr_orig)
            va_orig_arr = np.asarray(va_orig)

            clf_b_fold = self._fit_task_b(X_for_b[tr_mask], y_b_all[tr_mask])
            oof_p_b[va_orig_arr] = self._predict_task_b_proba(clf_b_fold, X_for_b[va_orig_arr])

        # 3) Task C classifier stacked with OOF P(A) and OOF P(B)
        X_for_c = hstack([
            X_all,
            csr_matrix(oof_p_a[parent_idx]),
            csr_matrix(oof_p_b[parent_idx]),
        ]).tocsr()
        self.clf_c = LogisticRegression(C=self.c_c, max_iter=3000, class_weight="balanced")
        self.clf_c.fit(X_for_c, y_c_all)
        return self

    def _predict_batch(self, clean_lower, n_msgs, noise):
        X_text = self._build_text_features(clean_lower, n_msgs)
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
            tech[noise] = 0
            action[noise] = self.action_by_label.get(0, self.default_action_)

        label = proba_a.argmax(1)
        return label, proba_a, action, tech

    def predict(self, df, preprocessed=False, batch_size=50_000):
        n = len(df)
        if preprocessed and {"clean", "n_messages", "noise_only"}.issubset(df.columns):
            clean_all = df["clean"].fillna("").str.lower().tolist()
            n_msgs_all = df["n_messages"].to_numpy(dtype=np.int32)
            noise_all = df["noise_only"].to_numpy(dtype=bool)
        else:
            prep = [preprocess(t) for t in df["text"]]
            clean_all = [p["clean"].lower() for p in prep]
            n_msgs_all = np.array([p["n_messages"] for p in prep], dtype=np.int32)
            noise_all = np.array([p["noise_only"] for p in prep], dtype=bool)

        if n <= batch_size:
            label, proba, action, tech = self._predict_batch(clean_all, n_msgs_all, noise_all)
            return {"label": label, "proba": proba, "action": action, "tech": tech}

        labels = np.empty(n, dtype=np.int64)
        probas = np.empty((n, config.NUM_CLASSES), dtype=np.float64)
        actions = np.empty(n, dtype=np.int64)
        techs = np.empty((n, self.n_tech_), dtype=np.int64)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            lb, pr, ac, tc = self._predict_batch(
                clean_all[start:end], n_msgs_all[start:end], noise_all[start:end]
            )
            labels[start:end] = lb
            probas[start:end] = pr
            actions[start:end] = ac
            techs[start:end] = tc
            print(f"  predicted {end:,}/{n:,} rows...", flush=True)

        return {"label": labels, "proba": probas, "action": actions, "tech": techs}


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
    test_prep = preprocess_frame(test_df[["id", "text"]], log_every=100_000)
    keep_cols = ["id", "clean", "n_messages", "noise_only", "n_noise_tokens", "template_key"]
    test_prep = test_prep[keep_cols]
    test_prep.to_pickle(cache_path)
    print(f"Done in {time.perf_counter() - t0:.1f}s -> saved {cache_path.name}", flush=True)
    return test_prep


def main():
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    res, summary = cross_validate(df, lambda tr, va: Baseline().fit(tr).predict(va), n_repeats=3)
    print(summary)
    res.to_csv(config.VALIDATION_DIR / "baseline_cv.csv", index=False)

    model = Baseline().fit(df)
    test, template = load_test(), load_submission_template()
    test_prep = load_or_build_preprocessed_test(test)

    t0 = time.perf_counter()
    pred = model.predict(test_prep, preprocessed=True, batch_size=50_000)
    print(f"Inference finished in {time.perf_counter() - t0:.1f}s", flush=True)

    sub = pd.DataFrame({
        "id": test["id"],
        "label": pred["label"].astype(int),
        "techniques": [multihot_to_techniques(r) for r in pred["tech"]],
        "action": pred["action"].astype(int),
        "confidence": np.clip(pred["proba"].max(1), 0.30, 0.98).round(4),
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