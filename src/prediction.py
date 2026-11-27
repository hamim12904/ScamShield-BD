"""End-to-end pipeline: fit on all labeled data, predict test, write submission.

Owned by Teammate B (T-010, Phase 2 E015, Phase 3 T-030/T-031 E021 M1+M2 ensemble).

Usage:
    python -m src.prediction
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src import config
from src.augment import expand
from src.data_loader import (load_sample, load_submission_template, load_test,
                             techniques_to_multihot)
from src.evaluation.calibration import apply as apply_calibration, fit_confidence
from src.evaluation.metrics import (log_experiment, macro_f1_multiclass,
                                    macro_f1_techniques, s_hat)
from src.features.tfidf_features import TfidfBlock, build_text_dict
from src.models.ensemble import blend, model_disagreement, tune_blend_weight
from src.models.task_a import TaskAHead
from src.models.task_b import TaskBHead
from src.models.task_c import TaskCHead
from src.postprocess import apply_rules_batch, tune_task_b_thresholds
from src.preprocessing import preprocess
from src.submission import build
from src.training.cross_validation import cross_validate, leave_cluster_out_cv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

CHUNK = 50_000   # rows per prediction chunk to bound memory

# Promoted config: word+char TF-IDF, Set A n_aug=5, LR+SVM+M2 blend, calibrated
_AUG_LEVELS = ("light", "medium", "heavy")
_AUG_WEIGHTS = (1.0, 0.8, 0.5)
DEFAULT_N_AUG = 5
DEFAULT_C = 5.0
DEFAULT_SVM_C = 1.0
DEFAULT_BLEND_W = 0.4   # 0.4 * LR + 0.6 * LinearSVC


# ── Fitted-model container ────────────────────────────────────────────────────

class FittedModel:
    """Container holding trained components for prediction."""

    def __init__(self, block: TfidfBlock,
                 head_a: TaskAHead,
                 head_b: TaskBHead,
                 head_c: TaskCHead,
                 head_a_svm: TaskAHead = None,
                 blend_w: float = DEFAULT_BLEND_W,
                 calibrator: dict = None):
        self.block = block
        self.head_a = head_a
        self.head_a_svm = head_a_svm
        self.head_b = head_b
        self.head_c = head_c
        self.blend_w = blend_w
        self.calibrator = calibrator


def _expand_train(df_labeled: pd.DataFrame, n_aug: int, seed: int,
                  extra_df: pd.DataFrame = None, extra_weight: float = 0.3,
                  aug_extra: bool = True) -> tuple:
    """Expand training rows with Set A corruption; return (full_df, sample_weights)."""
    parts = [df_labeled.assign(weight=1.0)]
    for i in range(n_aug):
        lvl = _AUG_LEVELS[i % len(_AUG_LEVELS)]
        w = _AUG_WEIGHTS[i % len(_AUG_WEIGHTS)]
        parts.append(expand(df_labeled, k=1, seed=seed + 100 * (i + 1),
                            level=lvl, ops="A", weight=w))
    if extra_df is not None and not extra_df.empty:
        extra_base = extra_df.assign(weight=float(extra_weight))
        parts.append(extra_base)
        if aug_extra and n_aug > 0:
            for i in range(n_aug):
                lvl = _AUG_LEVELS[i % len(_AUG_LEVELS)]
                w = _AUG_WEIGHTS[i % len(_AUG_WEIGHTS)] * float(extra_weight) * 0.5
                parts.append(expand(extra_df, k=1, seed=seed + 200 * (i + 1),
                                    level=lvl, ops="A", weight=w))
    full_df = pd.concat(parts, ignore_index=True)
    return full_df, full_df["weight"].values.astype(float)


def _noise_stats(texts) -> tuple:
    """Return (noise_ratio (n,), noise_only (n,)) arrays matching cross_validation.py."""
    ratios, only = [], []
    for t in texts:
        p = preprocess(t)
        n_tok = max(len(str(t).split()), 1)
        ratios.append(min(1.0, float(p["n_noise_tokens"]) / n_tok))
        only.append(bool(p["noise_only"]))
    return np.array(ratios, dtype=float), np.array(only, dtype=bool)


# ── Core API ──────────────────────────────────────────────────────────────────

def fit_all(df_labeled: pd.DataFrame,
            df_test: pd.DataFrame = None,
            seed: int = config.SEED,
            n_aug: int = DEFAULT_N_AUG,
            C: float = DEFAULT_C,
            svm_C: float = DEFAULT_SVM_C,
            blend_w: float = DEFAULT_BLEND_W,
            task_b_thresholds: list = None,
            calibrator: dict = None,
            extra_df: pd.DataFrame = None,
            extra_weight: float = 0.3,
            aug_extra: bool = True) -> FittedModel:
    """Fit TF-IDF + all task heads on labeled data (+ Set A augmentation).

    df_labeled:        DataFrame with text, label, techniques, action
    df_test:           optional DataFrame with text — used only for unsupervised TF-IDF IDF
    extra_df:          optional DataFrame with external/LLM labeled rows
    """
    config.set_seed(seed)
    full_train, weights = _expand_train(df_labeled, n_aug=n_aug, seed=seed,
                                        extra_df=extra_df, extra_weight=extra_weight,
                                        aug_extra=aug_extra)

    train_d = build_text_dict(full_train)
    train_d["struct_feats"] = None
    test_d = build_text_dict(df_test) if df_test is not None else None
    if test_d is not None:
        test_d["struct_feats"] = None

    block = TfidfBlock(use_msg_blocks=False)
    X     = block.fit_transform(train_d, test_d)

    y_a = full_train["label"].values
    Y_b = techniques_to_multihot(full_train["techniques"])
    y_c = full_train["action"].values

    head_a     = TaskAHead(C=C, model_type="lr", seed=seed).fit(X, y_a, sample_weight=weights)
    head_a_svm = TaskAHead(C=svm_C, model_type="svm", seed=seed).fit(X, y_a, sample_weight=weights)
    head_b     = TaskBHead(C=C, seed=seed).fit(X, Y_b, sample_weight=weights)
    if task_b_thresholds is not None:
        head_b.set_thresholds(task_b_thresholds)
    head_c = TaskCHead(C=C, seed=seed).fit(X, y_c, sample_weight=weights)

    block.save(config.CACHE_DIR / "tfidf_block.pkl")
    log.info("fit_all: %d labeled rows (%d with aug), %s test rows",
             len(df_labeled), len(full_train),
             len(df_test) if df_test is not None else "none")
    return FittedModel(block, head_a, head_b, head_c,
                       head_a_svm=head_a_svm, blend_w=blend_w, calibrator=calibrator)


def predict_df(df_text: pd.DataFrame,
               fitted: FittedModel,
               m2_test: dict = None,
               w_m1_a: float = 0.7,
               w_m1_b: float = 0.3,
               w_m1_c: float = 0.4) -> pd.DataFrame:
    """Predict on arbitrary new text; works on any input (empty, noise-only, conversations).

    df_text: DataFrame with 'id' and 'text' columns
    fitted:  FittedModel from fit_all()
    m2_test: optional dict/npz with Model 2 test arrays ('A', 'B', 'C', 'ood_sim')
    Returns: 5-column submission DataFrame
    """
    all_preds = []
    n = len(df_text)
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        chunk = df_text.iloc[start:end].copy()
        d     = build_text_dict(chunk)
        d["struct_feats"] = None
        X     = fitted.block.transform(d)

        proba_lr = fitted.head_a.predict_proba(X)
        if fitted.head_a_svm is not None:
            proba_svm = fitted.head_a_svm.predict_proba(X)
            m1_a = blend([proba_lr, proba_svm], weights=[fitted.blend_w, 1.0 - fitted.blend_w])
            dis = model_disagreement(proba_lr, proba_svm)
        else:
            m1_a = proba_lr
            dis = None

        m1_b = fitted.head_b.predict_proba(X)
        m1_c = fitted.head_c.predict_proba(X)

        if m2_test is not None:
            m2_a = np.asarray(m2_test["A"][start:end], dtype=float)
            m2_b = np.asarray(m2_test["B"][start:end], dtype=float)
            m2_c = np.asarray(m2_test["C"][start:end], dtype=float)
            proba_a = blend([m1_a, m2_a], weights=[w_m1_a, 1.0 - w_m1_a])
            proba_b = blend([m1_b, m2_b], weights=[w_m1_b, 1.0 - w_m1_b])
            proba_c = blend([m1_c, m2_c], weights=[w_m1_c, 1.0 - w_m1_c])
            dis = model_disagreement(m1_a, m2_a)
            ood = np.asarray(m2_test["ood_sim"][start:end], dtype=float)
        else:
            proba_a, proba_b, proba_c = m1_a, m1_b, m1_c
            ood = None

        label  = proba_a.argmax(axis=1)
        thr    = np.array(fitted.head_b.thresholds, dtype=float)
        tech   = (proba_b >= thr).astype(int)
        action = proba_c.argmax(axis=1)
        label, tech, action = apply_rules_batch(label, tech, action, proba_b, proba_c)

        if fitted.calibrator is not None:
            nr, no = _noise_stats(chunk["text"])
            cols = [nr]
            if dis is not None:
                cols.append(dis)
            if ood is not None:
                cols.append(ood)
            extra = np.column_stack(cols)
            conf = apply_calibration(fitted.calibrator, proba_a, extra_feats=extra, noise_only=no)
        else:
            conf = np.clip(proba_a.max(axis=1), *config.CONF_CLIP)

        all_preds.append(build(chunk["id"], label, tech, action, conf))
        log.info("predict_df: %d / %d rows done", end, n)

    return pd.concat(all_preds, ignore_index=True)


# ── Cross-validation helper ───────────────────────────────────────────────────

class _CVModel:
    """Wrapper around TfidfBlock + TaskA/B/C heads for cross_validation.cross_validate."""

    def __init__(self, seed: int = config.SEED,
                 n_aug: int = DEFAULT_N_AUG,
                 C: float = DEFAULT_C,
                 model_type: str = "lr",
                 extra_df: pd.DataFrame = None,
                 extra_weight: float = 0.3,
                 aug_extra: bool = True,
                 all_real_ids: Sequence[int] = None):
        self.seed = seed
        self.n_aug = n_aug
        self.C = C
        self.model_type = model_type
        self.extra_df = extra_df
        self.extra_weight = extra_weight
        self.aug_extra = aug_extra
        self.all_real_ids = all_real_ids

    def fit(self, train: pd.DataFrame) -> "_CVModel":
        safe_extras = None
        if self.extra_df is not None and not self.extra_df.empty:
            if self.all_real_ids is not None:
                val_ids = set(self.all_real_ids) - set(train["id"])
            else:
                val_ids = set()

            if val_ids and "near_real_ids" in self.extra_df.columns:
                def _safe(row):
                    near = row.get("near_real_ids")
                    if near is None or (isinstance(near, float) and np.isnan(near)):
                        return True
                    return not (set(near) & val_ids)

                safe_extras = self.extra_df[self.extra_df.apply(_safe, axis=1)].copy()
            else:
                safe_extras = self.extra_df.copy()

        full_train, weights = _expand_train(
            train, n_aug=self.n_aug, seed=self.seed,
            extra_df=safe_extras, extra_weight=self.extra_weight, aug_extra=self.aug_extra
        )
        train_d = build_text_dict(full_train)
        train_d["struct_feats"] = None
        self.block = TfidfBlock(use_msg_blocks=False)
        X_tr = self.block.fit_transform(train_d)   # only train; test IDF not used in CV

        y_a = full_train["label"].values
        Y_b = techniques_to_multihot(full_train["techniques"])
        y_c = full_train["action"].values

        self.head_a = TaskAHead(C=self.C, model_type=self.model_type, seed=self.seed).fit(
            X_tr, y_a, sample_weight=weights
        )
        self.head_b = TaskBHead(C=self.C, seed=self.seed).fit(X_tr, Y_b, sample_weight=weights)
        self.head_c = TaskCHead(C=self.C, seed=self.seed).fit(X_tr, y_c, sample_weight=weights)
        return self

    def predict(self, valid: pd.DataFrame) -> dict:
        valid_d = build_text_dict(valid)
        valid_d["struct_feats"] = None
        X_va = self.block.transform(valid_d)
        proba_a = self.head_a.predict_proba(X_va)
        proba_b = self.head_b.predict_proba(X_va)
        proba_c = self.head_c.predict_proba(X_va)
        return {
            "proba":   proba_a,
            "proba_b": proba_b,
            "proba_c": proba_c,
            "label":   proba_a.argmax(1),
            "tech":    self.head_b.predict(X_va),
            "action":  proba_c.argmax(1),
        }


def _cv_oof(df: pd.DataFrame, seed: int = config.SEED,
            exp_name: str = "E015", C: float = DEFAULT_C, model_type: str = "lr",
            extra_df: pd.DataFrame = None, extra_weight: float = 0.25,
            aug_extra: bool = True, all_real_ids: Sequence[int] = None):
    """Run repeated grouped CV. Returns (raw_df, summary_df, oof_bundle)."""
    return cross_validate(
        df,
        lambda: _CVModel(
            seed=seed,
            C=C,
            model_type=model_type,
            extra_df=extra_df,
            extra_weight=extra_weight,
            aug_extra=aug_extra,
            all_real_ids=all_real_ids,
        ),
        n_splits=config.N_SPLITS,
        n_repeats=config.N_REPEATS,
        exp_name=exp_name,
        model_tag="m1",
        return_oof=True,
    )


# ── Main entrypoint ───────────────────────────────────────────────────────────

def main(
    exp_lr: str = "E030",
    exp_svm: str = "E031",
    m2_exp: Optional[str] = "E020",
    ens_exp: str = "E032",
    use_extra: bool = True,
    extra_paths: Optional[Sequence[Path]] = None,
    llm_path: Optional[Path] = None,
    pseudo_path: Optional[Path] = None,
    include_handwritten: bool = False,
    require_unpadded_m2: bool = True,
    test_limit: Optional[int] = None,
) -> str:
    """Fit on all labeled data, blend M1+M2 if available, calibrate, predict test, write submission."""
    t0 = time.time()
    df       = load_sample()
    test     = load_test()
    if test_limit is not None and test_limit < len(test):
        test = test.head(test_limit).copy()
    template = load_submission_template()
    if len(template) != len(test):
        template = template.head(len(test)).copy()
    all_real_ids = df["id"].values.tolist()

    real_tech_count = techniques_to_multihot(df["techniques"]).sum(axis=1)
    log.info("Real rows label x technique-count x action==0:\n%s",
             pd.crosstab(df["label"], real_tech_count,
                         values=df["action"].eq(0), aggfunc="mean").round(3))

    # 1. Load promoted extra data (LLM + pseudo-labels) if enabled
    processed_extra = None
    if use_extra:
        from src.pseudo_label import load_pseudo_labels
        from src.training.train_llm_aug import (apply_class_cap,
                                               compute_near_real_ids,
                                               load_extra)

        parts = []
        if extra_paths:
            for p in extra_paths:
                parts.append(load_extra(Path(p), default_weight=0.25))
        else:
            llm_p = Path(llm_path) if llm_path is not None else (config.ROOT / "data" / "extra" / "llm_labels_v1.jsonl")
            pseudo_p = Path(pseudo_path) if pseudo_path is not None else (config.ROOT / "data" / "extra" / "pseudo_labels_v1.jsonl")
            if llm_p.exists():
                parts.append(load_extra(llm_p, default_weight=0.30))
            if pseudo_p.exists():
                parts.append(load_pseudo_labels(pseudo_p, weight=0.20))
            if include_handwritten:
                from data.extra.handwritten import load_handwritten
                hw = load_handwritten(weight=0.40)[["id", "text", "label", "techniques", "action", "source", "weight"]]
                parts.append(hw)

        if parts:
            comb = pd.concat(parts, ignore_index=True)
            capped = apply_class_cap(comb, max_per_class=400, seed=config.SEED)
            processed_extra = compute_near_real_ids(capped, df)
            near_count = (processed_extra["near_real_ids"].apply(len) > 0).sum()
            log.info("Loaded %d extra rows for training (%d with near-real matches)",
                     len(processed_extra), near_count)

            filtered_llm = processed_extra[
                processed_extra["source"].astype(str).eq("llm")
                & processed_extra["near_real_ids"].apply(lambda ids: len(ids) == 0)
            ]
            if not filtered_llm.empty:
                llm_tech_count = techniques_to_multihot(filtered_llm["techniques"]).sum(axis=1)
                log.info("Filtered LLM rows label x technique-count x action==0:\n%s",
                         pd.crosstab(filtered_llm["label"], llm_tech_count,
                                     values=filtered_llm["action"].eq(0), aggfunc="mean").round(3))

    log.info("Running %d-fold × %d-repeat grouped CV (LR: %s & LinearSVC: %s)...",
             config.N_SPLITS, config.N_REPEATS, exp_lr, exp_svm)
    res_lr, sum_lr, oof_lr = _cv_oof(
        df, exp_name=exp_lr, C=DEFAULT_C, model_type="lr",
        extra_df=processed_extra, extra_weight=0.25, aug_extra=True, all_real_ids=all_real_ids
    )
    _, sum_svm, oof_svm    = _cv_oof(
        df, exp_name=exp_svm, C=DEFAULT_SVM_C, model_type="svm",
        extra_df=processed_extra, extra_weight=0.25, aug_extra=True, all_real_ids=all_real_ids
    )
    res_lr.to_csv(config.VALIDATION_DIR / f"{exp_lr}_cv.csv", index=False)

    y_a = df["label"].values
    Y_true = techniques_to_multihot(df["techniques"])
    y_c = df["action"].values
    folds_arr = oof_lr["folds"]["fold_r0"].values

    p_lr, p_svm = oof_lr["clean"]["proba"], oof_svm["clean"]["proba"]
    w_lr, fa_m1, m1_a = tune_blend_weight(p_lr, p_svm, y_a, macro_f1_multiclass)
    log.info("LR/SVM OOF blend weight: w_lr=%.2f, F_A=%.4f", w_lr, fa_m1)
    m1_b = oof_lr["clean"]["proba_b"]
    m1_c = oof_lr["clean"]["proba_c"]

    m2_oof_path = (config.OOF_DIR / f"{m2_exp}_m2.npz") if m2_exp else None
    m2_test_path = (config.PREDICTION_DIR / f"{m2_exp}_m2_test.npz") if m2_exp else None
    nr, no = _noise_stats(df["text"])

    m2_test = None
    if m2_oof_path and m2_test_path and m2_oof_path.exists() and m2_test_path.exists():
        log.info("Loading Model 2 OOF & test probabilities (%s_m2)...", m2_exp)
        m2_oof = np.load(m2_oof_path, allow_pickle=True)
        m2_test_data = np.load(m2_test_path, allow_pickle=True)
        assert np.array_equal(m2_oof["ids"], df["id"].to_numpy(dtype=np.int64))
        assert np.array_equal(m2_test_data["ids"], test["id"].to_numpy(dtype=np.int64))

        # Check padding on Model 2 test arrays
        A_test = m2_test_data["A"]
        n_padded = int((A_test == A_test[-1]).all(axis=1).sum())
        if n_padded > 1000:
            msg = (
                f"Model 2 test arrays in {m2_test_path.name} are padded ({n_padded} identical rows)! "
                "To generate full unpadded embeddings on Kaggle GPU, run:\n"
                "    python -m src.training.train_embeddings --full-test"
            )
            if require_unpadded_m2:
                log.error(msg)
                raise RuntimeError(msg)
            else:
                log.warning("%s\nProceeding with M1 (LR+SVM) test predictions to prevent class skew.", msg)
                m2_test = None
        else:
            m2_test = m2_test_data

        w_m1_a, fa_ens, ens_a = tune_blend_weight(m1_a, m2_oof["A"], y_a, macro_f1_multiclass)
        w_m1_b, _, ens_b = tune_blend_weight(
            m1_b, m2_oof["B"], Y_true, macro_f1_techniques,
            grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), is_multilabel=True,
        )
        w_m1_c, _, ens_c = tune_blend_weight(
            m1_c, m2_oof["C"], y_c, macro_f1_multiclass,
            grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        dis = model_disagreement(m1_a, m2_oof["A"])
        extra = np.column_stack([nr, dis, m2_oof["ood_sim"]])

        np.savez_compressed(
            config.OOF_DIR / f"{ens_exp}_ens.npz",
            ids=m2_oof["ids"], repeat=m2_oof["repeat"],
            A=ens_a, B=ens_b, C=ens_c,
        )
        log.info("M1+M2 Ensemble OOF F_A=%.4f (w_m1_a=%.2f, w_m1_b=%.2f, w_m1_c=%.2f)",
                 fa_ens, w_m1_a, w_m1_b, w_m1_c)
    else:
        log.warning("Model 2 files not found or disabled; using M1 (LR+SVM) only.")
        ens_a, ens_b, ens_c = m1_a, m1_b, m1_c
        w_m1_a = w_m1_b = w_m1_c = 1.0
        dis = model_disagreement(p_lr, p_svm)
        extra = np.column_stack([nr, dis])

    thrs = tune_task_b_thresholds(ens_b, Y_true, folds_arr)
    nested_scores = []
    for f in np.unique(folds_arr):
        tr = folds_arr != f
        va = ~tr
        fold_thrs = np.asarray(tune_task_b_thresholds(ens_b[tr], Y_true[tr], folds_arr[tr]), dtype=float)
        pred_col = (ens_b[va] >= fold_thrs).astype(int)
        nested_scores.append(macro_f1_techniques(Y_true[va], pred_col))
    nested_fb = float(np.mean(nested_scores))
    log.info("Task B thresholds: %s (nested cross-fitted F_B estimate=%.4f)", thrs, nested_fb)

    cal = fit_confidence(ens_a, y_a, extra_feats=extra, folds=folds_arr, noise_only=no)

    # Fit final model on all labeled data (+ Set A aug + extra rows; F-07 train-only IDF)
    log.info("Fitting final model on %d labeled rows (+ %s extras)...",
             len(df), len(processed_extra) if processed_extra is not None else 0)
    fitted = fit_all(
        df,
        df_test=None,
        blend_w=w_lr,
        task_b_thresholds=thrs,
        calibrator=cal,
        extra_df=processed_extra,
        extra_weight=0.25,
        aug_extra=True,
    )

    # Predict test
    log.info("Predicting %d test rows...", len(test))
    sub = predict_df(test, fitted, m2_test=m2_test, w_m1_a=w_m1_a, w_m1_b=w_m1_b, w_m1_c=w_m1_c)

    if list(sub["id"].values) != list(template["id"].values):
        raise RuntimeError("Submission IDs do not match the template order!")
    if len(sub) != len(template):
        raise RuntimeError(
            f"Row count mismatch: {len(sub)} (ours) vs {len(template)} (template)"
        )

    exp_id   = datetime.now(timezone.utc).strftime("E%m%d_%H%M")
    out_path = config.PREDICTION_DIR / f"submission_{exp_id}.csv"
    sub.to_csv(out_path, index=False)
    elapsed  = (time.time() - t0) / 60
    log.info("Done → %s (%.1f min)", out_path, elapsed)
    log.info("Predicted class mix:\n%s",
             sub["label"].value_counts(normalize=True).sort_index().round(3))
    log.info("Predicted action mix:\n%s",
             sub["action"].value_counts(normalize=True).sort_index().round(3))
    return str(out_path)


if __name__ == "__main__":
    main()
