"""End-to-end pipeline: fit on all labeled data, predict test, write submission.

Owned by Teammate B (T-010, updated in Phase 2 T-020/T-021/T-024).

Usage:
    python -m src.prediction
"""
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.augment import expand
from src.data_loader import (load_sample, load_submission_template, load_test,
                             techniques_to_multihot)
from src.evaluation.metrics import log_experiment, macro_f1_multiclass, macro_f1_techniques, s_hat
from src.features.tfidf_features import TfidfBlock, build_text_dict
from src.models.task_a import TaskAHead
from src.models.task_b import TaskBHead
from src.models.task_c import TaskCHead
from src.postprocess import apply_rules_batch, tune_task_b_thresholds
from src.submission import build
from src.training.cross_validation import cross_validate, leave_cluster_out_cv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

CHUNK = 50_000   # rows per prediction chunk to bound memory

# Promoted Phase 2 config (E015: word+char only, C=5, n_aug=5 Set A)
_AUG_LEVELS = ("light", "medium", "heavy")
_AUG_WEIGHTS = (1.0, 0.8, 0.5)
DEFAULT_N_AUG = 5
DEFAULT_C = 5.0


# ── Fitted-model container ────────────────────────────────────────────────────

class FittedModel:
    """Container holding trained components for a single model variant."""

    def __init__(self, block: TfidfBlock,
                 head_a: TaskAHead,
                 head_b: TaskBHead,
                 head_c: TaskCHead):
        self.block  = block
        self.head_a = head_a
        self.head_b = head_b
        self.head_c = head_c


def _expand_train(df_labeled: pd.DataFrame, n_aug: int, seed: int) -> tuple:
    """Expand training rows with Set A corruption; return (full_df, sample_weights)."""
    parts = [df_labeled.assign(weight=1.0)]
    for i in range(n_aug):
        lvl = _AUG_LEVELS[i % len(_AUG_LEVELS)]
        w = _AUG_WEIGHTS[i % len(_AUG_WEIGHTS)]
        parts.append(expand(df_labeled, k=1, seed=seed + 100 * (i + 1),
                            level=lvl, ops="A", weight=w))
    full_df = pd.concat(parts, ignore_index=True)
    return full_df, full_df["weight"].values.astype(float)


# ── Core API ──────────────────────────────────────────────────────────────────

def fit_all(df_labeled: pd.DataFrame,
            df_test: pd.DataFrame = None,
            seed: int = config.SEED,
            n_aug: int = DEFAULT_N_AUG,
            C: float = DEFAULT_C,
            task_b_thresholds: list = None) -> FittedModel:
    """Fit TF-IDF + all task heads on labeled data (+ Set A augmentation).

    df_labeled:        DataFrame with text, label, techniques, action
    df_test:           optional DataFrame with text — used only for unsupervised TF-IDF IDF
    n_aug:             Set A corruption variants per row (default 5)
    C:                 regularisation parameter (default 5.0)
    task_b_thresholds: optional 6 per-technique thresholds from OOF tuning
    Returns:           FittedModel
    """
    config.set_seed(seed)
    full_train, weights = _expand_train(df_labeled, n_aug=n_aug, seed=seed)

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

    head_a = TaskAHead(C=C, seed=seed).fit(X, y_a, sample_weight=weights)
    head_b = TaskBHead(C=C, seed=seed).fit(X, Y_b, sample_weight=weights)
    if task_b_thresholds is not None:
        head_b.set_thresholds(task_b_thresholds)
    head_c = TaskCHead(C=C, seed=seed).fit(X, y_c, sample_weight=weights)

    block.save(config.CACHE_DIR / "tfidf_block.pkl")
    log.info("fit_all: %d labeled rows (%d with aug), %s test rows",
             len(df_labeled), len(full_train),
             len(df_test) if df_test is not None else "none")
    return FittedModel(block, head_a, head_b, head_c)


def predict_df(df_text: pd.DataFrame, fitted: FittedModel) -> pd.DataFrame:
    """Predict on arbitrary new text; works on any input (empty, noise-only, conversations).

    df_text: DataFrame with 'id' and 'text' columns
    fitted:  FittedModel from fit_all()
    Returns: 5-column submission DataFrame
    """
    all_preds = []
    n = len(df_text)
    for start in range(0, n, CHUNK):
        chunk = df_text.iloc[start:start + CHUNK].copy()
        d     = build_text_dict(chunk)
        d["struct_feats"] = None
        X     = fitted.block.transform(d)

        proba_a = fitted.head_a.predict_proba(X)
        proba_b = fitted.head_b.predict_proba(X)
        proba_c = fitted.head_c.predict_proba(X)

        label  = proba_a.argmax(axis=1)
        tech   = fitted.head_b.predict(X)
        action = proba_c.argmax(axis=1)
        label, tech, action = apply_rules_batch(label, tech, action, proba_b, proba_c)
        conf   = np.clip(proba_a.max(axis=1), *config.CONF_CLIP)

        all_preds.append(build(chunk["id"], label, tech, action, conf))
        log.info("predict_df: %d / %d rows done", min(start + CHUNK, n), n)

    return pd.concat(all_preds, ignore_index=True)


# ── Cross-validation helper ───────────────────────────────────────────────────

class _CVModel:
    """Wrapper around TfidfBlock + TaskA/B/C heads for cross_validation.cross_validate."""

    def __init__(self, seed: int = config.SEED, n_aug: int = DEFAULT_N_AUG, C: float = DEFAULT_C):
        self.seed = seed
        self.n_aug = n_aug
        self.C = C

    def fit(self, train: pd.DataFrame) -> "_CVModel":
        full_train, weights = _expand_train(train, n_aug=self.n_aug, seed=self.seed)
        train_d = build_text_dict(full_train)
        train_d["struct_feats"] = None
        self.block = TfidfBlock(use_msg_blocks=False)
        X_tr = self.block.fit_transform(train_d)   # only train; test IDF not used in CV

        y_a = full_train["label"].values
        Y_b = techniques_to_multihot(full_train["techniques"])
        y_c = full_train["action"].values

        self.head_a = TaskAHead(C=self.C, seed=self.seed).fit(X_tr, y_a, sample_weight=weights)
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


def _cv_oof(df: pd.DataFrame, seed: int = config.SEED, exp_name: str = "E015"):
    """Run repeated grouped CV. Returns (raw_df, summary_df, oof_bundle)."""
    return cross_validate(
        df,
        lambda: _CVModel(seed=seed),
        n_splits=config.N_SPLITS,
        n_repeats=config.N_REPEATS,
        exp_name=exp_name,
        model_tag="m1",
        return_oof=True,
    )


# ── Main entrypoint ───────────────────────────────────────────────────────────

def main() -> str:
    """Fit on all labeled data, run CV, tune Task B thresholds, predict test, write submission."""
    t0 = time.time()
    df       = load_sample()
    test     = load_test()
    template = load_submission_template()

    log.info("Running %d-fold × %d-repeat grouped CV...",
             config.N_SPLITS, config.N_REPEATS)
    res, summary, oof_bundle = _cv_oof(df)
    log.info("CV summary:\n%s", summary.to_string())
    res.to_csv(config.VALIDATION_DIR / "baseline_cv.csv", index=False)

    # Tune Task B thresholds on OOF
    oof_clean = oof_bundle["clean"]
    Y_true = techniques_to_multihot(df["techniques"])
    folds_arr = oof_bundle["folds"]["fold_r0"].values
    thrs = tune_task_b_thresholds(oof_clean["proba_b"], Y_true, folds_arr)

    # Fit final model on all labeled data (+ Set A aug + test for unsupervised TF-IDF)
    log.info("Fitting final model on %d rows + %d unlabeled test rows...",
             len(df), len(test))
    fitted = fit_all(df, test, task_b_thresholds=thrs)

    # Predict test
    log.info("Predicting %d test rows...", len(test))
    sub = predict_df(test, fitted)

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
    return str(out_path)


if __name__ == "__main__":
    main()
