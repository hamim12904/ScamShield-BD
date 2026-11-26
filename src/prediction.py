"""End-to-end pipeline: fit on all labeled data, predict test, write submission.

Owned by Teammate B (T-010).

Usage:
    python -m src.prediction
"""
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.data_loader import (load_sample, load_submission_template, load_test,
                              techniques_to_multihot)
from src.features.tfidf_features import TfidfBlock, build_text_dict
from src.models.task_a import TaskAHead
from src.models.task_b import TaskBHead
from src.models.task_c import TaskCHead
from src.submission import build
from src.training.cross_validation import cross_validate

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

CHUNK = 50_000   # rows per prediction chunk to bound memory


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


# ── Core API ──────────────────────────────────────────────────────────────────

def fit_all(df_labeled: pd.DataFrame,
            df_test: pd.DataFrame = None,
            seed: int = config.SEED) -> FittedModel:
    """Fit TF-IDF + all task heads on labeled data.

    df_labeled: DataFrame with text, label, techniques, action
    df_test:    optional DataFrame with text — used only for unsupervised TF-IDF IDF
                (declared in MEMORY.md §11 and notebook)
    Returns:    FittedModel
    """
    config.set_seed(seed)

    train_d = build_text_dict(df_labeled)
    test_d  = build_text_dict(df_test) if df_test is not None else None

    block = TfidfBlock()
    X     = block.fit_transform(train_d, test_d)

    y_a = df_labeled["label"].values
    Y_b = techniques_to_multihot(df_labeled["techniques"])
    y_c = df_labeled["action"].values

    head_a = TaskAHead(seed=seed).fit(X, y_a)
    head_b = TaskBHead(seed=seed).fit(X, Y_b)
    head_c = TaskCHead(seed=seed).fit(X, y_c)

    block.save(config.CACHE_DIR / "tfidf_block.pkl")
    log.info("fit_all: %d labeled rows, %s test rows",
             len(df_labeled), len(df_test) if df_test is not None else "none")
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
        X     = fitted.block.transform(d)

        proba_a = fitted.head_a.predict_proba(X)
        label   = proba_a.argmax(axis=1)
        tech    = fitted.head_b.predict(X)
        action  = fitted.head_c.predict(X)
        conf    = np.clip(proba_a.max(axis=1), *config.CONF_CLIP)

        all_preds.append(build(chunk["id"], label, tech, action, conf))
        log.info("predict_df: %d / %d rows done", min(start + CHUNK, n), n)

    return pd.concat(all_preds, ignore_index=True)


# ── Cross-validation helper ───────────────────────────────────────────────────

class _CVModel:
    """Wrapper around TfidfBlock + TaskA/B/C heads for cross_validation.cross_validate."""

    def __init__(self, seed: int = config.SEED):
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "_CVModel":
        train_d = build_text_dict(train)
        self.block = TfidfBlock()
        X_tr = self.block.fit_transform(train_d)   # only train; test IDF not used in CV

        y_a = train["label"].values
        Y_b = techniques_to_multihot(train["techniques"])
        y_c = train["action"].values

        self.head_a = TaskAHead(seed=self.seed).fit(X_tr, y_a)
        self.head_b = TaskBHead(seed=self.seed).fit(X_tr, Y_b)
        self.head_c = TaskCHead(seed=self.seed).fit(X_tr, y_c)
        return self

    def predict(self, valid: pd.DataFrame) -> dict:
        valid_d = build_text_dict(valid)
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


def _cv_oof(df: pd.DataFrame, seed: int = config.SEED, exp_name: str = "E001"):
    """Run repeated grouped CV. Returns (raw_df, summary_df)."""
    return cross_validate(
        df,
        lambda: _CVModel(seed=seed),
        n_splits=config.N_SPLITS,
        n_repeats=config.N_REPEATS,
        exp_name=exp_name,
        model_tag="m1",
    )


# ── Main entrypoint ───────────────────────────────────────────────────────────

def main() -> str:
    """Fit on all labeled data, run CV, predict test, validate, write submission.

    Returns the path of the written submission CSV.
    """
    t0 = time.time()
    df       = load_sample()
    test     = load_test()
    template = load_submission_template()

    # Grouped CV (uses provisional group=id until A delivers folds.json)
    log.info("Running %d-fold × %d-repeat grouped CV (provisional folds)...",
             config.N_SPLITS, config.N_REPEATS)
    res, summary = _cv_oof(df)
    log.info("CV summary:\n%s", summary.to_string())
    res.to_csv(config.VALIDATION_DIR / "baseline_cv.csv", index=False)

    # Log experiment
    try:
        from src.evaluation.metrics import log_experiment
        clean_row = summary.loc["clean"]
        log_experiment({
            "exp_id":       "E001",
            "config_json":  "baseline_tfidf_lr_C5",
            "F_A":          float(clean_row[("F_A",    "mean")]),
            "F_B":          float(clean_row[("F_B",    "mean")]),
            "F_C":          float(clean_row[("F_C",    "mean")]),
            "Cal":          float(1 - clean_row[("brier", "mean")]),
            "notes":        "baseline TF-IDF+LR, provisional group=id folds, no augmentation",
        })
    except Exception as exc:
        log.warning("Could not log experiment: %s", exc)

    # Fit final model on all labeled data (+ test for unsupervised TF-IDF)
    log.info("Fitting final model on %d rows + %d unlabeled test rows...",
             len(df), len(test))
    fitted = fit_all(df, test)

    # Predict test
    log.info("Predicting %d test rows...", len(test))
    sub = predict_df(test, fitted)

    # Validate alignment with submission template
    if list(sub["id"].values) != list(template["id"].values):
        raise RuntimeError("Submission IDs do not match the template order!")
    if len(sub) != len(template):
        raise RuntimeError(
            f"Row count mismatch: {len(sub)} (ours) vs {len(template)} (template)"
        )

    # Write submission
    exp_id   = datetime.utcnow().strftime("E%m%d_%H%M")
    out_path = config.PREDICTION_DIR / f"submission_{exp_id}.csv"
    sub.to_csv(out_path, index=False)
    elapsed  = (time.time() - t0) / 60
    log.info("Done → %s (%.1f min)", out_path, elapsed)
    log.info("Predicted class mix:\n%s",
             sub["label"].value_counts(normalize=True).sort_index().round(3))
    return str(out_path)


if __name__ == "__main__":
    main()
