"""Train Model 2: dense multilingual sentence embeddings + Logistic Regression heads.

Owned by Teammate C (T-023).

Run from the project root:
    python -m src.training.train_embeddings [--exp-id E020] [--test-limit 1000]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_loader import load_sample, load_test, techniques_to_multihot
from src.evaluation.metrics import log_experiment, score_all
from src.features.embedding_features import DEFAULT_MODEL_NAME, embed, max_cosine_to_train
from src.preprocessing import preprocess
from src.training.cross_validation import FOLDS_PATH, iter_saved_folds, load_or_create_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _align_proba(raw_proba: np.ndarray, classes: np.ndarray, n_classes: int = 6) -> np.ndarray:
    """Align sparse predict_proba output to fixed (n, n_classes) dimensions."""
    aligned = np.zeros((len(raw_proba), n_classes), dtype=np.float64)
    for col_idx, cls in enumerate(classes):
        if int(cls) < n_classes:
            aligned[:, int(cls)] = raw_proba[:, col_idx]
    return aligned


def cross_validate_embeddings(
    df: pd.DataFrame,
    emb_train: np.ndarray,
    folds_table: pd.DataFrame,
    C: float = 1.0,
    seed: int = config.SEED,
    n_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run 5x3 grouped CV on embeddings across Tasks A, B, and C.

    Returns repeat-averaged (n, 6) OOF probabilities for (A, B, C).
    """
    n = len(df)
    y_a = df["label"].to_numpy(dtype=np.int64)
    Y_b = techniques_to_multihot(df["techniques"]).astype(np.int64)
    y_c = df["action"].to_numpy(dtype=np.int64)

    repeat_cols = [c for c in folds_table.columns if c.startswith("fold_r")]
    if not repeat_cols:
        repeat_cols = ["fold_r0"]

    sum_p_a = np.zeros((n, n_classes), dtype=np.float64)
    sum_p_b = np.zeros((n, 6), dtype=np.float64)
    sum_p_c = np.zeros((n, n_classes), dtype=np.float64)

    for rep_idx, col in enumerate(repeat_cols):
        log.info("Cross-validating repeat %d (%s)...", rep_idx, col)
        rep_p_a = np.zeros((n, n_classes), dtype=np.float64)
        rep_p_b = np.zeros((n, 6), dtype=np.float64)
        rep_p_c = np.zeros((n, n_classes), dtype=np.float64)

        for tr, va in iter_saved_folds(folds_table, col=col):
            X_tr, X_va = emb_train[tr], emb_train[va]

            # Task A
            clf_a = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)
            clf_a.fit(X_tr, y_a[tr])
            rep_p_a[va] = _align_proba(clf_a.predict_proba(X_va), clf_a.classes_, n_classes=n_classes)

            # Task C
            clf_c = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)
            clf_c.fit(X_tr, y_c[tr])
            rep_p_c[va] = _align_proba(clf_c.predict_proba(X_va), clf_c.classes_, n_classes=n_classes)

            # Task B: 6 binary heads
            for j in range(6):
                y_tr_b = Y_b[tr, j]
                unique_y = np.unique(y_tr_b)
                if len(unique_y) < 2:
                    val = float(unique_y[0]) if len(unique_y) == 1 else 0.0
                    rep_p_b[va, j] = val
                else:
                    clf_b = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)
                    clf_b.fit(X_tr, y_tr_b)
                    if 1 in clf_b.classes_:
                        idx_1 = list(clf_b.classes_).index(1)
                        rep_p_b[va, j] = clf_b.predict_proba(X_va)[:, idx_1]
                    else:
                        rep_p_b[va, j] = 0.0

        sum_p_a += rep_p_a
        sum_p_b += rep_p_b
        sum_p_c += rep_p_c

    n_reps = len(repeat_cols)
    return sum_p_a / n_reps, sum_p_b / n_reps, sum_p_c / n_reps


def fit_all_heads(
    emb_train: np.ndarray,
    y_a: np.ndarray,
    Y_b: np.ndarray,
    y_c: np.ndarray,
    C: float = 1.0,
    seed: int = config.SEED,
) -> tuple[LogisticRegression, list[Optional[LogisticRegression]], LogisticRegression]:
    """Fit final estimators on all available training embeddings."""
    clf_a = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)
    clf_a.fit(emb_train, y_a)

    clf_c = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)
    clf_c.fit(emb_train, y_c)

    heads_b = []
    for j in range(6):
        y_b_j = Y_b[:, j]
        if len(np.unique(y_b_j)) < 2:
            heads_b.append(None)
        else:
            clf_b = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)
            clf_b.fit(emb_train, y_b_j)
            heads_b.append(clf_b)

    return clf_a, heads_b, clf_c


def predict_with_heads(
    emb: np.ndarray,
    clf_a: LogisticRegression,
    heads_b: list[Optional[LogisticRegression]],
    clf_c: LogisticRegression,
    n_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict probabilities for Tasks A, B, and C given embeddings."""
    p_a = _align_proba(clf_a.predict_proba(emb), clf_a.classes_, n_classes=n_classes)
    p_c = _align_proba(clf_c.predict_proba(emb), clf_c.classes_, n_classes=n_classes)

    p_b = np.zeros((len(emb), 6), dtype=np.float64)
    for j, head in enumerate(heads_b):
        if head is None:
            p_b[:, j] = 0.0
        else:
            if 1 in head.classes_:
                idx_1 = list(head.classes_).index(1)
                p_b[:, j] = head.predict_proba(emb)[:, idx_1]
            else:
                p_b[:, j] = 0.0

    return p_a, p_b, p_c


def run(
    exp_id: str = "E020",
    model_name: str = DEFAULT_MODEL_NAME,
    C: float = 1.0,
    test_limit: Optional[int] = None,
    batch_size: int = 128,
    max_len: int = 128,
) -> dict:
    """Execute Model 2 training, OOF generation, test prediction, and experiment logging."""
    log.info("=== Running Model 2 Pipeline (%s) with model=%s ===", exp_id, model_name)

    # 1. Load labeled sample data and canonical folds
    df = load_sample()
    n = len(df)
    folds_table = load_or_create_folds(df, path=FOLDS_PATH)
    log.info("Loaded %d labeled rows and folds table with columns: %s", n, list(folds_table.columns))

    # 2. Extract normalized text view and compute embeddings
    text_norm = [preprocess(t)["preserved"] for t in df["text"]]
    log.info("Extracting embeddings for %d training rows...", n)
    emb_train = embed(text_norm, model_name=model_name, batch=batch_size, max_len=max_len)
    log.info("Training embeddings shape: %s", emb_train.shape)

    # 3. Compute cross-fitted OOD similarity for training rows
    ood_sim_train = max_cosine_to_train(emb_train, emb_train, is_train=True)

    # 4. Run 5x3 grouped cross-validation
    oof_a, oof_b, oof_c = cross_validate_embeddings(df, emb_train, folds_table, C=C)

    # 5. Save OOF predictions to outputs/oof/<exp_id>_m2.npz
    oof_path = config.OOF_DIR / f"{exp_id}_m2.npz"
    oof_path.parent.mkdir(parents=True, exist_ok=True)

    y_a = df["label"].to_numpy(dtype=np.int64)
    Y_b = techniques_to_multihot(df["techniques"]).astype(np.int64)
    y_c = df["action"].to_numpy(dtype=np.int64)
    prep = [preprocess(t) for t in df["text"]]
    noise_only = np.array([p["noise_only"] for p in prep], dtype=bool)
    noise_ratio = np.array([p["noise_ratio"] for p in prep], dtype=np.float64)

    payload_oof = {
        "ids": df["id"].to_numpy(dtype=np.int64),
        "repeat": np.full(n, config.N_REPEATS, dtype=np.int64),
        "A": oof_a,
        "B": oof_b,
        "C": oof_c,
        "ood_sim": ood_sim_train,
        # Extended fields for full compatibility
        "y_a": y_a,
        "y_b": Y_b,
        "y_c": y_c,
        "noise_only": noise_only,
        "noise_ratio": noise_ratio,
        "clean_proba_a": oof_a,
        "clean_proba_b": oof_b,
        "clean_proba_c": oof_c,
        "folds": folds_table["fold_r0"].to_numpy(dtype=np.int64) if "fold_r0" in folds_table.columns else np.zeros(n, dtype=np.int64),
    }
    np.savez_compressed(oof_path, **payload_oof)
    log.info("Saved OOF predictions to %s", oof_path)

    # 6. Evaluate metrics & log experiment
    pred_label_a = oof_a.argmax(axis=1)
    pred_tech_b = (oof_b >= 0.5).astype(int)
    pred_action_c = oof_c.argmax(axis=1)
    conf = np.clip(oof_a.max(axis=1), *config.CONF_CLIP)

    metrics_res = score_all(
        y_true_A=y_a,
        y_pred_A=pred_label_a,
        Y_true_B=Y_b,
        Y_pred_B=pred_tech_b,
        y_true_C=y_c,
        y_pred_C=pred_action_c,
        conf=conf,
    )
    log.info("CV Metrics: F_A=%.4f, F_B=%.4f, F_C=%.4f, Cal=%.4f, S_hat=%.4f",
             metrics_res["F_A"], metrics_res["F_B"], metrics_res["F_C"], metrics_res["Cal"], metrics_res["S_hat"])

    exp_row = {
        "exp_id": exp_id,
        "timestamp": datetime.utcnow().isoformat(),
        "config_json": json.dumps({"model": model_name, "C": C, "batch": batch_size, "max_len": max_len}),
        "F_A": round(metrics_res["F_A"], 4),
        "F_B": round(metrics_res["F_B"], 4),
        "F_C": round(metrics_res["F_C"], 4),
        "Cal": round(metrics_res["Cal"], 4),
        "S_hat": round(metrics_res["S_hat"], 4),
        "notes": f"Model 2 {model_name} (T-023)",
    }
    log_experiment(exp_row)

    # 7. Fit final heads on all training data
    clf_a, heads_b, clf_c = fit_all_heads(emb_train, y_a, Y_b, y_c, C=C, seed=config.SEED)

    # 8. Predict on test set
    test_df = load_test()
    n_test_total = len(test_df)
    test_ids = test_df["id"].to_numpy(dtype=np.int64)
    log.info("Loaded test set with %d rows", n_test_total)

    test_out_path = config.PREDICTION_DIR / f"{exp_id}_m2_test.npz"
    test_out_path.parent.mkdir(parents=True, exist_ok=True)

    if test_limit is not None and test_limit < n_test_total:
        log.warning(
            "[DEV / CPU LIMIT] Computing real embeddings for first %d test rows; "
            "padding remaining %d rows with prior probabilities to maintain (500000, 6) shape.",
            test_limit, n_test_total - test_limit,
        )
        slice_test = test_df.iloc[:test_limit]
        slice_norm = [preprocess(t)["preserved"] for t in slice_test["text"]]
        emb_test_slice = embed(slice_norm, model_name=model_name, batch=batch_size, max_len=max_len)
        slice_p_a, slice_p_b, slice_p_c = predict_with_heads(emb_test_slice, clf_a, heads_b, clf_c)
        slice_ood = max_cosine_to_train(emb_test_slice, emb_train, is_train=False)

        # Pre-allocate full (500000, 6) and (500000,) arrays
        test_p_a = np.tile(oof_a.mean(axis=0), (n_test_total, 1))
        test_p_b = np.tile(oof_b.mean(axis=0), (n_test_total, 1))
        test_p_c = np.tile(oof_c.mean(axis=0), (n_test_total, 1))
        test_ood = np.full(n_test_total, float(np.median(ood_sim_train)), dtype=np.float32)

        test_p_a[:test_limit] = slice_p_a
        test_p_b[:test_limit] = slice_p_b
        test_p_c[:test_limit] = slice_p_c
        test_ood[:test_limit] = slice_ood
    else:
        log.info("Computing embeddings for all %d test rows...", n_test_total)
        test_norm = [preprocess(t)["preserved"] for t in test_df["text"]]
        emb_test = embed(test_norm, model_name=model_name, batch=batch_size, max_len=max_len)
        test_p_a, test_p_b, test_p_c = predict_with_heads(emb_test, clf_a, heads_b, clf_c)
        test_ood = max_cosine_to_train(emb_test, emb_train, is_train=False)

    np.savez_compressed(
        test_out_path,
        ids=test_ids,
        A=test_p_a.astype(np.float32),
        B=test_p_b.astype(np.float32),
        C=test_p_c.astype(np.float32),
        ood_sim=test_ood.astype(np.float32),
    )
    log.info("Saved test predictions to %s (A: %s, B: %s, C: %s, ood_sim: %s)",
             test_out_path, test_p_a.shape, test_p_b.shape, test_p_c.shape, test_ood.shape)

    return {
        "oof_path": str(oof_path),
        "test_out_path": str(test_out_path),
        "metrics": metrics_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Model 2 sentence embeddings + LR heads.")
    parser.add_argument("--exp-id", type=str, default="E020", help="Experiment ID (default: E020)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (default: 1.0)")
    parser.add_argument("--test-limit", type=int, default=1000, help="Test rows to embed on CPU dev mode (default: 1000; set None for full)")
    parser.add_argument("--full-test", action="store_true", help="Run full 500,000 test set embedding")
    args = parser.parse_args()

    limit = None if args.full_test else args.test_limit
    run(exp_id=args.exp_id, model_name=args.model_name, C=args.C, test_limit=limit)


if __name__ == "__main__":
    main()
