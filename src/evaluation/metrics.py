"""Evaluation metrics: macro-F1 A/B/C, Brier calibration, composite surrogate S_hat.

Owned by Teammate B (T-007).

Public API
----------
macro_f1_multiclass(y_true, y_pred, labels) -> float
macro_f1_techniques(Y_true, Y_pred) -> float
brier_calibration(conf, correct) -> float
s_hat(F_A, F_B, F_C, F_domain, F_robust, Cal, F_novel) -> float
score_all(...) -> dict
log_experiment(row) -> None
"""
import csv
import logging
import subprocess
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score

from src import config

log = logging.getLogger(__name__)


# ── Core metrics ──────────────────────────────────────────────────────────────

def macro_f1_multiclass(y_true, y_pred, labels=range(6)) -> float:
    """Macro-F1 for Task A or C. Classes absent from BOTH y_true and y_pred are skipped.

    Uses zero_division=0 so absent classes contribute 0 F1 (not error).
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    present = [l for l in list(labels) if l in y_true or l in y_pred]
    if not present:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def macro_f1_techniques(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Macro-F1 for Task B. Both arrays are (n, 6) binary matrices.

    F1 per technique averaged over T1–T6, computed over ALL rows.
    Legit rows (all-zero in Y_true) are negatives for every technique.
    """
    Y_true = np.asarray(Y_true, dtype=int)
    Y_pred = np.asarray(Y_pred, dtype=int)
    scores = [
        float(f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0))
        for j in range(Y_true.shape[1])
    ]
    return float(np.mean(scores))


def brier_calibration(conf, correct) -> float:
    """Calibration = 1 − Brier score. Higher is better (best = 1).

    conf:    confidence array (floats in [0, 1])
    correct: 1 when Task A prediction equals true label, else 0
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return float(1.0 - np.mean((conf - correct) ** 2))


def s_hat(F_A: float, F_B: float, F_C: float,
          F_domain: float, F_robust: float,
          Cal: float, F_novel: float) -> float:
    """Composite surrogate score per PRD §3.

    S = 0.40·F_A + 0.15·F_B + 0.10·F_C + 0.10·F_domain
        + 0.10·F_robust + 0.05·Cal + 0.10·F_novel
    """
    return (0.40 * F_A + 0.15 * F_B + 0.10 * F_C +
            0.10 * F_domain + 0.10 * F_robust +
            0.05 * Cal + 0.10 * F_novel)


# ── Aggregate scorer ──────────────────────────────────────────────────────────

def score_all(y_true_A, y_pred_A,
              Y_true_B, Y_pred_B,
              y_true_C, y_pred_C,
              conf,
              slice_masks: dict = None,
              F_domain: float = None,
              F_robust: float = None,
              F_novel: float = None) -> dict:
    """Compute all metrics. Returns dict with F_A, F_B, F_C, Cal, S_hat, per-slice F_A.

    slice_masks: dict[name -> bool array of length n]
    F_domain, F_robust, F_novel: proxy values; default to F_A when None.
    """
    y_true_A = np.asarray(y_true_A)
    y_pred_A = np.asarray(y_pred_A)
    Y_true_B = np.asarray(Y_true_B, dtype=int)
    Y_pred_B = np.asarray(Y_pred_B, dtype=int)
    y_true_C = np.asarray(y_true_C)
    y_pred_C = np.asarray(y_pred_C)
    correct = (y_pred_A == y_true_A).astype(float)

    F_A = macro_f1_multiclass(y_true_A, y_pred_A)
    F_B = macro_f1_techniques(Y_true_B, Y_pred_B)
    F_C = macro_f1_multiclass(y_true_C, y_pred_C)
    Cal = brier_calibration(conf, correct)

    _domain = F_domain if F_domain is not None else F_A
    _robust = F_robust if F_robust is not None else F_A
    _novel  = F_novel  if F_novel  is not None else _domain
    S = s_hat(F_A, F_B, F_C, _domain, _robust, Cal, _novel)

    result = {"F_A": F_A, "F_B": F_B, "F_C": F_C, "Cal": Cal, "S_hat": S}

    if slice_masks:
        for name, mask in slice_masks.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() > 0:
                result[f"F_A_{name}"] = macro_f1_multiclass(
                    y_true_A[mask], y_pred_A[mask])
            else:
                log.warning("Slice '%s' has 0 rows; skipping.", name)
                result[f"F_A_{name}"] = float("nan")

    return result


# ── Experiment logger ─────────────────────────────────────────────────────────

def log_experiment(row: dict) -> None:
    """Append one experiment row to outputs/validation_results/experiments.csv."""
    path = config.VALIDATION_DIR / "experiments.csv"
    row = dict(row)
    row.setdefault("timestamp", datetime.utcnow().isoformat())
    row.setdefault("git_hash", _git_hash())

    base_fields = [
        "exp_id", "timestamp", "git_hash", "config_json",
        "F_A", "F_B", "F_C", "Cal", "S_hat",
        "F_domain", "F_robust", "notes",
    ]
    all_fields = base_fields + [k for k in row if k not in base_fields]

    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("Experiment logged: %s", row.get("exp_id", "?"))


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
