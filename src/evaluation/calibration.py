"""Confidence calibration model for Task D.

Owned by Teammate B (T-031).

Target:  confidence = P(Task A prediction is correct), fit on OOF.
Metric:  Cal = 1 - Brier(confidence, correctness).
Gate:    Must beat constant-confidence baseline (OOF accuracy) in Brier;
         otherwise fall back to constant baseline and log a warning.

Public API (ARCHITECTURE.md §4):
  fit_confidence(oof_probs, y_true, extra_feats=None, folds=None, ...) -> dict
  apply(calibrator, probs, extra_feats=None, noise_only=None) -> np.ndarray
"""
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config

log = logging.getLogger(__name__)


def extract_cal_features(probs: np.ndarray, extra_feats: np.ndarray = None) -> np.ndarray:
    """Extract calibration features from class probabilities.

    Features: [max_prob, top2_margin, norm_entropy] (+ optional extra_feats).
    """
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    sorted_p = np.sort(probs, axis=1)
    max_p = sorted_p[:, -1]
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    entropy = -np.sum(probs * np.log(probs), axis=1) / np.log(probs.shape[1])

    feats = [max_p[:, None], margin[:, None], entropy[:, None]]
    if extra_feats is not None:
        ef = np.asarray(extra_feats, dtype=float)
        if ef.ndim == 1:
            ef = ef[:, None]
        feats.append(ef)
    return np.hstack(feats)


def _brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((conf - correct) ** 2))


def fit_confidence(oof_probs: np.ndarray,
                   y_true: np.ndarray,
                   extra_feats: np.ndarray = None,
                   folds: np.ndarray = None,
                   noise_only: np.ndarray = None,
                   C_grid=(0.05, 0.1, 0.5, 1.0)) -> dict:
    """Fit logistic calibrator on OOF predictions, cross-fitted if folds provided.

    Compares cross-fitted logistic calibrator against constant-confidence baseline
    (mean OOF accuracy). If logistic calibrator does not improve Brier, returns
    constant-baseline calibrator and logs a warning.
    """
    oof_probs = np.asarray(oof_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    correct = (oof_probs.argmax(axis=1) == y_true).astype(int)
    base_acc = float(np.mean(correct))
    const_brier = _brier(np.full_like(correct, base_acc, dtype=float), correct)

    X = extract_cal_features(oof_probs, extra_feats)

    if noise_only is not None and np.any(noise_only):
        noise_acc = float(np.mean(correct[np.asarray(noise_only, dtype=bool)]))
    else:
        noise_acc = min(base_acc, 0.5)

    best_C, best_brier, best_oof_conf = None, const_brier, None

    if folds is not None and len(np.unique(folds)) > 1:
        folds = np.asarray(folds)
        unique_folds = np.unique(folds)
        for C in C_grid:
            oof_conf = np.empty(len(y_true), dtype=float)
            ok = True
            for f in unique_folds:
                tr = folds != f
                va = folds == f
                if len(np.unique(correct[tr])) < 2:
                    ok = False
                    break
                scaler = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=C, max_iter=1000, random_state=config.SEED)
                clf.fit(scaler.transform(X[tr]), correct[tr])
                oof_conf[va] = clf.predict_proba(scaler.transform(X[va]))[:, 1]
            if not ok:
                continue
            oof_conf = np.clip(oof_conf, *config.CONF_CLIP)
            b = _brier(oof_conf, correct)
            if b < best_brier:
                best_brier = b
                best_C = C
                best_oof_conf = oof_conf
    else:
        # No folds provided: fit with strongest regularisation
        if len(np.unique(correct)) >= 2:
            C = C_grid[0]
            scaler = StandardScaler().fit(X)
            clf = LogisticRegression(C=C, max_iter=1000, random_state=config.SEED)
            clf.fit(scaler.transform(X), correct)
            oof_conf = np.clip(clf.predict_proba(scaler.transform(X))[:, 1], *config.CONF_CLIP)
            b = _brier(oof_conf, correct)
            if b < best_brier:
                best_brier = b
                best_C = C
                best_oof_conf = oof_conf

    if best_C is None:
        log.warning(
            "Calibration: logistic calibrator did not beat constant baseline "
            "(Brier=%.4f, Cal=%.4f); using constant baseline.",
            const_brier, 1.0 - const_brier,
        )
        return {
            "mode": "constant",
            "base_acc": float(np.clip(base_acc, *config.CONF_CLIP)),
            "noise_acc": float(np.clip(noise_acc, *config.CONF_CLIP)),
            "const_brier": const_brier,
            "cal_brier": const_brier,
            "oof_conf": np.full(len(y_true), np.clip(base_acc, *config.CONF_CLIP)),
        }

    # Fit final scaler + logistic model on all OOF rows
    final_scaler = StandardScaler().fit(X)
    final_clf = LogisticRegression(C=best_C, max_iter=1000, random_state=config.SEED)
    final_clf.fit(final_scaler.transform(X), correct)

    log.info(
        "Calibration: logistic C=%.3f beat constant baseline | "
        "Brier %.4f -> %.4f (Cal %.4f -> %.4f)",
        best_C, const_brier, best_brier, 1.0 - const_brier, 1.0 - best_brier,
    )
    return {
        "mode": "logistic",
        "C": best_C,
        "scaler": final_scaler,
        "clf": final_clf,
        "base_acc": float(np.clip(base_acc, *config.CONF_CLIP)),
        "noise_acc": float(np.clip(noise_acc, *config.CONF_CLIP)),
        "const_brier": const_brier,
        "cal_brier": best_brier,
        "oof_conf": best_oof_conf,
    }


def apply(calibrator: dict,
          probs: np.ndarray,
          extra_feats: np.ndarray = None,
          noise_only: np.ndarray = None) -> np.ndarray:
    """Apply fitted calibrator. Returns confidence array clipped to config.CONF_CLIP."""
    n = len(probs)
    if calibrator["mode"] == "constant":
        conf = np.full(n, calibrator["base_acc"], dtype=float)
    else:
        X = extract_cal_features(probs, extra_feats)
        X_s = calibrator["scaler"].transform(X)
        conf = calibrator["clf"].predict_proba(X_s)[:, 1]

    if noise_only is not None:
        mask = np.asarray(noise_only, dtype=bool)
        if np.any(mask):
            # Shrink noise-only confidence toward noise_acc
            conf[mask] = 0.5 * conf[mask] + 0.5 * calibrator["noise_acc"]

    return np.clip(conf, *config.CONF_CLIP)
