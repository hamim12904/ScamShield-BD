"""Stage 7 / Task D: cross-fitted confidence calibration (1 - Brier).

Contracts (ARCHITECTURE.md §4, DESIGN.md §9):
    fit_confidence(oof_probs, y_true, extra_feats=None, folds=None, seed=config.SEED) -> dict
    apply(calibrator, probs, extra_feats=None) -> np.ndarray
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src import config


def _extract_extra(extra_feats, n):
    if extra_feats is None:
        noise_ratio = np.zeros(n, dtype=np.float64)
        noise_only = np.zeros(n, dtype=bool)
    else:
        noise_ratio = np.asarray(extra_feats["noise_ratio"], dtype=np.float64)
        noise_only = np.asarray(extra_feats["noise_only"], dtype=bool)
    return noise_ratio, noise_only


def _build_features(probs, extra_feats=None):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    sorted_p = np.sort(p, axis=1)
    max_p = sorted_p[:, -1]
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    entropy = -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])
    noise_ratio, _ = _extract_extra(extra_feats, len(p))
    return np.column_stack([max_p, margin, entropy, noise_ratio])


def fit_confidence(oof_probs, y_true, extra_feats=None, folds=None, seed=config.SEED):
    """Cross-fit a compact logistic calibrator on OOF predictions."""
    probs = np.asarray(oof_probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    correct = (probs.argmax(axis=1) == y_true).astype(int)
    X = _build_features(probs, extra_feats)
    _, noise_only = _extract_extra(extra_feats, len(probs))

    base_acc = float(correct.mean())
    noise_acc = float(correct[noise_only].mean()) if noise_only.sum() >= 3 else base_acc
    const_brier = float(brier_score_loss(correct, np.full(len(correct), base_acc)))

    oof_conf = np.full(len(correct), base_acc, dtype=np.float64)
    if folds is not None and len(np.unique(folds)) >= 2:
        folds_arr = np.asarray(folds)
        for f in np.unique(folds_arr):
            tr = folds_arr != f
            va = folds_arr == f
            if len(np.unique(correct[tr])) < 2:
                oof_conf[va] = float(correct[tr].mean())
            else:
                clf = LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
                clf.fit(X[tr], correct[tr])
                oof_conf[va] = clf.predict_proba(X[va])[:, 1]
    else:
        if len(np.unique(correct)) >= 2:
            clf = LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
            clf.fit(X, correct)
            oof_conf = clf.predict_proba(X)[:, 1]

    if noise_only.any():
        oof_conf[noise_only] = 0.5 * oof_conf[noise_only] + 0.5 * noise_acc
    oof_conf = np.clip(oof_conf, config.CONF_CLIP[0], config.CONF_CLIP[1])
    oof_brier = float(brier_score_loss(correct, oof_conf))

    final_clf = None
    if len(np.unique(correct)) >= 2:
        final_clf = LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
        final_clf.fit(X, correct)

    return {
        "clf": final_clf,
        "base_acc": base_acc,
        "noise_acc": noise_acc,
        "oof_brier": oof_brier,
        "const_brier": const_brier,
        "use_const": final_clf is None,
    }


def apply(calibrator, probs, extra_feats=None):
    """Apply fitted calibrator and clip to config.CONF_CLIP."""
    probs = np.asarray(probs, dtype=np.float64)
    _, noise_only = _extract_extra(extra_feats, len(probs))
    if calibrator.get("use_const") or calibrator.get("clf") is None:
        conf = np.full(len(probs), calibrator["base_acc"], dtype=np.float64)
    else:
        X = _build_features(probs, extra_feats)
        conf = calibrator["clf"].predict_proba(X)[:, 1]
    if noise_only.any():
        conf[noise_only] = 0.5 * conf[noise_only] + 0.5 * calibrator["noise_acc"]
    return np.clip(conf, config.CONF_CLIP[0], config.CONF_CLIP[1])
