"""Ensemble blending of probability arrays from multiple models.

Owned by Teammate B (T-030).

Contract:
  blend(prob_list, weights=None) -> np.ndarray (n, k)
  tune_blend_weight(p1, p2, y_true, metric_fn, grid) -> float
  model_disagreement(p1, p2) -> np.ndarray (n,)
"""
import logging

import numpy as np

log = logging.getLogger(__name__)


def blend(prob_list: list, weights=None) -> np.ndarray:
    """Weighted average of class-probability arrays.

    prob_list: list of (n, k) numpy arrays — one per model
    weights:   list of non-negative floats (normalised internally); default = equal
    Returns:   (n, k) numpy array
    """
    if not prob_list:
        raise ValueError("prob_list must not be empty")

    if weights is None:
        weights = [1.0 / len(prob_list)] * len(prob_list)

    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        raise ValueError("All weights are zero")
    weights = weights / weights.sum()

    result = sum(w * np.asarray(p, dtype=float) for w, p in zip(weights, prob_list))
    log.debug("Ensemble blend: %d models, weights=%s", len(prob_list), weights.round(3))
    return result


def tune_blend_weight(p1: np.ndarray,
                      p2: np.ndarray,
                      y_true: np.ndarray,
                      metric_fn,
                      grid=(0.3, 0.4, 0.5, 0.6, 0.7),
                      is_multilabel: bool = False) -> tuple:
    """Pick mixing weight w in grid for w*p1 + (1-w)*p2 maximising metric_fn on OOF.

    Returns (best_w, best_score, blended_probs).
    """
    best_w, best_score, best_p = 0.5, -1.0, None
    for w in grid:
        p = blend([p1, p2], weights=[w, 1.0 - w])
        pred = (p >= 0.5).astype(int) if is_multilabel else p.argmax(axis=1)
        score = float(metric_fn(y_true, pred))
        if score > best_score:
            best_score = score
            best_w = float(w)
            best_p = p
    log.info("Ensemble tune: best_w=%.2f (1-w=%.2f) -> score=%.4f",
             best_w, 1.0 - best_w, best_score)
    return best_w, best_score, best_p


def model_disagreement(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute per-row L1 probability disagreement (total variation distance) in [0, 1]."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return 0.5 * np.sum(np.abs(p1 - p2), axis=1)
