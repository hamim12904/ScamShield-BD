"""Ensemble blending of probability arrays from multiple models.

Owned by Teammate B (T-030).

Contract: blend(prob_list, weights=None) -> np.ndarray (n, k)
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
