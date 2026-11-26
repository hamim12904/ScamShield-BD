"""Assemble + validate the 5-column submission DataFrame.

Owned by Teammate B (T-010).

Contract: build(ids, label, tech, action, conf) -> DataFrame
Raises ValueError on any invalid value before returning.
"""
import logging

import numpy as np
import pandas as pd

from src import config
from src.data_loader import multihot_to_techniques

log = logging.getLogger(__name__)

_VALID_LABELS  = set(range(config.NUM_CLASSES))
_VALID_ACTIONS = set(range(config.NUM_ACTIONS))


def build(ids, label, tech, action, conf) -> pd.DataFrame:
    """Assemble strict 5-column submission frame and validate all values.

    ids:    array-like of id values
    label:  int array (n,) — 0..5
    tech:   (n, 6) binary int matrix (multihot format)
    action: int array (n,) — 0..5
    conf:   float array (n,) — clipped to config.CONF_CLIP before returning
    Returns: DataFrame with columns [id, label, techniques, action, confidence]
    Raises: ValueError on any invalid value.
    """
    label  = np.asarray(label,  dtype=int)
    action = np.asarray(action, dtype=int)
    tech   = np.asarray(tech,   dtype=int)
    conf   = np.clip(np.asarray(conf, dtype=float), *config.CONF_CLIP)
    n      = len(ids)

    if tech.shape != (n, config.NUM_CLASSES):
        raise ValueError(
            f"tech shape must be ({n}, {config.NUM_CLASSES}), got {tech.shape}"
        )

    bad_labels = set(label.tolist()) - _VALID_LABELS
    if bad_labels:
        raise ValueError(f"Invalid label values: {bad_labels}")

    bad_actions = set(action.tolist()) - _VALID_ACTIONS
    if bad_actions:
        raise ValueError(f"Invalid action values: {bad_actions}")

    if np.any(np.isnan(conf)):
        raise ValueError("confidence array contains NaN")

    techniques_col = [multihot_to_techniques(row) for row in tech]

    return pd.DataFrame({
        "id":         list(ids),
        "label":      label.tolist(),
        "techniques": techniques_col,
        "action":     action.tolist(),
        "confidence": conf.round(4).tolist(),
    })
