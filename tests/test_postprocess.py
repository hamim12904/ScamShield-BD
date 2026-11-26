"""Tests for consistency post-processing and Task B threshold tuning.

Owned by Teammate B (T-024). Run: pytest -q tests/test_postprocess.py
"""
import numpy as np

from src.postprocess import apply_rules, apply_rules_batch, tune_task_b_thresholds


def test_non_scam_forces_none_and_action_zero():
    for lab in (0, 5):
        l, t, a = apply_rules(lab, [1, 0, 1, 0, 0, 0], 3, {"B": np.ones(6), "C": np.ones(6)})
        assert l == lab
        assert t.sum() == 0
        assert a == 0


def test_scam_fills_missing_technique_and_action():
    probs = {
        "B": np.array([0.1, 0.4, 0.2, 0.05, 0.05, 0.05]),
        "C": np.array([0.6, 0.1, 0.35, 0.05, 0.05, 0.05]),
    }
    l, t, a = apply_rules(1, [0, 0, 0, 0, 0, 0], 0, probs)
    assert l == 1
    assert t.tolist() == [0, 1, 0, 0, 0, 0]
    assert a == 2


def test_batch_matches_single_row():
    labels = np.array([0, 1, 5, 4])
    techs = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])
    actions = np.array([2, 0, 1, 0])
    pb = np.array([
        [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.4, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.4, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.4, 0.1, 0.1, 0.1],
    ])
    pc = np.array([
        [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.6, 0.3, 0.1, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.5, 0.0, 0.4, 0.1, 0.0, 0.0],
    ])
    lb, tb, ab = apply_rules_batch(labels, techs, actions, pb, pc)
    assert tb[0].sum() == 0 and ab[0] == 0
    assert tb[2].sum() == 0 and ab[2] == 0
    assert tb[1].tolist() == [0, 1, 0, 0, 0, 0] and ab[1] == 1
    assert tb[3].tolist() == [0, 0, 1, 0, 0, 0] and ab[3] == 2


def test_tune_task_b_thresholds_range():
    rng = np.random.default_rng(42)
    Y = rng.integers(0, 2, size=(50, 6))
    P = np.clip(Y * 0.6 + rng.uniform(0, 0.4, size=(50, 6)), 0, 1)
    folds = np.arange(50) % 5
    thrs = tune_task_b_thresholds(P, Y, folds)
    assert len(thrs) == 6
    assert all(0.05 <= t <= 0.95 for t in thrs)
