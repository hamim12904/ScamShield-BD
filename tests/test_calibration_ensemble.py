"""Tests for calibration (T-031) and ensemble (T-030).

Owned by Teammate B. Run: pytest -q tests/test_calibration_ensemble.py
"""
import numpy as np

from src import config
from src.evaluation.calibration import apply, fit_confidence
from src.evaluation.metrics import macro_f1_multiclass
from src.models.ensemble import blend, model_disagreement, tune_blend_weight


def test_calibration_beats_or_falls_back_cleanly():
    rng = np.random.default_rng(42)
    n = 100
    y = rng.integers(0, 6, size=n)
    probs = rng.uniform(0.01, 0.1, size=(n, 6))
    # Make half the rows confident and mostly right, half uncertain and mostly wrong
    for i in range(n):
        if i % 2 == 0:
            probs[i, y[i]] = 0.85
        else:
            probs[i, (y[i] + 1) % 6] = 0.40
    probs /= probs.sum(axis=1, keepdims=True)
    folds = np.arange(n) % 5

    cal = fit_confidence(probs, y, folds=folds)
    assert cal["mode"] in ("logistic", "constant")
    assert cal["cal_brier"] <= cal["const_brier"] + 1e-12

    out = apply(cal, probs)
    lo, hi = config.CONF_CLIP
    assert out.shape == (n,)
    assert np.all((out >= lo) & (out <= hi))


def test_calibration_noise_shrinkage():
    rng = np.random.default_rng(7)
    probs = rng.dirichlet(np.ones(6), size=30)
    y = rng.integers(0, 6, size=30)
    cal = fit_confidence(probs, y, folds=np.arange(30) % 3)
    noise_mask = np.zeros(30, dtype=bool)
    noise_mask[:5] = True
    out = apply(cal, probs, noise_only=noise_mask)
    assert out.shape == (30,)
    assert np.all((out >= config.CONF_CLIP[0]) & (out <= config.CONF_CLIP[1]))


def test_ensemble_blend_and_tune():
    p1 = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
    p2 = np.array([[0.6, 0.4], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    y = np.array([0, 1, 0, 1])

    b = blend([p1, p2], weights=[0.5, 0.5])
    assert np.allclose(b, 0.5 * p1 + 0.5 * p2)

    w, score, _ = tune_blend_weight(
        p1, p2, y, lambda yt, yp: macro_f1_multiclass(yt, yp, labels=[0, 1])
    )
    assert w in (0.3, 0.4, 0.5, 0.6, 0.7)
    assert score > 0.5

    dis = model_disagreement(p1, p2)
    assert dis.shape == (4,)
    assert np.all((dis >= 0.0) & (dis <= 1.0))
