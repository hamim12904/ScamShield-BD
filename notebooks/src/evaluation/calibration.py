"""Confidence calibration model for Task D.

Owned by Teammate B (T-031, Phase 3).

Target: confidence = P(Task A prediction is correct), on OOF.
Score:  Calibration = 1 − Brier(confidence, correctness).

Features for calibrator:
  max ensemble probability, top-2 margin, entropy,
  model disagreement, noise_ratio, OOD signal (from C's embeddings).

Model: logistic calibrator (2–4 params) or temperature scaling.
No isotonic unless OOF n ≥ several hundred. Cross-fitted. Clipped to [0.03, 0.97].
"""


def fit_confidence(oof_probs, y_true, extra_feats=None):
    """Fit calibration model on OOF predictions.

    oof_probs:   (n, 6) OOF probability array
    y_true:      (n,) true Task A labels
    extra_feats: optional (n, k) extra features (noise_ratio, OOD signal, etc.)

    Returns fitted calibrator object.
    """
    raise NotImplementedError("calibration.fit_confidence — implement in T-031 (Phase 3)")


def apply(calibrator, probs, extra_feats=None) -> "np.ndarray":
    """Apply fitted calibrator. Returns confidence array in [0.03, 0.97].

    calibrator:  object from fit_confidence()
    probs:       (n, 6) probability array
    extra_feats: optional (n, k) extra features
    """
    raise NotImplementedError("calibration.apply — implement in T-031 (Phase 3)")
