"""Task B: 6 independent binary technique classifiers.

Owned by Teammate B (T-010).
Each technique T1–T6 is treated as an independent binary problem.
Per-technique thresholds are tuned in T-024.

Contract: fit(X, Y, sample_weight=None) -> self
          predict_proba(X) -> (n, 6)  independent positive-class probabilities
          predict(X) -> (n, 6)  binary 0/1 using per-technique thresholds
          set_thresholds(list[6 floats]) -> None
"""
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config

log = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.5


class TaskBHead:
    """Six independent binary LRs for technique prediction (Task B)."""

    def __init__(self, C: float = 5.0, seed: int = config.SEED):
        self.C = C
        self.seed = seed
        self.thresholds = [_DEFAULT_THRESHOLD] * config.NUM_CLASSES

    def fit(self, X, Y, sample_weight=None) -> "TaskBHead":
        """Fit 6 binary heads. Y: (n, 6) int binary matrix (techniques_to_multihot)."""
        Y = np.asarray(Y, dtype=int)
        self.clfs_   = []
        self.priors_ = []
        for j in range(config.NUM_CLASSES):
            y_j = Y[:, j]
            pos = int(y_j.sum())
            if pos < 2:
                # Not enough positives: use a constant prior and log a warning
                prior = float(pos) / max(len(y_j), 1)
                self.clfs_.append(None)
                self.priors_.append(prior)
                log.warning(
                    "Technique T%d: only %d positive(s) in this fold; "
                    "using constant prior %.3f",
                    j + 1, pos, prior,
                )
            else:
                clf = LogisticRegression(
                    C=self.C, max_iter=3000, class_weight="balanced",
                    solver="lbfgs", random_state=self.seed,
                )
                clf.fit(X, y_j, sample_weight=sample_weight)
                self.clfs_.append(clf)
                self.priors_.append(None)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return (n, 6) positive-class probabilities."""
        n = X.shape[0]
        out = np.zeros((n, config.NUM_CLASSES), dtype=float)
        for j, (clf, prior) in enumerate(zip(self.clfs_, self.priors_)):
            if clf is None:
                out[:, j] = prior
            else:
                out[:, j] = clf.predict_proba(X)[:, 1]
        return out

    def predict(self, X) -> np.ndarray:
        """Return (n, 6) binary predictions using per-technique thresholds."""
        proba = self.predict_proba(X)
        thr   = np.array(self.thresholds, dtype=float)
        return (proba >= thr).astype(int)

    def set_thresholds(self, thresholds) -> None:
        """Set per-technique classification thresholds (list of 6 floats in [0,1])."""
        if len(thresholds) != config.NUM_CLASSES:
            raise ValueError(f"Expected {config.NUM_CLASSES} thresholds, got {len(thresholds)}")
        self.thresholds = [float(t) for t in thresholds]
