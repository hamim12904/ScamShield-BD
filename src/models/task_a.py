"""Task A: 6-class multinomial scam label classifier.

Owned by Teammate B (T-010).

Contract: fit(X, y, sample_weight=None) -> self
          predict_proba(X) -> (n, 6)
          predict(X) -> (n,)
"""
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src import config

log = logging.getLogger(__name__)


class TaskAHead:
    """6-class multinomial head for Task A (scam label classification).

    model_type='lr'  → LogisticRegression (multinomial, lbfgs)
    model_type='svm' → LinearSVC + softmax over scaled decision values
    """

    def __init__(self, C: float = 5.0,
                 model_type: str = "lr",
                 seed: int = config.SEED):
        self.C = C
        self.model_type = model_type
        self.seed = seed

    def fit(self, X, y, sample_weight=None) -> "TaskAHead":
        """Fit on sparse matrix X and label array y (integers 0–5)."""
        if self.model_type == "svm":
            self.clf_ = LinearSVC(
                C=self.C, class_weight="balanced",
                max_iter=5000, random_state=self.seed,
            )
        else:
            self.clf_ = LogisticRegression(
                C=self.C, max_iter=3000, class_weight="balanced",
                solver="lbfgs", random_state=self.seed,
            )
        self.clf_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return (n, 6) class probabilities (zero-filled for unseen classes)."""
        n = X.shape[0]
        proba = np.zeros((n, config.NUM_CLASSES), dtype=float)
        if self.model_type == "svm":
            decision = self.clf_.decision_function(X)
            if decision.ndim == 1:                            # binary edge case
                decision = np.column_stack([-decision, decision])
            decision -= decision.max(axis=1, keepdims=True)  # numerical stability
            exp_d = np.exp(decision)
            soft = exp_d / exp_d.sum(axis=1, keepdims=True)
            proba[:, self.clf_.classes_] = soft
        else:
            proba[:, self.clf_.classes_] = self.clf_.predict_proba(X)
        return proba

    def predict(self, X) -> np.ndarray:
        """Return (n,) argmax class labels."""
        return self.predict_proba(X).argmax(axis=1)
