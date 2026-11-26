"""Task C: 6-class intended action classifier.

Owned by Teammate B (T-010).
Optionally uses Task A OOF probabilities as extra features (cross-fitted).

Contract: fit(X, y, sample_weight=None, task_a_proba=None) -> self
          predict_proba(X, task_a_proba=None) -> (n, 6)
          predict(X, task_a_proba=None) -> (n,)
"""
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config

log = logging.getLogger(__name__)


class TaskCHead:
    """6-class multinomial head for Task C (intended action)."""

    def __init__(self, C: float = 5.0,
                 seed: int = config.SEED,
                 use_task_a_proba: bool = False):
        self.C = C
        self.seed = seed
        self.use_task_a_proba = use_task_a_proba

    @staticmethod
    def _augment_X(X, task_a_proba):
        """Optionally append Task A probabilities as extra columns."""
        if task_a_proba is not None:
            from scipy.sparse import csr_matrix, hstack
            return hstack([X, csr_matrix(np.asarray(task_a_proba, dtype=float))]).tocsr()
        return X

    def fit(self, X, y, sample_weight=None, task_a_proba=None) -> "TaskCHead":
        """Fit multinomial LR. Optionally append task_a_proba as extra features."""
        X_in = self._augment_X(X, task_a_proba) if self.use_task_a_proba else X
        self.clf_ = LogisticRegression(
            C=self.C, max_iter=3000, class_weight="balanced",
            solver="lbfgs", random_state=self.seed,
        )
        self.clf_.fit(X_in, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X, task_a_proba=None) -> np.ndarray:
        """Return (n, 6) action probabilities."""
        X_in = self._augment_X(X, task_a_proba) if self.use_task_a_proba else X
        n    = X_in.shape[0]
        proba = np.zeros((n, config.NUM_ACTIONS), dtype=float)
        proba[:, self.clf_.classes_] = self.clf_.predict_proba(X_in)
        return proba

    def predict(self, X, task_a_proba=None) -> np.ndarray:
        """Return (n,) argmax action labels."""
        return self.predict_proba(X, task_a_proba).argmax(axis=1)
