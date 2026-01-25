"""
K-Nearest Neighbors — implemented from scratch using NumPy.

Supports:
    - Classification (majority vote)
    - Regression (mean of neighbors)
    - Distance metrics: euclidean, manhattan, minkowski
"""

import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbors classifier / regressor.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider (default 3).
    task : str
        'classification' | 'regression'
    metric : str
        Distance metric: 'euclidean' | 'manhattan' | 'minkowski'
    p : float
        Power for Minkowski distance (only used when metric='minkowski').
    """

    def __init__(
        self,
        k: int = 3,
        task: str = "classification",
        metric: str = "euclidean",
        p: float = 2.0,
    ) -> None:
        self.k = k
        self.task = task
        self.metric = metric
        self.p = p
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """Store training data (lazy learner — no actual training)."""
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels/values for each sample in X."""
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy (classification) or R² (regression)."""
        y_pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(y_pred == y))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_single(self, x: np.ndarray):
        # NOTE: O(n) scan — a KD-tree would be much faster for large datasets
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[: self.k]
        k_labels  = self._y_train[k_indices]

        if self.task == "classification":
            # TODO: add distance-weighted voting
            return Counter(k_labels).most_common(1)[0][0]
        return float(np.mean(k_labels))

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self._X_train - x) ** 2, axis=1))
        if self.metric == "manhattan":
            return np.sum(np.abs(self._X_train - x), axis=1)
        if self.metric == "minkowski":
            return np.sum(np.abs(self._X_train - x) ** self.p, axis=1) ** (1 / self.p)
        raise ValueError(f"Unknown metric: {self.metric}")
