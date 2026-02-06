"""
Decision Tree — implemented from scratch using NumPy.

Uses Information Gain (ID3 criterion) with Entropy for splitting.
Supports max_depth, min_samples_split pruning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Node:
    """A node in the decision tree."""
    feature: int | None = None        # split feature index
    threshold: float | None = None    # split threshold value
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: Any = None                  # leaf prediction value
    gain: float = 0.0                  # information gain used


class DecisionTree:
    """
    Decision Tree classifier using Information Gain.

    Parameters
    ----------
    max_depth : int | None
        Maximum depth of the tree. None = grow fully.
    min_samples_split : int
        Minimum samples needed to split a node (default 2).
    n_features : int | None
        Number of features to consider per split (None = all).
        Set to int(sqrt(total_features)) for Random Forest usage.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        n_features: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self._root: _Node | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        X, y = np.array(X, dtype=float), np.array(y)
        self.n_features = self.n_features or X.shape[1]
        self._root = self._build(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self._root) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples, n_feats = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_classes == 1
            or n_samples < self.min_samples_split
        ):
            return _Node(value=self._majority(y))

        # Best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        if best_gain == 0:
            return _Node(value=self._majority(y))

        left_mask  = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        node = _Node(feature=best_feat, threshold=best_thresh, gain=best_gain)
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feat_idxs: np.ndarray
    ) -> tuple[int, float, float]:
        best_gain, best_feat, best_thresh = 0.0, 0, 0.0
        parent_entropy = self._entropy(y)

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                gain = self._information_gain(y, X[:, feat], t, parent_entropy)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t

        return best_feat, best_thresh, best_gain

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        """H(y) = -Σ p_k log2(p_k)"""
        # TODO: try Gini impurity as alternative — often faster to compute
        counts = np.bincount(y.astype(int))
        probs  = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-12) for p in probs if p > 0)

    def _information_gain(
        self,
        y: np.ndarray,
        X_feat: np.ndarray,
        threshold: float,
        parent_entropy: float,
    ) -> float:
        left  = y[X_feat <= threshold]
        right = y[X_feat >  threshold]
        if len(left) == 0 or len(right) == 0:
            return 0.0
        n = len(y)
        weighted_entropy = (len(left) / n) * self._entropy(left) + \
                           (len(right) / n) * self._entropy(right)
        return parent_entropy - weighted_entropy

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _traverse(self, x: np.ndarray, node: _Node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    @staticmethod
    def _majority(y: np.ndarray):
        return np.bincount(y.astype(int)).argmax()
