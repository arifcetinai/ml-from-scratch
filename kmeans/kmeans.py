"""
K-Means Clustering — implemented from scratch using NumPy.

Lloyd's algorithm with:
    - K-Means++ initialization
    - Inertia tracking
    - Multiple restarts for global-optimum approximation
"""

import numpy as np


class KMeans:
    """
    K-Means clustering via Lloyd's algorithm.

    Parameters
    ----------
    k : int
        Number of clusters (default 3).
    n_iterations : int
        Maximum iterations per run (default 300).
    n_init : int
        Number of independent runs; best is kept (default 10).
    tol : float
        Convergence tolerance — stops when centroid shift < tol.
    init : str
        'kmeans++' | 'random'
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 3,
        n_iterations: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        init: str = "kmeans++",
        random_state: int | None = None,
    ) -> None:
        self.k = k
        self.n_iterations = n_iterations
        self.n_init = n_init
        self.tol = tol
        self.init = init
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = float("inf")
        self.inertia_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best_centroids, best_labels, best_inertia = None, None, float("inf")

        for _ in range(self.n_init):
            centroids = self._init_centroids(X, rng)
            labels, inertia, history = self._lloyd(X, centroids)
            if inertia < best_inertia:
                best_centroids, best_labels, best_inertia = centroids, labels, inertia
                self.inertia_history = history

        self.centroids_ = best_centroids
        self.labels_    = best_labels
        self.inertia_   = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return self._assign(X, self.centroids_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lloyd(
        self, X: np.ndarray, centroids: np.ndarray
    ) -> tuple[np.ndarray, float, list[float]]:
        history: list[float] = []

        for _ in range(self.n_iterations):
            labels = self._assign(X, centroids)
            new_centroids = np.array([
                X[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c]
                for c in range(self.k)
            ])
            inertia = self._inertia(X, labels, new_centroids)
            history.append(inertia)

            if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < self.tol:
                centroids = new_centroids
                break
            centroids = new_centroids

        return labels, history[-1], history

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest centroid."""
        dists = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis], axis=2)
        return np.argmin(dists, axis=1)

    def _inertia(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> float:
        return float(sum(
            np.sum((X[labels == c] - centroids[c]) ** 2)
            for c in range(self.k)
        ))

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.init == "random":
            idxs = rng.choice(len(X), self.k, replace=False)
            return X[idxs].copy()

        # K-Means++ initialization
        centroids = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            dists = np.min(
                np.linalg.norm(X[:, np.newaxis] - np.array(centroids)[np.newaxis], axis=2),
                axis=1,
            ) ** 2
            probs = dists / dists.sum()
            centroids.append(X[rng.choice(len(X), p=probs)])
        return np.array(centroids)
