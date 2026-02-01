"""
Perceptron — implemented from scratch using NumPy.

The original Rosenblatt Perceptron (1957) — the building block of neural networks.

Convergence theorem: If data is linearly separable, the perceptron
will find a separating hyperplane in finite steps.
"""

import numpy as np


class Perceptron:
    """
    Binary Perceptron classifier.

    Parameters
    ----------
    learning_rate : float
        Step size for weight updates (default 0.01).
    n_iterations : int
        Maximum training epochs (default 1000).
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.errors_per_epoch: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train on X with labels y ∈ {0, 1}.

        Update rule (when misclassified):
            w ← w + η * (y - ŷ) * x
            b ← b + η * (y - ŷ)
        """
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_per_epoch = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred  = self._step(np.dot(xi, self.weights) + self.bias)
                delta   = self.learning_rate * (yi - y_pred)
                self.weights += delta * xi
                self.bias    += delta
                errors += int(delta != 0)
            self.errors_per_epoch.append(errors)
            if errors == 0:      # converged — linearly separable
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions."""
        X = np.array(X, dtype=float)
        return self._step(X @ self.weights + self.bias)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step(z: np.ndarray | float) -> np.ndarray | int:
        """Heaviside step function: 1 if z >= 0 else 0."""
        return (np.array(z) >= 0).astype(int)
