"""
Gaussian Naive Bayes — implemented from scratch using NumPy.

Assumes feature independence and Gaussian likelihood per class.
Applies log-space computation to avoid floating-point underflow.
"""

import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    Computes:
        P(y | x) ∝ P(y) * Π P(x_i | y)

    where P(x_i | y) is modelled as a Gaussian:
        P(x_i | y) = N(x_i; μ_{i,y}, σ_{i,y}²)
    """

    def __init__(self) -> None:
        self._classes: np.ndarray | None = None
        self._log_priors: np.ndarray | None = None   # log P(y_k)
        self._means: np.ndarray | None = None         # μ [n_classes, n_features]
        self._vars: np.ndarray | None = None          # σ² [n_classes, n_features]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X, y = np.array(X, dtype=float), np.array(y)
        self._classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self._classes)

        self._means      = np.zeros((n_classes, n_features))
        self._vars       = np.zeros((n_classes, n_features))
        self._log_priors = np.zeros(n_classes)

        for idx, cls in enumerate(self._classes):
            X_c = X[y == cls]
            self._means[idx]      = X_c.mean(axis=0)
            self._vars[idx]       = X_c.var(axis=0) + 1e-9   # smoothing
            self._log_priors[idx] = np.log(len(X_c) / n_samples)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return self._classes[np.argmax(self._log_posteriors(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized class probabilities."""
        log_post = self._log_posteriors(np.array(X, dtype=float))
        # softmax in log-space to normalize
        log_post -= log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        return probs / probs.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_posteriors(self, X: np.ndarray) -> np.ndarray:
        """Return log P(y_k | x) (unnormalized) for each class."""
        return np.array([
            self._log_prior_k + self._log_likelihood_k(X, k)
            for k, self._log_prior_k in enumerate(self._log_priors)
        ]).T                               # shape: [n_samples, n_classes]

    def _log_likelihood_k(self, X: np.ndarray, k: int) -> np.ndarray:
        """log N(x; μ_k, σ²_k) summed over features."""
        mu, var = self._means[k], self._vars[k]
        return np.sum(
            -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mu) ** 2) / var,
            axis=1,
        )
