"""
Logistic Regression from scratch — NumPy only.

Binary classification with sigmoid activation + BCE loss.
NOTE: binary only — for multiclass extend to softmax + CCE.

See loss.md for the math.
"""

import numpy as np


class LogisticRegression:
    """Logistic Regression via gradient descent (binary classification).

    Args:
        learning_rate: GD step size
        n_iterations: training epochs
        lambda_: L2 regularization strength
        threshold: decision cutoff for predict()
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        lambda_: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.threshold = threshold

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_hat = self._sigmoid(X @ self.weights + self.bias)

            dW = (1 / n) * (X.T @ (y_hat - y)) + (self.lambda_ / n) * self.weights
            db = (1 / n) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dW
            self.bias    -= self.learning_rate * db

            self.loss_history.append(self._bce(y, y_hat, n))

        return self

    def predict_proba(self, X):
        return self._sigmoid(np.array(X, dtype=float) @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _sigmoid(z):
        # numerically stable: avoid overflow for large negative z
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _bce(self, y, y_hat, n):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        loss = -(1 / n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss + (self.lambda_ / (2 * n)) * np.sum(self.weights ** 2)
