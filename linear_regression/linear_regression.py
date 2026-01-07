"""
Linear Regression from scratch — NumPy only.

Gradient descent (batch + mini-batch) and the normal equation.
Ridge regularization (L2) also supported.

See math.md for derivation.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression with optional L2 regularization.

    Args:
        learning_rate: step size for gradient descent
        n_iterations: how many GD steps to run
        method: 'gradient_descent', 'mini_batch', or 'normal_equation'
        batch_size: only used when method='mini_batch'
        lambda_: L2 penalty weight (0 = no regularization)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        method: str = "gradient_descent",
        batch_size: int = 32,
        lambda_: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.batch_size = batch_size
        self.lambda_ = lambda_

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        if self.method == "normal_equation":
            self._normal_equation(X, y)
        elif self.method == "mini_batch":
            self._mini_batch_gd(X, y, n_samples)
        else:
            self._batch_gd(X, y, n_samples)

        return self

    def predict(self, X):
        return np.array(X, dtype=float) @ self.weights + self.bias

    def score(self, X, y):
        """R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # ------------------------------------------------------------------ helpers

    def _mse(self, X, y, n):
        preds = X @ self.weights + self.bias
        mse = np.mean((preds - y) ** 2)
        l2 = (self.lambda_ / (2 * n)) * np.sum(self.weights ** 2)
        return mse + l2

    def _gradients(self, X, y, n):
        errors = X @ self.weights + self.bias - y
        dW = (2 / n) * (X.T @ errors) + (self.lambda_ / n) * self.weights
        db = (2 / n) * np.sum(errors)
        return dW, db

    def _batch_gd(self, X, y, n):
        # TODO: add learning rate decay (step or exponential)
        for _ in range(self.n_iterations):
            dW, db = self._gradients(X, y, n)
            self.weights -= self.learning_rate * dW
            self.bias    -= self.learning_rate * db
            self.loss_history.append(self._mse(X, y, n))

    def _mini_batch_gd(self, X, y, n):
        for _ in range(self.n_iterations):
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]
            for start in range(0, n, self.batch_size):
                Xb = X_s[start : start + self.batch_size]
                yb = y_s[start : start + self.batch_size]
                dW, db = self._gradients(Xb, yb, len(yb))
                self.weights -= self.learning_rate * dW
                self.bias    -= self.learning_rate * db
            self.loss_history.append(self._mse(X, y, n))

    def _normal_equation(self, X, y):
        # theta = (X^T X + lambda*I)^{-1} X^T y
        X_b = np.c_[np.ones(X.shape[0]), X]
        reg = self.lambda_ * np.eye(X_b.shape[1])
        reg[0, 0] = 0  # don't penalize bias term
        theta = np.linalg.pinv(X_b.T @ X_b + reg) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]
