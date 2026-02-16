"""
Neural Network — fully connected (MLP) implemented from scratch using NumPy.

Features:
    - Arbitrary depth and width (list of layer sizes)
    - Activations: ReLU, sigmoid, tanh, softmax
    - Backpropagation with chain rule
    - Loss: Binary/Categorical Cross-Entropy, MSE
    - Optimizer: SGD with momentum
    - Weight initialization: He / Xavier
    - L2 regularization
    - Mini-batch training

TODO: dropout regularization
TODO: Adam / RMSProp optimizer
"""

import numpy as np


# -----------------------------------------------------------------------
# Activation functions + their derivatives
# -----------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)

def tanh_act(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_grad(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

_ACTIVATIONS = {
    "relu":    (relu,     relu_grad),
    "sigmoid": (sigmoid,  sigmoid_grad),
    "tanh":    (tanh_act, tanh_grad),
    "softmax": (softmax,  None),
}


# -----------------------------------------------------------------------
# Neural Network
# -----------------------------------------------------------------------

class NeuralNetwork:
    """
    Fully connected neural network with backpropagation.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer including input and output.
        Example: [784, 128, 64, 10] → 2 hidden layers.
    activations : list[str]
        Activation per layer (excluding input).
        Example: ['relu', 'relu', 'softmax']
    loss : str
        'bce' (binary), 'cce' (categorical), 'mse'
    learning_rate : float
    lambda_ : float
        L2 regularization weight (default 0.0).
    momentum : float
        SGD momentum coefficient (default 0.9).
    batch_size : int
        Mini-batch size (default 32).
    n_epochs : int
        Training epochs (default 100).

    Example
    -------
    >>> model = NeuralNetwork([784, 128, 10], ['relu', 'softmax'], loss='cce')
    >>> model.fit(X_train, y_train_onehot)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str],
        loss: str = "cce",
        learning_rate: float = 0.01,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        batch_size: int = 32,
        n_epochs: int = 100,
    ) -> None:
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 1

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_name = loss
        self.lr = learning_rate
        self.lambda_ = lambda_
        self.momentum = momentum
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self._weights: list[np.ndarray] = []
        self._biases:  list[np.ndarray] = []
        self._vW:      list[np.ndarray] = []   # momentum buffers
        self._vb:      list[np.ndarray] = []
        self.loss_history: list[float] = []

        self._init_params()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetwork":
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n = len(X)

        for epoch in range(self.n_epochs):
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]
            epoch_loss = 0.0
            batches = 0

            for start in range(0, n, self.batch_size):
                Xb = X_s[start : start + self.batch_size]
                yb = y_s[start : start + self.batch_size]

                activations, zs = self._forward(Xb)
                loss = self._compute_loss(yb, activations[-1])
                epoch_loss += loss
                batches += 1

                dW, db = self._backward(Xb, yb, activations, zs)
                self._update_params(dW, db)

            self.loss_history.append(epoch_loss / batches)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.array(X, dtype=float))[0][-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.shape[1] == 1:
            return (proba >= 0.5).astype(int).ravel()
        return np.argmax(proba, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_arr = np.array(y)
        if y_arr.ndim > 1:
            y_arr = np.argmax(y_arr, axis=1)
        return float(np.mean(self.predict(X) == y_arr))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        for i in range(len(self.layer_sizes) - 1):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            act = self.activations[i]

            # He init for ReLU, Xavier for others
            scale = np.sqrt(2 / fan_in) if act == "relu" else np.sqrt(1 / fan_in)
            W = np.random.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))

            self._weights.append(W)
            self._biases.append(b)
            self._vW.append(np.zeros_like(W))
            self._vb.append(np.zeros_like(b))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(
        self, X: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (activations list, pre-activations list)."""
        a, zs = [X], []
        current = X

        for W, b, act_name in zip(self._weights, self._biases, self.activations):
            z = current @ W + b
            fn, _ = _ACTIVATIONS[act_name]
            current = fn(z)
            zs.append(z)
            a.append(current)

        return a, zs

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: list[np.ndarray],
        zs: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        m = len(X)
        n_layers = len(self._weights)
        dW = [None] * n_layers
        db = [None] * n_layers

        # Output layer delta
        delta = activations[-1] - y       # works for both CCE+softmax and BCE+sigmoid

        for i in reversed(range(n_layers)):
            dW[i] = (activations[i].T @ delta) / m + (self.lambda_ / m) * self._weights[i]
            db[i] = delta.mean(axis=0, keepdims=True)

            if i > 0:
                _, grad_fn = _ACTIVATIONS[self.activations[i - 1]]
                delta = (delta @ self._weights[i].T) * grad_fn(zs[i - 1])

        return dW, db

    # ------------------------------------------------------------------
    # Parameter update (SGD + momentum)
    # ------------------------------------------------------------------

    def _update_params(
        self, dW: list[np.ndarray], db: list[np.ndarray]
    ) -> None:
        for i in range(len(self._weights)):
            self._vW[i] = self.momentum * self._vW[i] + self.lr * dW[i]
            self._vb[i] = self.momentum * self._vb[i] + self.lr * db[i]
            self._weights[i] -= self._vW[i]
            self._biases[i]  -= self._vb[i]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        if self.loss_name == "mse":
            return float(np.mean((y - y_pred) ** 2))
        y_pred = np.clip(y_pred, eps, 1 - eps)
        if self.loss_name == "bce":
            return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))
        # categorical cross-entropy (default)
        return float(-np.mean(np.sum(y * np.log(y_pred), axis=1)))
