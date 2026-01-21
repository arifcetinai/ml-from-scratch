# Logistic Regression — Loss Function Derivation

## 1. Model

The model applies a **sigmoid** (logistic) activation to a linear combination:

$$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

### Sigmoid Properties
- Output range: $(0, 1)$ — interpreted as $P(y=1 \mid \mathbf{x})$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

---

## 2. Why Not MSE?

MSE with sigmoid produces a **non-convex** loss surface — full of local minima, making gradient descent unreliable.

We instead use the **log-likelihood** of a Bernoulli distribution.

---

## 3. Binary Cross-Entropy (BCE) Loss

For $m$ training examples with $y^{(i)} \in \{0, 1\}$:

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

- When $y=1$: loss $= -\log(\hat{y})$ — penalizes low confidence on positive class
- When $y=0$: loss $= -\log(1-\hat{y})$ — penalizes high confidence on negative class

With **L2 regularization**:

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \|\mathbf{w}\|^2$$

---

## 4. Gradient Derivation

Because $\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$ and the BCE chain rule simplifies beautifully:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

These are structurally identical to Linear Regression gradients — the difference lives entirely in how $\hat{y}$ is computed.

---

## 5. Decision Boundary

The model predicts class 1 when $\hat{y} \geq 0.5$, i.e., when:

$$\mathbf{w}^T \mathbf{x} + b \geq 0$$

This is a **linear** boundary in the input space.

---

## 6. Multi-Class Extension

For $K$ classes, replace sigmoid with **softmax**:

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

and use **Categorical Cross-Entropy** loss.
