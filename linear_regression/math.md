# Linear Regression — Math Derivation

## 1. Model

Given input features $\mathbf{x} \in \mathbb{R}^n$, the model predicts:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

where $\mathbf{w} \in \mathbb{R}^n$ are weights and $b \in \mathbb{R}$ is the bias.

---

## 2. Cost Function — Mean Squared Error

For $m$ training examples:

$$J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

With **L2 regularization (Ridge)**:

$$J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 + \frac{\lambda}{2m} \|\mathbf{w}\|^2$$

---

## 3. Gradient Descent

We minimize $J$ by iteratively updating parameters in the direction of the negative gradient.

### Partial Derivatives

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2}{m} X^T (\hat{\mathbf{y}} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}$$

$$\frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### Update Rules

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

where $\alpha$ is the **learning rate**.

---

## 4. Normal Equation (Closed-Form)

Instead of iterating, we can solve analytically:

$$\boldsymbol{\theta} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$$

- **Pros**: Exact solution, no learning rate needed
- **Cons**: $O(n^3)$ matrix inversion — slow for large feature spaces

---

## 5. R² Score

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- $R^2 = 1$ → perfect fit
- $R^2 = 0$ → model is no better than predicting the mean
- $R^2 < 0$ → model is worse than predicting the mean
