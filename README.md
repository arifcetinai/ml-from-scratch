# ML From Scratch

Implementing classic machine learning algorithms from scratch using only NumPy — no scikit-learn or other ML libraries.

Built as a learning project to understand the math behind each model.

---

## Algorithms

| Algorithm | Folder | Notes |
|-----------|--------|-------|
| Linear Regression | `linear_regression/` | gradient descent + normal equation, L2 regularization |
| Logistic Regression | `logistic_regression/` | sigmoid + binary cross-entropy |
| K-Nearest Neighbors | `knn/` | euclidean / manhattan / minkowski distance |
| Perceptron | `perceptron/` | Rosenblatt perceptron |
| Decision Tree | `decision_tree/` | ID3 algorithm, entropy-based splits |
| Gaussian Naive Bayes | `naive_bayes/` | log-likelihood for numerical stability |
| K-Means Clustering | `kmeans/` | Lloyd's algorithm + K-Means++ initialisation |
| Neural Network | `neural_network/` | MLP, backprop, momentum SGD |

---

## Structure

Each algorithm lives in its own folder:

```
algorithm_name/
    algorithm_name.py   # implementation
    demo.ipynb          # notebook with a real dataset example
    __init__.py
```

Exceptions:
- `linear_regression/` also has `math.md` (gradient derivations)
- `logistic_regression/` also has `loss.md` (BCE derivation)

---

## Requirements

```
numpy
scikit-learn   # datasets + evaluation metrics only
matplotlib
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

```python
from linear_regression.linear_regression import LinearRegression

model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

---

## Notes

- Everything is NumPy-only on the algorithm side
- scikit-learn is used only for loading datasets and computing metrics
- Notebooks use real-world datasets (California Housing, Breast Cancer, Iris, etc.)
