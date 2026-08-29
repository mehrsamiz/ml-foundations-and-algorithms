# 01 — Linear Regression from Scratch

A NumPy-only implementation of Linear Regression, trained two different ways and
benchmarked on the **California Housing** dataset (loaded directly via
`sklearn.datasets.fetch_california_housing`, no external file needed).

## What's implemented

- **Batch gradient descent** — manual forward pass, MSE loss, and gradient computation.
- **Normal equation** — closed-form solution using the Moore-Penrose pseudo-inverse
  (`np.linalg.pinv`), with a written justification for why `pinv` is preferred over
  `inv` for numerical stability.
- Learning-rate comparison (`0.001`, `0.01`, `0.1`) with loss curves.
- Evaluation via MSE and R² on train/test splits.
- A short analysis comparing the weights learned by gradient descent vs. the normal
  equation.

## Files

| File | Description |
|---|---|
| [`linear_regression.py`](linear_regression.py) | The `LinearRegression` class (gradient descent + normal equation). |
| [`notebook.ipynb`](notebook.ipynb) | Full pipeline: EDA → split → scaling → training → evaluation → analysis. |

## Results

| Method | Test R² |
|---|---|
| Normal Equation | 0.5768 |
| Gradient Descent (lr=0.1) | comparable, see notebook |

## Run it

```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
