# 02 — Logistic Regression from Scratch

A NumPy-only implementation of Logistic Regression, applied to a CDC BRFSS-style health
survey dataset to predict self-reported COVID-19 test results from demographic and
health indicators (~40 raw features).

## What's implemented

- **From-scratch logistic regression** — sigmoid activation, binary cross-entropy loss,
  gradient descent, trained at two learning rates for comparison.
- A full preprocessing pipeline: missing-value imputation (median/mode, fit on train
  only), one-hot encoding of categorical features, and standardization (fit on train
  only, applied to test).
- Manual implementations of accuracy / precision / recall and an **ROC curve computed
  from scratch** (no `sklearn.metrics.roc_curve`).
- A scikit-learn `LogisticRegression` baseline for comparison.
- Feature pruning based on a correlation-with-target threshold and a collinearity check
  (pairs with |corr| > 0.85).

## Files

| File | Description |
|---|---|
| [`logistic_regression.py`](logistic_regression.py) | The `LogisticRegression` class. |
| [`notebook.ipynb`](notebook.ipynb) | Full pipeline: cleaning → encoding → training → evaluation → feature pruning. |

## Dataset

The notebook expects a `covid_survey.csv` in this folder — a CDC BRFSS-derived health
survey with a binary `CovidPos` target and features such as weight, height, and
lifestyle/health indicators. The raw file isn't included in this repo (large, and
subject to its own license); grab a matching BRFSS/health-indicators dataset from Kaggle
and drop it in as `covid_survey.csv` to reproduce the results.

## Run it

```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
