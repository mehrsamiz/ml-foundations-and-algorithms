# Machine Learning Coursework

Implementations of core machine learning algorithms **from scratch (NumPy only)** where
it matters, benchmarked against scikit-learn counterparts, with written analysis of the
results. Built as coursework for an introductory Machine Learning course, plus a paper
reproduction project.

> Project — a from-scratch reproduction/implementation of a Q1-journal paper, with a
> full report — lives in a separate repo: https://github.com/mehrsamiz/clinical-ml-paper-replication.git.

## Why this repo

The goal here was to implement the underlying math by hand — gradients, loss functions, distance
metrics, eigendecomposition — verify each implementation against the equivalent
scikit-learn model, and actually reason about *why* the results look the way they do.

## Contents

| # | Topic | Highlights |
|---|---|---|
| [01](01-linear-regression/) | **Linear Regression** | Gradient descent + normal equation, on California Housing |
| [02](02-logistic-regression/) | **Logistic Regression** | Full preprocessing pipeline, ROC curve from scratch, health-survey data |
| [03](03-knn-svm/) | **KNN & SVM** | Custom KNN, distance-metric comparison, SVM kernel/margin analysis |
| [04](04-clustering-pca/) | **K-Means & PCA** | Custom clustering + PCA-based image compression |
| [05](05-decision-trees-ensembles/) | **Decision Trees & Ensembles** | Depth tuning, feature importance, Random Forest vs. AdaBoost |
| 06 | **Paper Reproduction** | Full implementation + report of a Q1-journal paper — see  https://github.com/mehrsamiz/clinical-ml-paper-replication.git  |

Each folder has its own README with details, results, and instructions to run it.

## What's demonstrated

- Deriving and implementing gradient-based optimization (gradient descent) and
  closed-form solutions (normal equation, PCA via eigendecomposition) by hand.
- Correct train/test methodology: fitting scalers, imputers, and encoders on the
  training set only.
- Building evaluation metrics (accuracy, precision, recall, ROC/AUC) from scratch rather
  than relying on `sklearn.metrics` alone.
- Tree-based models: depth/overfitting trade-offs, feature importance, and bagging vs.
  boosting.
- Validating custom implementations numerically against scikit-learn baselines.
- Reading and explaining results — not just producing plots.

## Tech stack

`Python` · `NumPy` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn` · `Jupyter`

## Setup

```bash
git clone https://github.com/mehrsamiz/ml-foundations-and-algorithms
cd ml-coursework
pip install -r requirements.txt
jupyter notebook
```

Each notebook is self-contained within its folder — open the one you want and run it
top to bottom.

## Repo structure

```
.
├── 01-linear-regression/
│   ├── linear_regression.py
│   ├── notebook.ipynb
│   └── README.md
├── 02-logistic-regression/
│   ├── logistic_regression.py
│   ├── notebook.ipynb
│   └── README.md
├── 03-knn-svm/
│   ├── knn_from_scratch.ipynb
│   ├── svm_kernels.ipynb
│   └── README.md
├── 04-clustering-pca/
│   ├── kmeans.ipynb
│   ├── pca_image_compression.ipynb
│   ├── animal.jpg
│   └── README.md
├── 05-decision-trees-ensembles/
│   ├── part1_decision_trees.ipynb
│   ├── part2_ensembles.ipynb
│   ├── Cryotherapy.xlsx
│   └── README.md
├── requirements.txt
└── README.md
```

