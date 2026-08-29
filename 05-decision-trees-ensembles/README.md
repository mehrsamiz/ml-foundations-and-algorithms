# 05 — Decision Trees & Ensemble Methods

Two notebooks exploring tree-based models on the **Cryotherapy** dataset (90 patients,
6 features), which records whether cryotherapy successfully removed warts.

## `part1_decision_trees.ipynb`

Tuning a single `DecisionTreeClassifier`.

- `max_depth` sweep over `[2, 4, 6, 8, None]`, selecting the best depth by test F1-score.
- Feature importance analysis — and why a depth-2 tree assigns zero importance to most
  features.
- Full walkthrough of the resulting tree structure, node by node.

## `part2_ensembles.ipynb`

Comparing a single tree against two ensemble strategies.

- **Random Forest** (bagging) vs. **AdaBoost** (boosting) vs. a plain Decision Tree —
  accuracy, F1-score, and training time.
- Accuracy vs. `n_estimators` for both ensemble methods, including why performance
  saturates early on a dataset this small.
- Discussion of *why* boosting is slower than bagging (sequential vs. parallel tree
  construction).

## Dataset

[`Cryotherapy.xlsx`](Cryotherapy.xlsx) — included directly, it's small (90 rows, 7
columns): `sex`, `age`, `Time`, `Number_of_Warts`, `Type`, `Area`, and the target
`Result_of_Treatment`.

## Run it

```bash
pip install -r ../requirements.txt
jupyter notebook part1_decision_trees.ipynb
jupyter notebook part2_ensembles.ipynb
```
