# 03 — KNN & SVM

Two notebooks covering instance-based and margin-based classification.

## `knn_from_scratch.ipynb`

A **K-Nearest Neighbors classifier implemented from scratch** in pure Python/NumPy
(Euclidean distance, majority vote), evaluated on the Iris dataset.

- Custom KNN vs. scikit-learn's `KNeighborsClassifier` across `k = [1, 3, 5, 7, 11]`.
- Comparison of distance metrics (Euclidean, Manhattan, Minkowski p=3) at `k=5`, with an
  analysis of why Manhattan distance came out ahead on this dataset.

## `svm_kernels.ipynb`

An exploration of **SVM kernels and decision boundaries** on non-linearly-separable toy
data (`make_moons`, `make_circles`), using scikit-learn's `SVC`.

- Linear, polynomial, and RBF kernels compared by accuracy and by their decision
  boundary shape.
- The effect of the regularization parameter `C` on margin width.
- The soft-margin SVM objective, derived and explained term by term.

## Run it

```bash
pip install -r ../requirements.txt
jupyter notebook knn_from_scratch.ipynb
jupyter notebook svm_kernels.ipynb
```
