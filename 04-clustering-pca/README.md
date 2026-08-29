# 04 — K-Means & PCA from Scratch

Two unsupervised-learning implementations, both built from scratch with NumPy only.

## `kmeans.ipynb`

A **from-scratch K-Means** implementation applied to the petal length/width features of
the Iris dataset.

- Clustering at `k = 2, 3, 4, 5` to study under- and over-clustering relative to the
  true number of species (3).
- Sensitivity of the result to centroid initialization (seed), including why `k=3`
  converges stably while larger, non-optimal `k` values don't.

## `pca_image_compression.ipynb`

A **from-scratch PCA** implementation (eigendecomposition of the covariance matrix),
verified numerically against scikit-learn's `PCA`, then used for **lossy image
compression**: each RGB channel of [`animal.jpg`](animal.jpg) is compressed and
reconstructed with varying numbers of principal components (`K = 8, 32, 128`).

- Includes a short discussion of why general-purpose PCA is not competitive with a
  specialized codec like JPEG for image compression.

## Run it

```bash
pip install -r ../requirements.txt
jupyter notebook kmeans.ipynb
jupyter notebook pca_image_compression.ipynb
```
