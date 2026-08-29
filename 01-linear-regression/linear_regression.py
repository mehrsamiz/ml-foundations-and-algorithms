
"""
Linear Regression implemented from scratch with NumPy.

Supports two fitting strategies:
  - 'gradient_descent': iterative batch gradient descent on the MSE loss.
  - 'normal_equation':  closed-form solution via the Moore-Penrose pseudo-inverse.
"""

import numpy as np


class LinearRegression:
    def __init__(self, method='gradient_descent'):
        self.weights = None
        self.bias = None
        self.method = method
        self.loss_history = []

    def fit(self, X, y, epochs=1000, learning_rate=0.01):

        if self.method == 'gradient_descent':
            self._fit_gradient_descent( X, y, epochs, learning_rate)
        elif self.method == 'normal_equation':
            self._fit_normal_equation( X, y)
        else:
            raise ValueError("Method must be 'gradient_descent' or 'normal_equation'")



    def _fit_gradient_descent(self, X, y, epochs, learning_rate):
        #initial
        n_samples, m_features = X.shape
        self.weights = np.zeros(m_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(epochs):

                # 1.	Prediction      y ̂=Xw+b
                y_hat = X.dot(self.weights) + self.bias

                # 2.	Loss      MSE=n1∑(y^−y)2
                error = y_hat - y
                loss = np.mean(error ** 2)  # or (1/n_examples)np.sum(error)**2
                self.loss_history.append(loss)

                # 3.	Gradient      ∇wL=2/n X^T (y^−y)
                grad_w = (2 / n_samples) * (X.T @ error)
                grad_b = (2 / n_samples) * np.sum(error)

                # 4.	Update Parameters      w:=w−η∇wL   and for bias   b:=b−η∇bL
                self.weights -= learning_rate * grad_w
                self.bias -= learning_rate * grad_b

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss = {loss}")
                # Debug at the end of epoch np.linalg.norm(grad_w)

    def _fit_normal_equation(self, X, y):

        # 1.	First we create the augmented X
        X_aug = np.c_[np.ones(X.shape[0]), X]

        #2.	Compute optimal parameters.theta (9,1)
        theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
        # or theta = np.linalg.pinv(X_aug) @ y


        #the first element in theta is bias
        self.bias = theta[0]
        self.weights = theta[1:]
        self.loss_history = []


    def predict(self, X):
        return X.dot(self.weights) + self.bias