"""
Logistic Regression implemented from scratch with NumPy.

Trained via batch gradient descent on the binary cross-entropy loss.
"""

import numpy as np


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=1000, learning_rate=0.01):

        self.loss_history = []

        m_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        y = y.reshape(-1, 1)

        for i in range(epochs):
            # 1. Forward Pass (Predictions)
            z = X.dot(self.weights) + self.bias
            y_hat = self._sigmoid(z)

            # 2. Compute Loss (for tracking performance)
            y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
            loss = - (1 / m_samples) * np.sum(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Epoch {i:4d} | Loss: {loss:.6f}")

            # 3. Backward Pass (Gradients)
            error = y_hat - y
            dw = (1 / m_samples) * (X.T @ error)
            db = (1 / m_samples) * np.sum(error)

            # 4. Update Parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        print(f"Training Complete | Final Loss: {self.loss_history[-1]:.6f}")


    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        y_predicted= self._sigmoid(z)
        return y_predicted

    def predict(self, X):
        probabilities = self.predict_proba(X)
        # If probability >= threshold, return 1, else return 0
        return np.where(probabilities >= 0.5, 1, 0)
    