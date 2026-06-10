import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin

class ElasticNetGDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.01, max_iter=1000, l1=0.0, l2=0.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1 = l1
        self.l2 = l2

    def mse_gradient(self, X, Y, theta):
        n = X.shape[0]
        y_pred = X @ theta
        return (2 / n) * X.T @ (y_pred - Y)

    def fit(self, X, y):
        Y = np.asarray(y)

        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))

        if sp.issparse(X):
            X_calc = sp.hstack([ones, X], format='csr')
        else:
            X = np.asarray(X)
            X_calc = np.hstack((ones, X))

        n, f = X_calc.shape
        theta = np.zeros(f)

        for i in range(self.max_iter):
            grad_loss = self.mse_gradient(X_calc, Y, theta)

            grad_reg = np.zeros_like(theta)
            if f > 1:
                grad_reg[1:] += 2 * self.l2 * theta[1:]
                grad_reg[1:] += self.l1 * np.sign(theta[1:])

            theta = theta - self.learning_rate * (grad_loss + grad_reg)

        self.theta = theta

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        
        if sp.issparse(X):
            X_calc = sp.hstack([ones, X], format='csr')
        else:
            X = np.asarray(X)
            X_calc = np.hstack((ones, X))

        return X_calc @ self.theta