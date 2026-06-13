import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class DumbRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        
        return np.full(shape=n_samples, fill_value=self.mean_)