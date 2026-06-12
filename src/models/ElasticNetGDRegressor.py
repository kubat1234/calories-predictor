import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin

class ElasticNetGDRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        l1=0.0,
        l2=0.0,
        verbose=False,
        tol=1e-6,
        patience=20,
        lr_decay=0.5,
        lr_patience=10,
        min_learning_rate=1e-3,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1 = l1
        self.l2 = l2
        self.verbose = verbose
        self.tol = tol
        self.patience = patience
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.min_learning_rate = min_learning_rate

    def fit(self, X, y):
        n_samples, n_features = X.shape

        effective_l1 = self.l1 / n_samples
        effective_l2 = self.l2 / n_samples
        current_lr = float(self.learning_rate)

        Y = np.asarray(y).ravel()
        ones = np.ones((n_samples, 1))

        if sp.issparse(X):
            X_calc = sp.hstack([ones, X], format='csr')
        else:
            X = np.asarray(X)
            X_calc = np.hstack((ones, X))

        n, f = X_calc.shape
        theta = np.zeros(f)
        best_theta = theta.copy()
        best_loss = np.inf
        no_improve_count = 0
        no_improve_lr_count = 0
        
        for i in range(self.max_iter):
            y_pred = X_calc @ theta
            
            grad_mse = (2 / n) * X_calc.T @ (y_pred - Y)
            
            grad_l2 = np.zeros_like(theta)
            if f > 1 and effective_l2 > 0:
                grad_l2[1:] = 2 * effective_l2 * theta[1:]

            theta = theta - current_lr * (grad_mse + grad_l2)

            if f > 1 and effective_l1 > 0:
                l1_penalty = current_lr * effective_l1
                theta[1:] = np.sign(theta[1:]) * np.maximum(np.abs(theta[1:]) - l1_penalty, 0)

            y_pred_new = X_calc @ theta
            mse = np.mean((y_pred_new - Y) ** 2)
            reg_l2 = effective_l2 * np.sum(theta[1:] ** 2)
            reg_l1 = effective_l1 * np.sum(np.abs(theta[1:]))
            loss = mse + reg_l1 + reg_l2

            if best_loss - loss > self.tol:
                best_loss = loss
                best_theta = theta.copy()
                no_improve_count = 0
                no_improve_lr_count = 0
            else:
                no_improve_count += 1
                no_improve_lr_count += 1

            if no_improve_lr_count >= self.lr_patience and current_lr > self.min_learning_rate:
                new_lr = max(current_lr * self.lr_decay, self.min_learning_rate)
                if new_lr < current_lr:
                    current_lr = new_lr
                    no_improve_count = 0
                no_improve_lr_count = 0

            if self.verbose:
                print(
                    f"Iteracja {i + 1}/{self.max_iter} | Loss: {loss:.4f} | LR: {current_lr:.6f}",
                    end="\r",
                    flush=True,
                )

            self.n_iter_ = i + 1

            if no_improve_count >= self.patience:
                if self.verbose:
                    print(
                        f"\nEarly stopping: iteracja {self.n_iter_}/{self.max_iter}, "
                        f"loss={loss:.6f}, lr={current_lr:.6f}"
                    )
                break

        if self.verbose and no_improve_count < self.patience:
            print()

        self.theta = best_theta
        self.learning_rate_ = current_lr
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