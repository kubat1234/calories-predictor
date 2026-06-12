import numpy as np
import copy
import os
import pickle

class WeightedPerceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y, sample_weights=None):
        m, n = X.shape
        if sample_weights is None:
            sample_weights = np.ones(m) / m
        
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.epochs):
            for i in range(m):
                prediction = np.sign(np.dot(X[i], self.w) + self.b)
                
                if prediction != y[i]:
                    update = self.lr * sample_weights[i] * y[i]
                    self.w += update * X[i]
                    self.b += update

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

class AdaBoost:
    def __init__(self, base_classifier, n_estimators=50):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.n_features_in_ = None

    def fit(self, X, y, val_X=None, val_y=None):
        self.n_features_in_ = X.shape[1]
        m = len(y)
        weights = np.ones(m) / m

        validation_accuracies = []
        training_accuracies = []
        
        for t in range(self.n_estimators):
            learner = copy.deepcopy(self.base_classifier)

            learner.fit(X, y, sample_weights=weights)
            
            predictions = learner.predict(X)
            errors = (predictions != y).astype(int)
            epsilon_t = np.sum(weights * errors) / np.sum(weights)
            
            epsilon_t = max(1e-10, min(1 - 1e-10, epsilon_t))
            
            alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            
            weights *= np.exp(-alpha_t * y * predictions)
            weights /= np.sum(weights)
            
            self.models.append(learner)
            self.alphas.append(alpha_t)

            train_predictions = self.predict(X)
            train_accuracy = np.mean(train_predictions == y)
            training_accuracies.append(train_accuracy)

            if val_X is not None and val_y is not None:
                val_predictions = self.predict(val_X)
                val_accuracy = np.mean(val_predictions == val_y)
                validation_accuracies.append(val_accuracy)

        self.training_accuracies = training_accuracies
        self.validation_accuracies = validation_accuracies
        return (training_accuracies, validation_accuracies)

    def get_feature_importance(self):
        importances = np.zeros(self.n_features_in_, dtype=float)

        for alpha, model in zip(self.alphas, self.models):
            root_feature = model.get_root_feature()
            importances[root_feature] += abs(alpha)

        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance

        return importances

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def predict(self, X):
        weighted_preds = np.zeros(len(X))
        for alpha, model in zip(self.alphas, self.models):
            weighted_preds += alpha * model.predict(X)
            
        return np.sign(weighted_preds)