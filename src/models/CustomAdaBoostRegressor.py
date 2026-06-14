import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class CustomNodeRegressor:
    def __init__(self, depth):
        self.depth = depth
        self.is_leaf = False
        self.value = None
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

class CustomDecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=4, min_samples_leaf=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None

    def fit(self, X, y, sample_weights=None):
        X = np.asarray(X)
        y = np.asarray(y).ravel() 
        n_samples = X.shape[0]
        
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples
        else:
            sample_weights = np.asarray(sample_weights)
            
        self.root = self._build_tree(X, y, sample_weights, depth=0)

    def _build_tree(self, X, y, weights, depth):
        node = CustomNodeRegressor(depth)
        
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.ptp(y) == 0:
            node.is_leaf = True
            node.value = np.average(y, weights=weights) if np.sum(weights) > 0 else np.mean(y)
            return node

        best_feature, best_threshold = self._find_best_split(X, y, weights)

        if best_feature is None:
            node.is_leaf = True
            node.value = np.average(y, weights=weights) if np.sum(weights) > 0 else np.mean(y)
            return node

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            node.is_leaf = True
            node.value = np.average(y, weights=weights) if np.sum(weights) > 0 else np.mean(y)
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold

        node.left = self._build_tree(X[left_mask], y[left_mask], weights[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], weights[right_mask], depth + 1)

        return node

    def _find_best_split(self, X, y, weights):
        n_samples, n_features = X.shape
        best_score = -float('inf')
        best_feature = None
        best_threshold = None

        if self.max_features is None:
            n_subset = n_features
        elif isinstance(self.max_features, int):
            n_subset = min(self.max_features, n_features)
        elif self.max_features == "sqrt":
            n_subset = max(1, int(np.sqrt(n_features)))
        else:
            n_subset = n_features

        features_to_check = np.random.choice(n_features, n_subset, replace=False)

        for feature in features_to_check:
            X_f = X[:, feature]
            order = np.argsort(X_f)
            
            X_f_sorted = X_f[order]
            y_sorted = y[order]
            weights_sorted = weights[order]

            cum_w = np.cumsum(weights_sorted)
            cum_yw = np.cumsum(y_sorted * weights_sorted)

            sum_left_w = cum_w[:-1]
            sum_left_yw = cum_yw[:-1]
            
            sum_right_w = np.maximum(0, cum_w[-1] - sum_left_w)
            sum_right_yw = cum_yw[-1] - sum_left_yw

            n_left_samples = np.arange(1, n_samples)
            n_right_samples = n_samples - n_left_samples

            valid_mask = (sum_left_w > 1e-9) & (sum_right_w > 1e-9) & \
                         (X_f_sorted[:-1] != X_f_sorted[1:]) & \
                         (n_left_samples >= self.min_samples_leaf) & \
                         (n_right_samples >= self.min_samples_leaf)

            if not np.any(valid_mask):
                continue

            scores = np.full(len(valid_mask), -np.inf)
            scores[valid_mask] = (sum_left_yw[valid_mask] ** 2) / sum_left_w[valid_mask] + \
                                 (sum_right_yw[valid_mask] ** 2) / sum_right_w[valid_mask]

            max_idx = np.argmax(scores)
            
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_feature = feature
                best_threshold = (X_f_sorted[max_idx] + X_f_sorted[max_idx + 1]) / 2.0

        return best_feature, best_threshold

    def predict(self, X):
        X = np.asarray(X)
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds[i] = node.value
        return preds


class CustomAdaBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=30, max_depth=4, min_samples_split=4, min_samples_leaf=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_samples = len(y)
        
        self.estimators_ = []
        self.estimator_weights_ = []
        
        weights = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X, y, sample_weights=weights)
            
            preds = tree.predict(X)
            
            errors = np.abs(preds - y)
            max_error = np.max(errors)
            
            if max_error <= 1e-10:
                self.estimators_.append(tree)
                self.estimator_weights_.append(10.0)
                break
                
            relative_errors = errors / max_error
            mean_error = np.sum(weights * relative_errors)
            
            if mean_error >= 0.5 or mean_error <= 1e-10:
                if t == 0 or mean_error <= 1e-10:
                    self.estimators_.append(tree)
                    weight = 10.0 if mean_error <= 1e-10 else 1.0 
                    self.estimator_weights_.append(weight)
                break
                
            beta = mean_error / (1.0 - mean_error)
            estimator_weight = np.log(1.0 / beta)
            
            self.estimators_.append(tree)
            self.estimator_weights_.append(estimator_weight)
            
            weights *= beta ** (1.0 - relative_errors)
            
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(n_samples) / n_samples

        return self

    def predict(self, X):
        X = np.asarray(X)
        if not self.estimators_:
            raise ValueError("Model nie został jeszcze wytrenowany (brak estymatorów)!")
            
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
        n_samples = X.shape[0]
        final_preds = np.zeros(n_samples)
        weights = np.array(self.estimator_weights_)
        
        weights /= np.sum(weights)
        
        for i in range(n_samples):
            sample_preds = predictions[:, i]
            sorted_idx = np.argsort(sample_preds)
            
            sorted_preds = sample_preds[sorted_idx]
            sorted_weights = weights[sorted_idx]
            
            cumulative_weights = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
            
            median_idx = min(median_idx, len(sorted_preds) - 1)
            
            final_preds[i] = sorted_preds[median_idx]
            
        return final_preds