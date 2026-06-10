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
    def __init__(self, max_depth=10, min_samples_split=4, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        node = CustomNodeRegressor(depth)
        
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _find_best_split(self, X, y):
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
        sum_total = np.sum(y)

        for feature in features_to_check:
            X_f = X[:, feature]

            order = np.argsort(X_f)
            X_f_sorted = X_f[order]
            y_sorted = y[order]

            sum_left = 0.0
            n_left = 0

            for i in range(n_samples - 1):
                sum_left += y_sorted[i]
                n_left += 1
                n_right = n_samples - n_left
                
                if X_f_sorted[i] == X_f_sorted[i + 1]:
                    continue

                sum_right = sum_total - sum_left
                score = (sum_left * sum_left) / n_left + (sum_right * sum_right) / n_right

                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = (X_f_sorted[i] + X_f_sorted[i + 1]) / 2.0

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)


class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=4, max_features="sqrt", patience=10, tolerance=0.001):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.patience = patience
        self.tolerance = tolerance

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        self.estimators_ = []
        
        oob_pred_sum = np.zeros(n_samples)
        oob_pred_count = np.zeros(n_samples)
        
        best_oob_mse = float('inf')
        lack_of_improvement = 0

        print(f"Rozpoczęcie trenowania Random Forest Regressor (drzewa: {self.n_estimators})...")

        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

            oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
            if len(oob_indices) > 0:
                preds = tree.predict(X[oob_indices])
                oob_pred_sum[oob_indices] += preds
                oob_pred_count[oob_indices] += 1

            valid_oob = oob_pred_count > 0
            if np.any(valid_oob):
                current_oob_mse = np.mean((y[valid_oob] - (oob_pred_sum[valid_oob] / oob_pred_count[valid_oob])) ** 2)
                
                if current_oob_mse <= best_oob_mse - self.tolerance:
                    best_oob_mse = current_oob_mse
                    lack_of_improvement = 0
                else:
                    lack_of_improvement += 1

                if lack_of_improvement >= self.patience:
                    print(f"Przerwanie trenowania po {i+1} drzewach z powodu braku poprawy OOB MSE: {current_oob_mse:.4f}")
                    break

        print("Trenowanie zakończone.\n")
        return self

    def predict(self, X):
        X = np.asarray(X)
        tree_predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(tree_predictions, axis=0)