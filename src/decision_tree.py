import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, count=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.count = count

class DecisionTree:
    def __init__(self, max_depth=1, min_samples=1):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def _weighted_error(self, y, weights):
        if len(y) == 0: return 0
        
        unique_classes = np.unique(y)
        class_weights = [np.sum(weights[y == c]) for c in unique_classes]
        
        total_weight = np.sum(weights)
        return total_weight - np.max(class_weights)

    def _best_split(self, X, y, weights):
        best_cost = float('inf')
        split = {'feature': None, 'threshold': None}
        
        m, n = X.shape
        for feat_idx in range(n):

            thresholds = np.unique(X[:, feat_idx])
            potential_thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            for threshold in potential_thresholds:
                left_idx = X[:, feat_idx] <= threshold
                right_idx = ~left_idx
                
                if not np.any(left_idx) or not np.any(right_idx):
                    continue
                
                cost = self._weighted_error(y[left_idx], weights[left_idx]) + \
                       self._weighted_error(y[right_idx], weights[right_idx])
                
                if cost < best_cost:
                    best_cost = cost
                    split = {'feature': feat_idx, 'threshold': threshold}
        
        return split

    def _build_tree(self, X, y, weights, depth=0):
        unique_classes = np.unique(y)
        class_sums = [np.sum(weights[y == c]) for c in unique_classes]
        majority_class = unique_classes[np.argmax(class_sums)]
        node_count = len(y)

        if depth >= self.max_depth or len(unique_classes) == 1 or len(y) < self.min_samples:
            return Node(value=majority_class, count=node_count)

        split = self._best_split(X, y, weights)
        if split['feature'] is None:
            return Node(value=majority_class, count=node_count)

        left_idx = X[:, split['feature']] <= split['threshold']
        right_idx = ~left_idx
        
        left = self._build_tree(X[left_idx], y[left_idx], weights[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], weights[right_idx], depth + 1)
        
        return Node(feature=split['feature'], threshold=split['threshold'], left=left, right=right, count=node_count)

    def fit(self, X, y, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones(len(y)) / len(y)
        self.root = self._build_tree(X, y, sample_weights)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def get_root_feature(self):
        return self.root.feature if self.root else None

    def draw_tree(self, filename=None):
        positions = {}
        x_counter = [0]
        
        def _get_positions(node, depth=0):
            if node is None:
                return
            if node.left is None and node.right is None:
                positions[id(node)] = (x_counter[0], -depth)
                x_counter[0] += 1
                return
            _get_positions(node.left, depth + 1)
            _get_positions(node.right, depth + 1)
            left_pos = positions.get(id(node.left), (0, 0))
            right_pos = positions.get(id(node.right), (0, 0))
            positions[id(node)] = ((left_pos[0] + right_pos[0]) / 2, -depth)
        
        _get_positions(self.root)
        
        fig, ax = plt.subplots(figsize=(max(6, x_counter[0] // 2), max(4, self.max_depth)))
        
        def _draw(node):
            if node is None or id(node) not in positions:
                return
            x, y = positions[id(node)]
            label = f"f{node.feature}<={node.threshold:.1f}\nn={node.count}" if node.feature is not None else f"v:{node.value}\nn={node.count}"
            ax.text(x, y, label, ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
            if node.left:
                x2, y2 = positions.get(id(node.left), (x, y))
                ax.plot([x, x2], [y, y2], 'k-', alpha=0.5)
                _draw(node.left)
            if node.right:
                x2, y2 = positions.get(id(node.right), (x, y))
                ax.plot([x, x2], [y, y2], 'k-', alpha=0.5)
                _draw(node.right)
        
        _draw(self.root)
        ax.set_axis_off()
        if filename:
            fig.savefig(filename, bbox_inches='tight')

            
    def _predict_one(self, x, node):
        if node.value is not None: return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)