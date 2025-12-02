import numpy as np

class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=5, min_samples_split=8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        if len(y) == 0: return 0
        return np.mean((y - np.mean(y)) ** 2)

    def best_split(self, X, y):
        best_feat, best_thresh, best_mse = None, None, float('inf')
        for f in range(X.shape[1]):
            for thresh in np.unique(X[:, f]):
                left_mask, right_mask = X[:, f] <= thresh, X[:, f] > thresh
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                mse_total = (left_mask.sum()*self.mse(y[left_mask]) + right_mask.sum()*self.mse(y[right_mask])) / len(y)
                if mse_total < best_mse:
                    best_feat, best_thresh, best_mse = f, thresh, mse_total
        return best_feat, best_thresh

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)
        feat, thresh = self.best_split(X, y)
        if feat is None: return np.mean(y)
        left_mask, right_mask = X[:, feat] <= thresh, X[:, feat] > thresh
        return {
            'feature': feat,
            'threshold': thresh,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, row, node):
        if not isinstance(node, dict): return node
        return self.predict_one(row, node['left']) if row[node['feature']] <= node['threshold'] else self.predict_one(row, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(row, self.tree) for row in X])
