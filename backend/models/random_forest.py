import numpy as np
from .decision_tree import DecisionTreeRegressorCustom

class RandomForestRegressorCustom:
    def __init__(self, n_estimators=10, max_depth=6, min_samples_split=8, sample_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_ratio = sample_ratio
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = int(len(X) * self.sample_ratio)
        idx = np.random.choice(len(X), n_samples, replace=True)
        return X[idx], y[idx]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            Xs, ys = self.bootstrap_sample(X, y)
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(Xs, ys)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)
