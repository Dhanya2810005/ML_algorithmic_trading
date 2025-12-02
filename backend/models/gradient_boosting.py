import numpy as np
from .decision_tree import DecisionTreeRegressorCustom

class GradientBoostingRegressorCustom:
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = y.mean()
        pred = np.full(y.shape, self.initial_pred)
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth)
            tree.fit(X, residual)
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
