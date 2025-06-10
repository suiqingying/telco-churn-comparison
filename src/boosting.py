from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class BoostingModel:
    def __init__(self):
        self.model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)

    def train(self, X, y):
        self.model.fit(X, y.ravel())

    def evaluate(self, X, y):
        return self.model.score(X, y.ravel())