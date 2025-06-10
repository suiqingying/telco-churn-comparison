from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=2000, solver='lbfgs', verbose=1)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)