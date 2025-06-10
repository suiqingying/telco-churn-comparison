class DecisionTreeModel:
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy