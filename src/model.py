from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    """Trains a logistic regression model."""
    model = LogisticRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    """Returns the predictions of a model."""
    return model.predict(X)
