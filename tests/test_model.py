import numpy as np
from sklearn.datasets import make_classification
from src.model import train_model, predict

def test_train_model():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = train_model(X, y)
    # Check that the model is trained (coefficients not nil)
    assert hasattr(model, "coef_")

def test_predict():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = train_model(X, y)
    predictions = predict(model, X)
    # Verify that the predictions are the same length as the data
    assert len(predictions) == len(y)
