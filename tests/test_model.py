import numpy as np
from sklearn.datasets import make_classification
from src.model import train_model, predict

def test_train_model():
    # Create a fake classification dataset with 100 samples and 5 features
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    # Train the model using the generated data
    model = train_model(X, y)
    
    # Check that the model has been trained by verifying it has coefficients
    assert hasattr(model, "coef_")

def test_predict():
    # Generate the same fake dataset
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    # Train the model on the data
    model = train_model(X, y)
    
    # Use the trained model to make predictions
    predictions = predict(model, X)
    
    # Check that the number of predictions is equal to the number of samples
    assert len(predictions) == len(y)
