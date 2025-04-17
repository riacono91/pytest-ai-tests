from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    """Trains a logistic regression model."""
    
    # Create an instance of the logistic regression model
    model = LogisticRegression()
    
    # Train the model using input features X and labels y
    model.fit(X, y)
    
    # Return the trained model so it can be used later
    return model

def predict(model, X):
    """Returns the predictions of a model."""
    
    # Use the trained model to make predictions on new data X
    return model.predict(X)
