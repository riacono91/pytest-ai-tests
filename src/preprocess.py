import numpy as np

def normalize_data(X):
    """Normalizes the data so that they have mean 0 and standard deviation 1."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
