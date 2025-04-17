import numpy as np

def normalize_data(X):
    """Normalizes the data so that they have mean 0 and standard deviation 1."""
    
    # Calculate the mean of each column (feature) in the dataset
    mean = np.mean(X, axis=0)
    
    # Calculate the standard deviation of each column (feature)
    std = np.std(X, axis=0)
    
    # Subtract the mean and divide by the standard deviation for each column
    # This scales the data so that each feature has mean 0 and std 1
    return (X - mean) / std
