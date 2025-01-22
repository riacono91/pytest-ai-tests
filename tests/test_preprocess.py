import numpy as np
from src.preprocess import normalize_data

def test_normalize_data():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    normalized_X = normalize_data(X)
    # Verify that the average of each column is 0
    assert np.allclose(np.mean(normalized_X, axis=0), 0)
    # Verify that the standard deviation of each column is 1
    assert np.allclose(np.std(normalized_X, axis=0), 1)
