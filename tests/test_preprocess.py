# import numpy as np
# from src.preprocess import normalize_data

# def test_normalize_data():
#     # Create a small 2D array with 3 rows and 2 columns
#     X = np.array([[1, 2], [3, 4], [5, 6]])
    
#     # Normalize the data using the function from preprocess.py
#     normalized_X = normalize_data(X)
    
#     # Check that the mean (average) of each column is close to 0
#     assert np.allclose(np.mean(normalized_X, axis=0), 0)
    
#     # Check that the standard deviation of each column is close to 1
#     assert np.allclose(np.std(normalized_X, axis=0), 1)

import numpy as np
from src.preprocess import normalize_data

def test_normalize_data():
    # Create a small 2D array with 3 rows and 2 columns
    X = np.array([[1, 2], [3, 4], [5, 6]])
    print("Original Data:")
    print(X)

    # Normalize the data using the function from preprocess.py
    normalized_X = normalize_data(X)
    print("\nNormalized Data:")
    print(normalized_X)

    # Print mean and standard deviation to verify visually
    print("\nMean of columns (should be ~0):")
    print(np.mean(normalized_X, axis=0))

    print("Standard deviation of columns (should be ~1):")
    print(np.std(normalized_X, axis=0))

    # Check that the mean of each column is close to 0
    assert np.allclose(np.mean(normalized_X, axis=0), 0)

    # Check that the standard deviation of each column is close to 1
    assert np.allclose(np.std(normalized_X, axis=0), 1)
