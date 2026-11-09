import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """

    X_arr = np.array(X, dtype = float)
    N = X_arr.shape[0]
    
    # Compute feature means
    mu = np.mean(X_arr, axis = 0)
    
    # Center the data
    X_c = X_arr - mu
    
    # Compute sample covariance matrix
    sigma = (X_c.T @ X_c) / (N - 1)
    
    return sigma