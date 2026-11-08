import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """

    X_arr = np.array(X, dtype = float)
    
    # 1. Center the data by subtracting the mean of each feature
    mean = np.mean(X_arr, axis = 0)
    X_c = X_arr - mean
    
    # 2. Compute the d x d sample covariance matrix
    n = X_arr.shape[0]
    cov_matrix = (X_c.T @ X_c) / (n - 1)
    
    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvectors by eigenvalues in descending order and select top-k
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_indices = sorted_indices[:k]
    W = eigenvectors[:, top_k_indices]
    
    # 5. Project the centered data onto the top-k eigenvectors
    X_proj = X_c @ W
    
    return X_proj.tolist()