import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Point to X, y as NumPy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # w = (X^T @ X)^-1 @ (X ^T @ y)
    # closed-form solution: (X^T @ X) = (X ^T @ y)
    return np.linalg.solve(X.T @ X, X.T @ y)