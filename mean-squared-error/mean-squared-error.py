import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    
    # Point to existing lists as NumPy arrays
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)

    # Compute, square, and store each error residual
    squared_error = np.square(pred - true)

    # Compute and return the MSE
    return np.mean(squared_error)