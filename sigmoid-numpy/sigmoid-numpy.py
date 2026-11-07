import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    
    z = np.array(x, dtype = float)
    return 1.0 / (1.0 + np.exp(-z))