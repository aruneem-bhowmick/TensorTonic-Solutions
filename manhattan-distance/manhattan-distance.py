import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    a = np.asarray(x)
    b = np.asarray(y)

    # Manhattan distance: sum of absolute differences 
    return np.sum(np.abs(a - b), dtype = 'float')