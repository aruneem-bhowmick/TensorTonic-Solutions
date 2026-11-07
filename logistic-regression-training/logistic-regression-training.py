import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    
    # X is n by d, where n represents # samples and d represents # features
    n, d = X.shape

    # Initialize trained weights and bias (w, b)
    w = np.zeros(d)
    b = 0.0

    # Apply gradient descent for established number of steps
    for _ in range(steps):
        # z = X @ w + b
        z = np.dot(X, w) + b

        # p is probability of z being 1 (according to sigmoid function)
        p = _sigmoid(z)

        # Calculate gradient of loss with respect to w, b
        dw = (1 / n) * np.dot(X.T, (p - y))
        db = (1 / n) * np.sum(p - y)

        # Adjust w and b accordingly
        w -= lr * dw
        b -= lr * db

    # Return learned parameters (w, b)
    return w, float(b)