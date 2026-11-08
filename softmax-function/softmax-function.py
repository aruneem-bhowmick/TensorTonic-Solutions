import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """

    # Point to logits as a NumPy array x
    x = np.asarray(x, dtype = float)

    # Get the maximum logit in the row
    m = np.max(x, axis = -1, keepdims = True)

    # Store each logit raised as a power of e
    exp_x = np.exp(x - m)

    # Convert all logits into probabilities
    return exp_x / np.sum(exp_x, axis = -1, keepdims = True)