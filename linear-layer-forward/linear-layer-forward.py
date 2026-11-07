def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Point to X, W, and b as corresponding NumPy arrays
    input = np.asarray(X)
    weight = np.asarray(W)
    bias = np.asarray(b)

    # Apply matrix operations for affine transformation
    # linear layer output = input matrix * weight matrix + bias vector
    Y = input @ weight + bias

    # Return as a list
    return Y.tolist()