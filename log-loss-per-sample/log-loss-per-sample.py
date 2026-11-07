import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Create a list to store our losses
    losses = []
    
    for y, p in zip(y_true, y_pred):
        # Clip probability before applying logarithm
        p_clipped = min(1.0 - eps, max(eps, float(p)))

        # Apply BCE log loss function on current sample
        loss = -(y * math.log(p_clipped) + (1 - y) * math.log(1.0 - p_clipped))

        # Add computed loss to our list
        losses.append(loss)

    # Return losses
    return losses
    