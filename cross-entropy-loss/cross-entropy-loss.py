import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_samples = len(y_true)

    # Extract predicted probability for the true class in each sample i: p[i, y_true[i]]
    correct_class_probs = y_pred_arr[np.arange(n_samples), y_true_arr]

    # Compute cross-entropy loss for each sample: L_i = -log(p[i, y_i])
    sample_losses = -np.log(correct_class_probs)

    # Compute and return the mean loss across all samples
    return float(np.mean(sample_losses))