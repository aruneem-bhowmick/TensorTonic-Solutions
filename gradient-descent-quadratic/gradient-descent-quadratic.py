def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    
    # Initialize the parameter with given starting point
    x = x0

    # Update our parameter in given number of steps
    for _ in range(steps):
        # f' = d/dx [ax^2 + bx + c] = 2ax + b
        # f' is our gradient, and represents the slope of the objective function f at x
        gradient = 2 * a * x + b

        # Update our parameter by stepping in the negative gradient direction
        # Learning rate lr controls the magnitude of each update
        x -= lr * gradient

    # Return our final computed parameter
    return x