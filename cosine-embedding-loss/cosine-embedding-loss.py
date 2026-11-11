import math

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """

    # Compute dot product and vector norms for cosine similarity
    dot_product = sum(a * b for a, b in zip(x1, x2))
    norm_x1 = math.sqrt(sum(a * a for a in x1))
    norm_x2 = math.sqrt(sum(b * b for b in x2))
    
    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    
    # Compute loss based on label
    if label == 1:
        loss = 1.0 - cosine_similarity
    else:
        loss = max(0.0, cosine_similarity - margin)
        
    return float(loss)