import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Write code here
    mask = np.triu(np.ones(scores.shape), 1).astype(bool)

    output = np.where(mask, mask_value, scores)

    return output