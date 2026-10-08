import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))
    denom = np.array(base ** ((np.arange(0, d_model, 2)) / d_model))
    odd_len = d_model // 2
    # 1,3,5
    # 0 2 4 6
    for pos in range(seq_len):
        pe[pos, ::2] = np.sin(pos / denom)
        pe[pos, 1::2] = np.cos(pos / denom)[:odd_len]

    return pe