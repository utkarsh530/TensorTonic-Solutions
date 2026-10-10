import numpy as np

def softmax(S):

    Smax = np.max(S, axis = -1, keepdims=True)
    S_shift = S - Smax
    P = np.exp(S_shift) / np.sum(np.exp(S_shift), axis = -1, keepdims = True)
    return P

def scaled_dot_product_attention(Q: list, K: list, V: list) -> np.ndarray:
    """
    Returns the scaled-attention output as a float64 array.
    """
    Q = np.array(Q)
    K = np.array(K)
    V = np.array(V)

    d = Q.shape[-1]

    score = Q @ K.transpose((1,0))
    score = score / (d ** 0.5)
    output = softmax(score) @ V
    return output