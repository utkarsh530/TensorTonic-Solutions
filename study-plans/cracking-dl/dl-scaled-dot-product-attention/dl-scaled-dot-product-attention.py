import numpy as np


def softmax(S):

    max_row = np.max(S, axis = -1, keepdims=True)
    S_ = S - max_row
    S_ = np.exp(S_)
    A = S_ / np.sum(S_, axis = -1, keepdims=True)
    return A

def scaled_dot_product_attention(Q: list, K: list, V: list, mask: list | None = None, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Returns forward results and optional backward gradients as a dictionary of lists.
    """
    Q = np.array(Q)
    K = np.array(K)
    V = np.array(V)

    sq, d = Q.shape
    sk, d = K.shape

    print(Q.shape, K.transpose(1,0).shape)

    score = Q @ np.transpose(K, (1,0))

    S = score / (d ** 0.5)
    if mask is not None:
        mask = np.array(mask)
        M = np.where(mask, 0, -1e9)
        S = S + M
        
    # print(score, score.shape)
    # max_row = np.max(S, axis = -1, keepdims=True)
    # print(max_row, max_row.shape)
    # S_ = S - max_row
    # print(score)
    # S_ = np.exp(S_)

    # A = S_ / np.sum(S_, axis = -1, keepdims=True)
    A = softmax(S)

    # print(score, score.shape)
    O = A @ V

    # A = np.round(A, decimals=4)
    # O = np.round(O, decimals=4)

    ans = {"output": np.round(O, decimals=4).tolist(), "attention_weights": np.round(A, decimals=4).tolist()}

    if mode == "forward":
        return ans
    elif mode == "backward":
        d_output = np.array(d_output)
        grad_V = A.transpose(1,0) @ d_output
        grad_A = d_output @ V.transpose(1,0)
        grad_S = A * grad_A - A * np.sum(grad_A * A, axis = -1, keepdims = True)
        grad_Q = grad_S @ K / (d ** 0.5)
        grad_K = grad_S.transpose(1,0) @ Q / (d ** 0.5)

        ans['dK'] = np.round(grad_K, decimals=4).tolist()
        ans['dQ'] = np.round(grad_Q, decimals=4).tolist()
        ans['dV'] = np.round(grad_V, decimals=4).tolist()

        return ans
    
