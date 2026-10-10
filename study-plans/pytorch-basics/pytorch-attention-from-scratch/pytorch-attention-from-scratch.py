import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 attention tensor of shape (batch, seq_q, d_v).
    """
    d = Q.shape[-1]
    score = (Q @ K.transpose(-2,-1)) / (d ** 0.5)

    out = torch.softmax(score, dim = -1) @ V

    return out