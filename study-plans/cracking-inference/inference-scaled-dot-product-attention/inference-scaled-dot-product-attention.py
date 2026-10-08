import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    B,Sq,d = query.shape 
    score = query @ key.transpose(-2,-1)
    score = score / (d ** 0.5)
    if mask is not None:
        score = score.masked_fill(mask, float("-inf"))
    Y = torch.softmax(score, dim = -1)
    Y = Y @ value
    return Y
