import torch

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 tensor of shape (batch, seq_q, d_v).
    """
    B, Sq, D = Q.shape
    

    score = Q @ K.transpose(-2,-1)
    score = score / (D ** 0.5)

    mask = torch.triu(torch.ones(score.shape), diagonal = 1).bool()
    score = score.masked_fill(mask, float("-inf"))
    output = torch.softmax(score, dim = -1) @ V
    return output
