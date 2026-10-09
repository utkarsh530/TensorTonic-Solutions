import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    Bq,Sq,D= q.shape
    Bk, Sk, D = k.shape
    d_h = D // num_heads
    q = q.reshape(Bq,Sq, num_heads, d_h)
    k = k.reshape(Bk,Sk, num_heads, d_h)
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    scores = q @ k.transpose(-2,-1)
    scores = scores / (d_h ** 0.5)
    return scores
