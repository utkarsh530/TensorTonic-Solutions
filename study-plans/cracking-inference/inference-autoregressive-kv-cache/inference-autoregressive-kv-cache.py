import torch

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """
    Returns (outputs, key_cache, value_cache), float32 tensors in sequence order.
    """
    B,S,dq = query.shape
    B,S,dk = key.shape
    B,S,dv = value.shape
    kv = torch.zeros(B,S,S)

    output = torch.tensor([])
    k_cache = torch.tensor([])
    v_cache = torch.tensor([])
    score = torch.tensor([])
    for t in range(S):
        # k_t = key[:,:t+1, :]
        # v_t = value[:,:t+1, :]
        # q_t = query[:,t:t+1, :]
        # # print(k_t,v_t, q_t)
        # score_t = q_t @ k_t.transpose(-2,-1)
        # score_t = torch.softmax(score_t, dim = -1) @ v_t
        # print(score_t)

        k_t = key[:,t:t+1, :]
        k_cache = torch.cat((k_cache, k_t), dim = 1)
        v_t = value[:,t:t+1, :]
        v_cache = torch.cat((v_cache, v_t), dim = 1)
        q_t = query[:,t:t+1, :]
        # print(k_t,v_t, q_t)
        # score_t = q_t @ k_t.transpose(-2,-1)
        # print(k_cache.shape)
        score = q_t @ k_cache.transpose(-2,-1)
        # score = torch.cat((score, score_t))
        # print(score.shape)
        score = score / (dk ** 0.5)
        # score_t = q_t @ k_cache.transpose(-2,-1)
        output = torch.cat((output, torch.softmax(score, dim = -1) @ v_cache), dim = 1)
        # print(score)

    return (output, k_cache, v_cache)
