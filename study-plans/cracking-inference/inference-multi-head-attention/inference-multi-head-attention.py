import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states.
    """

    Q = hidden_states @ w_q
    K = hidden_states @ w_k
    V = hidden_states @ w_v
    B,S,D = hidden_states.shape
    d = D // num_heads
    Q = Q.view(B, S, num_heads, D // num_heads).permute(0,2,1,3)
    K = K.view(B, S, num_heads, D // num_heads).permute(0,2,1,3)
    V = V.view(B, S, num_heads, D // num_heads).permute(0,2,1,3)

    scores = Q @ K.transpose(-2,-1)
    scores = scores / (d ** 0.5)


    # if causal:
    #     mask = torch.zeros(scores.shape)
    #     for i in range(S):
    #         for j in range(S):
    #             if i < j:
    #                 mask[:,:,i,j] = float("-inf")
    #     scores = scores + mask

    if causal:
    # 1. Create an upper triangular matrix of True values (excluding the diagonal)
    #    Shape: (S, S)
        mask = torch.triu(torch.ones(S, S, device=scores.device), diagonal=1).bool()
        
        # 2. Fill all True positions in the scores with -inf
        scores = scores.masked_fill(mask, float("-inf"))
        
    Y  = torch.softmax(scores, dim = -1)
    print("Y", Y.shape)
    Y  = Y @ V
    #B H S D
    Y = Y.transpose(1,2).reshape(B, S, D)
    Y = Y @ w_o
    # print(Y.shape)

    return Y
