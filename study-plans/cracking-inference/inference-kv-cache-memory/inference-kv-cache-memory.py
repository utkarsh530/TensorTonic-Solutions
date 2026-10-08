import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns an int64 tensor of byte counts ordered [MHA, MQA, GQA, MLA].
    """
    ans = []
    mha = 2 * batch_size * seq_len * num_layers * num_query_heads * head_dim * bytes_per_element
    
    mqa = 2 * batch_size * seq_len * num_layers  * head_dim * bytes_per_element

    gqa = 2 * batch_size * seq_len * num_layers * gqa_kv_heads * head_dim * bytes_per_element
    
    mla = batch_size * seq_len * num_layers *(mla_latent_dim + mla_rotary_key_dim) * bytes_per_element
    

    return torch.tensor([mha,mqa,gqa,mla], dtype = torch.int64)
    pass
