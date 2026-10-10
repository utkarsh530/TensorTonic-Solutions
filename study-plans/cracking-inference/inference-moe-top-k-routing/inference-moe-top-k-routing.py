import torch

def route_tokens_to_experts(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple:
    """
    Returns (expert_indices, routing_weights), tensors shaped (num_tokens, top_k).
    """
    # values, indices = torch.topk(router_logits, top_k, dim = -1)
    # print(values, indices)
    # experts = values.to(torch.float32)
    # expert_indices = indices.to(torch.int64)
    # print(expert_indices, experts)
    # routing_weights = torch.softmax(experts, dim = -1)
    # return (expert_indices, routing_weights)
    sorted_vals, indices = torch.sort(router_logits, dim = -1, descending = True)
    experts = sorted_vals[:, :top_k]
    expert_indices = indices[:, :top_k]
    routing_weights = torch.softmax(experts, dim = -1)

    print(expert_indices, routing_weights)
    return (expert_indices, routing_weights)