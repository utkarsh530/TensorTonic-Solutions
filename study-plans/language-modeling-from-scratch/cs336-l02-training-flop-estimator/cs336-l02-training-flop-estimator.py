def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """
    F_forward = 0
    for i in range(len(matmuls)):
        F_forward += 2 * matmuls[i][0] * matmuls[i][1] * matmuls[i][2]

    F_forward += attention_flops
    F_bkd = 2 * F_forward
    F_tot = F_forward + F_bkd
    
    return {"forward_flops":F_forward, "backward_flops":F_bkd,"total_flops": F_tot}
