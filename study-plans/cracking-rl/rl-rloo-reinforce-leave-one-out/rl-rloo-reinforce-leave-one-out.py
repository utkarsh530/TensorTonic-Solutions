def rloo_advantages(rewards: list, group_indices: list) -> list:
    """
    Returns the leave-one-out advantage list, rounded to four decimals.
    """
    n = len(rewards)
    A = [0.0] * n
    grp_dict = {}
    for i in range(n):
        grp_dict[group_indices[i]] = grp_dict.get(group_indices[i], [])
        grp_dict[group_indices[i]].append(i)
        
    for i in range(n):
        sum_i  = 0
        for idx in grp_dict[group_indices[i]]:
            if i != idx:
                sum_i += rewards[idx]
        A[i] = rewards[i] - (1 / (len(grp_dict[group_indices[i]]) - 1)) * sum_i
        A[i] = round(A[i], 4)
        
    return A