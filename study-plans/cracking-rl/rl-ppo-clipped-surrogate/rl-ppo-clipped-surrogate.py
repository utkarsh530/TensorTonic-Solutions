import math

def ppo_clipped_loss(log_probs_new: list, log_probs_old: list, advantages: list, clip_eps: float) -> float:
    """
    Returns the PPO clipped loss as a float rounded to four decimals.
    """
    n = len(log_probs_new)
    reward = [math.exp(log_probs_new[i] - log_probs_old[i]) for i in range(n)]

    ans = 0

    for i in range(n):
        ans += min(reward[i] * advantages[i], min(max(reward[i], 1 - clip_eps), 1 + clip_eps) * advantages[i])
        
    return (-1 / n) * ans