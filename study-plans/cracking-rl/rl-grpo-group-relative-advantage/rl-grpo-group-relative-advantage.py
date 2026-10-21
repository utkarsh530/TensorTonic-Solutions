
import math

def grpo_advantages(rewards: list, group_indices: list, eps: float = 1e-8) -> list:
    groups = {}

    for i, grp in enumerate(group_indices):
        groups.setdefault(grp, []).append(i)

    ans = [0.0] * len(rewards)

    for indices in groups.values():
        vals = [rewards[i] for i in indices]

        # Only constant-reward groups get zero advantages
        if all(x == vals[0] for x in vals):
            continue

        mean = math.fsum(vals) / len(vals)

        variance = math.fsum(
            (x - mean) ** 2 for x in vals
        ) / len(vals)

        std = math.sqrt(variance)

        for idx in indices:
            ans[idx] = round(
                (rewards[idx] - mean) / (std + eps),
                4
            )

    return ans
