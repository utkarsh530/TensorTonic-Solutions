def per_priorities_and_weights(td_errors: list, alpha: float, beta: float, epsilon: float = 1e-6) -> tuple:
    """
    Returns (probs, is_weights), two float lists rounded to four decimals.
    """
    N = len(td_errors)
    p = [(abs(td_error) + epsilon) ** alpha for td_error in td_errors]
    sump = sum(p)
    P = [pi / sump for pi in p]
    w = [(N*Pi) ** (-beta) for Pi in P]
    maxw = max(w)
    wtilde = [wi / maxw for wi in w]

    # wtilde = w / max(w)
    for i in range(N):
        wtilde[i] = round(wtilde[i], 4)
        P[i] = round(P[i], 4)
    return (P, wtilde)