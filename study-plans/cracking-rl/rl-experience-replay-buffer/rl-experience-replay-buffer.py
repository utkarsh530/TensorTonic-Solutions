def replay_buffer_sample(capacity: int, transitions: list, sample_indices: list) -> tuple:
    """
    Returns (states, actions, rewards, next_states, dones), each a list in sample order.
    """
    n = len(sample_indices)
    states = [0.0] * capacity
    actions = [0.0] * capacity
    rewards = [0.0] * capacity
    next_states = [0.0] * capacity
    dones = [0.0] * capacity
    if not sample_indices:
        return ([], [], [], [], [])

    nt = len(transitions)

    for i in range(nt):
        states[ i % capacity] = transitions[i][0]
        actions[ i % capacity] = transitions[i][1]
        rewards[ i % capacity] = transitions[i][2]
        next_states[ i % capacity] = transitions[i][3]
        dones[ i % capacity] = transitions[i][4]

    astates = []
    aactions = []
    arewards = []
    anext_states = []
    adones = []

    for i in sample_indices:
        astates.append(states[i])
        aactions.append(actions[i])
        arewards.append(round(float(rewards[i]), 4))
        anext_states.append(next_states[i])
        adones.append(dones[i])

    return (astates, aactions, arewards, anext_states, adones)