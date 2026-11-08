import numpy as np


def higher_block_transition(Q, order, num_symbols):
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if isinstance(num_symbols, (bool, np.bool_)) or not isinstance(num_symbols, (int, np.integer)) or num_symbols < 2:
        raise ValueError("num_symbols must be an integer >= 2")

    m, k = int(num_symbols), int(order)
    n = m ** k
    if np.iscomplexobj(Q):
        raise ValueError("Q must contain real probabilities")
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (n, m) or not np.all(np.isfinite(Q)):
        raise ValueError("Q has invalid shape or nonfinite entries")
    if np.any(Q < 0) or not np.allclose(Q.sum(axis=1), 1.0):
        raise ValueError("Q must be row-stochastic")

    T = np.zeros((n, n), dtype=float)
    for history in range(n): # encoded history goes from 0 to n-1
        for symbol in range(m):
            next_history = (history % (m ** (k - 1))) * m + symbol
            T[history, next_history] = Q[history, symbol]
    return T
