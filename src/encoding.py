import numpy as np


def _integer_array(values, name):
    values = np.asarray(values)
    if values.size and values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    return values.astype(int)


def encode_states(states, mapping):
    mapping = _integer_array(mapping, "mapping") # [0, 1, 1]
    states = _integer_array(states, "states") # [0, 2, 1, 2, 0]
    if mapping.ndim != 1 or not mapping.size or np.any(mapping < 0):
        raise ValueError("mapping must be a nonempty vector of nonnegative labels")
    if np.any(states < 0) or np.any(states >= len(mapping)):
        raise ValueError("states contain an invalid hidden-state index")
    return mapping[states] # [0, 1, 1, 1, 0]


def symbol_matrices(T, mapping, num_symbols):
    if (isinstance(num_symbols, (bool, np.bool_))
            or not isinstance(num_symbols, (int, np.integer)) or num_symbols < 1):
        raise ValueError("num_symbols must be a positive integer")
    if np.iscomplexobj(T):
        raise ValueError("T must contain real probabilities")
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] == 0 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a nonempty square matrix")
    if (not np.all(np.isfinite(T)) or np.any(T < 0)
            or not np.allclose(T.sum(axis=1), 1)):
        raise ValueError("T must be finite, nonnegative, and row-stochastic")
    mapping = _integer_array(mapping, "mapping")
    if mapping.shape != (len(T),) or np.any(mapping < 0) or np.any(mapping >= num_symbols):
        raise ValueError("mapping must give one label in 0..num_symbols-1 per state")

    M = np.zeros((num_symbols, len(T), len(T))) # T x T matrix for each symbol
    for a in range(num_symbols):
        M[a] = T * (mapping[None, :] == a) # multiply T by a mask that selects columns corresponding to symbol a
    return M


def word_probability(word, T, mapping, initial, num_symbols):
    M = symbol_matrices(T, mapping, num_symbols)

    word = _integer_array(word, "word")
    if word.ndim != 1 or np.any(word < 0) or np.any(word >= num_symbols):
        raise ValueError("Invalid observed word")

    if np.iscomplexobj(initial):
        raise ValueError("Invalid initial distribution")
    initial = np.asarray(initial, dtype=float)
    if (initial.shape != (M.shape[1],)
            or not np.all(np.isfinite(initial))
            or np.any(initial < 0)
            or not np.isclose(initial.sum(), 1)):
        raise ValueError("Invalid initial distribution")

    v = initial.copy()

    for a in word:
        v = v @ M[a]

    return float(v.sum())
