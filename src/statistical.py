from math import fsum, log
from numbers import Integral
import warnings

import numpy as np


def _validate(sequence, max_order, m): #sequence + max order we want to test + alphabet size
    for value, name, minimum in ((m, "m", 2), (max_order, "max_order", 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        sequence = tuple(sequence)
    except TypeError as exc:
        raise ValueError("sequence must be a one-dimensional sequence of integers") from exc
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
           or x < 0 or x >= m for x in sequence):
        raise ValueError("sequence must contain integer symbols in 0..m-1")
    if max_order >= len(sequence) or len(sequence) - max_order <= 1:
        raise ValueError("Require max_order < sequence length and N - max_order > 1")
    return tuple(int(x) for x in sequence), int(max_order), int(m)


def transition_counts(sequence, k, m, max_order):
    sequence, max_order, m = _validate(sequence, max_order, m)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or not 0 <= k <= max_order:
        raise ValueError("k must be an integer in 0..max_order")
    counts = {} # dict that stores histories and symbols that follow
    for target_idx in range(max_order, len(sequence)): # start every model at max_order so we train on the same targets
        history = sequence[target_idx-k:target_idx]
        row = counts.setdefault(history, {}) # return or create a dict for the history
        symbol = sequence[target_idx]
        row[symbol] = row.get(symbol, 0) + 1 # increment the count for the symbol that follows the history
    return counts


def fitted_log_likelihood(counts):
    terms = []
    for row in counts.values(): #iterate over histories
        total = sum(row.values())
        for count in row.values(): #iterate over symbols that follow the history
            if count > 0:
                terms.append(count * log(count / total))
    return fsum(terms)


def estimate_markov_order(sequence, max_order=6, m=2):
    sequence, max_order, m = _validate(sequence, max_order, m)
    n_eff = len(sequence) - max_order
    if m ** max_order > n_eff: # means some histories have 9 observations
        warnings.warn("More possible histories than prediction targets at max_order; "
                      "history counts may be sparse.", RuntimeWarning, stacklevel=2)
    scores = [] #candidate orders and their BIC scores
    best_bic = None
    best_counts = None #transition counts for the best order
    best_order = None #order of the current best model
    for k in range(max_order + 1):
        counts = transition_counts(sequence, k, m, max_order)
        likelihood = fitted_log_likelihood(counts)
        parameters = m ** k * (m - 1)
        bic = -2 * likelihood + parameters * log(n_eff)
        scores.append({"order": k, "log_likelihood": likelihood,
                       "parameters": parameters, "bic": bic})
        if best_bic is None or bic < best_bic:
            best_bic, best_order, best_counts = bic, k, counts
    probabilities = {
        history: {symbol: row.get(symbol, 0) / sum(row.values()) for symbol in range(m)}
        for history, row in best_counts.items()
    } # return a dict of dicts with conditional probabilities for each history and symbol
    return {"estimated_order": best_order, "n_eff": n_eff, "scores": scores,
            "conditional_probabilities": probabilities}
