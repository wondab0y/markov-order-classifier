from numbers import Integral

from sympy import Matrix, ones, zeros

from .minimization import _rational, minimize_representation
from .rank_one import classify_minimal


def _matrix(rows): # convert to matrix of exact rationals
    try:
        matrix = Matrix(rows)
        return matrix.applyfunc(_rational)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Use a matrix of exact rational numbers") from exc


def _validate_transition(transition):
    n = transition.rows
    if n == 0 or transition.cols != n:
        raise ValueError("T must be a nonempty square matrix")
    if any(x < 0 for x in transition) or transition * ones(n, 1) != ones(n, 1):
        raise ValueError("T must be nonnegative and exactly row-stochastic")


def _stationary(transition, pi): # compute stationary distribution (possibly with free parameters) or validate supplied pi
    n = transition.rows
    if pi is None:
        system = (transition.T - Matrix.eye(n)).col_join(ones(1, n)) #left eigenvector equation with normalization, n+1 x n matrix
        rhs = zeros(n, 1).col_join(Matrix([1])) # rhs is n zeros and 1 for normalization, n+1 x 1 vector
        solution, parameters = system.gauss_jordan_solve(rhs)
        if parameters.rows:
            return solution.T
        initial = solution.T # transpose right eigenvector to get a row vector
    else:
        initial = _matrix([pi])
    if (initial.shape != (1, n) or any(x < 0 for x in initial)
            or sum(initial) != 1 or initial * transition != initial):
        raise ValueError("pi must be a stationary probability vector: pi T = pi")
    return initial


def classify_symbol_matrices(matrices, pi=None): # interface for classification of symbol matrices, including validation and minimization
    if not isinstance(matrices, dict) or not matrices:
        raise ValueError("matrices must be a nonempty object keyed by symbol")
    ms = {symbol: _matrix(rows) for symbol, rows in matrices.items()}
    n = next(iter(ms.values())).rows
    if any(not isinstance(s, str) or not s for s in ms):
        raise ValueError("Each symbol must be a nonempty string")
    if any(m.shape != (n, n) or any(x < 0 for x in m) for m in ms.values()):
        raise ValueError("Symbol matrices must be nonnegative and equally sized square matrices")
    transition = sum(ms.values(), zeros(n, n))
    _validate_transition(transition)
    initial = _stationary(transition, pi)
    if initial.free_symbols:
        return {
            "stationary_family": initial,
            "free_parameters": sorted(initial.free_symbols, key=str),
            "constraints": [entry >= 0 for entry in initial],
        }
    reduced = minimize_representation(
        list(initial), {s: m.tolist() for s, m in ms.items()}, [1] * n)
    return classify_minimal(reduced["matrices"])


def classify_hidden_chain(T, mapping, pi=None): # interface for hidden Markov chain classification
    transition = _matrix(T)
    _validate_transition(transition)
    n = transition.rows
    if (len(mapping) != n or any(isinstance(s, bool) or not isinstance(s, Integral)
                               or s < 0 for s in mapping)):
        raise ValueError("mapping must contain one nonnegative integer label per state")
    matrices = {}
    for symbol in sorted(set(mapping)):
        matrices[str(symbol)] = [
            [transition[i, j] if mapping[j] == symbol else 0 for j in range(n)] #choose cols corresponding to selected symbol
            for i in range(n)]
    return classify_symbol_matrices(matrices, pi=pi)


def classify_higher_block(Q, order, m=2, pi=None): # interface for higher-block classification of a Markov chain
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError("order must be a positive integer")
    if isinstance(m, bool) or not isinstance(m, Integral) or m < 2:
        raise ValueError("m must be an integer >= 2")
    table = _matrix(Q)
    n = m ** order
    if table.shape != (n, m):
        raise ValueError("Q must have shape (m**order, m)")
    transition = zeros(n, n)
    for history in range(n):
        for symbol in range(m):
            next_history = (history % (m ** (order - 1))) * m + symbol
            transition[history, next_history] = table[history, symbol]
    return classify_hidden_chain(transition, [h % m for h in range(n)], pi=pi)
