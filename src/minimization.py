from fractions import Fraction

from sympy import Matrix, Rational, zeros


EXAMPLE_4_2 = {
    "alpha": ["1/3", "1/3", "1/3"],
    "matrices": {
        "a": [[0, 0, 0], ["1/2", 0, 0], ["1/2", 0, 0]],
        "b": [[0, "1/2", "1/2"], [0, "1/2", 0], [0, 0, "1/2"]],
    },
    "beta": [1, 1, 1],
}


def _rational(value):
    if isinstance(value, bool):
        raise ValueError("Entries must be rational numbers, not booleans")
    fraction = Fraction(str(value))
    return Rational(fraction.numerator, fraction.denominator)


def parse_representation(alpha, matrices, beta):
    try:
        if not isinstance(matrices, dict) or not matrices:
            raise ValueError("matrices must be a nonempty object keyed by symbol")
        a = Matrix(1, len(alpha), [_rational(x) for x in alpha])
        b = Matrix(len(beta), 1, [_rational(x) for x in beta])
        ms = {}
        n = a.cols
        if n == 0 or b.rows != n:
            raise ValueError("alpha and beta must be nonempty vectors of equal length")
        for symbol, rows in matrices.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("Each symbol must be a nonempty string")
            if len(rows) != n or any(len(row) != n for row in rows):
                raise ValueError(f"Matrix {symbol!r} must have shape {n} x {n}")
            ms[symbol] = Matrix([[_rational(x) for x in row] for row in rows])
        return a, ms, b
    except (TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Use flat vectors and square matrices of rational numbers") from exc


def _row_basis(start, matrices):
    basis = []
    if start.is_zero_matrix:
        return zeros(0, start.cols)
    basis.append(start)
    index = 0
    while index < len(basis):
        row = basis[index]
        for matrix in matrices:
            candidate = row * matrix
            if Matrix.vstack(*basis, candidate).rank() > len(basis):
                basis.append(candidate)
        index += 1
    return Matrix.vstack(*basis)


def minimize_representation(alpha, matrices, beta):
    a, ms, b = parse_representation(alpha, matrices, beta)
    reachable = _row_basis(a, list(ms.values()))
    observable = _row_basis(b.T, [m.T for m in ms.values()]).T
    hankel = reachable * observable
    row_indices = hankel.T.rref()[1] # return a tuple of pivot-col indices
    p = reachable[list(row_indices), :]
    column_indices = (p * observable).rref()[1]
    q = observable[:, list(column_indices)]
    inverse = (p * q).inv()
    return {
        "original_dimension": a.cols,
        "reachable_dimension": reachable.rows,
        "observable_dimension": observable.cols,
        "minimal_dimension": len(row_indices),
        "alpha": a * q * inverse,
        "matrices": {symbol: p * m * q * inverse for symbol, m in ms.items()},
        "beta": p * b,
    }


def word_probability(alpha, matrices, beta, word):
    row = alpha
    for symbol in word:
        row = row * matrices[symbol]
    return (row * beta)[0, 0]
