from pathlib import Path

import numpy as np


def generate_zero_order_processes(
    num_processes=20, sequence_length=1000, num_symbols=2, seed=42
):
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.ones(num_symbols), size=num_processes) #gen probability vector
    sequences = np.array([
        rng.choice(num_symbols, size=sequence_length, p=p)
        for p in probabilities
    ]) # generate sequences based on established p
    return sequences, probabilities


def generate_first_order_processes(
    num_processes=20, sequence_length=1000, num_symbols=2, seed=42
):
    if num_processes < 1 or sequence_length < 1 or num_symbols < 2:
        raise ValueError("Use at least one process, one sample, and two symbols.")

    rng = np.random.default_rng(seed)
    sequences = np.empty((num_processes, sequence_length), dtype=int)
    initial_probabilities = np.empty((num_processes, num_symbols))
    transitions = np.empty((num_processes, num_symbols, num_symbols)) #generate empty transition matrices for each process
    dependence = np.where(np.arange(num_processes) < num_processes // 2, "weak", "strong") # assign weak dependence to first 10 processes and strong to last 10

    for i, kind in enumerate(dependence):
        alpha = np.full(num_symbols, 10.0 if kind == "weak" else 0.5) #larger alpha for weak depencencies, as it's closer to uniform distribution, which makes transition matrices more similar
        initial_probabilities[i] = rng.dirichlet(alpha)

        while True:
            matrix = rng.dirichlet(alpha, size=num_symbols)
            distance = np.abs(matrix[:, None, :] - matrix[None, :, :]).sum(axis=-1).max() / 2 #max total variation distance
            if (kind == "weak" and distance <= 0.2) or (kind == "strong" and distance >= 0.6):
                break

        transitions[i] = matrix
        sequences[i, 0] = rng.choice(num_symbols, p=initial_probabilities[i]) # generate 0 state using initial probabilities
        for t in range(1, sequence_length):
            sequences[i, t] = rng.choice(num_symbols, p=matrix[sequences[i, t - 1]]) # sample next state based on previous state row

    return sequences, initial_probabilities, transitions, dependence


def generate_k_order_processes(
    order, num_processes=20, sequence_length=10_000, num_symbols=2, seed=42
):
    for value in (order, num_processes, sequence_length, num_symbols):
        if not isinstance(value, (int, np.integer)):
            raise ValueError("Order, counts, and lengths must be integers.")
    if order < 2 or num_processes < 1 or num_symbols < 2 or sequence_length < order:
        raise ValueError(
            "Use order >= 2, at least one process, at least two symbols, "
            "and sequence_length >= order."
        )

    rng = np.random.default_rng(seed)
    num_histories = num_symbols ** order
    probabilities = rng.dirichlet(
        np.ones(num_symbols), size=(num_processes, num_histories)
    ) #3d array of probabilities for each history in each process
    sequences = np.empty((num_processes, sequence_length), dtype=int)

    for i in range(num_processes):
        # Independent uniform symbols give a uniform initial history.
        sequences[i, :order] = rng.integers(num_symbols, size=order) #assign initial history to first k symbols
        history = 0
        for symbol in sequences[i, :order]:
            history = history * num_symbols + int(symbol) #for each symbol in the initial history, shift to the left and add new symbol

        for t in range(order, sequence_length): # 0 to order-1 are already filled
            symbol = rng.choice(num_symbols, p=probabilities[i, history]) #choose next symbol based on conditional probability table
            sequences[i, t] = symbol
            history = (history % (num_symbols ** (order - 1))) * num_symbols + int(symbol) #explained in notion

    return sequences, probabilities, np.full(num_processes, order, dtype=int)


def infinite_order_model(family, p):
    if not np.isfinite(p) or not 0 < p < 1:
        raise ValueError("Use 0 < p < 1 to avoid degeneracies.")
    if family == "even":
        matrices = np.array([[[p, 0], [0, 0]], [[0, 1 - p], [1, 0]]])
        stationary = np.array([1, 1 - p]) / (2 - p)
    elif family == "sns":
        matrices = np.array([[[p, 1 - p], [0, p]], [[0, 0], [1 - p, 0]]])
        stationary = np.array([0.5, 0.5])
    else:
        raise ValueError("Unknown family; choose 'even' or 'sns'.")
    return matrices, stationary


def generate_infinite_order_processes(num_processes=40, sequence_length=10_000, seed=42):
    for value in (num_processes, sequence_length, seed):
        if not isinstance(value, (int, np.integer)):
            raise ValueError("Counts, length, and seed must be integers.")
    if num_processes < 2 or sequence_length < 1 or seed < 0:
        raise ValueError("Use at least two processes, one symbol, and a nonnegative seed.")

    rng = np.random.default_rng(seed)
    families = np.where(np.arange(num_processes) < num_processes // 2, "even", "sns")
    parameters = rng.uniform(0.2, 0.8, size=num_processes)
    sequences = np.empty((num_processes, sequence_length), dtype=int)
    symbol_matrices = np.empty((num_processes, 2, 2, 2)) # indexing convention: [process, symbol, current_state, next_state]
    stationary_distributions = np.empty((num_processes, 2))

    for i, (family, p) in enumerate(zip(families, parameters)):
        matrices, stationary = infinite_order_model(family, p)
        symbol_matrices[i] = matrices
        stationary_distributions[i] = stationary
        state = rng.choice(2, p=stationary) # sample initial state
        for t in range(sequence_length):
            outcome = rng.choice(4, p=matrices[:, state, :].ravel()) # flatten probability matrix for current state and sample next outcome
            symbol, state = divmod(int(outcome), 2) # 0 gives (0,0), 1 gives (0,1), 2 gives (1,0), 3 gives (1,1)
            sequences[i, t] = symbol

    return {
        "sequences": sequences,
        "symbol_matrices": symbol_matrices,
        "stationary_distributions": stationary_distributions,
        "families": families,
        "p": parameters,
        "orders": np.full(num_processes, np.inf),
        "seed": np.array(seed),
    }


AIRPORT_PROBABILITIES = {
    2: np.array([0.0410, 0.4978, 0.1140, 0.8374]),
    3: np.array([
        0.0400, 0.4815, 0.0883, 0.6848,
        0.0675, 0.6270, 0.1400, 0.8671,
    ]),
    4: np.array([
        0.0387, 0.4946, 0.0881, 0.6655,
        0.0559, 0.5451, 0.0996, 0.8231,
        0.0678, 0.3023, 0.0956, 0.7987,
        0.0798, 0.6809, 0.1581, 0.8738,
    ]),
}


def airport_transition_table(order):
    if order not in AIRPORT_PROBABILITIES:
        raise ValueError("available orders: 2, 3, 4.")
    p1 = AIRPORT_PROBABILITIES[order]
    return np.column_stack((1 - p1, p1))


def generate_airport_processes(sequence_length=10_000, seed=42):
    if not isinstance(sequence_length, (int, np.integer)) or sequence_length < 4:
        raise ValueError("use an integer sequence_length >= 4.")
    rng = np.random.default_rng(seed)
    orders = np.array([2, 3, 4])
    sequences = np.empty((3, sequence_length), dtype=int)
    transition_tables = {}

    for i, order in enumerate(orders):
        table = airport_transition_table(int(order))
        transition_tables[int(order)] = table
        sequences[i, :order] = rng.integers(0, 2, size=order)
        history = 0
        for symbol in sequences[i, :order]:
            history = history * 2 + int(symbol)
        for t in range(order, sequence_length):
            symbol = rng.choice(2, p=table[history])
            sequences[i, t] = symbol
            history = (history % (2 ** (order - 1))) * 2 + int(symbol)

    return sequences, orders, transition_tables


if __name__ == "__main__":
    sequences, probabilities = generate_zero_order_processes()
    output_path = Path(__file__).resolve().parents[1] / "data" / "zero_order.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, sequences=sequences, probabilities=probabilities)
    print(f"Saved {len(sequences)} processes to {output_path}")

    for order in range(2, 6):
        sequences, probabilities, orders = generate_k_order_processes(order)
        higher_order_path = output_path.with_name(f"order_{order}.npz")
        np.savez(
            higher_order_path,
            sequences=sequences,
            conditional_probabilities=probabilities,
            orders=orders,
            seed=42,
            num_symbols=2,
        )
        print(f"Saved {len(sequences)} order-{order} processes to {higher_order_path}")

    sequences, initial_probabilities, transitions, dependence = generate_first_order_processes()
    output_path = output_path.with_name("first_order.npz")
    np.savez(
        output_path,
        sequences=sequences,
        initial_probabilities=initial_probabilities,
        transitions=transitions,
        dependence=dependence,
    )
    print(f"Saved {len(sequences)} processes to {output_path}")

    infinite_data = generate_infinite_order_processes()
    infinite_path = output_path.with_name("infinite_order.npz")
    np.savez(infinite_path, **infinite_data)
    print(f"Saved {len(infinite_data['sequences'])} infinite-order processes to {infinite_path}")

    sequences, orders, transition_tables = generate_airport_processes()
    airport_path = output_path.with_name("airport.npz")
    np.savez(
        airport_path,
        sequences=sequences,
        orders=orders,
        **{f"transition_table_{order}": table for order, table in transition_tables.items()},
    )
    print(f"Saved {len(sequences)} airport processes to {airport_path}")
