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
    """Generate chains with weak or strong differences between transition rows."""
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


if __name__ == "__main__":
    sequences, probabilities = generate_zero_order_processes()
    output_path = Path(__file__).resolve().parents[1] / "data" / "zero_order.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, sequences=sequences, probabilities=probabilities)
    print(f"Saved {len(sequences)} processes to {output_path}")

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
