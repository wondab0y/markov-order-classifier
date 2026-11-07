# Markov-Order Classifier

A project for developing a Markov-order classifier.

Current objective: **Part 1: Classifier Development — Stage 1: Data Generation**.
Step 1 generates 20 zero-order Markov processes. Each process gets one random
probability vector from a symmetric Dirichlet distribution with all parameters
equal to 1. Every symbol is sampled independently using that process's fixed vector.

## Setup

With Python 3 installed, run:

```sh
git clone https://github.com/wondab0y/markov-order-classifier.git
cd markov-order-classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Generate data

```sh
python src/generate_data.py
```

The defaults are 20 processes, 1,000 symbols per process, two possible symbols
(`0` and `1`), and seed 42 for reproducibility. These can be changed through the
arguments to `generate_zero_order_processes()` in `src/generate_data.py`.

The output is `data/zero_order.npz`, containing `sequences` (20 × 1,000) and
`probabilities` (20 × 2). Load it with `np.load("data/zero_order.npz")`.

Step 2 also saves `data/first_order.npz`: 20 first-order chains, each with 1,000
samples and two states by default. Use `generate_first_order_processes()` to
change these settings. The file contains `sequences`, `initial_probabilities`,
`transitions`, and `dependence` labels.

For each chain, the initial distribution and transition rows are sampled from
Dirichlet(a, ..., a). The first state uses the initial distribution; each next
state uses the transition row of the current state. Half the chains use a = 10
for weak dependence and half use a = 0.5 for strong dependence. Matrices are
resampled until the largest total variation distance between rows is at most
0.2 (weak) or at least 0.6 (strong). These labels describe transition-row
differences; they do not guarantee a particular empirical correlation.

Step 3 saves `data/order_2.npz` through `data/order_5.npz`, with **20 processes
per order** (80 total). Each binary sequence has 10,000 symbols, including its
initial k symbols. Use `generate_k_order_processes(order, ...)` for other orders,
counts, lengths, or alphabet sizes; the sequence length must be at least the order.

Each process gets its own conditional probability table, with independently
sampled Dirichlet(1, ..., 1) rows. A row corresponds to a length-k history encoded
as a base-m integer, oldest symbol first. The initial history is uniform; each
subsequent symbol uses the current history's row. The history index updates as
`h = (h % m**(k - 1)) * m + symbol`.

Each file contains `sequences` (20 × 10,000), `conditional_probabilities`
(20 × 2**k × 2), `orders` (one true order per process), `seed` (42), and
`num_symbols` (2). Positive independent Dirichlet rows give exact order k almost
surely in the mathematical model. Finite sequences may not reveal that order;
increase the length for larger orders or rare histories. Table size grows as m**k.

Step 4 saves `data/infinite_order.npz`: **40 infinite-order examples**, split
between 20 Even Processes and 20 Simple Nonunifilar Sources (SNS, q=p).
Use `generate_infinite_order_processes()` to change the count, length, or seed.
Each binary sequence has 10,000 symbols and starts in stationarity. Parameters
p are drawn independently from [0.2, 0.8), avoiding degenerate endpoints.
The file includes observed sequences, family names, p, symbol-labeled matrices,
stationary distributions, infinite-order labels (`np.inf`), and seed.
Only sequences are classifier inputs. See [the models and order proofs](docs/infinite_order.md),
based on Step 4 of the Notion notebook and the cited research.

Run the tests with:

```sh
python -m unittest discover -s tests
```
