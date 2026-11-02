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

Run the tests with:

```sh
python -m unittest discover -s tests
```
