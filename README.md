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

## Stage 2: Higher-block presentation

`src/higher_block.py` converts a conditional probability table into a first-order
transition matrix on overlapping length-k histories:

```python
from src.higher_block import higher_block_transition

Q = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]]
T = higher_block_transition(Q, order=2, num_symbols=2)
```

For a saved Stage 1 dataset, pass one table at a time, such as
`data["conditional_probabilities"][0]`. Rows and columns of T use base-m history
indices (00, 01, 10, 11 in the example). Only overlapping histories can connect.
The dense matrix has shape `(m**k, m**k)`, so storage grows quickly with order.

Sequence probabilities agree when both representations use the same initial
history distribution. Keep Stage 1's uniform initialization for direct comparison.
A stationary distribution instead satisfies `rho @ T == rho` and sums to 1;
for the example it is `(1/4, 1/12, 1/12, 7/12)`.

## Stage 3: Symbolic encoding

`src/encoding.py` labels hidden states and calculates observed-word probabilities:

```python
from src.encoding import encode_states, symbol_matrices, word_probability

T = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.4, 0.1, 0.5]]
mapping = [0, 1, 1]
initial = [0.6, 0.3, 0.1]
observations = encode_states([0, 1, 2, 0], mapping)  # [0, 1, 1, 0]
M = symbol_matrices(T, mapping, num_symbols=2)
p = word_probability([0, 1], T, mapping, initial, num_symbols=2)  # 0.2
```

Here `M[0] = [[0.5, 0, 0], [0.2, 0, 0], [0.4, 0, 0]]` and
`M[1] = [[0, 0.3, 0.2], [0, 0.6, 0.2], [0, 0.1, 0.5]]`. They label
destination states and sum to T. The example word starts after the first transition,
so its probability is `(0.6*0.5 + 0.3*0.2 + 0.1*0.4) * (0.3 + 0.2) = 0.2`.

`word_probability` computes `P(X_1,...,X_r)` with `X_t = mapping[S_t]`,
using the initial distribution of `S_0`. It omits the initial state's label.
To compare with an encoded hidden trajectory, use `encode_states(states[1:], mapping)`.
For Stage 2 order-k block states labeled by their last symbol, the first transition
emits original `X_k`; the k symbols of the initial history are omitted.

Merging states can make earlier observations relevant to predicting the next
symbol. If states sharing a label have equal transition probabilities into every
label class (strong lumpability), the observed process is first-order Markov
for every initial distribution. Failure of that condition alone does not prove
non-Markov behavior for a particular initial distribution.

## Stage 4: Exact classifier

```python
from src.stage4_pipeline import classify_hidden_chain, classify_higher_block

result = classify_hidden_chain(
    T=[["4/5", "1/5"], ["3/10", "7/10"]],
    mapping=[0, 1],
)
print(result["order"])  # 1

# Q is a rational conditional table with m**k rows and m columns.
# result = classify_higher_block(Q, order=k, m=2)
```

The pipeline validates the model, computes its stationary distribution, minimizes
the representation, and tests symbol products using exact rank-one checks.
Products of rank at most one are pruned. The search stops at the first successful
order or at `d*(d-1)//2`, where `d` is the minimal dimension. Dimension one gives
order zero; failure through the bound certifies infinite order.

Results contain `order` (`None` for infinity), `minimal_dimension`, `search_bound`,
and `diagnostics` with the number of surviving products at each tested order.
For edge-labeled models, use `classify_symbol_matrices({"0": M0, "1": M1})`.
All entry points accept an optional stationary `pi`. When it is omitted and the
stationary distribution is not unique, they return `stationary_family` (a SymPy
row matrix), `free_parameters`, and nonnegativity `constraints` instead of a
classification. Choose real parameter values satisfying these constraints and
pass the resulting probability vector as `pi`:

```python
T = [[1, 0], [0, 1]]
family = classify_hidden_chain(T, [0, 1])
parameter = family["free_parameters"][0]
pi = list(family["stationary_family"].subs(parameter, "1/2"))
result = classify_hidden_chain(T, [0, 1], pi=pi)
```

Nonstationary initial distributions and negative probabilities are rejected.

Use integers, fractions, or quoted rational strings. Decimals are interpreted
exactly as written, and row sums must equal one exactly. Existing floating-point
Stage 2–3 helpers are not used by this pipeline. This classifier needs known
model parameters, not sample sequences or estimated frequencies. The Stage 1
sample-only classifier remains a separate task. Larger higher-block models can
be expensive because the state space and exact arithmetic grow with order.

Run the tests with:

```sh
python -m unittest discover -s tests
```
