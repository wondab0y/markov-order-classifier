# Step 4: Infinite-order ground truth

Theory backbone: [Step 4 in the Notion notebook](https://app.notion.com/p/407f3f38ee8e8187b1baf4648f3994e4).
We use its convention `M[a, i, j] = P(X_t=a, S_(t+1)=j | S_t=i)`.
Rows of `A = M[0] + M[1]` sum to one. Word probabilities are
`pi @ M[x_0] @ ... @ M[x_(n-1)] @ ones(2)`.

The dataset contains 20 parameter settings per family, 40 sequences total,
each of length 10,000. Each example independently draws p uniformly in [0.2, 0.8)
and starts its hidden state from the stationary distribution. Seed 42 reproduces
the dataset. Ground truth comes from the constructions and proofs below;
testing finitely many histories cannot establish infinite order by itself.

## Even Process

The notebook's model is also discussed in Marzen and Crutchfield,
[Circumventing the Curse of Dimensionality in Prediction](https://arxiv.org/abs/1412.2859),
Fig. 3. States are A (even parity) and B (odd parity).

```text
M[0] = [[p, 0],       M[1] = [[0, 1-p],
        [0, 0]]              [1, 0]]
pi = [1, 1-p] / (2-p)
```

Allowed domain: 0 < p < 1. Only even-length runs of ones occur between zeros.
Following a zero, histories `0 1^k` and `0 1^(k+1)` have the same last k
symbols but opposite parity. Their next-zero probabilities are p and 0
(in parity-dependent order). Both histories have positive probability for
every finite k. Thus no finite history length suffices. Endpoints are excluded.
With stationary initialization, an incomplete boundary run need not be even.

## Simple Nonunifilar Source (SNS)

Source: Marzen and Crutchfield,
[Informational and Causal Architecture of Discrete-Time Renewal Processes](https://arxiv.org/abs/1408.6876),
Sec. V, Fig. 7, Eq. (15). We restrict the published two-parameter family to q=p.
Its edges are A --0,p--> A, A --0,1-p--> B, B --0,p--> B,
and B --1,1-p--> A. Two different next states can emit 0 from A.

```text
M[0] = [[p, 1-p],     M[1] = [[0, 0],
        [0, p]]              [1-p, 0]]
pi = [1/2, 1/2]
```

Allowed domain: 0 < p < 1. Consecutive ones are forbidden.
Here is a direct derivation of the infinite-order property for this restriction.
After a 1 the state resets to A. If N is the number of zeros until the next 1,
the waiting time is the sum of two geometric stages, giving
`P(N=n) = n (1-p)^2 p^(n-1)` for n >= 1.
Summing the tail gives `P(N>=n) = p^(n-1) [p+n(1-p)]`.
Consequently the next-one probability after `1 0^n` is
`H(n) = n(1-p)^2 / [p+n(1-p)]`.
Its increment is
`H(n+1)-H(n) = p(1-p)^2 / ([p+n(1-p)] [p+(n+1)(1-p)]) > 0`.
For every finite k, histories `1 0^k` and `1 0^(k+1)` therefore share their
last k symbols but give different predictions. Endpoints are excluded.

## Validation and use

Tests check stochastic matrices, stationary distributions, family constraints,
observed word frequencies, SNS gap probabilities, exact rational history
witnesses, reproducibility, and serialization without pickle. The analytic
arguments justify `orders = np.inf`; empirical checks validate the implementation.

`data/infinite_order.npz` stores `sequences`, `families`, `p`, `symbol_matrices`,
`stationary_distributions`, `orders`, and `seed`. For SNS, q=p is
part of the family definition. Only `sequences` should be classifier inputs.
Metadata supplies labels and exact models for evaluation. Finite samples may
not reveal all predictive memory; exact order and effective memory differ.
