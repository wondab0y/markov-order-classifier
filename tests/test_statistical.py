from math import isfinite, log
import unittest

import numpy as np

from src.generate_data import (
    generate_first_order_processes, generate_infinite_order_processes,
    generate_k_order_processes, generate_zero_order_processes,
)
from src.statistical import (
    estimate_markov_order, fitted_log_likelihood, transition_counts,
)


class TestBICEstimator(unittest.TestCase):
    def test_manual_counts_likelihood_mle_and_bic(self):
        sequence = [0, 1, 0, 0, 1, 1]
        counts = transition_counts(sequence, k=0, m=2, max_order=2)
        self.assertEqual(counts, {(): {0: 2, 1: 2}})
        self.assertEqual(transition_counts(sequence, 1, 2, 2),
                         {(1,): {0: 1, 1: 1}, (0,): {0: 1, 1: 1}})
        self.assertEqual(transition_counts(sequence, 2, 2, 2),
                         {(0, 1): {0: 1, 1: 1}, (1, 0): {0: 1}, (0, 0): {1: 1}})
        self.assertAlmostEqual(fitted_log_likelihood(counts), -4 * log(2))
        result = estimate_markov_order(sequence, max_order=2)
        self.assertEqual(result["estimated_order"], 0)
        self.assertEqual(result["n_eff"], 4)
        self.assertEqual(result["conditional_probabilities"], {(): {0: 0.5, 1: 0.5}})
        self.assertAlmostEqual(result["scores"][0]["bic"], 8 * log(2) + log(4))
        self.assertEqual([row["parameters"] for row in result["scores"]], [1, 2, 4])

    def test_common_evaluation_window(self):
        sequence = np.random.default_rng(42).integers(3, size=100)
        for k in range(5):
            counts = transition_counts(sequence, k, 3, 4)
            self.assertEqual(sum(sum(row.values()) for row in counts.values()), 96)

    def test_unobserved_histories_and_zero_counts(self):
        counts = transition_counts([0] * 100, 3, 2, 4)
        self.assertEqual(counts, {(0, 0, 0): {0: 96}})
        self.assertEqual(fitted_log_likelihood(counts), 0)
        self.assertEqual(fitted_log_likelihood({(): {0: 0, 1: 2}, (1,): {0: 0}}), 0)
        result = estimate_markov_order([0] * 100, max_order=4)
        self.assertEqual(result["estimated_order"], 0)
        self.assertEqual(result["conditional_probabilities"], {(): {0: 1, 1: 0}})
        self.assertTrue(all(isfinite(row["bic"]) for row in result["scores"]))

    def test_generated_orders_zero_one_two(self):
        for seed in (7, 42):
            iid, _ = generate_zero_order_processes(num_processes=1, sequence_length=20_000, seed=seed)
            first, _, _, _ = generate_first_order_processes(num_processes=2, sequence_length=20_000, seed=seed)
            second, _, _ = generate_k_order_processes(2, num_processes=1, sequence_length=20_000, seed=seed)
            for order, sequence in ((0, iid[0]), (1, first[1]), (2, second[0])):
                with self.subTest(seed=seed, order=order):
                    result = estimate_markov_order(sequence, max_order=5)
                    self.assertEqual(result["estimated_order"], order)
                    for row in result["conditional_probabilities"].values():
                        self.assertAlmostEqual(sum(row.values()), 1)

    def test_infinite_processes_have_finite_estimates(self):
        data = generate_infinite_order_processes(num_processes=2, sequence_length=5000, seed=7)
        for sequence in data["sequences"]:
            result = estimate_markov_order(sequence, max_order=5)
            self.assertIn(result["estimated_order"], range(6))
            self.assertEqual(len(result["scores"]), 6)
            self.assertTrue(all(isfinite(row["bic"]) for row in result["scores"]))
            self.assertNotIn("order", result)  # No exact certification field.

    def test_sparse_histories_use_sparse_counts(self):
        sequence = [0] * 100
        with self.assertWarnsRegex(RuntimeWarning, "sparse"):
            result = estimate_markov_order(sequence, max_order=50)
        self.assertEqual(len(result["scores"]), 51)
        self.assertEqual(len(transition_counts(sequence, 50, 2, 50)), 1)

    def test_order_zero_and_numpy_arguments(self):
        result = estimate_markov_order(np.array([0, 1, 0]), max_order=np.int64(0), m=np.int64(2))
        self.assertEqual(result["n_eff"], 3)
        self.assertEqual(result["estimated_order"], 0)
        self.assertAlmostEqual(result["conditional_probabilities"][()][0], 2 / 3)

    def test_invalid_input(self):
        for sequence in ([], [0], [0, 2, 0], [0, -1, 0], [0.0, 1.0, 0.0],
                         [0, True, 0], [[0, 1], [1, 0]], "010", None):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                estimate_markov_order(sequence, max_order=0)
        for kwargs in ({"max_order": -1}, {"max_order": 1.5}, {"max_order": True},
                       {"max_order": 3}, {"max_order": 2}, {"max_order": 0, "m": 1},
                       {"max_order": 0, "m": 2.0}, {"max_order": 0, "m": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_markov_order([0, 1, 0], **kwargs)
        for k in (-1, 3, 0.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                transition_counts([0, 1, 0, 1], k, 2, 2)


if __name__ == "__main__":
    unittest.main()
