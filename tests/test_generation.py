import unittest

import numpy as np

from src.generate_data import (
    generate_first_order_processes,
    generate_k_order_processes,
    generate_zero_order_processes,
)


class TestGeneration(unittest.TestCase):
    def test_default_shapes_and_probabilities(self):
        sequences, probabilities = generate_zero_order_processes()
        self.assertEqual(sequences.shape, (20, 1000))
        self.assertEqual(probabilities.shape, (20, 2))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertTrue(np.all((sequences >= 0) & (sequences < 2)))

    def test_seed_reproduces_output(self):
        first = generate_zero_order_processes(seed=7)
        second = generate_zero_order_processes(seed=7)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_samples_follow_each_process_probability_vector(self):
        sequences, probabilities = generate_zero_order_processes(
            num_processes=3, sequence_length=100_000, num_symbols=4
        )
        for sequence, p in zip(sequences, probabilities):
            frequencies = np.bincount(sequence, minlength=4) / len(sequence)
            np.testing.assert_allclose(frequencies, p, atol=0.01, rtol=0)


class TestFirstOrderGeneration(unittest.TestCase):
    def test_probabilities_and_dependence_groups(self):
        sequences, initial, transitions, dependence = generate_first_order_processes()
        self.assertEqual(sequences.shape, (20, 1000))
        self.assertEqual(initial.shape, (20, 2))
        self.assertEqual(transitions.shape, (20, 2, 2))
        self.assertTrue(np.all((sequences >= 0) & (sequences < 2)))
        self.assertTrue(np.all(initial >= 0))
        self.assertTrue(np.all(transitions >= 0))
        np.testing.assert_allclose(initial.sum(axis=1), 1)
        np.testing.assert_allclose(transitions.sum(axis=2), 1)
        self.assertEqual(np.count_nonzero(dependence == "weak"), 10)
        self.assertEqual(np.count_nonzero(dependence == "strong"), 10)
        distances = np.abs(transitions[:, 0] - transitions[:, 1]).sum(axis=1) / 2
        self.assertTrue(np.all(distances[dependence == "weak"] <= 0.2))
        self.assertTrue(np.all(distances[dependence == "strong"] >= 0.6))

    def test_seed_reproduces_output(self):
        first = generate_first_order_processes(num_symbols=4, sequence_length=100, seed=7)
        second = generate_first_order_processes(num_symbols=4, sequence_length=100, seed=7)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_samples_follow_transition_rows(self):
        sequences, _, transitions, _ = generate_first_order_processes(
            num_processes=2, sequence_length=30_000
        )
        for sequence, matrix in zip(sequences, transitions):
            for state, row in enumerate(matrix):
                next_states = sequence[1:][sequence[:-1] == state]
                self.assertGreater(len(next_states), 100)
                frequencies = np.bincount(next_states, minlength=2) / len(next_states)
                np.testing.assert_allclose(frequencies, row, atol=0.03, rtol=0)


class TestHigherOrderGeneration(unittest.TestCase):
    def test_shapes_probabilities_and_orders(self):
        for order in range(2, 6):
            with self.subTest(order=order):
                sequences, probabilities, orders = generate_k_order_processes(
                    order, sequence_length=100
                )
                self.assertEqual(sequences.shape, (20, 100))
                self.assertEqual(probabilities.shape, (20, 2 ** order, 2))
                self.assertTrue(np.all((sequences >= 0) & (sequences < 2)))
                self.assertTrue(np.all(probabilities > 0))
                np.testing.assert_allclose(probabilities.sum(axis=-1), 1)
                np.testing.assert_array_equal(orders, np.full(20, order))
                for table in probabilities:
                    self.assertFalse(np.array_equal(table[0], table[2 ** (order - 1)]))

    def test_seed_reproduces_output(self):
        first = generate_k_order_processes(3, num_symbols=3, sequence_length=100, seed=7)
        second = generate_k_order_processes(3, num_symbols=3, sequence_length=100, seed=7)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_initial_history_fills_output_when_length_equals_order(self):
        sequences, _, _ = generate_k_order_processes(
            3, num_processes=2000, sequence_length=3, num_symbols=3
        )
        self.assertEqual(sequences.shape, (2000, 3))
        histories = sequences @ np.array([9, 3, 1])
        frequencies = np.bincount(histories, minlength=27) / len(histories)
        np.testing.assert_allclose(frequencies, 1 / 27, atol=0.02, rtol=0)

    def test_samples_follow_history_rows(self):
        for order, num_symbols in ((3, 2), (2, 3)):
            sequences, tables, _ = generate_k_order_processes(
                order, num_processes=1, sequence_length=100_000,
                num_symbols=num_symbols, seed=7,
            )
            sequence = sequences[0]
            windows = np.lib.stride_tricks.sliding_window_view(sequence, order + 1)
            weights = num_symbols ** np.arange(order - 1, -1, -1)
            histories = windows[:, :order] @ weights
            checked = 0
            for history, row in enumerate(tables[0]):
                next_symbols = windows[histories == history, -1]
                if len(next_symbols) < 500:
                    continue
                frequencies = np.bincount(next_symbols, minlength=num_symbols) / len(next_symbols)
                np.testing.assert_allclose(frequencies, row, atol=0.06, rtol=0)
                checked += 1
            self.assertGreater(checked, 1)

    def test_invalid_parameters(self):
        for kwargs in (
            {"order": 1}, {"order": 2.5}, {"order": 3, "sequence_length": 2},
            {"order": 2, "num_processes": 0}, {"order": 2, "num_symbols": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_k_order_processes(**kwargs)


if __name__ == "__main__":
    unittest.main()
