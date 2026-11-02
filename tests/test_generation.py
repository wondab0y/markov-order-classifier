import unittest

import numpy as np

from src.generate_data import generate_first_order_processes, generate_zero_order_processes


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


if __name__ == "__main__":
    unittest.main()
