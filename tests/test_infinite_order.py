from fractions import Fraction
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.generate_data import generate_infinite_order_processes, infinite_order_model


class TestInfiniteOrderGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_infinite_order_processes()

    def test_dataset_and_models(self):
        data = self.data
        self.assertEqual(data['sequences'].shape, (40, 10_000))
        self.assertEqual(data['symbol_matrices'].shape, (40, 2, 2, 2))
        self.assertTrue(np.all(np.isin(data['sequences'], [0, 1])))
        self.assertTrue(np.all(np.isposinf(data['orders'])))
        self.assertEqual(np.count_nonzero(data['families'] == 'even'), 20)
        self.assertEqual(np.count_nonzero(data['families'] == 'sns'), 20)
        self.assertTrue(np.all((data['p'] >= 0.2) & (data['p'] < 0.8)))
        for matrices, pi in zip(data['symbol_matrices'], data['stationary_distributions']):
            transition = matrices.sum(axis=0)
            self.assertTrue(np.all(matrices >= 0))
            self.assertTrue(np.all(pi > 0))
            np.testing.assert_allclose(transition.sum(axis=1), 1)
            np.testing.assert_allclose(pi.sum(), 1)
            np.testing.assert_allclose(pi @ transition, pi)

    def test_family_constraints_in_every_sequence(self):
        for sequence, family in zip(self.data['sequences'], self.data['families']):
            if family == 'even':
                # Exclude boundary runs: stationary samples may start mid-run.
                zeros = np.flatnonzero(sequence == 0)
                self.assertGreater(len(zeros), 100)
                self.assertTrue(np.all((np.diff(zeros) - 1) % 2 == 0))
            else:
                ones = np.flatnonzero(sequence == 1)
                self.assertGreater(len(ones), 100)
                self.assertTrue(np.all(np.diff(ones) >= 2))  # SNS forbids 11.

    def test_observed_word_probabilities(self):
        # Validate the complete observation law, including joint edge sampling
        # and stationary initialization, without using hidden states.
        for i in (0, 19, 20, 39):
            sequence = self.data['sequences'][i]
            matrices = self.data['symbol_matrices'][i]
            pi = self.data['stationary_distributions'][i]
            windows = np.lib.stride_tricks.sliding_window_view(sequence, 3)
            for word in product((0, 1), repeat=3):
                weights = pi.copy()
                for symbol in word:
                    weights = weights @ matrices[symbol]
                expected = weights.sum()
                observed = np.mean(np.all(windows == word, axis=1))
                self.assertAlmostEqual(observed, expected, delta=0.025)

    def test_sns_gap_distribution(self):
        # Independently derived two-geometric waiting-time distribution.
        for i in (20, 25, 30, 39):
            p = self.data['p'][i]
            ones = np.flatnonzero(self.data['sequences'][i] == 1)
            gaps = np.diff(ones) - 1
            for n in range(1, 7):
                expected = n * (1 - p) ** 2 * p ** (n - 1)
                self.assertAlmostEqual(np.mean(gaps == n), expected, delta=0.04)

    def test_same_suffix_has_different_predictions(self):
        # Exact rational arithmetic avoids rounding away SNS's memory.
        # These are regression witnesses; the proof for all k is in the docs.
        def next_probability(matrices, history, symbol):
            weights = np.array([Fraction(1, 2), Fraction(1, 2)])
            for observation in history:
                weights = weights @ matrices[observation]
                self.assertGreater(sum(weights), 0)
                weights = weights / sum(weights)
            return sum(weights @ matrices[symbol])

        for p in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            for family in ('even', 'sns'):
                matrices, _ = infinite_order_model(family, float(p))
                matrices = np.vectorize(Fraction, otypes=[object])(matrices)
                reset, repeated = (0, 1) if family == 'even' else (1, 0)
                for k in (1, 2, 3, 5, 10, 32, 64):
                    first = next_probability(matrices, [reset] + [repeated] * k, reset)
                    second = next_probability(matrices, [reset] + [repeated] * (k + 1), reset)
                    self.assertNotEqual(first, second)
                    if family == 'even':
                        self.assertEqual(first, p if k % 2 == 0 else 0)
                    else:
                        self.assertEqual(first, k * (1 - p) ** 2 / (p + k * (1 - p)))

    def test_reproducibility_and_npz_round_trip(self):
        first = generate_infinite_order_processes(num_processes=4, sequence_length=50, seed=7)
        second = generate_infinite_order_processes(num_processes=4, sequence_length=50, seed=7)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        different = generate_infinite_order_processes(num_processes=4, sequence_length=50, seed=8)
        self.assertFalse(np.array_equal(first['sequences'], different['sequences']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'infinite.npz'
            np.savez(path, **first)
            with np.load(path, allow_pickle=False) as saved:
                for key in first:
                    np.testing.assert_array_equal(saved[key], first[key])

    def test_invalid_arguments_and_degenerate_models(self):
        for family in ('even', 'sns'):
            for p in (0, 1, -0.1, 1.1, np.nan, np.inf):
                with self.subTest(family=family, p=p), self.assertRaises(ValueError):
                    infinite_order_model(family, p)
        with self.assertRaises(ValueError):
            infinite_order_model('unknown', 0.5)
        for kwargs in (
            {'num_processes': 1}, {'num_processes': 2.5},
            {'sequence_length': 0}, {'sequence_length': 1.5},
            {'seed': -1}, {'seed': None},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_infinite_order_processes(**kwargs)


if __name__ == '__main__':
    unittest.main()
