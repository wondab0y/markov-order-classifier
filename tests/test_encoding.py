import itertools
import unittest

import numpy as np

from src.encoding import encode_states, symbol_matrices, word_probability
from src.generate_data import generate_k_order_processes
from src.higher_block import higher_block_transition


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.T = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.4, 0.1, 0.5]])
        self.mapping = np.array([0, 1, 1])
        self.initial = np.array([0.6, 0.3, 0.1])

    def probability(self, word):
        return word_probability(word, self.T, self.mapping, self.initial, 2)

    def test_encoding_and_reproducibility(self):
        def sample(seed):
            rng = np.random.default_rng(seed)
            states = [rng.choice(3, p=self.initial)]
            for _ in range(20):
                states.append(rng.choice(3, p=self.T[states[-1]]))
            return encode_states(states, self.mapping)

        np.testing.assert_array_equal(sample(42), sample(42))
        np.testing.assert_array_equal(encode_states([[0, 1], [2, 0]], self.mapping),
                                      [[0, 1], [1, 0]])
        self.assertEqual(encode_states([], self.mapping).size, 0)

    def test_documented_symbol_matrices(self):
        M = symbol_matrices(self.T, self.mapping, 2)
        np.testing.assert_array_equal(M[0], [[0.5, 0, 0], [0.2, 0, 0], [0.4, 0, 0]])
        np.testing.assert_array_equal(M[1], [[0, 0.3, 0.2], [0, 0.6, 0.2], [0, 0.1, 0.5]])
        np.testing.assert_allclose(M.sum(axis=0), self.T)
        np.testing.assert_allclose(M.sum(axis=(0, 2)), 1)
        self.assertAlmostEqual(self.probability([0, 1]), 0.2)
        self.assertAlmostEqual(word_probability(
            [0, 1], self.T.tolist(), self.mapping.tolist(), self.initial.tolist(), 2
        ), 0.2)
        self.assertAlmostEqual(self.probability([]), 1)
        # A nonstationary start makes observation timing matter.
        # P(X_1=0)=0.4, whereas the omitted initial label has P(X_0=0)=0.6.
        self.assertAlmostEqual(self.probability([0]), 0.4)

    def test_word_probabilities_against_hidden_path_enumeration(self):
        for length in range(1, 4):
            total = 0
            for word in itertools.product(range(2), repeat=length):
                expected = 0
                for path in itertools.product(range(3), repeat=length + 1):
                    if tuple(self.mapping[list(path[1:])]) != word:
                        continue
                    p = self.initial[path[0]]
                    for i, j in zip(path, path[1:]):
                        p *= self.T[i, j]
                    expected += p
                actual = self.probability(word)
                self.assertAlmostEqual(actual, expected)
                total += actual
            self.assertAlmostEqual(total, 1)

    def test_stage_two_integration(self):
        _, tables, _ = generate_k_order_processes(2, num_processes=1, sequence_length=2)
        Q = tables[0]
        T = higher_block_transition(Q, 2, 2)
        mapping = [0, 1, 0, 1]
        for mu in ([0.25] * 4, [0, 1, 0, 0]):
            for length in range(1, 5):
                for word in itertools.product(range(2), repeat=length):
                    expected = 0
                    # First transition emits original X_2; sum out initial X_0,X_1.
                    for initial_history in range(4):
                        h = initial_history
                        p = mu[h]
                        for symbol in word:
                            p *= Q[h, symbol]
                            h = (h % 2) * 2 + symbol
                        expected += p
                    self.assertAlmostEqual(word_probability(word, T, mapping, mu, 2), expected)

    def test_merged_states_can_retain_memory(self):
        p_zero_after_01 = self.probability([0, 1, 0]) / self.probability([0, 1])
        p_zero_after_11 = self.probability([1, 1, 0]) / self.probability([1, 1])
        self.assertNotAlmostEqual(p_zero_after_01, p_zero_after_11)

    def test_invalid_inputs(self):
        for T in ([], [[1, 0]], [[-1, 2], [0, 1]], [[0.2]], [[np.nan]], [[np.inf]], [[1j]]):
            with self.subTest(T=T), self.assertRaises(ValueError):
                symbol_matrices(T, [0], 2)
        for mapping in ([0, 1], [0, -1, 1], [0, 2, 1], [0, 1.5, 1], [[0, 1, 1]]):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                symbol_matrices(self.T, mapping, 2)
        for m in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                symbol_matrices(self.T, self.mapping, m)
        for states in ([-1], [3], [0.5], [True]):
            with self.assertRaises(ValueError):
                encode_states(states, self.mapping)
        for word in ([-1], [2], [0.5], [[0]]):
            with self.assertRaises(ValueError):
                self.probability(word)
        for initial in ([1, 0], [-1, 1, 1], [0, 0, 0], [np.nan, 0, 1], [1j, 0, 1]):
            with self.assertRaises(ValueError):
                word_probability([0], self.T, self.mapping, initial, 2)


if __name__ == "__main__":
    unittest.main()
