import itertools
import unittest

import numpy as np

from src.higher_block import higher_block_transition


class TestHigherBlock(unittest.TestCase):
    def setUp(self):
        self.Q = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
        self.T = higher_block_transition(self.Q, 2, 2)

    def test_documented_example_and_stationarity(self):
        expected = [[0.8, 0.2, 0, 0], [0, 0, 0.3, 0.7],
                    [0.6, 0.4, 0, 0], [0, 0, 0.1, 0.9]]
        np.testing.assert_array_equal(self.T, expected)
        rho = np.array([1 / 4, 1 / 12, 1 / 12, 7 / 12])
        np.testing.assert_allclose(rho @ self.T, rho)

    def test_overlap_normalization_and_round_trip(self):
        rng = np.random.default_rng(42)
        for k, m in [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (2, 3)]:
            with self.subTest(order=k, num_symbols=m):
                Q = rng.dirichlet(np.ones(m), size=m ** k)
                T = higher_block_transition(Q, k, m)
                self.assertEqual(T.shape, (m ** k, m ** k))
                self.assertTrue(np.all(T >= 0))
                np.testing.assert_allclose(T.sum(axis=1), 1)
                recovered = np.empty_like(Q)
                for h in range(m ** k):
                    history = np.unravel_index(h, (m,) * k)
                    for j in range(m ** k):
                        successor = np.unravel_index(j, (m,) * k)
                        if history[1:] == successor[:-1]:
                            recovered[h, successor[-1]] = T[h, j]
                        else:
                            self.assertEqual(T[h, j], 0)
                np.testing.assert_array_equal(recovered, Q)

    def test_word_probabilities_with_matching_initialization(self):
        # Include uniform, stationary, and fixed-history starts.
        for mu in ([0.25] * 4, [1 / 4, 1 / 12, 1 / 12, 7 / 12], [0, 1, 0, 0]):
            for length in range(2, 7):
                for word in itertools.product(range(2), repeat=length):
                    h = 2 * word[0] + word[1]
                    conditional_probability = mu[h]
                    for symbol in word[2:]:
                        conditional_probability *= self.Q[h, symbol]
                        h = (h % 2) * 2 + symbol
                    # Index each overlapping block directly from the word.
                    blocks = [2 * a + b for a, b in zip(word, word[1:])]
                    block_probability = mu[blocks[0]]
                    for current, following in zip(blocks, blocks[1:]):
                        block_probability *= self.T[current, following]
                    self.assertAlmostEqual(conditional_probability, block_probability)

    def test_zero_probability_transitions(self):
        Q = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        T = higher_block_transition(Q, 2, 2)
        np.testing.assert_array_equal(T, [[1, 0, 0, 0], [0, 0, 0, 1],
                                        [1, 0, 0, 0], [0, 0, 0, 1]])

    def test_invalid_inputs(self):
        for order in (0, -1, 2.5, True, np.bool_(True)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                higher_block_transition(self.Q, order, 2)
        for m in (0, 1, 2.5, True):
            with self.subTest(num_symbols=m), self.assertRaises(ValueError):
                higher_block_transition(self.Q, 2, m)
        for Q in (np.ones((2, 2)), np.full((4, 2), np.nan),
                  np.full((4, 2), np.inf), np.full((4, 2), 0.4),
                  [[-0.1, 1.1]] * 4, np.full((4, 2), 0.5 + 0.1j)):
            with self.subTest(Q=Q), self.assertRaises(ValueError):
                higher_block_transition(Q, 2, 2)


if __name__ == "__main__":
    unittest.main()
