import unittest

import numpy as np

from src.generate_data import (
    AIRPORT_PROBABILITIES,
    airport_transition_table,
    generate_airport_processes,
)


class TestAirportGeneration(unittest.TestCase):
    def test_transition_tables(self):
        for order in (2, 3, 4):
            table = airport_transition_table(order)
            self.assertEqual(table.shape, (2 ** order, 2))
            self.assertTrue(np.all((table >= 0) & (table <= 1)))
            np.testing.assert_allclose(table.sum(axis=1), 1)
            np.testing.assert_array_equal(table[:, 1], AIRPORT_PROBABILITIES[order])
            np.testing.assert_allclose(table[[0, -1], 1], {
                2: [0.0410, 0.8374],
                3: [0.0400, 0.8671],
                4: [0.0387, 0.8738],
            }[order])

    def test_sequences_and_reproducibility(self):
        first = generate_airport_processes(sequence_length=100, seed=7)
        second = generate_airport_processes(sequence_length=100, seed=7)
        sequences, orders, tables = first
        self.assertEqual(sequences.shape, (3, 100))
        self.assertTrue(np.all(np.isin(sequences, [0, 1])))
        np.testing.assert_array_equal(orders, [2, 3, 4])
        np.testing.assert_array_equal(sequences, second[0])
        for order in orders:
            np.testing.assert_array_equal(tables[order], second[2][order])
        self.assertEqual(generate_airport_processes(sequence_length=4)[0].shape, (3, 4))

    def test_observations_follow_tables(self):
        sequences, orders, tables = generate_airport_processes(sequence_length=100_000)
        for sequence, order in zip(sequences, orders):
            windows = np.lib.stride_tricks.sliding_window_view(sequence, order + 1)
            histories = windows[:, :order] @ (2 ** np.arange(order - 1, -1, -1))
            for history, row in enumerate(tables[order]):
                observations = windows[histories == history, -1]
                self.assertGreater(len(observations), 100)
                self.assertAlmostEqual(observations.mean(), row[1], delta=0.08)

    def test_invalid_arguments(self):
        for order in (0, 1, 5):
            with self.assertRaises(ValueError):
                airport_transition_table(order)
        for length in (0, 3, 4.5):
            with self.assertRaises(ValueError):
                generate_airport_processes(sequence_length=length)


if __name__ == '__main__':
    unittest.main()
