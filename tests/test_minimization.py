import itertools
import unittest

from src.minimization import (
    EXAMPLE_4_2,
    minimize_representation,
    parse_representation,
    word_probability,
)


class TestMinimization(unittest.TestCase):
    def assert_equivalent(self, model, expected_dimension):
        result = minimize_representation(**model)
        self.assertEqual(result["minimal_dimension"], expected_dimension)
        a, ms, b = parse_representation(**model)
        for length in range(6):
            for word in itertools.product(ms, repeat=length):
                self.assertEqual(
                    word_probability(a, ms, b, word),
                    word_probability(result["alpha"], result["matrices"], result["beta"], word),
                )
        return result

    def test_paper_example(self):
        result = self.assert_equivalent(EXAMPLE_4_2, 2)
        self.assertEqual(result["original_dimension"], 3)

    def test_unreachable_and_unobservable_states(self):
        result = self.assert_equivalent({
            "alpha": [1, 0, 0], "beta": [1, 0, 0],
            "matrices": {"a": [["1/2", 1, 0], [0, 1, 0], [0, 0, 1]]},
        }, 1)
        self.assertEqual(result["reachable_dimension"], 2)
        self.assertEqual(result["observable_dimension"], 1)

    def test_minimal_representation(self):
        self.assert_equivalent({"alpha": ["1/2", "1/2"], "beta": [1, 1],
                                "matrices": {"a": [[1, 0], [0, "1/2"]]}}, 2)

    def test_zero_representation(self):
        for a, b in (([0, 0], [1, 1]), ([1, 1], [0, 0]), ([1, 0], [0, 1])):
            self.assert_equivalent({"alpha": a, "beta": b,
                                    "matrices": {"a": [[1, 0], [0, 1]]}}, 0)

    def test_invalid_inputs(self):
        for changes in ({"alpha": []}, {"beta": [1]}, {"matrices": {}},
                        {"matrices": {"a": [[1]]}}, {"alpha": ["nan", 1, 1]},
                        {"alpha": ["1/0", 1, 1]}, {"alpha": [True, 1, 1]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                minimize_representation(**(EXAMPLE_4_2 | changes))


if __name__ == "__main__":
    unittest.main()
