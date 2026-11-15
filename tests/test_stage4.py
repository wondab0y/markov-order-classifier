from fractions import Fraction
import random
import unittest

from sympy import Matrix

from src.minimization import EXAMPLE_4_2
from src.rank_one import rank_at_most_one
from src.stage4_pipeline import (
    _stationary, classify_hidden_chain, classify_higher_block, classify_symbol_matrices,
)


class TestExactClassifier(unittest.TestCase):
    def test_exact_rank(self):
        for rows, expected in [([[0, 0], [0, 0]], True),
                               ([[0, 2], [0, 4]], True),
                               ([[1, 2], [2, 4]], True),
                               ([[1, 0], [0, 1]], False),
                               ([[1, 1], [1, Fraction(10**30 + 1, 10**30)]], False)]:
            with self.subTest(rows=rows):
                self.assertEqual(rank_at_most_one(Matrix(rows)), expected)

    def test_iid_and_first_order(self):
        iid = classify_hidden_chain([["1/3", "2/3"]] * 2, [0, 1])
        self.assertEqual(iid, dict(order=0, minimal_dimension=1,
                                   search_bound=0, diagnostics=[]))
        result = classify_hidden_chain([["4/5", "1/5"], ["3/10", "7/10"]], [0, 1])
        self.assertEqual(result["order"], 1)
        self.assertEqual(result["diagnostics"], [{"order": 1, "surviving_products": 0}])
        self.assertEqual(classify_hidden_chain([[1]], [0])["order"], 0)

    def test_generated_exact_orders(self):
        for order in (1, 2, 3):
            for seed in (7, 42, 99):
                rng = random.Random(seed)
                # Same recent suffix, different oldest symbol: disjoint ranges
                # force distinct next-symbol probabilities. All rows positive.
                probabilities = [Fraction(rng.randint(1, 4) + 5 * (h // 2**(order - 1)), 10)
                                 for h in range(2**order)]
                table = [[p, 1 - p] for p in probabilities]
                with self.subTest(order=order, seed=seed):
                    result = classify_higher_block(table, order)
                    self.assertEqual(result["order"], order)
                    self.assertEqual(len(result["diagnostics"]), order)
                    self.assertTrue(all(d["surviving_products"] > 0
                                        for d in result["diagnostics"][:-1]))

    def test_higher_block_can_have_lower_order(self):
        self.assertEqual(classify_higher_block([["1/3", "2/3"]] * 8, 3)["order"], 0)
        table = [["4/5", "1/5"], ["3/10", "7/10"]] * 2
        self.assertEqual(classify_higher_block(table, 2)["order"], 1)

    def test_infinite_order_models(self):
        for p in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            even = {"0": [[p, 0], [0, 0]], "1": [[0, 1-p], [1, 0]]}
            sns = {"0": [[p, 1-p], [0, p]], "1": [[0, 0], [1-p, 0]]}
            for matrices in (even, sns):
                result = classify_symbol_matrices(matrices)
                self.assertIsNone(result["order"])
                self.assertEqual(result["minimal_dimension"], 2)
                self.assertEqual(result["search_bound"], 1)
                self.assertGreater(result["diagnostics"][-1]["surviving_products"], 0)

    def test_paper_example_minimizes_before_classifying(self):
        result = classify_symbol_matrices(EXAMPLE_4_2["matrices"], EXAMPLE_4_2["alpha"])
        self.assertEqual(result["minimal_dimension"], 2)
        self.assertEqual(result["order"], 1)

    def test_stationary_family_if_nonunique(self):
        transition = [[1, 0], [0, 1]]
        family = _stationary(Matrix(transition), None)
        parameter, = family.free_symbols
        self.assertEqual(family, Matrix([[1 - parameter, parameter]]))
        result = classify_hidden_chain(transition, [0, 1])
        self.assertEqual(result["stationary_family"], family)
        self.assertEqual(result["free_parameters"], [parameter])
        self.assertEqual(result["constraints"], [1 - parameter >= 0, parameter >= 0])
        chosen = list(family.subs(parameter, Fraction(1, 2)))
        self.assertEqual(classify_hidden_chain(transition, [0, 1], pi=chosen)["order"], 1)
        with self.assertRaises(ValueError):
            classify_hidden_chain(transition, [0, 1], pi=list(family.subs(parameter, 2)))
        self.assertEqual(classify_hidden_chain(transition, [0, 1], pi=[1, 0])["order"], 0)
        self.assertEqual(classify_hidden_chain(transition, [0, 1], pi=["1/2", "1/2"])["order"], 1)

    def test_invalid_models(self):
        for transition, mapping, pi in [
            ([["4/5", "1/5"], ["3/10", "7/10"]], [0, 1], [1, 0]),
            ([[1, 0], [0, 1]], [0, 1], [-1, 2]),
            ([["0.3", "0.6999999999999999"], [0, 1]], [0, 1], None),
            ([[1, -1], [0, 1]], [0, 1], None),
            ([[1]], [], None), ([[1]], [True], None),
            ([[1, 0]], [0], None),
        ]:
            with self.subTest(T=transition, mapping=mapping, pi=pi), self.assertRaises(ValueError):
                classify_hidden_chain(transition, mapping, pi)
        for kwargs in ({"order": 0}, {"order": True}, {"order": 2, "m": 1}):
            with self.assertRaises(ValueError):
                classify_higher_block([[1, 0]], **kwargs)


if __name__ == "__main__":
    unittest.main()
