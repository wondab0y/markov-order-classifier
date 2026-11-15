"""Small integration checks using Streamlit's built-in test runner."""

import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest
from sympy import Matrix, diag

from app import generate_dataset, load_sequence_file
from src.minimization import EXAMPLE_4_2
from src.statistical import estimate_markov_order


APP = Path(__file__).resolve().parents[1] / "app.py"


class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP)).run()
        self.assertFalse(self.app.exception)

    def load_exact(self, name):
        self.app.selectbox(key="exact_preset").select(name)
        next(button for button in self.app.button if button.label == "Load example").click().run()
        self.app.button(key="classify").click().run()
        self.assertFalse(self.app.exception)

    def test_first_order_and_minimization_preset(self):
        self.assertEqual([tab.label for tab in self.app.tabs], [
            "Exact Classification", "Process Generation", "Representation Minimization",
            "Statistical Classification"])
        self.load_exact("First-order chain")
        self.assertEqual(self.app.session_state["exact_result"]["order"], 1)
        next(button for button in self.app.button if button.label == "Load Example 4.2").click().run()
        self.app.button(key="minimize").click().run()
        self.assertEqual(self.app.session_state["min_result"]["minimal_dimension"], 2)
        self.assertFalse(self.app.exception)

    def test_stationary_parameter_selection_and_validation(self):
        self.load_exact("Multiple stationary distributions")
        self.assertIn("stationary_family", self.app.session_state["exact_result"])
        self.app.text_input(key="parameter:tau0").set_value("2")
        self.app.button(key="choose_parameters").click().run()
        self.assertTrue(self.app.error)
        self.assertIn("stationary_family", self.app.session_state["exact_result"])
        self.app.text_input(key="parameter:tau0").set_value("1/3")
        self.app.button(key="choose_parameters").click().run()
        self.assertEqual(self.app.session_state["exact_result"]["order"], 1)
        self.assertFalse(self.app.exception)
        self.load_exact("Multiple stationary distributions")
        self.app.text_input(key="parameter:tau0").set_value("0")
        self.app.button(key="choose_parameters").click().run()
        self.assertEqual(self.app.session_state["exact_result"]["order"], 0)

    def test_infinite_and_higher_block_modes(self):
        self.load_exact("Even Process")
        self.assertIsNone(self.app.session_state["exact_result"]["order"])
        self.app.selectbox(key="exact_mode").select("Higher-block process").run()
        self.app.button(key="classify").click().run()
        self.assertEqual(self.app.session_state["exact_result"]["order"], 0)
        self.assertFalse(self.app.exception)

    def test_general_nonminimal_representation(self):
        alpha = Matrix([EXAMPLE_4_2["alpha"]])
        matrices = {s: Matrix(rows) for s, rows in EXAMPLE_4_2["matrices"].items()}
        beta = Matrix(EXAMPLE_4_2["beta"])
        # Same process in a basis with negative entries, non-probability alpha,
        # and beta different from ones. No stochastic validation is applicable.
        basis = diag(2, -1, 1)
        self.app.selectbox(key="exact_mode").select("General Linear Representation")
        self.app.number_input(key="exact_n").set_value(3)
        for transform in (Matrix.eye(3), basis):
            inputs = {
                "general_alpha": alpha * transform,
                "general_beta": transform.inv() * beta,
                **{f"general_M{i}": transform.inv() * matrix * transform
                   for i, matrix in enumerate(matrices.values())},
            }
            for key, matrix in inputs.items():
                self.app.session_state[f"values:{key}"] = [
                    [str(x) for x in row] for row in matrix.tolist()]
                self.app.session_state[f"shape:{key}"] = None
            self.app.run()
            self.assertFalse(any(radio.label == "Stationary distribution π" for radio in self.app.radio))
            self.app.button(key="classify").click().run()
            self.assertFalse(self.app.exception)
            self.assertFalse(self.app.error)
            result = self.app.session_state["exact_result"]
            self.assertEqual(result["original_dimension"], 3)
            self.assertEqual(result["minimal_dimension"], 2)
            self.assertEqual(result["order"], 1)
            self.assertEqual(result["search_bound"], 1)
            self.assertEqual(result["diagnostics"], [{"order": 1, "surviving_products": 0}])
        revision = self.app.session_state["revision:general_alpha"]
        self.app.session_state[f"grid:general_alpha:{revision}"] = {
            "edited_rows": {0: {"0": "invalid"}}, "added_rows": [], "deleted_rows": []}
        self.app.button(key="classify").click().run()
        self.assertTrue(self.app.error)
        self.assertFalse(self.app.exception)

    def test_editor_preserves_fraction_strings_and_resizing(self):
        self.app.session_state["grid:exact_T:1"] = {
            "edited_rows": {0: {"0": "1/3", "1": "2/3"}}, "added_rows": [], "deleted_rows": []}
        self.app.run()
        expected = [["1/3", "2/3"], ["3/10", "7/10"]]
        self.assertEqual(self.app.session_state["values:exact_T"], expected)
        self.app.selectbox(key="exact_mode").select("Symbol matrices").run()
        self.app.selectbox(key="exact_mode").select("Hidden Markov chain").run()
        self.assertEqual(self.app.session_state["values:exact_T"], expected)
        self.app.number_input(key="exact_n").set_value(3).run()
        self.assertEqual(self.app.session_state["values:exact_T"][0], ["1/3", "2/3", "0"])
        self.app.number_input(key="gen_seed").set_value(7).run()
        self.app.number_input(key="exact_n").set_value(2).run()
        self.assertEqual(self.app.session_state["values:exact_T"], expected)
        self.app.selectbox(key="exact_mode").select("Symbol matrices").run()
        self.app.selectbox(key="exact_mode").select("Hidden Markov chain").run()
        self.assertEqual(self.app.session_state["values:exact_T"], expected)
        self.app.button(key="classify").click().run()
        self.assertEqual(self.app.session_state["exact_result"]["order"], 1)
        self.assertFalse(self.app.exception)

    def test_empty_entries_show_error_and_changed_inputs_hide_result(self):
        self.app.button(key="classify").click().run()
        self.app.session_state["grid:exact_T:1"] = {
            "edited_rows": {0: {"0": ""}}, "added_rows": [], "deleted_rows": []}
        self.app.run()
        self.assertFalse(self.app.metric)
        self.app.button(key="classify").click().run()
        self.assertTrue(self.app.error)
        self.assertFalse(self.app.exception)

    def test_generation_and_npz_download(self):
        self.app.number_input(key="gen_length").set_value(50).run()
        self.app.button(key="generate").click().run()
        self.assertFalse(self.app.exception)
        with np.load(io.BytesIO(self.app.session_state["generated_npz"]), allow_pickle=False) as data:
            self.assertEqual(data["sequences"].shape, (2, 50))
            self.assertIn("probabilities", data)
        self.app.radio(key="stat_source").set_value("Previous batch").run()
        self.assertEqual(len(self.app.selectbox(key="stat_sequence").options), 2)

    def test_bic_generated_alphabet_results_and_stale_inputs(self):
        data = generate_dataset("i.i.d.", count=2, length=100, m=3, order=0, seed=7)
        self.app.session_state["generated"] = data
        self.app.session_state["generated_family"] = "i.i.d."
        self.app.session_state["generated_npz"] = b""
        self.app.run().radio(key="stat_source").set_value("Previous batch").run()
        self.app.button(key="estimate_bic").click().run()
        self.assertFalse(self.app.exception)
        result = self.app.session_state["stat_result"]
        self.assertEqual(result["n_eff"], 94)
        self.assertEqual(result["scores"][0]["parameters"], 2)  # m=3, not binary.
        self.assertEqual(len(result["scores"]), 7)
        self.assertIn("BIC-selected finite-order estimate", [metric.label for metric in self.app.metric])
        chart = json.loads(self.app.get("vega_lite_chart")[0].proto.spec)
        self.assertEqual(chart["layer"][0]["mark"]["type"], "line")
        self.assertEqual(chart["layer"][1]["transform"][0]["filter"]["field"], "Selected minimum")
        frames = [frame.value for frame in self.app.dataframe]
        scores = next(frame for frame in frames if "BIC" in frame.columns)
        self.assertEqual(list(scores.columns), ["Candidate order", "Maximized log-likelihood", "Parameters", "BIC"])
        probabilities = next(frame for frame in frames if "P(2 | history)" in frame.columns)
        self.assertTrue(np.allclose(probabilities.iloc[:, 1:].sum(axis=1), 1))
        self.app.selectbox(key="stat_sequence").select(1).run()
        self.assertNotIn("BIC-selected finite-order estimate", [metric.label for metric in self.app.metric])

    def test_bic_upload_validation_sparsity_and_boundary(self):
        upload = io.BytesIO(b"0,1,2,0,1,2")
        upload.name = "sequence.csv"
        original_uploader = st.file_uploader

        def uploader(*args, **kwargs):
            return upload if kwargs.get("key") == "stat_upload" else original_uploader(*args, **kwargs)

        with patch("streamlit.file_uploader", side_effect=uploader):
            self.app.radio(key="stat_source").set_value("CSV upload").run()
            self.assertEqual(self.app.number_input(key="stat_m").value, 3)
            self.app.number_input(key="stat_max_order").set_value(2).run()
            self.app.button(key="estimate_bic").click().run()
            self.assertFalse(self.app.exception)
            self.assertTrue(any("sparse" in warning.value for warning in self.app.warning))
            self.app.number_input(key="stat_m").set_value(4).run()
            self.app.number_input(key="stat_max_order").set_value(0).run()
            self.app.button(key="estimate_bic").click().run()
            self.assertEqual(self.app.session_state["stat_result"]["scores"][0]["parameters"], 3)
            self.assertTrue(any("search boundary" in warning.value for warning in self.app.warning))
            self.app.number_input(key="stat_max_order").set_value(5).run()
            self.app.button(key="estimate_bic").click().run()
            self.assertTrue(self.app.error)
            self.assertFalse(self.app.exception)
            upload.seek(0)
            upload.truncate()
            upload.write(b"0,not-a-symbol")
            self.app.run()
            self.assertTrue(self.app.error)
            self.assertFalse(self.app.exception)
            upload.seek(0)
            upload.truncate()
            self.app.run()
            self.assertTrue(self.app.error)
            self.assertFalse(self.app.exception)

    def test_bic_direct_generators_end_to_end(self):
        self.assertEqual(self.app.radio(key="stat_source").options,
                         ["Generated sequence", "Manual sequence", "CSV upload"])
        for family in ("i.i.d.", "First-order Markov", "Higher-order Markov", "Even Process", "SNS"):
            with self.subTest(family=family):
                self.app.selectbox(key="stat_gen_family").select(family).run()
                self.app.number_input(key="stat_gen_length").set_value(200)
                self.app.number_input(key="stat_gen_seed").set_value(7)
                m = 2 if family in ("Even Process", "SNS") else 3
                if m == 3:
                    self.app.number_input(key="stat_gen_m").set_value(m)
                self.app.button(key="stat_generate").click().run()
                self.assertFalse(self.app.exception)
                self.assertFalse(self.app.error)
                sequence = self.app.session_state["stat_generated"]["sequences"][0]
                self.assertEqual(len(sequence), 200)
                self.app.number_input(key="stat_max_order").set_value(3).run()
                self.app.button(key="estimate_bic").click().run()
                self.assertFalse(self.app.exception)
                self.assertEqual(self.app.session_state["stat_result"],
                                 estimate_markov_order(sequence, max_order=3, m=m))
                self.assertEqual(len(self.app.get("vega_lite_chart")), 1)
        self.app.number_input(key="stat_gen_seed").set_value(8).run()
        self.assertNotIn("BIC-selected finite-order estimate", [metric.label for metric in self.app.metric])

    def test_bic_manual_sequence_end_to_end(self):
        self.app.radio(key="stat_source").set_value("Manual sequence").run()
        self.app.text_area(key="stat_manual").set_value("0, 1, 2\n0 1 2 0,1,2 0 1 2").run()
        self.assertEqual(self.app.number_input(key="stat_m").value, 3)
        self.app.number_input(key="stat_max_order").set_value(3).run()
        self.app.button(key="estimate_bic").click().run()
        self.assertFalse(self.app.exception)
        with self.assertWarns(RuntimeWarning):
            expected = estimate_markov_order([0, 1, 2] * 4, max_order=3, m=3)
        self.assertEqual(self.app.session_state["stat_result"], expected)
        self.assertTrue(any("sparse" in warning.value for warning in self.app.warning))
        self.assertEqual(len(self.app.get("vega_lite_chart")), 1)
        for invalid in ("0 1.5 2", "0 -1 2", "0 word 2", ""):
            self.app.text_area(key="stat_manual").set_value(invalid).run()
            self.assertFalse(self.app.exception)
            self.assertFalse(self.app.metric)
            if invalid:
                self.assertTrue(self.app.error)

    def test_bic_infinite_process_is_a_finite_estimate(self):
        self.app.session_state["generated"] = generate_dataset("Even Process", 1, 100, 2, 0, 7)
        self.app.session_state["generated_family"] = "Even Process"
        self.app.session_state["generated_npz"] = b""
        self.app.run().radio(key="stat_source").set_value("Previous batch").run()
        self.app.button(key="estimate_bic").click().run()
        self.assertFalse(self.app.exception)
        self.assertIn(self.app.session_state["stat_result"]["estimated_order"], range(7))
        self.assertNotIn("Exact Markov order", [metric.label for metric in self.app.metric])


class TestSequenceLoading(unittest.TestCase):
    def test_csv_rows_column_and_npz_metadata(self):
        for content, expected in [(b"0,1,2", [[0, 1, 2]]),
                                  (b"0\n1\n2", [[0, 1, 2]]),
                                  (b"0,1\n1,2", [[0, 1], [1, 2]])]:
            sequences, m = load_sequence_file(content, "sequences.csv")
            np.testing.assert_array_equal(sequences, expected)
            self.assertIsNone(m)
        for values in (np.array([0, 1, 2]), np.array([[0, 1, 2], [2, 1, 0]])):
            buffer = io.BytesIO()
            np.savez(buffer, sequences=values, num_symbols=4)
            sequences, m = load_sequence_file(buffer.getvalue(), "batch.npz")
            self.assertEqual(sequences.ndim, 2)
            self.assertEqual(m, 4)

    def test_reject_malformed_or_noninteger_files(self):
        for content in (b"", b"0.1,1.0", b"0,-1", b"0,1\n2"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_sequence_file(content, "bad.csv")
        for arrays in ({"other": [0, 1]}, {"sequences": [0.0, 1.0]},
                       {"sequences": [0, 1], "num_symbols": 1},
                       {"sequences": [0, 1], "num_symbols": 2.5},
                       {"sequences": np.ones((2, 2, 2), dtype=int)}):
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            with self.subTest(arrays=arrays.keys()), self.assertRaises(ValueError):
                load_sequence_file(buffer.getvalue(), "bad.npz")


class TestGenerationAdapters(unittest.TestCase):
    def test_existing_generator_formats_and_family_selection(self):
        for family, field in [("i.i.d.", "probabilities"), ("First-order Markov", "transitions"),
                              ("Higher-order Markov", "conditional_probabilities"),
                              ("Even Process", "symbol_matrices"), ("SNS", "symbol_matrices")]:
            with self.subTest(family=family):
                data = generate_dataset(family, count=2, length=50, m=2, order=2, seed=7)
                self.assertEqual(data["sequences"].shape, (2, 50))
                self.assertIn(field, data)
                if "families" in data:
                    self.assertTrue(np.all(data["families"] == ("even" if family == "Even Process" else "sns")))


if __name__ == "__main__":
    unittest.main()
