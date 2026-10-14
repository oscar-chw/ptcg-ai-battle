"""Stdlib comparison and CLI validation, independent of a tensor framework."""

import importlib
import math
import subprocess
import sys
import unittest
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
SCRIPT = DEMO / "serving_diagnostic.py"


class Comparison(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "serving diagnostic is not implemented yet")
        self.api = importlib.import_module("serving_diagnostic")

    def test_matching_vectors_pass_and_have_no_mismatch(self):
        result = self.api.compare_rows([[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]])
        self.assertTrue(result["passed"])
        self.assertIsNone(result["first_mismatch"])
        self.assertEqual(result["choice_disagreements"], 0)

    def test_logit_drift_is_not_hidden_by_matching_choices(self):
        result = self.api.compare_rows([[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 4.0]])
        self.assertFalse(result["passed"])
        self.assertEqual(result["choice_disagreements"], 0)
        first = result["first_mismatch"]
        self.assertEqual(first["row"], 1)
        self.assertEqual(first["max_gap_option"], 1)
        self.assertEqual(first["max_abs_error"], 1.0)

    def test_first_mismatch_is_not_replaced_by_a_later_larger_error(self):
        result = self.api.compare_rows([[1.0, 2.0], [3.0, 4.0]], [[9.0, 2.0], [3.0, 99.0]])
        self.assertEqual(result["first_mismatch"]["row"], 0)
        self.assertEqual(result["first_mismatch"]["max_gap_option"], 0)
        self.assertEqual(result["max_abs_error"], 95.0)

    def test_choice_drift_fails_even_below_the_existing_score_threshold(self):
        result = self.api.compare_rows([[0.0, 1e-6]], [[1e-6, 0.0]])
        self.assertFalse(result["passed"])
        self.assertEqual(result["choice_disagreements"], 1)
        self.assertEqual(result["first_mismatch"]["reference_choice"], 1)
        self.assertEqual(result["first_mismatch"]["serving_choice"], 0)

    def test_threshold_boundary_fails_and_tied_choices_use_first_index(self):
        self.assertFalse(self.api.compare_rows([[0.0]], [[1e-4]])["passed"])
        tied = self.api.compare_rows([[1.0, 1.0]], [[1.0, 1.0]])
        self.assertEqual(tied["rows"][0]["reference_choice"], 0)

    def test_empty_missing_wrong_shape_and_nonfinite_vectors_fail(self):
        for left, right in [([], []), ([[]], [[]]), ([[1.0]], []), ([[1.0]], [[1.0, 2.0]]),
                            ([[math.nan]], [[0.0]]), ([[0.0]], [[math.inf]]),
                            ([[True]], [[0.0]]), ([["0"]], [[0.0]])]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                self.api.compare_rows(left, right)


class CliContract(unittest.TestCase):
    def test_bad_budget_mode_and_seed_fail_before_tensor_imports(self):
        for args, field in [(["--boards", "0"], "boards"), (["--boards", "65"], "boards"),
                            (["--seed", "-1"], "seed"), (["--serving-mode", "unknown"], "serving-mode"),
                            (["--reference-mode", "no-relations"], "reference-mode")]:
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(field, result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
