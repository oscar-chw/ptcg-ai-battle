"""Actual CPU Torch/NumPy differential witnesses on explicitly synthetic features."""

import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
SCRIPT = DEMO / "serving_diagnostic.py"


class ServingDiagnostic(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "serving diagnostic is not implemented yet")
        self.api = importlib.import_module("serving_diagnostic")

    def test_actual_matching_forward_passes_and_choices_validate(self):
        report = self.api.run_diagnostic(boards=8, seed=0, reference_mode="masked", serving_mode="masked")
        self.assertEqual(report["result"], "PASS")
        self.assertTrue(report["control"]["passed"])
        self.assertTrue(report["candidate"]["passed"])
        self.assertIn("SYNTHETIC", report["scope"])
        self.assertFalse(report["historical_training_verified"])
        self.assertTrue(all(row["menu_choice_valid"] for row in report["candidate"]["rows"]))

    def test_actual_unmasked_forward_disagrees_with_masked_reference(self):
        report = self.api.run_diagnostic(boards=8, seed=0, reference_mode="masked", serving_mode="unmasked")
        self.assertEqual(report["result"], "MISMATCH")
        self.assertTrue(report["control"]["passed"])
        first = report["candidate"]["first_mismatch"]
        self.assertIsNotNone(first)
        self.assertEqual(len(first["reference_logits"]), len(first["serving_logits"]))
        self.assertGreater(first["max_abs_error"], 1e-4)

    def test_same_unmasked_configuration_passes_without_becoming_training_proof(self):
        report = self.api.run_diagnostic(boards=8, seed=0, reference_mode="unmasked", serving_mode="unmasked")
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["reference_mode"], "unmasked")
        self.assertFalse(report["historical_training_verified"])

    def test_same_seed_is_deterministic_and_changed_seed_changes_fixture_hashes(self):
        args = dict(boards=3, seed=0, reference_mode="masked", serving_mode="masked")
        first, second = self.api.run_diagnostic(**args), self.api.run_diagnostic(**args)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        args["seed"] = 1
        third = self.api.run_diagnostic(**args)
        self.assertNotEqual(first["input_sha256"], third["input_sha256"])
        self.assertNotEqual(first["weight_sha256"], third["weight_sha256"])

    def test_cpu_fixture_is_independent_of_caller_default_device_and_dtype(self):
        import torch
        args = dict(boards=3, seed=0, reference_mode="masked", serving_mode="masked")
        expected = self.api.run_diagnostic(**args)
        old_device, old_dtype = torch.get_default_device(), torch.get_default_dtype()
        try:
            torch.set_default_device("meta")  # No accelerator allocation; exposes implicit construction defaults.
            torch.set_default_dtype(torch.float64)
            actual = self.api.run_diagnostic(**args)
            self.assertEqual(torch.get_default_device().type, "meta")
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_device(old_device)
            torch.set_default_dtype(old_dtype)
        self.assertEqual(actual["result"], "PASS")
        self.assertEqual(actual["weight_sha256"], expected["weight_sha256"])
        self.assertEqual(actual["input_sha256"], expected["input_sha256"])

    def test_failed_positive_control_prevents_candidate_certification(self):
        def broken(weights, row, mode):
            return [100.0] * sum(kind >= 0 for kind in row["option_type"])

        with patch.object(self.api, "_numpy_logits", side_effect=broken):
            report = self.api.run_diagnostic(boards=2, seed=0, reference_mode="masked", serving_mode="unmasked")
        self.assertEqual(report["result"], "CONTROL_FAILED")
        self.assertFalse(report["control"]["passed"])
        self.assertIsNone(report["candidate"])

    def test_reference_globals_are_restored_when_inference_raises(self):
        import model_demo
        provider = model_demo.syn
        provider.set_training_semantics(mask_state=False, mask_options=True, relations=False)
        names = ["TRUNCATE_SEQ", "MASK_PAD_STATE", "MASK_PAD_OPTIONS", "BUILD_RELATIONS", "NUM_W_RUN", "OPT_NUM_W_RUN"]
        before = [getattr(provider.train_ss, name) for name in names]
        with patch.object(self.api, "_numpy_logits", side_effect=RuntimeError("inference failed")), \
                self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.api.run_diagnostic(boards=2, seed=0, reference_mode="masked", serving_mode="masked")
        self.assertEqual([getattr(provider.train_ss, name) for name in names], before)

    def test_cli_reports_computed_mismatch_and_recovery(self):
        wrong = subprocess.run([sys.executable, str(SCRIPT), "--boards", "4", "--serving-mode", "unmasked"],
                               capture_output=True, text=True)
        self.assertEqual(wrong.returncode, 1, wrong.stderr)
        report = json.loads(wrong.stdout)
        self.assertEqual(report["result"], "MISMATCH")
        recovered = subprocess.run([sys.executable, str(SCRIPT), "--boards", "4", "--serving-mode", "masked"],
                                   capture_output=True, text=True)
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        good = json.loads(recovered.stdout)
        self.assertEqual(good["input_sha256"], report["input_sha256"])
        self.assertEqual(good["weight_sha256"], report["weight_sha256"])


if __name__ == "__main__":
    unittest.main()
