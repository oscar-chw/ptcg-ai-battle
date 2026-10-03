"""The serving flags are stamped from the run's own manifest, and absence must raise.

Two readers decide what a stamp means: the exporter (stamps) and the gate
(verifies). They must agree on every manifest, including the ones that omit a key.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "gates"))
from export_ss_numpy import serve_flags_from_manifest  # noqa: E402
from gate_serve_stamp import training_flags  # noqa: E402

FULL = {"no_mask_pad_state": False, "mask_pad_options": True, "relations": True}


def manifest(cfg):
    d = Path(tempfile.mkdtemp())
    path = d / "manifest.json"
    path.write_text(json.dumps({"config": cfg}))
    return path


class ServeFlags(unittest.TestCase):
    def test_full_manifest_is_read_faithfully(self):
        self.assertEqual(
            serve_flags_from_manifest(manifest(FULL)),
            {"mask_pad_state": True, "mask_pad_options": True, "relations": True})

    def test_no_mask_pad_state_true_means_state_masking_off(self):
        cfg = dict(FULL, no_mask_pad_state=True)
        self.assertFalse(serve_flags_from_manifest(manifest(cfg))["mask_pad_state"])

    def test_absent_key_raises_unless_legacy_is_named(self):
        cfg = {"mask_pad_options": True, "relations": True}  # no_mask_pad_state absent
        with self.assertRaises(ValueError) as ctx:
            serve_flags_from_manifest(manifest(cfg))
        self.assertIn("no_mask_pad_state", str(ctx.exception))
        legacy = serve_flags_from_manifest(manifest(cfg), assume_legacy=True)
        self.assertFalse(legacy["mask_pad_state"])

    def test_gate_reads_an_absent_key_as_off_not_as_todays_default(self):
        # The gate must NOT reconstruct `not cfg.get("no_mask_pad_state", False)`,
        # which would be True and describe a run that masked when it did not.
        flags = training_flags(manifest({"mask_pad_options": False, "relations": False}))
        self.assertFalse(flags["mask_pad_state"])

    def test_exporter_and_gate_agree_when_every_key_is_present(self):
        path = manifest(FULL)
        self.assertEqual(serve_flags_from_manifest(path), training_flags(path))


if __name__ == "__main__":
    unittest.main()
