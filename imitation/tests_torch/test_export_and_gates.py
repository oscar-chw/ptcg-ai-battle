"""Export stamp, provenance and the gates, end to end on a SYNTHETIC checkpoint.

Nothing here touches the engine or real weights: a tiny random model is saved,
exported, stamped, and then fed to the gates, which must pass a faithful package and
fail each corrupted one.
"""
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import synthetic as syn

ROOT = Path(__file__).resolve().parents[1]
RUN_CFG = {"d": 32, "layers": syn.LAYERS, "heads": syn.HEADS, "hidden": 48,
           "num_scale": "log1p", "dropout": 0.0,
           "no_mask_pad_state": False, "mask_pad_options": True, "relations": True}


def run(*args, cwd=None):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                          text=True, cwd=cwd)


class Fixture:
    """runs/<run>/{manifest.json,checkpoints/step-1.pt} plus an exported package."""

    def __init__(self, seed=0, cfg=None):
        self.dir = Path(tempfile.mkdtemp())
        self.run = self.dir / "runs" / "r1"
        (self.run / "checkpoints").mkdir(parents=True)
        self.model = syn.make_model(seed)
        self.ckpt = self.run / "checkpoints" / "step-1.pt"
        torch.save(self.model.state_dict(), self.ckpt)
        self.manifest = self.run / "manifest.json"
        self.manifest.write_text(json.dumps({"config": cfg or RUN_CFG}))
        self.pkg = self.dir / "pkg"
        self.pkg.mkdir()

    def export(self, manifest=None):
        args = [ROOT / "serving/export_ss_numpy.py", "--ckpt", self.ckpt,
                "--out", self.pkg / "weights.npz"]
        if manifest is not False:
            args += ["--run-manifest", manifest or self.manifest]
        return run(*args)


class Export(unittest.TestCase):
    def test_export_stamps_the_run_flags_and_keeps_every_tensor(self):
        fx = Fixture()
        self.assertEqual(fx.export().returncode, 0)
        z = np.load(fx.pkg / "weights.npz")
        stamps = {k: int(z[k]) for k in z.files if k.startswith("serve.")}
        self.assertEqual(stamps, {"serve.mask_pad_state": 1, "serve.mask_pad_options": 1,
                                  "serve.relations": 1})
        weights = [k for k in z.files if not k.startswith("serve.")]
        self.assertEqual(sorted(weights), sorted(fx.model.state_dict()))
        self.assertEqual(z["card.weight"].dtype, np.float16)      # big matrix: fp16
        self.assertEqual(z["emb_norm.weight"].dtype, np.float32)  # norm: fp32

    def test_export_refuses_a_manifest_that_omits_a_serving_flag(self):
        fx = Fixture(cfg={k: v for k, v in RUN_CFG.items() if k != "relations"})
        r = fx.export()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("relations", r.stderr)


class ServeStampGate(unittest.TestCase):
    def build(self, cfg_for_gate=None):
        fx = Fixture()
        self.assertEqual(fx.export().returncode, 0)
        r = run(ROOT / "serving/write_provenance.py", "--package", fx.pkg,
                "--ckpt", fx.ckpt, "--force")
        self.assertEqual(r.returncode, 0, r.stderr)
        if cfg_for_gate is not None:                   # the run "really" trained this
            fx.manifest.write_text(json.dumps({"config": cfg_for_gate}))
        return fx

    def test_a_stamp_equal_to_its_runs_flags_passes(self):
        fx = self.build()
        r = run(ROOT / "gates/gate_serve_stamp.py", "--package", fx.pkg)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_stamp_copied_from_a_sibling_run_fails(self):
        # weights stamped relations=1, but the run that produced them trained without
        fx = self.build(cfg_for_gate=dict(RUN_CFG, relations=False))
        r = run(ROOT / "gates/gate_serve_stamp.py", "--package", fx.pkg)
        self.assertEqual(r.returncode, 1)
        self.assertIn("serve.relations=1", r.stdout)

    def test_a_package_without_provenance_fails(self):
        fx = Fixture()
        fx.export()
        r = run(ROOT / "gates/gate_serve_stamp.py", "--package", fx.pkg)
        self.assertEqual(r.returncode, 1)
        self.assertIn("PROVENANCE", r.stdout)

    def test_the_gates_own_negative_controls_are_green(self):
        r = run(ROOT / "gates/gate_serve_stamp.py", "--self-test")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("GREEN", r.stdout)


class ServingSeamGate(unittest.TestCase):
    STUB = "NUMERIC_WIDTH = {w}\nOPTION_NUMERIC_WIDTH = {w}\n"

    def package(self, served_width=syn.NUM_W, trained_width=syn.NUM_W):
        fx = Fixture()
        fx.export()
        (fx.pkg / "main.py").write_text("# reads card_type, energy_type\n")
        (fx.pkg / "effect_tags.json").write_text("{}")
        (fx.pkg / "featurize.py").write_text(self.STUB.format(w=served_width))
        trained = fx.dir / "featurize_training.py"
        trained.write_text(self.STUB.format(w=trained_width))
        return fx, trained

    def seam(self, fx, trained):
        return run(ROOT / "gates/gate_serving_seam.py", "--submission", fx.pkg,
                   "--training-featurizer", trained)

    def test_matching_featurizer_and_widths_pass(self):
        fx, trained = self.package()
        self.assertEqual(self.seam(fx, trained).returncode, 0)

    def test_a_featurizer_wider_than_the_weights_fails(self):
        # the defect that scored 355.7 behind green offline gates
        fx, trained = self.package(served_width=80, trained_width=80)
        r = self.seam(fx, trained)
        self.assertEqual(r.returncode, 1)
        self.assertIn("numeric features", r.stdout)

    def test_a_drifted_featurizer_fails_when_the_reference_could_be_the_trainer(self):
        fx, trained = self.package()
        (fx.pkg / "featurize.py").write_text(self.STUB.format(w=syn.NUM_W) + "# edited\n")
        r = self.seam(fx, trained)
        self.assertEqual(r.returncode, 1)
        self.assertIn("differs from the training", r.stdout)

    def test_a_reference_that_cannot_be_the_trainer_is_incomplete_not_a_pass(self):
        fx, trained = self.package()
        trained.write_text(self.STUB.format(w=269))
        (fx.pkg / "featurize.py").write_text(self.STUB.format(w=syn.NUM_W))
        self.assertEqual(self.seam(fx, trained).returncode, 2)

    def test_a_missing_effect_tags_file_fails(self):
        fx, trained = self.package()
        (fx.pkg / "effect_tags.json").unlink()
        r = self.seam(fx, trained)
        self.assertEqual(r.returncode, 1)
        self.assertIn("effect_tags.json", r.stdout)


class ReachabilityGate(unittest.TestCase):
    def test_the_gates_own_negative_controls_are_green(self):
        r = run(ROOT / "gates/gate_package_reachability.py", "--self-test")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("GREEN", r.stdout)


class ComputeParityGate(unittest.TestCase):
    def corpus(self, fx, rows):
        data = fx.dir / "corpus"
        data.mkdir(exist_ok=True)
        with gzip.open(data / "frames.jsonl.gz", "wt") as fh:
            for row in rows:
                fh.write(json.dumps(dict(row, split="validation")) + "\n")
        return data

    def gate(self, fx, npz, rows):
        out = fx.dir / "parity.json"
        r = run(ROOT / "gates/gate_compute_parity.py", "--data", self.corpus(fx, rows),
                "--ckpt", fx.ckpt, "--manifest", fx.manifest, "--npz", npz, "--out", out,
                "--rows", 40, "--min-rows", 20, "--batch", 8)
        return r, (json.loads(out.read_text()) if out.exists() else None)

    def test_a_faithful_export_passes_and_a_foreign_one_fails(self):
        fx = Fixture(seed=0)
        self.assertEqual(fx.export().returncode, 0)
        rows = syn.make_rows(60)
        ok, report = self.gate(fx, fx.pkg / "weights.npz", rows)
        self.assertEqual(ok.returncode, 0, ok.stdout[-800:] + ok.stderr[-800:])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["agreement_v7_fp32_strict"], 1.0)
        self.assertIsNone(report["agreement_v6_decidable"])   # --main-v6 not given

        # same shapes, different weights: the gate must go red
        other = Fixture(seed=7)
        self.assertEqual(other.export().returncode, 0)
        bad, report = self.gate(fx, other.pkg / "weights.npz", rows)
        self.assertEqual(bad.returncode, 1)
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
