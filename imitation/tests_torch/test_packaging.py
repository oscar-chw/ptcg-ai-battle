"""package_arms.sh / package_and_tar.sh, run end to end on a SYNTHETIC checkpoint.

The packager is copied into a scratch tree first, so the test never writes into the
repository. The "engine" is an empty cg/ package: the packager only copies it.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

import synthetic  # noqa: F401  (puts training/ and gates/ on sys.path; keep it first)
import model_ss

ROOT = Path(__file__).resolve().parents[1]
CFG = {"d": 32, "layers": 2, "heads": 4, "hidden": 48, "num_scale": "log1p",
       "no_mask_pad_state": False, "mask_pad_options": True, "relations": True}


class Packaging(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tree = self.tmp / "imitation"
        shutil.copytree(ROOT, self.tree, ignore=shutil.ignore_patterns(
            "tests", "tests_torch", "__pycache__", "build", "runs", ".gate"))
        tags = self.tree / ".gate/marnie-sixthsense"
        tags.mkdir(parents=True)
        (tags / "effect_tags.json").write_text("{}")
        self.run_dir = self.tree / "runs" / "r1"
        (self.run_dir / "checkpoints").mkdir(parents=True)
        torch.manual_seed(0)
        model = model_ss.SixthSenseNet(d=32, layers=2, heads=4, hidden=48,
                                       dropout=0.0, num_scale="log1p")
        self.ckpt = self.run_dir / "checkpoints" / "step-1.pt"
        torch.save(model.state_dict(), self.ckpt)
        self.manifest = self.run_dir / "manifest.json"
        self.manifest.write_text(json.dumps({"config": CFG}))
        self.engine = self.tmp / "engine"
        (self.engine / "cg").mkdir(parents=True)
        (self.engine / "cg/__init__.py").write_text("")
        self.deck = self.tmp / "deck.csv"
        self.deck.write_text("\n".join(str(1 + i % 7) for i in range(60)) + "\n")  # SYNTHETIC
        self.env = dict(os.environ, PYTHON=sys.executable, PTCG_ENGINE_DIR=str(self.engine),
                        PTCG_BUILD_DIR=str(self.tmp / "build"))

    def sh(self, script, *args, env=None):
        return subprocess.run(["bash", str(self.tree / "serving" / script), *map(str, args)],
                              capture_output=True, text=True, env=env or self.env)

    def test_the_wrapper_builds_a_stamped_package_local_tarball(self):
        r = self.sh("package_arms.sh", "r1", self.deck, self.ckpt)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
        self.assertIn("SERVING SEAM PASS", r.stdout)
        with tarfile.open(self.tmp / "build" / "r1.tar.gz") as tar:
            names = {m.name for m in tar.getmembers()}
            main_bytes = tar.extractfile("./main.py").read()
            link = tar.getmember("./featurize.py")
        for want in ("./lib/featurize.py", "./weights.npz", "./PROVENANCE.json", "./deck.csv",
                     "./cg/__init__.py", "./.gate/marnie-sixthsense/effect_tags.json"):
            self.assertIn(want, names)
        self.assertTrue(link.issym())          # featurize.py is a link into lib/
        shipped = hashlib.sha256((ROOT / "serving" / "main_v7.py").read_bytes()).hexdigest()
        self.assertEqual(hashlib.sha256(main_bytes).hexdigest(), shipped)

    def build_directly(self, *layout_flags):
        r = self.sh("package_and_tar.sh", "r1", "r1", self.deck, self.ckpt,
                    "--main", self.tree / "serving" / "main_v7.py",
                    "--run-manifest", self.manifest, *layout_flags)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
        with tarfile.open(self.tmp / "build" / "r1.tar.gz") as tar:
            return {m.name: m for m in tar.getmembers()}

    def test_the_package_local_layout_is_the_default(self):
        members = self.build_directly()          # no layout flag at all
        self.assertIn("./lib/featurize.py", members)
        self.assertTrue(members["./featurize.py"].issym())

    def test_the_flat_layout_has_to_be_asked_for_by_name(self):
        members = self.build_directly("--legacy-tags")
        self.assertNotIn("./lib/featurize.py", members)
        self.assertTrue(members["./featurize.py"].isfile())

    def test_a_stamp_is_required_unless_waived_by_name(self):
        r = self.sh("package_and_tar.sh", "r1", "r1", self.deck, self.ckpt,
                    "--main", self.tree / "serving" / "main_v7.py")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--no-stamp", r.stderr)

    def test_the_serving_file_has_no_default(self):
        r = self.sh("package_and_tar.sh", "r1", "r1", self.deck, self.ckpt,
                    "--run-manifest", self.manifest)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--main is required", r.stderr)

    def test_a_missing_run_manifest_refuses_to_build(self):
        self.manifest.unlink()
        r = self.sh("package_arms.sh", "r1", self.deck, self.ckpt)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no run manifest", r.stderr)

    def test_without_an_engine_directory_it_stops_with_a_message(self):
        env = {k: v for k, v in self.env.items() if k != "PTCG_ENGINE_DIR"}
        r = self.sh("package_arms.sh", "r1", self.deck, self.ckpt, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("PTCG_ENGINE_DIR", r.stderr)


if __name__ == "__main__":
    unittest.main()
