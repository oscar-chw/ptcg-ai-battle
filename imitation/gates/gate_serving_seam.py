#!/usr/bin/env python3
"""Gate the TRAIN/SERVE seam -- the one thing export parity structurally cannot see.

WHY THIS EXISTS. `gate_export_parity.py` compares the PyTorch argmax against the
NumPy agent's argmax over the SAME featurized input. That makes it blind to the
featurizer: a featurizer that is wrong is wrong identically on both sides, and
parity reads a perfect 1.0000.

Measured 2026-08-13. Every submissions/*/featurize.py on disk was the same stale
copy (md5 3198aa09, NUMERIC_WIDTH 64, zero UNSEEN_CARD) while tools/featurize.py
had moved to NUMERIC_WIDTH 80 with card_type, energy_type and per-card deck
tokens:

    submissions/S1-xerosic    weights num_proj (512, 80)  featurizer 64  -> 355.7
    submissions/P-cohort-929  weights num_proj (384, 64)  featurizer 64  -> 760.7

The 80/64 package shipped with export parity 1.0000, val top-1 0.6724 and Brier
0.3799 all green, then played on the ladder without attaching energy -- because
every card-semantic feature the model was trained to rely on arrived as zeros.

Three checks, each of which would have caught it in under a second:

  1. the shipped featurize.py is byte-identical to the training one
  2. weights' numeric projection width == the featurizer's NUMERIC_WIDTH
  3. main.py actually reads every semantic tensor the weights carry

    python gates/gate_serving_seam.py --submission build/<package>

Check 1 only means something on the machine that trained the weights. Off that
tree it differs by construction, and an unconditionally-failing gate is an
ignored one. So check 1 is skipped -- reported UNVERIFIABLE, exit 2, never a
silent pass -- when the reference featurizer's width disagrees with the width the
package's own weights were trained at, which proves it is not the trainer. Checks
2 and 3 are local to the package and stay binding on every tree.

Exit: 0 PASS, 1 FAIL, 2 INCOMPLETE (nothing failed, check 1 did not run).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_FEATURIZER = ROOT / "training" / "featurize.py"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _const(path: Path, name: str) -> int | None:
    """Read a module-level int constant without importing (heavy imports)."""
    m = re.search(rf"^{name}\s*=\s*(\d+)", path.read_text(), re.M)
    return int(m.group(1)) if m else None


def _numeric_width(path: Path) -> int | None:
    return _const(path, "NUMERIC_WIDTH")


def _weights_num_w(npz: Path) -> int | None:
    """The numeric width the WEIGHTS were trained at, from num_proj.weight.

    This is the only fact in the package that identifies the featurizer that
    actually produced it, so it is what decides whether a reference tree could
    have been the trainer.
    """
    if not npz.is_file():
        return None
    try:
        import numpy as np
        w = np.load(npz)
        keys = [k for k in w.files if "num_proj" in k and "weight" in k]
        if not keys or w[keys[0]].ndim != 2:
            return None
        return int(w[keys[0]].shape[1])
    except Exception:
        return None


# (featurizer constant, weight tensor, human name). The OPTION side matters as
# much as the state side: the L2 block extends option_numeric, so opt_num is
# exactly where the next width drift will appear.
WIDTH_PAIRS = [
    ("NUMERIC_WIDTH", "num_proj", "state numeric"),
    ("OPTION_NUMERIC_WIDTH", "opt_num", "option numeric"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--training-featurizer", type=Path, default=TRAINING_FEATURIZER)
    args = p.parse_args()

    sub = args.submission
    fails: list[str] = []
    unverifiable: list[str] = []
    print(f"submission: {sub}")

    # --- 1. featurizer identity -------------------------------------------
    # THE md5 COMPARISON IS ONLY MEANINGFUL ON THE TRAINING TREE. Run anywhere
    # else it differs by construction, so the gate failed unconditionally --
    # measured 2026-08-15, this file byte-identical on both machines, PASSing on
    # the box (269) and FAILing every package on the Mac (304, a tree that has
    # trained nothing anywhere). An always-failing gate is not a strict gate, it
    # is an ignored one, and it takes the width check below down with it.
    #
    # So the reference tree has to EARN the comparison. The package's own
    # num_proj.weight says what width its weights were trained at; if the
    # reference featurizer emits a different width it cannot have produced them,
    # and calling the difference DRIFT asserts something unestablished. Scoped by
    # measured width, never by path or hostname: same width means the tree is a
    # plausible trainer and the comparison stays a hard FAIL.
    served = sub / "featurize.py"
    trained = args.training_featurizer
    trained_w = _numeric_width(trained) if trained.is_file() else None
    weights_w = _weights_num_w(sub / "weights.npz")
    plausible_trainer = (trained_w is None or weights_w is None
                         or trained_w == weights_w)
    if not served.is_file():
        fails.append(f"{served} does not exist -- the package cannot featurize")
    elif not trained.is_file():
        fails.append(f"{trained} does not exist -- nothing to compare against")
    else:
        a, b = _md5(served), _md5(trained)
        same = a == b
        print(f"  featurize.py served={a[:8]} trained={b[:8]} "
              f"{'MATCH' if same else 'DRIFT'}")
        if not same and plausible_trainer:
            fails.append(
                f"served featurize.py ({a[:8]}) differs from the training "
                f"featurizer ({b[:8]}); the model will be fed features it was "
                f"never trained on")
        elif not same:
            unverifiable.append(
                f"featurizer identity UNVERIFIABLE from this tree: reference "
                f"{trained} emits {trained_w} numeric columns but these weights "
                f"were trained at {weights_w}, so it is not the featurizer that "
                f"produced them. Re-run on the training machine, or pass "
                f"--training-featurizer pointing at the tree that trained this "
                f"package. The width check below is unaffected and still binding.")

    # --- 1b. the data files the featurizer's WIDTH depends on --------------
    # Measured 2026-08-13: .gate/marnie-sixthsense/effect_tags.json was absent
    # on the training box. The old featurizer took EFFECT_TAG_NAMES from that
    # file, so its 21 effect-tag columns silently became 0.0 -- a 21-feature
    # dropout with no shape change, therefore no shape check to fire and no
    # gate to catch it. featurize.py now pins the tag list and refuses to
    # import without the file; this check makes the same requirement visible
    # for a SHIPPED package instead of at first inference.
    for data_file in ("effect_tags.json",):
        here = ROOT / ".gate/marnie-sixthsense" / data_file
        packaged = sub / data_file
        if not here.is_file() and not packaged.is_file():
            fails.append(
                f"{data_file} is missing from both {here.parent} and {sub}; "
                f"the effect-tag columns would be emitted as zeros")
        else:
            print(f"  {data_file:<20} present "
                  f"({'package' if packaged.is_file() else 'repo'})")

    # --- 2. numeric width agreement ---------------------------------------
    width = _numeric_width(served) if served.is_file() else None
    npz = sub / "weights.npz"
    if not npz.is_file():
        fails.append(f"{npz} does not exist")
    elif width is None:
        fails.append(f"could not read NUMERIC_WIDTH from {served}")
    else:
        import numpy as np
        w = np.load(npz)
        keys = [k for k in w.files if "num_proj" in k and "weight" in k]
        if not keys:
            fails.append("weights.npz has no num_proj weight to check width against")
        else:
            shape = w[keys[0]].shape
            expect = shape[1]
            print(f"  numeric width: featurizer={width} weights={expect} "
                  f"({keys[0]} {shape})")
            if expect != width:
                fails.append(
                    f"weights expect {expect} numeric features, featurizer emits "
                    f"{width}. This is the exact defect that scored 355.7.")

        # --- 3. every semantic tensor is actually consumed by main.py ------
        main_py = sub / "main.py"
        if not main_py.is_file():
            fails.append(f"{main_py} does not exist")
        else:
            src = main_py.read_text()
            for tensor in ("card_type", "energy_type", "card_bow", "word.weight"):
                present = any(tensor.split(".")[0] in k for k in w.files)
                used = tensor.split(".")[0] in src
                if present and not used:
                    fails.append(
                        f"weights carry '{tensor}' but main.py never reads it; "
                        f"that signal is silently dropped at serve time")
                if present:
                    print(f"  {tensor:<12} in weights, read by main.py: {used}")

    print()
    if unverifiable:
        # Printed whether or not anything failed: a check that did not run is a
        # fact about this run, and burying it under a FAIL list is how it stops
        # being read.
        print(f"SERVING SEAM UNVERIFIABLE ({len(unverifiable)})")
        for u in unverifiable:
            print(f"  ? {u}")
    if fails:
        print(f"SERVING SEAM FAIL ({len(fails)})")
        for f in fails:
            print(f"  - {f}")
        return 1
    if unverifiable:
        # Exit 2, not 0. An incomplete run must never be readable as a clean
        # pass -- that would hide the problem rather than name it.
        print("SERVING SEAM INCOMPLETE -- widths agree and every semantic tensor "
              "is consumed, but featurizer identity was NOT checked (see above)")
        return 2
    print("SERVING SEAM PASS -- served featurizer matches training, widths "
          "agree, every semantic tensor is consumed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
