#!/usr/bin/env python3
"""Write PROVENANCE.json into a submission package.

WHY THIS EXISTS. gate_serve_stamp.py checks that a package's serve.* stamps
equal the training flags of the run that produced its weights. To do that it has
to know WHICH run produced them -- and until this file is written, nothing in
the package says. The stamp on A2-v7-stamped was correct, and the gate still
failed it, because the only way to verify it was to supply --run from outside
knowledge. A stamp that only its author can check is a stamp on trust.

The numbers below are COMPUTED FROM THIS PACKAGE'S OWN weights.npz, never copied
from a sibling package. A provenance file carrying another model's params is a
stamp about a different model -- the substitution antipattern wearing the
costume of the fix. FIXED-312 has params 45435348; A2-v7-stamped has 45547400.
Those differing is the evidence that neither was copied.

Schema mirrors ptcg-ai/ss-weights/v1, already emitted by export_ss_numpy.py, so
there is one provenance schema on this tree rather than two.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint path RELATIVE TO THE REPO ROOT; "
                         "gate_serve_stamp resolves the run as parents[1] of it")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing PROVENANCE.json")
    a = ap.parse_args()

    npz_path = a.package / "weights.npz"
    if not npz_path.is_file():
        print(f"FATAL: no weights.npz in {a.package}", file=sys.stderr)
        return 1

    out = a.package / "PROVENANCE.json"
    if out.is_file() and not a.force:
        print(f"FATAL: {out} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    # THE RUN MUST BE RESOLVABLE THE WAY THE GATE RESOLVES IT, OR THE FILE IS
    # DECORATION. gate_serve_stamp does Path(ckpt).parent.parent and then looks
    # for manifest.json there. Check that here rather than discovering it at
    # gate time -- a provenance file naming a run with no manifest would pass
    # "the package has provenance" and fail the thing provenance is FOR.
    run_dir = Path(a.ckpt).parent.parent
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        print(f"FATAL: --ckpt {a.ckpt} resolves the run to {run_dir}, which has "
              f"no manifest.json. The gate would be unable to read the training "
              f"flags, so this provenance would name a run it cannot check.",
              file=sys.stderr)
        return 1

    z = np.load(npz_path)
    # serve.* are STAMPS, not model weights. Counting them as tensors would make
    # `tensors` disagree with export_ss_numpy's own count for the same model.
    names = sorted(k for k in z.files if not k.startswith("serve."))
    if not names:
        print(f"FATAL: {npz_path} carries no model tensors, only stamps",
              file=sys.stderr)
        return 1

    nbytes = os.path.getsize(npz_path)
    prov = {
        "schema": "ptcg-ai/ss-weights/v1",
        "checkpoint": str(a.ckpt),
        "params": int(sum(int(np.prod(z[k].shape)) for k in names)),
        "tensors": len(names),
        "bytes": nbytes,
        "mb": round(nbytes / 1e6, 1),
        "keys": names[:8],
    }
    out.write_text(json.dumps(prov, indent=2) + "\n")

    print(f"wrote {out}")
    print(f"  checkpoint  {prov['checkpoint']}")
    print(f"  run         {run_dir}  (manifest present)")
    print(f"  params      {prov['params']}  tensors {prov['tensors']}  "
          f"mb {prov['mb']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
