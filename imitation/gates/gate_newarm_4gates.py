#!/usr/bin/env python3
"""D4: does a package built from a NEW post-fix arm pass all four gates?

This is the evidence that defects I-1 and I-2 are closed. It is deliberately NOT
a transcriber.

WHY IT RE-VERIFIES INSTEAD OF READING PRIOR ARTEFACTS. `.gate/` holds 214 json
files, several of them stamped hours ago against packages that have since been
rebuilt. A combiner that reads them would report PASS from stale evidence -- the
same class as the seam gate that asked whether effect_tags.json existed
SOMEWHERE, and the same class as a `test -s` gate passing on a stub. Three of
the four checks are cheap, so they are re-run here against the package as it
exists right now.

Parity is the exception: it costs minutes, so it is READ -- but its artefact
must be NEWER than the package's main.py and weights.npz, or it is rejected.
Freshness is asserted, never assumed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PY = sys.executable
GATES = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--tarball", required=True, type=Path)
    ap.add_argument("--parity", required=True, type=Path,
                    help="the compute-parity artefact for THIS package")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    checks: dict[str, dict] = {}

    # 1. SEAM -- re-run
    rc, out = _run([PY, str(GATES / "gate_serving_seam.py"), "--submission", str(a.package)])
    checks["seam"] = {"ok": rc == 0, "exit": rc,
                      "detail": out.strip().splitlines()[-1] if out.strip() else ""}

    # 2. SERVE STAMP -- re-run, and with NO --run so it must resolve from the
    #    package's own PROVENANCE.json. A stamp only its author can check is a
    #    stamp on trust.
    rc, out = _run([PY, str(GATES / "gate_serve_stamp.py"), "--package", str(a.package)])
    checks["serve_stamp"] = {"ok": rc == 0, "exit": rc,
                             "detail": out.strip().splitlines()[-1] if out.strip() else ""}

    # 3. REACHABILITY -- extract the tarball to a directory it has never been in
    #    and import the featurizer there. Existence is not reachability: 13
    #    packages satisfied `is_file()` and could not reach their data file.
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(a.tarball) as tf:
            tf.extractall(td)
        # run it *inside* the extraction, not the repo
        p = subprocess.run([PY, "-c",
                            "import sys; sys.path.insert(0,'.')\n"
                            "import featurize\n"
                            "print('%s %d' % (featurize.NUMERIC_WIDTH, len(featurize.CARD_TAGS)))"],
                           cwd=td, capture_output=True, text=True)
        ok = p.returncode == 0
        checks["reachability"] = {
            "ok": ok, "exit": p.returncode,
            "detail": (p.stdout.strip() if ok else (p.stdout + p.stderr).strip().splitlines()[-1]),
        }

    # 4. PARITY -- read, but ONLY if it is newer than what it claims to describe.
    if not a.parity.is_file():
        checks["parity"] = {"ok": False, "detail": f"absent: {a.parity}"}
    else:
        pm = a.parity.stat().st_mtime
        newer_than = []
        for f in ("main.py", "weights.npz"):
            t = (a.package / f)
            if t.is_file() and t.stat().st_mtime > pm:
                newer_than.append(f)
        if newer_than:
            checks["parity"] = {
                "ok": False,
                "detail": f"STALE -- {', '.join(newer_than)} is newer than the parity "
                          f"artefact, so it describes a package that no longer exists",
            }
        else:
            d = json.loads(a.parity.read_text())
            agree = d.get("agreement", d.get("top1_agreement"))
            checks["parity"] = {"ok": bool(d.get("status") == "PASS"),
                                "agreement": agree, "detail": f"status={d.get('status')}"}

    # AN EMPTY WORK-LIST MUST FAIL. Four checks are expected; if the dict is
    # short, something did not run and reporting a pass over it would be the
    # exact defect this file exists to catch.
    assert len(checks) == 4, f"expected 4 checks, built {len(checks)}: {sorted(checks)}"

    status = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"schema": "ptcg-ai/newarm-4gates/v1", "package": str(a.package),
         "tarball": str(a.tarball), "checks": checks, "status": status}, indent=2) + "\n")

    for k, c in checks.items():
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {k:14s} {c.get('detail','')[:96]}")
    print(f"status {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
