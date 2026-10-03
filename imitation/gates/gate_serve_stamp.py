#!/usr/bin/env python3
"""Is the `serve.*` stamp TRUE OF THESE WEIGHTS, or merely present?

THE RULE THIS ENFORCES. **A serve flag is not a package setting. It is a property of
the training run that produced the weights.** It must be read from that run, never
inherited from a sibling package, a template, or the last thing that worked.

THE TRAP THIS EXISTS FOR, LIVE AS OF 2026-08-15 21:0x. `A2-v7-stamped` carries
`serve.relations=1`, and that is CORRECT -- its weights come from
`A2-deck-then-teacher`, which trained `relations=True`. The postfix arms now
training print `BUILD_RELATIONS=False`. If they are packaged with the A2 stamp
copied across, `serve.relations=1` against a relations-off checkpoint **recreates
I-1 exactly**: serving computes a function the weights were never fit to, at the
same seam, one week later.

WHY EXISTING CHECKS CANNOT CATCH IT. The stamp would be PRESENT and the package
would load; parity would even pass if it were run in the stamped mode on both sides,
because both sides would be wrong in the same way -- which is precisely how
`gate_export_parity.py` read green through I-1 the first time. Checking that a stamp
exists is the same error as checking that a file exists: presence is not truth.

ABSENT IS NOT FALSE-BY-TODAY'S-DEFAULT. A manifest key missing means the flag did
not exist when that run trained, so the behaviour was the pre-flag one:

    no_mask_pad_state absent  -> MASK_PAD_STATE was OFF   (code masked
                                 unconditionally; reading `not cfg.get(k, False)`
                                 gives True and reconstructs a run that masked when
                                 it did not)
    no_mask_pad_state False   -> MASK_PAD_STATE ON
    no_mask_pad_state True    -> MASK_PAD_STATE OFF
    mask_pad_options absent   -> OFF        relations absent -> OFF

PROVENANCE IS REQUIRED, AND ITS ABSENCE IS A FAILURE. A package that does not record
which run produced its weights cannot have its stamp checked by anyone, ever. As of
this writing `A2-v7-stamped` and `FIXED-717` carry no PROVENANCE.json; `FIXED-312`
does. `--run` lets a caller supply it explicitly, but the missing-provenance failure
is still reported, because a stamp nobody can trace is a claim nobody can check.

    python gates/gate_serve_stamp.py --package build/<package> \
        --run runs/<run> --out build/serve_stamp.json
    python gates/gate_serve_stamp.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVE_KEYS = ("mask_pad_state", "mask_pad_options", "relations")


def training_flags(manifest_path: Path) -> dict:
    """The three serve-relevant switches AS THE RUN ACTUALLY EXECUTED THEM."""
    cfg = json.loads(Path(manifest_path).read_text())
    cfg = cfg.get("config", cfg)
    return {
        # absent => the flag did not exist => the run did NOT mask
        "mask_pad_state": (False if "no_mask_pad_state" not in cfg
                           or cfg.get("no_mask_pad_state") is None
                           else not bool(cfg["no_mask_pad_state"])),
        "mask_pad_options": bool(cfg.get("mask_pad_options") or False),
        "relations": bool(cfg.get("relations") or False),
    }


def stamped_flags(npz_path: Path) -> dict:
    import numpy as np
    z = np.load(npz_path)
    out = {}
    for k in SERVE_KEYS:
        key = f"serve.{k}"
        out[k] = bool(z[key]) if key in z.files else None   # None = not stamped
    return out


def evaluate(package: Path, run: Path | None):
    package = Path(package)
    npz = package / "weights.npz"
    prov = package / "PROVENANCE.json"
    fails, notes = [], []

    if not npz.is_file():
        return {"status": "FAIL", "package": package.name,
                "checks": {"weights_present": {"ok": False,
                                               "detail": f"no weights.npz in {package}"}}}

    resolved_run = run
    if prov.is_file():
        try:
            ckpt = json.loads(prov.read_text()).get("checkpoint")
            if ckpt and resolved_run is None:
                resolved_run = Path(ckpt).parent.parent
            notes.append(f"provenance names {ckpt}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"PROVENANCE.json unreadable: {e}")
    else:
        fails.append("package carries no PROVENANCE.json, so nothing links these "
                     "weights to the run that produced them; the stamp cannot be "
                     "checked by anyone who did not build it")

    stamp = stamped_flags(npz)
    unstamped = [k for k, v in stamp.items() if v is None]
    if unstamped:
        fails.append(f"serve.* missing from weights.npz for {unstamped}; serving "
                     f"falls back to defaults and cannot know what the run did")

    train = None
    if resolved_run is not None and (Path(resolved_run) / "manifest.json").is_file():
        train = training_flags(Path(resolved_run) / "manifest.json")
        for k in SERVE_KEYS:
            if stamp[k] is None:
                continue
            if stamp[k] != train[k]:
                fails.append(
                    f"serve.{k}={int(stamp[k])} but {Path(resolved_run).name} trained "
                    f"{k}={train[k]} -- the package would serve a function these "
                    f"weights were never fit to (this is I-1)")
    else:
        fails.append(f"no training manifest resolved (run={resolved_run}); the stamp "
                     f"is unverifiable, which is a FAILURE and not a pass")

    return {
        "schema": "ptcg-ai/serve-stamp/v1",
        "gate": ("every serve.* in weights.npz must equal the corresponding flag in "
                 "the manifest of the run that produced those weights; a serve flag "
                 "is a property of the checkpoint, not a package setting"),
        "status": "PASS" if not fails else "FAIL",
        "package": package.name,
        "run": str(resolved_run) if resolved_run else None,
        "stamped": {k: (None if v is None else int(v)) for k, v in stamp.items()},
        "trained": train,
        "notes": notes,
        "failures": fails,
    }


def self_test():
    """NEGATIVE CONTROLS. Each planted mismatch must turn the gate RED."""
    import tempfile

    import numpy as np
    out = []
    with tempfile.TemporaryDirectory(prefix="stamp_") as td:
        base = Path(td)

        def make(pkg_name, stamp, cfg, provenance=True):
            pkg = base / pkg_name
            pkg.mkdir(parents=True, exist_ok=True)
            np.savez(pkg / "weights.npz",
                     **{f"serve.{k}": np.array(int(v)) for k, v in stamp.items()})
            run = base / "runs" / (pkg_name + "-run")
            run.mkdir(parents=True, exist_ok=True)
            (run / "manifest.json").write_text(json.dumps({"config": cfg}))
            if provenance:
                (pkg / "PROVENANCE.json").write_text(json.dumps(
                    {"checkpoint": str(run / "checkpoints" / "step-1.pt")}))
            return pkg, run

        pkg, run = make("match", {"mask_pad_state": 1, "mask_pad_options": 1, "relations": 1},
                        {"no_mask_pad_state": False, "mask_pad_options": True, "relations": True})
        r = evaluate(pkg, run)
        out.append(("a stamp matching its run PASSES", r["status"] == "PASS", r["failures"][:1]))

        pkg, run = make("relmismatch", {"mask_pad_state": 1, "mask_pad_options": 1, "relations": 1},
                        {"no_mask_pad_state": False, "mask_pad_options": True, "relations": False})
        r = evaluate(pkg, run)
        out.append(("THE LIVE TRAP: serve.relations=1 on a relations-OFF run is REJECTED",
                    r["status"] == "FAIL", r["failures"][:1]))

        pkg, run = make("absent", {"mask_pad_state": 1, "mask_pad_options": 0, "relations": 0},
                        {"mask_pad_options": False, "relations": False})
        r = evaluate(pkg, run)
        out.append(("an ABSENT no_mask_pad_state is read as OFF, not True",
                    r["status"] == "FAIL", r["failures"][:1]))

        pkg, run = make("absent_ok", {"mask_pad_state": 0, "mask_pad_options": 0, "relations": 0},
                        {"mask_pad_options": False, "relations": False})
        r = evaluate(pkg, run)
        out.append(("...and a stamp agreeing with that reading PASSES",
                    r["status"] == "PASS", r["failures"][:1]))

        pkg, run = make("noprov", {"mask_pad_state": 1, "mask_pad_options": 1, "relations": 1},
                        {"no_mask_pad_state": False, "mask_pad_options": True, "relations": True},
                        provenance=False)
        r = evaluate(pkg, None)
        out.append(("a package with NO provenance and no --run is REJECTED",
                    r["status"] == "FAIL", r["failures"][:1]))

    print(f"{'CONTROL':<66} {'RESULT':<7} FIRST FAILURE")
    print("-" * 122)
    bad = 0
    for name, ok, detail in out:
        print(f"{name:<66} {'ok' if ok else 'BROKEN':<7} {str(detail)[:48]}")
        bad += 0 if ok else 1
    print("-" * 122)
    print("SELF-TEST:", "GREEN -- the gate discriminates" if not bad
          else f"RED -- {bad} control(s) did not fire")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path)
    ap.add_argument("--run", type=Path, default=None,
                    help="the run that produced these weights; preferred source is "
                         "the package's own PROVENANCE.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.package:
        ap.error("--package is required unless --self-test")

    r = evaluate(a.package, a.run)
    print(f"package  {r['package']}")
    print(f"run      {r.get('run')}")
    print(f"stamped  {r.get('stamped')}")
    print(f"trained  {r.get('trained')}")
    for f in r.get("failures", []):
        print(f"  FAIL  {f}")
    print(f"status {r['status']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(r, indent=2, sort_keys=True))
        print(f"-> {a.out}")
    return 0 if r["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
