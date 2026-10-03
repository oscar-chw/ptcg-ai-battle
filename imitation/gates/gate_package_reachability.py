#!/usr/bin/env python3
"""Can a submission package actually LOAD ITSELF from a clean extraction?

WHY EXISTENCE CHECKS ARE NOT ENOUGH, MEASURED. An earlier version of
`gates/gate_serving_seam.py` asked whether `effect_tags.json` is present in the repo OR at the package root:

    if not here.is_file() and not packaged.is_file():
        fails.append(...)

It never asks whether the CODE can reach it from where the code looks. The
featurizer computes `ROOT = Path(__file__).resolve().parents[1]` and reads
`ROOT/.gate/marnie-sixthsense/effect_tags.json`. For a package whose featurize.py
sits at the package root, `parents[1]` is one directory ABOVE the extraction, so the
file can be present at the package root -- passing the existence check -- and still
be unreachable.

WHAT THAT COSTS, MEASURED 2026-08-15 across all 32 packages in submissions/:

    OK           3   featurizer imports, CARD_TAGS populated
    OK-ZEROTAGS  9   imports fine, CARD_TAGS == 0 -- 21 effect-tag columns silently
                     emitted as zeros, no shape change, no gate fires
    RAISES      20   featurizer cannot import at all

The RAISES case is the severe one. `main.py` imports featurize INSIDE agent()'s try
block and its except returns `random.sample(range(n_opt), ...)`. So a package that
cannot import its featurizer does not crash and does not report an error -- it plays
UNIFORMLY AT RANDOM on every decision, with the weights never consulted, and looks
like a working submission the whole time.

THE ONLY HONEST TEST IS TO DO WHAT THE SANDBOX DOES: copy the package somewhere it
has never been, import it, and see. Everything else is an inference about a path.

    python gates/gate_package_reachability.py --all --out build/package_reachability.json
    python gates/gate_package_reachability.py --package build/<package>
    python gates/gate_package_reachability.py --self-test        # negative controls
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Run inside the copied package. Prints one line of JSON; never raises out.
PROBE = r"""
import json, sys
sys.path.insert(0, ".")
out = {}
try:
    import featurize as F
    ct = getattr(F, "CARD_TAGS", None)
    out["status"] = "OK" if (ct is None or len(ct) > 0) else "OK-ZEROTAGS"
    out["card_tags"] = -1 if ct is None else len(ct)
    out["numeric_width"] = getattr(F, "NUMERIC_WIDTH", None)
    out["root"] = str(getattr(F, "ROOT", ""))
except BaseException as e:
    out["status"] = "RAISES"
    out["error_type"] = type(e).__name__
    out["error"] = str(e)[:300]
print(json.dumps(out))
"""


def probe_package(pkg: Path, python: str, timeout: int = 120) -> dict:
    """Copy to a fresh directory and import there. The copy is the whole point."""
    with tempfile.TemporaryDirectory(prefix="reach_") as td:
        dest = Path(td) / "agent"
        try:
            shutil.copytree(pkg, dest, symlinks=True)
        except Exception as e:  # noqa: BLE001
            return {"status": "COPY-FAILED", "error": str(e)[:200]}
        try:
            r = subprocess.run([python, "-c", PROBE], cwd=dest, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT"}
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return {"status": "NO-OUTPUT", "error": (r.stderr or "")[-200:]}
        try:
            return json.loads(line[-1])
        except Exception:  # noqa: BLE001
            return {"status": "UNPARSEABLE", "error": line[-1][:200]}


def plays_randomly_on_failure(pkg: Path) -> bool:
    """Does main.py swallow an import error into a random legal move?

    This is what turns 'cannot import' into 'plays randomly' rather than 'errors
    out'. A package that crashes loudly is far safer than one that silently
    degrades, so the distinction belongs in the report.
    """
    main = pkg / "main.py"
    if not main.is_file():
        return False
    src = main.read_text(errors="replace")
    return "random.sample" in src and "except Exception" in src


def evaluate(packages, python):
    rows = []
    for pkg in packages:
        res = probe_package(pkg, python)
        res["package"] = pkg.name
        res["random_on_failure"] = plays_randomly_on_failure(pkg)
        res["ok"] = res.get("status") == "OK"
        rows.append(res)
    return rows


def _print(rows):
    w = max([len(r["package"]) for r in rows] + [12])
    print(f"{'PACKAGE':<{w}}  {'STATUS':<12} {'WIDTH':>6} {'TAGS':>6}  NOTE")
    print("-" * (w + 46))
    for r in sorted(rows, key=lambda x: (x["ok"], x["package"])):
        note = ""
        if r["status"] == "RAISES":
            note = (f"{r.get('error_type', '')} -- "
                    + ("PLAYS RANDOMLY (main.py swallows it)" if r["random_on_failure"]
                       else "would error out"))
        elif r["status"] == "OK-ZEROTAGS":
            note = "silent zero-fill of the effect-tag columns"
        print(f"{r['package']:<{w}}  {r['status']:<12} "
              f"{str(r.get('numeric_width', '')):>6} {str(r.get('card_tags', '')):>6}  {note}")
    print("-" * (w + 46))


def self_test(python):
    """NEGATIVE CONTROLS: plant each of the three failure modes, confirm detection.

    Built from scratch rather than by mutating a real package -- a control that
    edits submissions/ would be a control that can damage the thing it checks.
    """
    out = []
    with tempfile.TemporaryDirectory(prefix="reach_selftest_") as td:
        base = Path(td)

        good = base / "good"
        good.mkdir()
        (good / "featurize.py").write_text(
            "from pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "NUMERIC_WIDTH = 269\n"
            "CARD_TAGS = {1: ['a'], 2: ['b']}\n")
        (good / "main.py").write_text("import random\ntry:\n    pass\nexcept Exception:\n"
                                      "    random.sample([], 0)\n")
        r = probe_package(good, python)
        out.append(("a loadable package reads OK", r.get("status") == "OK", r))

        zero = base / "zerotags"
        zero.mkdir()
        (zero / "featurize.py").write_text("NUMERIC_WIDTH = 64\nCARD_TAGS = {}\n")
        r = probe_package(zero, python)
        out.append(("an empty CARD_TAGS reads OK-ZEROTAGS",
                    r.get("status") == "OK-ZEROTAGS", r))

        raises = base / "raises"
        raises.mkdir()
        (raises / "featurize.py").write_text(
            "from pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "p = ROOT / '.gate/marnie-sixthsense/effect_tags.json'\n"
            "if not p.exists():\n"
            "    raise RuntimeError(f'{p} is missing')\n")
        # the file IS present at the package root -- exactly the shape that fools an
        # existence check while remaining unreachable to the code
        (raises / "effect_tags.json").write_text("{}")
        (raises / "main.py").write_text(
            "import random\n"
            "def agent(o):\n"
            "    try:\n"
            "        from featurize import build_tokens\n"
            "    except Exception:\n"
            "        return random.sample(range(3), 1)\n")
        r = probe_package(raises, python)
        detected = r.get("status") == "RAISES"
        swallows = plays_randomly_on_failure(raises)
        out.append(("an unreachable tags path reads RAISES", detected, r))
        out.append(("...and is flagged as PLAYS RANDOMLY", swallows,
                    {"random_on_failure": swallows}))
        exists_check_would_pass = (raises / "effect_tags.json").is_file()
        out.append(("...while a mere existence check would have PASSED it",
                    exists_check_would_pass,
                    {"effect_tags.json at package root": exists_check_would_pass}))

    print(f"{'CONTROL':<58} {'RESULT':<7} DETAIL")
    print("-" * 112)
    bad = 0
    for name, ok, detail in out:
        print(f"{name:<58} {'ok' if ok else 'BROKEN':<7} {json.dumps(detail)[:44]}")
        bad += 0 if ok else 1
    print("-" * 112)
    print("SELF-TEST:", "GREEN -- the gate discriminates all three modes" if not bad
          else f"RED -- {bad} control(s) did not fire")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, action="append", default=[])
    ap.add_argument("--all", action="store_true",
                    help="every package under --root that has a main.py")
    ap.add_argument("--root", type=Path, default=Path("build"),
                    help="directory whose subdirectories are built packages")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test(a.python)

    pkgs = list(a.package)
    if a.all:
        pkgs += [p for p in sorted(a.root.iterdir())
                 if p.is_dir() and (p / "main.py").is_file()]
    if not pkgs:
        ap.error("pass --package or --all")

    rows = evaluate(pkgs, a.python)
    _print(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    n_random = sum(1 for r in rows if r["status"] == "RAISES" and r["random_on_failure"])
    report = {
        "schema": "ptcg-ai/package-reachability/v1",
        "gate": ("a package must import its own featurizer from a clean extraction "
                 "directory with a populated CARD_TAGS; existence of a file "
                 "somewhere is not reachability from where the code looks"),
        "status": "PASS" if n_ok == len(rows) else "FAIL",
        "packages": len(rows), "ok": n_ok,
        "zerotags": sum(1 for r in rows if r["status"] == "OK-ZEROTAGS"),
        "raises": sum(1 for r in rows if r["status"] == "RAISES"),
        "raises_and_plays_randomly": n_random,
        "rows": rows,
    }
    print(f"\n{n_ok}/{len(rows)} packages load correctly; "
          f"{n_random} would PLAY RANDOMLY rather than fail visibly")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"-> {a.out}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
