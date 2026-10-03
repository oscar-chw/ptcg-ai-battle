#!/usr/bin/env bash
# Build a submission package, gate it, and TAR IT IN THE SAME BREATH.
#   package_and_tar.sh <NAME> <RUN_ID> <DECK_CSV> <CHECKPOINT> --main <path> [options]
#
# Run it from anywhere; it works inside imitation/. Packages and tarballs are
# written to $PTCG_BUILD_DIR (default imitation/build, git-ignored). It needs:
#   PTCG_ENGINE_DIR   the directory that holds the official engine's cg/ package.
#                     A package must carry cg/ (the sandbox runs it), so cg/ is
#                     copied from YOUR install into the build output, never into
#                     this repository.
#   .gate/marnie-sixthsense/effect_tags.json
#                     derived from the engine, so it is not shipped here.
#   PYTHON            an interpreter with numpy and torch (default python3).
#
# --main is REQUIRED. The other options default to the safe form. Passing no
# flags used to build an unmasked v6 forward + an unstamped weights.npz + a flat
# package, which is bit-for-bit the shape that played at random on the ladder.
#
#   --main <path>          the serving forward to ship as main.py (required;
#                          serving/main_v7.py serves the graph as trained).
#   --run-manifest <path>  stamp the three serving flags into weights.npz from
#                          this run's manifest.json, so main_v7 resolves the
#                          mode from the run's OWN RECORD instead of inferring
#                          it from the weights. See below.
#   --pkg-local-tags       lay the package out so featurize.py resolves
#                          effect_tags.json INSIDE the extracted package.
#
# WHY --run-manifest EXISTS, AND WHY SHIPPING main_v7 WITHOUT IT IS NOT A FIX.
# main_v7 serves padding masks and the relational bias the way training ran
# them, but weights.npz carries no serving flags, so a package cannot recover
# WHICH mode to serve. --mask-pad-options in particular changes the input and
# never the parameters, so it leaves no trace in the weights at all. main_v7
# then falls back to auto-detection and finally to the champion defaults, and
# serving a masked graph for a checkpoint that trained unmasked is the same
# defect pointing the other way. Measured on the original tree: for the champion
# D2-3121746f2b28 auto-detection resolves mask_pad_options=False where the run
# manifest says True. So v7 and the stamp ship TOGETHER or neither ships.
#
# WHY THE UNIQUE NAME AND THE IMMEDIATE TAR. Another session was concurrently
# writing into the shared submissions directory, and it overwrote the fixed
# main.py in two packages with an older copy (md5 81e9f328) at 16:46 and 17:26 --
# measured, not suspected. That file hardcodes heads=12, which on d512/8-head
# weights raises inside agent()'s fail-closed guard and turns the agent into a
# random legal move generator. A package gated at T and tarred at T+20min is
# therefore not the package that was gated. This builds under a name nothing
# else has a reason to touch, re-checks the hash immediately before tarring, and
# verifies the hash INSIDE the finished tarball (sha256 here; the incident above
# was found with md5).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python3}
OUT=${PTCG_BUILD_DIR:-build}
: "${PTCG_ENGINE_DIR:?set PTCG_ENGINE_DIR to the directory that contains the official cg/ package}"
[ -d "$PTCG_ENGINE_DIR/cg" ] || { echo "FATAL: $PTCG_ENGINE_DIR/cg is not a directory" >&2; exit 2; }
[ -f .gate/marnie-sixthsense/effect_tags.json ] || {
  echo "FATAL: .gate/marnie-sixthsense/effect_tags.json is missing. It is derived from" >&2
  echo "  the official engine, so it is not shipped here; supply your own." >&2
  exit 2
}
[ $# -ge 4 ] || {
  echo "usage: package_and_tar.sh <NAME> <RUN_ID> <DECK_CSV> <CHECKPOINT> --main <path> [options]" >&2
  exit 2
}
NAME=$1; RUN=$2; DECK_CSV=$3; CK=$4
shift 4
sha256_of() { "$PY" -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'; }
# F-11: THESE DEFAULTS USED TO BE THE BUG WITH A FLAG IN FRONT OF IT.
# an old main + no stamp + flat layout is EXACTLY the A2/A4 package: an unmasked
# forward, a serving mode resolved by guessing, and effect_tags.json where
# nothing can read it -- measured at 260.9 and 156.9 against 800.5, playing
# uniformly at random. Any caller who omitted the flags got that, and the only
# exit 1s in this file compare the shipped main.py to the SOURCE main.py, so
# they would faithfully confirm the old main had shipped correctly.
#
# Safety is no longer opt-in. --main must be named, the stamp must be supplied
# or explicitly waived, and the package-local layout is the default.
MAIN_SRC=
RUN_MANIFEST=
NO_STAMP=0
PKG_LOCAL_TAGS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --main)           MAIN_SRC=$2; shift 2 ;;
    --run-manifest)   RUN_MANIFEST=$2; shift 2 ;;
    --no-stamp)       NO_STAMP=1; shift ;;
    # accepted and now redundant: the safe layout is the default. Kept so
    # existing callers do not break, and because a caller that names it is
    # stating an intent worth honouring rather than an error worth rejecting.
    --pkg-local-tags) PKG_LOCAL_TAGS=1; shift ;;
    --legacy-tags)    PKG_LOCAL_TAGS=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$MAIN_SRC" ] || {
  echo "FATAL: --main is required and has no default." >&2
  echo "  An earlier main_v6 (not in this repo) masked NOTHING: measured 0.7926" >&2
  echo "                    argmax agreement with its own trained graph." >&2
  echo "  serving/main_v7.py  serves the graph as trained. Needs --run-manifest." >&2
  echo "  Naming it is the point -- the old forward stays buildable, never accidental." >&2
  exit 2
}
[ -f "$MAIN_SRC" ] || { echo "FATAL: --main $MAIN_SRC does not exist" >&2; exit 2; }
[ -f "$DECK_CSV" ] || { echo "FATAL: deck csv $DECK_CSV does not exist" >&2; exit 2; }

[ -n "$RUN_MANIFEST" ] || [ "$NO_STAMP" = 1 ] || {
  echo "FATAL: no --run-manifest, and --no-stamp was not passed." >&2
  echo "  The serving mode is a property of the run that trained these weights." >&2
  echo "  Without the stamp main_v7 falls back to auto-detection, which is" >&2
  echo "  measurably wrong: on D2-3121746f2b28 it resolves mask_pad_options=False" >&2
  echo "  where the run manifest says True. Pass --run-manifest <path>, or pass" >&2
  echo "  --no-stamp to state deliberately that this package carries no mode." >&2
  exit 2
}
[ -z "$RUN_MANIFEST" ] || [ -f "$RUN_MANIFEST" ] || {
  echo "FATAL: --run-manifest $RUN_MANIFEST does not exist" >&2; exit 2; }
D=$OUT/$NAME
WANT=$(sha256_of < "$MAIN_SRC")

mkdir -p "$OUT"
rm -rf "$D"; mkdir -p "$D"
cp -r "$PTCG_ENGINE_DIR/cg" "$D"/
cp "$MAIN_SRC" "$D"/main.py
cp training/deck_tracker.py "$D"/
cp "$DECK_CSV" "$D"/deck.csv
cp .gate/marnie-sixthsense/effect_tags.json "$D"/effect_tags.json
cp training/featurize.py "$D"/featurize.py
# featurize.py imports require_option_type from strict_get (the silent-default
# work). WITHOUT THIS COPY the featurizer raises ModuleNotFoundError inside the
# extraction, main.py's fail-closed guard swallows it, and the agent plays
# UNIFORMLY AT RANDOM -- F-9 exactly, reintroduced by the fix that closed it.
# Caught by gate_package_reachability, which is why that gate exists.
cp training/strict_get.py "$D"/strict_get.py

# WHERE effect_tags.json HAS TO SIT FOR featurize.py TO FIND IT.
#
# featurize.py computes ROOT = Path(__file__).resolve().parents[1] and reads
# ROOT/.gate/marnie-sixthsense/effect_tags.json, refusing to import without it.
# Kaggle extracts the package into /kaggle_simulations/agent/, so a featurize.py
# at the package root resolves that ONE LEVEL ABOVE the extraction -- where this
# package shipped nothing. It raises, main.py's fail-closed guard swallows the
# exception, and the agent plays random legal moves for the entire game. That is
# the 293.4 ladder score, and the package-root copy of effect_tags.json above
# does NOT fix it, because nothing ever reads that copy.
#
# --pkg-local-tags puts the real featurize.py one directory down and leaves a
# symlink at the package root. .resolve() follows the symlink, so ROOT becomes
# the package directory ITSELF and the shipped .gate/ copy is the one read.
# featurize.py stays BYTE-IDENTICAL to the training copy -- the layout moves,
# the file does not, so the seam gate's hash check is unaffected.
if [ "$PKG_LOCAL_TAGS" = 1 ]; then
  mkdir -p "$D"/lib "$D"/.gate/marnie-sixthsense
  mv "$D"/featurize.py "$D"/lib/featurize.py
  ln -s lib/featurize.py "$D"/featurize.py
  cp .gate/marnie-sixthsense/effect_tags.json "$D"/.gate/marnie-sixthsense/effect_tags.json
fi

"$PY" serving/export_ss_numpy.py --ckpt "$CK" --out "$D"/weights.npz \
    ${RUN_MANIFEST:+--run-manifest "$RUN_MANIFEST"} > "$OUT/export_$NAME.json"
find "$D" -name __pycache__ -type d -prune -exec rm -r {} + 2>/dev/null || true

# PROVENANCE. Without this the package records nothing linking these weights to
# the run that produced them, so gate_serve_stamp.py can only verify the serve.*
# stamps if the caller supplies --run from outside knowledge -- i.e. only the
# person who built it can check it. Measured: A2-v7-stamped carried a CORRECT
# stamp and still failed the gate for exactly this reason.
"$PY" serving/write_provenance.py --package "$D" --ckpt "$CK" --force

if [ -n "$RUN_MANIFEST" ]; then
  echo "=== SERVING-MODE STAMP $NAME ==="
  "$PY" - "$D/weights.npz" "$RUN_MANIFEST" <<'PYEOF'
import sys
sys.path.insert(0, "serving")
import numpy as np
from export_ss_numpy import SERVE_FLAGS, serve_flags_from_manifest

npz, man = sys.argv[1], sys.argv[2]
W = np.load(npz)
keys = sorted(k for k in W.files if k.startswith("serve."))
# AN EMPTY WORK-LIST MUST FAIL. A check that iterates the stamps it finds and
# finds none would otherwise print nothing and exit 0 -- reporting "the stamp is
# fine" as the indistinguishable twin of "there is no stamp", which is the exact
# defect this flag exists to close.
assert keys, "no serve.* keys in %s -- the stamp did NOT land" % npz
got = {k[len("serve."):]: bool(np.asarray(W[k]).reshape(-1)[0] != 0) for k in keys}
assert sorted(got) == sorted(SERVE_FLAGS), \
    "stamped %s, expected all of %s" % (sorted(got), sorted(SERVE_FLAGS))
truth = serve_flags_from_manifest(man)
assert got == truth, "stamp %s disagrees with manifest %s" % (got, truth)
for f in SERVE_FLAGS:
    print("  serve.%-18s %s" % (f, got[f]))
print("  3/3 stamps present and equal to %s" % man)
PYEOF
fi

if [ "$PKG_LOCAL_TAGS" = 1 ]; then
  echo "=== PACKAGE-LOCAL effect_tags $NAME ==="
  "$PY" - "$D" <<'PYEOF'
import pathlib
import sys

d = pathlib.Path(sys.argv[1]).resolve()
root = (d / "featurize.py").resolve().parents[1]
tags = root / ".gate/marnie-sixthsense/effect_tags.json"
assert root == d, \
    "featurize.py resolves ROOT to %s, which is OUTSIDE the package %s" % (root, d)
assert tags.is_file(), "%s missing" % tags
print("  featurize.py ROOT resolves to the package itself: %s" % root)
print("  effect_tags.json read from: %s" % tags.relative_to(d))
PYEOF
fi

echo "=== SEAM GATE $NAME ==="
"$PY" gates/gate_serving_seam.py --submission "$D"

GOT=$(sha256_of < "$D/main.py")
[ "$GOT" = "$WANT" ] || { echo "ABORT: main.py drifted before tar ($GOT != $WANT)"; exit 1; }

TAR=$OUT/$NAME.tar.gz
tar -czf "$TAR" -C "$D" .
# verify what is actually INSIDE the tarball, not what was on disk
INTAR=$(tar -xzOf "$TAR" ./main.py | sha256_of)
[ "$INTAR" = "$WANT" ] || { echo "ABORT: tarball main.py is $INTAR, expected $WANT"; exit 1; }
echo "TAR_OK $TAR"
echo "main.py sha256 in tarball: $INTAR ($MAIN_SRC)"
tar -tzf "$TAR" | grep -E '^\./(main|featurize|deck_tracker)\.py$|^\./lib/featurize\.py$|weights.npz|deck.csv|effect_tags.json|^\./PROVENANCE\.json$' | sort
du -h "$TAR"
