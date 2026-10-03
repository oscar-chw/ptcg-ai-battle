#!/usr/bin/env bash
# Package one deck arm for submission.
#   package_arms.sh <RUN_ID> <DECK_CSV> <CHECKPOINT>   (run from imitation/ or anywhere)
#
# THIS FILE IS A WRAPPER, AND THAT IS THE ENTIRE POINT OF IT.
#
# It used to be a second, independent packager -- 28 lines that duplicated
# package_and_tar.sh badly. On 2026-08-14 the effect_tags.json reachability bug
# was found, root-caused to featurize.py's ROOT = parents[1], fixed, and proven
# by a fidelity gate. The fix landed in package_and_tar.sh. It never landed
# here, because nobody remembered this path existed.
#
# Nineteen hours later this script built A2-deckcold-then-teacherflg and
# A4-deckgeneral-then-teacherflg. Both shipped effect_tags.json at the package
# root, where nothing reads it; featurize.py raised on import; main.py's
# fail-closed guard swallowed it and returned random.sample() for EVERY
# decision. They scored 260.9 and 156.9 against the champion's 800.5, and both
# submission descriptions asserted "effect_tags.json package-local" -- false as
# written, because the seam gate certified it by asking whether the file existed
# SOMEWHERE rather than whether the code could reach it.
#
# Measured afterwards across all 32 packages on this tree: 3 load correctly,
# 9 import with CARD_TAGS silently 0, and 20 raise -- i.e. play at random.
#
# A FIX THAT LANDS IN ONE BUILD PATH IS NOT A FIX. So there is now exactly one
# packager, and this file forwards to it. Do not reintroduce a second one: if
# arms need behaviour submissions do not, add a flag to package_and_tar.sh.
#
# The three forwarded options are not optional and each one is a defect closed:
#   --main serving/main_v7.py I-1: v6 masks nothing, so it serves a different
#                             function than the graph trained. Measured 0.7926
#                             argmax agreement with its own checkpoint.
#   --run-manifest            I-2/F-10: the serving mode is a property of THE
#                             RUN, recovered from its own record. v7 without the
#                             stamp guesses, and guesses wrong -- auto-detection
#                             resolves mask_pad_options=False where the champion
#                             manifest says True.
#   --pkg-local-tags          F-9: the lib/ + symlink layout so .resolve() lands
#                             inside the package and parents[1] IS the package.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ $# -eq 3 ] || { echo "usage: package_arms.sh <RUN_ID> <DECK_CSV> <CHECKPOINT>" >&2; exit 2; }
RUN=$1; DECK=$2; CK=$3
MAN=runs/$RUN/manifest.json

# THE MANIFEST IS REQUIRED AND ITS ABSENCE MUST FAIL HERE, LOUDLY.
# Defaulting to "no stamp" would produce a well-formed package that resolves its
# serving mode by guessing -- the substitution antipattern, and the reason A2/A4
# passed every gate they were given. An absent key fails; it never defaults.
[ -f "$MAN" ] || {
  echo "FATAL: no run manifest at $MAN" >&2
  echo "  The serving mode is a property of the run that trained these weights." >&2
  echo "  Without it the package cannot record which mode to serve, and main_v7" >&2
  echo "  would fall back to auto-detection. Refusing to build an unstamped arm." >&2
  exit 1
}
[ -f "$CK" ] || { echo "FATAL: no checkpoint at $CK" >&2; exit 1; }

exec bash serving/package_and_tar.sh "$RUN" "$RUN" "$DECK" "$CK" \
    --main serving/main_v7.py \
    --run-manifest "$MAN" \
    --pkg-local-tags
