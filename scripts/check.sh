#!/usr/bin/env bash
# One command that runs everything this repository can check, and says plainly what it
# did not run. Exit 0 only if every suite that RAN passed. It needs no engine.
#
#   bash scripts/check.sh
#
# WHAT RUNS WITH THE STANDARD LIBRARY ALONE (system python3, 3.9 or newer) -- always:
#   report    analyze_results.py reproduces results/analysis_output.txt exactly, and
#             rejects edited inputs                               report/tests
#   imitation serve-flag logic, strict_get, deck_tracker          imitation/tests
#   demo      agents and match loop against a test double of the engine; and the demo
#             itself with PTCG_ENGINE_DIR unset, which must exit 2 with a clear message
#
# WHAT NEEDS numpy + torch (+ pytest for ppo) -- SKIPPED, NOT PASSED, WITHOUT THEM:
#   imitation-numeric   model, NumPy-vs-torch parity, export, the four gates, packaging
#                       imitation/tests_torch  (needs numpy, torch)
#   ppo                 the 115 ptcg_ppo tests                    ppo/tests
#                       (needs numpy, torch, pytest)
#   TODO-DEPENDENCY: this machine's system python3 has none of them, and nothing is
#   installed by this script. Point PTCG_PYTHON at an interpreter that has them
#   (or create ./.venv with them); a skipped suite is printed as SKIPPED below and never
#   counted as a pass.
#
# THE DEMO WITH THE REAL ENGINE runs only if PTCG_ENGINE_DIR is set (and python is 3.10+);
# see demo/README.md. Otherwise only the no-engine behaviour above is checked.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1

PY=${PYTHON:-python3}
if [ -n "${PTCG_PYTHON:-}" ]; then NUM_PY=$PTCG_PYTHON
elif [ -x .venv/bin/python ]; then NUM_PY=.venv/bin/python
else NUM_PY=$PY; fi

failed=0; ran=0; skipped=0
ok()   { echo "PASS     $1"; ran=$((ran + 1)); }
bad()  { echo "FAIL     $1"; ran=$((ran + 1)); failed=$((failed + 1)); }
skip() { echo "SKIPPED  $1 -- missing dependency: $2"; skipped=$((skipped + 1)); }
run()  { # run <label> <command...>; show the tail of the output only on failure
  local label=$1; shift
  local out
  if out=$("$@" 2>&1); then ok "$label"; else bad "$label"; echo "$out" | tail -25; fi
}

has() { "$NUM_PY" -c "import $1" >/dev/null 2>&1; }

echo "== standard library suites ($("$PY" --version 2>&1)) =="
run "report: analyze_results reproduces the committed table" \
    bash -c "diff <($PY report/analyze_results.py) results/analysis_output.txt"
run "report: tests" "$PY" -m unittest discover -s report/tests
run "imitation: tests (serve flags, strict_get, deck_tracker)" "$PY" -m unittest discover -s imitation/tests
run "demo: tests (agents, match loop, no-engine message)" "$PY" -m unittest discover -s demo/tests

# The demo itself. Without an engine it must refuse clearly (exit 2, naming the variable).
if [ -z "${PTCG_ENGINE_DIR:-}" ]; then
  out=$(env -u PTCG_ENGINE_DIR "$PY" demo/run_match.py 2>&1 >/dev/null); code=$?
  if [ "$code" = 2 ] && echo "$out" | grep -q PTCG_ENGINE_DIR; then
    ok "demo: exits 2 and names PTCG_ENGINE_DIR when no engine is given"
  else bad "demo: no-engine behaviour (exit $code)"; echo "$out" | tail -5; fi
else
  if "$NUM_PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    out=$("$NUM_PY" demo/run_match.py --games 100 2>&1); code=$?
    if [ "$code" = 0 ] && echo "$out" | grep -q greedy; then ok "demo: 100 games on the engine at PTCG_ENGINE_DIR"
    else bad "demo: run on the engine (exit $code)"; echo "$out" | tail -8; fi
  else
    skip "demo: run on the engine" "PTCG_ENGINE_DIR is set but $NUM_PY is older than Python 3.10"
  fi
fi

echo "== numeric suites (interpreter: $NUM_PY) =="
if has numpy && has torch; then
  run "imitation-numeric: tests" "$NUM_PY" -m unittest discover -s imitation/tests_torch
else
  skip "imitation-numeric (imitation/tests_torch)" "numpy and torch are not importable by $NUM_PY"
fi
if has numpy && has torch && has pytest; then
  run "ppo: tests" env PYTHONPATH=ppo "$NUM_PY" -m pytest ppo/tests -q -p no:cacheprovider
else
  skip "ppo (ppo/tests)" "numpy, torch and pytest are not all importable by $NUM_PY"
fi

echo "== $ran ran, $failed failed, $skipped skipped =="
[ "$failed" = 0 ]
