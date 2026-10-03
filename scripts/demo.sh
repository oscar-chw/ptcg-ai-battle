#!/usr/bin/env bash
# The demo a fresh clone can run: standard-library python3 only, no engine, a few seconds.
#
#   bash scripts/demo.sh
#
# 1. Recomputes the results table from results/results.csv with report/analyze_results.py
#    and fails unless it equals the committed results/analysis_output.txt byte for byte.
# 2. Prints the headline: the final standing and the vacuous value gate, with sources.
# 3. Recovers the PPO head-to-head game counts from their recorded intervals.
# 4. If PTCG_PYTHON points at an interpreter with numpy and torch, runs the NumPy serving
#    model on SYNTHETIC boards and shows the compute-parity check catching a wrong
#    serving mode (demo/model_demo.py). Otherwise says how to enable it.
# 5. Explains how to play a live match on the official engine, which is not in this
#    repository. If PTCG_ENGINE_DIR is already set, it plays that match (Python 3.10+).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
PY=${PYTHON:-python3}

echo "== 1. Results table, recomputed from results/results.csv =="
table=$("$PY" report/analyze_results.py)
echo "$table"
if [ "$table" != "$(cat results/analysis_output.txt)" ]; then
  echo "FAIL: the recomputed table differs from results/analysis_output.txt" >&2
  exit 1
fi
echo "-> identical to the committed results/analysis_output.txt"

echo
echo "== 2. Headline =="
"$PY" -c '
import json
s = json.load(open("results/final_standing.json", encoding="utf-8"))
print("Final Simulation standing of %s: rank %s of %s, score %s (source: %s)." % (
    s["submitted_agent"], format(s["rank"], ","), format(s["ranked_out_of"], ","),
    s["final_simulation_score"], "results/final_standing.json"))
'
echo "A value gate of 'Brier below 0.5' admits zero skill: the constant predictor scores"
echo "0.487022 on 17,392 validation rows (the Brier line of the table above)."

echo
echo "== 3. PPO head-to-head: game counts recovered from the recorded intervals =="
"$PY" figures/ppo_counts.py

echo
echo "== 4. The serving model on SYNTHETIC boards (needs numpy and torch) =="
if [ -n "${PTCG_PYTHON:-}" ] && "$PTCG_PYTHON" -c "import numpy, torch" >/dev/null 2>&1; then
  "$PTCG_PYTHON" demo/model_demo.py
else
  echo "Skipped: set PTCG_PYTHON to an interpreter with numpy and torch to run"
  echo "demo/model_demo.py (about 1 s): the NumPy forward scores a board's legal options,"
  echo "and the compute-parity check passes the trained serving mode and rejects a wrong one."
fi

echo
echo "== 5. A live match on the official engine =="
if [ -z "${PTCG_ENGINE_DIR:-}" ]; then
  cat <<'TEXT'
PTCG_ENGINE_DIR is not set, so no live match is played. The engine is licensed for
competition use only and is not in this repository. To play one (Python 3.10+):

  1. Join the Pokemon TCG AI Battle competition on Kaggle and download its data.
  2. Point PTCG_ENGINE_DIR at the unzipped folder that contains cg/, deck.csv and
     main.py (sample_submission/sample_submission), then run:

     PTCG_ENGINE_DIR=/path/to/sample_submission python3 demo/run_match.py --games 1000

demo/README.md has the details.
TEXT
else
  "$PY" demo/run_match.py --games 1000
fi
