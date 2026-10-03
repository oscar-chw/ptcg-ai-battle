#!/usr/bin/env bash
# The demo a fresh clone can run: standard-library python3 only, no engine, a few seconds.
#
#   bash scripts/demo.sh
#
# 1. Recomputes the results table from results/results.csv with report/analyze_results.py
#    and fails unless it equals the committed results/analysis_output.txt byte for byte.
# 2. Prints the headline: the serving-path spread and the final standing, with sources.
# 3. Explains how to play a live match on the official engine, which is not in this
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
echo "One checkpoint scored 851.5, 305.9 and 133.1 on the ladder depending only on its"
echo "serving path (source: ppo/docs/BASELINE.md, section 2)."
"$PY" - <<'EOF'
import json
s = json.load(open("results/final_standing.json", encoding="utf-8"))
print(f"Final Simulation standing of {s['submitted_agent']}: rank {s['rank']:,} of "
      f"{s['ranked_out_of']:,}, score {s['final_simulation_score']} "
      "(source: results/final_standing.json).")
EOF

echo
echo "== 3. A live match on the official engine =="
if [ -z "${PTCG_ENGINE_DIR:-}" ]; then
  cat <<'EOF'
PTCG_ENGINE_DIR is not set, so no live match is played. The engine is licensed for
competition use only and is not in this repository. To play one (Python 3.10+):

  1. Join the Pokemon TCG AI Battle competition on Kaggle and download its data.
  2. Point PTCG_ENGINE_DIR at the unzipped folder that contains cg/, deck.csv and
     main.py (sample_submission/sample_submission), then run:

     PTCG_ENGINE_DIR=/path/to/sample_submission python3 demo/run_match.py --games 1000

demo/README.md has the details.
EOF
else
  "$PY" demo/run_match.py --games 1000
fi
