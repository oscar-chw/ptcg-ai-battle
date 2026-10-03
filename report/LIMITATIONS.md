# Evidence and publication boundaries

The public-facing package contains a participant-authored report, original diagrams, aggregate experimental results, mathematical derivations and a statistics script. It contains no raw competition replays, official engine files, card images, credentials, private machine configuration or model weights.

## What was verified

- Preserved local implementations and machine-readable gate records support the historical architecture, corpus, baseline, ISMCTS and rollout claims.
- The exact Mega Lucario deck cohort was reconstructed from two matching replay deck actions and associated with the final agent through its Kaggle submission description.
- Live Kaggle pages supported the final standing and recorded final-agent configuration.
- Primary publication sources support the cited prior methods.
- The supplied script reproduces arithmetic from aggregate records.

## What this package does not establish

- The exact final archive, optimizer, learning-rate schedule, training seed and data split were not reconstructed in the report audit.
- No preserved controlled final-agent evaluation across starting seats, repeated matches and matchups was recovered. Its robustness remains unestablished.
- Historical development win rates are not the final agent's win rates.
- The original trained-weight padding probe and v2 comparison are documented experiments, not reruns included here.
- The engine speedup applies to its tested macOS build and workload. It does not establish Linux or GPU performance.
- Self-play instrumentation passed a finite mechanics gate; successful reinforcement learning or expert iteration was not demonstrated.
- Recalculating aggregate statistics does not reproduce the original matches or training.

These limitations are part of the result. They identify the exact work needed to strengthen the next release and prevent a portfolio reader from inferring evidence that does not exist.

## Publication state

The Strategy report this case study accompanies was submitted to the competition's Strategy Main Track on 13 September 2026, and Kaggle confirmed the submission (see `../results/final_standing.json`). No licence is asserted over third-party game materials; none are included in this repository.
