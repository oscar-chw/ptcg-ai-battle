# ptcg-ai-battle

**Rank 2,043 of 6,807 on the final Simulation leaderboard of the Kaggle Pokemon TCG AI
Battle competition, final score 709.1, with the submitted agent `BEST1_fixed`**
([results/final_standing.json](results/final_standing.json)). It was a team effort: the
submitted agent and the imitation pipeline behind it are the team's work, and this
repository is a consolidation of that work and of two follow-up experiments that were never
submitted. No medal, and no claim that reinforcement learning improved the submitted agent.

Implemented with AI coding agents under Oscar's design and review. The consolidation, the
cleanup, the tests, the demo and the write-ups in this repository were produced that way;
the competition work itself is credited to the team.

## The problem

The competition asks for an agent that plays a two-player trading-card game with hidden
information (each player's hand, deck order and prize cards), where the set of legal
responses changes with every prompt (play a card, attach energy, choose targets, choose how
many cards to take). The game is run by an official simulator that supplies each observation
and its legal options. A submission is a package (`main.py`, a deck and weights) executed in
a sandbox where a deep-learning framework is not guaranteed to exist, and it is ranked on a
ladder of simulated games.

Two things made this harder than it looks. A model can pass every offline gate and still
play badly, because what is deployed can silently differ from what was trained: two packages
that passed every gate scored 260.9 and 156.9 against an 800.5 champion because the deployed
agent was playing at random, and one checkpoint scored 851.5, 305.9 and 133.1 depending only
on its serving path ([imitation/README.md](imitation/README.md),
[ppo/README.md](ppo/README.md)). So the project's real subject became *how to know a result
is true*.

## Approach (methods and algorithms)

- **Deck-specialist imitation.** One model per deck, trained on strong players' winning
  games. The team measured that imitating one strong player beats imitating an archetype:
  a corpus of five demonstrators gave the same board several conflicting "correct" moves
  ([results/team_results.md](results/team_results.md)).
- **A set transformer over board tokens** with a learned attention bias over typed edges,
  scoring each legal option from its own token, so legality is a property of the input and
  an illegal action has no logit. Inference is hand-written NumPy; torch is only used to
  train ([imitation/](imitation/README.md), [docs/architecture.pdf](docs/architecture.pdf)).
- **Four packaging gates** that check a built package, not an opinion about it: the served
  featurizer is the trained one, the serving flags equal the training run's own flags, the
  package can import itself from a clean directory, and the NumPy forward reproduces torch's
  choice on identical weights.
- **Evaluation discipline** ([report/](report/README.md)): seat-balanced match batteries,
  Wilson intervals, a constant-predictor baseline for any value-loss threshold, and
  negative controls for every gate.
- **Search, tried and rejected:** ISMCTS and a Gumbel candidate against a frozen baseline.
- **Self-play PPO with a STOP head** ([ppo/](ppo/README.md)): the champion cannot decide
  how many cards to take, so the number of picks became a policy decision. Never submitted.
- **A GPU engine slice**, kept as numbers only ([engine/](engine/README.md)).

Methods are credited to their sources in [report/REFERENCES.md](report/REFERENCES.md) (set
transformers, FiLM, DAgger, ISMCTS, expert iteration) and in
[ppo/docs/RESEARCH.md](ppo/docs/RESEARCH.md) (PPO, RLOO, KL anchoring, league training).

## Results (real numbers with their source; synthetic clearly labelled)

No result below is synthetic. The fixtures inside the test suites are, and are labelled
SYNTHETIC where they are defined. Intervals are nominal Wilson 95% intervals for the recorded
samples, not uncertainty over all opponents.

| Result | Value | Source |
|---|---|---|
| **Final Simulation standing, `BEST1_fixed`** | **rank 2,043 of 6,807; score 709.1** | [final_standing.json](results/final_standing.json) |
| `BEST1_fixed` as recorded in its submission | d512, 12 layers, 8 heads, FFN 1365, 269 numeric features; trained on 400 winning games from four demonstrators; epoch-12 checkpoint, 58.16% contested validation accuracy. Recorded facts, not reconstructed: the final archive was not recovered | [REPORT.md, section 1](report/REPORT.md) |
| A preserved imitation corpus | 89,048 decisions from 926 games, 741/185 train/validation games | [REPORT.md, section 3](report/REPORT.md) |
| Development validation top-1 by deck | Marnie/Froslass 0.7785, Alakazam 0.7024, Mega Lopunny 0.6332 | [team_results.md](results/team_results.md) |
| Rule/search baseline, seat-balanced battery | 198/400 = 49.50% [44.63, 54.38]; first seat 116/200, second seat 82/200 (17 points) | [analysis_output.txt](results/analysis_output.txt), [results.csv](results/results.csv) |
| Gumbel candidate vs same-deck baseline | 204/400 = 51.00% [46.11, 55.87]: no evidence either way | [analysis_output.txt](results/analysis_output.txt) |
| Value-head threshold | the constant predictor scores Brier 0.487022 on 17,392 rows from 185 games, so "below 0.5" admits zero skill | [analysis_output.txt](results/analysis_output.txt) |
| Official-engine speedup, macOS arm64 | 1.497701x over the preceding implementation; 50 paired games matched all outcomes and 9,844 moves | [analysis_output.txt](results/analysis_output.txt) |
| Self-play instrumentation gate | 32/32 games, 5,513 decision records, 0 illegal actions: mechanics only, no learning claim | [analysis_output.txt](results/analysis_output.txt) |
| **PPO vs its frozen parent (never submitted)** | **0.8104 [0.756, 0.855]**; control, the parent against itself, 0.4875 [0.419, 0.556]. The number of games is not recorded in the sources | [ppo_RESULTS.md](results/ppo_RESULTS.md) |
| GPU engine slice, M2 Max, Metal | 110.14M env-steps/s and 572.2k games/s at batch 262,144; 213 Rust tests passed (a partial game, not the full rules) | [engine/README.md](engine/README.md) |
| This repository's tests | 115 ppo (pytest), 31 imitation numeric, 19 imitation stdlib, 12 demo, 5 report (unittest) | `bash scripts/check.sh` |

**Negative results**, in full in [results/negative_results.md](results/negative_results.md):

| Result | Value | Source |
|---|---|---|
| Spidops deck model | top-1 **0.5714 against its 0.5943 baseline**; failed, deck dropped (the imitated player won 6.2% of its 48 episodes) | [negative_results.md](results/negative_results.md), [team_results.md](results/team_results.md) |
| Older distillation line, fidelity check | top-1 agreement **0.376667 against a 0.55 target**; best after later phases 0.496667, never passed | [negative_results.md](results/negative_results.md) |
| ISMCTS vs same-deck baseline | **111/400 = 27.75% [23.59, 32.33]**, rejected | [analysis_output.txt](results/analysis_output.txt) |
| ISMCTS cross-deck battery | **108/400 = 27.00% [22.88, 31.55]**, rejected (its opponents differ from the 198/400 baseline row's, so the rows are not pooled) | [analysis_output.txt](results/analysis_output.txt) |

## How to run (under 5 minutes)

```bash
bash scripts/check.sh        # no engine needed; about 1 s without torch, about 18 s with it
```

It runs the standard-library suites, recomputes the results table with
`report/analyze_results.py` and checks it against [results/analysis_output.txt](results/analysis_output.txt),
and runs the demo with no engine, which must exit with a message. The suites that need
numpy, torch and pytest (`imitation/tests_torch`, `ppo/tests`) are **skipped, and printed as
SKIPPED, not passed**, unless `PTCG_PYTHON` points at an interpreter that has them
(TODO-DEPENDENCY: none is installed by this repository).

To play two small agents on the real engine (needs Python 3.10+; about 3 s for 1,000 games):

```bash
PTCG_ENGINE_DIR=/path/to/sample_submission python3 demo/run_match.py --games 1000
```

The engine is not in this repository and cannot be: it is licensed for competition use only.
[demo/README.md](demo/README.md) says how to download it from Kaggle. With the variable
unset the demo prints that and exits with status 2.

## Architecture

```
README.md            this file
scripts/check.sh     everything checkable without the engine
results/             every number in this README, as a file: final standing, results.csv,
                     analysis_output.txt, the PPO results, the negative results
imitation/           set-transformer model, featurizer, trainer, NumPy serving, 4 packaging gates
  training/ serving/ gates/ tests/ tests_torch/
ppo/                 ptcg_ppo (STOP head, PPO objective, advantage, opponent pool) + 115 tests + design docs
report/              the case study and analyze_results.py (standard library only)
engine/              GPU engine slice: numbers and design only, no code
demo/                two agents, one local match on the engine you supply
docs/                the architecture paper
```

![Policy architecture and evaluation boundaries](report/architecture.svg)

Data flows left to right: replays become tokens (`featurize`), tokens train the set
transformer (`train_ss`), the checkpoint is exported to NumPy with its serving flags stamped
from the run's own manifest (`export_ss_numpy`), a package is built by one script
(`package_arms.sh`) and is gated four ways before it is tarred. `ppo/` starts from a
checkpoint and replaces the cardinality constant with a learned STOP decision.

## Limits

- **The submitted agent cannot be reproduced from this repository.** Weights, corpus and the
  final archive are not included, and the exact optimizer, schedule, seed and split of
  `BEST1_fixed` were not reconstructed. No controlled evaluation of it across seats, repeated
  matches and matchups was recovered, so its robustness is unestablished
  ([report/LIMITATIONS.md](report/LIMITATIONS.md)). Development win rates above are earlier
  branches, not `BEST1_fixed`'s.
- **The featurizer and trainer cannot run here.** They need the official engine and
  engine-derived files that are not shipped (see [imitation/README.md](imitation/README.md)).
  What is tested is the model, the NumPy serving forward, the export and stamping, and the
  gates, on a tiny synthetic model.
- **The PPO number is documented, not reproducible here.** The drivers that produced it need
  the 45M checkpoint and tooling that are not included, the number of games is not recorded,
  the opponent pool was not wired in, and nothing was submitted
  ([results/ppo_RESULTS.md](results/ppo_RESULTS.md), "Still open").
- **Ladder scores are not comparable across time.** The same package scored 800.5 and then
  851.5 two days apart ([ppo/docs/BASELINE.md](ppo/docs/BASELINE.md)), and 851.5 and 790.8 are
  mid-competition scores of other submissions; only the final standing, 709.1, is the result.
- **The GPU numbers are a partial game on one laptop GPU**, with no CUDA measurement.
- **The engine speedup applies to one macOS build and workload.** It is not a strength gain.
- **The case study discusses individual cards** from a participant's point of view; Pokemon
  and associated names are third-party trademarks, and no official artwork, card text or
  engine files are included.
- **Licence.** The code and text here are MIT ([LICENSE](LICENSE)). The official engine is
  `LicenseRef-PTCG-ABC-Competition-Use-Only` and is not included; `featurize.py` and the
  packager read it from a directory you supply through `PTCG_ENGINE_DIR`.

## What I learned

TODO-OSCAR: the sources record no conclusions of Oscar's own for this project, so this
section is left for him to write.
