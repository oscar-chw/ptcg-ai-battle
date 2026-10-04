# ptcg-ai-battle

A case study in verifying that a game-AI result is real: imitation learning, self-play PPO
and search baselines for the Kaggle Pokemon TCG AI Battle.

The team's submitted agent, `BEST1_fixed`, finished **rank 2,043 of 6,807 on the final
Simulation leaderboard, score 709.1**, no medal
([results/final_standing.json](results/final_standing.json)). The team's write-up was
submitted to the Kaggle strategy track
([write-up](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/from-imitation-to-reliable-play-a-ptcg-agent-stud)).

**A parity check built the wrong way read green while two submissions scored 256.7 and
183.1 against the team's 800.5 champion.** The serving forward attended over padding the
weights were trained to mask, and the check collated its torch reference the same wrong
way, so both sides agreed ([gate_compute_parity.py](imitation/gates/gate_compute_parity.py),
docstring). How much of the gap that defect caused is the team's diagnosis, not a
measurement: the 800.5 champion was itself served without masks or relations, with
recorded served-vs-trained agreement 0.8736 against 0.8200 for the two failed arms (team
records, private). What is certain is that the defect existed and the check could not see
it; [demo/model_demo.py](demo/model_demo.py) reproduces that false green on a SYNTHETIC
model, next to the corrected check catching it. The repository is about catching
failures like that before they ship: four gates check the built package, and every win
rate carries an interval and a control.

```bash
bash scripts/demo.sh    # standard-library python3, no engine: results table, headline, PPO game counts
bash scripts/check.sh   # every suite that runs without the engine
PTCG_PYTHON=/path/to/python-with-numpy-torch-pytest bash scripts/demo.sh   # adds the model demo
```

![Policy architecture and evaluation boundaries](report/architecture.svg)

Implemented with AI coding agents under Oscar's design and review: the consolidation,
tests, demo and write-ups here; the competition work itself is the team's. The licence is
split by authorship, and no engine code or data is included ([Limits](#limits)).

## The problem

An agent must play a two-player trading-card game with hidden information (hands, deck
order, prize cards), where the set of legal responses changes with every prompt. The
official simulator supplies each observation and its legal options. A submission is a
package (`main.py`, a deck, weights) run in a sandbox where a deep-learning framework is
not guaranteed, and it is ranked on a ladder of simulated games.

The hard part was knowing what was deployed. Besides the false-green parity check above, a
second build path re-shipped a fixed defect 19 hours later (260.9 and 156.9 against 800.5),
and a sweep of 32 packages found 20 that raised inside the sandbox and silently played
random moves ([imitation/README.md](imitation/README.md), "Packaging"). So the project's
real subject became *how to know a result is true*.

## Approach (methods and algorithms)

- **Deck-specialist imitation** on strong players' winning games. A single strong
  demonstrator looked better than a five-player corpus, but that comparison is confounded by
  deck and corpus size: an observation, not a measurement ([docs/details.md](docs/details.md)).
- **A set transformer over board tokens** with a learned attention bias over typed edges,
  scoring each legal option from its own token, so an illegal action has no logit. Inference
  is hand-written NumPy; torch only trains ([imitation/](imitation/README.md),
  [docs/architecture.pdf](docs/architecture.pdf)).
- **Four packaging gates** on the built package: featurizer, serving flags, clean-directory
  import, and NumPy-vs-torch choice on identical weights.
- **Evaluation** ([report/](report/README.md)): seat-balanced batteries, Wilson intervals,
  constant-predictor baselines, negative controls. **Search** (ISMCTS, Gumbel): rejected.
- **Self-play PPO with a STOP head** ([ppo/](ppo/README.md)): how many cards to take became a
  policy decision. Never submitted.

Methods are credited in [report/REFERENCES.md](report/REFERENCES.md) and
[ppo/docs/RESEARCH.md](ppo/docs/RESEARCH.md).

**Who did what.** This was a team entry; teammates are not named. By Oscar's account, he did the
project's engineering, experiments and analysis himself. The folders recorded as team work
(`imitation/`, `report/`) keep "all rights reserved" until the team agrees to a licence. The code
was implemented with AI coding agents under Oscar's design and review.

## Results (real numbers with their source; synthetic clearly labelled)

![Win rates with intervals: search batteries and PPO against its frozen parent](figures/results.png)

Rendered by [figures/plot_results.py](figures/plot_results.py) from committed CSVs;
`figures/tests` checks every plotted number against its source. No result here is
synthetic. Intervals are nominal Wilson 95%.

| Result | Value | Source |
|---|---|---|
| **Final standing, `BEST1_fixed`** | **rank 2,043 of 6,807; score 709.1** | [final_standing.json](results/final_standing.json) |
| Value head vs a zero-skill constant | value head Brier 0.5771; a class-frequency constant scores 0.487022 on 17,392 validation rows with no draws, so the head did worse than no skill **in aggregate** (the team had compared it with a 0.6667 three-class reference; a later head was useful late in games, see details) | [analysis_output.txt](results/analysis_output.txt), [details](docs/details.md#the-value-head) |
| ISMCTS, rejected | 111/400 = 27.75% [23.59, 32.33] vs same-deck baseline; 108/400 cross-deck | [analysis_output.txt](results/analysis_output.txt) |
| PPO vs frozen parent (never submitted) | 0.8104 [0.756, 0.855] on 240 games; control, parent vs itself, 0.4875 [0.419, 0.556] on 200 | [ppo_RESULTS.md](results/ppo_RESULTS.md), [ppo_counts.py](figures/ppo_counts.py) |

PPO caveats: the game counts were not recorded; they are the only ones the intervals allow
(194.5/240, 97.5/200), and the half-points mean draws scored 0.5 there, not 0 as in the
battery rows. 0.8104 is the best of three arms on the same opponent; Bonferroni over 3 gives
[0.743, 0.863]. Against an opponent never trained against: PPO 0.7167 [0.663, 0.765],
champion 0.5333 [0.409, 0.654] on only about 60 games (why the samples differ is not
recorded).

[docs/details.md](docs/details.md) has the full table, the negative results, per-deck
accuracies with the one recorded baseline, and one checkpoint that scored 851.5, 305.9 and
133.1 in three serving packages, **cause not recorded**.

## How to run (under 5 minutes)

```bash
bash scripts/demo.sh     # no engine, standard-library python3; under a second when observed
bash scripts/check.sh    # no engine; observed about 1 s without torch, about 20 s with it
```

`demo.sh` fails unless the recomputed table equals
[results/analysis_output.txt](results/analysis_output.txt). With `PTCG_PYTHON` set it also runs
[demo/model_demo.py](demo/model_demo.py): the NumPy serving forward scores SYNTHETIC boards
beside torch, a badly built parity check reads a false PASS, and the correct one catches it. `check.sh` prints suites it
cannot run as SKIPPED, never passed. [CI](.github/workflows/ci.yml) is configured to run both,
installing numpy and CPU torch on the runner; it has not run yet (not pushed). Timings are
single local observations.

A live match needs the official engine, licensed for competition use only and not included
([demo/README.md](demo/README.md); Python 3.10+):

```bash
PTCG_ENGINE_DIR=/path/to/sample_submission python3 demo/run_match.py --games 1000
```

## Architecture

```
imitation/   set transformer, featurizer, trainer, NumPy serving, 4 packaging gates, tests
ppo/         STOP head, PPO objective, advantage, opponent pool; 115 tests; design docs
report/      the case study and analyze_results.py (standard library)
results/     every number as a file;  figures/  the figure, PPO count recovery, tests
demo/        live match on your engine; model demo on SYNTHETIC boards
docs/        architecture paper, details.md;  engine/  GPU prototype: numbers, parity output
```

The figure at the top shows the data flow: replays to tokens, training, NumPy export with
stamped serving flags, then one script that builds and gates the package.

### Design decisions and trade-offs

- **Imitation, with RL only as a fine-tune.** Imitation is bounded by its teachers (Spidops
  copied a player who won 6.2% of games); the team's self-play loop had shown no learning gain.
- **NumPy at inference.** Torch was not guaranteed in the sandbox; the cost is two forwards
  that can drift, which the parity and serve-stamp gates police.
- **No search shipped.** ISMCTS lost, Gumbel showed nothing, the value head scored worse
  than a constant predictor in aggregate, and search made repeated runs disagree (132/200).
- **PPO never submitted.** No reason is recorded; the opponent pool was not wired in and the
  offline harness failed its own control.
- **A counted serving fallback.** The team's file played a random legal move on any exception,
  silently; this copy plays the lowest legal indices, counts and logs each event, and a test
  guards it ([imitation/README.md](imitation/README.md)).

Sources: [docs/details.md](docs/details.md).

### GPU prototype (simplified game loop)

A batched Rust/GPU simulator of a **simplified slice** of the game loop (setup, draw, energy,
attacks with weakness and resistance, knock-outs, prizes, retreat, bench, three win conditions,
legal-action mask, observation): **~110M environment steps/s on an Apple M2 Max** (Metal, batch
262,144), about 4.4–4.7x the same slice on 12 CPU cores. Measured 110.14M originally and
109.72M / 109.47M when re-run on 2026-10-03 ([raw output](engine/results/gpubench-2026-10-03.txt)).
GPU output is checked against the prototype's own Rust CPU reference: 4,096 seeded games over
48 steps, full state and legal masks identical after every step, observations within float
tolerance; 3/3 parity tests pass ([output](engine/results/parity-2026-10-03.txt)). Not
rule-complete, not parity-tested against the official engine, never used for training; the code
is a derivative of the competition-use-only engine and stays private. Details and how it is
reproduced: [engine/README.md](engine/README.md).

## Limits

- **`BEST1_fixed` cannot be reproduced here** (no weights, corpus or final archive), and no
  controlled evaluation of it was recovered ([report/LIMITATIONS.md](report/LIMITATIONS.md)).
  The battery rows are earlier development branches.
- **The featurizer and trainer need the engine**; tests use a tiny SYNTHETIC model. The PPO
  numbers are records: their drivers and the 45M checkpoint are not included.
- **Ladder scores drift**: one package scored 800.5 and 851.5 two days apart
  ([ppo/docs/BASELINE.md](ppo/docs/BASELINE.md)). Only the final standing is the result.
- **Final submission.** An earlier checkpoint (`BEST1_fixed`) was the final submission; the
  stronger `FIXED-312` (851.5 mid-competition) could not be swapped in before the deadline.
- **Pokemon names are third-party trademarks**; no artwork, card text or engine files.
- **Licence, split by authorship.** The root MIT licence ([LICENSE](LICENSE)) covers Oscar's
  parts only: `ppo/`, `demo/`, `scripts/`, `figures/`, `.github/`, `engine/README.md` and this
  README. `imitation/` and `report/` are the team's and carry their own all-rights-reserved
  `LICENSE` files; `results/` and `docs/` record the team's work and are not MIT either. The
  engine (`LicenseRef-PTCG-ABC-Competition-Use-Only`) is read from `PTCG_ENGINE_DIR`.

## What I learned

Lessons from conclusions the records state, confirmed by Oscar on 2026-10-03.

1. **A threshold can be met by a model with no skill.** The constant predictor scores Brier
   0.487022, so a gate of "Brier below 0.5" would admit zero skill; a value-loss gate needs
   that baseline beside it ([report/REPORT.md](report/REPORT.md) section 4).

2. **A sophisticated method can lose to a plain baseline, and the loss belongs to the
   implementation.** ISMCTS won 111/400 and 108/400 and was rejected "in our evaluated
   implementation, not as a general research direction"
   ([report/REPORT.md](report/REPORT.md) section 5).

3. **A go/no-go gate only means something if a miss stops the line.** The older distillation
   line scored 0.376667 against a 0.55 gate, never passed, and was not carried forward
   ([results/negative_results.md](results/negative_results.md) section 2).

4. **Passing every gate does not show the deployed agent is the trained one.** A parity check
   built the same wrong way as the serving forward could not see that the served function
   differed from the trained one, and a second build path re-shipped a fixed defect. The packaging became four gates run on the built package
   ([imitation/README.md](imitation/README.md), "Packaging";
   [gate_compute_parity.py](imitation/gates/gate_compute_parity.py)).
