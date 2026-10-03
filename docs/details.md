# Details behind the README

The README keeps four headline rows. This file holds the full results table, the
negative results, the serving-package scores whose cause was not recorded, and the
long form of the design decisions. Every number cites its source. None is synthetic.

## Full results

Intervals are nominal Wilson 95% intervals for the recorded samples, not uncertainty over
all opponents. In the battery rows a draw counts as a non-win; no draws occurred. In the
PPO rows a draw or unfinished game scores 0.5 (the evaluator's rule), so the two kinds of
row are not on the same scale.

| Result | Value | Source |
|---|---|---|
| Final Simulation standing, `BEST1_fixed` | rank 2,043 of 6,807; score 709.1 | [final_standing.json](../results/final_standing.json) |
| `BEST1_fixed` as recorded in its submission | d512, 12 layers, 8 heads, FFN 1365, 269 numeric features; 400 winning games from four demonstrators; epoch-12 checkpoint, 58.16% contested validation accuracy. Recorded facts, not reconstructed: the final archive was not recovered | [REPORT.md, section 1](../report/REPORT.md) |
| A preserved imitation corpus | 89,048 decisions from 926 games, 741/185 train/validation games | [REPORT.md, section 3](../report/REPORT.md) |
| Development validation top-1 by deck | Marnie/Froslass 0.7785 against the 0.5943 dense-scorer baseline, which was measured on the Marnie episode split. Alakazam 0.7024 and Mega Lopunny 0.6332: no baseline recorded for those decks | [team_results.md](../results/team_results.md); baseline: [train_ss.py](../imitation/training/train_ss.py), docstring |
| Rule/search baseline, seat-balanced battery | 198/400 = 49.50% [44.63, 54.38]; first seat 116/200, second seat 82/200 (17 points) | [analysis_output.txt](../results/analysis_output.txt) |
| Gumbel candidate vs same-deck baseline | 204/400 = 51.00% [46.11, 55.87]: no evidence either way | [analysis_output.txt](../results/analysis_output.txt) |
| Zero-skill Brier reference | a class-frequency constant predictor scores Brier 0.487022 on the 17,392 validation rows (185 games; no draws), so a "Brier below 0.5" threshold would pass a model with no skill | [analysis_output.txt](../results/analysis_output.txt) |
| Value head, Brier | 0.5771, worse than that zero-skill constant. See "The value head" below for how comparable the two numbers are | [main_v7.py](../imitation/serving/main_v7.py), module docstring |
| Official-engine speedup, macOS arm64 | 1.497701x over the preceding implementation; 50 paired games matched all outcomes and 9,844 moves | [analysis_output.txt](../results/analysis_output.txt) |
| Self-play instrumentation gate | 32/32 games, 5,513 decision records, 0 illegal actions: mechanics only, no learning claim | [analysis_output.txt](../results/analysis_output.txt) |
| PPO vs its frozen parent (never submitted) | 0.8104 [0.756, 0.855], score 194.5/240; best of three arms against the same opponent, Bonferroni-adjusted (3 arms) [0.743, 0.863]. Control, the parent against itself, 0.4875 [0.419, 0.556], 97.5/200 | [ppo_RESULTS.md](../results/ppo_RESULTS.md); counts: `python3 figures/ppo_counts.py` |
| PPO vs an opponent never trained against | 0.7167 [0.663, 0.765] for PPO, 0.5333 [0.409, 0.654] for the champion (the intervals allow 32/60 for the champion; the PPO count is ambiguous, 210/293 or 215/300) | [ppo_RESULTS.md](../results/ppo_RESULTS.md), gate 3 |
| GPU prototype, M2 Max, Metal | 110.14M env-steps/s at batch 262,144 for a batched Rust/GPU prototype of a simplified game-loop slice, verified against its own CPU reference; not rule-complete, not parity-tested against the official engine, never used for training, not published | [engine/README.md](../engine/README.md) |
| This repository's tests | 115 ppo (pytest), 33 imitation numeric, 19 imitation stdlib, 12 demo, 5 report, 6 figures (unittest) | `bash scripts/check.sh` |

## Negative results

In full in [results/negative_results.md](../results/negative_results.md).

| Result | Value | Source |
|---|---|---|
| Spidops deck model | top-1 0.5714 against the 0.5943 baseline (a Marnie-split number the trainer applies to every run); failed, deck dropped. The imitated player won 6.2% of its 48 episodes | [negative_results.md](../results/negative_results.md), [team_results.md](../results/team_results.md) |
| Older distillation line, fidelity check | top-1 agreement 0.376667 against a 0.55 target; best after later phases 0.496667, never passed | [negative_results.md](../results/negative_results.md) |
| ISMCTS vs same-deck baseline | 111/400 = 27.75% [23.59, 32.33], rejected | [analysis_output.txt](../results/analysis_output.txt) |
| ISMCTS cross-deck battery | 108/400 = 27.00% [22.88, 31.55], rejected (its opponents differ from the 198/400 row's, so the rows are not pooled) | [analysis_output.txt](../results/analysis_output.txt) |

## Serving-package scores, cause not recorded

One checkpoint scored 851.5 (package `FIXED-312`, serving file v6), 305.9 (`CHAMP-v7`) and
133.1 (`CHAMP-v7max`) on the ladder ([ppo/docs/BASELINE.md](../ppo/docs/BASELINE.md),
section 2). The weights were the same; the serving packages differed. What the v7 and
v7max packages changed, and on which dates they were scored, was not recorded, and
`main_v7.py` with every serving flag off computes the same forward as v6. So this is a
record of "same weights, different package, very different score", not a measured
serving-path effect. Ladder drift is smaller than the spread: the identical `FIXED-312`
package scored 800.5 and 851.5 two days apart (same file, section 1), but that is one
repeat, not a noise estimate.

The serving-forward defects the team recorded with a diagnosed cause are these:

- two submissions trained with padding masks and relations, and served by a forward that
  attended over padding and dropped the relation bias, scored 256.7 and 183.1 against the
  800.5 champion, while a parity check that collated torch the same wrong way read green
  ([gate_compute_parity.py](../imitation/gates/gate_compute_parity.py), docstring). The
  link from defect to score is the team's diagnosis, not a controlled measurement: the
  champion was itself served by the same unmasked v6 forward. The team's ledger records
  served-vs-trained argmax agreement of 0.8200 for those arms and 0.8736 for the champion.
  The repository's packaging notes record a third figure, 0.7926, for the v6 forward
  against "its own checkpoint" in a packaging run
  ([package_arms.sh](../imitation/serving/package_arms.sh)); which checkpoint that was is
  not recorded, so the three are not one series;
- a second build path shipped a fixed defect again 19 hours later, scoring 260.9 and 156.9
  ([imitation/README.md](../imitation/README.md), "Packaging");
- a sweep of 32 packages found 3 correct, 9 importing with all card tags silently zero, and
  20 raising into a random-move fallback (same section).

## The value head

The team switched the value head off because it "measured Brier 0.5771 against a 0.6667
uniform reference" ([main_v7.py](../imitation/serving/main_v7.py), docstring, as originally
written). 0.6667 is the reference for three equally likely outcomes, but this data has no
draws, and the right zero-skill reference is the class-frequency constant: 0.487022 on the
185-game validation split of the same 926-game Sixth Sense corpus
([results.csv](../results/results.csv), recomputed by `scripts/demo.sh`). Against that
reference the value head did **worse than a predictor with no skill** (0.5771 > 0.487022).

How comparable the two numbers are: the team's records place the 0.5771 on the held-out
split of the 926-game corpus (run `ss-tf-ptr-001`, recorded 2026-08-07) and the 0.487022 on
that corpus's 185-game, 17,392-row validation split; the team's later win-probability notes
make the same comparison and call 0.6667 "the wrong" reference. Those records are in the
team's private repository and are not included here, and they do not state that the two
evaluations used identical rows. So the comparison is between two numbers on the same
corpus and split definition, not a paired measurement.

The aggregate also hides where the signal is. The team's later architecture notes call
the "noise" verdict a mis-diagnosis: a later value head (a different checkpoint from the
0.5771 one) had aggregate Brier 0.5282 against the 0.4870 base rate, which hid a phase
split: worse than the base rate for the first 40% of a game, but Brier 0.331 and 82.8%
accuracy in the final tenth. The notes conclude that late-game-only search would steer on
signal (team records, private). So "worse than no skill" holds in aggregate only, and an
aggregate metric hid where the value head was useful. Nothing in this repository tested a
late-game search.

## Design decisions, long form

**Imitation for the submitted agent; RL only as a fine-tune of it.** The submitted agent is
behaviour cloning on winning games ([REPORT.md](../report/REPORT.md), sections 1 and 3). Its
quality is bounded by its teachers. The Spidops model faithfully copied a player who won
6.2% of 48 episodes. The team also observed that a five-demonstrator Lopunny corpus had a
wider train/validation gap than a mostly single-player Alakazam corpus (+0.284 against
+0.192) and read it as conflicting labels; that comparison is across decks and corpus
sizes, so it is an observation, not a controlled measurement
([team_results.md](../results/team_results.md)). The team's own self-play loop passed its
instrumentation gate but had not demonstrated a learning improvement (REPORT.md, section
6), so RL entered only as PPO on top of an imitation champion ([ppo/](../ppo/README.md)).

**Hand-written NumPy at inference, torch only to train.** Torch is not guaranteed in the
submission sandbox, and both earlier neural submissions returned ERROR
([main_v7.py](../imitation/serving/main_v7.py), module docstring). The price is two
implementations of one forward pass that can drift apart: the recorded agreements of the
v6 forward with trained checkpoints (0.7926, 0.8200 and 0.8736 above) come from different
checkpoints and are not one series, but none is close to 1. The compute-parity and
serve-stamp gates pay for this choice; `demo/model_demo.py` shows, on a SYNTHETIC model, a
badly built parity check reading a false PASS and the correct one catching the wrong mode.

**Search lost; the shipped policy is one greedy forward pass.** ISMCTS won 111/400 and
108/400 and was rejected; Gumbel won 204/400, no evidence either way. The case study puts
the loss on that implementation (beliefs, transition fidelity, rollout evaluation), not on
search in general (REPORT.md, section 5). Two further recorded costs: the value head that
would steer a search scored worse than a constant predictor in aggregate (above), so it is
unused; and identical search-enabled arms agreed on 132/200 outcomes against 200/200
without search (REPORT.md, section 4).

**Why PPO was never submitted.** The records state that no submission slot was used, not why.
What they record as still open at that point: the opponent pool was built but not wired in,
so training was a best response to one frozen opponent; the cross-seed overfitting check was
not run; the offline harness did not pass its own absolute control
([ppo_RESULTS.md](../results/ppo_RESULTS.md), "Still open"). The design also treats whether a
better policy survives packaging as a separate experiment
([ppo/docs/DESIGN.md](../ppo/docs/DESIGN.md), section 6). So 0.8104 is a head-to-head against
the frozen parent, never a ladder result.

**A counted fallback, not a random one.** The team's serving file answered any exception
with a random legal move and no trace. This repository's copy answers with the lowest legal
indices, counts each event and logs it to stderr
([imitation/README.md](../imitation/README.md)), with a test that fails if the fallback is
random or silent again.
