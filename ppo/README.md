# ppo/: self-play PPO with an explicit STOP head

A self-play PPO fine-tune of the 45M-parameter imitation champion (`FIXED-312`, ladder
score 851.5 on 2026-08-16), compared against the frozen original. **It was never
submitted to Kaggle.** No submission slot was used for it, so its win rate below says
nothing about the final standing.

```
ptcg_ppo/action.py     the selection MDP with an explicit STOP        <- the core idea
ptcg_ppo/objective.py  PPO objective, KL diagnostics, gate G-RATIO
ptcg_ppo/advantage.py  potential-based shaping, RLOO, normalisation
ptcg_ppo/policy.py     architecture derivation, the STOP head, calibration
ptcg_ppo/pool.py       the opponent distribution
ptcg_ppo/collate.py    the single path from featuriser rows to a model batch
tests/                 115 tests, each fix paired with a negative control

docs/BASELINE.md       what it starts from, with every number's provenance
docs/DESIGN.md         the method, and why each choice is the one made
docs/PPO_KL_ANCHOR.md  the PPO + supervised-KL spec ("do not re-derive it")
docs/RESEARCH.md       the external evidence, cited and confidence-labelled
../results/ppo_RESULTS.md   the measured results
```

## The finding this is built on

The champion is not allowed to decide how many cards to take. Its serving code computes
`hi = maxCount` and returns the top-`maxCount` options by logit, always: the network chooses
*which*, a constant chooses *how many*. Measured on 6,219 held-out decisions (these three
figures are from the project's own records, quoted in `docs/BASELINE.md`):

```
always maxCount (the champion)   54.93% exact-k
always minCount (v7)             17.43%
a bounds lookup table (v8)       56.33%
```

The same code cannot pass on the 13.99% of decisions where the engine offers it. Because a
supervised loss is fitting noise in the cells that carry the residual (the pilots' own
behaviour is a near-tie there), the fix is a cardinality head trained from outcomes: a STOP
token in the selection MDP, so the number of picks is a decision like any other.

## Results

Win rate against the frozen champion, from [../results/ppo_RESULTS.md](../results/ppo_RESULTS.md):
**0.8104 [0.756, 0.855]** for iteration 57 with a greedy-calibrated STOP head. The game
counts were not written down; the intervals admit only 194.5/240 for this arm and 97.5/200
for the control, the champion against itself, 0.4875 [0.419, 0.556]
(`python3 figures/ppo_counts.py`). Draws and unfinished games score 0.5 in this evaluator.
0.8104 is the best of three PPO arms scored against the same opponent; adjusted for that
choice (Bonferroni over 3) its interval is [0.743, 0.863]. Against an opponent never
trained against (the heuristic agent), PPO scored 0.7167 [0.663, 0.765] and the champion
0.5333 [0.409, 0.654].
The same file lists what is still open: the opponent pool is built and tested but not wired
in, joint-policy correlation across seeds was not run, and the offline replay harness does
not pass its own absolute control, so only its paired delta is quoted.

The gain came from better option ranking, not from the cardinality fix: iteration 30 reached
0.7975 while passing on 0 of 2,035 pass-legal decisions, and the cardinality fix adds roughly
1 to 3 points on top. A negative result worth keeping: left free, PPO destroyed the pass
behaviour (pass rate 2.38% at iteration 0 to 0.05% at iteration 7), because passing rarely
moves the prize differential. The STOP head's bias moved +0.005 over 30 iterations when
closed-form calibration said +15.65 was needed, which is why `--freeze-stop` exists.

## What is and is not reproducible from this repository

- **Tests: yes.** `python -m pytest ppo/tests -q` with `PYTHONPATH=ppo` (needs torch, numpy
  and pytest). Every property the design relies on has a test, and each fix is paired with a
  negative control.
- **The 0.8104: no.** It was produced by drivers (self-play training, the head-to-head
  arena, and the gates run against the real checkpoint) that need the 45M checkpoint, the
  team's rollout tooling and the official engine. None of those is included, so the number
  is documented, not re-runnable here.
- The gate status recorded on 2026-08-17 (G-ARCH, G-WIDTH, G-TIE, G-STOP, all PASS on the
  real checkpoint) is likewise a record, not something this repository can re-run.

## Traps this hit, recorded so the next person does not

1. `model_ss.py` hardcodes `NUM_W = 269` under a comment saying it must equal the
   featurizer's width: a comment, not a check. The champion is 160 wide, so building without
   an explicit width silently builds a different network.
2. The champion carries `card_bow` and `word.weight`, and `self.word` only exists when
   `card_bow` is passed to the constructor. Omit it and the card-semantics path disappears
   while most tensors still load.
3. The champion's own training corpus no longer existed, so the board states used to gate it
   were synthetic, run through the real collate and the real model. No behavioural claim
   rests on them.

## What this design does not claim

- Not that the ladder would move. The same weights scored 851.5, 305.9 and 133.1 in three
  serving packages, a 718-point spread of unrecorded cause; the experiment holds the
  serving path constant and varies only the weights.
- Not that the cardinality head helps: it claims the champion cannot express cardinality, that
  the corpus cannot teach it, and that outcomes can. The head-to-head decides whether that
  is worth anything.
- The 35/30/25/10 opponent mix is transposed from AlphaStar's main-agent proportions onto a
  Restricted-Nash framing. It is inferred, not measured.
