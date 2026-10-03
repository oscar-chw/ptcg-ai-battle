# Results — self-play PPO on FIXED-312

The numbers below were recorded by the PPO project in August 2026; each instrument
was checked against a control before its output was read, and unproven points are
marked. The head-to-head intervals are 95% Wilson intervals on a score in which a
draw or unfinished game counts 0.5; the game counts were not written down, and
`figures/ppo_counts.py` recovers them from the intervals (240 games for 0.8104, 200 for
the other rows).

---

## Headline

**Self-play PPO improved FIXED-312, and the improvement survives every gate that
distinguishes real strength from beating its own parent.**

```
                                        vs frozen champion        pass rate
champion (baseline, main_v6 rule)            ~0.500                 0.0000
PPO iter-030, no cardinality fix              0.7975 [.736,.847]    0.0000
iter-057 + sampling-calibrated STOP           0.7675 [.704,.821]    0.1565
iter-057 + GREEDY-calibrated STOP             0.8104 [.756,.855]    0.0509   <- best
```

Control that makes those readable: **champion vs itself 0.4875 [0.419, 0.556]** —
straddles 0.5, so no seat bias, no harness artefact.

---

## The five gates

| # | gate | result |
|---|---|---|
| 1 | beats frozen parent, CI lower bound > 0.5 | **PASS** 0.8104 [0.7561, 0.8550] |
| 2 | no regression vs own history (**minimum**) | **PASS** min 0.5625; monotone 0.775 → 0.667 → 0.5625 |
| 3 | beats an opponent **never trained against** | **PASS** 0.7167 [0.663, 0.765] vs champion 0.5333 [0.409, 0.654] — disjoint |
| 4 | 0 exceptions, 0 illegal actions | **PASS** over 900+ games |
| 5 | move agreement not collapsed (guardrail) | **PASS** 0.5690 vs 0.5579, McNemar p = 1.4e-03 |

Gate 3 is the one that matters most and was the last to land. The opponent is the
project's `GenericHeuristicAgent`, deliberately excluded from the training mix so
it stays a measurement rather than a memorisation target.

---

## Offline, out-of-time, paired

Replays harvested **2026-08-14**; the champion checkpoint is **2026-08-13**. The
data did not exist in its training corpus — a genuine temporal holdout for both
models.

```
same deck (exact 60-card multiset), 150 replays, 10,646 decisions
                     CHAMPION       PPO      delta
top-1 headline         0.5579     0.5690   +0.0111
top-1 contested        0.4590     0.4702   +0.0113
exact set match        0.5502     0.5606   +0.0104

McNemar: PPO-only 1030 · champion-only 889 · discordant 1919
         chi2 10.214   p = 1.4e-03   SIGNIFICANT
PPO beats the champion on 8 of the top 10 pilots by decision count
(same 8/10 on the winners-only subset)
```

Paired on identical decisions, so common-mode harness error cancels in the delta
even though the absolute level is still suspect.

**Why the level is 0.55 and not 0.79, and why that is mostly not a defect:** the
0.79 figure is agreement with the **one** 1142-rated pilot the model was cloned
from. This set is a mixture of 43 pilots, and per-pilot agreement spans
**0.4500 – 0.6916 (range 0.2416)**. Agreeing with a mixture is a different and
harder task than agreeing with your own teacher. Not fully cleared — the best
single pilot still does not reach 0.79 — so the harness stays labelled
imperfect and only the paired delta is quoted as evidence.

---

## What actually produced the gain

**Not the cardinality fix.** PPO iter-030 reached 0.7975 while passing on
**0 of 2,035** pass-legal decisions. The large gain is better option *ranking*.
The cardinality fix adds roughly 1–3 points on top.

**The STOP head could not learn from reward, and here is the proof.** Over 30
iterations `stop_bias` moved **+0.005**; closed-form calibration showed the
required move was **+15.65**. It was starved, not slow: `P(STOP) ≈ 0.004` means
STOP is almost never sampled, so almost no credit reaches it and the gradient
flips sign and cancels.

**Worse — left free, PPO actively destroys the behaviour:**

```
iter 0:  PASS 2.38%    max 96.7%
iter 6:  PASS 0.29%    max 99.7%
iter 7:  PASS 0.05%    max 99.9%
```

Passing rarely moves the prize differential, so under a shaped reward it reads as
neutral-to-bad against acting. Since the calibrated 5.09% pass rate *beats* both
never-passing (0.7975) and over-passing (0.7675), RL's gradient on that head
points the wrong way. Hence `--freeze-stop`: calibrate the cardinality, lock it,
and spend every gradient on ranking.

**Calibrate the GREEDY rate, not the sampling rate.** Deployment decodes greedily,
and greedy picks STOP whenever its logit merely wins the argmax — a looser
condition than `P(STOP) = p`. Calibrating the sampling rate to 4.2% produced a
greedy rate of **15.65%**, ~4× too often, and cost win rate (0.7675 vs 0.8104).

---

## Bugs the gates caught, in order

Each was found by refusing to read past a failing control.

1. **The 2026-08-08 run was broken before its first gradient step** —
   `approx_kl 7.486`, `ratio_max 125.2` at epoch 0 minibatch 0, where the ratio
   must be exactly 1. A rollout/trainer seam, not a tuning problem. Fixed
   structurally: one collate, one scorer, both sides. Ours reads **4e-06**.
2. **Temperature seam.** `sample_selection` scales logits by temperature; the
   replay did not. Arm B died at `8.4e-01`. Arm A survived only because
   temperature 1.0 makes it a no-op.
3. **`/dev/shm` filename collision** — one worker taking two jobs silently
   clobbered a whole slice of episodes.
4. **Empty deck counter** into `build_tokens`, zeroing a feature family.
5. **`decks_from` searched a key that does not exist**, skipping all 200 replays.
   Decks live at `steps[0][0]["visualize"][0]["action"]`.
6. **Replay off-by-one.** The action at step *t* answers the observation at step
   ***t−1***. Proved by index legality; `out_of_range` went 27,545 → **0**.

---

## Still open

- **The opponent pool is built and tested but not wired.** Training is still a
  best response to one frozen opponent. Gate 3 says that has not hurt us yet;
  wiring it is the main remaining robustness work.
- **Joint-Policy Correlation** across independent seeds — the canonical
  overfitting measurement — not run.
- **The offline harness does not pass its own absolute control**, so only its
  paired delta is quoted.
- **Nothing has been submitted to Kaggle.** No submission slot was used.
