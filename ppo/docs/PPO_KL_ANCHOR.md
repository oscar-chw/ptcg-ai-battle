# The PPO + supervised-KL anchor

**This is the reference spec for how PPO with a supervised KL is done here. Do not
re-derive it. Do not accept a library default in place of a line of it.** Every
row below is either read from a primary source or measured on this project, and
where a value exists only because getting it wrong cost us a run, it says so.

---

## The objective

```
L = −E[ min( r·A , clip(r, 1±ε)·A ) ]  −  c_H·H(π)  +  β·KL(π ‖ π_ref)

r = π(a|s) / π_old(a|s)
```

**Three quantities are routinely conflated. Keep them apart or nothing works:**

| symbol | what it is | what it governs | how it is enforced |
|---|---|---|---|
| `π_old` | the **rollout** policy that generated this batch | step size within one update | clip **and** a target-KL abort |
| `π_ref` | the **frozen supervised** policy | forgetting, across the whole run | `β`, adaptive |
| `π` | the policy being trained | — | — |

**Anchoring to `π_ref` places no bound whatsoever on within-update drift.** Our
2026-08-08 run logged the first quantity, penalised the second, and constrained
neither — which is how it reached `approx_kl 7.486` at epoch 0, minibatch 0,
*before a single gradient step*.

---

## The rules

| | value | why |
|---|---|---|
| ratio at epoch 0, minibatch 0 | **exactly 1.0** | canonical assertion. If it is not, it is a seam bug and no amount of tuning will help. The rebuilt pipeline reads `4e-06`; the old run read `125.2`. |
| `target_kl` | 0.01–0.02, abort at **1.5×** | clipping does **not** bound KL. Spinning Up, verbatim: the policy *"can still go farther than the clip_ratio says, but it doesn't help on the objective anymore"* |
| abort granularity | **per minibatch** (SB3) | CleanRL breaks only per *epoch*. The old run reached minibatch 325 inside epoch 0, so an epoch check would never have fired. |
| `β` | **0.01–0.02**, adaptive | Ouyang's sweep: both 0 and 2 perform poorly; the optimum is "around 0.01 and 0.02" |
| `β` controller | `e = clip((KL−target)/target, ±0.2)`, `β ← β(1 + 0.1·e)` | Ziegler's log-space proportional controller. A fixed `β` is a nudge with no feedback: at \|A\|≈1 it is outweighed ~50×. |
| KL estimator | report **k3 and k1** | `k3 = (r−1) − log r` ≥ 0 always, outlier-dominated. `k1 = −log r` goes negative for half of samples *by construction*. Disagreement localises drift to the tail. |
| reference KL | **exact over the full masked categorical** | not the single-sample estimator. See below. |
| advantage | normalise per minibatch, **shuffle by decision** | a game-homogeneous minibatch has `std≈0`, and `/(std+1e-8)` then yields advantages of order 1e8 |
| grad clip / Adam eps | **0.5 / 1e-5** | reference implementations; not PyTorch's 1e-8 |
| masking | at the **logits**, before the softmax | Huang & Ontañón: this zeroes the gradient at invalid entries and stays a valid policy gradient. Renormalising after the softmax does not. |

### The reference KL must be exact, not sampled

The RLHF-lineage shortcut estimates `KL(π‖π_ref)` from the one action that
happened to be sampled. That is unbiased but high-variance, and — the part that
matters — **the anchor then exerts no force on the rest of the distribution**,
which is most of what the supervised policy knows. Our action set is a masked
categorical of at most ~51 options, so the exact sum is cheap:

```
KL(p‖q) = Σ_a p(a)·( log p(a) − log q(a) )      over valid options only
```

Direction is `KL(current ‖ reference)` — mode-seeking, and the direction the
RLHF lineage penalises. Implemented as `objective.exact_kl`.

### One caveat that does NOT apply to us

Gao et al. found KL penalties *"akin to early stopping"* and set the penalty to
zero for their other experiments. That result is about defending against a
**learned, exploitable reward model** — the proxy-vs-gold gap. **Our reward is
engine win/loss: a true reward, not a learnable proxy, and not hackable.** The
finding does not transfer, and the setting where the anchor *is* measured to pay
is exactly ours.

---

## The promotion gate: GAMES WON, never an offline metric

**A checkpoint ships only if it wins games.** Offline top-1 has inverted against
this ladder four times; it is a guardrail, never a criterion.

| # | gate | necessary? |
|---|---|---|
| 1 | beats the frozen parent head-to-head, **CI lower bound > 0.5** | yes |
| 2 | no regression against its own history — **MINIMUM** win rate > 0.5, not the mean | yes |
| 3 | beats an opponent it **never trained against** | yes |
| 4 | 0 exceptions, 0 illegal actions | yes |
| 5 | move agreement has not **collapsed** (guardrail: a fall to ~0.30 means the KL anchor failed) | advisory |

**Why 2 and 3 exist, and why 1 alone is not enough.** Beating the parent 77% is
consistent with genuine improvement AND with a best response fitted to that one
opponent — the two are indistinguishable from gate 1 alone. Johanson's frequentist
best responses beat their targets *harder than anything before them* and then lost
to opponents they were not fitted to. Gate 2 catches cycling; gate 3 catches
opponent-specific overfitting.

**Why 5 is not a gate.** If PPO is working, agreement with the demonstrator should
drift DOWN — the model is deviating in ways that win. Promoting on agreement would
select against the thing being bought. Direction matters more than level: a slow
decline is healthy, a collapse is an alarm, and a RISE means PPO changed nothing.

*Status as measured:* gate 1 **PASS** (0.7675 [0.7042, 0.8207]); gate 2 **PASS**
(min 0.5625, monotone 0.775 → 0.667 → 0.5625); gate 4 **PASS** (600+ games);
gate 5 **PASS** (0.5690 vs champion 0.5579, +0.0111, McNemar p=1.4e-03);
**gate 3 NOT YET RUN — this is the honest weak point.**

## Proving it is not just beating itself

Four tests, weakest to strongest. Run more than one.

1. **Held-out opponents.** The heuristic, plus the three ladder-reconstructed
   decks in `LADDER_CONTROLS.md` (`mega_top`, `alakazam_ladder`,
   `archaludon_ladder`) — real lists from games that beat us.
2. **Round-robin against its own history, reporting the MINIMUM win rate, not
   the mean.** A best response has a great mean and a terrible minimum.
   AlphaStar's retention figures: naive self-play **46%** minimum against past
   selves versus **71%** for league play.
   *Our measurement:* `0.775 → 0.667 → 0.5625` against iterations 0/10/20 —
   monotone, minimum **0.5625 > 0.5**, no regression against any past self.
3. **Joint-Policy Correlation** (Lanctot et al.). Train N seeds independently,
   cross-play them, compute `R⁻ = (D̄ − Ō)/D̄`. Measured **34–72%** reward loss
   in directly competitive environments. The canonical measurement for this
   exact question. *Not yet run.*
4. **Transitivity.** If A beats B and B beats C, does A beat C? Failure means
   cycling, not improving.

---

## Checkpoint selection: validation agreement INVERTED here

The published guidance is that validation **action agreement** predicts real
performance (r = -0.89) while validation **loss** does not (r = -0.04), and that
picking by val loss or by the last checkpoint costs 10-100% of achievable
performance. The first half of that is why this project selects on `top1` rather
than on loss, and that was the right call.

**But measured on our own data, agreement inverted against play.** Arm
`WIN-ed7c14bab680`, three checkpoints, 200 games per pair, same deck both sides:

```
            val top-1   train loss    head-to-head
EARLY  e6     0.6332       0.9935     beats PEAK 65-35   [0.287,0.418] SIGNIFICANT
PEAK   e12    0.6381       0.8269     --
LATE   e30    0.6242       0.4080     ties PEAK 46-54    [0.471,0.608] not separable
```

The **best-validation checkpoint is significantly worse in play than an earlier
one with lower agreement**, and the checkpoint whose training loss more than
halved is indistinguishable from it. So on this data neither offline metric ranks
checkpoints reliably, and `argmax(val_top1)` picks the wrong model.

This is the **fifth** recorded inversion of an offline metric against play in this
project. The standing rule follows: **never ship `argmax(offline)` — run the
arena over several checkpoints and pick by games.** Peak-val, a late low-train-loss
one, and an early control is a sufficient screen; it costs minutes.

Corollary worth keeping: a difference the arena cannot resolve is not a
difference. PEAK vs LATE differ by 0.0139 val and a halved training loss and are
statistically tied at n=200 -- resolving a true 4pp gap needs ~600 games a pair.

## Validating an instrument before believing it

**The instrument gets a control, and the control gets checked first.** This is
the whole lesson of the `approx_kl 7.486` incident, and it recurred immediately
in the offline harness.

- Head-to-head evaluator: control = **champion vs itself → 0.4875 [0.419, 0.556]**.
  Straddles 0.5, so no seat bias. Only then is `0.7975` readable.
- Offline replay harness: control = **champion's recorded 0.79 agreement**.
  It read 0.396, so the harness was quarantined rather than reported.

Bugs found by refusing to read past a failing control, in order:

1. **Empty deck counter** passed to `build_tokens`, zeroing a feature family.
2. **`decks_from` searched for a key that does not exist** — silently skipped all
   200 replays. The decks live at `steps[0][0]["visualize"][0]["action"]`.
3. **Off-by-one in replay alignment.** The action recorded at step *t* answers
   the observation at step ***t−1***. Proved by index legality:
   `t=7 obs n_opt=4` paired with `t=7 act=[5]` is out of range; paired with
   `t=8 act=[2]` it is in range, and `out_of_range` went 27,545 → **0**.

**What the control still says:** the champion reads 0.558 on a *mixture* of 43
pilots, against 0.79 on the single pilot it was cloned from. Per-pilot spread is
**0.4500–0.6916, range 0.2416** — so pilot identity is the dominant term and 0.5x
is largely expected rather than a defect. It is not fully cleared, because the
best pilot does not reach 0.79.

### Paired comparison survives a biased instrument

Both models are scored through the **identical** featurisation on the
**identical** decisions, so common-mode harness error cancels in the *delta* even
while the *level* is wrong. Use **McNemar** on the discordant pairs — agreements
carry no information about which model is better and would dilute the test.

*Measured, 150 replays, both models on data neither had seen:*
```
top-1 headline    champion 0.5579   PPO 0.5690   +0.0111
McNemar           PPO-only 1030 · champion-only 889 · discordant 1919
                  chi2 10.214   p = 1.4e-03   SIGNIFICANT
PPO beats the champion on 8 of the top 10 pilots by decision count
```

---

## Status of this project against the spec

**Correct:** per-sub-step ratios (not a joint over *k* draws, which compounds
geometrically), the G-RATIO gate, per-minibatch abort, frozen reference never in
the optimiser, adaptive `β`, exact full-distribution reference KL, grad-clip 0.5,
Adam eps 1e-5, decision-level shuffle, logit-level masking, potential-based
shaping with the terminal zeroing.

**Still missing:** the opponent pool (`pool.py`, built and tested, not wired) —
so training is still a best response to one frozen opponent.
