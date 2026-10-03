# Design — self-play PPO for FIXED-312

Read `BASELINE.md` first; it establishes the facts this design reacts to.
`RESEARCH.md` holds the external evidence with citations.

---

## 0. What we are actually trying to fix

The champion scores 851.5 with a policy that **is not allowed to decide how many
cards to take**. `main.py:242-268` computes `hi = maxCount` and returns
`order[:hi]` — the top-`maxCount` options by logit, always. The network chooses
*which*; a constant chooses *how many*.

Measured on 6,219 held-out decisions where `minCount < maxCount`:

```
always maxCount  (the champion)   54.93% exact-k
always minCount  (v7)             17.43% exact-k
a bounds lookup table (v8)        56.33% exact-k
```

and the project's own note on the residual:

> the count plainly depends on the board rather than on the bounds — **a
> cardinality head is the real answer and this table is not a substitute for one**

Nobody built the head. That is this project.

**Why RL and not more imitation.** In exactly the cells that carry the residual,
the pilots' own behaviour is a near-tie — `(0,3)` splits 35.8 / 34.5 / 25.7 across
k=1/3/2, and `(0,4)` splits 33.0 / 27.8 / 27.4. Imitation fits noise there, and
the project measured that it does: the per-cell training modes *lost* to the flat
cap on held-out episodes. A near-tie in the label distribution is not a signal a
supervised loss can extract. **But taking one card too many has a real cost that
shows up in whether you win.** Cardinality is the textbook case for learning from
outcomes rather than from labels, and it is the single thing self-play can teach
this model that its corpus cannot.

It is also the defect the user reports seeing in play: *"sometimes it just doesn't
know when to stop and plays every single possible thing."* That is
`order[:maxCount]`, exactly.

---

## 1. The action space — sequential selection with an explicit STOP

Today, one decision = one atomic draw of `k = min(maxCount, n_options)` options,
scored as a joint log-probability. `selfplay_rollout.py:821` hard-wires the same
constant, so **PPO on the existing rollout would reinforce the defect**, not fix it.

We replace the atomic draw with a sequential MDP over sub-steps:

```
state:  (legal options remaining, chosen so far)
step:   score {remaining options} + {STOP}, mask illegal, sample one
STOP is offered iff  len(chosen) >= minCount
the sub-episode ends at STOP or when len(chosen) == maxCount
```

Consequences, in order of importance:

1. **`k = 0` becomes reachable.** With `minCount == 0`, STOP is available at
   sub-step 0. That is the pass the engine offers on 13.99% of decisions and that
   the champion is structurally incapable of taking.
2. **Cardinality is learned.** The STOP logit is a per-board prediction, which is
   what the residual demands.
3. **The importance ratio stops compounding.** The old joint ratio is
   `exp(Σ over k terms)`; per-term drift multiplies, which is how `ratio_max`
   reached 32.8 and then 373. Each sub-step is now its own categorical with its
   own ratio, clipped independently — the clip range 0.2 means what PPO intends
   it to mean.
4. **Masking is at the logits**, replacing invalid entries with a large negative
   before the softmax. Huang & Ontañón show this zeroes the gradient at the
   invalid logits and remains a valid policy gradient; masking after the softmax
   does not.

### Getting a STOP logit out of a checkpoint that has none

`self.policy = Sequential(Linear(2d, d), SiLU, Linear(d, 1))` scores
`concat(option_token, pooled_state)`. We add **one learned vector**
`stop_token ∈ R^d`, scored through the *same* policy MLP as
`concat(stop_token, pooled_state)`. Every other tensor loads unchanged.

`stop_token` is initialised so that `P(STOP) ≈ 0` at every sub-step before
`maxCount`. **This makes the extended policy behaviourally identical to the
parent at initialisation** — same moves, same games, same result. That is the
experimental design: the comparison against the frozen original starts from a
true tie, so every point of divergence is attributable to training rather than to
the surgery.

> **Gate G-TIE:** at step 0, the extended policy and the frozen parent must play
> N seeded games with **identical action sequences**. A single divergence fails.

---

## 2. The trainer — what the old one got wrong

The old run's own metrics, at **minibatch 0 of epoch 0, before any gradient step**:

```
approx_kl 7.486    clipfrac 0.578    ratio_max 125.2
max_old_logp_delta 9.2e-07
```

The canonical PPO assertion is that the first minibatch of the first epoch has
`ratio ≡ 1`, hence `approx_kl = 0` and `clipfrac = 0`. It was 7.49. **The run was
never a tuning failure; the ratio never measured policy drift at all.**

`max_old_logp_delta 9.2e-07` is the trap. It compares the *stored* logp against a
recomputation from the *stored* logits — it validates logits→logp arithmetic and
is blind to model→logits. `model.eval()` is set and the checkpoint is the same
one, so dropout is excluded and the remaining explanation is that the observation
re-collated at training time is not the observation the rollout scored (the mask
flags and the option-truncation convention are the two candidates, and
`collate()` vs `collate_fast()` differ on exactly the mask flags).

**The gate that would have caught it in one second:**

> **Gate G-RATIO:** on the first minibatch of the first epoch,
> `max |ratio − 1| < 1e-5`. Otherwise refuse to train, and print the worst 10
> decisions with their stored and recomputed logits side by side.

This is the highest-yield line in the whole project and it costs nothing.

### The rest of the trainer changes

| change | why |
|---|---|
| **per-minibatch target-KL abort** at `1.5 × target_kl`, `target_kl = 0.02` | Clipping does not bound KL — it removes the *incentive* to diverge, not the motion. Early stopping is the only actual constraint. SB3 aborts per minibatch; CleanRL only per epoch, which at KL 6 is far too coarse. Both ship it **off** by default, which is how this is missed. |
| **advantage normalisation, but per batch and after shaping** | With a terminal ±1 reward and an RLOO baseline the advantage is two-valued and identical across all ~78 decisions of a game. Normalising *that* maps every sample to ±1 — a maximum-magnitude uniform-sign gradient with no credit assignment. Shaping (below) is what makes normalisation safe. |
| **shuffle at decision granularity, never at game granularity** | A game-homogeneous minibatch has `std ≈ 0`, and `/(std + 1e-8)` then yields advantages of order 1e8. |
| **KL-to-reference computed per sub-step, not on a joint delta** | `ref_kl = exp(Δ) − Δ − 1` with `Δ` a *joint* log-probability difference does not restrain, it detonates. |
| **separate the two KLs and report both** | `kl_weight` penalises `KL(π ‖ π_BC)`; `approx_kl` measures `KL(π ‖ π_old)`. Anchoring to the BC reference places **no bound** on within-update drift. The old config had nothing constraining the quantity it was logging. |
| **report k1 and k3 side by side** | k3 `(r−1) − log r` is always ≥ 0 and outlier-dominated; k1 `−log r` is unbiased and goes negative for half of samples by construction. Disagreement between them localises drift to the tail. |
| **global grad-norm clip 0.5** | was 1.0; the reference implementations use 0.5. |
| **Adam eps 1e-5** | not PyTorch's 1e-8. |

### Hyperparameters, and where each comes from

```
lr              3e-5     0.1x the supervised LR; the PG-fine-tuning band is 0.1-0.3x
clip_eps        0.2      universal across every implementation read
target_kl       0.02     Spinning Up ships 0.01 and fires at 1.5x; ours fires at 0.03
kl_weight       0.02     Ouyang's swept optimum band is 0.01-0.02; adaptive, see below
entropy_weight  1e-3     between the RLHF lineage's 0 and the Atari default 0.01.
                         The BC seed sits at 0.33-0.53 nats, below ln 2 = 0.69 -- it is
                         already near-deterministic, so the anchor does the regularising
                         and entropy only has to arrest collapse.
epochs          2        with a per-minibatch KL abort, not a fixed step count
minibatch       1024 decisions, shuffled across games
grad_clip       0.5
adam_eps        1e-5
```

`kl_weight` is **adaptive**, on the original PPO paper's rule: `β ← β/2` when
`d < d_targ/1.5`, `β ← β·2` when `d > d_targ·1.5`. A fixed β is a nudge with no
feedback: with normalised advantages at |A| ≈ 1, a β of 0.02 is outweighed ~50x
and the policy simply walks away.

---

## 3. Reward — potential-based shaping on the prize differential

Terminal win/loss over ~78 decisions is a single bit of credit spread across the
whole game. The natural intermediate signal is the prize differential, which moves
monotonically toward the win condition:

```
Phi(s) = their_prizes_remaining - my_prizes_remaining
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

Ng, Harada & Russell's theorem: a shaping term of exactly this form leaves the
optimal policy unchanged. That is the whole reason to use this form rather than a
hand-tuned bonus — it densifies credit assignment **without** inventing a new
objective that trades winning for prize-farming.

**The condition that is easy to get wrong:** policy invariance requires
`Phi(terminal) = 0`. If the potential is left at its natural value on the last
transition, the shaping is no longer a telescoping sum, the guarantee is void, and
the agent is rewarded for reaching a *state* rather than for winning.

> **Gate G-SHAPE:** on any complete episode, `sum(F) + Phi(s_0) == 0` to fp
> tolerance, and `sum(shaped return) == sum(terminal return)` per episode.
> A unit test asserts this and a negative control removes the terminal zeroing
> and asserts the gate goes red.

Shaping is behind `--shaping {none,prize}` and the first production run uses
`prize`, with a `none` arm as the control.

---

## 4. The opponent pool

Deployment is a fixed field of frozen submissions, matched at a similar rating —
not self-play. Two results shape the mix:

- **Pure best-response is brittle.** Johanson et al.: a Frequentist Best Response
  beats the opponent it was fitted to, and then *loses to opponents it was not
  fitted to* — including weak ones an equilibrium strategy beats comfortably.
  The Restricted Nash Response is the correct object: put mass `p` on the modelled
  field and `1 − p` on free play. The exploitation/exploitability curve is steep
  near `p = 1`, so a small concession buys a large reduction in worst case.
- **Naive self-play is only ~21 Elo behind the best league method on strength, but
  far more forgetful** — 46% vs 71% minimum win rate against its own history. What
  a pool buys is retention, not peak strength.

Sampling per episode:

```
35%  current self                   generates transitive strength; do not go lower
30%  own frozen checkpoints         PFSP-weighted, f_var(x) = x(1-x), full history
25%  the frozen parent FIXED-312    the thing we must beat, and the field's stand-in
10%  "forgotten" slot               past opponents whose win rate has fallen below
                                    ~50%; reverts to self-play when empty
```

`f_var(x) = x(1−x)` rather than AlphaStar's default `f_hard(x) = (1−x)^p`, on the
reasoning that AlphaStar wanted to be unexploitable by *anyone* while we are
scored on expected result against a **rating-proximate** sample. Train the
matchmaking you are scored under. This is inferred, not cited, and is the first
knob to sweep.

Pool bookkeeping uses OpenAI Five's quality-score rule — `p_i ∝ e^{q_i}`, and on a
win by the current agent `q_i ← q_i − η/(N·p_i)` with `η = 0.01`. It is four lines
and self-balancing.

**The heuristic is deliberately excluded from the training mix.** It is the
standing behavioural gate. An opponent you train against stops measuring
generalisation and starts measuring memorisation.

---

## 5. Evaluation

The primary instrument is **head-to-head against the frozen parent**, same deck,
same serving path, seats alternated, common random numbers. It is the only
measurement that isolates the one variable being changed. Offline top-1 is not a
gate — it has inverted against the ladder four times.

```
G-TIE      step 0 plays the parent's moves exactly            required before training
G-WIDTH    featurizer width == num_proj.shape[1]              refuse otherwise
G-RATIO    |ratio - 1| < 1e-5 at epoch 0 minibatch 0          refuse otherwise
G-SHAPE    shaping telescopes to zero                         refuse otherwise
G-FLOOR    >= 45% vs `--b heuristic` over 200 games,          standing project rule
           exceptions == 0, illegal_actions == 0
G-HEAD     >= 52.5% vs frozen parent over 2,000 games         promotion gate
           (SE 1.1pp, so this is ~2 sigma)
G-STOP     pass rate and exact-k are reported every iteration; a run that drives
           P(STOP) to 0 or 1 everywhere has collapsed, not learned
```

`G-HEAD` at 2,000 games costs ~8 minutes at the measured 4.0 episodes/s. Cheap
enough to gate every iteration.

---

## 6. What this design does not claim

- **It does not claim the ladder will move.** Serving-path choice currently
  dominates the ladder signal — the same checkpoint scored 851.5, 305.9 and 133.1
  on three serving paths. This experiment holds the serving path constant and
  measures the weights. Whether a better policy survives packaging is a separate
  question with its own separate experiment.
- **It does not claim the cardinality head will help.** It claims the champion
  cannot express cardinality, that the corpus cannot teach it, and that outcomes
  can. `G-HEAD` decides whether that is worth anything.
- **The 35/30/25/10 pool split is inferred**, transposed from AlphaStar's
  main-agent proportions onto an RNR framing. No published work measures the
  optimum for a ~78-decision card game.
- **`f_var` over `f_hard` is reasoning, not a citation.**
