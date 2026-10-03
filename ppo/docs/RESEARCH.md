# Research — the external evidence this design rests on

Four independent sweeps, 2026-08-16. Every figure below was read from a primary
source, not from a search snippet. Confidence is labelled per claim.
**Verified** = ≥2 independent origins. **Single-source** = named inline.
**Inferred** = reasoning from evidence, not retrieved. **Unknown** = searched, not found.

---

## 1. Does RL fine-tuning actually beat behaviour cloning?

### AlphaStar Fig. 3E — the ablation this whole design is built around
Single-source but primary: Vinyals et al., Nature 575 (2019).
https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf

| arm | Test Elo |
|---|---|
| no human data (RL from scratch) | **149** |
| supervised (the BC policy) | **936** |
| + human init, RL **without** KL | **1020** |
| + **supervised KL** | **1400** |
| + statistics `z` | **1540** |

**Initialising from the BC policy bought +84 Elo. Anchoring to it bought +380
more.** The anchor is the mechanism, not the initialisation — and RL from scratch
landed **487 Elo below** the BC policy it was meant to improve on.

The KL is continuous, not a one-off init: *"we initialise the policy parameters to
the supervised policy and continually minimise the KL divergence between the
supervised and current policy."* The coefficient is **Unknown** — it lives in a
supplementary file the sweep could not reach.

Caveats carried: one map, one match-up, main agents only, RL capped at 10^10 steps.

### Legends of Code and Magic — the closest published analogue
Verified (read in full): Haluška & Schmid, ALA 2024, https://arxiv.org/abs/2404.16689

A two-player collectible card game, BC→PPO vs PPO-from-scratch, head to head.

```
best BC policy                    42.4%  win rate vs ByteRL
BC -> PPO                         90.4%
PPO from scratch                  81.2%
```

BC init won 5 of 6 deck-pool settings, and the gap *widens* with pool size. Notes
that transfer directly: the value head was **randomly initialised with no shared
weights**; entropy coefficient **0.01**; filtering pass actions out of the BC data
was worth **~+7 points** on its own.

**The number that decides feasibility: pretrained runs crossed 50% in under 100
training episodes and 75% in under 500.** From-scratch needed more than twice as
long. Hundreds of games, not millions — at our measured 4.0 episodes/s that is
minutes, not days. Every five-figure game count in the table below belongs to a
from-scratch or league system.

| system | cost for RL to pass BC | BC init |
|---|---|---|
| **LOCM (BC→PPO)** | **<100 episodes to 50%, <500 to 75%** | yes |
| DouZero | 2 days, 4 GPUs, to pass a 226k-match SL agent | no |
| Suphx | 1.5–2.5M games, 44 GPUs, 2 days | yes |
| AlphaStar ablation arm | 10^10 steps ≈ 6M games (Inferred) | yes |
| OpenAI Five | 770 PFlop/s-days, 10 months | no |

### Where RL fine-tuning made a BC policy worse

- **AlphaStar's no-human-data arm**, above. The clearest negative result found.
- **Pommerman** (https://arxiv.org/pdf/1911.04947): the *default* PPO entropy
  coefficient caused **catastrophic forgetting of imitation-learned skills**; they
  set it to zero. LOCM used 0.01 without trouble, so this is domain-dependent —
  but entropy is the first knob to check when a fine-tune degrades.
- **MINDGAMES DeceptionNet** (https://arxiv.org/abs/2605.29512, §A.2.7), verbatim:
  *"Self-play PPO proved too brittle… Performance consistently degraded from the
  IL baseline."* Caveat: cloned from 300 synthetic demonstrations, and a team
  report rather than a controlled ablation.
- **Suphx reports no SL-vs-RL number at all.** It is routinely cited as evidence
  RL beats BC; its Fig. 8 is unlabelled box plots. **Unknown**, not supportive.

**Absent prior art:** arXiv and OpenAlex are empty of RL or imitation work on
Pokémon TCG or Pokémon TCG Pocket. There is no published baseline to beat.

---

## 2. The KL explosion — mechanism and fix

### The estimators
Verified: Schulman, http://joschu.net/blog/kl-approx.html

```
k1 = -log r            unbiased, NEGATIVE for ~half of samples by construction
k3 = (r - 1) - log r   unbiased, always >= 0, outlier-dominated (contains raw r)
```

Read from source: **CleanRL** logs both and gates on k3; **SB3** uses k3 only;
**Spinning Up** uses k1. A negative k1 next to a positive k3 is normal, not a bug.

### Which implementation ships the abort

| impl | `target_kl` default | trigger | abort granularity |
|---|---|---|---|
| Spinning Up | **0.01 — ON** | `kl > 1.5 × target` | per gradient step |
| CleanRL | `None` — **OFF** | `approx_kl > target` | end of epoch only |
| SB3 | `None` — **OFF** | `> 1.5 × target` | **per minibatch** |

Two of the three ship it off. That is how this failure is missed. We use SB3's
granularity with Spinning Up's philosophy of having it on.

### Clipping does not bound KL
Verified (Spinning Up docs and source agree verbatim): the new policy *"can still
go farther than the clip_ratio says, but it doesn't help on the objective
anymore."* Clipping removes the **incentive** to diverge, not the **motion**.
Mechanisms: shared parameters (zeroing one sample's gradient doesn't freeze its
probability while the rest of the minibatch moves the same weights); stale
ordering (a sample first seen at minibatch *j* was already displaced by 1..*j−1*);
and the clip being one-sided per sample.

Only early stopping is an actual constraint.

### The canonical diagnostic
Single-source but primary: ICLR blog track detail #12,
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ —
`approx_kl` *"generally stays below 0.02, and if approx_kl becomes too high it
usually means the policy is changing too quickly and there is a bug."*
**In the first minibatch of the first epoch the ratio must be exactly 1.0.**

Our old run reported 7.486 there. That is 370× the threshold, before any step.

### Other details that bear on this failure
- **#7 advantage normalisation is per *minibatch***, matching `openai/baselines`.
  With a two-valued terminal advantage, standardising maps every sample to ±1 —
  a maximum-magnitude uniform-sign gradient with no credit assignment left.
  Sub-case: minibatches formed over *games* rather than shuffled over *decisions*
  give `std ≈ 0`, and `/(std + 1e-8)` then yields advantages of order 1e8.
- **#11 global grad-norm clip 0.5**; **#3 Adam eps 1e-5**.
- **Invalid-action masking belongs at the logits.** Verified: Huang & Ontañón,
  https://arxiv.org/abs/2006.14171 — masking at the logits *"makes the gradient
  corresponding to the logits of the invalid action to zero"* and remains a valid
  policy gradient; masking after the softmax does not. It also **scales** as the
  invalid space grows.
- **The two KLs are different quantities.** `kl_weight` penalises `KL(π ‖ π_BC)`;
  `approx_kl` measures `KL(π ‖ π_old)` within one update. Anchoring to the BC
  reference places no bound on within-update drift.

---

## 3. Opponent pools against a fixed field

### Naive self-play is not weak — it is forgetful
Single-source but primary (AlphaStar Fig. 3C/D):

| algorithm | Test Elo | min win rate vs all past selves |
|---|---|---|
| pFSP + SP | **1540** | **71%** |
| SP (naive) | 1519 | **46%** |
| pFSP alone | 1273 | 70% |
| FSP alone | 1143 | 69% |

Naive self-play was only **21 Elo** behind the best method, and beat *pure*
historical play by 246–376 Elo. What a pool buys is **retention**, not peak
strength. This is the argument for keeping ≥35% self-play, not for replacing it.

### PFSP, exactly
Sample frozen opponent `B` from candidates `C` with probability
`f(P[A beats B]) / Σ_C f(P[A beats C])`, where

```
f_hard(x) = (1 - x)^p     no games against opponents already beaten;
                          "a smooth approximation of max-min optimisation"
f_var(x)  = x(1 - x)      preferentially plays opponents around its own level
```

AlphaStar main agents: **35% self-play / 50% PFSP over all past players / 15%
against forgotten players and past main exploiters**, reverting to self-play when
that set is empty. League exploiters freeze in at >70% win rate over the whole
league, with a 25% chance of reset. Exploiters were worth **+284 Elo and ~10× the
Relative Population Performance** (6% → 62%).

OpenAI Five used plain **80% latest self / 20% past**, sampling `p_i ∝ e^{q_i}`
and updating `q_i ← q_i − η/(N·p_i)` with `η = 0.01` **only when the current agent
wins**. Verified across main text, appendix and hyperparameter table.
https://arxiv.org/abs/1912.06680

### Best response is brittle — the reason not to go past ~35% field mass
Single-source but primary: Johanson, Zinkevich & Bowling, NIPS 2007,
https://poker.cs.ualberta.ca/publications/NIPS07-rnash.pdf

A Frequentist Best Response beats the opponent it was fitted to (PsOpti4:
75 → 137 mb/h) and then **loses to opponents it was not fitted to** — including a
weak program that an approximate equilibrium beats comfortably. Verbatim:
*"best response is, in practice, a brittle computation, and can perform poorly
when the model is wrong."*

The correct object is the **Restricted Nash Response**: force the opponent to play
the modelled field with probability `p` and play freely with `1−p`. `p=1` is the
pure best response, `p=0` is Nash, and every `p ∈ (0,1]` is an ε-safe best
response. The exploitation/exploitability curve is **steep near p=1**, so
conceding a little exploitation buys a large reduction in worst case.

Joint-Policy Correlation quantifies the co-training overfit: reward loss of
**34.2 / 62.5 / 71.7%** in directly competitive environments, but 0.3–2.2% in
coordination ones (Lanctot et al., https://arxiv.org/abs/1711.00832). Ours is
directly competitive; assume the bad end.

### Disconfirmation that survived
**League training is not necessary for strength.** OpenAI Five reached superhuman
Dota 2 on plain 80/20 self-play with no league, and DeepMind's own numbers concede
naive SP was 21 Elo off. The league buys unexploitability and costs 12 concurrent
actor-learner setups. We take the cheap part (a frozen-checkpoint pool and one
forgotten slot) and skip the expensive part.

---

## 4. Reward shaping and KL anchoring

### The theorem
Verified: Ng, Harada & Russell, ICML 1999,
https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf

`F(s,a,s') = γΦ(s') − Φ(s)` is **necessary and sufficient** for the optimal policy
to be preserved. Remark 1 extends it to arbitrary policies, so *near*-optimal
policies are preserved too. Two conditions bind us: `Φ` must be bounded, and
**§5 is explicit that using `Φ(s') − Φ(s)` when `γ ≠ 1` voids the theorem** — the
same γ must appear in the shaping as in the returns.

### The terminal-state condition — our live risk, not a footnote
Verified: Grześ, AAMAS 2017, https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf

For a trajectory ending at `s_N`:

```
U_Φ(trajectory) = U(trajectory) + γ^N Φ(s_N) - Φ(s_0)
```

`Φ(s_0)` cannot change the policy — it is action-independent. **`γ^N Φ(s_N)` can**,
because which terminal state you reach depends on earlier actions. The paper's
counterexample inverts the optimal policy for any `γ ≳ 0.101`.

This is precisely our exposure: **`Φ(s) = their_prizes − my_prizes` is maximal at
the winning terminal state.** Left unzeroed, the agent is paid for *reaching a
state* rather than for winning.

The fix is one line — `Φ(s_N) := 0` at every trajectory-terminating state — with
two details that are easy to miss:
- it applies to **step-limit truncations**, not only genuine terminals;
- the same state keeps its normal potential when it is *non*-terminal in a
  different trajectory;
- and zeroing only the **final transition's shaping reward** does **not** work —
  the same imbalance re-forms at the terminal states' predecessors.

In the two-player case this condition is also what protects Nash invariance:
Devlin & Kudenko showed potential shaping preserves the Nash equilibria, and
Grześ corrects them — *"Devlin and Kudenko did not consider this requirement"* —
with a counterexample where a non-zero terminal potential **introduces a new
equilibrium**.

Also verified: state-only `Φ` needs no change to action selection, whereas
state-**action** potentials (Wiewiora's look-ahead advice) only recover the
optimal policy under a *biased greedy* rule `argmax_a [Q(s,a) + Φ(s,a)]`. We use
state-only `Φ` and avoid that entirely.

### KL anchoring — the coefficients, from the sources

**Ouyang et al., InstructGPT, Appendix E.7, verbatim** (the claim was put to the
sweep to verify or refute, and it verified):
> *"Both 0 and 2 for KL reward coefficient result in poor performance. **The
> optimal value is around 0.01 and 0.02.**"*

Default used: **β = 0.02**. https://arxiv.org/abs/2203.02155

**Ziegler et al.'s adaptive controller**, verbatim — a log-space proportional
controller, and better than the original PPO paper's halve/double rule:

```
e_t     = clip( (KL(pi_t, rho) - KL_target) / KL_target,  -0.2, 0.2 )
beta_t+1 = beta_t * (1 + K_beta * e_t)          K_beta = 0.1
```
https://arxiv.org/abs/1909.08593

**Stiennon et al.** used a *fixed* β = 0.05, and their Table 9 shows the mapping is
strongly nonlinear — β 0.35 → 0.05 is a 7× change that produces a **10× change in
realised KL** (1.8 → 19.0 nats). Anchor on measured KL, not on β.
https://arxiv.org/abs/2009.01325

**Ziegler's Table 10 is the failure mode the anchor prevents:** without a KL
penalty, a policy reached 99.97% proxy reward and produced gibberish — *and an
entropy bonus was not a substitute*.

### The disconfirmation, and why it does not apply to us
Verified: Gao, Schulman & Hilton, *Scaling Laws for Reward Model
Overoptimization*, https://arxiv.org/abs/2210.10760, contains a bullet headed
**"KL penalty ineffectiveness"**: the penalty *"does not correspond to a
measurable improvement in the gold RM score–KL_RL frontier"*, and its effect is
*"akin to early stopping"*. They set the KL penalty to 0 for all other experiments.

That result is about defending against a **learned, exploitable reward model** —
the proxy-vs-gold gap. **Our reward is win/loss from the engine: a true reward,
not a learnable proxy, and not hackable.** The setting where the anchor is
measured to pay is exactly ours: AlphaStar's +380 Elo on a ground-truth win/loss
reward, and piKL's Theorem 1, which bounds `D_KL(π ‖ τ) ≤ (1/λ)(R/T + D)` and
Theorem 2, which makes the limit points `max λβ`-approximate Nash equilibria of
the *original* utilities. https://arxiv.org/abs/2112.07544

---

## What none of this establishes

- **Unknown:** any measured optimum for a current/frozen/scripted opponent ratio
  in a ~78-decision card game. The 35/30/25/10 split is transposed, not measured.
- **Unknown:** AlphaStar's actual supervised-KL coefficient.
- **Unknown:** whether entropy/KL should be normalised by `log(n)` when the legal
  set size varies. The mechanism is real; no source prescribes it.
- **Unknown:** whether `f_var` beats `f_hard` for a rating-proximate ladder. That
  is reasoning, and it is the first knob to sweep.
- **No paper measures RL fine-tuning making a BC policy *more exploitable* than
  its BC parent.** Both studies that measure both endpoints show large
  head-to-head gains while *both* endpoints stay highly exploitable.
