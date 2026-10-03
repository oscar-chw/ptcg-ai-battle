# Baseline — what we are starting from

## Provenance of every number in this document

Read this first. The distinction below is the difference between evidence and
repetition, and this project has been burned by the second.

**MEASURED directly on 2026-08-16, with the command or file named — the primary evidence:**

| claim | how |
|---|---|
| `FIXED-312` = 851.5, top of 46 scored submissions | `kaggle competitions submissions` |
| `main.py:242` `lo = max(1, minCount)`; `:268` `random.sample` | read the shipped file |
| d512 · 12L · 8 heads · FFN 1365 · numeric 160 · 45,435,348 params | read the tensors |
| `card_bow (1270, 1651)`, value 2-class, aux `[4,2,2,2,2,4,18,2,4,2,2,3]` | read the tensors |
| `approx_kl 7.486`, `clipfrac 0.578`, `ratio_max 125.2` at epoch 0 mb 0 | read `metrics.jsonl` |
| 679.47 decisions/s, 1024/1024 episodes, 0 illegal | read `manifest.json` |
| no 160-wide corpus exists on the training host | scanned every `ss_*` corpus |
| **declining legal on 15.23% of decisions** (2,015 of 13,232) | **60 games through the engine** |
| **86% of those are the `(0,1)` cell** — act or don't, not "how many" | same run |
| 107 tests, 4 gates green on the real checkpoint | ran them |

**QUOTED from the training repository's own recorded measurements — not re-derived here:**

- `54.93% / 17.43% / 56.33%` exact-k for maxCount / minCount / the v8 table
- `145,834 decisions = 13.99%` with `minCount == 0`, and pilots passing `4.20%`
  of offers *(note: `CODING_PLAN.md` reports `20.23%` for what sounds like the
  same statistic on a different corpus — the two have not been reconciled, and
  neither has been verified here)*
- the `(0,2)…(1,5)` pilot-behaviour table
- "ALL 840 pass rows lack an END option"

These come with the command that produced them cited in the source, which is
better than most claims in this tree, but they are still someone else's
measurement. The engine measurement above is the only independent check, and it
agrees on the order of magnitude (15.23% vs 13.99%) while disagreeing on the
emphasis.

**INFERRED — reasoning, not measurement, and labelled as such where used:**

- that the `collate` / `collate_fast` split is *the* cause of the epoch-0 KL
  blowup. Strongly supported (the rollout backend calls `collate`, the trainer
  calls `collate_fast`, and they differ on exactly the mask and truncation flags)
  but **the experiment that isolates it has not been run.**
- the 35/30/25/10 opponent mix.
- `f_var` over `f_hard`.

**PROVEN — mathematics, not measurement:**

Softmax cross-entropy is invariant to a uniform shift of the logits. For target
`j`, `L = -z_j + log Σ_k exp(z_k)`; substituting `z_k + c` gives
`-(z_j + c) + c + log Σ_k exp(z_k) = L` exactly. So training determines logit
*differences* and never their level — there is no absolute scale on which "not
worth taking" could be expressed, and no fixed threshold rule is well-defined.
`z_i > z_STOP` **is** well-defined under that invariance, because both sides
shift together. That is the formal reason a STOP token is the right fix rather
than a patch.

---

Every number below was read off the tree or the Kaggle API on 2026-08-16, not recalled.
Where something is unmeasured, it says so.

---

## 1. The model we are improving

**`FIXED-312` — public score 851.5, the highest this project has ever scored.**

```
kaggle submission   "FIXED-312 RESTORE"  851.5   2026-08-16 01:45  COMPLETE
same package        "FIXED-312"          800.5   2026-08-14 06:57  COMPLETE
checkpoint          runs/D2-3121746f2b28/checkpoints/step-00004647.pt
package             submissions/FIXED-312/        (on the training host)
params              45,435,348   ·  202 tensors  ·  80.4 MB weights.npz
```

### Architecture, derived from the checkpoint tensors

Not from `--help` defaults, not from a README, not from memory. From the shapes:

```
num_proj.weight       (512, 160)   ->  d_model 512 · NUMERIC_WIDTH 160
card.weight          (1270, 512)   ->  1270-entry card vocabulary
blocks.*.attn.qkv    (1536, 512)   ->  qkv fused, 12 blocks
blocks.*.attn.k_norm      (64,)    ->  head_dim 64  ->  8 heads
blocks.0.attn.rel_bias    (8, 8)   ->  8 typed relations x 8 heads
```

**The featurizer is 160 columns wide.** The workspace's current `tools/featurize.py`
emits **269**. Feeding 269 columns into a `(512, 160)` projection is the exact defect
class that once scored 355.7 with every offline gate green. The 160-wide featurizer is
preserved at `.gate/feat160/featurize.py` and `tools/featurize_rv160.py`.

> **Gate G-WIDTH:** the rollout refuses to start unless
> `featurizer.NUMERIC_WIDTH == ckpt["num_proj.weight"].shape[1]`.

`weights.npz` carries **no `serve.*` stamps** — it predates stamping. So the serving
mode cannot be read back from the package and must be asserted, not defaulted.

---

## 2. The two defects in the live agent — confirmed, both present

The champion package ships **`main_v6.py`**. Both defects observed in play are in it.

### 2a. It cannot pass — "plays every single possible thing"

```python
submissions/FIXED-312/main.py:242
    lo = max(1, sd.get("minCount") or 0)      # forces at least one selection
```

The engine signals "you may decline" with `minCount == 0`. Measured on the corpus:

```
minCount == 0 offered      145,834 decisions   13.99%
pass TAKEN by strong humans   20.23% of offers   (UNRECONCILED: 4.20% elsewhere; see top)
minCount absent from a prompt  0 of 22,169     (it is never missing)
```

So on ~1 decision in 7 the agent has **no representation of the legal option to
decline**, which strong players take somewhere between 4.20% and 20.23% of the time:
the two recorded rates have not been reconciled. Fixed in `main_v8.py`, which the champion
package predates.

### 2b. It plays random moves on any exception — silently

```python
submissions/FIXED-312/main.py:268
    return random.sample(range(n_opt), min(hi, n_opt))
```

One exception anywhere in featurisation or inference and the agent plays uniformly at
random *for the rest of the game*, while every offline metric stays green. This is the
defect class recorded as "20/32 packages played RANDOM behind a green gate".
`main_v8.py` replaces it with the lowest legal index and **counts** the event to stderr.

### Why this does not change the PPO plan

It is tempting to fix serving first. The ladder record argues for caution:

```
FIXED-312   (v6 serving)   851.5
CHAMP-v7    same weights, v7 serving    305.9
CHAMP-v7max same weights, v7max serving 133.1
```

Provenance of the two lower scores: they were copied from the project's ladder notes
and are in neither class at the top of this document. The submission dates and what
exactly the v7 and v7max packages changed (flags, featurizer, count rule) were not
recorded, and `main_v7.py` with every serving flag off computes the same forward as
v6, so the cause of the drop is **not recorded**. Read them as "same weights, different
serving package", not as a measured serving-path effect.

Same weights, different packages, a 718-point spread of unrecorded cause. Because a
packaging change alone can move the ladder that far, the PPO experiment holds the
serving path *constant* and varies only the weights, or its result would be unreadable.
The serving fix is a separate, later, one-variable experiment.

---

## 3. The prior PPO attempt and why it collapsed

Two runs existed on the training host, both 2026-08-08:
`runs/ppo-rloo-20260808-102714` (plumbing proof) and `runs/ppo-rloo-researched`.

The rollout half worked. Its manifest:

```
episodes_completed        1024 / 1024        episodes_failed 0
decisions                 172,479            illegal_actions 0
decisions_per_second      679.47             corrupted_decisions 0
logp_recompute_mismatches 0                  max_logp_recompute_delta 0.0
```

The trainer half diverged. Final minibatches of `ppo-rloo-researched`:

```
approx_kl   6.05 - 6.70      (healthy PPO: 0.01 - 0.03)
clipfrac    0.51 - 0.56      (healthy: 0.1 - 0.2)
ratio_max   32.8             ( = e^3.49 )
entropy     0.51 -> 0.26     collapsing
minibatch index reached 325+ within epoch 0
```

### Diagnosis from reading `tools/train_rloo.py`

Four defects, ranked by suspected contribution. Ranking is mine and is **inferred**;
the fixes are individually justified regardless of the ranking.

1. **No target-KL abort, and 326 sequential gradient steps on one frozen rollout.**
   `run()` walks every minibatch of the batch with no KL check. Reference
   implementations (Spinning Up, CleanRL, SB3) break out of the update when
   `approx_kl` exceeds a target. Clipping does **not** bound KL: it zeroes the
   gradient for the clipped samples while the other ~45% keep moving the policy, so
   drift accumulates monotonically across minibatches. `clipfrac 0.55` is that
   process, observed.

2. **The ratio is the JOINT probability of a multi-select action.**
   `objective()` takes one `logp` per decision, and the rollout defines it as the joint
   log-probability of *k* options drawn without replacement. So
   `ratio = exp(sum of k log-ratio terms)` — per-term drift compounds geometrically in
   *k*. `ratio_max 32.8` is what a modest per-term drift looks like after 8 selections.
   The clip range 0.2 was chosen for single-action ratios and does not mean the same
   thing here.

3. **Advantages are never normalised.** `_assign_rloo` writes a leave-one-out
   advantage per *episode* and stamps it on all ~78 of that episode's decisions
   unchanged. Reward is terminal win/loss, so the advantage is near-binary, of
   magnitude ~1-2, identical across every decision of an episode, and perfectly
   correlated within an episode.

4. **The KL-to-reference penalty is itself exponential in the joint delta.**
   `ref_kl = (ref_delta.exp() - ref_delta - 1).mean()` with `ref_delta` a *joint*
   log-probability difference. When the policy drifts, this term does not gently
   restrain — it explodes, and its gradient dominates.

The rollout plumbing is reusable. **The trainer is what gets rewritten.**

---

## 4. Compute available

Measured on 2026-08-16:

```
host  4x RTX PRO 6000 Blackwell   97,887 MiB each   0% util   4 MiB used   IDLE
      192 cores · 754 GB RAM · load 2.01
```

Throughput implied by the prior rollout manifest: **679 decisions/s → ~4.0 episodes/s
→ ~14,500 episodes/hour**, with a neural policy on both seats. A 1,024-episode PPO
iteration costs ~4 minutes of rollout. This is not a compute-bound problem.

Engine floor for reference (heuristic vs heuristic, no network):
`0.0295 s/game` optimised, `0.1273 s/game` on the round-2 harness.

---

## 5. Evaluation instruments, and how much to trust each

| instrument | trust | evidence |
|---|---|---|
| head-to-head vs frozen parent, same deck, same serving path | **primary** | isolates the one variable we change |
| `arena_checkpoints.py --b heuristic`, 200+ games | **floor gate** | standing project rule; `exceptions` and `illegal_actions` must be 0 |
| offline top-1 / contested top-1 | **do not gate on it** | has inverted against the ladder 4 times |
| paired CRN arena between unrelated agents | **weak** | two runs returned 0.500 and 0.505 on pairs the ladder separates by hundreds |
| the Kaggle ladder itself | ground truth, but | seeding noise moves identical code by ~100 ranks; 5 submissions/day; a resubmit evicts the prior entry before it matures |

Head-to-head at *n* games has standard error ≈ `0.5/sqrt(n)`: **2,000 games → 1.1pp**.
At 4 episodes/s that is ~8 minutes. That is affordable enough to gate every iteration.
