# imitation/: set-transformer policy, NumPy serving, and the packaging gates

The imitation-learning code of the team's agents, cleaned and consolidated: the model,
the featurizer, the trainer, the NumPy forward that is actually shipped to the Kaggle
sandbox, and the checks that decide whether a built package computes the function it was
trained to compute. This is the team's code. What this repository added is the cleanup
(host paths, a loader for the reader's own engine, dead command-line tools removed) and the
tests in `tests/` and `tests_torch/`.

The architecture write-up is [../docs/architecture.pdf](../docs/architecture.pdf) (9 pages,
with a "Claims withdrawn" section: read it before proposing changes).

## What runs here, and what does not

| piece | runs from this repository? | needs |
|---|---|---|
| `model_ss.py`, `main_v7._forward`, `export_ss_numpy.py`, `write_provenance.py` | yes, tested | numpy, torch |
| the gates and `package_arms.sh` / `package_and_tar.sh` | yes, tested on a synthetic checkpoint | numpy, torch |
| `strict_get.py`, `deck_tracker.py`, serving-flag logic | yes, tested | standard library only |
| `featurize.py` | no | the official engine (`PTCG_ENGINE_DIR`) and an engine-derived `effect_tags.json` |
| `train_ss.py` | no | a corpus built by tooling that is not included, plus the engine-derived card-text file |
| trained weights | not included | they are not published |

`featurize.py` reads the engine's `cg.api` (enums, card and attack tables) at import time
and ships none of it. It contains no embedded card data: the card tables come from
the reader's own engine install through `PTCG_ENGINE_DIR`. The one data file it also needs,
`.gate/marnie-sixthsense/effect_tags.json` (21 effect tags per card), was generated from the
engine's source, so it cannot be shipped; supply your own at that path. See
[../demo/README.md](../demo/README.md) for how to obtain the engine.

## The model

A **set transformer** over board tokens. The board is a set (bench slots have no meaningful
order), so permutation-equivariance is correct. Structure a bare set would lose is injected
as a learned attention bias over 8 typed edges: same-owner, same-zone, pointer, destination,
active-active, prompt. One forward pass scores every legal option from its own token.

Two sizes appear in this repository, and they should not be confused:

- the default configuration of `SixthSenseNet` (d 384, 12 layers, 12 heads, FFN 1024) has
  **23,782,511 parameters**, asserted by `tests_torch/test_model_ss.py`;
- the checkpoints behind the 45M-parameter agents are d 512, 12 layers, 8 heads, FFN 1365,
  with **45,435,348 parameters** read from the champion's tensors
  ([../ppo/docs/BASELINE.md](../ppo/docs/BASELINE.md)). The submitted `BEST1_fixed` is
  recorded as d 512, 12 layers, 8 heads, FFN 1365, 269 numeric features
  ([../report/REPORT.md](../report/REPORT.md), section 1).

(The original README of this code gave "d 384 ... ~45M params" together, which is
inconsistent; the numbers above are the measured ones.)

**The v2 architecture (`--arch-v2`) is training-only.** FiLM family conditioning,
multi-seed pooling and per-column standardization exist in `model_ss.py`, but
`serving/main_v7.py` implements the v1 forward (single-seed pooling, no FiLM, no
standardization); a v2 checkpoint has no NumPy serving path here. Under v2 the three
changes cannot be separated (`model_ss.py` raises if `arch_v2` and `pool_seeds < 3`, and
ignores `pool_seeds` under v1), so no configuration isolates FiLM from multi-seed pooling.
The v1-versus-v2 result is not decomposable and should not be reported as if it were.

## Packaging: "identical" is four gates, not an opinion

Build only through `serving/package_arms.sh`. A second build path is how a fixed defect
shipped again 19 hours later, scoring 260.9 and 156.9 against a champion's 800.5, with both
submissions asserting the fix was applied. (These ladder scores and the measurements below
are the team's records of what happened on the original tree; they are not re-measurable
here. The tests in this repository check that each gate still fires on a planted defect.)

| gate | file | asks | planted-defect test |
|---|---|---|---|
| seam | `gates/gate_serving_seam.py` | is the served featurizer the trained one, and do the weights' numeric width and the featurizer's agree? | `tests_torch/test_export_and_gates.py::ServingSeamGate` |
| serve stamp | `gates/gate_serve_stamp.py` | does every `serve.*` flag in `weights.npz` equal the flag in the manifest of the run that trained those weights? | `...::ServeStampGate` |
| reachability | `gates/gate_package_reachability.py` | does a copy of the package, in a directory it has never been in, import its own featurizer? | `...::ReachabilityGate` (the gate's own `--self-test`) |
| compute parity | `gates/gate_compute_parity.py` | does the NumPy forward reproduce torch's argmax on identical weights, with the masks and relations the run trained with? | `...::ComputeParityGate`, `test_numpy_parity.py` |

`gates/gate_newarm_4gates.py` re-runs the first three against a built package and reads the
fourth only if its artifact is newer than the package.

Why each exists, from the original records:

- **Reachability.** `featurize.py` resolved a data file through `parents[1]`, which from
  inside a Kaggle extraction points above the package. The import raised, `main.py`'s
  `except Exception` returned a random legal move, and the agent played uniformly at random
  for entire games behind a gate that only asked whether the file existed somewhere. A sweep
  of 32 packages found 3 correct, 9 importing with all card tags silently zero, and 20
  raising.
- **Serve stamp.** The serving mode is a property of the training run, not of the package.
  One arm trains with relations on and another with them off; copying a stamp between them
  serves a graph the weights never saw.
- **Compute parity.** An earlier parity check collated its torch side without the padding
  masks and the relation bias the checkpoint was trained with, so both sides were wrong in
  the same way and it read 1.0000. The old serving forward agreed with its own checkpoint
  on 0.7926 of decisions. `tests_torch/test_numpy_parity.py` includes the control that
  matters: serving the wrong mode must be *detected*.

**One change from the team's shipped serving file.** On any exception inside `agent()`, the
team's `main_v7.py` returned a random legal move with no trace: the mechanism by which a
broken package played at random behind green offline gates. This copy answers with the
lowest legal indices instead, increments `main_v7.FALLBACKS` and logs each event to
stderr, following the fix the PPO baseline records for `main_v8.py`
([../ppo/docs/BASELINE.md](../ppo/docs/BASELINE.md), section 2b; `main_v8.py` itself is not
in the team records available here). `tests_torch/test_fallback.py` fails if the fallback
is random or silent again. The minimum-one-pick floor in that fallback is deliberate:
`main_v7` passes only through its explicit pass branch.

`gate_compute_parity.py` takes an optional `--main-v6` (the earlier unmasked serving file,
which is not in this repository) to print the size of that old defect beside the fix.

## Verification standards used here

Each was earned by a gate that passed while something was broken:

1. Every test needs a negative control: delete the fix, prove it goes red.
2. Assert what the code does, not what it looks like.
3. The probe must be able to observe success. If every assertion expects failure, a broken
   probe passes everything.
4. A gate that discovers its own work-list must assert it is non-empty.
5. An absent key must fail, never default. A default that silently selects broken behaviour
   is the bug with a flag in front of it (`training/strict_get.py`).
6. Use true exit codes, never `$?` after a pipe, and name the hash algorithm.

## Training notes

These are the team's measurements on the training host, kept because they are the kind of
thing that costs a run. They are not re-measured here.

- `--epochs` is a learning-rate hyperparameter, not a stopping rule: the cosine spans
  `total_steps`, so raising it stretches the whole decay. At 24 epochs every arm peaked
  earlier and lower.
- fp32 only. bf16 diverged on small decks; `--no-tf32` sets both the matmul and the cuDNN
  halves.
- `--decisions-only` drops rows with `label < 0`, which is every pass decision. There is no
  pass slot in the option table, so passing cannot be represented at all.
- Report `contested_top1` (rows with a real choice), not `top1`: about 8% of rows are forced
  (one legal option) and inflate it. Neither settles anything alone, since top-1 measures
  agreement with a strong player, not winning.

## What this does not do

- No search and no lookahead at inference: one forward pass per decision. The 12 auxiliary
  heads predict forward facts (will my Active be knocked out, how do prizes swing over two
  turns), which is amortised lookahead, not a plan it can check.
- Cross-turn history is near-absent: 4 tokens carrying a count and one type id for 32 log
  entries. Intra-turn memory is real (40 ordered tokens).
- No opponent modelling: every auxiliary head is about us.
- Passing is unrepresentable (no slot in the option table).
- The auxiliary heads were never supervised: across 219 runs none supplied a label table,
  so six heads are built from a fallback and carry about 448k parameters frozen at
  initialisation. Set `n_aux=0` or supply the table.

## Reading the comments

The source files keep their original, dense comments. Two conventions to know: paths such
as `tools/measure_select_stats.py` name measurement scripts of the team's working tree that
are not part of this repository, and ids such as `I-1` or `F-9` refer to that team's
internal defect register, which is not included either. The numbers quoted in those
comments (for example the pass-rate and cardinality tables in `serving/main_v7.py`) are
the team's measurements on its own corpus.

## Status

The state of the original repository (2026-08-16): the trainer and serving path were fixed
and gated, and the recipe was not frozen. Single-variable runs (relations on/off, head
width, model width) were in flight, and anything marked provisional in the paper is
provisional here.
