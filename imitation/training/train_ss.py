#!/usr/bin/env python3
"""Train the marnie-sixthsense transformer on the Sixth Sense imitation corpus.

Gate: held-out top-1 must beat **0.5943**, the measured dense-scorer baseline on
the same episode split (runs/marnie-sixthsense-bc-002).

Reports more than top-1 on purpose. `END` is legal in 45.7% of decisions and
chosen in 7.1% of those, so a model that never predicts END scores 92.9% on
every decision where ending is legal — top-1 alone would call that a success
while the agent dithers or passes, which the owner's doctrine names as the
agent's most persistent defect. END precision/recall and per-SelectType accuracy
are therefore first-class outputs.

    python training/train_ss.py --data <corpus dir> --run-id <name> --epochs 200
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_ss import (  # noqa: E402
    ATTACK_VOCAB, CARD_VOCAB, N_CARD_TYPE, N_ENERGY_TYPE, N_FAMILY, N_WDL,
    N_OPTION_TYPE, N_OWNER, NUM_W, OPT_NUM_W,
    SixthSenseNet, param_report,
)
from run_recorder import RunRecorder  # noqa: E402

END_OPTION_TYPE = 14
TRUNCATE_SEQ = False   # set from --truncate-seq; off reproduces the 790.8 run
SUBSAMPLE_SEED = 0      # set from --seed; makes the random subsample reproducible
DECISIONS_ONLY = False  # set from --decisions-only; see stream_precollate
MASK_PAD_OPTIONS = False  # set from --mask-pad-options; see collate_fast
# Widths come from the CORPUS, not from model_ss's globals. Those globals track
# whatever featurizer is current (160); training an OLD-architecture arm on a
# 64-wide corpus would otherwise allocate 160-wide tensors and hand the model 96
# dead columns -- silently training something that is not the old architecture.
NUM_W_RUN = None
# The value head became 2-class when draws were folded into losses. Corpora built
# under the OLD 3-class layout still carry wdl==2, and feeding that to a 2-class
# head raises "Assertion t >= 0 && t < n_classes" -- surfaced as an opaque CUBLAS
# error. Derive the class count from the DATA, exactly as NUM_W_RUN is derived,
# so an older corpus stays trainable instead of failing three runs in a row.
N_WDL_RUN = None
MASK_PAD_STATE = True
OPT_NUM_W_RUN = None
CARD_SEMANTICS_RUN = True
OUTCOME_MODE = "off"      # set from --outcome-weighting; see outcome_weight
OUTCOME_PARAM = 0.0       # the number after the colon, if the mode takes one
# Weight given to a row whose episode is absent from --demo-weights. Named, not
# inlined, so the fallback is greppable and so counting it is not optional --
# see demo_weight_targets (F-7).
DEMO_WEIGHT_ON_MISS = 1.0
TRUNC_STATE = 160   # measured max across ALL decks: Marnie 112, Alakazam 127,
                    # Lopunny 130. 128 was Marnie-only and overflowed Lopunny --
                    # the label-range guard caught it instead of silently training
                    # against the wrong option.
TRUNC_OPT = 64      # option maxima: Marnie 39, Alakazam 44, Lopunny 57. 64 is the
                    # full slot width, so options are no longer truncated at all;
                    # the state side still saves 192->160.
FAM_ACTIVE, FAM_PROMPT = 3, 12
# relation matrix is correct but unvectorised; off until measured as an ablation
BUILD_RELATIONS = False   # indices into featurize.FAMILIES
BASELINE = 0.5943


def load(path: Path, skip_train: bool = False, max_val: int = 0):
    """Read the frame file into dict rows.

    skip_train exists because a training split held as dicts costs ~594 KB/row
    against ~71 KB as a tensor. Callers that stream the training split into
    tensors (stream_precollate) must not pay that cost for rows they will never
    use as dicts.

    PASS skip_train=True IF YOU DISCARD THE FIRST RETURN VALUE. Writing
    `_train, rows = load(...)` still materialises every training row first:
    on the 26,369-game corpus that is 2,147,439 dicts, over a terabyte, built
    and then thrown away. Six tools did exactly that -- including
    gate_export_parity and gate_release_ss, both part of the release pipeline.
    It never bit only because those tools had previously run against 895- to
    1,500-game corpora.
    """
    train, val = [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] == "validation":
                # A capped validation set is a sampling choice, not a shortcut:
                # 60k decisions estimate top-1 to well under a tenth of a point,
                # while the full 380k costs ~226 GB as dicts and stalls the NCCL
                # handshake past its 600 s timeout.
                if max_val and len(val) >= max_val:
                    continue
                val.append(row)
            elif not skip_train:
                train.append(row)
    return train, val


# Per-head weights. The critic probe measured these are NOT equally learnable:
# future KO learns cleanly (Brier 0.378 vs 0.487 base rate) while prizes-in-2-
# turns is unpredictable and got worse with training (0.890). Averaging all
# heads equally dilutes real signal with noise. Intra-turn heads (6-11) carry
# the within-turn plan and are weighted up.
AUX_HEAD_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 0.5, 0.5,   # cross-turn
                    1.0, 1.5, 1.0, 1.5, 1.0, 0.5]   # intra-turn
# mirrors build_lookahead_v2.HEAD_CLASSES; 18 = OptionType count + TURN_ENDS
AUX_CLASSES = [4, 2, 2, 2, 2, 4, 18, 2, 4, 2, 2, 3]


def load_value_table(path):
    """(episode, step) -> soft 3-vector value target, or None if not supplied.

    Supplied-but-absent is an error here for the same reason as in
    load_aux_table: a silent None turns a requested training signal off and
    reports nothing.
    """
    if path is None:
        return None
    if not Path(path).exists():
        raise SystemExit(
            f"--value-targets {path} does not exist. Refusing to silently "
            f"train without soft value targets; build it or omit the flag.")
    table = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            table[(r["episode"], r["step"])] = r["soft"]
    return table


def soft_value_targets(rows, table, device, prior=(0.5788, 0.4212)):
    """NOTE: the value head is now 2-class (win / not-win); draws are folded into
    losses by build_ss_dataset_par because the objective is to win, not to avoid
    losing. Any --value-targets table built against the old 3-class layout will
    mismatch, so the width is checked below rather than broadcasting silently.

    RETURNS (targets, misses). Same shape of defect as F-7 and found by the sweep
    for it: `table.get(key, prior)` handed an unjoined row the corpus BASE RATE,
    which is a thoroughly plausible soft target and byte-identical to a genuine
    table entry at the base rate. A --value-targets file keyed against a
    different episode/step convention would therefore train the value head on a
    constant, at the base rate, and report nothing. The prior is still the right
    fallback; the count is what makes its use visible.
    """
    out = torch.zeros(len(rows), N_WDL)
    misses = 0
    for i, r in enumerate(rows):
        v = table.get((r.get("episode"), r.get("step")))
        if v is None:
            misses += 1
            v = prior
        if len(v) != N_WDL:
            raise SystemExit(
                f"--value-targets row has width {len(v)}, the value head is "
                f"{N_WDL}-class. The table was built for the old 3-class layout; "
                f"rebuild it with draws folded into losses.")
        out[i] = torch.tensor(v)
    return out.to(device), misses


def load_aux_table(path):
    """episode+optional seat+step -> label vector, or None if not supplied.

    A path that was SUPPLIED but does not exist is a hard error, not a None.
    Treating the two the same cost a full-corpus multi-GPU run: it was launched with
    --lookahead artifacts/lookahead_marnie-ALL.jsonl.gz, which had never been
    built, so aux_table came back None, the 12 look-ahead heads silently
    vanished, --aux-weight 0.30 weighted nothing, and the arm trained the pure
    imitation recipe that measured 699.8 on the ladder instead of the
    multi-task one that measured 790.8. Nothing in the log said so.
    """
    if path is None:
        return None
    if not Path(path).exists():
        raise SystemExit(
            f"--lookahead {path} does not exist. Refusing to silently train "
            f"without the auxiliary heads; build it or omit --lookahead.")
    table = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            # 739/60,000 sampled generalist rows collide on (episode, step).
            # New tables retain both seats; old seatless tables keep their
            # exact key shape so already-running arms remain compatible.
            key = ((r["episode"], r["seat"], r["step"])
                   if "seat" in r else (r["episode"], r["step"]))
            table[key] = r["aux"]
    return table


def load_demo_weights(path):
    """episode -> demonstrator weight, or None if not supplied."""
    if path is None:
        return None
    if not Path(path).exists():
        raise SystemExit(
            f"--demo-weights {path} does not exist. Refusing to silently "
            f"train without demonstrator weights; build it or omit the flag.")
    return {episode: float(weight)
            for episode, weight in json.loads(Path(path).read_text()).items()}


def demo_weight_targets(rows, table, device):
    """Per-SEAT lookup, falling back to per-episode.

    A game has two players of different strength, so one weight per episode
    would score a 1200-rated pilot and a 900-rated one identically. Keys are
    tried as "episode|seat" first; the bare episode key still works so existing
    weight files keep functioning.

    This multiplies into the POLICY loss only -- the value and 12 aux heads see
    every row at full weight. That is the point: weak players' STATES are the
    only source of bad-position coverage, which is exactly what compounding
    error needs, while their CHOICES are not worth copying.

    RETURNS (weights, misses). F-7: this was a double join ending in a literal
    default, so a row that matched NEITHER key received full weight -- byte
    identical to a row that genuinely joined at 1.0, with nothing recording that
    it happened. A weight table keyed differently than the corpus therefore
    trained every row at 1.0 while the log reported that weights were supplied.
    The default value is not the defect and 1.0 is still the right one; being
    unable to tell it was used is. `misses` is what makes it tellable, and the
    callers turn a TOTAL miss into a hard exit -- see _report_demo_join. This is
    the same accounting the aux join already does with `matched` (aux_targets).
    """
    out, misses = [], 0
    for row in rows:
        v = table.get(f"{row.get('episode')}|{row.get('seat')}")
        if v is None:
            v = table.get(str(row.get("episode")))
        if v is None:
            misses += 1
            v = DEMO_WEIGHT_ON_MISS
        out.append(v)
    return torch.tensor(out, dtype=torch.float32, device=device), misses


def _report_demo_join(where, misses, rows):
    """Print the demonstrator-weight join rate; exit if NOTHING joined.

    A partial miss is legitimate (a weight file need not cover every episode),
    so it is reported and not fatal. A TOTAL miss over a non-empty pass means
    the table is keyed differently than the corpus -- the exact failure F-7 made
    invisible -- and there is no reading of that which is worth a training run.
    """
    if not rows:
        return
    if misses == rows:
        raise SystemExit(
            f"--demo-weights joined 0 of {rows} rows in {where}. Every row would "
            f"train at the fallback weight {DEMO_WEIGHT_ON_MISS}, which is "
            f"indistinguishable from a table of all-1.0 weights. The table is "
            f"keyed differently than the corpus (expected 'episode|seat' or "
            f"'episode'); rebuild it or drop --demo-weights.")
    if misses:
        print(f"demo-weight join [{where}]: {rows - misses}/{rows} rows matched, "
              f"{misses} fell back to {DEMO_WEIGHT_ON_MISS}", flush=True)


def policy_cross_entropy(logits, label, weights=None):
    if weights is None:
        return F.cross_entropy(logits.float(), label, ignore_index=-1)
    per = F.cross_entropy(logits.float(), label, ignore_index=-1,
                          reduction="none")
    valid = label >= 0
    weighted_valid = weights * valid
    # Demonstrators agree on the exact card 85.4% of the time; weighting should
    # change whose judgment wins only on the measured disagreement remainder.
    return ((per * weighted_valid).sum()
            / weighted_valid.sum().clamp(min=torch.finfo(per.dtype).eps))


def outcome_weight(wdl, demo_weight):
    """Per-row policy weight derived from the demonstrator's GAME RESULT.

    wdl is 0 win / 1 draw / 2 loss (build_ss_dataset_par.py:82), so the outcome
    is already on every row and no manifest join is needed.

    WHY NOT SIMPLY PASS NEGATIVE WEIGHTS. policy_cross_entropy normalises by
    weights.sum(); a negative weight does not flip one row's gradient, it poisons
    the denominator for the whole batch and can drive the mean to +-inf as the
    sum crosses zero. Pushing probability DOWN needs its own term, which is what
    unlikelihood_term does. Everything here stays non-negative by construction.

    Modes:
      off           every row weight 1 (unchanged behaviour)
      winners-only  keep wins, drop losses -- filtered behavioural cloning
      soft:<w>      keep everything, losses scaled by w -- the same idea without
                    throwing away ~35% of an already small corpus
      unlikelihood  winners-only cross-entropy PLUS an explicit push away from
                    the losing actions; see unlikelihood_term
    """
    if OUTCOME_MODE == "off":
        return demo_weight
    if OUTCOME_MODE in ("winners-only", "unlikelihood"):
        w = (wdl == 0).float()
    elif OUTCOME_MODE == "soft":
        w = torch.ones_like(wdl, dtype=torch.float32)
        w = torch.where(wdl == 2, torch.full_like(w, OUTCOME_PARAM), w)
    else:
        return demo_weight
    return w if demo_weight is None else w * demo_weight


def unlikelihood_term(logits, label, wdl):
    """Push probability AWAY from the action taken in games the demonstrator lost.

    Welleck et al. 2019 form: -log(1 - p(negative)). Stable where a negated
    cross-entropy is not, because it is bounded below by 0 and only diverges as
    p -> 1, which is exactly the case worth punishing.

    THE CAVEAT THAT DECIDES WHETHER THIS HELPS. A lost game is mostly GOOD moves;
    the loss usually comes from a few bad decisions or from variance. Labelling
    every row of a lost game "bad" is crude credit assignment, and the honest
    expectation is that it hurts unless the weight is small. That is why this is
    an experiment with a swept lambda and an arena gate, not a default.
    """
    if OUTCOME_MODE != "unlikelihood" or not OUTCOME_PARAM:
        return logits.new_zeros(())
    neg = (wdl == 2) & (label >= 0)
    if not bool(neg.any()):
        return logits.new_zeros(())
    logp = F.log_softmax(logits.float(), -1)
    taken = logp.gather(1, label.clamp(min=0).unsqueeze(1)).squeeze(1).exp()
    ul = -torch.log((1.0 - taken).clamp(min=1e-6))
    return OUTCOME_PARAM * (ul * neg).sum() / neg.sum().clamp(min=1)


def aux_targets(rows, table, device):
    n_heads = len(AUX_HEAD_WEIGHTS)
    out = torch.full((len(rows), n_heads), -1, dtype=torch.long)
    matched = torch.zeros(len(rows), dtype=torch.bool)
    for i, r in enumerate(rows):
        episode, step = r.get("episode"), r.get("step")
        lab = table.get((episode, r.get("seat"), step))
        if lab is None:
            lab = table.get((episode, step))
        if lab is not None:
            matched[i] = True
            for h in range(min(n_heads, len(lab))):
                out[i, h] = int(lab[h])
    return out.to(device), matched.to(device)


def _grad_norm(mod):
    """L2 norm of one submodule's gradients, as a device tensor.

    Returned unsynced so the caller can accumulate it without stalling the step;
    must be read BEFORE clip_grad_norm_, which rescales gradients in place.
    """
    ss = [(p.grad.detach().float() ** 2).sum()
          for p in mod.parameters() if p.grad is not None]
    if not ss:
        # On the module's own device, not the CPU default: the caller adds this
        # to a device accumulator, and find_unused_parameters means a head can
        # legitimately have no gradient on a given step.
        p = next(mod.parameters(), None)
        return torch.zeros((), device=p.device if p is not None else "cpu")
    return torch.sqrt(torch.stack(ss).sum())


def aux_cross_entropy(head_out, target):
    target = target.clamp(max=head_out.shape[-1] - 1)
    valid = (target != -1).sum()
    return (F.cross_entropy(head_out.float(), target, ignore_index=-1,
                            reduction="sum") / valid.clamp(min=1))


def collate(rows, device):
    b = len(rows)
    ns = max(len(r["family"]) for r in rows)
    no = max(len(r["option_type"]) for r in rows)
    fam = torch.zeros(b, ns, dtype=torch.long)
    own = torch.zeros(b, ns, dtype=torch.long)
    card = torch.zeros(b, ns, dtype=torch.long)
    ctype = torch.zeros(b, ns, dtype=torch.long)
    etype = torch.zeros(b, ns, dtype=torch.long)
    num = torch.zeros(b, ns, NUM_W_RUN or NUM_W)
    otype = torch.zeros(b, no, dtype=torch.long)
    onum = torch.zeros(b, no, OPT_NUM_W_RUN or OPT_NUM_W)
    mask = torch.zeros(b, ns + no, dtype=torch.bool)
    omask = torch.zeros(b, no, dtype=torch.bool)
    optr = torch.full((b, no, 2), -1, dtype=torch.long)
    oatk = torch.zeros(b, no, dtype=torch.long)
    label = torch.zeros(b, dtype=torch.long)
    wdl = torch.zeros(b, dtype=torch.long)
    for i, r in enumerate(rows):
        s = min(len(r["family"]), ns)
        o = min(len(r["option_type"]), no)
        fam[i, :s] = torch.tensor(r["family"]).clamp(min=0, max=N_FAMILY - 1)
        own[i, :s] = torch.tensor(r["owner"]).clamp(min=0, max=N_OWNER - 1)
        card[i, :s] = torch.tensor(r["card"]).clamp(min=0, max=CARD_VOCAB - 1)
        # .get(): corpora built before card semantics existed simply omit these,
        # and index 0 is the padding_idx "no card" row, so an old corpus trains
        # exactly as it did before rather than crashing.
        if r.get("card_type") is not None:
            ctype[i, :s] = torch.tensor(r["card_type"][:s]).clamp(min=0, max=N_CARD_TYPE - 1)
        if r.get("energy_type") is not None:
            etype[i, :s] = torch.tensor(r["energy_type"][:s]).clamp(min=0, max=N_ENERGY_TYPE - 1)
        for j, vec in enumerate(r["numeric"]):
            num[i, j, :len(vec)] = torch.tensor(vec)
        # featurize pads opt_type with -1; nn.Embedding rejects negative indices
        # and row N_OPTION_TYPE is reserved for exactly this. Same for the
        # state-side ids, which pad with 0 (PAD) already.
        otype[i, :o] = torch.tensor(r["option_type"]).clamp(min=0, max=N_OPTION_TYPE)
        oa = r.get("option_attack") or []
        if oa:
            oatk[i, :min(o, len(oa))] = torch.tensor(oa[:o]).clamp(min=0, max=ATTACK_VOCAB - 1)
        for j, vec in enumerate(r["option_numeric"]):
            onum[i, j, :len(vec)] = torch.tensor(vec)
        mask[i, :s] = True
        mask[i, ns:ns + o] = True
        omask[i, :o] = True
        for j, (src, dst) in enumerate(r["option_ptr"]):
            optr[i, j, 0] = src if 0 <= src < s else -1
            optr[i, j, 1] = dst if 0 <= dst < s else -1
        # label -1 means "pass"; the engine offers no option token for it, so it
        # is folded onto the END option when one exists, else the row is skipped
        lbl = r["label"]
        if lbl < 0:
            ends = [k for k, t in enumerate(r["option_type"]) if t == END_OPTION_TYPE]
            # Measured: ALL 840 pass rows lack an END option, so the old fallback
            # sent every one of them to index 0 -- a fabricated target with a
            # positional bias. Keep -1 instead; the policy loss masks these rows
            # out, while value and aux still learn from the state.
            lbl = ends[0] if ends else -1
        label[i] = lbl
        wdl[i] = r["wdl"]
    # ---- typed relation matrix over [state ; option] ----
    # 0 none · 1 same-owner · 2 same-zone · 3 option->referent
    # 4 option->destination · 5 active<->active · 6 prompt->option
    #
    # WHY THIS MATTERS. The encoder has NO positional embedding -- the board is a
    # set -- so without typed edges attention is fully permutation-invariant and
    # cannot distinguish "this energy is attached to THAT Pokemon" from "this
    # energy exists somewhere". That is an expressivity gap, not a hyperparameter,
    # and it is the one lever the 17-config shape matrix never varied.
    #
    # VECTORISED. The original built this with nested Python loops per row
    # (for a in act: for c in act:) and was left disabled as "correct but
    # unvectorised" -- at 8.8M rows that cost dominates the step. Everything here
    # is derived from family / owner / option_ptr, all already present in the row,
    # so NO corpus rebuild is needed: BUILD_RELATIONS is a trainer flag.
    n = ns + no
    rel = None
    if BUILD_RELATIONS:
        rel = torch.zeros(b, n, n, dtype=torch.long)
        fam_b = torch.from_numpy(np.asarray(
            [r["family"][:ns] + [0] * (ns - len(r["family"][:ns])) for r in rows],
            dtype=np.int64))
        own_b = torch.from_numpy(np.asarray(
            [r["owner"][:ns] + [2] * (ns - len(r["owner"][:ns])) for r in rows],
            dtype=np.int64))
        # order matters: same-zone overwrites same-owner, matching the original
        block = torch.where(own_b[:, :, None] == own_b[:, None, :], 1, 0)
        block = torch.where(fam_b[:, :, None] == fam_b[:, None, :], 2, block)
        # active<->active only across DIFFERENT owners
        act = (fam_b == FAM_ACTIVE)
        cross = act[:, :, None] & act[:, None, :] & (own_b[:, :, None] != own_b[:, None, :])
        block = torch.where(cross, 5, block)
        rel[:, :ns, :ns] = block

        ptr = torch.from_numpy(np.asarray(
            [[list(pd) for pd in r["option_ptr"][:no]]
             + [[-1, -1]] * (no - len(r["option_ptr"][:no])) for r in rows],
            dtype=np.int64))                       # (b, no, 2)
        src, dst = ptr[..., 0], ptr[..., 1]
        rows_ix = torch.arange(b)[:, None].expand(b, no)
        cols = torch.arange(no)[None, :].expand(b, no) + ns
        for val, tgt in ((3, src), (4, dst)):
            ok = (tgt >= 0) & (tgt < ns)
            if bool(ok.any()):
                bi, oi = rows_ix[ok], cols[ok]
                ti = tgt[ok]
                rel[bi, oi, ti] = val
                rel[bi, ti, oi] = val
        prm = (fam_b == FAM_PROMPT)                # (b, ns)
        if bool(prm.any()):
            pb, pi = prm.nonzero(as_tuple=True)
            # every prompt token <-> every option column of the same row
            for off in range(no):
                rel[pb, pi, ns + off] = 6
                rel[pb, ns + off, pi] = 6

    to = lambda t: t.to(device, non_blocking=True)  # noqa: E731
    return ({"family": to(fam), "owner": to(own), "card": to(card),
             "card_type": to(ctype), "energy_type": to(etype), "numeric": to(num),
             "option_type": to(otype), "option_numeric": to(onum),
             "mask": to(mask), "option_mask": to(omask),
             "option_ptr": to(optr), "option_attack": to(oatk),
             "relation": (to(rel) if rel is not None else None)}, to(label), to(wdl))


def _attach_relation(b):
    """Fill b["relation"] in place, per batch, when --relations is on.

    WHY IT IS BUILT HERE AND NOT PRECOLLATED: the per-row matrix is 392 KiB and
    quadratic in sequence length -- ~215 GB per rank if stored, which is exactly
    why stream_precollate forces relation to None. At batch 96 it is 38.5 MB,
    allocated and freed per step.

    WHY IT IS SAFE FOR A WARM START: rel_bias is nn.Parameter(zeros), so a model
    given a relation matrix computes bit-identically to one given None until
    gradients move it. Enabling this changes nothing at initialization.

    *** SERVING OBLIGATION -- READ BEFORE PACKAGING ***
    Once rel_bias is TRAINED to non-zero, a package whose main.py passes no
    relation is running a DIFFERENT GRAPH from the one that was trained. That is
    the identical failure mode as the attention mask: train and serve currently
    agree only because both are absent. Any checkpoint trained with --relations
    MUST ship with a main.py that builds the same matrix, and the flag belongs
    stamped in weights.npz so the two cannot be mispaired.

    *** F-8: A MISSING INPUT KEY RAISES, IT DOES NOT RETURN QUIETLY ***
    This used to be `if fam is None or own is None or optr is None: return b`,
    so a collate path that stopped emitting one of them left relation=None while
    every log still printed BUILD_RELATIONS=True. The flag would read as ON for
    the whole run and the relational bias would train on nothing. The inputs are
    all produced unconditionally by collate/collate_fast, so an absent one means
    an upstream path changed -- there is no benign reading of it, and guessing
    one is how this project loses runs.
    """
    if not BUILD_RELATIONS or b.get("relation") is not None:
        return b
    fam, own, optr = b.get("family"), b.get("owner"), b.get("option_ptr")
    otype = b.get("option_type")
    absent = [name for name, v in (("family", fam), ("owner", own),
                                   ("option_ptr", optr), ("option_type", otype))
              if v is None]
    if absent:
        raise KeyError(
            f"--relations is on but the batch is missing {absent}. "
            f"_attach_relation cannot build the relation matrix without them, "
            f"and skipping it silently would leave relation=None while the run "
            f"kept reporting BUILD_RELATIONS=True (F-8).")
    b["relation"] = build_relation_batch(
        fam, own, optr, fam.shape[1], otype.shape[1])
    return b


def build_relation_batch(fam, own, optr, ns, no):
    """(b, ns) family, (b, ns) owner, (b, no, 2) option_ptr -> (b, ns+no, ns+no).

    Built fresh each step from tensors already on the device and never stored:
    the per-row form is 392 KiB and precollating it would need ~215 GB per rank.
    Edge ids and their overwrite order match the reference builder exactly.
    """
    import torch

    b = fam.shape[0]
    n = ns + no
    dev = fam.device
    rel = torch.zeros(b, n, n, dtype=torch.long, device=dev)

    # --- state x state ---------------------------------------------------
    same_own = own[:, :, None] == own[:, None, :]
    same_fam = fam[:, :, None] == fam[:, None, :]
    block = torch.where(same_own, 1, 0)
    block = torch.where(same_fam, 2, block)          # zone beats owner
    act = fam == FAM_ACTIVE
    cross = act[:, :, None] & act[:, None, :] & ~same_own
    block = torch.where(cross, 5, block)             # active-active beats both
    rel[:, :ns, :ns] = block

    # --- option -> pointed-at state token, symmetric ----------------------
    src_i, dst_i = optr[..., 0], optr[..., 1]        # (b, no)
    rows_ix = torch.arange(b, device=dev)[:, None].expand(b, no)
    cols = torch.arange(no, device=dev)[None, :].expand(b, no) + ns
    for val, tgt in ((3, src_i), (4, dst_i)):
        ok = (tgt >= 0) & (tgt < ns)
        if bool(ok.any()):
            bi, oi, ti = rows_ix[ok], cols[ok], tgt[ok]
            rel[bi, oi, ti] = val
            rel[bi, ti, oi] = val

    # --- prompt <-> every option, LAST so it wins on prompt tokens --------
    prm = (fam == FAM_PROMPT)                        # (b, ns)
    if bool(prm.any()):
        pm = prm[:, :, None].expand(b, ns, no)
        q = rel[:, :ns, ns:]
        rel[:, :ns, ns:] = torch.where(pm, torch.full_like(q, 6), q)
        q2 = rel[:, ns:, :ns]
        rel[:, ns:, :ns] = torch.where(pm.transpose(1, 2),
                                       torch.full_like(q2, 6), q2)
    return rel


def collate_fast(rows, device, truncate=None):
    if truncate is None:
        truncate = TRUNCATE_SEQ
    """Vectorised collate. `truncate` slices each batch to its REAL token count.

    The dataset arrives pre-padded to 192 state / 64 option slots, and the mask
    was set True across the full width -- so every attention layer computed a
    256x256 matrix. Measured over 4,000 rows: the mean real length is 57.5 of
    256, i.e. 22% useful, and attention is O(n^2), so ~20x of the attention work
    was spent on padding. main.py has always sliced (its own comment: "attending
    over PAD tokens as if they were board entities cost 29 points of agreement");
    training never did, which is both a waste and a train/serve mismatch.

    PAD is family id 0 for state tokens and option_type < 0 for options.
    """
    b = len(rows)
    if truncate:
        # FIXED widths, not batch-max. Measured over all 89,048 rows: real state
        # tokens max out at 112 (p99 83) and real options at 39 (p99 21), while the
        # dataset pads them to 192 and 64. 128/48 covers both maxima with margin.
        #
        # Fixed beats batch-max here for a specific reason: batch-max varies per
        # step, and varying shapes make torch.compile recompile -- its limit is 8,
        # after which it silently falls back to eager and the compile win is gone.
        # A constant shape compiles once. Padding is RIGHT-side (trailing 0 for
        # family, -1 for option_type), verified, so a head slice keeps the content.
        ns, no = min(TRUNC_STATE, len(rows[0]["family"])), min(TRUNC_OPT, len(rows[0]["option_type"]))
    else:
        ns = max(len(r["family"]) for r in rows)
        no = max(len(r["option_type"]) for r in rows)
    fam = np.zeros((b, ns), dtype=np.int64)
    own = np.zeros((b, ns), dtype=np.int64)
    card = np.zeros((b, ns), dtype=np.int64)
    ctype = np.zeros((b, ns), dtype=np.int64)
    etype = np.zeros((b, ns), dtype=np.int64)
    num = np.zeros((b, ns, NUM_W_RUN or NUM_W), dtype=np.float32)
    otype = np.zeros((b, no), dtype=np.int64)
    onum = np.zeros((b, no, OPT_NUM_W_RUN or OPT_NUM_W), dtype=np.float32)
    mask = np.zeros((b, ns + no), dtype=np.bool_)
    omask = np.zeros((b, no), dtype=np.bool_)
    optr = np.full((b, no, 2), -1, dtype=np.int64)
    oatk = np.zeros((b, no), dtype=np.int64)
    label = np.zeros(b, dtype=np.int64)
    wdl = np.zeros(b, dtype=np.int64)
    for i, r in enumerate(rows):
        # clamp to the buffer width, or a row longer than ns overflows it
        s, o = min(len(r["family"]), ns), min(len(r["option_type"]), no)
        fam[i, :s] = np.clip(r["family"][:s], 0, N_FAMILY - 1)
        own[i, :s] = np.clip(r["owner"][:s], 0, N_OWNER - 1)
        card[i, :s] = np.clip(r["card"][:s], 0, CARD_VOCAB - 1)
        if r.get("card_type") is not None:
            ctype[i, :s] = np.clip(r["card_type"][:s], 0, N_CARD_TYPE - 1)
        if r.get("energy_type") is not None:
            etype[i, :s] = np.clip(r["energy_type"][:s], 0, N_ENERGY_TYPE - 1)
        numeric = np.asarray(r["numeric"][:s], dtype=np.float32)
        num[i, :len(numeric), :numeric.shape[1]] = numeric
        otype[i, :o] = np.clip(r["option_type"][:o], 0, N_OPTION_TYPE)
        oa = r.get("option_attack") or []
        if oa:
            n_attack = min(o, len(oa))
            oatk[i, :n_attack] = np.clip(oa[:o], 0, ATTACK_VOCAB - 1)
        option_numeric = np.asarray(r["option_numeric"][:o], dtype=np.float32)
        onum[i, :len(option_numeric), :option_numeric.shape[1]] = option_numeric
        if MASK_PAD_STATE:
            # family 0 is PAD and nothing else, so the real token count is
            # recoverable without a corpus rebuild -- the same trick the option
            # side uses with its -1 sentinel.
            _fam = np.asarray(r["family"][:s])
            _real = int((_fam > 0).sum())
            mask[i, :_real] = True
        else:
            mask[i, :s] = True
        # MASK_PAD_OPTIONS: mark only the options the engine actually offered.
        #
        # featurize.py:647 emits a correct option_mask, but
        # build_ss_dataset_par.py never writes it into the row, so `o` is the
        # PADDED width and both masks below marked all 64 slots valid. Measured
        # on the live corpus: option_type is 64 long in 200,000/200,000 rows
        # while only 5.61 options are real, and 0 rows carry an option_mask. So
        # model_ss.py's masked_fill(~option_mask) masked nothing, the policy
        # softmax spanned 64 slots with ~58 of them padding, and the padding's
        # option_type was clipped from -1 to 0 -- OptionType.NUMBER, a real
        # type. Serving does the opposite: main.py slices to sum(option_mask),
        # about 5.6 tokens. Train and serve therefore ran different sequence
        # lengths through the same attention.
        #
        # The padding is still recoverable because it is stored as -1, which is
        # what makes this fixable without rebuilding the corpus.
        n_real = int((np.asarray(r["option_type"][:o]) >= 0).sum()) \
            if MASK_PAD_OPTIONS else o
        mask[i, ns:ns + (n_real if MASK_PAD_OPTIONS else o)] = True
        omask[i, :n_real] = True
        ptr = np.asarray(r["option_ptr"][:o], dtype=np.int64)
        ptr = np.where((0 <= ptr) & (ptr < s), ptr, -1)
        optr[i, :len(ptr)] = ptr
        lbl = r["label"]
        if lbl < 0:
            ends = np.flatnonzero(np.asarray(r["option_type"][:o]) == END_OPTION_TYPE)
            # -1, NOT 0. collate() was fixed for this and carries the measurement:
            # "ALL 840 pass rows lack an END option, so the old fallback sent every
            # one of them to index 0 -- a fabricated target with a positional bias."
            # collate_fast is the ONLY collate on the training path, so until now
            # the ignore_index=-1 machinery was unreachable and evaluate()'s
            # `scored = label >= 0` was always True -- meaning top-1, the sole
            # checkpoint-selection criterion, was measured on a wrong denominator.
            lbl = int(ends[0]) if len(ends) else -1
        # Truncation must never cut off the chosen action. o counts options with
        # option_type >= 0, and a real label always indexes one of those, so this
        # should be unreachable -- but a silently out-of-range label would train
        # the model against the wrong option, which is exactly the class of bug
        # that is invisible until the agent plays badly.
        if lbl < 0:
            pass  # kept as -1; masked out of the policy loss, see collate()
        elif not 0 <= lbl < o:
            raise ValueError(
                f"label {lbl} outside truncated option count {o} "
                f"(row episode={r.get('episode')} step={r.get('step')})")
        label[i] = lbl
        wdl[i] = r["wdl"]

    to = lambda a: torch.from_numpy(a).to(device, non_blocking=True)  # noqa: E731
    return ({"family": to(fam), "owner": to(own), "card": to(card),
             "card_type": to(ctype), "energy_type": to(etype), "numeric": to(num),
             "option_type": to(otype), "option_numeric": to(onum),
             "mask": to(mask), "option_mask": to(omask),
             "option_ptr": to(optr), "option_attack": to(oatk),
             "relation": None}, to(label), to(wdl))


def _shard_dir(args, world, name):
    """Path to a pre-sharded file if shard_frames.py has been run, else None.

    Opt-in by presence, not by flag: a run launched against a dataset that has
    shards should use them, and one against a dataset that does not must still
    work. Returning None lets every call site keep its old behaviour verbatim.
    """
    p = Path(args.data) / f"shards_w{world}" / name
    return p if p.exists() else None


def stream_precollate(path, want_split, device, aux_table, chunk=4096,
                      rank=0, world=1, n_rows=None, demo_weights=None):
    """Collate a split straight from disk into one set of tensors.

    WHY THIS EXISTS. load() materialises every row as a Python dict before any
    tensor is built, and a dict row costs ~594 KB against ~71 KB as a tensor --
    8x. Measured: 609k rows reached 362 GB RSS and never finished epoch 1;
    2.5M rows reached 490 GB and died. That ceiling, not the GPU and not the
    data, is why every arm was capped at 1,500 games out of 26,369.

    Reading a chunk, collating it, and dropping the dicts holds only one chunk
    of dicts at a time, so peak memory is the final tensors plus ~4k rows. The
    per-chunk call is collate_fast -- the same function the per-batch path uses
    -- so the result is bit-identical to precollate_split, just built without
    ever holding the whole split as Python objects.

    device="cpu" keeps the tensors in host RAM (880 GB here) for datasets far
    larger than a 96 GB card; the training loop moves each batch across.
    """
    import torch

    keys, parts, labels, wdls, auxes, demos = None, [], [], [], [], []
    big = big_label = big_wdl = big_aux = big_demo = None
    buf, n, seen = [], 0, 0
    demo_miss = 0   # F-7: an unjoined demonstrator weight must be counted
    _rng = random.Random((SUBSAMPLE_SEED << 8) ^ (rank << 4) ^ world)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            # STRIDE BEFORE PARSE when world > 1. json.loads is essentially the
            # ENTIRE cost of this function, so skipping a row AFTER parsing it
            # saves memory and not one second of load time -- measured 3.24 MB/s
            # of compressed input either way. world > 1 occurs only on the
            # pre-sharded path (see the caller), where every line already carries
            # split == want_split, so counting lines and counting rows coincide.
            # The split test is KEPT below as a cheap assertion rather than
            # deleted: if that assumption is ever false, this fails loudly
            # instead of silently training each rank on the wrong rows.
            if world > 1:
                # RANDOM keep, not a modulo stride. This file is ALREADY
                # round-robin sharded, so every-Nth-row here is every (N*world)th
                # decision of the original corpus -- a fixed period that can beat
                # against episode length and systematically favour certain
                # positions within a turn. A seeded random draw has no period to
                # collide with. Drawn BEFORE json.loads because the parse is
                # essentially the entire cost of this function.
                if _rng.randrange(world) != 0:
                    continue
                seen += 1
            row = json.loads(line)
            if row["split"] != want_split:
                continue
            if DECISIONS_ONLY:
                # A row the engine offered one option on is not a decision: it
                # scores 1.0 by construction and teaches nothing, while diluting
                # both the training signal and the top-1 that selects the
                # checkpoint. Same for a row with no real label -- pass rows are
                # given a fabricated index elsewhere in this file.
                _ot = row.get("option_type") or []
                if sum(1 for _t in _ot if _t >= 0) < 2:
                    continue
                if int(row.get("label", -1)) < 0:
                    continue
            # Round-robin shard. Each rank keeps every world-th row, so the four
            # GPUs together hold the split exactly once. Striding rather than
            # slicing contiguously keeps each shard mixed across episodes, so no
            # rank trains on one narrow slice of the corpus.
            if world <= 1:
                seen += 1
            buf.append(row)
            if len(buf) >= chunk:
                b, lab, w = collate_fast(buf, device)
                if keys is None:
                    keys = [k for k, v in b.items() if v is not None]
                if n_rows is None:
                    parts.append({k: b[k] for k in keys})
                    labels.append(lab)
                    wdls.append(w)
                    if aux_table is not None:
                        auxes.append(aux_targets(buf, aux_table, device)[0])
                    if demo_weights is not None:
                        _d, _m = demo_weight_targets(buf, demo_weights, device)
                        demo_miss += _m
                        demos.append(_d)
                else:
                    aux = (aux_targets(buf, aux_table, device)[0]
                           if aux_table is not None else None)
                    demo = None
                    if demo_weights is not None:
                        demo, _m = demo_weight_targets(buf, demo_weights, device)
                        demo_miss += _m
                    if big is None:
                        # 2,214,025 rows/rank at the measured 63,632 bytes/row
                        # is ~131 GB. Allocate the destination once: cat would
                        # retain another ~131 GB of parts until it completed.
                        big = {k: b[k].new_empty((n_rows, *b[k].shape[1:]))
                               for k in keys}
                        big_label = lab.new_empty((n_rows, *lab.shape[1:]))
                        big_wdl = w.new_empty((n_rows, *w.shape[1:]))
                        if aux is not None:
                            big_aux = aux.new_empty((n_rows, *aux.shape[1:]))
                        if demo is not None:
                            big_demo = demo.new_empty((n_rows, *demo.shape[1:]))
                    # CLAMP to the preallocation. With a modulo stride the kept
                    # count was exact and a chunk could never overshoot; a RANDOM
                    # keep is binomial, so the last chunk can run past n_rows.
                    # Copy only what fits rather than crashing on a size mismatch.
                    end = n + len(buf)
                    if end > n_rows:
                        end = n_rows
                    take = end - n
                    for k in keys:
                        big[k][n:end].copy_(b[k][:take])
                    big_label[n:end].copy_(lab[:take])
                    big_wdl[n:end].copy_(w[:take])
                    if aux is not None:
                        big_aux[n:end].copy_(aux[:take])
                    if demo is not None:
                        big_demo[n:end].copy_(demo[:take])
                    n = end
                    buf = []
                    if n >= n_rows:
                        break
                if n_rows is None:
                    n += len(buf)
                buf = []
                if n_rows is not None:
                    del b, lab, w, aux, demo
    if buf:
        b, lab, w = collate_fast(buf, device)
        if keys is None:
            keys = [k for k, v in b.items() if v is not None]
        if n_rows is None:
            parts.append({k: b[k] for k in keys})
            labels.append(lab)
            wdls.append(w)
            if aux_table is not None:
                auxes.append(aux_targets(buf, aux_table, device)[0])
            if demo_weights is not None:
                _d, _m = demo_weight_targets(buf, demo_weights, device)
                demo_miss += _m
                demos.append(_d)
        else:
            aux = (aux_targets(buf, aux_table, device)[0]
                   if aux_table is not None else None)
            demo = None
            if demo_weights is not None:
                demo, _m = demo_weight_targets(buf, demo_weights, device)
                demo_miss += _m
            if big is None:
                big = {k: b[k].new_empty((n_rows, *b[k].shape[1:]))
                       for k in keys}
                big_label = lab.new_empty((n_rows, *lab.shape[1:]))
                big_wdl = w.new_empty((n_rows, *w.shape[1:]))
                if aux is not None:
                    big_aux = aux.new_empty((n_rows, *aux.shape[1:]))
                if demo is not None:
                    big_demo = demo.new_empty((n_rows, *demo.shape[1:]))
            end = n + len(buf)
            for k in keys:
                big[k][n:end].copy_(b[k])
            big_label[n:end].copy_(lab)
            big_wdl[n:end].copy_(w)
            if aux is not None:
                big_aux[n:end].copy_(aux)
            if demo is not None:
                big_demo[n:end].copy_(demo)
        n += len(buf)
        if n_rows is not None:
            del b, lab, w, aux, demo
    if n_rows is None:
        if not parts:
            raise SystemExit(f"stream_precollate: no rows with split={want_split}")
        big = {k: torch.cat([p[k] for p in parts], 0) for k in keys}
        big["relation"] = None
        out = (big, torch.cat(labels, 0), torch.cat(wdls, 0),
               torch.cat(auxes, 0) if auxes else None)
        if demo_weights is not None:
            out = (*out, torch.cat(demos, 0))
            _report_demo_join(f"stream_precollate/{want_split}", demo_miss, n)
        return out, n
    if keys is None:
        raise SystemExit(f"stream_precollate: no rows with split={want_split}")
    if n > n_rows:
        raise SystemExit(
            f"stream_precollate: OVERFILLED {n} > {n_rows} -- rows were written past the "
            f"preallocation, which means the clamp in the chunk copy failed")
    if n < n_rows:
        # Undershoot is EXPECTED, not a fault: n_rows is derived from shard_report before
        # the row filters run, while --subsample draws randomly (binomial, not exact) and
        # --decisions-only drops a further ~8.7% (rows the engine offered <2 real options
        # on). The tail of `big` is UNINITIALISED memory, so it must be truncated rather
        # than tolerated -- iterating it would train on whatever was in the allocation.
        big = {k: v[:n] for k, v in big.items()}
        big_label = big_label[:n]
        big_wdl = big_wdl[:n]
        if big_aux is not None:
            big_aux = big_aux[:n]
        if big_demo is not None:
            big_demo = big_demo[:n]
        print(f"stream_precollate: kept {n} of {n_rows} preallocated rows "
              f"({100.0 * n / n_rows:.1f}%); truncated the unwritten tail", flush=True)
    big["relation"] = None
    out = (big, big_label, big_wdl, big_aux)
    if demo_weights is not None:
        out = (*out, big_demo)
        _report_demo_join(f"stream_precollate/{want_split}", demo_miss, n)
    return out, n


def precollate_split(rows, device, aux_table, chunk=1024, demo_weights=None):
    """Collate a whole split once into GPU tensors, plus its aux targets.

    Returns (tensors, label, wdl, aux) where every value is one big tensor indexed
    by row. Bit-identical to calling collate_fast per batch -- it IS collate_fast,
    just called once over the static data instead of once per epoch.
    """
    import torch

    keys, parts, labels, wdls, auxes, demos = None, [], [], [], [], []
    demo_miss = 0   # F-7: an unjoined demonstrator weight must be counted
    for i in range(0, len(rows), chunk):
        sl = rows[i:i + chunk]
        b, lab, w = collate_fast(sl, device)
        if keys is None:
            keys = [k for k, v in b.items() if v is not None]
        parts.append({k: b[k] for k in keys})
        labels.append(lab)
        wdls.append(w)
        if aux_table is not None:
            auxes.append(aux_targets(sl, aux_table, device)[0])
        if demo_weights is not None:
            _d, _m = demo_weight_targets(sl, demo_weights, device)
            demo_miss += _m
            demos.append(_d)
    big = {k: torch.cat([p[k] for p in parts], 0) for k in keys}
    big["relation"] = None
    out = (big, torch.cat(labels, 0), torch.cat(wdls, 0),
           torch.cat(auxes, 0) if auxes else None)
    if demo_weights is not None:
        out = (*out, torch.cat(demos, 0))
        _report_demo_join("precollate_split", demo_miss, len(rows))
    return out


@torch.no_grad()
def evaluate(model, rows, device, batch=64):
    model.eval()
    correct = total = 0
    by_type = defaultdict(lambda: [0, 0])
    # Contested top-1, emitted ALONGSIDE the headline. ~9% of val rows have a
    # single legal option and are correct by construction, inflating the headline
    # by ~5 points. The headline is what metrics.jsonl shows while a run is in
    # flight -- which is exactly when decisions get made -- and the contested
    # number has until now required a separate deliberate pass. That asymmetry is
    # why comparisons keep getting stated in diluted units.
    contested_correct = contested_total = forced_total = 0
    end_tp = end_fp = end_fn = 0
    brier = 0.0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        b, label, wdl = collate_fast(chunk, device)
        _attach_relation(b)
        out = model(b)
        pred = out["policy"].argmax(-1)
        # pass rows carry label -1 and have no defined correct option; scoring
        # them as misses would understate top-1 and, worse, make this run's
        # numbers incomparable with every previous run
        scored = label >= 0
        correct += ((pred == label) & scored).sum().item()
        total += int(scored.sum().item())
        probs = out["value"].softmax(-1)
        # N_WDL, not a hardcoded 3. The value head became 2-class when draws
        # were folded into losses (measured draw base rate 0.000), and this line
        # was missed -- it crashed rank 0 at the FIRST evaluation, after a full
        # epoch of training and before any checkpoint existed, leaving the other
        # three ranks spinning on a collective that would never complete.
        onehot = F.one_hot(wdl, N_WDL_RUN or N_WDL).float()
        brier += ((probs - onehot) ** 2).sum(-1).sum().item()
        for k, row in enumerate(chunk):
            # featurize pads option_type with -1; real options are the >=0 ones
            _nreal = sum(1 for t in (row.get("option_type") or []) if t is not None and t >= 0)
            if _nreal == 1:
                forced_total += 1
            if _nreal >= 3 and label[k].item() >= 0:
                contested_total += 1
                contested_correct += int(pred[k].item() == label[k].item())
            st = row["select_type"] or 0
            by_type[st][1] += 1
            by_type[st][0] += int(pred[k].item() == label[k].item())
            types = row["option_type"]
            p_end = pred[k].item() < len(types) and types[pred[k].item()] == END_OPTION_TYPE
            t_end = label[k].item() < len(types) and types[label[k].item()] == END_OPTION_TYPE
            end_tp += int(p_end and t_end)
            end_fp += int(p_end and not t_end)
            end_fn += int(t_end and not p_end)
    model.train()
    return {
        "top1": correct / max(total, 1),
        "contested_top1": contested_correct / max(contested_total, 1),
        "contested_rows": contested_total,
        "forced_rows": forced_total,
        "rows": total,
        "brier": brier / max(total, 1),
        "end_precision": end_tp / max(end_tp + end_fp, 1),
        "end_recall": end_tp / max(end_tp + end_fn, 1),
        "end_support": end_tp + end_fn,
        "by_select_type": {str(k): {"acc": v[0] / max(v[1], 1), "n": v[1]}
                           for k, v in sorted(by_type.items())},
    }


# fam_gamma is a GAIN initialised at 1.0 meaning "pass the numeric channel
# through unchanged". AdamW's decoupled decay pulls it toward 0, i.e. "numeric
# channel off" -- the opposite of the correct prior, and it happens with no
# gradient behind it. Measured with zero gradient: 200 steps at lr 1e-3 leaves
# fam_gamma.mean() = 0.98019; over a 30k-step cosine at peak 3e-4 that is ~0.64,
# a 36% attenuation of the whole numeric input. fam_beta is a shift and decaying
# it toward 0 is harmless, but it is excluded with its partner so the FiLM
# transform is not pulled apart.
#
# Matched by SUFFIX, not exact name: by the time the optimiser is built the model
# has been through torch.compile (prefix "_orig_mod.") and DDP (prefix
# "module."), so an exact match silently catches nothing and the gains decay
# anyway.
_V2_GAIN_SUFFIXES = ("fam_gamma.weight", "fam_beta.weight")


def split_decay_params(model):
    """Return (decayed, not_decayed) as lists of (name, parameter).

    Module-level so scripts/verify_v2.py gates the SAME predicate the trainer
    runs. A gate that reimplements this rule passes whether or not the trainer
    agrees with it, which is the substitution antipattern in gate form.

    The v2 clause ADDS to the pre-existing exclusions rather than replacing
    them; with arch_v2 off the FiLM tensors do not exist and the split is
    exactly what it was before v2 landed.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # norms, biases and the learned PMA seed query are excluded; only real
        # weight matrices are decayed. The seed is (1,1,d) so ndim alone misses it.
        if (p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
                or name.endswith("seed") or name.endswith(_V2_GAIN_SUFFIXES)):
            no_decay.append((name, p))
        else:
            decay.append((name, p))
    return decay, no_decay


def uncompiled(model):
    """The module underneath a torch.compile wrapper, or `model` itself.

    The wrapper prefixes every state_dict key with "_orig_mod.", which the loaders
    (policy.architecture_from_state_dict, main_v7._arch) cannot read. Parameters are shared,
    so saving and loading through the inner module is the same weights under the real names.
    """
    return getattr(model, "_orig_mod", model)


def apply_init_from_then_num_stats(model, args, device):
    """--init-from FIRST, --num-stats SECOND, then VERIFY the file won.

    THE ORDER IS THE POINT AND IT IS NOW CHECKED, NOT MERELY INTENDED.
    num_mu / num_sigma / opt_num_mu / opt_num_sigma are registered BUFFERS, so
    they travel inside the state_dict. Running --init-from after --num-stats
    therefore copies the OLD corpus's statistics straight back over the file that
    was just installed -- and the "num-stats loaded from ..." line below still
    prints, so the log claims the new statistics while the run trains on the
    checkpoint's. Both orders look natural at the call site, which is exactly why
    a comment was not enough to keep it right.

    This lives in one function, called once, so the order is a property of a unit
    that can be driven by a test rather than of a hundred lines of main().
    """
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device)
        # Aux tensors are DROPPED, not loaded. An old 6-head checkpoint has
        # different head shapes than a 12-head model, and strict=False tolerates
        # missing keys but still raises on size mismatch. Aux heads are
        # training-only and discarded at export, so reinitialising them is
        # correct as well as necessary; the trunk is what carries over.
        # Drop aux tensors ONLY when they do not fit. The original blanket drop
        # was written for warm-starting a 12-head model from a 6-head
        # checkpoint, where the shapes genuinely differ. Applied to a resume of
        # the SAME architecture it throws away 12 trained look-ahead heads and
        # restarts them from noise, which spikes the aux term and drags the
        # trunk with it -- the opposite of continuing a run.
        want = uncompiled(model).state_dict()
        dropped = [k for k in sd
                   if k.startswith("aux.")
                   and (k not in want or want[k].shape != sd[k].shape)]
        sd = {k: v for k, v in sd.items() if k not in dropped}
        if dropped:
            print(f"dropped {len(dropped)} aux tensors on shape mismatch",
                  flush=True)
        missing, unexpected = uncompiled(model).load_state_dict(sd, strict=False)
        bad = [k for k in list(missing) + list(unexpected)
               if not k.startswith("aux.")]
        if bad:
            raise SystemExit(f"checkpoint mismatch outside aux heads: {bad[:6]}")
        print(f"warm-started from {args.init_from} "
              f"(trunk loaded, {len(missing)} aux tensors reinitialised)")

    if args.arch_v2 and args.num_stats:
        # Per-column statistics are a property of the CORPUS, so they are loaded
        # from it and shipped inside the state_dict -- serving cannot then
        # disagree with training about what "standardised" means. sigma == 0
        # marks a constant column and is passed through, never divided.
        import numpy as _np
        _st = _np.load(args.num_stats)
        _scale_in_file = str(_st["num_scale"]) if "num_scale" in _st else None
        if _scale_in_file is not None and _scale_in_file != args.num_scale:
            # Standardization runs after _scale, so raw-space stats applied to
            # post-LayerNorm data crush every column onto ~one value, silently.
            raise SystemExit(
                f"--num-stats was measured in '{_scale_in_file}' space but "
                f"--num-scale is '{args.num_scale}'; the statistics are invalid")
        model.set_num_stats(_st["mu"], _st["sigma"],
                            _st["opt_mu"] if "opt_mu" in _st else None,
                            _st["opt_sigma"] if "opt_sigma" in _st else None)
        assert_num_stats_came_from_file(model, _st, args.num_stats)
        _live = int((_st["sigma"] > 0).sum())
        _opt = ("yes" if "opt_mu" in _st else
                "NO -- option numerics stay unstandardized")
        print(f"num-stats loaded from {args.num_stats}: "
              f"{_live}/{_st['sigma'].size} state columns live (sigma>0); "
              f"option stats: {_opt}; buffers verified against the file",
              flush=True)
    return model


def assert_num_stats_came_from_file(model, st, path):
    """The live buffers must hold the FILE's statistics, not a checkpoint's.

    Printed "num-stats loaded from X" is a statement of intent; this is the
    statement of fact, and it is what a reordered --init-from would trip. mu is
    copied verbatim, so it must match exactly; sigma is passed through
    set_num_stats' SIGMA_FLOOR, which zeroes constant columns, so a column may
    legitimately read 0 where the file is below the floor -- and nothing else.
    """
    from model_ss import SixthSenseNet as _Net

    def _check(buf_name, want, floored):
        buf = getattr(model, buf_name, None)
        if buf is None:
            raise SystemExit(
                f"--num-stats {path} was applied but the model has no "
                f"{buf_name} buffer; standardization is not actually installed")
        want_t = torch.as_tensor(np.asarray(want), dtype=buf.dtype).reshape(-1)
        if floored:
            want_t = torch.where(want_t < _Net.SIGMA_FLOOR,
                                 torch.zeros_like(want_t), want_t)
        if not torch.equal(buf.detach().cpu(), want_t):
            n_bad = int((buf.detach().cpu() != want_t).sum())
            raise SystemExit(
                f"LOAD ORDER: after --init-from and --num-stats, model.{buf_name} "
                f"differs from {path} in {n_bad}/{want_t.numel()} columns. The "
                f"standardization buffers ride inside the state_dict, so this is "
                f"what a checkpoint load running AFTER set_num_stats looks like: "
                f"the log says the new statistics are in use and the run trains "
                f"on the old ones. --init-from must come first.")

    _check("num_mu", st["mu"], floored=False)
    _check("num_sigma", st["sigma"], floored=True)
    if "opt_mu" in st and "opt_sigma" in st:
        _check("opt_num_mu", st["opt_mu"], floored=False)
        _check("opt_num_sigma", st["opt_sigma"], floored=True)


def main() -> None:
    # Declared here, before argparse reads them as flag defaults: a `global`
    # statement must precede every use of the name in the same scope, and
    # ast.parse does NOT catch the violation -- only compile() does.
    global TRUNCATE_SEQ, MASK_PAD_OPTIONS, TRUNC_STATE, TRUNC_OPT
    global N_WDL_RUN
    global NUM_W_RUN, OPT_NUM_W_RUN, CARD_SEMANTICS_RUN
    global OUTCOME_MODE, OUTCOME_PARAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    # Dropout was hardcoded at the model default of 0.15 and unreachable from the
    # CLI, which quietly ruled out a whole class of experiment. Grokking is driven
    # by weight decay and shows up only when training runs far past the point the
    # training set is fit; dropout injects noise that works against the
    # memorise-then-generalise transition, so a grokking arm needs 0 here.
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model. model_ss.py records 2.2-3.9x for "
                         "the fused-SDPA attention under compile. Costs a slow first "
                         "epoch while it traces, which is irrelevant over a long "
                         "grokking run and dominant over a short one.")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast + TF32 matmuls. Training ran pure fp32, which "
                         "on Blackwell throws away most of the card. bf16 keeps fp32's "
                         "exponent range so no loss scaling is needed, and autocast "
                         "keeps master weights in fp32 -- this is the standard choice "
                         "for transformer training, not a risky one.")
    ap.add_argument("--truncate-seq", action="store_true",
                    help="slice to 128 state / 48 option instead of the padded "
                         "192/64. Measured real maxima over 89,048 rows are 112 "
                         "and 39, so this drops only padding -- but it changes "
                         "what the model attends over, so it stays opt-in.")
    ap.add_argument("--val-max-rows", type=int, default=60000,
                    help="cap validation rows. They are held as dicts (~594 KB "
                         "each), so the full 15%% of a 2.5M-row corpus is ~226 GB "
                         "and slow enough to time out the NCCL handshake. 60k "
                         "decisions is ample for a stable top-1.")
    ap.add_argument("--precollate-host", action="store_true",
                    help="stream the training split into tensors in HOST RAM "
                         "instead of GPU memory, and move each batch across. "
                         "Removes both the 594 KB/row dict ceiling in load() and "
                         "the 96 GB card limit, so the FULL corpus is trainable: "
                         "2.5M Marnie rows are ~180 GB as tensors, which fits in "
                         "880 GB of host RAM but never on one GPU.")
    ap.add_argument("--precollate", action="store_true",
                    help="collate the whole split ONCE into GPU tensors and index it "
                         "each step. The dataset is static, so re-collating the same "
                         "71,656 rows every epoch is pure waste; measured, collate is "
                         "about half of a 198s epoch even after vectorising it. Costs "
                         "roughly 5 GB of VRAM on a 96 GB card.")
    ap.add_argument("--target-val-top1", type=float, default=0.0,
                    help="stop once held-out top-1 reaches this; 0 disables")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N consecutive EVALS with no new best; 0 disables. "
                         "For grokking set this HIGH -- the premise is that "
                         "generalisation arrives long after the training set is fit, "
                         "so a short patience guarantees stopping before the thing "
                         "you were waiting for.")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--num-scale", default="layernorm", choices=["layernorm", "log1p"])
    ap.add_argument("--aux-weight", type=float, default=0.10)
    ap.add_argument("--entropy-weight", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--card-text", type=Path,
                    help="artifacts/card_text_bow.npz -- the fixed card->concept "
                         "matrix from the card-text builder (not included). Supplied-but-absent "
                         "raises rather than silently training without it, the same "
                         "rule as --lookahead, which once disabled 12 aux heads in "
                         "silence for 1.6h at a perfectly healthy top-1.")
    ap.add_argument("--card-id-dropout", type=float, default=0.0,
                    help="probability of withholding a token's CARD ID during "
                         "training (id 0 = padding = zero vector), forcing the "
                         "model to read the 64 numeric features instead of "
                         "memorising the 230 card ids it has ever seen. Aimed at "
                         "opponents playing cards outside the training corpus.")
    ap.add_argument("--arch-v2", action="store_true",
                    help="enable the v2 model path: family-conditioned numeric "
                         "projection (column meaning is family-dependent and one "
                         "shared linear cannot represent that), multi-seed pooling "
                         "(v1 feeds value + every aux head + policy from ONE "
                         "vector), and per-column standardization. Default OFF; "
                         "v2-off is bit-identical to the pre-v2 file, asserted by "
                         "scripts/verify_v2.py against a pristine module copy.")
    ap.add_argument("--pool-seeds", type=int, default=4,
                    help="pooled reads under --arch-v2: seed 0 policy, 1 value, "
                         "2+ aux. Ignored without --arch-v2.")
    ap.add_argument("--num-stats", type=Path, default=None,
                    help="npz with mu/sigma per numeric column, built from the "
                         "corpus being trained on (builder not included). "
                         "Requires --arch-v2.")
    ap.add_argument("--block-dropout", type=float, default=0.0,
                    help="dropout INSIDE the 12 transformer blocks: attention "
                         "probabilities, FFN hidden activation, and sublayer "
                         "residuals. --dropout only ever touched the embedding "
                         "(~5%% of params), which is why 0.15 vs 0.0 measured as "
                         "noise. Default 0.0 reproduces every existing run.")
    ap.add_argument("--outcome-weighting", default="off",
                    help="off | winners-only | soft:<w> | unlikelihood:<lambda>. "
                         "Weight or filter the policy loss by the demonstrator's "
                         "game result. Negative weights are NOT supported and are "
                         "not the same thing: cross-entropy is normalised by the "
                         "weight sum, so a negative weight poisons the batch mean "
                         "rather than reversing one row. Use unlikelihood for that.")
    ap.add_argument("--ckpt-every-steps", type=int, default=0,
                    help="also checkpoint every N optimiser STEPS. --ckpt-every "
                         "counts EPOCHS, and one epoch on the 8.8M-row generalist "
                         "is ~1.4h, so a 3-epoch run yields only 3 checkpoints and "
                         "a crash at 2.5 epochs loses an hour. Step checkpoints "
                         "bound that loss. They are written under the same "
                         "step-XXXXXXXX.pt naming, so nothing downstream changes.")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="also checkpoint every N epochs regardless of val top-1, "
                         "so a run keeps weights from regions where top-1 declined; "
                         "0 keeps the old best-only behaviour")
    # One WDL label per GAME is broadcast to ~78 correlated decisions, so the
    # value head has ~895 effective examples against the policy head's 69,490.
    # At equal weight it memorises and its Brier degrades every epoch.
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--value-targets", type=Path,
                    help="discounted SOFT value targets. Without this the value "
                         "head trains on the raw terminal result stamped "
                         "identically on all ~78 decisions of a game, which is "
                         "the measured cause of the degrading critic.")
    ap.add_argument("--init-from", type=Path,
                    help="warm-start from a saved state_dict. GPU 3 was unbound "
                         "by the driver mid-run and 20 epochs had no path back "
                         "into a process; this is that path.")
    ap.add_argument("--lookahead", type=Path,
                    help="lookahead_v2.jsonl.gz; without it aux falls back to "
                         "the old wdl placeholder")
    ap.add_argument("--demo-weights", type=Path,
                    help="JSON episode weights applied to the policy loss only")
    ap.add_argument("--trunc-state", type=int, default=TRUNC_STATE,
                    help="state-token cap under --truncate-seq. Measured real "
                         "maximum on the Marnie corpus is 127, so 128 truncates "
                         "0.000%% of rows; the 160 default carries a deck-family "
                         "margin that a single-deck cohort does not need.")
    ap.add_argument("--trunc-opt", type=int, default=TRUNC_OPT,
                    help="option-token cap under --truncate-seq. Real maximum "
                         "is 42, so 48 truncates nothing. 128+48 cuts attention "
                         "to 62%% and the FFN to 79%% against 160+64.")
    ap.add_argument("--relations", action="store_true",
                    help="supply the typed-edge relation matrix to attention. "
                         "Without it the encoder is PERMUTATION-INVARIANT: there is "
                         "no positional embedding either, so it cannot tell which "
                         "Tool is attached to which Pokemon. Built per batch, never "
                         "stored. A checkpoint trained with this MUST be served with "
                         "it -- see _attach_relation.")
    ap.add_argument("--decisions-only", action="store_true",
                    help="drop rows the engine offered fewer than 2 real options "
                         "on, and rows with no real label. Those score 1.0 by "
                         "construction, teach nothing, and dilute the top-1 that "
                         "selects the checkpoint.")
    ap.add_argument("--subsample", type=int, default=1,
                    help="keep every Nth row of each pre-sharded file. The "
                         "precollated tensors, not the GPUs, are the memory "
                         "ceiling; N=2 halves them. RE-SIZE --epochs to keep "
                         "total_steps comparable when you use this.")
    ap.add_argument("--mask-pad-options", action="store_true",
                    help="mask the ~58 padding option slots out of attention "
                         "and the policy softmax, so training sees the same "
                         "option count serving does (measured 5.61 real of "
                         "64 slots). Off by default: the 802.4 submission was "
                         "trained without it, so this is an arm, not a fix to "
                         "apply silently.")
    ap.add_argument("--no-tf32", action="store_true",
                    help="run matmuls at full fp32 instead of TF32. Only reason "
                         "to use this is to test TF32 as a divergence suspect.")
    ap.add_argument("--no-mask-pad-state", action="store_true",
                    help="disable PAD-state masking in pooling. Default is ON: "
                         "unmasked, 96.6%% of pooling attention lands on padding "
                         "and the board is 3.4%% of the summary vector.")
    ap.add_argument("--gate", type=Path)
    args = ap.parse_args()

    # Checked HERE, at the parser, not at the model construction site minutes
    # into a run and after DDP has already spun up four ranks. A contradiction
    # the parser can see should never cost that.
    if args.num_stats and not args.arch_v2:
        raise SystemExit("--num-stats requires --arch-v2; without it the "
                         "standardization buffers do not exist and the file "
                         "would be silently ignored")
    if args.arch_v2 and int(args.pool_seeds) < 3:
        raise SystemExit(f"--arch-v2 needs --pool-seeds >= 3 (policy/value/aux); "
                         f"got {args.pool_seeds}. With 2 the aux read aliases "
                         f"onto the policy read; with 1 the forward raises.")
    if args.arch_v2 and not args.num_stats:
        # Not fatal -- an unstandardized v2 is a legitimate ablation -- but it
        # must be LOUD. mu=0/sigma=1 makes _standardize the identity, and with
        # gamma=1/beta=0 at init the entire v2 numeric path is then bit-identical
        # to v1. The first v2 smoke run trained in exactly this state and its
        # manifest recorded arch_v2=True, which reads as "standardization on".
        print("WARNING: --arch-v2 without --num-stats. Standardization is an "
              "EXACT NO-OP (mu=0, sigma=1); only multi-seed pooling and FiLM "
              "are active. This is an ablation, not the full v2.", flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # ---- DDP: one process per GPU, launched by torchrun -------------------
    # Sharding the precollated tensors across the four cards is what makes the
    # FULL corpus trainable at all: 2.54M Marnie rows are ~180 GB, far past one
    # 96 GB card, but ~45 GB per rank across four. It also avoids shuttling every
    # batch from host RAM, which is the cost of the --precollate-host path.
    import os
    ddp_rank = int(os.environ.get("RANK", -1))
    is_ddp = ddp_rank >= 0
    if is_ddp:
        import torch.distributed as dist
        ddp_local = int(os.environ["LOCAL_RANK"])
        ddp_world = int(os.environ["WORLD_SIZE"])
        # 4 h, not the 600 s default. The default is sized for a collective that
        # hangs, but ours legitimately straddles minutes of disk work: a rank
        # reaching an all-reduce while a peer is still building tensors is normal
        # here, and at 600 s the watchdog SIGABRTs the whole job (observed:
        # `WorkNCCL(SeqNum=4, OpType=ALLREDUCE) ran for 600015 ms`). Pre-sharding
        # is what actually removes the skew; this stops a slow disk from being
        # fatal.
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(ddp_local)
        device = f"cuda:{ddp_local}"
    else:
        ddp_local, ddp_world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_main = (not is_ddp) or ddp_rank == 0
    # With --precollate-host the training rows are streamed into tensors later
    # and must NEVER be materialised as dicts here -- that list IS the 594 KB/row
    # ceiling that capped every arm at 1,500 of 26,369 games. Validation is small
    # (15% of episodes) and evaluate() still wants dict rows, so it is loaded.
    # DEFERRED under DDP. --val-max-rows caps what is KEPT, not what is PARSED:
    # rank 0 still reads all 2.5M lines to find validation rows, which takes
    # longer than NCCL's 600 s handshake and timed out two full-corpus launches.
    # The ranks therefore sync at DDP construction FIRST, and rank 0 loads
    # validation afterwards while the others wait at the first all-reduce.
    if is_ddp:
        train, val = [], []
    elif getattr(args, "precollate_host", False):
        _, val = load(args.data / "frames.jsonl.gz", skip_train=True,
                      max_val=args.val_max_rows)
        train = []
        if not val:
            raise SystemExit("empty validation split")
    else:
        train, val = load(args.data / "frames.jsonl.gz",
                          max_val=args.val_max_rows)
        if not train or not val:
            raise SystemExit("empty split")

    value_table = load_value_table(args.value_targets)
    aux_table = load_aux_table(args.lookahead)
    demo_weights = load_demo_weights(args.demo_weights)
    aux_classes = AUX_CLASSES if aux_table is not None else None
    n_aux = len(AUX_HEAD_WEIGHTS) if aux_table is not None else 6
    card_bow = None
    if args.card_text is not None:
        if not Path(args.card_text).exists():
            raise SystemExit(
                f"--card-text {args.card_text} does not exist. Refusing to train "
                f"without the card concept basis; build it with "
                f"the card-text builder (not included) or omit the flag.")
        import numpy as _np  # noqa: PLC0415
        card_bow = _np.load(args.card_text, allow_pickle=True)["bow"]
        print(f"card text basis {card_bow.shape} from {args.card_text}", flush=True)

    # ---- widths, read from the CORPUS itself -------------------------------
    # An old-architecture arm trains on a 64-wide corpus; a new one on 160-wide.
    # Taking these from model_ss's globals would build today's shape regardless
    # and pad the difference with dead columns.
    import gzip as _gz, json as _js
    with _gz.open(args.data / "frames.jsonl.gz", "rt") as _fh:
        _r0 = _js.loads(next(iter(_fh)))
    _n0 = _r0["numeric"][0]
    NUM_W_RUN = len(_n0) if isinstance(_n0, list) else len(_r0["numeric"])
    _o0 = (_r0.get("option_numeric") or [[]])[0]
    OPT_NUM_W_RUN = len(_o0) if _o0 else NUM_W_RUN
    CARD_SEMANTICS_RUN = "card_type" in _r0
    print(f"corpus widths: numeric={NUM_W_RUN} option_numeric={OPT_NUM_W_RUN} "
          f"card_semantics={CARD_SEMANTICS_RUN}", flush=True)
    # ---- value-head class count, also read from the CORPUS -----------------
    # Row 0 is enough for widths but NOT for wdl: a 3-class corpus need not show
    # a 2 in its first row. One full pass, once, at startup -- cheaper than a
    # crash 40 minutes in that surfaces as CUBLAS_STATUS_INTERNAL_ERROR.
    # Cached: under DDP this ran once PER RANK, four single-threaded passes over
    # the same gzip for one integer. Rank 0 computes, everyone reads.
    _wdl_cache = args.data / "wdl_classes.json"
    if is_main and not _wdl_cache.exists():
        _m = 0
        with _gz.open(args.data / "frames.jsonl.gz", "rt") as _fh:
            for _line in _fh:
                _w = _js.loads(_line).get("wdl")
                if isinstance(_w, int) and _w > _m:
                    _m = _w
        _wdl_cache.write_text(_js.dumps({"max_wdl": _m}))
        print(f"wdl scan: computed max_wdl={_m} -> {_wdl_cache}", flush=True)
    if is_ddp:
        # unconditional: every rank must take the same collective path
        dist.barrier()
    _maxwdl = int(_js.loads(_wdl_cache.read_text())["max_wdl"])
    N_WDL_RUN = max(N_WDL, _maxwdl + 1)
    print(f"corpus value classes: max_wdl={_maxwdl} -> n_wdl={N_WDL_RUN} "
          f"(module default {N_WDL})", flush=True)

    model = SixthSenseNet(d=args.d, layers=args.layers, heads=args.heads,
                          hidden=args.hidden, n_aux=n_aux,
                          aux_classes=aux_classes, dropout=args.dropout,
                          block_dropout=args.block_dropout,
                          card_id_dropout=args.card_id_dropout,
                          card_bow=card_bow,
                          num_w=NUM_W_RUN, opt_num_w=OPT_NUM_W_RUN,
                          n_wdl=N_WDL_RUN or N_WDL,
                          card_semantics=CARD_SEMANTICS_RUN,
                          num_scale=args.num_scale,
                          arch_v2=bool(args.arch_v2),
                          pool_seeds=int(args.pool_seeds)).to(device)
    global MASK_PAD_STATE
    TRUNCATE_SEQ = bool(args.truncate_seq)
    TRUNC_STATE, TRUNC_OPT = int(args.trunc_state), int(args.trunc_opt)
    MASK_PAD_OPTIONS = bool(args.mask_pad_options)
    global SUBSAMPLE_SEED, DECISIONS_ONLY
    SUBSAMPLE_SEED = int(args.seed)
    DECISIONS_ONLY = bool(args.decisions_only)
    global BUILD_RELATIONS
    BUILD_RELATIONS = bool(args.relations)
    print(f"BUILD_RELATIONS={BUILD_RELATIONS}", flush=True)
    print(f"SUBSAMPLE_SEED={SUBSAMPLE_SEED}  DECISIONS_ONLY={DECISIONS_ONLY}", flush=True)
    MASK_PAD_STATE = not bool(getattr(args, 'no_mask_pad_state', False))
    print(f'MASK_PAD_STATE={MASK_PAD_STATE}  MASK_PAD_OPTIONS='
          f'{MASK_PAD_OPTIONS}', flush=True)
    spec = str(args.outcome_weighting)
    OUTCOME_MODE, _, tail = spec.partition(":")
    if OUTCOME_MODE not in ("off", "winners-only", "soft", "unlikelihood"):
        raise SystemExit(f"--outcome-weighting {spec!r} is not a known mode")
    if OUTCOME_MODE in ("soft", "unlikelihood"):
        if not tail:
            raise SystemExit(f"--outcome-weighting {OUTCOME_MODE} needs a value, "
                             f"e.g. {OUTCOME_MODE}:0.25")
        OUTCOME_PARAM = float(tail)
        if OUTCOME_PARAM < 0:
            raise SystemExit("--outcome-weighting values must be non-negative; "
                             "see outcome_weight for why negatives are unsound")
    # TF32 for ALL fp32 runs, not just --amp. It was gated behind --amp, which is
    # the bf16 flag that diverged in four arms of four -- so every fp32 arm we
    # ever ran did its matmuls at the card's slow full-fp32 rate, roughly an
    # eighth of TF32 on this Blackwell part, for no benefit anyone chose.
    #
    # This does not reintroduce the bf16 risk. Those arms died with grad_norm
    # ~1e5 while clipped at 1.0, which is a dynamic-range failure (the -inf
    # option mask through softmax), not accumulated rounding. TF32 keeps fp32's
    # full 8-bit exponent and still accumulates in fp32; only the multiplier
    # mantissa narrows to 10 bits. --no-tf32 exists to settle the question by
    # experiment if an arm ever does diverge.
    # SET BOTH DIRECTIONS EXPLICITLY. This was `if not args.no_tf32: <set True>`,
    # which never turned anything OFF -- it only declined to turn it on, and then
    # inherited whatever torch defaults to. Measured in this venv 2026-08-15:
    # torch.backends.cuda.matmul.allow_tf32 defaults False (so --no-tf32 got the
    # right answer by luck) but torch.backends.cudnn.allow_tf32 defaults TRUE, so
    # every "--no-tf32" run to date has trained with cuDNN TF32 ENABLED while its
    # manifest recorded no_tf32=True. A flag whose effect depends on a library
    # default is not a control; if torch ever flips the matmul default the flag
    # becomes a total no-op silently.
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    print(f"TF32: matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn={torch.backends.cudnn.allow_tf32} "
          f"(--no-tf32={bool(args.no_tf32)})", flush=True)
    if args.compile:
        model = torch.compile(model)
    # BEFORE the DDP wrap, so set_num_stats reaches the module directly rather
    # than through .module.
    apply_init_from_then_num_stats(model, args, device)

    rep = param_report(model)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters is REQUIRED here, not defensive: the net has a
        # value head and 12 auxiliary heads, and not every head contributes a
        # gradient on every step (a head whose weight is 0, or whose targets are
        # absent for that batch). Without it DDP raises "Expected to have
        # finished reduction in the prior iteration" on the second step.
        model = DDP(model, device_ids=[ddp_local], find_unused_parameters=True)
        if is_main:
            # Prefer the pre-sharded val file. Reading it costs seconds; finding
            # the same rows inside frames.jsonl.gz means parsing all 2.5M lines,
            # which took longer than the NCCL watchdog allowed and killed the run
            # while ranks 1-3 sat at the next all-reduce.
            _, val = load(_shard_dir(args, ddp_world, "val.jsonl.gz")
                          or args.data / "frames.jsonl.gz",
                          skip_train=True, max_val=args.val_max_rows)
            if not val:
                raise SystemExit("empty validation split")
            print(f"[rank 0] validation rows: {len(val)}", flush=True)
    config = {**vars(args), "device": device, "params": rep["total"],
              "baseline_to_beat": BASELINE}
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    # Only rank 0 owns the run directory. All four ranks constructing a
    # RunRecorder for the same run_id race on its atomic manifest write and the
    # loser dies -- which is exactly how the first DDP attempt failed.
    class _NullRecorder:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    rec = RunRecorder(args.run_id, config) if is_main else _NullRecorder()
    if is_main:
        rec.record_dataset(args.data / "frames.jsonl.gz", role="tokenized_corpus")
    splits = json.loads((args.data / "splits.json").read_text())
    if is_main:
        rec.record_splits(**splits)

    _decay_named, _no_decay_named = split_decay_params(model)
    _decay = [p for _, p in _decay_named]
    _no_decay = [p for _, p in _no_decay_named]
    opt = torch.optim.AdamW(
        [{"params": _decay, "weight_decay": args.weight_decay},
         {"params": _no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    print(f"param groups: {len(_decay)} decayed / {len(_no_decay)} not decayed",
          flush=True)
    if args.arch_v2:
        _gains = [n for n, _ in _no_decay_named
                  if n.endswith(_V2_GAIN_SUFFIXES)]
        print(f"weight decay excluded from {len(_gains)} v2 FiLM tensors "
              f"(gain init 1.0 must not decay toward 0): {_gains}", flush=True)
    steps_per_epoch = max(1, math.ceil(len(train) / args.batch))
    # --epochs IS A LEARNING-RATE HYPERPARAMETER, not just a stopping condition.
    #
    # The cosine below spans total_steps, so declaring more epochs STRETCHES the
    # whole decay rather than extending training at the same schedule. Two runs
    # at the same step can therefore sit at very different learning rates.
    # Measured, at step 29,880 where the 802.4 run peaked:
    #
    #   v5-fwd   45 ep x   747 =  33,615 total   ->  LR 3.84e-05  (annealed)
    #   H1       64 ep x   747 =  47,808 total   ->  LR 1.15e-04
    #   E4       40 ep x 7,454 = 298,160 total   ->  LR 2.94e-04  (never anneals)
    #
    # E4 was launched with the same --epochs 40 that suited an 895-game corpus,
    # on one 7.6x larger. Its schedule stretched with it, so it never reached the
    # low-LR phase where v5-fwd made its final gains (0.7785 -> 0.7878). Its flat
    # validation curve was read as convergence; it had only stopped improving AT
    # THAT LR. Fixing it moved the arena result from 35.0% to 45.2% against v5-A.
    #
    # WHEN THE CORPUS SIZE CHANGES, RE-SIZE --epochs so total_steps is comparable.
    # The same coupling already caused the steps_per_epoch=1 bug noted below.
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        p = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 3e-5 + 0.5 * (args.lr - 3e-5) * (1 + math.cos(math.pi * min(p, 1.0)))

    best = {"top1": 0.0}
    stale = 0
    stop_reason = "epoch ceiling"
    step = 0
    started = time.time()
    # --precollate-host: build the training tensors straight off disk into HOST
    # RAM. The whole reason every arm was capped at 1,500 games out of 26,369 is
    # that load() holds each row as a Python dict (~594 KB) before any tensor
    # exists; 609k rows hit 362 GB RSS and never finished an epoch. Streaming
    # holds ~4k dicts at a time, and at ~71 KB/row as tensors the full 2.5M-row
    # Marnie split is ~180 GB -- far past a 96 GB card, comfortable in 880 GB of
    # host RAM. Batches move to the GPU per step.
    pre_host = bool(getattr(args, "precollate_host", False))
    if is_ddp:
        # Each rank streams ONLY its shard. With
        # pre-sharded files it reads just its own ~635k rows; without them it
        # falls back to striding over the shared file, which is correct but
        # parses the whole 2.5M-row corpus once per rank -- 4x the JSON work for
        # one copy of the data. A shard writer (not included) writes the shards, using
        # this same stride so the rows and their order are unchanged.
        shard = _shard_dir(args, ddp_world, f"train_rank{ddp_rank}.jsonl.gz")
        report = Path(args.data) / f"shards_w{ddp_world}" / "shard_report.json"
        n_rows = (json.loads(report.read_text())["per_rank"][ddp_rank]
                  if report.exists() else None)
        pre_device = torch.device("cpu") if pre_host else torch.device(device)
        # --subsample N: keep every Nth row of this rank's shard.
        # The PRECOLLATED TENSORS are the binding constraint, not the GPUs.
        # Measured on this corpus: 249,776 bytes/row at 269-wide, trunc_state
        # 160 -- so 789,752 rows/rank is 197 GB/rank and 789 GB across four
        # ranks, against 880 GB of host RAM that drops to ~796 GB when a module
        # is removed. N=2 halves it. The data cost is small and measured:
        # 2.67x less corpus cost 1.67 points.
        # n_rows MUST be divided too. It preallocates `big` (see
        # stream_precollate), so leaving it at the full count allocates the very
        # memory the subsample exists to save.
        _sub = max(1, int(getattr(args, "subsample", 1)))
        if _sub > 1 and not shard:
            raise SystemExit(
                "--subsample is only supported on the pre-sharded path; without "
                "a shard, `world` already carries the DDP stride and overloading "
                "it would silently change which rows each rank sees.")
        if _sub > 1 and n_rows:
            n_rows = (n_rows + _sub - 1) // _sub
        pre, n_pre = stream_precollate(
            shard or Path(args.data) / "frames.jsonl.gz", "train",
            pre_device, aux_table,
            rank=0 if shard else ddp_rank,
            world=(_sub if shard else ddp_world),
            n_rows=n_rows, demo_weights=demo_weights)
        # ---- _EQUALIZE ROWS ACROSS RANKS -------------------------------------
        # DDP requires every rank to run the SAME number of steps per epoch: each
        # optimizer step is a collective, so a rank with one extra batch calls an
        # all-reduce nobody joins and the job hangs -- spinning at 100% GPU and
        # 100% CPU, because NCCL busy-waits. It looks exactly like healthy work.
        # This bites because --subsample draws RANDOMLY (binomial per rank) and
        # --decisions-only drops a further ~8.7% at rank-dependent rates. Measured
        # on this corpus: 34,316 / 34,430 / 34,443 / 34,360 rows -> 358/359/359/358
        # steps at batch 96. Epoch 0 completed; the next collective deadlocked and
        # burned 4 hours before anyone noticed.
        # A modulo stride would have been exact; a random draw is not, so the row
        # count must be reconciled explicitly rather than assumed.
        if is_ddp and n_pre:
            _t = torch.tensor([n_pre], dtype=torch.long, device=device)
            dist.all_reduce(_t, op=dist.ReduceOp.MIN)
            _n_min = int(_t.item())
            if _n_min < n_pre:
                print(f"[rank {ddp_rank}] equalising {n_pre} -> {_n_min} rows so every "
                      f"rank runs the same step count", flush=True)
                # NOTE: values INSIDE the dict can be None -- stream_precollate sets
                # big["relation"] = None before returning. Slicing None raises, so the
                # None check has to be at BOTH levels, not just the outer one.
                pre = tuple(
                    ({k: (v[:_n_min] if v is not None else None) for k, v in x.items()}
                     if isinstance(x, dict) else
                     (x[:_n_min] if x is not None else None))
                    for x in pre)
                n_pre = _n_min
        if _sub > 1:
            print(f"[rank {ddp_rank}] --subsample {_sub}: keeping every {_sub}th "
                  f"row of the shard", flush=True)
        print(f"[rank {ddp_rank}] precollated {n_pre} rows onto {pre_device}",
              flush=True)
        train_n = n_pre
    elif pre_host:
        pre, n_pre = stream_precollate(
            Path(args.data) / "frames.jsonl.gz", "train",
            torch.device("cpu"), aux_table, demo_weights=demo_weights)
        print(f"stream-precollated {n_pre} rows into host RAM", flush=True)
        train_n = n_pre
    else:
        pre = (precollate_split(train, device, aux_table,
                                demo_weights=demo_weights)
               if args.precollate else None)
        if pre is not None:
            print(f"precollated {len(train)} rows onto {device}", flush=True)
        train_n = len(train)
    # Recompute now that the shard size is known. len(train) is 0 whenever the
    # rows were streamed into tensors, which silently set steps_per_epoch=1:
    # reported loss came out ~338x too high and the LR schedule spanned one step.
    steps_per_epoch = max(1, math.ceil(train_n / args.batch))
    total_steps = steps_per_epoch * args.epochs

    for epoch in range(args.epochs):
        if pre is None:
            random.shuffle(train)
        else:
            perm_device = torch.device("cpu") if pre_host else torch.device(device)
            perm = torch.randperm(train_n, device=perm_device)
        run_loss = 0.0
        ep_hits = ep_seen = 0
        ep_gnorm = 0.0
        # F-7 / same-shape sweep: rows whose demonstrator weight or soft value
        # target failed to join, counted over the epoch and reported at its end.
        ep_demo_miss = ep_demo_rows = 0
        ep_val_miss = ep_val_rows = 0
        # zero-dim device tensors, not floats: see the accumulation site below
        _z = lambda: torch.zeros((), device=device)
        ep_pol, ep_val, ep_aux, ep_ent = _z(), _z(), _z(), _z()
        ep_gn_value, ep_gn_policy = _z(), _z()
        for i in range(0, train_n, args.batch):
            if pre is None:
                chunk = train[i:i + args.batch]
                b, label, wdl = collate_fast(chunk, device)
                demo_weight = None
                if demo_weights is not None:
                    demo_weight, _m = demo_weight_targets(
                        chunk, demo_weights, device)
                    ep_demo_miss += _m
                    ep_demo_rows += len(chunk)
            else:
                idx = perm[i:i + args.batch]
                pb, plab, pwdl, paux = pre[:4]
                b = {k: (v[idx] if v is not None else None) for k, v in pb.items()}
                label, wdl = plab[idx], pwdl[idx]
                batch_aux = paux[idx] if paux is not None else None
                demo_weight = pre[4][idx] if demo_weights is not None else None
                if pre_host:
                    # tensors live in host RAM; move just this batch across
                    b = {k: (v.to(device, non_blocking=True)
                             if v is not None and hasattr(v, "to") else v)
                         for k, v in b.items()}
                    label = label.to(device, non_blocking=True)
                    wdl = wdl.to(device, non_blocking=True)
                    if batch_aux is not None:
                        batch_aux = batch_aux.to(device, non_blocking=True)
                    if demo_weight is not None:
                        demo_weight = demo_weight.to(device, non_blocking=True)
                chunk = None
            amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if args.amp and "cuda" in str(device)
                       else contextlib.nullcontext())
            with amp_ctx:
                _attach_relation(b)
                out = model(b)
            logp = F.log_softmax(out["policy"].float(), dim=-1)
            entropy = -(logp.exp() * logp).sum(-1).mean()
            # Aux heads carry the FORWARD-LOOKING signal. Each head gets its own
            # target from the replay's future; feeding all of them `wdl` (as this
            # did) made six copies of one label -- and the label the critic gate
            # proved unlearnable -- so there was no lookahead signal at all.
            aux_loss = torch.zeros((), device=logp.device)
            if aux_table is not None:
                tgt = (aux_targets(chunk, aux_table, device)[0]
                       if pre is None else batch_aux)
                for h, head_out in enumerate(out["aux"]):
                    aux_loss = aux_loss + AUX_HEAD_WEIGHTS[h] * \
                        aux_cross_entropy(head_out, tgt[:, h])
                aux_loss = aux_loss / sum(AUX_HEAD_WEIGHTS)
            else:
                for head_out in out["aux"]:
                    aux_loss = aux_loss + F.cross_entropy(
                        head_out.float(), wdl.clamp(max=head_out.shape[-1] - 1))
                aux_loss = aux_loss / max(len(out["aux"]), 1)
            if value_table is not None:
                if pre is not None:
                    raise SystemExit("--precollate does not support --value-targets yet; "
                                     "soft targets are looked up per row")
                soft, _vm = soft_value_targets(chunk, value_table, logp.device)
                ep_val_miss += _vm
                ep_val_rows += len(chunk)
                value_loss = -(soft * F.log_softmax(out["value"].float(), -1)).sum(-1).mean()
            else:
                value_loss = F.cross_entropy(out["value"].float(), wdl)
            # ignore_index=-1 drops the pass rows from the POLICY term only; the
            # value and aux heads still see them, which is the point -- the state
            # is real, only the "which option" target is undefined.
            policy_loss = policy_cross_entropy(
                out["policy"], label, outcome_weight(wdl, demo_weight))
            policy_loss = policy_loss + unlikelihood_term(
                out["policy"], label, wdl)
            loss = (policy_loss
                    + args.value_weight * value_loss
                    + args.aux_weight * aux_loss
                    - args.entropy_weight * entropy)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Per-head gradient norms, taken BEFORE clipping because
            # clip_grad_norm_ rescales in place and would flatten exactly the
            # signal we are after.
            #
            # WHY THIS EXISTS. Four of four bf16 arms diverged and the cause is
            # still not established. The leading hypothesis -- logit blow-up
            # through the -inf option mask -- was tested and REFUTED: at the
            # last checkpoint before divergence bf16 cross-entropy was 0.4868
            # against fp32's 0.4873, i.e. marginally better, and the mechanism's
            # ceiling is ln(64) = 4.159 nats against observed losses of 751 to
            # 35,263. What survives is a weight-space runaway: grad_norm leads
            # train_loss by roughly 20:1 in relative excursion at onset, the
            # ramp spans 400-800 optimizer steps, and no NaN or Inf appears
            # anywhere. Which HEAD initiates it cannot be answered from what is
            # on disk, because the per-term loss split has never been recorded.
            # These six numbers are that missing measurement.
            # Accumulated ON DEVICE and read once per epoch. Calling float() per
            # step would force six extra host syncs on every one of the 5,593
            # steps in an epoch, which is a real cost to pay for a diagnostic.
            with torch.no_grad():
                _m = model.module if is_ddp else model
                ep_gn_value = ep_gn_value + _grad_norm(_m.value)
                ep_gn_policy = ep_gn_policy + _grad_norm(_m.policy)
                ep_pol = ep_pol + policy_loss.detach()
                ep_val = ep_val + value_loss.detach()
                ep_aux = ep_aux + aux_loss.detach()
                ep_ent = ep_ent + entropy.detach()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # A non-finite grad norm means the run is already dead: clipping
            # propagates the nan/inf into every parameter on the next step, and
            # from there every later checkpoint is garbage that still looks like
            # a checkpoint. Fail here rather than burn the remaining hours.
            if not torch.isfinite(gnorm):
                raise SystemExit(
                    f"DIVERGED at epoch {epoch} step {step}: grad norm {gnorm} is "
                    f"not finite. Checkpoints before this point remain valid.")
            # TRAIN accuracy, tracked per epoch. Grokking's whole signature is
            # train pinned near 100% while validation sits flat and then starts
            # rising -- without the train curve the transition is invisible, and
            # this run exists to look for it.
            with torch.no_grad():
                # count only rows the policy loss actually trained on; pass rows
                # carry label -1 and would otherwise be scored as permanent misses
                scored = label >= 0
                ep_hits += int(((out["policy"].argmax(-1) == label) & scored).sum().item())
                ep_seen += int(scored.sum().item())
                ep_gnorm += float(gnorm)
            opt.step()
            run_loss += loss.item()
            step += 1
            # Step checkpoints bound crash loss. --ckpt-every counts EPOCHS, and
            # one epoch on 8.8M rows is ~1.4h, so a 3-epoch run has only three
            # save points. Rank 0 only: every rank writing the same weights would
            # quadruple the IO and race on the filename.
            if (args.ckpt_every_steps and is_main
                    and step % args.ckpt_every_steps == 0):
                rec.save_checkpoint(
                    step, lambda p: torch.save(
                        uncompiled(model.module if is_ddp else model).state_dict(), p))
        # Reported once per epoch, and fatal on a total miss. Without this the
        # per-batch counts would be computed and thrown away, which is the same
        # silence F-7 was about, just one level up.
        _report_demo_join(f"epoch {epoch}", ep_demo_miss, ep_demo_rows)
        if ep_val_rows and ep_val_miss:
            if ep_val_miss == ep_val_rows:
                raise SystemExit(
                    f"--value-targets joined 0 of {ep_val_rows} rows in epoch "
                    f"{epoch}. Every row would train on the base-rate prior, "
                    f"which is indistinguishable from a table that says the base "
                    f"rate. The table is keyed differently than the corpus "
                    f"(expected (episode, step)); rebuild it or drop the flag.")
            print(f"value-target join [epoch {epoch}]: "
                  f"{ep_val_rows - ep_val_miss}/{ep_val_rows} rows matched, "
                  f"{ep_val_miss} fell back to the prior", flush=True)
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            core = model.module if is_ddp else model
            # ONLY rank 0 evaluates. Every rank loading the validation split as
            # dicts would cost ~226 GB each (380k rows x 594 KB) and OOM the box;
            # only rank 0 holds it. The stop decision is then broadcast, because
            # if one rank breaks out of the loop and the others do not, the next
            # all-reduce blocks forever.
            if not is_main:
                stop_flag = torch.zeros(1, dtype=torch.uint8, device=device)
                import torch.distributed as dist
                dist.broadcast(stop_flag, src=0)
                if int(stop_flag.item()):
                    stop_reason = "stopped by rank 0"
                    break
                continue
            metrics = evaluate(core, val, device)
            rec.log_metrics(epoch=epoch, step=step,
                            train_loss=run_loss / steps_per_epoch,
                            train_top1=ep_hits / max(ep_seen, 1),
                            grad_norm=ep_gnorm / max(steps_per_epoch, 1),
                            lr=lr_at(step),
                            gen_gap=(ep_hits / max(ep_seen, 1)) - metrics["top1"],
                            # The per-term split, absent from every metrics.jsonl
                            # on disk, which is why four bf16 divergences could
                            # never be attributed to a head. gn_value/gn_policy
                            # are pre-clip.
                            loss_policy=float(ep_pol) / max(steps_per_epoch, 1),
                            loss_value=float(ep_val) / max(steps_per_epoch, 1),
                            loss_aux=float(ep_aux) / max(steps_per_epoch, 1),
                            entropy_term=float(ep_ent) / max(steps_per_epoch, 1),
                            gn_value=float(ep_gn_value) / max(steps_per_epoch, 1),
                            gn_policy=float(ep_gn_policy) / max(steps_per_epoch, 1),
                            **metrics)
            # Saving ONLY on a val-top1 record loses every checkpoint from a
            # decline, and top-1 is measurably not the ranking we care about: E3
            # ranked first on top-1 and last in the arena, and P-cohort-ed7 beat
            # v5-A 60.99% head-to-head while scoring 540.1 to its 790.1 on the
            # ladder. cohort-312 and cohort-929 peaked at epoch 4 and declined
            # for the rest of training, so no checkpoint exists anywhere in that
            # region -- if the strongest PLAYER was at epoch 8 it is unrecoverable.
            # --ckpt-every writes on a schedule as well, so the gauntlet can pick
            # the checkpoint instead of top-1 picking it.
            improved = metrics["top1"] > best["top1"]
            periodic = bool(args.ckpt_every) and epoch % args.ckpt_every == 0
            if improved:
                best = {**metrics, "epoch": epoch}
                stale = 0
            else:
                stale += 1
            if improved or periodic:
                rec.save_checkpoint(step, lambda p: torch.save(uncompiled(core).state_dict(), p))
            print(f"epoch {epoch:4d} step {step:6d} loss {run_loss/steps_per_epoch:.4f} "
                  f"tr1 {ep_hits/max(ep_seen,1):.4f} gap {(ep_hits/max(ep_seen,1))-metrics['top1']:+.4f} "
                  f"top1 {metrics['top1']:.4f} end_p {metrics['end_precision']:.3f} "
                  f"end_r {metrics['end_recall']:.3f} brier {metrics['brier']:.4f} "
                  f"stale {stale}", flush=True)
            want_stop = bool(
                (args.target_val_top1 and metrics["top1"] >= args.target_val_top1)
                or (args.patience and stale >= args.patience))
            if is_ddp:
                import torch.distributed as dist
                dist.broadcast(
                    torch.tensor([1 if want_stop else 0], dtype=torch.uint8,
                                 device=device), src=0)
            if want_stop:
                stop_reason = (
                    f"target val top1 {args.target_val_top1} reached"
                    if args.target_val_top1 and metrics["top1"] >= args.target_val_top1
                    else f"no new best for {stale} evals (patience {args.patience})")
                break

    if is_ddp and not is_main:
        import torch.distributed as dist
        dist.destroy_process_group()
        return
    status = "PASS" if best["top1"] > BASELINE else "FAIL"
    result = {"status": status, "best_top1": best["top1"], "baseline": BASELINE,
              "delta_vs_baseline": round(best["top1"] - BASELINE, 4),
              "best_epoch": best.get("epoch"), "end_precision": best.get("end_precision"),
              "end_recall": best.get("end_recall"), "brier": best.get("brier"),
              "by_select_type": best.get("by_select_type"),
              "train_rows": train_n * (ddp_world if is_ddp else 1),
              "val_rows": len(val), "ddp_world": ddp_world,
              "params": rep["total"], "elapsed_s": round(time.time() - started, 1),
              "stop_reason": stop_reason, "epochs_run": epoch + 1}
    rec.finish(result, {"beats_dense_baseline": status})
    if args.gate:
        args.gate.parent.mkdir(parents=True, exist_ok=True)
        args.gate.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
