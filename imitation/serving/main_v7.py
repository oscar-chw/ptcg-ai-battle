"""marnie-sixthsense phase-1 agent: policy-only, greedy over legal options.

v7 = v6 with the serving forward brought back into agreement with the trained
graph. main_v6.py is UNCHANGED and still ships the 800.5 champion; do not edit it.

WHAT v6 COMPUTES THAT TRAINING DOES NOT (the whole reason this file exists).
Checkpoints trained since 2026-08-14 run with --relations, --mask-pad-options and
MASK_PAD_STATE on. featurize.build_tokens pads every observation to 192 state and
64 option slots, so a live board of ~55 real tokens arrives with ~200 PADs.
Training masks those out of attention and out of the PMA pool and adds a typed
relational bias to the attention logits; v6 attends over the PADs as if they were
board entities and adds no bias at all, so the same weights are evaluated on a
different function from the one they were fit to. Two Kaggle submissions scored
256.7 and 183.1 against the champion's 800.5 through exactly that gap.

Each fix mirrors a named site in the training code:

  1. state PAD mask      train_ss.collate_fast:593-601  (family 0 is PAD)
  2. option PAD mask     train_ss.collate_fast:617-620  (option_type -1 is PAD,
                         counted BEFORE the clip to 0 that the embedding needs)
  3. typed relation bias train_ss.build_relation_batch  ->  model_ss.Attention,
                         F.embedding(rel, rel_bias) added to the attention
                         logits, PAD keys then set to the float min
  4. the PMA pools under the SAME mask (model_ss.PMA.forward)
  5. the policy softmax masks non-existent options (model_ss.forward:323)

*** THE MODE IS NOT HARDCODED. ***
Two arms exist: one trained WITHOUT any of this (the 800.5 champion, whose
manifest carries neither `relations` nor `no_mask_pad_state`) and one trained
WITH it. Serving the wrong one is the same class of bug in the other direction,
so every switch is a FLAG, resolved once per process by resolve_mode() below,
PRINTED to stderr, and readable afterwards from serve_mode(). The defaults are
the champion's -- all three off -- so an unlabelled weights.npz keeps behaving
exactly as v6 did rather than silently changing.

With all three off, v7's forward is arithmetically identical to v6's: the mask is
all-True so no additive term is built at all, and the policy mask is empty.

No search. The value head is deliberately unused — it measured Brier 0.5771 on
the held-out split of the 926-game Sixth Sense corpus. The 0.6667 three-class
uniform reference it was first compared with is the wrong one, since that data has
no draws: the class-frequency constant on the corpus's 185-game validation split
scores 0.487022, so the value head did WORSE than a zero-skill predictor (the
team's later win-probability notes say the same). Steering a search with it would
be steering by worse than noise.

Inference is hand-written NumPy. Torch is not guaranteed in the sandbox, and
both prior neural submissions from this project returned ERROR; removing the
framework removes it from the failure surface.

Fail-closed, and counted: any exception inside agent()'s guard falls back to the
lowest legal indices, because returning an illegal index forfeits the game. The
team's shipped file returned a RANDOM legal choice here with no trace, so a broken
package played uniformly at random while every offline metric stayed green (the
"20/32 packages played RANDOM behind a green gate" defect). This consolidated copy
follows the fix the PPO baseline records for main_v8 (not itself included): the
fallback is deterministic, every occurrence increments FALLBACKS, and each one is
logged to stderr with the exception, so a run's log shows how many decisions the
model did not make.
"""
import json
import os
import sys

import numpy as np

try:
    from cg.api import to_observation_class
except ModuleNotFoundError:
    # The engine is absent only when this module is imported without it, for the
    # tests of _forward. agent() is called by the engine's own runner, which has
    # `cg` on the path; None makes the parse below raise into the guard.
    to_observation_class = None

_W = None
_DECK = None
_STATE = {"turn": None, "intra": []}
FALLBACKS = 0   # decisions answered by the fallback below, not by the model
_MODE = None

# featurize.FAMILIES indices, mirrored from train_ss.FAM_ACTIVE / FAM_PROMPT.
FAM_ACTIVE, FAM_PROMPT = 3, 12
# Relation ids, from train_ss.build_relation_batch: 0 none, 1 same-owner,
# 2 same-zone, 3 option->referent, 4 option->destination, 5 active<->active,
# 6 prompt<->option.
#
# Masked-out attention keys. Torch uses torch.finfo(float32).min; in NumPy that
# value OVERFLOWS to -inf when added to a finite logit and raises a RuntimeWarning
# on every block. -1e30 underflows the softmax to exactly 0.0 just the same, which
# is the only property either side depends on.
_NEG = np.float32(-1e30)

# ---------------------------------------------------------------------------
# Serving mode
# ---------------------------------------------------------------------------
# The three switches that have to agree with training, and the champion's values.
# `None` in an override means "not stated here, ask the next source down".
SERVE_FLAGS = ("mask_pad_state", "mask_pad_options", "relations")
CHAMPION_MODE = {"mask_pad_state": False, "mask_pad_options": False,
                 "relations": False}
# Module-level override, for a packager that would rather edit one line than set
# an environment variable. Anything left None falls through to the next source.
SERVE_MODE_OVERRIDE = {"mask_pad_state": None, "mask_pad_options": None,
                       "relations": None}
# Precedence, highest first:
#   1. the `mode` argument to _forward / resolve_mode   (the parity gate uses it)
#   2. SERVE_MODE_OVERRIDE above
#   3. environment: PTCG_SERVE_MASK_PAD_STATE / _MASK_PAD_OPTIONS / _RELATIONS
#   4. serve_flags.json sitting next to weights.npz
#   5. keys stamped INTO weights.npz by export_ss_numpy.py --run-manifest
#   6. auto-detection from the weights themselves (see _autodetect)
#   7. CHAMPION_MODE
_ENV = {f: "PTCG_SERVE_" + f.upper() for f in SERVE_FLAGS}


def _path(name):
    p = name
    if not os.path.exists(p):
        p = "/kaggle_simulations/agent/" + name
    return p


def read_deck_csv():
    with open(_path("deck.csv"), "r") as fh:
        rows = fh.read().split("\n")
    return [int(rows[i]) for i in range(60)]


def _weights():
    global _W
    if _W is None:
        raw = np.load(_path("weights.npz"))
        _W = {k: raw[k].astype(np.float32) for k in raw.files}
    return _W


def _as_bool(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        return None
    return bool(value)


def _rel_bias_keys(W):
    return sorted(k for k in W if k.endswith(".attn.rel_bias"))


def _rel_bias_trained(W):
    """Was this checkpoint trained WITH the relational bias?

    Decisive, not a guess. model_ss.Attention always allocates
    rel_bias = nn.Parameter(zeros(N_RELATIONS, heads)), so the tensor is in every
    state_dict. When the trainer passes rel=None the parameter receives no
    gradient, and AdamW's decoupled weight decay is a multiply, so zero stays
    exactly zero. Measured: D1-marnie-final (relations off) has abs max 0.0 across
    all 12 blocks; A2-deck-then-teacher step-00001456 (relations on) has 0.0531.
    """
    keys = _rel_bias_keys(W)
    if not keys:
        return None                      # export predates the parameter entirely
    return any(bool(np.any(W[k] != 0.0)) for k in keys)


def _autodetect(W):
    """What the weights alone can prove about the mode they were trained in.

    relations is RECOVERABLE from the weights (see _rel_bias_trained).
    The two padding masks are NOT: they change the input, never the parameters,
    so nothing in weights.npz distinguishes them. What follows is therefore an
    INFERENCE and is labelled as one wherever it is printed: across all 200 run
    manifests in this workspace, every run with relations=true also has
    mask_pad_options=true and no_mask_pad_state=false. So a trained rel_bias
    implies both masks. The converse does NOT hold -- many runs mask options
    without relations -- which is why an untrained rel_bias infers nothing and
    falls through to CHAMPION_MODE. Stamp the flags with
    export_ss_numpy.py --run-manifest to remove the guess entirely.
    """
    trained = _rel_bias_trained(W)
    if not trained:
        return {f: None for f in SERVE_FLAGS}, None
    return ({"relations": True, "mask_pad_state": True, "mask_pad_options": True},
            "rel_bias is non-zero -> trained with --relations; masks inferred "
            "from it (every relations=true run in this workspace also set "
            "--mask-pad-options and left MASK_PAD_STATE on)")


def _sidecar():
    p = _path("serve_flags.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r") as fh:
            return json.load(fh) or {}
    except Exception:  # noqa: BLE001 - a malformed sidecar must not kill the agent
        return {}


def _stamped(W, flag):
    arr = W.get("serve." + flag)
    if arr is None:
        return None
    try:
        return bool(np.asarray(arr).reshape(-1)[0] != 0)
    except Exception:  # noqa: BLE001
        return None


def resolve_mode(W, mode=None, strict=False, log=True):
    """Decide the three switches once, record where each came from, and say so.

    `strict` turns the relation mismatch below into an exception. Tooling (the
    parity gate, a packaging script) should pass strict=True. The agent must not:
    inside the sandbox a raise here becomes a fallback move on every decision,
    which is strictly worse than a known-wrong graph.
    """
    auto, auto_why = _autodetect(W)
    sidecar = _sidecar()
    resolved, source = {}, {}
    for flag in SERVE_FLAGS:
        for value, where in (
            (_as_bool((mode or {}).get(flag)), "argument"),
            (_as_bool(SERVE_MODE_OVERRIDE.get(flag)), "SERVE_MODE_OVERRIDE"),
            (_as_bool(os.environ.get(_ENV[flag])), "env " + _ENV[flag]),
            (_as_bool(sidecar.get(flag)), "serve_flags.json"),
            (_stamped(W, flag), "weights.npz serve.* stamp"),
            (auto.get(flag), "auto-detected from weights"),
            (CHAMPION_MODE[flag], "champion default"),
        ):
            if value is not None:
                resolved[flag], source[flag] = bool(value), where
                break

    trained = _rel_bias_trained(W)
    warnings = []
    # THE MISMATCH THIS FILE EXISTS TO PREVENT. A non-zero rel_bias in the file
    # and no relation matrix in the forward is precisely the 183.1 submission.
    if trained and not resolved["relations"]:
        warnings.append(
            "weights.npz carries a TRAINED relational bias (rel_bias abs max "
            f"{max(float(np.abs(W[k]).max()) for k in _rel_bias_keys(W)):.6g}) "
            "but relations are being served OFF. The serving graph is not the "
            "graph these weights were fitted to.")
    if trained and not (resolved["mask_pad_state"] and resolved["mask_pad_options"]):
        warnings.append(
            "weights.npz carries a TRAINED relational bias but padding masks are "
            "not both on; every run in this workspace that trained relations also "
            "masked padding.")
    if resolved["relations"] and trained is None:
        warnings.append(
            "relations are ON but this export has no rel_bias tensor; the bias "
            "term is identically zero, so only the padding masks take effect.")
    if not trained and not any(resolved.values()):
        warnings.append(
            "no flags stated and rel_bias is untrained: falling back to CHAMPION "
            "defaults (no masks, no relations). If this checkpoint was trained "
            "with --mask-pad-options that CANNOT be detected from the weights -- "
            "re-export with export_ss_numpy.py --run-manifest to stamp it.")

    if strict and warnings:
        raise ValueError("serving mode mismatch:\n  - " + "\n  - ".join(warnings))
    if log:
        out = sys.stderr
        print("[main_v7] serving mode:", file=out)
        for flag in SERVE_FLAGS:
            print("    %-18s %-5s  <- %s" % (flag, resolved[flag], source[flag]),
                  file=out)
        if auto_why:
            print("    detection: " + auto_why, file=out)
        for w in warnings:
            print("[main_v7] *** MODE WARNING *** " + w, file=out)
        out.flush()
    return {**resolved, "_source": source, "_warnings": warnings,
            "_rel_bias_trained": trained}


def serve_mode(W=None):
    """The resolved mode for this process; resolves and prints on first call."""
    global _MODE
    if _MODE is None:
        _MODE = resolve_mode(_weights() if W is None else W)
    return _MODE


def _scale_num(x):
    """Signed log1p, mirroring model_ss.SixthSenseNet._scale for num_scale="log1p".

    The v5 arms all trained with --num-scale log1p, where model_ss.py:161-164
    returns sign(x)*log1p(|x|) INSTEAD of the num_norm LayerNorm -- not in
    addition to it. The stock main.py applies the LayerNorm branch
    unconditionally, because it was written for a layernorm-scaled model. Feeding
    a log1p model through the LayerNorm path scored 38% action agreement against
    its own torch weights: the same silent export drift that made two earlier
    0.78+ models collapse to 0.5726 live.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def _rms(x, w, eps=1e-6):
    return w * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _relation(fam, own, ptr, ns, no):
    """(ns,) family, (ns,) owner, (no,2) option_ptr -> (ns+no, ns+no) edge ids.

    The single-row form of train_ss.build_relation_batch, fed the same clipped
    arrays train_ss.collate_fast feeds it. THE OVERWRITE ORDER IS THE SPEC:
    same-zone overwrites same-owner, active<->active overwrites both, and the
    prompt edge is written LAST so it wins on prompt tokens. Reordering these
    silently relabels edges and the model reads a bias it was never trained with.
    """
    n = ns + no
    rel = np.zeros((n, n), dtype=np.int64)

    # --- state x state ---------------------------------------------------
    same_own = own[:, None] == own[None, :]
    same_fam = fam[:, None] == fam[None, :]
    block = np.where(same_own, 1, 0)
    block = np.where(same_fam, 2, block)          # zone beats owner
    act = fam == FAM_ACTIVE
    cross = act[:, None] & act[None, :] & ~same_own
    block = np.where(cross, 5, block)             # active-active beats both
    rel[:ns, :ns] = block

    # --- option -> pointed-at state token, symmetric ----------------------
    cols = np.arange(no) + ns
    for val, tgt in ((3, ptr[:, 0]), (4, ptr[:, 1])):
        ok = (tgt >= 0) & (tgt < ns)
        if ok.any():
            oi, ti = cols[ok], tgt[ok]
            rel[oi, ti] = val
            rel[ti, oi] = val

    # --- prompt <-> every option, LAST so it wins on prompt tokens --------
    prm = np.flatnonzero(fam == FAM_PROMPT)
    if prm.size:
        rel[np.ix_(prm, cols)] = 6
        rel[np.ix_(cols, prm)] = 6
    return rel


def _attn(x, W, p, heads, add):
    """`add` is the (heads, n, n) additive attention mask: the per-head relation
    bias on real keys, the float floor on PAD keys, or None when neither applies.
    model_ss.Attention hands the same tensor to scaled_dot_product_attention,
    which adds it to the SCALED QK logits — so it goes in after the 1/sqrt(dh)
    divide, not before."""
    n, d = x.shape
    dh = d // heads
    qkv = x @ W[p + ".attn.qkv.weight"].T
    q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    q = q.reshape(n, heads, dh)
    k = k.reshape(n, heads, dh)
    v = v.reshape(n, heads, dh).transpose(1, 0, 2)
    q = _rms(q, W[p + ".attn.q_norm.weight"]).transpose(1, 0, 2)
    k = _rms(k, W[p + ".attn.k_norm.weight"]).transpose(1, 0, 2)
    logits = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh)
    if add is not None:
        logits = logits + add
    y = (_softmax(logits) @ v).transpose(1, 0, 2).reshape(n, d)
    return y @ W[p + ".attn.out.weight"].T


def _swiglu(x, W, p):
    g = x @ W[p + ".ffn.gate.weight"].T
    u = x @ W[p + ".ffn.up.weight"].T
    return (_silu(g) * u) @ W[p + ".ffn.down.weight"].T


def _arch(W):
    """Read the block count and head count from the weights themselves.

    `layers=12, heads=12` were hardcoded defaults that happened to be right for
    the d384 v5-A model and are wrong for anything else. On a d512/8-head
    checkpoint, q.reshape(n, 12, 512 // 12) asks for 12*42 = 504 elements out of
    512 and raises; the exception is swallowed by agent()'s fail-closed guard, so
    the agent plays a RANDOM LEGAL MOVE on every single decision while every
    offline metric stays green. Both counts are recoverable exactly:
    q_norm is a per-head RMSNorm so its width IS head_dim, and the block count is
    simply how many blocks.<i> prefixes the export carries.
    """
    layers = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("blocks."))
    d = W["emb_norm.weight"].shape[0]
    heads = d // W["blocks.0.attn.q_norm.weight"].shape[0]
    return layers, heads


def _forward(tok, W, layers=None, heads=None, mode=None):
    if layers is None or heads is None:
        layers, heads = _arch(W)
    m = serve_mode(W) if mode is None else resolve_mode(W, mode=mode, log=False)
    ns = len(tok["family"])
    no = len(tok["option_type"])
    fam_raw = np.asarray(tok["family"], dtype=np.int64)
    # RAW option_type: PAD is -1 and the clip below turns it into 0, which is
    # OptionType.NUMBER, a real type. Counting after the clip marks all 64 slots
    # real and the mask does nothing -- the exact trap train_ss.collate_fast:617
    # calls out.
    otype_raw = np.asarray(tok["option_type"], dtype=np.int64)
    # train_ss.collate_fast:593-601 / 617-620. When a flag is off, training marked
    # the WHOLE padded width valid, so serving must too.
    n_state_real = int((fam_raw > 0).sum()) if m["mask_pad_state"] else ns
    n_opt_real = int((otype_raw >= 0).sum()) if m["mask_pad_options"] else no

    fam = np.clip(fam_raw, 0, W["family.weight"].shape[0] - 1)
    own = np.clip(np.asarray(tok["owner"], dtype=np.int64),
                  0, W["owner.weight"].shape[0] - 1)
    card = np.clip(np.asarray(tok["card"], dtype=np.int64),
                   0, W["card.weight"].shape[0] - 1)
    # featurize pads every block to its own NUMERIC_WIDTH (32); the trained
    # weights define the real widths. Truncate to what the model expects rather
    # than trusting the producer's padding — they disagree for option numerics.
    nw = W["num_proj.weight"].shape[1]
    num = np.zeros((len(fam), nw), dtype=np.float32)
    for i, row in enumerate(tok["numeric"][:ns]):
        num[i, :min(len(row), nw)] = row[:nw]
    x = (W["family.weight"][fam] + W["owner.weight"][own] + W["card.weight"][card]
         + _scale_num(num)
         @ W["num_proj.weight"].T + W["num_proj.bias"])
    # Card SEMANTICS (cardType / energyType). Gated on the weight being present so
    # this file still serves checkpoints exported before these tables existed --
    # and gated on the featuriser emitting them, so a mismatch degrades to the old
    # behaviour instead of raising inside the sandbox. Torch adds these to x the
    # same way; if either side skips them the export-parity gate fails loudly.
    if "card_type.weight" in W and tok.get("card_type") is not None:
        ct = np.clip(np.asarray(tok["card_type"][:ns], dtype=np.int64),
                     0, W["card_type.weight"].shape[0] - 1)
        x = x + W["card_type.weight"][ct]
    if "energy_type.weight" in W and tok.get("energy_type") is not None:
        et = np.clip(np.asarray(tok["energy_type"][:ns], dtype=np.int64),
                     0, W["energy_type.weight"].shape[0] - 1)
        x = x + W["energy_type.weight"][et]
    # Card rules-text concepts. card_bow is a BUFFER, so export_ss_numpy carries
    # it in weights.npz alongside the learned word vectors and nothing extra has
    # to be shipped. Torch computes (card_bow @ word) then gathers by card id;
    # this must match exactly or the export-parity gate fails.
    if "card_bow" in W and "word.weight" in W:
        x = x + (W["card_bow"].astype(np.float32)
                 @ W["word.weight"].astype(np.float32))[card]

    # collate clamps the raw -1 PAD option type to row 0. NumPy negative
    # indexing selected the last embedding row and poisoned every later block.
    ot = np.clip(otype_raw, 0, W["opt_type.weight"].shape[0] - 1)
    ow = W["opt_num.weight"].shape[1]
    onum = np.zeros((len(ot), ow), dtype=np.float32)
    for i, row in enumerate(tok["option_numeric"][:no]):
        onum[i, :min(len(row), ow)] = row[:ow]
    opt = (W["opt_type.weight"][ot]
           + _scale_num(onum)
           @ W["opt_num.weight"].T + W["opt_num.bias"])

    # Attack embedding. model_ss.py:178-181 adds attack_proj(attack(option_attack))
    # to every option, "two attacks with equal cost and damage were previously
    # identical". main.py never applied it, so the exported agent was missing an
    # entire input and disagreed with its own torch weights on 37% of decisions.
    # featurize.attack_tok already emits these ids; nothing upstream needed changing.
    atk = tok.get("option_attack")
    if atk is not None and len(atk):
        a = np.zeros(len(ot), dtype=np.int64)
        take = min(len(ot), len(atk))
        a[:take] = np.clip(np.asarray(atk[:take], dtype=np.int64),
                           0, W["attack.weight"].shape[0] - 1)
        opt = opt + W["attack.weight"][a] @ W["attack_proj.weight"].T
    ptr = np.asarray(tok["option_ptr"][:no], dtype=np.int64)
    n_state = x.shape[0]
    for j, (src, dst) in enumerate(ptr):
        if 0 <= src < n_state:
            opt[j] += x[src] @ W["ptr_src.weight"].T
        if 0 <= dst < n_state:
            opt[j] += x[dst] @ W["ptr_dst.weight"].T

    h = np.concatenate([x, opt], axis=0)
    h = _rms(h, W["emb_norm.weight"])

    # --- the mask and the relation matrix, built once for every block --------
    mask = np.zeros(n_state + no, dtype=bool)
    mask[:n_state_real] = True
    mask[n_state:n_state + n_opt_real] = True
    keep = mask[None, None, :]          # broadcast over head and query: KEYS only
    masked = not bool(mask.all())
    rel = _relation(fam, own, ptr, n_state, no) if m["relations"] else None
    for i in range(layers):
        p = "blocks.%d" % i
        rb = W.get(p + ".attn.rel_bias") if rel is not None else None
        if rb is None and not masked:
            # Nothing to add. This is the champion path and it is arithmetically
            # identical to main_v6 -- no epsilon, no reordered sum.
            add = None
        elif rb is None:
            add = np.where(keep, np.float32(0.0), _NEG)
        else:
            # F.embedding(rel, rel_bias).permute(0, 3, 1, 2) then masked_fill:
            # PAD keys are REPLACED by the floor, not offset by it.
            add = np.where(keep, rb[rel].transpose(2, 0, 1), _NEG)
        h = h + _attn(_rms(h, W[p + ".n1.weight"]), W, p, heads, add)
        h = h + _swiglu(_rms(h, W[p + ".n2.weight"]), W, p)
    h = _rms(h, W["final.weight"])

    d = h.shape[1]
    dh = d // heads
    q = W["pool.seed"].reshape(1, heads, dh).transpose(1, 0, 2)
    k = (h @ W["pool.k.weight"].T).reshape(-1, heads, dh).transpose(1, 0, 2)
    v = (h @ W["pool.v.weight"].T).reshape(-1, heads, dh).transpose(1, 0, 2)
    # model_ss.PMA masks the pooling logits with the SAME state+option mask. An
    # unmasked pool averages ~200 PAD tokens into the state vector that every
    # option logit is conditioned on.
    pool_logits = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh)
    if masked:
        pool_logits = np.where(keep, pool_logits, _NEG)
    pooled = _softmax(pool_logits) @ v
    state = pooled.transpose(1, 0, 2).reshape(1, d) @ W["pool.out.weight"].T

    opt_h = h[n_state:]
    joined = np.concatenate([opt_h, np.repeat(state, opt_h.shape[0], axis=0)], axis=1)
    hid = _silu(joined @ W["policy.0.weight"].T + W["policy.0.bias"])
    logits = (hid @ W["policy.2.weight"].T + W["policy.2.bias"]).reshape(-1)
    # "An option that does not exist has no logit at all" (model_ss.forward:323,
    # masked_fill(~option_mask)). n_opt_real is the full width when option masking
    # is off, so this is a no-op in champion mode.
    logits[n_opt_real:] = _NEG
    return logits


# ---------------------------------------------------------------------------
# minCount == 0: "you may pass"
# ---------------------------------------------------------------------------
# v6 wrote `lo = max(1, minCount or 0)`, so the empty action the engine offers was
# unreachable. v7 keeps the 0 and routes it through ONE named constant, because
# what to do with it is a judgement the code should state out loud.
#
# MEASURED on the elite1150 corpus -- 1,042,568 decisions from 8,943 episodes,
# with the ACTING seat's OWN rating >= 1150.0 on every row (min 1150.002, max
# 1343.92). Per-seat rating is the point: highelo_gold is filtered on
# max(seat_score) >= 1150 per EPISODE (tools/harvest_days.py:151), so a statistic
# taken over both of its seats mixes a gold pilot with whatever opponent he drew.
#     tools/measure_select_stats.py --out-prefix analysis/elite1150 \
#         --src <elite1150 corpus csv>
#         minCount == 0 on 145,834 decisions        13.99%
#         of those the pilot PASSED on   6,127       4.20%
#     Pass-legality is confined to select_type 1; no other type is ever optional.
#     minCount is NEVER absent -- 0 of 22,169 prompts over 150 episodes omit it --
#     so the "absent means 1" branch in agent() is defensive, not load-bearing.
#
# PASSING TRACKS HOW LITTLE IS ON OFFER, not the position:
#     n_opt  1 16.58%   2 6.40%   3 2.65%   4 1.61%   5 0.53%
#            6  0.13%   7 0.04%   8 0.04%  >=9 0.02%
# Sweeping every (n_opt, context, maxCount) cell with n >= 30, NOT ONE has a pass
# majority; the most pass-heavy cell in the corpus (n_opt 1, context 5, n=2,968)
# reaches 43.83%. Acting is the majority answer in every observable situation.
#
# DOES THE MODEL KNOW BETTER? Measured, not assumed. Real pass-legal prompts were
# replayed through THIS FILE's own forward on each package's own featurizer and
# split by what the pilot did -- tools/measure_pass_logits.py then
# tools/analyse_pass_logits.py:
#
#     marnie-sixthsense, champion mode   |  A2-deckcold-then-teacherflg, rel ON
#     2,025 pass / 8,030 act rows        |  990 pass / 3,993 act rows
#       max logit PASS mean +12.76 sd 4.06 |    max logit PASS mean -2.67 sd 2.14
#       max logit ACT  mean +11.65 sd 3.54 |    max logit ACT  mean -0.21 sd 2.15
#       AUC 0.4246                         |    AUC 0.7923
#
# THREE INDEPENDENT REASONS NO SINGLE SCALAR SURVIVES THAT TABLE.
#  1. THE SCALE IS NOT A PROPERTY OF THE GAME. One checkpoint puts its pass-legal
#     logits near +12, the other near -1. A floor of +10 never fires on the first
#     and fires on EVERY prompt on the second. The policy loss is a softmax over
#     options and is shift-invariant -- adding a constant to every option logit at
#     a prompt changes no loss -- so nothing in training pins the absolute level
#     this constant is compared against. The 14-point gap is that
#     non-identifiability made visible, and it moves with every re-export.
#  2. THE SIGN IS NOT STABLE. AUC 0.4246 is BELOW 0.5: on the champion, "pass when
#     the model dislikes everything" is worse than a coin flip, because the pilots
#     passed where its max logit was HIGHER. On A2 the same rule points the right
#     way. Whichever sign is chosen is wrong on one of the two checkpoints.
#  3. WHERE THE SIGNAL IS REAL IT STILL NEVER PAYS. A2's AUC 0.7923 is genuine, and
#     its best operating point over EVERY threshold is precision 45.11% at recall
#     2.93% -- 29 true, 6 false, firing on 0.037% of all decisions. Below 50%
#     precision a rule that fires is wrong more often than right, and any threshold
#     with useful recall is far worse (12.78% precision at 52.22% recall). The
#     reversed rule was measured too: 12.66% precision at 7.16% recall on the
#     champion. Neither direction on either checkpoint reaches 50% anywhere.
#
# Base rates and sweeps agree: the best rule this evidence supports is "always
# act", which is exactly PASS_LOGIT_FLOOR = None. Setting it to a float F makes the
# agent pass when max(logit) < F; leaving it None means take one option whenever
# one is offered.
#
# WHY THE MODEL CANNOT ARBITRATE IT. build_ss_dataset_par.py:163 keeps pass rows,
# but train_ss.collate_fast:627-637 gives them label -1 and the policy loss masks
# those rows out, so NO gradient ever taught this model when passing is right.
#
# WHAT WOULD SETTLE IT, none of which is a number to be guessed here: a pass row
# carrying a real label (a trained pass logit, or an explicit pass entry in the
# option list -- 0 of 3,847 pass-legal prompts carry one today), or a behavioural
# gate over >= 200 games against `--b heuristic` scoring WIN RATE rather than
# agreement, which is the only instrument that can price an asymmetric mistake.
# Until one exists this stays None. An absent number must stay absent, not default
# to something plausible.
PASS_LOGIT_FLOOR = None


# ---------------------------------------------------------------------------
# HOW MANY to take when the engine allows a range
# ---------------------------------------------------------------------------
# v6 took `order[:hi]` -- the maximum, always. v7's first cut took `lo` -- the
# minimum, always. Both are constants, and the corpus says they are not equally
# wrong. MEASURED on the elite1150 corpus (1,042,568 decisions, 8,943 episodes,
# the ACTING seat's own rating >= 1150.0 on every row), over the 32,334 decisions
# -- 3.10% -- where minCount < maxCount and maxCount > 1, scored on HELD-OUT
# episodes (episode_id % 5 == 0, 6,219 rows) so no table is scored on its own fit:
#
#     tools/measure_select_stats.py --src .../elite1150_marnie.csv \
#         --out-prefix analysis/elite1150
#     tools/measure_rules.py --cardinality analysis/elite1150_cardinality.jsonl \
#         --passlegal analysis/elite1150_passlegal.jsonl --out analysis/rules.json
#
#         v6, always maxCount     3,416 / 6,219 exact-k   54.93%
#         v7, always minCount     1,084 / 6,219 exact-k   17.43%
#         the table below         3,503 / 6,219 exact-k   56.33%
#
# Taking the minimum is not the safe constant. It is a three-times-worse one, and
# shipping it was a regression against the champion on this slice.
#
# WHAT THE PILOTS DID, per (minCount, maxCount), on the training episodes:
#     (0,2)  n=12,273   k=2 74.5%                          -> the CAP
#     (0,3)  n= 2,452   k=1 35.8%  k=3 34.5%  k=2 25.7%    -> flat
#     (0,4)  n= 1,766   k=1 33.0%  k=2 27.8%  k=4 27.4%    -> flat
#     (0,5)  n= 8,949   k=5 41.0%  k=4 22.9%  k=3 18.7%    -> the CAP
#     (1,2)  n=   291   k=2 88.3%                          -> the CAP
#     (1,3)  n=   202   k=1 99.0%                          -> the FLOOR
#     (1,4)  n=   137   k=1 97.1%                          -> the FLOOR
#     (1,5)  n=    45   k=1 95.6%                          -> the FLOOR
# The answer inverts inside one corpus: at maxCount 2 the pilots take everything,
# and at minCount 1 with maxCount >= 3 they take exactly one thing. No single
# constant can express that. That is the finding, and the table is its minimum
# expression -- not a model of the game.
#
# WHY NOT THE PER-CELL MODE EVERYWHERE. (0,3) and (0,4) are near-ties (35.8 vs
# 34.5, 33.0 vs 27.4) and their training modes both LOSE to the cap on held-out
# episodes (34.6% vs 36.7%, 28.8% vs 30.4%). Fitting them would be fitting noise,
# so they keep the cap; only cells whose evidence is decisive override it.
#
# WHAT THIS DOES NOT FIX. 56.33% exact-k is not a solved problem. The residual
# sits in (0,3), (0,4) and (0,5), where the count plainly depends on the board
# rather than on the bounds -- a cardinality head is the real answer and this
# table is not a substitute for one. See DEFECTS.md I-4.
#
# UNMEASURED BOUNDS ARE NOT GUESSED. A (minCount, maxCount) pair absent here never
# occurred in 1,042,568 decisions. Those keep v7's existing floor rather than
# inheriting a number fitted to a different prompt.
SELECT_COUNT_BY_BOUNDS = {
    (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5,
    (1, 2): 2,
    (1, 3): 1, (1, 4): 1, (1, 5): 1,
}


def agent(obs_dict):
    global _DECK, FALLBACKS
    # to_observation_class raises KeyError on any missing field (e.g. 'logs').
    # It MUST NOT sit outside the guard: an exception escaping agent() returns
    # no action at all, which is how a submission comes back ERROR rather than
    # merely playing badly. Read the raw dict as the fallback.
    try:
        obs = to_observation_class(obs_dict)
        sel_raw = obs.select
    except Exception:  # noqa: BLE001
        obs = None
        sel_raw = obs_dict.get("select")
    if sel_raw is None:
        _STATE["turn"], _STATE["intra"] = None, []
        if _DECK is None:
            _DECK = read_deck_csv()
        return _DECK

    sd = obs_dict.get("select") or {}
    n_opt = len(sd.get("option") or [])
    if n_opt == 0:
        return []
    # minCount 0 is honoured as 0. An ABSENT minCount is not permission to pass,
    # so it still floors at 1 exactly as v6 did.
    mc = sd.get("minCount")
    lo = 1 if mc is None else max(0, int(mc))
    hi = max(lo, int(sd.get("maxCount") or 1))
    # The ENGINE's bounds, captured BEFORE the clip to n_opt. SELECT_COUNT_BY_BOUNDS
    # is keyed on what the engine stated, which is what the corpus was keyed on;
    # looking it up with a maxCount already clipped to the option count reroutes a
    # (1,3) prompt offering 2 options into the (1,2) cell, whose measured answer is
    # the opposite one.
    bounds = (lo, hi)
    hi = min(hi, n_opt)
    lo = min(lo, hi)
    try:
        sys.path.insert(0, os.path.dirname(_path("featurize.py")))
        from collections import Counter

        from featurize import build_tokens
        if _DECK is None:
            _DECK = read_deck_csv()
        cur = obs_dict.get("current") or {}
        # SEAT IS NOT DEFAULTABLE. `cur.get("yourIndex", 0)` returned seat 0 --
        # a REAL seat -- when the key was absent, so playing seat 1 with a seat-0
        # featurization silently mirrors the board: the model reads the opponent's
        # side as its own and then picks confidently from it. That is worse than
        # random and indistinguishable from working.
        #
        # Raising is safe HERE specifically, and only here: this whole block sits
        # inside the guard below, whose documented behaviour is a counted, legal
        # fallback move. So absence costs one badly-played decision instead of a whole game
        # played from a mirrored board. An exception escaping agent() would be an
        # ERROR, which is why the outer read at the top of this function must keep
        # its bare except.
        #
        # Measured over data/decision_corpus_v4.jsonl.gz: 954,217 of 954,217
        # `current` objects carry yourIndex (100.0000%), so this never fires on
        # real engine output.
        if "yourIndex" not in cur:
            raise KeyError(
                "observation['current'] has no 'yourIndex'; seat 0 is a real "
                "seat, so defaulting would featurize a mirrored board"
            )
        seat = cur["yourIndex"]
        if cur.get("turn") != _STATE["turn"]:
            _STATE["turn"], _STATE["intra"] = cur.get("turn"), []
        tok = build_tokens(obs_dict, seat, Counter(_DECK), _STATE["intra"])
        logits = _forward(tok, _weights())
        if lo == 0 and PASS_LOGIT_FLOOR is not None and \
                float(logits[:n_opt].max()) < PASS_LOGIT_FLOOR:
            return []           # legal only because the engine said minCount == 0
        # HOW MANY -- measured per (minCount, maxCount), see SELECT_COUNT_BY_BOUNDS.
        # Clamped to >= 1 here so no table cell can ever become a back-door pass:
        # passing is the ONE branch above, and nothing else may return [].
        take = SELECT_COUNT_BY_BOUNDS.get(bounds, max(lo, 1))
        take = min(max(take, lo, 1), hi, n_opt)
        order = list(np.argsort(-logits[:n_opt]))
        picked = [int(i) for i in order[:take]]
        chosen = (sd.get("option") or [])[picked[0]]
        # UNKNOWN IS -1, NOT 0. option type 0 is OptionType.NUMBER, a real type
        # that occurs in 0.02% of corpus option slots, so recording 0 for an
        # option that never declared a type files it under a genuine, rare label
        # where no distribution check can see it. -1 is the sentinel the
        # featurizer already uses for "no option here" (featurize.py:1573), and
        # it falls outside `range(len(OptionType))`, so an unknown is DROPPED from
        # the intra-turn counts rather than miscounted into a real bucket.
        #
        # Raising is not an option on this line the way it is for the seat above:
        # `picked` is already computed and legal, and discarding a good move over
        # a bookkeeping gap would trade a real decision for a fallback one.
        # Measured 6,438,967 of 6,438,967 engine options carry "type", so this
        # sentinel is unreachable on real engine output.
        _STATE["intra"].append({
            "type": chosen.get("type", -1) if isinstance(chosen, dict) else -1,
            "card": chosen.get("cardId") if isinstance(chosen, dict) else None,
        })
        return picked
    except Exception as err:  # noqa: BLE001
        # Fail closed: an illegal index forfeits. cg.api validates
        # minCount <= len(action) <= maxCount, and lo <= hi <= n_opt, so this count
        # is always legal. It floors at 1 for the same reason the success path does:
        # passing is not the safe default here. Counted and logged, never silent.
        FALLBACKS += 1
        print(f"FALLBACK {FALLBACKS}: {type(err).__name__}: {err}", file=sys.stderr)
        return list(range(min(max(lo, 1), n_opt)))
