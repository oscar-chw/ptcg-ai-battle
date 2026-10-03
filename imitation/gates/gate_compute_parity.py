#!/usr/bin/env python3
"""Does the NumPy serving forward compute the SAME FUNCTION as the trained graph?

WHY THIS EXISTS AND WHY gate_export_parity.py IS NOT ENOUGH. That gate collates
its torch side with `collate` -- which marks the whole padded width valid and
never builds a relation matrix -- so it compares the NumPy agent against a torch
model running WITHOUT the padding masks and WITHOUT the relational bias the
checkpoint was actually trained with. Both sides are wrong in the same way and
the gate reads green. Two submissions scored 256.7 and 183.1 against a 800.5
champion behind exactly that green light.

This gate rebuilds the torch side the way TRAINING ran it, read from the run
manifest:

    collate_fast      MASK_PAD_STATE, MASK_PAD_OPTIONS, TRUNCATE_SEQ/TRUNC_*
    _attach_relation  BUILD_RELATIONS -> build_relation_batch

and compares the argmax over the REAL options -- the choice the agent would send
to the engine -- against both serving forwards on identical inputs:

    main_v6._forward   the shipped 800.5 champion (no masks, no relations)
    main_v7._forward   the corrected one, driven by the SAME flags

Reporting v6 beside v7 is the point: it shows the size of the defect and it
proves the harness can measure disagreement at all. A parity gate that can only
ever print 1.0000 measures nothing.

main_v7 is run a third time on the checkpoint's own fp32 weights, because
"is the forward right" and "what did fp16 export cost" are different questions
with different answers, and averaging them hides both. See TIE_EPS below.

ABSENT IS NOT FALSE-BY-TODAY'S-DEFAULT. `no_mask_pad_state` missing from a
manifest means the flag did not exist when that run trained, and the code then
wrote mask[:s] = True unconditionally -- i.e. MASK_PAD_STATE was OFF. Reading it
as `not cfg.get("no_mask_pad_state", False)` gives True, silently reconstructing
a torch side that masks when the run did not. An absent key is answered from the
history of the flag, never from the module default.

    python gates/gate_compute_parity.py \
        --data <corpus dir holding frames.jsonl.gz> \
        --ckpt runs/<run>/checkpoints/step-XXXXXXXX.pt \
        --manifest runs/<run>/manifest.json \
        --npz <package>/weights.npz --out build/compute_parity.json \
        [--main-v6 <the earlier unmasked serving file, to measure its disagreement>]

--main-v6 is optional. Without it the gate still answers the question it exists
for (does main_v7 compute the trained function?) and reports the v6 columns as
null; with it, the size of the defect is printed beside the fix.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

# TWO QUESTIONS, TWO NUMBERS, AND WHY THE GATE ASKS THEM SEPARATELY.
#
# (1) Does main_v7 compute the SAME FUNCTION as the trained graph? That is the
#     train/serve mismatch, and it is asked with the checkpoint's own fp32
#     weights on both sides so that storage precision is not part of the answer.
#
# (2) What does SHIPPING cost? export_ss_numpy stores every tensor above 4096
#     elements as fp16, so the packaged agent runs quantised weights. That is a
#     packaging decision, not a train/serve mismatch, and it gets its own number
#     rather than being averaged into (1).
#
# WHERE THE REFERENCE STOPS BEING A REFERENCE. Some prompts offer two options the
# model scores to within a rounding error -- two copies of a card in hand are two
# option tokens with near-identical features. Which index wins is then decided by
# whichever GEMM ran, and TORCH DISAGREES WITH ITSELF: the same row scored inside
# a batch of 32 and alone in a batch of 1 can order those two options differently.
# A gate cannot demand that NumPy reproduce a choice the reference does not make.
#
# So the bar is MEASURED, never chosen. Every run scores the torch side TWICE at
# two batch sizes and reports `torch_reference_noise`, the largest disagreement
# between them. A row is DECIDABLE when the reference's own top-2 gap exceeds
# that noise, and the gated agreement is computed over decidable rows only. This
# cannot launder a broken forward: main_v6 on this same checkpoint misses by
# gaps of 1.2 to 8.7 logits, five orders of magnitude above the noise, so every
# one of its disagreements is on a decidable row and counts against it.
#
# TIE_EPS below feeds one human-readable "tie-aware" column. It is reported and
# never gated; no hand-picked epsilon appears in the pass/fail decision.
TIE_EPS = 1e-5


def _load_serving_module(name: str, path: Path, cg_dir: Path | None):
    """Import a serving file (main_vN.py) as a module.

    An older main_vN.py does `from cg.api import ...` at import time. cg_dir, the
    directory that holds the engine's cg/ package, goes on sys.path so the file
    under test is imported EXACTLY as written, rather than being copied or edited
    into something importable.
    """
    if cg_dir is not None and str(cg_dir) not in sys.path:
        sys.path.insert(0, str(cg_dir))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def training_semantics(cfg: dict) -> dict:
    """The three switches, plus the widths, exactly as this run trained them."""
    truncate = bool(cfg.get("truncate_seq") or False)
    if truncate and ("trunc_state" not in cfg or "trunc_opt" not in cfg):
        # The widths decide the sequence length, and with masking off the
        # sequence length changes the answer. Guessing today's default here is
        # how a gate certifies a graph nobody ran.
        raise SystemExit(
            "manifest sets truncate_seq but records no trunc_state/trunc_opt; "
            "the trained sequence width is unrecoverable, refusing to guess")
    return {
        # absent => the flag did not exist => the old code masked nothing
        "MASK_PAD_STATE": (not bool(cfg["no_mask_pad_state"]))
        if "no_mask_pad_state" in cfg and cfg["no_mask_pad_state"] is not None
        else False,
        "MASK_PAD_OPTIONS": bool(cfg.get("mask_pad_options") or False),
        "BUILD_RELATIONS": bool(cfg.get("relations") or False),
        "TRUNCATE_SEQ": truncate,
        "TRUNC_STATE": int(cfg["trunc_state"]) if truncate else None,
        "TRUNC_OPT": int(cfg["trunc_opt"]) if truncate else None,
    }


def _build_model(manifest: dict, state_dict: dict, model_class, aux_classes_fn):
    """Like eval_ss_per_episode._build_model, plus the two things it cannot infer.

    A champion-era checkpoint predates card semantics and carries a 3-class value
    head; the shared helper always builds today's shape (card_type/energy_type
    present, n_wdl=2) and load_state_dict rejects it. Both are readable straight
    off the state_dict, so read them rather than requiring a newer checkpoint.
    """
    cfg = manifest["config"]
    missing = [k for k in ("d", "layers", "heads", "hidden", "num_scale")
               if k not in cfg]
    if missing:
        raise SystemExit(f"manifest config is missing {missing}")
    aux_classes = aux_classes_fn(state_dict)
    kwargs = {
        "d": int(cfg["d"]), "layers": int(cfg["layers"]),
        "heads": int(cfg["heads"]), "hidden": int(cfg["hidden"]),
        "n_aux": len(aux_classes), "aux_classes": aux_classes,
        "num_scale": cfg["num_scale"],
        "card_semantics": "card_type.weight" in state_dict,
    }
    if "dropout" in cfg:
        kwargs["dropout"] = float(cfg["dropout"])
    if "card_bow" in state_dict:
        kwargs["card_bow"] = state_dict["card_bow"]
    if "num_proj.weight" in state_dict:
        kwargs["num_w"] = int(state_dict["num_proj.weight"].shape[1])
    if "opt_num.weight" in state_dict:
        kwargs["opt_num_w"] = int(state_dict["opt_num.weight"].shape[1])
    if "value.2.weight" in state_dict:
        kwargs["n_wdl"] = int(state_dict["value.2.weight"].shape[0])
    model = model_class(**kwargs)
    model.load_state_dict(state_dict)
    return model


def _fmt(value, spec: str) -> str:
    """Format a number, or 'n/a' for the v6 columns when --main-v6 was not given."""
    return "n/a" if value is None else format(value, spec)


def _collate_widths(chunk, sem):
    """The (ns, no) train_ss.collate_fast would use for THIS chunk.

    The NumPy side has to be fed the same width or the two are not comparing the
    same input: with masking off, 192 state slots and 160 are different graphs.
    """
    if sem["TRUNCATE_SEQ"]:
        return (min(sem["TRUNC_STATE"], len(chunk[0]["family"])),
                min(sem["TRUNC_OPT"], len(chunk[0]["option_type"])))
    return (max(len(r["family"]) for r in chunk),
            max(len(r["option_type"]) for r in chunk))


def _row_to_tok(row: dict, ns: int, no: int) -> dict:
    """A dataset row in the shape build_tokens returns at serve time, sliced and
    padded to the widths collate_fast used, and PAD-filled the way collate_fast
    fills (zeros for the state ids, -1 for option_type and option_ptr).

    option_type keeps its RAW -1 sentinel: main_v7 has to count real options
    BEFORE the clip to 0 that the embedding needs, which is the trap in
    train_ss.collate_fast:617.
    """
    import numpy as np

    s, o = min(len(row["family"]), ns), min(len(row["option_type"]), no)
    if s != ns or o != no:
        # Never silently pad a short row into a wide one: collate_fast marks only
        # the first s / o slots valid even with the masks off, so the two sides
        # would diverge for a reason that has nothing to do with this gate.
        raise SystemExit(
            f"row is {len(row['family'])}x{len(row['option_type'])} but collate "
            f"width is {ns}x{no}; this gate assumes fixed-width corpus rows")

    def pad1(seq, slots, fill):
        seq = list(seq)[:slots]
        return seq + [fill] * (slots - len(seq))

    def pad2(vecs, slots, width, fill=0.0):
        out = [list(v) for v in list(vecs)[:slots]]
        out += [[fill] * width for _ in range(slots - len(out))]
        return out

    num_w = len(row["numeric"][0]) if row["numeric"] else 0
    onum_w = len(row["option_numeric"][0]) if row["option_numeric"] else 0
    # collate_fast:654 -- a pointer outside the kept state width becomes -1
    ptr = np.asarray(row["option_ptr"][:o], dtype=np.int64)
    ptr = np.where((0 <= ptr) & (ptr < s), ptr, -1)
    return {
        "family": pad1(row["family"], ns, 0),
        "owner": pad1(row["owner"], ns, 0),
        "card": pad1(row["card"], ns, 0),
        "card_type": pad1(row.get("card_type") or [], ns, 0)
        if row.get("card_type") is not None else None,
        "energy_type": pad1(row.get("energy_type") or [], ns, 0)
        if row.get("energy_type") is not None else None,
        "numeric": pad2(row["numeric"], ns, num_w),
        "option_type": pad1(row["option_type"], no, -1),
        "option_numeric": pad2(row["option_numeric"], no, onum_w),
        "option_ptr": [list(p) for p in ptr] + [[-1, -1]] * (no - len(ptr)),
        "option_attack": pad1(row.get("option_attack") or [], no, 0),
    }


def _n_real_options(row: dict) -> int:
    """Options the engine actually offered. featurize pads option_type with -1."""
    return sum(1 for t in (row.get("option_type") or []) if t is not None and t >= 0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--cg-dir", type=Path,
                   default=Path(os.environ["PTCG_ENGINE_DIR"])
                   if os.environ.get("PTCG_ENGINE_DIR") else None,
                   help="directory holding the engine's cg/ package; defaults to "
                        "$PTCG_ENGINE_DIR. Only --main-v6 needs it.")
    p.add_argument("--main-v6", type=Path, default=None,
                   help="the earlier, unmasked serving file; optional")
    p.add_argument("--rows", type=int, default=500)
    p.add_argument("--min-rows", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ref-batch", type=int, default=1,
                   help="second torch batch size; the disagreement between the "
                        "two passes IS the reference's resolution")
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-agreement", type=float, default=0.999)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import numpy as np
    import torch

    import train_ss
    from train_ss import _attach_relation, collate_fast, load
    from model_ss import SixthSenseNet
    from eval_ss_per_episode import _aux_classes, _load_state_dict

    manifest = json.loads(args.manifest.read_text())
    cfg = manifest["config"]
    sem = training_semantics(cfg)

    # ---- put the TRAINER in the state this checkpoint was trained in --------
    # These are module globals, not arguments, so a caller that forgets one gets
    # the default -- which for MASK_PAD_STATE is True, i.e. silently the POST
    # 2026-08-14 graph on a champion-era checkpoint.
    train_ss.TRUNCATE_SEQ = sem["TRUNCATE_SEQ"]
    train_ss.MASK_PAD_OPTIONS = sem["MASK_PAD_OPTIONS"]
    train_ss.MASK_PAD_STATE = sem["MASK_PAD_STATE"]
    train_ss.BUILD_RELATIONS = sem["BUILD_RELATIONS"]
    if sem["TRUNCATE_SEQ"]:
        train_ss.TRUNC_STATE, train_ss.TRUNC_OPT = sem["TRUNC_STATE"], sem["TRUNC_OPT"]

    rows_all = load(args.data / "frames.jsonl.gz", skip_train=True)[1]
    if not rows_all:
        raise SystemExit(f"{args.data} has no validation rows")
    # widths come from the CORPUS, exactly as train_ss.main derives them
    train_ss.NUM_W_RUN = len(rows_all[0]["numeric"][0])
    train_ss.OPT_NUM_W_RUN = len(rows_all[0]["option_numeric"][0])

    # A single legal option cannot disagree; those rows only dilute the number
    # the gate exists to protect.
    rows = [r for r in rows_all if _n_real_options(r) >= 2][: args.rows]
    if len(rows) < args.min_rows:
        raise SystemExit(
            f"only {len(rows)} contested validation decisions available; the gate "
            f"requires at least {args.min_rows} and will not be run on fewer")

    state_dict = _load_state_dict(args.ckpt, torch)
    device = torch.device(args.device)
    model = _build_model(manifest, state_dict, SixthSenseNet,
                         _aux_classes).to(device).eval()

    # fp16 storage, fp32 compute -- the same cast main_vN._weights() does, so the
    # gate measures the shipped precision and not a friendlier one.
    _raw = np.load(args.npz)
    W = {k: _raw[k].astype(np.float32) for k in _raw.files}
    # The SAME weights torch is holding, with no fp16 round trip. Running main_v7
    # on these isolates "is the forward right" from "what did fp16 storage cost".
    W32 = {k: v.detach().cpu().numpy().astype(np.float32)
           for k, v in state_dict.items()}
    v6 = (_load_serving_module("main_v6_under_test", args.main_v6, args.cg_dir)
          if args.main_v6 else None)
    v7 = _load_serving_module("main_v7_under_test", ROOT / "serving" / "main_v7.py",
                              args.cg_dir)
    layers, heads = int(cfg["layers"]), int(cfg["heads"])

    # main_v7 is flag-driven. Drive it from the manifest -- the same source the
    # torch side above is driven from -- and separately record what it would have
    # chosen ON ITS OWN from the weights file, because that is what a packaged
    # submission gets when nobody passes flags.
    serve_mode = {"mask_pad_state": sem["MASK_PAD_STATE"],
                  "mask_pad_options": sem["MASK_PAD_OPTIONS"],
                  "relations": sem["BUILD_RELATIONS"]}
    auto = v7.resolve_mode(W)
    auto_mode = {k: auto[k] for k in v7.SERVE_FLAGS}
    auto_source = {k: auto["_source"][k] for k in v7.SERVE_FLAGS}
    auto_ok = auto_mode == serve_mode
    champion_mode = dict(v7.CHAMPION_MODE)

    def torch_pass(batch_size, collect_toks=False):
        """Reference policy logits over the real options of every row."""
        out, tk = [], []
        with torch.inference_mode():
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                ns, no = _collate_widths(chunk, sem)
                batch, _label, _wdl = collate_fast(chunk, device)
                _attach_relation(batch)
                if sem["BUILD_RELATIONS"] and batch.get("relation") is None:
                    raise SystemExit(
                        "manifest says relations=true but none was built")
                if tuple(batch["family"].shape[1:]) != (ns,) or \
                        tuple(batch["option_type"].shape[1:]) != (no,):
                    raise SystemExit(
                        f"collate produced {tuple(batch['family'].shape)} /"
                        f"{tuple(batch['option_type'].shape)}, gate predicted "
                        f"{ns}/{no}")
                policy = model(batch)["policy"].float().cpu().numpy()
                for k, row in enumerate(chunk):
                    n = _n_real_options(row)
                    out.append(policy[k, :n].astype(np.float64))
                    if collect_toks:
                        tk.append(_row_to_tok(row, ns, no))
        return out, tk

    torch_logits, toks = torch_pass(args.batch, collect_toks=True)
    torch_choice = [int(l.argmax()) for l in torch_logits]
    # THE SAME MODEL, THE SAME ROWS, A DIFFERENT BATCH SIZE. Anything these two
    # passes disagree about is a question the reference does not answer.
    torch_logits_b, _ = torch_pass(args.ref_batch)
    reference_noise = max(float(np.abs(a - b).max())
                          for a, b in zip(torch_logits, torch_logits_b))
    torch_self_disagreements = sum(
        1 for a, b in zip(torch_logits, torch_logits_b)
        if int(a.argmax()) != int(b.argmax()))

    def top2_gap(logits):
        s = np.sort(logits)
        return float(s[-1] - s[-2]) if len(s) >= 2 else float("inf")

    decidable = [top2_gap(l) > reference_noise for l in torch_logits]
    n_decidable = sum(decidable)
    if n_decidable < args.min_rows:
        raise SystemExit(
            f"only {n_decidable} of {len(rows)} rows are decidable at the "
            f"reference's own resolution ({reference_noise:.3e}); the gate needs "
            f"at least {args.min_rows} and will not certify on fewer")

    # v6 has no mode. v7 is run three ways: as trained on the shipped fp16
    # export, as trained on the checkpoint's own fp32 weights (the compute-parity
    # question), and in champion mode -- which must reproduce v6 bit for bit, or
    # "v7 is a strict superset of v6" is false and the champion could not be
    # served from this file.
    arms = (*((("v6", v6, None, W),) if v6 is not None else ()),
            ("v7", v7, serve_mode, W),
            ("v7_fp32", v7, serve_mode, W32),
            ("v7_champion_mode", v7, champion_mode, W))
    names = [a[0] for a in arms]
    agree = {n: 0 for n in names}
    strict = {n: 0 for n in names}
    dec_agree = {n: 0 for n in names}      # over decidable rows only -- the gate
    tied_gap = {n: 0.0 for n in names}
    errors = {n: 0 for n in names}
    bad: dict[str, list] = {n: [] for n in names}
    maxdiff = {n: 0.0 for n in names}
    worst_flip = {n: 0.0 for n in names}   # widest reference gap this arm missed
    v7champ_vs_v6 = 0.0
    quant_noise = 0.0          # how far fp16 storage moves a policy logit

    for idx, row in enumerate(rows):
        tok = toks[idx]
        n = _n_real_options(row)
        ref = torch_logits[idx]
        got = {}
        for name, mod, mode, weights in arms:
            try:
                kw = {} if mode is None else {"mode": mode}
                logits = np.asarray(
                    mod._forward(tok, weights, layers=layers, heads=heads, **kw),
                    dtype=np.float64)
            except Exception as exc:  # noqa: BLE001 - reported as gate evidence
                errors[name] += 1
                if len(bad[name]) < 10:
                    bad[name].append({"row": idx,
                                      "error": f"{type(exc).__name__}: {exc}"})
                continue
            got[name] = logits
            choice = int(logits[:n].argmax())
            # Logit distance, not just argmax: an argmax can agree by luck while
            # the function underneath is still the wrong one.
            maxdiff[name] = max(maxdiff[name],
                                float(np.abs(logits[:n] - ref).max()))
            if choice == torch_choice[idx]:
                strict[name] += 1
                agree[name] += 1
                if decidable[idx]:
                    dec_agree[name] += 1
                continue
            gap = float(ref.max() - ref[choice])
            worst_flip[name] = max(worst_flip[name], gap)
            if gap <= TIE_EPS:
                agree[name] += 1
                tied_gap[name] = max(tied_gap[name], gap)
            if len(bad[name]) < 10:
                bad[name].append({"row": idx, "torch": torch_choice[idx],
                                  name: choice, "n_options": n,
                                  "torch_gap": gap, "tie": gap <= TIE_EPS,
                                  "decidable": bool(decidable[idx])})
        if "v6" in got and "v7_champion_mode" in got:
            v7champ_vs_v6 = max(
                v7champ_vs_v6,
                float(np.abs(got["v7_champion_mode"] - got["v6"]).max()))
        if "v7" in got and "v7_fp32" in got:
            quant_noise = max(quant_noise,
                              float(np.abs(got["v7"][:n] - got["v7_fp32"][:n]).max()))

    compared = len(rows)
    rate = {k: agree[k] / compared for k in agree}
    rate_strict = {k: strict[k] / compared for k in strict}
    rate_dec = {k: dec_agree[k] / n_decidable for k in dec_agree}
    # (1) the train/serve mismatch, on identical fp32 weights, over the rows the
    #     reference actually decides. Strict index equality, no epsilon.
    compute_ok = (rate_dec["v7_fp32"] >= args.min_agreement
                  and errors["v7_fp32"] == 0)
    # (2) the cost of shipping fp16, held to the same bar: quantisation may
    #     reorder options the reference cannot separate, and nothing else.
    export_ok = (rate_dec["v7"] >= args.min_agreement and errors["v7"] == 0)
    passed = compute_ok and export_ok
    # A packaged submission carries no manifest. If main_v7 cannot recover this
    # mode from weights.npz alone, the fix is one re-export away and the gate says
    # so rather than certifying a file that will serve itself wrong.
    stamped = any(k.startswith("serve.") for k in W)
    mode_recoverable = auto_ok or stamped
    gate = {
        "schema": "ptcg-ai/compute-parity/v3",
        "status": "PASS" if (passed and mode_recoverable) else "FAIL",
        "gate": "the NumPy serving forward must reproduce the argmax of the torch "
                "graph AS TRAINED on identical fp32 weights; the shipped fp16 "
                "export may only reorder options it cannot resolve; and the mode "
                "must be recoverable from the weights file alone",
        "compute_parity_ok": compute_ok,
        "export_quantisation_ok": export_ok,
        "mode_recoverable": mode_recoverable,
        # ---- the gated numbers: decidable rows only --------------------------
        "agreement_v7_fp32_decidable": rate_dec["v7_fp32"],
        "agreement_v7_decidable": rate_dec["v7"],
        "agreement_v6_decidable": rate_dec.get("v6"),
        "agreement_v7_champion_mode_decidable": rate_dec["v7_champion_mode"],
        "n_decidable": n_decidable,
        # ---- how the reference's own resolution was measured -----------------
        "torch_reference_noise": reference_noise,
        "torch_self_disagreements": torch_self_disagreements,
        "ref_batch_a": args.batch,
        "ref_batch_b": args.ref_batch,
        # ---- raw, over every compared row ------------------------------------
        "agreement_v7_fp32_strict": rate_strict["v7_fp32"],
        "agreement_v7_strict": rate_strict["v7"],
        "agreement_v6_strict": rate_strict.get("v6"),
        "agreement_v7_champion_mode_strict": rate_strict["v7_champion_mode"],
        "max_abs_logit_diff_v7_fp32_vs_torch": maxdiff["v7_fp32"],
        "export_quantisation_noise": quant_noise,
        "worst_flip_gap": worst_flip,
        "min_agreement": args.min_agreement,
        "compared": compared,
        "agreed_v7_fp32_strict": strict["v7_fp32"],
        "agreed_v7_strict": strict["v7"],
        "agreed_v6_strict": strict.get("v6"),
        "agreement_v7_tie_aware": rate["v7"],
        "agreement_v6_tie_aware": rate.get("v6"),
        "tie_eps": TIE_EPS,
        "max_tied_gap_v7": tied_gap["v7"],
        "max_tied_gap_v6": tied_gap.get("v6"),
        "errors": errors,
        "max_abs_logit_diff_v7_vs_torch": maxdiff["v7"],
        "max_abs_logit_diff_v6_vs_torch": maxdiff.get("v6"),
        "max_abs_logit_diff_v7championmode_vs_v6": v7champ_vs_v6 if v6 is not None else None,
        "training_semantics": sem,
        "v7_mode_from_manifest": serve_mode,
        "v7_mode_autoresolved_from_npz": auto_mode,
        "v7_mode_autoresolve_source": auto_source,
        "v7_mode_autoresolve_matches_training": auto_ok,
        "npz_carries_serve_stamp": stamped,
        "npz_rel_bias_trained": auto["_rel_bias_trained"],
        "mode_warnings": auto["_warnings"],
        "ckpt": str(args.ckpt),
        "npz": str(args.npz),
        "manifest": str(args.manifest),
        "data": str(args.data),
        "disagreement_examples": bad,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2))
    print(json.dumps({k: v for k, v in gate.items()
                      if k != "disagreement_examples"}, indent=2))
    print(f"\nGATE {gate['status']}  {compared} contested decisions, "
          f"{n_decidable} of them decidable at the reference's own resolution "
          f"{reference_noise:.2e} (torch disagrees with itself on "
          f"{torch_self_disagreements})\n"
          f"  agreement on decidable rows   fp32 v7 {rate_dec['v7_fp32']:.4f}   "
          f"shipped v7 {rate_dec['v7']:.4f}   v6 {_fmt(rate_dec.get('v6'), '.4f')}   "
          f"v7-in-champion-mode {rate_dec['v7_champion_mode']:.4f}\n"
          f"  agreement on every row        fp32 v7 {rate_strict['v7_fp32']:.4f}"
          f"   shipped v7 {rate_strict['v7']:.4f}   v6 {_fmt(rate_strict.get('v6'), '.4f')}\n"
          f"  max|dlogit| v7 fp32 {maxdiff['v7_fp32']:.2e}   v6 "
          f"{_fmt(maxdiff.get('v6'), '.2e')}   fp16 quantisation {quant_noise:.2e}")
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
