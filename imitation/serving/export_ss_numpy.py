#!/usr/bin/env python3
"""Export the trained checkpoint to a NumPy weight file for CPU inference.

The submission sandbox is not guaranteed to have Torch, and this project's
prior neural submissions both returned ERROR. Shipping NumPy arrays plus a
hand-written forward removes the framework from the failure surface entirely.

fp16 storage, fp32 compute. A 45,435,348-parameter export measured 80.4 MB
(ppo/docs/BASELINE.md), inside the 200 MB package limit.

    python serving/export_ss_numpy.py --ckpt runs/<run>/checkpoints/step-XXXX.pt \
        --out <package>/weights.npz --run-manifest runs/<run>/manifest.json

--run-manifest STAMPS THE SERVING MODE into the npz (three `serve.*` scalars).
main_v7.py reads them and serves the padding masks and the relational bias the
way this run trained them; without a stamp it falls back to auto-detection and
then to the champion defaults, and --mask-pad-options in particular leaves NO
trace in the weights and cannot be recovered any other way. Optional, and off by
default, so an existing export reproduces byte for byte.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SERVE_FLAGS = ("mask_pad_state", "mask_pad_options", "relations")


def serve_flags_from_manifest(manifest_path: Path, *,
                              assume_legacy: bool = False) -> dict:
    """The three serving switches as this run actually trained them.

    THE OLD JUSTIFICATION FOR DEFAULTING WAS TESTED AND IS FALSE. It read: "an
    ABSENT key means the flag did not exist yet, so the code of the day did the
    un-flagged thing". That is an era argument, and the eras do not separate.
    Measured across all 221 manifests in runs/, sorted by started_utc:

        relations          declared in  41/221; the FIRST run declaring it is
                           RELD-smoke (2026-08-14T15:10:37, relations=True), and
                           7 manifests that OMIT the key started LATER -- four of
                           them (RELD-ctlBASE, RELD-ctlBASE2, RELD-ilB1, RELD-ilB2)
                           within FIVE MINUTES of it, in the same ablation family.
        no_mask_pad_state  declared in  46/221, 3 later omissions
        mask_pad_options   declared in 119/221
        num_scale          declared in 208/221

    So absence is not an era signal; it varies between sibling runs minutes apart.
    Worse, the two sides disagree about what absence MEANS: this function read a
    missing `no_mask_pad_state` as mask_pad_state=False, while train_ss.py:1539
    reads the same missing flag as `not bool(getattr(args, ..., False))` = **True**.
    For the 175 manifests lacking the key those two statements cannot both hold,
    and the stamp is the one that ships.

    That could not be measured any further: this workspace is not a git
    repository, so what "the code of the day" actually did is not recoverable.
    An unverifiable default that decides the serving graph is a MUST-RAISE.

    `assume_legacy=True` restores the old behaviour for genuinely champion-era
    checkpoints, which must stay exportable -- but it has to be NAMED, which is
    the same shape as the packager's --legacy-tags and --no-stamp opt-outs.
    """
    cfg = json.loads(Path(manifest_path).read_text())["config"]
    keys = ("no_mask_pad_state", "mask_pad_options", "relations")
    absent = [k for k in keys if k not in cfg]
    if absent and not assume_legacy:
        raise ValueError(
            f"{manifest_path} config omits {absent}; these decide the SERVING "
            f"graph and are stamped into weights.npz, where nothing downstream "
            f"can tell a stamped guess from a stamped fact. Absence is not an "
            f"era signal (RELD-ctlBASE omits 'relations' five minutes after "
            f"RELD-smoke declares it), and train_ss.py reads a missing "
            f"'no_mask_pad_state' as mask_pad_state=True while this exporter "
            f"read it as False. Re-record the manifest, or pass "
            f"--assume-legacy-serve-flags to state that this really is a "
            f"pre-flag checkpoint."
        )
    nmps = cfg.get("no_mask_pad_state")
    return {
        "mask_pad_state": (not bool(nmps)) if nmps is not None else False,
        "mask_pad_options": bool(cfg.get("mask_pad_options") or False),
        "relations": bool(cfg.get("relations") or False),
    }


def main() -> None:
    # numpy and torch are imported here so that serve_flags_from_manifest, the
    # part that decides what gets stamped, can be tested with the standard
    # library alone.
    import numpy as np
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--run-manifest", type=Path,
                    help="stamp the serving mode from this run's manifest.json")
    ap.add_argument("--assume-legacy-serve-flags", action="store_true",
                    help="the manifest predates the serving flags, so an absent "
                         "key really does mean the un-flagged behaviour. Must be "
                         "named; see serve_flags_from_manifest for why absence "
                         "is not self-evidently legacy.")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    arrays = {}
    total = 0
    for key, tensor in state.items():
        arr = tensor.detach().cpu().numpy()
        # norms and biases stay fp32; big matrices go fp16
        arrays[key] = arr.astype(np.float16 if arr.size > 4096 else np.float32)
        total += arr.size
    flags = None
    if args.run_manifest:
        flags = serve_flags_from_manifest(
            args.run_manifest, assume_legacy=args.assume_legacy_serve_flags)
        for name, value in flags.items():
            # int8 scalars, and a `serve.` prefix no weight name can collide with
            arrays["serve." + name] = np.array(int(value), dtype=np.int8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    meta = {
        "schema": "ptcg-ai/ss-weights/v1",
        "checkpoint": str(args.ckpt),
        "params": total,
        "tensors": len(arrays),
        "bytes": args.out.stat().st_size,
        "mb": round(args.out.stat().st_size / 1e6, 2),
        "keys": sorted(arrays)[:8],
        "serve_flags": flags,
    }
    if args.meta:
        args.meta.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
