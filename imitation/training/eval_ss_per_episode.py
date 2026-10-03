#!/usr/bin/env python3
"""Score Sixth Sense validation rows and emit paired per-episode metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

EXPECTED_EPISODES = 185
EXPECTED_ROWS = 17_392
BRIER_BASELINE = 0.487022
TOLERANCE = 1e-6
BATCH_SIZE = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_state_dict(path: Path, torch):
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} is not a state_dict")
    return state_dict


def _checkpoint_step(manifest: dict, checkpoint: Path) -> int:
    digest = _sha256(checkpoint)
    matches = [
        entry for entry in manifest.get("checkpoints", [])
        if entry.get("sha256") == digest
    ]
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint hash must match exactly one manifest entry; found {len(matches)}"
        )
    return int(matches[0]["step"])


def _logged_metrics(path: Path, step: int) -> dict:
    matches = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["step"]) == step:
                matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"step {step} must have exactly one metrics row; found {len(matches)}")
    return matches[0]


def _aux_classes(state_dict: dict) -> list[int]:
    found = {}
    for name, tensor in state_dict.items():
        match = re.fullmatch(r"aux\.(\d+)\.2\.weight", name)
        if match:
            found[int(match.group(1))] = int(tensor.shape[0])
    if sorted(found) != list(range(len(found))):
        raise ValueError("checkpoint aux heads are not contiguous")
    return [found[index] for index in range(len(found))]


def _build_model(manifest: dict, state_dict: dict, model_class):
    config = manifest["config"]
    required = ("d", "layers", "heads", "hidden", "num_scale")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"manifest config is missing model hyperparameters: {missing}")
    aux_classes = _aux_classes(state_dict)
    kwargs = {
        "d": int(config["d"]),
        "layers": int(config["layers"]),
        "heads": int(config["heads"]),
        "hidden": int(config["hidden"]),
        "n_aux": len(aux_classes),
        "aux_classes": aux_classes,
        "num_scale": config["num_scale"],
    }
    if "dropout" in config:
        kwargs["dropout"] = float(config["dropout"])
    # The rules-text basis ships INSIDE the checkpoint as a buffer, so a model
    # built without it cannot load one that has it -- "Unexpected key(s):
    # card_bow, word.weight". Rebuild it from the state_dict rather than
    # requiring the caller to supply the npz.
    if "card_bow" in state_dict:
        kwargs["card_bow"] = state_dict["card_bow"]
    model = model_class(**kwargs)
    model.load_state_dict(state_dict)
    return model


def _auto_device(torch):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    for index in range(torch.cuda.device_count()):
        try:
            processes = torch.cuda.list_gpu_processes(index).lower()
        except Exception:  # NVML unavailable means an idle GPU cannot be proved.
            return torch.device("cpu")
        if "no processes are running" in processes or "no active processes" in processes:
            return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _device(requested: str, torch):
    if requested == "auto":
        return _auto_device(torch)
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def _score(model, rows: list[dict], device, torch, functional, collate_fn) -> dict:
    totals = defaultdict(lambda: {"rows": 0, "top1_correct": 0, "brier_sum": 0.0})
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), BATCH_SIZE):
            chunk = rows[offset:offset + BATCH_SIZE]
            batch, labels, wdl = collate_fn(chunk, device)
            output = model(batch)
            predictions = output["policy"].argmax(-1)
            probabilities = output["value"].softmax(-1)
            onehot = functional.one_hot(wdl, 3).float()
            correct = (predictions == labels).to("cpu").tolist()
            brier = ((probabilities - onehot) ** 2).sum(-1).to("cpu").tolist()
            for row, is_correct, row_brier in zip(chunk, correct, brier, strict=True):
                episode = row["episode"]
                totals[episode]["rows"] += 1
                totals[episode]["top1_correct"] += int(is_correct)
                totals[episode]["brier_sum"] += float(row_brier)
    return totals


def _write_episode_dump(path: Path, totals: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode, metrics in totals.items():
            record = {"episode": episode, **metrics}
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _aggregate_dump(path: Path) -> tuple[int, int, float, float]:
    episodes = rows = correct = 0
    brier_sum = 0.0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            episodes += 1
            rows += int(record["rows"])
            correct += int(record["top1_correct"])
            brier_sum += float(record["brier_sum"])
    return episodes, rows, correct / max(rows, 1), brier_sum / max(rows, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit validation metrics per independent Sixth Sense episode."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()

    import torch  # noqa: PLC0415 - keep --help usable on CPU-only control hosts
    import torch.nn.functional as F  # noqa: N812, PLC0415

    from train_ss import load, collate  # noqa: PLC0415
    from model_ss import SixthSenseNet  # noqa: PLC0415

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint_step = _checkpoint_step(manifest, args.ckpt)
    logged = _logged_metrics(args.manifest.parent / "metrics.jsonl", checkpoint_step)
    state_dict = _load_state_dict(args.ckpt, torch)
    device = _device(args.device, torch)
    model = _build_model(manifest, state_dict, SixthSenseNet).to(device)
    _train_rows, validation_rows = load(args.data / "frames.jsonl.gz", skip_train=True)

    totals = _score(model, validation_rows, device, torch, F, collate)
    _write_episode_dump(args.out, totals)
    episodes, rows, recomputed_top1, recomputed_brier = _aggregate_dump(args.out)
    top1_delta = abs(recomputed_top1 - float(logged["top1"]))
    brier_delta = abs(recomputed_brier - float(logged["brier"]))
    status = "PASS" if (
        episodes == EXPECTED_EPISODES
        and rows == EXPECTED_ROWS
        and top1_delta <= TOLERANCE
        and brier_delta <= TOLERANCE
    ) else "FAIL"
    gate = {
        "status": status,
        "episodes": episodes,
        "rows": rows,
        "recomputed_top1": recomputed_top1,
        "recomputed_brier": recomputed_brier,
        "brier_skill_score": 1.0 - recomputed_brier / BRIER_BASELINE,
        "checkpoint_step": checkpoint_step,
        "logged_top1": float(logged["top1"]),
        "logged_brier": float(logged["brier"]),
        "top1_abs_delta": top1_delta,
        "brier_abs_delta": brier_delta,
        "device": str(device),
    }
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.gate.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
