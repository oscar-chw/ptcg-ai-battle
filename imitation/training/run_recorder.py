#!/usr/bin/env python3
"""Durable recording for every training run, so any run can be revisited.

A run that cannot be reconstructed later is a run that has to be repeated.  This
module stamps one directory per run with everything needed to answer, months
later: what data was this trained on, with what settings, what did it do while
training, and which checkpoint is which.

    rec = RunRecorder("marnie-sixthsense-bc-001", config)
    rec.record_dataset(corpus_csv)
    rec.record_splits(train_eps, val_eps, test_eps)
    ...
    rec.log_metrics(step=1000, epoch=1, train_loss=..., val_top1=...)
    rec.save_checkpoint(step, state_dict_bytes)
    rec.finish(result_dict, gates_dict)

Layout under ``runs/<run_id>/``::

    manifest.json     data hashes, resolved config, seeds, environment, git state
    splits.json       the exact episode ids per split - without this an eval is
                      not reproducible, only re-runnable with different luck
    metrics.jsonl     one JSON object per evaluation point, append-only
    checkpoints/      step-XXXXXXXX.pt plus a .sha256 beside each
    RESULT.json       final summary and gate outcomes
    stdout.log        whatever the trainer printed

Everything is written atomically (tmp + replace) so a killed run leaves valid
files rather than truncated ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
SCHEMA = "ptcg-ai/run-record/v1"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True, timeout=15
            ).stdout.strip() or None
        except Exception:  # noqa: BLE001 - git state is context, never fatal
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch  # noqa: PLC0415 - optional, recorded when present

        env["torch"] = torch.__version__
        env["cuda"] = getattr(torch.version, "cuda", None)
        env["gpu_count"] = torch.cuda.device_count()
        env["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception:  # noqa: BLE001 - CPU-only hosts are legitimate
        env["torch"] = None
    return env


class RunRecorder:
    """One run directory, written as the run proceeds rather than at the end."""

    def __init__(self, run_id: str, config: dict[str, Any], runs_dir: Path | None = None):
        self.run_id = run_id
        self.dir = (runs_dir or RUNS_DIR) / run_id
        if self.dir.exists() and (self.dir / "RESULT.json").exists():
            raise SystemExit(
                f"{self.dir} already holds a finished run; pick a new run id "
                "rather than overwriting recorded evidence"
            )
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.started = time.time()
        self.manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "run_id": run_id,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "config": config,
            "git": _git_state(),
            "environment": _environment(),
            "datasets": [],
            "checkpoints": [],
        }
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        _atomic_write(self.dir / "manifest.json", json.dumps(self.manifest, indent=2))

    # -- provenance ----------------------------------------------------------

    def record_dataset(self, path: Path, role: str = "train", **extra: Any) -> str:
        """Hash an input file so a later reader can prove which data this was."""
        path = Path(path)
        digest = sha256_file(path)
        self.manifest["datasets"].append({
            "role": role,
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            **extra,
        })
        self._flush_manifest()
        return digest

    def record_splits(self, **splits: list[str]) -> None:
        """Persist the exact episode ids per split.

        Splitting by game is only reproducible if the assignment itself is
        stored; a seed alone stops being enough the moment the corpus changes.
        """
        _atomic_write(
            self.dir / "splits.json",
            json.dumps({k: sorted(v) for k, v in splits.items()}, indent=1),
        )
        self.manifest["split_sizes"] = {k: len(v) for k, v in splits.items()}
        self._flush_manifest()

    # -- during training -----------------------------------------------------

    def log_metrics(self, **fields: Any) -> None:
        """Append one evaluation point. Append-only, so a crash keeps history."""
        record = {"wall_s": round(time.time() - self.started, 3), **fields}
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def save_checkpoint(self, step: int, save_fn, keep_every: int = 1) -> Path | None:
        """Write ``checkpoints/step-XXXXXXXX.pt`` and hash it.

        ``save_fn(path)`` does the actual serialisation, so this module never
        imports torch.  Every checkpoint is hashed on write: a checkpoint whose
        identity is not recorded cannot be tied back to the metrics that
        justified promoting it.
        """
        if step % keep_every:
            return None
        path = self.dir / "checkpoints" / f"step-{step:08d}.pt"
        save_fn(path)
        digest = sha256_file(path)
        (path.parent / f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n")
        self.manifest["checkpoints"].append({
            "step": step,
            "file": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
        })
        self._flush_manifest()
        return path

    # -- end -----------------------------------------------------------------

    def finish(self, result: dict[str, Any], gates: dict[str, Any] | None = None) -> None:
        payload = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "elapsed_s": round(time.time() - self.started, 1),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result": result,
            "gates": gates or {},
            "checkpoints": self.manifest["checkpoints"],
            "datasets": self.manifest["datasets"],
        }
        _atomic_write(self.dir / "RESULT.json", json.dumps(payload, indent=2))
        self.manifest["finished_utc"] = payload["finished_utc"]
        self._flush_manifest()


def summarise(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    """One row per recorded run — the index for revisiting past work."""
    base = runs_dir or RUNS_DIR
    rows = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        result_path = manifest_path.parent / "RESULT.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        rows.append({
            "run_id": manifest.get("run_id"),
            "started": manifest.get("started_utc"),
            "finished": manifest.get("finished_utc"),
            "checkpoints": len(manifest.get("checkpoints", [])),
            "status": "finished" if result else "incomplete",
            "result": result.get("result", {}),
        })
    return rows


if __name__ == "__main__":
    for row in summarise():
        print(json.dumps(row))
