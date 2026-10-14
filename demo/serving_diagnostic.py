#!/usr/bin/env python3
"""Inspect actual CPU Torch/NumPy forward differences on explicitly SYNTHETIC features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

from decision_audit import decision_record

ROOT = Path(__file__).resolve().parents[1]
SCORE_LIMIT = 1e-4  # The existing demo/parity tests' threshold; not tuned by this diagnostic.
SERVING_MODES = ("masked", "unmasked", "no-state-mask", "no-option-mask", "no-relations")
REFERENCE_MODES = ("masked", "unmasked")


class DiagnosticError(ValueError):
    """A bounded fixture or comparison cannot be evaluated faithfully."""


def _vectors(values, where: str) -> list[list[float]]:
    if not isinstance(values, list) or not values:
        raise DiagnosticError(f"{where}: expected a nonempty row list")
    out = []
    for row_index, row in enumerate(values):
        if not isinstance(row, list) or not row:
            raise DiagnosticError(f"{where}[{row_index}]: expected nonempty logits")
        converted = []
        for index, value in enumerate(row):
            if type(value) not in (int, float):
                raise DiagnosticError(f"{where}[{row_index}][{index}]: expected a finite number")
            try:
                number = float(value)
            except OverflowError as error:
                raise DiagnosticError(f"{where}[{row_index}][{index}]: number exceeds float range") from error
            if not math.isfinite(number):
                raise DiagnosticError(f"{where}[{row_index}][{index}]: nonfinite logit")
            converted.append(number)
        out.append(converted)
    return out


def compare_rows(reference, serving) -> dict:
    reference, serving = _vectors(reference, "reference"), _vectors(serving, "serving")
    if len(reference) != len(serving):
        raise DiagnosticError("row count differs between reference and serving")
    rows, first = [], None
    for row_index, (left, right) in enumerate(zip(reference, serving)):
        if len(left) != len(right):
            raise DiagnosticError(f"row {row_index}: option count differs between reference and serving")
        gaps = [abs(a - b) for a, b in zip(left, right)]
        if not all(math.isfinite(gap) for gap in gaps):
            raise DiagnosticError(f"row {row_index}: score difference exceeds float range")
        gap, option = max(gaps), gaps.index(max(gaps))
        expected, actual = left.index(max(left)), right.index(max(right))
        passed = gap < SCORE_LIMIT and expected == actual
        row = {"row": row_index, "option_count": len(left), "max_abs_error": gap,
               "max_gap_option": option, "reference_choice": expected, "serving_choice": actual,
               "reference_logits": left, "serving_logits": right, "passed": passed}
        rows.append(row)
        if not passed and first is None:
            first = row
    return {"passed": first is None, "max_abs_error": max(row["max_abs_error"] for row in rows),
            "choice_disagreements": sum(row["reference_choice"] != row["serving_choice"] for row in rows),
            "first_mismatch": first, "rows": rows}


def _flags(mode: str) -> dict[str, bool]:
    enabled = mode != "unmasked"
    flags = {"mask_pad_state": enabled, "mask_pad_options": enabled, "relations": enabled}
    disabled = {"no-state-mask": "mask_pad_state", "no-option-mask": "mask_pad_options",
                "no-relations": "relations"}
    if mode in disabled:
        flags[disabled[mode]] = False
    return flags


def _numpy_logits(weights, row, mode):
    import model_demo
    return model_demo.numpy_logits(weights, row, mode).tolist()


def _source_hashes() -> dict[str, str]:
    names = ("demo/serving_diagnostic.py", "demo/model_demo.py", "imitation/tests_torch/synthetic.py",
             "imitation/training/model_ss.py", "imitation/training/train_ss.py",
             "imitation/serving/main_v7.py", "imitation/gates/gate_compute_parity.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def run_diagnostic(*, boards: int, seed: int, reference_mode: str, serving_mode: str) -> dict:
    if type(boards) is not int or not 1 <= boards <= 64:
        raise DiagnosticError("boards: must be an integer in1..64")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise DiagnosticError("seed: must be an integer in0..4294967295")
    if reference_mode not in REFERENCE_MODES or serving_mode not in SERVING_MODES:
        raise DiagnosticError("reference/serving mode: unsupported configuration")
    try:
        import model_demo
    except ImportError as error:
        raise DiagnosticError(f"runtime import failed: {error}; use PTCG_PYTHON with NumPy/Torch "
                              "and a complete repository checkout") from error
    provider = model_demo.syn
    names = ("TRUNCATE_SEQ", "MASK_PAD_STATE", "MASK_PAD_OPTIONS", "BUILD_RELATIONS", "NUM_W_RUN", "OPT_NUM_W_RUN")
    before = {name: getattr(provider.train_ss, name) for name in names}
    dtype_before = provider.torch.get_default_dtype()
    reference_flags, serving_flags = _flags(reference_mode), _flags(serving_mode)
    try:
        # Fixture construction must not inherit an embedding caller's accelerator or float64 defaults.
        provider.torch.set_default_dtype(provider.torch.float32)
        with provider.torch.random.fork_rng(devices=[]), provider.torch.device("cpu"):
            model = provider.make_model(seed=seed).cpu()
            weights = provider.numpy_weights(model)
            rows = provider.make_rows(boards, seed=seed + 1)
            provider.set_training_semantics(mask_state=reference_flags["mask_pad_state"],
                                            mask_options=reference_flags["mask_pad_options"],
                                            relations=reference_flags["relations"])
            reference = [row.tolist() for row in provider.torch_logits(model, rows)]
            control_scores = [_numpy_logits(weights, row, reference_flags) for row in rows]
            control = compare_rows(reference, control_scores)
            candidate = None
            if control["passed"]:
                scores = control_scores if serving_flags == reference_flags else [
                    _numpy_logits(weights, row, serving_flags) for row in rows]
                candidate = compare_rows(reference, scores)
            for comparison in (control, candidate):
                if comparison is None:
                    continue
                for encoded, witness in zip(rows, comparison["rows"]):
                    kinds = [kind for kind in encoded["option_type"] if kind >= 0]
                    menu = {"current": {"result": -1, "yourIndex": 0},
                            "select": {"type": 0, "minCount": 1, "maxCount": 1,
                                       "option": [{"type": kind} for kind in kinds]}}
                    decision_record(menu, [witness["serving_choice"]], "synthetic_numpy", 0, witness["row"])
                    witness["menu_choice_valid"] = True
            digest = hashlib.sha256()
            for name, array in sorted(weights.items()):
                description = json.dumps([name, array.dtype.str, list(array.shape)], separators=(",", ":"))
                digest.update(description.encode() + b"\0" + array.tobytes(order="C"))
            result = "CONTROL_FAILED" if not control["passed"] else ("PASS" if candidate["passed"] else "MISMATCH")
            return {"schema": "ptcg-serving-diagnostic/v1", "result": result,
                    "scope": "SYNTHETIC seeded random weights and encoded feature rows; CPU forward mechanics only, "
                             "not legal game states, engine replay or playing strength",
                    "historical_training_verified": False, "boards": boards, "seed": seed,
                    "reference_mode": reference_mode, "serving_mode": serving_mode,
                    "reference_flags": reference_flags, "serving_flags": serving_flags,
                    "criterion": {"score_error_strictly_below": SCORE_LIMIT, "choice_disagreements": 0},
                    "input_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                                               allow_nan=False).encode()).hexdigest(),
                    "weight_sha256": digest.hexdigest(), "source_sha256": _source_hashes(),
                    "runtime": {"python": platform.python_version(), "numpy": model_demo.np.__version__,
                                "torch": provider.torch.__version__, "device": "cpu", "fixture_dtype": "float32"},
                    "control": control, "candidate": candidate,
                    "recovery": "Match serving flags to the intended reference; prove historical flags from "
                                "the original run manifest before treating this as trained-model parity."}
    finally:
        provider.torch.set_default_dtype(dtype_before)
        for name, value in before.items():
            setattr(provider.train_ss, name, value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=8, help="synthetic encoded rows,1..64")
    parser.add_argument("--seed", type=int, default=0, help="model seed; input seed=seed+1")
    parser.add_argument("--reference-mode", choices=REFERENCE_MODES, default="masked")
    parser.add_argument("--serving-mode", choices=SERVING_MODES, default="masked")
    args = parser.parse_args(argv)
    if not 1 <= args.boards <= 64:
        parser.error("boards must be1..64")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be0..4294967295")
    try:
        report = run_diagnostic(boards=args.boards, seed=args.seed,
                                reference_mode=args.reference_mode, serving_mode=args.serving_mode)
    except DiagnosticError as error:
        print(f"diagnostic failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
