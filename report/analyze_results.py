#!/usr/bin/env python3
"""Recompute aggregate statistics. No dependencies, network, games or training."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys


OUTCOMES = (
    "baseline", "baseline_first", "baseline_second",
    "ismcts_same", "ismcts_cross", "gumbel_same",
)
SCHEMAS = {name: {"wins", "losses", "draws", "games"} for name in OUTCOMES}
SCHEMAS.update({
    "brier_baseline": {"p_win", "p_draw", "p_loss", "recorded_brier", "rows", "episodes"},
    "engine_round2": {"before_seconds", "after_seconds", "reported_speedup", "parity_games",
                      "parity_moves", "timing_batches_per_arm", "games_per_timing_batch"},
    "rollout_gate": {"completed", "requested", "decision_records", "illegal_actions"},
})
SOURCES = {name: name for name in OUTCOMES}
SOURCES.update({"baseline_first": "baseline", "baseline_second": "baseline",
                "brier_baseline": "brier", "engine_round2": "engine", "rollout_gate": "rollout"})
RATIOS = {"p_win", "p_draw", "p_loss", "recorded_brier", "reported_speedup"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(wins, games):
    """Two-sided nominal 95% Wilson interval for win versus non-win."""
    z = NormalDist().inv_cdf(0.975)
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return centre - radius, centre + radius


def read_results(path):
    data = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["experiment", "metric", "value", "unit", "source_id"],
                "unexpected CSV columns")
        for row_number, row in enumerate(reader, start=2):
            require(None not in row and all(value is not None for value in row.values()),
                    f"CSV row {row_number}: malformed row")
            experiment, metric = row["experiment"], row["metric"]
            require(experiment in SCHEMAS, f"unknown experiment: {experiment}")
            require(metric in SCHEMAS[experiment], f"unknown metric: {experiment}.{metric}")
            values = data.setdefault(experiment, {})
            require(metric not in values, f"duplicate metric: {experiment}.{metric}")
            require(row["source_id"] == SOURCES[experiment], f"wrong source: {experiment}.{metric}")
            unit = "seconds" if metric.endswith("_seconds") else "ratio" if metric in RATIOS else "count"
            require(row["unit"] == unit, f"wrong unit: {experiment}.{metric}")
            value = float(row["value"])
            require(math.isfinite(value) and value >= 0, f"invalid value: {experiment}.{metric}")
            if unit == "count":
                require(value.is_integer(), f"noninteger count: {experiment}.{metric}")
                value = int(value)
            values[metric] = value
    require(set(data) == set(SCHEMAS), "missing experiments")
    for experiment, metrics in SCHEMAS.items():
        require(set(data[experiment]) == metrics, f"missing metrics: {experiment}")
    for experiment in OUTCOMES:
        values = data[experiment]
        require(values["games"] > 0, f"{experiment}: games must be positive")
        require(values["wins"] + values["losses"] + values["draws"] == values["games"],
                f"{experiment}: wins + losses + draws must equal games")
    for metric in SCHEMAS["baseline"]:
        require(data["baseline_first"][metric] + data["baseline_second"][metric] == data["baseline"][metric],
                f"baseline seat strata do not sum to baseline {metric}")
    brier = data["brier_baseline"]
    rates = [brier[name] for name in ("p_win", "p_draw", "p_loss")]
    require(all(value <= 1 for value in rates) and math.isclose(sum(rates), 1, abs_tol=1e-6),
            "Brier class rates must be probabilities summing to one")
    require(0 < brier["episodes"] <= brier["rows"], "Brier row/game counts inconsistent")
    calculated = 1 - sum(value * value for value in rates)
    require(abs(calculated - brier["recorded_brier"]) <= 2e-6,
            "Brier identity disagrees beyond rounded-input tolerance")
    engine = data["engine_round2"]
    require(all(value > 0 for value in engine.values()), "engine measurements must be positive")
    require(abs(engine["before_seconds"] / engine["after_seconds"] - engine["reported_speedup"]) <= 5e-7,
            "speed ratio disagrees with reported six-decimal value")
    rollout = data["rollout_gate"]
    require(0 < rollout["completed"] <= rollout["requested"], "rollout completion counts inconsistent")
    require(rollout["illegal_actions"] <= rollout["decision_records"], "rollout illegal count inconsistent")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--results", type=Path, default=results / "results.csv")
    parser.add_argument("--provenance", type=Path, default=results / "provenance.json")
    args = parser.parse_args()
    try:
        data = read_results(args.results)  # Check contradictions before the snapshot digest.
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        require(provenance["schema"] == "ptcg-aggregate-reproduction/v1", "unknown provenance schema")
        require(hashlib.sha256(args.results.read_bytes()).hexdigest() == provenance["results_csv_sha256"],
                "CSV SHA256 differs from the recorded aggregate snapshot")
        require(set(SOURCES.values()) <= set(provenance["sources"]), "missing source provenance")
        for source in provenance["sources"].values():
            digest = source["sha256"]
            require(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
                    "invalid source SHA256")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("AGGREGATE ARITHMETIC REPRODUCTION — no games or training executed")
    print("Nominal Wilson 95% intervals; draws count as non-wins.")
    for experiment in OUTCOMES:
        values = data[experiment]
        low, high = wilson(values["wins"], values["games"])
        print(f"{experiment}: {values['wins']}/{values['games']} = "
              f"{100 * values['wins'] / values['games']:.2f}% [{100 * low:.2f}, {100 * high:.2f}]")
    first, second = data["baseline_first"], data["baseline_second"]
    difference = first["wins"] / first["games"] - second["wins"] / second["games"]
    print(f"Baseline first-minus-second seat difference: {100 * difference:.2f} percentage points (descriptive)")
    brier = data["brier_baseline"]
    calculated = 1 - sum(brier[name] ** 2 for name in ("p_win", "p_draw", "p_loss"))
    print(f"Brier constant baseline: {calculated:.6f} from rounded class rates; "
          f"{brier['rows']} rows / {brier['episodes']} games; recorded {brier['recorded_brier']:.6f}")
    engine = data["engine_round2"]
    print(f"Engine incremental speed ratio: {engine['before_seconds'] / engine['after_seconds']:.6f}x; "
          f"macOS arm64; {engine['parity_games']} parity games / {engine['parity_moves']} moves")
    print("Engine additional 2x target: NOT MET; no portability or playing-strength inference.")
    rollout = data["rollout_gate"]
    print(f"Rollout instrumentation: {rollout['completed']}/{rollout['requested']} complete; "
          f"{rollout['decision_records']} records; {rollout['illegal_actions']} illegal actions")
    print("Scope: experiments remain separate; seat strata reuse baseline games. "
          "Gumbel engine shuffles were unpaired. No pooled strength estimate.")
    print("PASS: arithmetic, internal counts and aggregate snapshot SHA256; "
          "original source bytes and full environment are not reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
