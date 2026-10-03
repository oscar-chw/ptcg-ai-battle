#!/usr/bin/env python3
"""Recover the PPO head-to-head game counts from the recorded intervals. Standard library only.

    python3 figures/ppo_counts.py

results/ppo_RESULTS.md records each PPO win rate with an interval but not the number of
games. The evaluator that produced them (not included in this repository) scored a win 1,
a loss 0 and a draw or unfinished game 0.5, and used a 95% Wilson interval on that
score. So for each recorded "rate [low, high]" there are only a few (score, games) pairs,
with score a multiple of 0.5, that reproduce all three numbers at their printed
precision. This script finds them by exhaustive search, then widens the headline arm's
interval for having been picked as the best of the three PPO arms on the same battery
(Bonferroni over 3).
"""
import csv
import math
import re
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
QUOTE = re.compile(r"^(\d*\.(\d+)) \[(\d*\.(\d+)), ?(\d*\.(\d+))\]$")
Z95 = NormalDist().inv_cdf(0.975)
MAX_GAMES = 2000


def wilson(score, games, z=Z95):
    p = score / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return centre - radius, centre + radius


def solutions(quote):
    """Every (score, games), score in half-points, that reproduces the quote as printed."""
    m = QUOTE.match(quote)
    if not m:
        raise ValueError(f"unparseable quote: {quote!r}")
    rate, low, high = (float(m.group(i)) for i in (1, 3, 5))
    dp_rate, dp_low, dp_high = (len(m.group(i)) for i in (2, 4, 6))
    found = []
    for games in range(2, MAX_GAMES + 1):
        lo_half = math.ceil((rate - 10 ** -dp_rate) * games * 2)
        for half in range(max(lo_half, 0), min(2 * games, math.floor((rate + 10 ** -dp_rate) * games * 2)) + 1):
            score = half / 2
            if round(score / games, dp_rate) != rate:
                continue
            a, b = wilson(score, games)
            if round(a, dp_low) == low and round(b, dp_high) == high:
                found.append((score, games))
    return found


def rows():
    with (ROOT / "figures" / "ppo_head_to_head.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    print("PPO head-to-head: (score, games) consistent with each recorded interval")
    print("(draws and unfinished games score 0.5 here; the battery table counts them as non-wins)")
    best = None
    for row in rows():
        found = solutions(row["quote"])
        shown = ", ".join(f"{s:g}/{g}" for s, g in found) or "none"
        print(f"  {row['label']:<40} {row['quote']:<24} -> {shown}")
        if row["role"] == "candidate" and len(found) == 1:
            score, games = found[0]
            if best is None or score / games > best[1] / best[2]:
                best = (row["label"], score, games)
    label, score, games = best
    z = NormalDist().inv_cdf(1 - 0.05 / 3 / 2)
    low, high = wilson(score, games, z)
    print(f"Best of 3 arms, {label}: {score:g}/{games} = {score / games:.4f}; "
          f"Bonferroni-adjusted (3 arms) Wilson interval [{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    main()
