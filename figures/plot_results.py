#!/usr/bin/env python3
"""Render figures/results.png: every win rate in the README's results, with its interval.

    python figures/plot_results.py      # needs matplotlib; rows() alone needs only the stdlib

Two panels, because the rows do not share an opponent or an interval method:
  left   the seat-balanced batteries in results/results.csv, nominal Wilson 95% intervals
         recomputed by report/analyze_results.py (the same function that writes
         results/analysis_output.txt)
  right  PPO against its frozen parent, from figures/ppo_head_to_head.csv, whose `quote`
         column is copied verbatim from results/ppo_RESULTS.md (intervals as recorded
         there). Draws score 0.5 in that evaluator; the game counts are derived from the
         intervals by figures/ppo_counts.py.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
from analyze_results import read_results, wilson  # noqa: E402
from ppo_counts import solutions  # noqa: E402

PPO_CSV = ROOT / "figures" / "ppo_head_to_head.csv"
OUT = ROOT / "figures" / "results.png"
QUOTE = re.compile(r"^(\d*\.\d+) \[(\d*\.\d+), ?(\d*\.\d+)\]$")

# (experiment in results.csv, label, role); "rejected" marks the candidates that failed.
BATTERY = [
    ("baseline", "Rule/search baseline vs cross-deck control", "baseline"),
    ("baseline_first", "  same games, first seat only", "baseline"),
    ("baseline_second", "  same games, second seat only", "baseline"),
    ("gumbel_same", "Gumbel candidate vs same-deck baseline", "candidate"),
    ("ismcts_same", "ISMCTS vs same-deck baseline (rejected)", "rejected"),
    ("ismcts_cross", "ISMCTS cross-deck battery (rejected)", "rejected"),
]


def battery_rows():
    """(label, role, rate, low, high, annotation) for each battery, recomputed from the CSV."""
    data = read_results(ROOT / "results" / "results.csv")
    rows = []
    for key, label, role in BATTERY:
        wins, games = data[key]["wins"], data[key]["games"]
        low, high = wilson(wins, games)
        rate = wins / games
        rows.append((label, role, rate, low, high,
                     f"{wins}/{games} = {100 * rate:.2f}% [{100 * low:.2f}, {100 * high:.2f}]"))
    return rows


def ppo_rows():
    """(label, role, rate, low, high, annotation) parsed from the recorded quotes."""
    rows = []
    with PPO_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            match = QUOTE.match(row["quote"])
            if not match:
                raise ValueError(f"unparseable quote: {row['quote']!r}")
            rate, low, high = (float(x) for x in match.groups())
            derived = ", ".join(f"{s:g}/{g}" for s, g in solutions(row["quote"]))
            rows.append((row["label"], row["role"], rate, low, high,
                         f"{row['quote']}  = {derived}"))
    return rows


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colour = {"baseline": "#8a8984", "control": "#8a8984", "candidate": "#2a78d6", "rejected": "#eb6834"}
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    panels = [
        (battery_rows(), "Seat-balanced batteries, 400 games each (seat rows 200)",
         "source: results/results.csv, recomputed by report/analyze_results.py; nominal Wilson 95%; "
         "opponents differ by row, so rows are not pooled; no draws occurred"),
        (ppo_rows(), "Self-play PPO vs its frozen parent (never submitted)",
         "source: results/ppo_RESULTS.md, intervals as recorded (not adjusted for picking the best of "
         "3 arms); draws score 0.5;\nscore/games derived from the intervals by figures/ppo_counts.py"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), facecolor=surface,
                             gridspec_kw={"height_ratios": [6, 4], "hspace": 0.75})
    for ax, (rows, title, source) in zip(axes, panels):
        ax.set_facecolor(surface)
        for y, (label, role, rate, low, high, note) in enumerate(reversed(rows)):
            c = colour[role]
            ax.plot([100 * low, 100 * high], [y, y], color=c, lw=2, solid_capstyle="round")
            ax.plot(100 * rate, y, "o", ms=8, color=c, mec=surface, mew=2)
            ax.text(101.5, y, note, va="center", fontsize=9, color=muted)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)], color=ink)
        ax.axvline(50, color=muted, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xticks(range(0, 101, 20))
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.set_xlabel("win rate, % (dashed line: 50%)", color=muted)
        ax.set_title(title, loc="left", color=ink, fontsize=11, fontweight="bold",
                     pad=18 + 12 * source.count("\n"))
        ax.text(0, 1.02, source, transform=ax.transAxes, fontsize=8.5, color=muted)
    handles = [plt.Line2D([], [], marker="o", lw=2, ms=7, color=colour[k], label=t)
               for k, t in (("baseline", "baseline / control"), ("candidate", "candidate"),
                            ("rejected", "rejected candidate"))]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.98, 0.995), labelcolor=ink)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=surface)
    return OUT


if __name__ == "__main__":
    print(render())
