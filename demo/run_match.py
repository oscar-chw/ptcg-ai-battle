#!/usr/bin/env python3
"""Play two small agents against each other on the official engine and report the result.

The engine is NOT part of this repository (licence: competition use only). Point
PTCG_ENGINE_DIR at the directory that contains its `cg/` package; demo/README.md says
how to get it from Kaggle. Without it this script prints that and exits with status 2.

    PTCG_ENGINE_DIR=/path/to/sample_submission python3 demo/run_match.py --games 1000

Needs Python 3.10 or newer, because the engine's own api module uses 3.10 syntax.
Both players use the engine's sample deck (deck.csv, in the same directory), so the
match is a mirror and the only difference between the sides is the agent.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "report"))

from agents import GreedyAgent, RandomAgent  # noqa: E402
from analyze_results import wilson  # noqa: E402

NO_ENGINE = """\
PTCG_ENGINE_DIR is not set, so there is no engine to play on.

The official engine is not included in this repository: it is licensed for
competition use only (LicenseRef-PTCG-ABC-Competition-Use-Only). To run the demo:

  1. Join the Pokemon TCG AI Battle competition on Kaggle and accept its rules.
  2. Download the competition data (the Data tab) and unzip it.
  3. Set PTCG_ENGINE_DIR to the folder that contains cg/, deck.csv and main.py
     (sample_submission/sample_submission in the unzipped data), then re-run.

See demo/README.md. Needs Python 3.10 or newer.
"""
MAX_STEPS = 5000   # a game that has not ended by then is counted as a timeout


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def load_engine(directory: Path) -> SimpleNamespace:
    """Import the engine's game and api modules from `directory`, or raise ImportError."""
    sys.path.insert(0, str(directory))
    try:
        game = importlib.import_module("cg.game")
        api = importlib.import_module("cg.api")
    except (ImportError, TypeError, OSError) as err:   # TypeError: api needs Python 3.10+
        raise ImportError(
            f"could not load the engine from {directory}: {type(err).__name__}: {err}. "
            f"Is this the folder that contains cg/ ? Is Python 3.10 or newer in use?") from err
    return SimpleNamespace(game=game, api=api)


def read_deck(directory: Path) -> list[int]:
    deck = [int(token) for token in (directory / "deck.csv").read_text().split()]
    if len(deck) != 60:
        raise ValueError(f"{directory / 'deck.csv'} holds {len(deck)} cards, expected 60")
    return deck


def play_game(game, deck: list[int], agents: list) -> int | None:
    """Play one game; return the winning seat, or None for a draw or a timeout."""
    obs, start = game.battle_start(deck, deck)
    if obs is None:
        raise RuntimeError(f"the engine refused the deck (player {start.errorPlayer}, "
                           f"error {start.errorType})")
    try:
        for _ in range(MAX_STEPS):
            result = obs["current"]["result"]
            if result != -1:
                return result if result in (0, 1) else None
            obs = game.battle_select(agents[obs["current"]["yourIndex"]].act(obs))
        return None
    finally:
        game.battle_finish()


def run(engine, deck: list[int], games: int, seed: int) -> dict:
    """Alternate which agent sits first, so neither side owns the first-turn advantage."""
    damage = {a.attackId: a.damage for a in engine.api.all_attack()}
    wins = {"greedy": 0, "random": 0}
    undecided = 0
    for i in range(games):
        greedy = GreedyAgent(engine.api.OptionType, int(engine.api.SelectType.MAIN), damage)
        agents = [greedy, RandomAgent(seed * 1_000_003 + i)]
        if i % 2:
            agents.reverse()
        winner = play_game(engine.game, deck, agents)
        if winner is None:
            undecided += 1
        else:
            wins[agents[winner].name] += 1
    return {"games": games, "wins": wins, "undecided": undecided}


def report(summary: dict, seconds: float) -> str:
    lines = [f"{summary['games']} games, seats alternated, mirror deck"]
    for name, count in summary["wins"].items():
        low, high = wilson(count, summary["games"])
        lines.append(f"  {name:<7} {count:>4}/{summary['games']} = "
                     f"{100 * count / summary['games']:5.1f}%  "
                     f"[{100 * low:.1f}, {100 * high:.1f}] nominal Wilson 95%")
    lines.append(f"  draws or timeouts: {summary['undecided']}")
    lines.append(f"  {seconds:.1f} s. The engine's shuffles are not seeded here, so counts "
                 f"vary slightly between runs.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="seeds the random agent only")
    args = parser.parse_args(argv)

    directory = os.environ.get("PTCG_ENGINE_DIR")
    if not directory:
        return fail(NO_ENGINE)
    directory = Path(directory)
    if not (directory / "cg").is_dir():
        return fail(f"PTCG_ENGINE_DIR={directory} has no cg/ folder inside it; "
                    f"see demo/README.md.")
    try:
        engine = load_engine(directory)
        deck = read_deck(directory)
    except (ImportError, OSError, ValueError) as err:
        return fail(str(err))

    started = time.time()
    summary = run(engine, deck, args.games, args.seed)
    print(report(summary, time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
