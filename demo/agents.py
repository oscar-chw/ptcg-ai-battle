"""Two small agents for the demo: uniform-random, and a fixed-priority greedy player.

Neither is a competition agent. They read the engine's plain-dict observation, take the
engine's enums and attack-damage table as arguments (so they run against a test double
as well as the real engine), and return a list of option indices.
"""
from __future__ import annotations

import random


class RandomAgent:
    """Uniformly random legal response: a random count in [min, max], random options."""

    name = "random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def act(self, obs: dict) -> list[int]:
        select = obs["select"]
        n = len(select["option"])
        count = self.rng.randint(select["minCount"], select["maxCount"])
        return self.rng.sample(range(n), count)


class GreedyAgent:
    """Set up first (evolve, attach, play a card, use an ability), then attack, then end.

    Attacking ends the turn, so it comes after everything that can still be done; among
    attacks the one with the most printed damage is chosen.

    Anything that is not the main-phase menu takes the first `maxCount` options, with
    YES preferred to NO. It never retreats or discards in play on its own initiative.
    `option_type` is the engine's OptionType enum, `select_main` the value of
    SelectType.MAIN, and `damage` maps attackId to printed damage.
    """

    name = "greedy"
    MAX_ACTIONS_PER_TURN = 60   # a stuck loop ends the turn instead of the process

    def __init__(self, option_type, select_main: int, damage: dict[int, int]):
        self.t = option_type
        self.select_main = select_main
        self.damage = damage
        self.rank = {option_type.EVOLVE: 1, option_type.ATTACH: 2, option_type.PLAY: 3,
                     option_type.ABILITY: 4}

    def act(self, obs: dict) -> list[int]:
        select = obs["select"]
        options = select["option"]
        if select["type"] == self.select_main:
            return [self._main_phase(obs, options)]
        order = sorted(range(len(options)),
                       key=lambda i: options[i]["type"] != self.t.YES)
        return order[:max(select["maxCount"], select["minCount"])]

    def _main_phase(self, obs: dict, options: list[dict]) -> int:
        end = next(i for i, o in enumerate(options) if o["type"] == self.t.END)
        if obs["current"]["turnActionCount"] > self.MAX_ACTIONS_PER_TURN:
            return end
        ranked = [(self.rank[o["type"]], i) for i, o in enumerate(options)
                  if o["type"] in self.rank]
        if ranked:
            return min(ranked)[1]
        attacks = [i for i, o in enumerate(options) if o["type"] == self.t.ATTACK]
        if attacks:
            return max(attacks, key=lambda i: self.damage.get(options[i]["attackId"], 0))
        return end
