"""The demo's agents and match loop, run against a TEST DOUBLE of the engine.

The double is a three-hit duel with the same call surface as the engine's cg.game
(battle_start / battle_select / battle_finish) and plain-dict observations. It carries
no game data and none of the engine's code, so these tests need no engine.
"""
import os
import subprocess
import sys
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import run_match  # noqa: E402
from agents import GreedyAgent, RandomAgent  # noqa: E402


class OptionType(IntEnum):
    YES, NO, PLAY, ATTACH, EVOLVE, ABILITY, ATTACK, END = 1, 2, 7, 8, 9, 10, 13, 14


MAIN = 0


def option(kind, **extra):
    return dict(type=int(kind), **extra)


def main_obs(options, turn_actions=0):
    return {"select": {"type": MAIN, "minCount": 1, "maxCount": 1, "option": options},
            "current": {"turnActionCount": turn_actions, "yourIndex": 0, "result": -1}}


def greedy():
    return GreedyAgent(OptionType, MAIN, {1: 10, 2: 50})


class Agents(unittest.TestCase):
    def test_greedy_sets_up_before_it_attacks(self):
        obs = main_obs([option(OptionType.ATTACK, attackId=2), option(OptionType.PLAY),
                        option(OptionType.ATTACH), option(OptionType.END)])
        self.assertEqual(greedy().act(obs), [2])          # ATTACH outranks PLAY and ATTACK

    def test_greedy_attacks_with_the_highest_damage_when_nothing_is_left_to_set_up(self):
        obs = main_obs([option(OptionType.ATTACK, attackId=1),
                        option(OptionType.ATTACK, attackId=2), option(OptionType.END)])
        self.assertEqual(greedy().act(obs), [1])

    def test_greedy_ends_the_turn_when_it_can_do_nothing_else(self):
        self.assertEqual(greedy().act(main_obs([option(OptionType.END)])), [0])

    def test_a_stuck_turn_is_ended_not_looped(self):
        obs = main_obs([option(OptionType.ABILITY), option(OptionType.END)],
                       turn_actions=GreedyAgent.MAX_ACTIONS_PER_TURN + 1)
        self.assertEqual(greedy().act(obs), [1])

    def test_outside_the_main_phase_greedy_prefers_yes_and_takes_max_count(self):
        obs = {"select": {"type": 9, "minCount": 1, "maxCount": 1,
                          "option": [option(OptionType.NO), option(OptionType.YES)]}}
        self.assertEqual(greedy().act(obs), [1])
        obs = {"select": {"type": 1, "minCount": 0, "maxCount": 2,
                          "option": [option(OptionType.PLAY)] * 4}}
        self.assertEqual(len(greedy().act(obs)), 2)

    def test_random_agent_always_answers_legally(self):
        agent = RandomAgent(seed=3)
        for lo, hi, n in [(0, 0, 3), (1, 1, 1), (0, 3, 5), (2, 4, 4)]:
            obs = {"select": {"minCount": lo, "maxCount": hi,
                              "option": [option(OptionType.PLAY)] * n}}
            for _ in range(50):
                picked = agent.act(obs)
                self.assertTrue(lo <= len(picked) <= hi)
                self.assertEqual(len(set(picked)), len(picked))
                self.assertTrue(all(0 <= i < n for i in picked))

    def test_random_agent_clips_max_count_to_the_option_count(self):
        agent = RandomAgent(seed=0)
        obs = {"select": {"minCount": 1, "maxCount": 3,
                          "option": [option(OptionType.PLAY)] * 2}}
        for _ in range(200):
            picked = agent.act(obs)
            self.assertTrue(1 <= len(picked) <= 2)
            self.assertEqual(sorted(set(picked)), sorted(picked))


class Duel:
    """First player to land 3 hits wins. `hits_to_win=None` makes a game that never ends."""

    def __init__(self, hits_to_win=3):
        self.hits_to_win = hits_to_win
        self.finished = 0
        self.seat_zero_agent = []

    def battle_start(self, deck0, deck1):
        assert len(deck0) == len(deck1) == 60
        self.hp, self.turn = [self.hits_to_win] * 2 if self.hits_to_win else [10**9] * 2, 0
        return self._obs(), SimpleNamespace(errorPlayer=-1, errorType=0)

    def _obs(self, result=-1):
        return {"select": {"type": MAIN, "minCount": 1, "maxCount": 1,
                           "option": [option(OptionType.ATTACK, attackId=1),
                                      option(OptionType.END)]},
                "current": {"turnActionCount": 0, "yourIndex": self.turn, "result": result}}

    def battle_select(self, picked):
        if picked == [0]:                       # ATTACK
            self.hp[1 - self.turn] -= 1
            if self.hp[1 - self.turn] <= 0:
                return self._obs(result=self.turn)
        self.turn = 1 - self.turn
        return self._obs()

    def battle_finish(self):
        self.finished += 1


class MatchLoop(unittest.TestCase):
    def test_every_game_is_decided_and_released(self):
        duel = Duel()
        agents = [greedy(), greedy()]
        agents[1].name = "other"
        winner = run_match.play_game(duel, [1] * 60, agents)
        self.assertEqual(winner, 0)               # seat 0 moves first and lands 3 hits first
        self.assertEqual(duel.finished, 1)

    def test_a_game_that_never_ends_is_a_timeout_and_still_released(self):
        duel = Duel(hits_to_win=None)
        run_match.MAX_STEPS, saved = 50, run_match.MAX_STEPS
        try:
            winner = run_match.play_game(duel, [1] * 60, [greedy(), greedy()])
        finally:
            run_match.MAX_STEPS = saved
        self.assertIsNone(winner)
        self.assertEqual(duel.finished, 1)

    def test_run_alternates_seats_and_accounts_for_every_game(self):
        duel = Duel()
        api = SimpleNamespace(all_attack=lambda: [SimpleNamespace(attackId=1, damage=30)],
                              OptionType=OptionType, SelectType=SimpleNamespace(MAIN=MAIN))
        first_seat = []
        real_play = run_match.play_game

        def recording(game, deck, agents):
            first_seat.append(agents[0].name)
            return real_play(game, deck, agents)

        run_match.play_game = recording
        try:
            summary = run_match.run(SimpleNamespace(game=duel, api=api), [1] * 60, games=40, seed=0)
        finally:
            run_match.play_game = real_play
        self.assertEqual(sum(summary["wins"].values()) + summary["undecided"], 40)
        self.assertEqual(duel.finished, 40)
        # the first-turn advantage is real, so each agent must open exactly half the games
        self.assertEqual(first_seat, ["greedy", "random"] * 20)

    def test_report_names_both_agents_and_the_interval(self):
        text = run_match.report({"games": 10, "wins": {"greedy": 7, "random": 3},
                                 "undecided": 0}, 1.0)
        self.assertIn("greedy", text)
        self.assertIn("70.0%", text)
        self.assertIn("Wilson", text)


class Cli(unittest.TestCase):
    def run_cli(self, **env):
        base = {k: v for k, v in os.environ.items() if k != "PTCG_ENGINE_DIR"}
        return subprocess.run([sys.executable, str(DEMO / "run_match.py"), "--games", "2"],
                              capture_output=True, text=True, env={**base, **env})

    def test_without_the_engine_variable_it_exits_2_and_says_how_to_get_the_engine(self):
        r = self.run_cli()
        self.assertEqual(r.returncode, 2)
        self.assertIn("PTCG_ENGINE_DIR", r.stderr)
        self.assertIn("Kaggle", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_a_directory_without_cg_is_refused_with_its_name(self):
        r = self.run_cli(PTCG_ENGINE_DIR=str(DEMO))        # DEMO has no cg/ inside
        self.assertEqual(r.returncode, 2)
        self.assertIn("no cg/ folder", r.stderr)


if __name__ == "__main__":
    unittest.main()
