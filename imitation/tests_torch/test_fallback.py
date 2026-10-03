"""agent()'s guard must answer a failed decision legally, deterministically and VISIBLY.

The team's shipped serving file returned a random legal move here with no trace, so a
broken package played at random behind green offline gates. These tests force the guard
(no engine is on the path, so the featurizer import raises) on a SYNTHETIC prompt.
"""
import contextlib
import io
import unittest

import synthetic  # noqa: F401  (puts serving/ on sys.path)
import main_v7


def prompt(n_opt, min_count, max_count):
    """A SYNTHETIC select prompt: option dicts carry no game data."""
    return {"select": {"option": [{"type": 7} for _ in range(n_opt)],
                       "minCount": min_count, "maxCount": max_count},
            "current": {"yourIndex": 0, "turn": 1}}


class Fallback(unittest.TestCase):
    def setUp(self):
        main_v7.FALLBACKS = 0
        main_v7._DECK = ["SYNTHETIC"]   # skip deck.csv, which a test tree does not have

    def run_agent(self, obs):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = main_v7.agent(obs)
        return out, err.getvalue()

    def test_a_failed_decision_is_counted_logged_and_deterministic(self):
        answers = []
        for i in range(1, 6):
            out, log = self.run_agent(prompt(50, 1, 1))
            answers.append(out)
            self.assertEqual(main_v7.FALLBACKS, i)
            self.assertIn(f"FALLBACK {i}:", log)
        self.assertEqual(answers, [[0]] * 5)   # the lowest legal index, every time

    def test_the_fallback_count_respects_the_engine_bounds(self):
        out, _ = self.run_agent(prompt(4, 3, 3))
        self.assertEqual(out, [0, 1, 2])
        out, _ = self.run_agent(prompt(2, 0, 3))   # may pass, but the fallback does not
        self.assertEqual(out, [0])
        self.assertEqual(main_v7.FALLBACKS, 2)


if __name__ == "__main__":
    unittest.main()
