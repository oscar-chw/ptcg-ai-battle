"""Selection-menu boundary tests, using original synthetic prompts, never game rules."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import run_match  # noqa: E402


def observation(minimum=1, maximum=1):
    return {"current": {"result": -1, "yourIndex": 0},
            "select": {"type": 9, "minCount": minimum, "maxCount": maximum,
                       "option": [{"type": 1}, {"type": 2}]}}


def record(selected=None):
    return {"schema": "ptcg-decision/v1", "game": 0, "step": 0, "agent": "fixture",
            "observation": observation(), "selected": [0] if selected is None else selected}


class OneDecision:
    def __init__(self, obs=None):
        self.obs = observation() if obs is None else obs
        self.actions = []
        self.finished = 0

    def battle_start(self, *_):
        return copy.deepcopy(self.obs), SimpleNamespace(errorPlayer=-1, errorType=0)

    def battle_select(self, action):
        self.actions.append(action)
        return {"current": {"result": 0}}

    def battle_finish(self):
        self.finished += 1


class FixedAgent:
    name = "fixture"

    def __init__(self, selected):
        self.selected = selected
        self.calls = 0

    def act(self, _):
        self.calls += 1
        return self.selected


class Dispatch(unittest.TestCase):
    def test_startup_error_and_deck_refusal_release_engine(self):
        class StartupFailure(OneDecision):
            def battle_start(self, *_):
                raise RuntimeError("startup failed")

        class RefusedDeck(OneDecision):
            def battle_start(self, *_):
                return None, SimpleNamespace(errorPlayer=0, errorType=9)

        for game in [StartupFailure(), RefusedDeck()]:
            with self.subTest(game=type(game).__name__), self.assertRaises(RuntimeError):
                try:
                    run_match.play_game(game, [1] * 60, [FixedAgent([0])] * 2)
                finally:
                    self.assertEqual(game.finished, 1)
                    self.assertEqual(game.actions, [])

    def test_an_agent_cannot_relax_the_issued_menu_by_mutating_its_observation(self):
        class MutatingAgent:
            def act(self, obs):
                obs["select"]["minCount"] = 0
                return []

        game = OneDecision()
        with self.assertRaisesRegex(ValueError, "selected"):
            run_match.play_game(game, [1] * 60, [MutatingAgent()] * 2)
        self.assertEqual(game.actions, [])
        self.assertEqual(game.finished, 1)

    def test_invalid_responses_never_reach_the_engine(self):
        for selected, field in [([-1], "selected"), ([2], "selected"), ([True], "selected"),
                                ([0.0], "selected"), (["0"], "selected"),
                                ([0, 0], "selected"), ([], "selected"),
                                ([0, 1], "selected"), (None, "selected"), ((0,), "selected")]:
            with self.subTest(selected=selected):
                game = OneDecision()
                with self.assertRaisesRegex(ValueError, field):
                    run_match.play_game(game, [1] * 60, [FixedAgent(selected)] * 2)
                self.assertEqual(game.actions, [])
                self.assertEqual(game.finished, 1)

    def test_invalid_state_is_rejected_before_agent_dispatch(self):
        cases = [("current", "yourIndex", -1), ("current", "yourIndex", True),
                 ("current", "yourIndex", 2), ("current", "result", True),
                 ("select", "minCount", -1), ("select", "minCount", True),
                 ("select", "minCount", 3), ("select", "maxCount", 0),
                 ("select", "maxCount", 1.5), ("select", "type", "MAIN")]
        for section, key, value in cases:
            with self.subTest(key=key, value=value):
                obs = observation()
                obs[section][key] = value
                game, agent = OneDecision(obs), FixedAgent([0])
                with self.assertRaisesRegex(ValueError, key):
                    run_match.play_game(game, [1] * 60, [agent] * 2)
                self.assertEqual(agent.calls, 0)
                self.assertEqual(game.actions, [])
                self.assertEqual(game.finished, 1)

    def test_valid_empty_and_capped_choices_are_not_repaired(self):
        for obs, selected in [(observation(0, 3), []), (observation(1, 3), [1, 0])]:
            with self.subTest(selected=selected):
                game = OneDecision(obs)
                self.assertEqual(run_match.play_game(game, [1] * 60, [FixedAgent(selected)] * 2), 0)
                self.assertEqual(game.actions, [selected])
                self.assertEqual(game.finished, 1)

    def test_generated_trace_audits_and_contains_only_the_selection_menu(self):
        import io
        trace = io.StringIO()
        obs = observation()
        obs["hidden_hand"] = "not recorded"
        game = OneDecision(obs)
        run_match.play_game(game, [1] * 60, [FixedAgent([1])] * 2, trace=trace, game_id=4)
        entry = json.loads(trace.getvalue())
        self.assertEqual(entry["game"], 4)
        self.assertEqual(entry["step"], 0)
        self.assertEqual(entry["selected"], [1])
        self.assertEqual(set(entry["observation"]), {"current", "select"})
        self.assertNotIn("hidden_hand", trace.getvalue())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            path.write_text(trace.getvalue())
            audit = cli(path)
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertEqual(json.loads(audit.stdout)["decisions"], 1)

    def test_trace_write_failure_still_releases_engine(self):
        class BrokenTrace:
            def write(self, _):
                raise OSError("trace disk full")

        game = OneDecision()
        with self.assertRaisesRegex(OSError, "trace disk full"):
            run_match.play_game(game, [1] * 60, [FixedAgent([0])] * 2, trace=BrokenTrace())
        self.assertEqual(game.actions, [])
        self.assertEqual(game.finished, 1)


def cli(path):
    return subprocess.run([sys.executable, str(DEMO / "audit_trace.py"), str(path)],
                          capture_output=True, text=True)


class AuditCli(unittest.TestCase):
    def test_excessive_json_nesting_has_source_line_instead_of_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested.jsonl"
            path.write_text(json.dumps(record()) + "\n" + "[" * 10000 + "0" + "]" * 10000 + "\n")
            result = cli(path)
        self.assertEqual(result.returncode, 1)
        self.assertIn("nested.jsonl:2", result.stderr)
        self.assertIn("nesting", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_nonstandard_json_number_in_an_extra_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nonstandard.jsonl"
            content = json.dumps(record()).replace('"game": 0', '"extra": NaN, "game": 0')
            path.write_text(content)
            result = cli(path)
        self.assertEqual(result.returncode, 1)
        self.assertIn("JSON", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_trace_cli_refuses_overwrite_before_running_a_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cg").mkdir()
            target = root / "existing.jsonl"
            target.write_text("keep this evidence\n")
            with patch.dict(os.environ, {"PTCG_ENGINE_DIR": str(root)}), \
                    patch.object(run_match, "load_engine", return_value=object()), \
                    patch.object(run_match, "read_deck", return_value=[1] * 60), \
                    patch.object(run_match, "run") as run:
                self.assertEqual(run_match.main(["--games", "1", "--trace", str(target)]), 2)
                run.assert_not_called()
            self.assertEqual(target.read_text(), "keep this evidence\n")

    def test_valid_trace_counts_decisions_and_passes_without_an_engine(self):
        first, second = record(), record([])
        second["step"], second["observation"] = 1, observation(0, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.jsonl"
            path.write_text(json.dumps(first) + "\n\n" + json.dumps(second) + "\n")
            before = path.read_bytes()
            env = {k: v for k, v in os.environ.items() if k != "PTCG_ENGINE_DIR"}
            result = subprocess.run([sys.executable, str(DEMO / "audit_trace.py"), str(path)],
                                    capture_output=True, text=True, env=env)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["agents"]["fixture"], {"decisions": 2, "passes": 1, "selected_options": 1})
        self.assertIn("menu", summary["scope"])

    def test_invalid_second_line_reports_location_and_no_success_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.jsonl"
            path.write_text(json.dumps(record()) + "\n" + json.dumps(record([0, 0])) + "\n")
            result = cli(path)
        self.assertEqual(result.returncode, 1)
        self.assertIn("broken.jsonl:2", result.stderr)
        self.assertIn("duplicate", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_malformed_schema_empty_and_noninteger_inputs_fail(self):
        bad_schema = record()
        bad_schema["schema"] = "another/v1"
        bad_game = record()
        bad_game["game"] = True
        bad_option = record()
        del bad_option["observation"]["select"]["option"][0]["type"]
        cases = [('{"schema":', "JSON"), ('{"schema":"a","schema":"b"}', "duplicate"),
                 (json.dumps(bad_schema), "schema"), ("\n", "empty"),
                 (json.dumps(bad_game), "game"), (json.dumps(bad_option), "type")]
        for content, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "invalid.jsonl"
                path.write_text(content)
                result = cli(path)
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_missing_file_names_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.jsonl"
            result = cli(path)
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing.jsonl", result.stderr)


if __name__ == "__main__":
    unittest.main()
