"""Every number drawn in figures/results.png traces to a committed source. Standard library only."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_results  # noqa: E402  (matplotlib is imported only inside render())
import ppo_counts  # noqa: E402


class FigureData(unittest.TestCase):
    def test_battery_annotations_match_the_committed_analysis(self):
        # The analysis file is what analyze_results.py prints; check.sh diffs the two.
        analysis = (ROOT / "results" / "analysis_output.txt").read_text(encoding="utf-8")
        rows = plot_results.battery_rows()
        self.assertEqual(len(rows), len(plot_results.BATTERY))
        for (key, _, _), row in zip(plot_results.BATTERY, rows):
            self.assertIn(f"{key}: {row[5]}\n", analysis)
            self.assertLessEqual(row[3], row[2])
            self.assertLessEqual(row[2], row[4])

    def test_each_battery_label_names_its_experiment(self):
        # A label drawn over another experiment's numbers is the error this catches.
        expected = {"baseline": "Rule/search baseline", "baseline_first": "first seat",
                    "baseline_second": "second seat", "gumbel_same": "Gumbel candidate vs same-deck",
                    "ismcts_same": "ISMCTS vs same-deck", "ismcts_cross": "ISMCTS cross-deck"}
        self.assertEqual({key: label for key, label, _ in plot_results.BATTERY if expected[key] in label},
                         {key: label for key, label, _ in plot_results.BATTERY})
        self.assertEqual(sorted(k for k, _, _ in plot_results.BATTERY), sorted(expected))

    def test_ppo_quotes_are_verbatim_in_the_results_file(self):
        recorded = (ROOT / "results" / "ppo_RESULTS.md").read_text(encoding="utf-8")
        rows = plot_results.ppo_rows()
        self.assertEqual(len(rows), 4)
        for label, role, rate, low, high, note in rows:
            quote, derived = note.split("  = ")
            self.assertIn(quote, recorded, label)
            self.assertRegex(derived, r"^\d+(\.5)?/\d+$")
            self.assertIn(role, {"candidate", "control"})
            self.assertTrue(low <= rate <= high, label)
        best = max(rows, key=lambda r: r[2])
        self.assertEqual((best[2], best[3], best[4]), (0.8104, 0.756, 0.855))

    def test_game_counts_are_the_only_ones_the_intervals_allow(self):
        got = {r["quote"]: ppo_counts.solutions(r["quote"]) for r in ppo_counts.rows()}
        self.assertEqual(got, {"0.4875 [0.419, 0.556]": [(97.5, 200)],
                               "0.7975 [.736,.847]": [(159.5, 200)],
                               "0.7675 [.704,.821]": [(153.5, 200)],
                               "0.8104 [.756,.855]": [(194.5, 240)]})

    def test_the_search_can_fail(self):
        # An interval no half-point score reproduces must come back empty, not forced.
        self.assertEqual(ppo_counts.solutions("0.8104 [.600,.990]"), [])

    def test_an_unparseable_quote_is_refused(self):
        self.assertIsNone(plot_results.QUOTE.match("about 0.81"))
        self.assertIsNotNone(re.match(plot_results.QUOTE, "0.4875 [0.419, 0.556]"))


if __name__ == "__main__":
    unittest.main()
