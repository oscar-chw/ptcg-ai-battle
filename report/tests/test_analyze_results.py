"""analyze_results.py reproduces the committed results table, and rejects edited inputs."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
RESULTS = REPORT.parent / "results"
sys.path.insert(0, str(REPORT))
from analyze_results import wilson  # noqa: E402


def run(*args):
    return subprocess.run([sys.executable, str(REPORT / "analyze_results.py"), *args],
                          capture_output=True, text=True)


def edited_csv(old, new):
    text = (RESULTS / "results.csv").read_text()
    assert old in text
    path = Path(tempfile.mkdtemp()) / "results.csv"
    path.write_text(text.replace(old, new, 1))
    return path


class AnalyzeResults(unittest.TestCase):
    def test_output_equals_the_snapshot_the_readme_table_is_built_from(self):
        r = run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, (RESULTS / "analysis_output.txt").read_text())

    def test_wilson_interval_matches_the_published_baseline_row(self):
        low, high = wilson(198, 400)
        self.assertEqual((round(100 * low, 2), round(100 * high, 2)), (44.63, 54.38))

    def test_contradictory_counts_are_rejected_before_the_digest_is_checked(self):
        r = run("--results", str(edited_csv("baseline,wins,198", "baseline,wins,401")))
        self.assertEqual(r.returncode, 2)
        self.assertIn("wins + losses + draws must equal games", r.stderr)

    def test_a_consistent_edit_still_fails_the_snapshot_digest(self):
        r = run("--results", str(edited_csv("rollout_gate,requested,32", "rollout_gate,requested,33")))
        self.assertEqual(r.returncode, 2)
        self.assertIn("SHA256", r.stderr)

    def test_the_negative_results_are_in_the_data(self):
        text = (RESULTS / "results.csv").read_text()
        for row in ("ismcts_same,wins,111", "ismcts_cross,wins,108", "baseline,wins,198"):
            self.assertIn(row, text)


if __name__ == "__main__":
    unittest.main()
