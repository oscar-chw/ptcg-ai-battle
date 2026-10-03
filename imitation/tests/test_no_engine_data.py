"""Guard for the licence boundary: nothing from the official engine is in this repository.

The engine is licensed for competition use only (LicenseRef-PTCG-ABC-Competition-Use-Only).
`featurize.py` must read card and attack tables from the reader's own install at run time,
so it must not carry them. This cannot prove a negative, but it fails on the shapes the
data would take: a large literal table, an engine binary or source file, a hardcoded
engine path.
"""
import ast
import os
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAX_CONSTANTS_IN_ONE_LITERAL = 30     # the largest real one today holds 21 (effect-tag names)
ENGINE_EXTENSIONS = {".dylib", ".so", ".dll", ".h", ".hpp", ".cpp", ".cc", ".rs"}
FORBIDDEN_TEXT = ("libcg", "content/data", "ptcg_engine/")


def tracked_files():
    for folder, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv"}]
        for name in files:
            yield Path(folder) / name


def constants_in(node):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return sum(isinstance(e, ast.Constant) for e in node.elts)
    if isinstance(node, ast.Dict):
        return sum(isinstance(v, ast.Constant) for v in node.values)
    return 0


def largest_literal(source):
    return max((constants_in(n) for n in ast.walk(ast.parse(source))), default=0)


class NoEngineData(unittest.TestCase):
    def test_no_python_file_embeds_a_large_constant_table(self):
        for path in tracked_files():
            if path.suffix == ".py" and "tests" not in path.parts:
                size = largest_literal(path.read_text())
                self.assertLessEqual(
                    size, MAX_CONSTANTS_IN_ONE_LITERAL,
                    f"{path.relative_to(REPO)} holds a literal with {size} constants; "
                    f"card or attack tables must be read from PTCG_ENGINE_DIR, not embedded")

    def test_the_detector_sees_a_planted_table(self):
        table = "CARDS = [" + ", ".join(str(i) for i in range(500)) + "]"
        self.assertGreater(largest_literal(table), MAX_CONSTANTS_IN_ONE_LITERAL)

    def test_no_engine_binary_or_source_file_is_present(self):
        found = [p.relative_to(REPO) for p in tracked_files() if p.suffix in ENGINE_EXTENSIONS]
        self.assertEqual(found, [])

    def test_no_python_file_names_an_engine_location(self):
        for path in tracked_files():
            if path.suffix in {".py", ".sh"} and path.name != Path(__file__).name:
                text = path.read_text()
                for needle in FORBIDDEN_TEXT:
                    self.assertNotIn(needle, text, f"{path.relative_to(REPO)} mentions {needle!r}")


if __name__ == "__main__":
    unittest.main()
