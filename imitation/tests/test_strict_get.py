"""strict_get: an absent key must raise, naming the key; present keys pass through."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from strict_get import AbsentKeyError, require_key, require_option_type  # noqa: E402


class StrictGet(unittest.TestCase):
    def test_present_key_is_returned(self):
        self.assertEqual(require_key({"a": 3}, "a", "here", "why"), 3)

    def test_absent_key_raises_and_names_it(self):
        with self.assertRaises(AbsentKeyError) as ctx:
            require_key({"b": 1}, "a", "here", "because")
        self.assertIn("'a'", str(ctx.exception))
        self.assertIn("here", str(ctx.exception))

    def test_zero_is_a_real_type_not_a_default(self):
        # option type 0 is OptionType.NUMBER; a present 0 must survive...
        self.assertEqual(require_option_type({"type": 0}), 0)

    def test_missing_option_type_is_not_silently_zero(self):
        # ...and a missing type must NOT become 0.
        with self.assertRaises(AbsentKeyError):
            require_option_type({"index": 4})

    def test_non_mapping_is_rejected(self):
        with self.assertRaises(AbsentKeyError):
            require_key(["type"], "type", "here", "why")


if __name__ == "__main__":
    unittest.main()
