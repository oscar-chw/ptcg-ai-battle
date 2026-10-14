"""An explicit JSON depth budget must not depend on decoder recursion behavior."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import decision_audit as api  # noqa: E402


def record_with(extra):
    record = json.loads((DEMO / "examples/decision-menu.synthetic.jsonl").read_text().splitlines()[0])
    record["extra"] = extra
    return json.dumps(record)


class JsonNesting(unittest.TestCase):
    def test_permissive_decoder_does_not_change_excessive_nesting_diagnostic(self):
        # A decoder that completes the deep array still returns an array, not a record.
        # The audit must classify the depth before schema/type validation instead.
        line = "[" * 10000 + "0" + "]" * 10000
        with patch.object(api.json, "loads", return_value=[]), \
                self.assertRaisesRegex(api.DecisionError, r"trace.jsonl:2:.*nesting"):
            api.audit_lines(["\n", line], "trace.jsonl")

    def test_256_container_boundary_is_admitted(self):
        extra = 0
        for _ in range(255):  # The record object contributes the remaining level.
            extra = [extra]
        result = api.audit_lines([record_with(extra)], "boundary.jsonl")
        self.assertEqual(result["decisions"], 1)

    def test_257_container_boundary_is_rejected_with_source_line(self):
        extra = 0
        for _ in range(256):
            extra = [extra]
        with self.assertRaisesRegex(api.DecisionError, r"boundary.jsonl:1:.*nesting.*256"):
            api.audit_lines([record_with(extra)], "boundary.jsonl")

    def test_quoted_brackets_and_escaped_quotes_do_not_consume_depth(self):
        extra = {"text": ('\\\"' + "[{" * 10000 + ']}\\\\\"')}
        self.assertEqual(api.audit_lines([record_with(extra)], "quoted.jsonl")["decisions"], 1)

    def test_mixed_arrays_and_objects_share_the_same_depth_budget(self):
        extra = 0
        for _ in range(128):
            extra = {"next": [extra]}
        with self.assertRaisesRegex(api.DecisionError, r"mixed.jsonl:1:.*nesting"):
            api.audit_lines([record_with(extra)], "mixed.jsonl")


if __name__ == "__main__":
    unittest.main()
