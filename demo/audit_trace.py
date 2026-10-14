#!/usr/bin/env python3
"""Inspect a JSONL decision trace and identify the first invalid selection menu/response."""

import argparse
import json
import sys
from pathlib import Path

from decision_audit import DecisionError, audit_lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="ptcg-decision/v1 JSONL; engine not required")
    args = parser.parse_args(argv)
    try:
        with args.trace.open(encoding="utf-8") as source:
            summary = audit_lines(source, str(args.trace))
    except DecisionError as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError) as error:
        print(f"audit failed: {args.trace}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
