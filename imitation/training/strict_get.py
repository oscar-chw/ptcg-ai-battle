#!/usr/bin/env python3
"""Absent keys that must FAIL rather than default to a plausible value.

WHY THIS MODULE EXISTS. Four incidents in one night shared one shape:

    --index    defaulted to the partial pairs dump   -> 725 phantom episodes
    --no-tf32  inherited torch's library defaults    -> cuDNN TF32 on regardless
    --main     defaulted to the unmasked v6 forward  -> 0.7926 self-agreement
    --pkg-local-tags off by default                  -> agent played at random

In every case the artefact produced was complete and well-formed, and every gate
read green, because the substituted value was a legal one. A default is only
acceptable when its correctness has been MEASURED and the measurement is written
down next to it. Everything else raises, naming the key.

THE MEASUREMENT BEHIND `require_option_type`. The engine's option records were
counted directly over data/decision_corpus_v4.jsonl.gz (8,869 decision rows):

    option dicts scanned : 6,438,967
    carrying "type"      : 6,438,967
    missing "type"       : 0          (100.0000% presence)

So the old `option.get("type", 0)` never legitimately fired -- raising costs
nothing on real data. What it DID do is silently relabel any option that ever
lacked the key as OptionType.NUMBER, because **0 is a real option type**, not a
sentinel. Measured over 4,001 rows of a built corpus, option_type 0 occurs 48
times (0.02% of slots), so a mislabelled row lands inside the expected frequency
of a genuine value and no downstream distribution check can see it.
"""
from __future__ import annotations

from typing import Any, Mapping


class AbsentKeyError(KeyError):
    """A key whose absence means something is wrong, not something is default."""

    def __str__(self) -> str:  # KeyError repr quotes the message; this does not
        return self.args[0] if self.args else ""


def require_key(mapping: Mapping[str, Any], key: str, where: str, why: str) -> Any:
    """Return ``mapping[key]`` or raise an error that names the key and the fix."""
    if not isinstance(mapping, Mapping) or key not in mapping:
        present = sorted(mapping)[:12] if isinstance(mapping, Mapping) else type(mapping).__name__
        raise AbsentKeyError(
            f"{where}: required key {key!r} is absent. {why} "
            f"Present keys: {present}"
        )
    return mapping[key]


def require_option_type(option: Mapping[str, Any],
                        where: str = "engine option record") -> int:
    """The option's declared type, never a guessed one.

    See the module docstring for the 6,438,967-of-6,438,967 presence measurement
    and for why 0 is the worst possible default here.
    """
    return int(require_key(
        option, "type", where,
        "Every engine option carries it (measured 6,438,967 of 6,438,967). "
        "Defaulting to 0 would relabel this option as OptionType.NUMBER -- a "
        "REAL type -- and poison the corpus invisibly.",
    ))
