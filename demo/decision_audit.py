"""Validate an issued selection menu, not Pokemon card rules or game transitions."""

from __future__ import annotations

import json
from collections import Counter
from typing import Iterable

SCHEMA = "ptcg-decision/v1"


class DecisionError(ValueError):
    """A state, response or trace cannot satisfy the declared selection contract."""


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise DecisionError(f"{where}: expected an object")
    return value


def _integer(value, where: str) -> int:
    if type(value) is not int:
        raise DecisionError(f"{where}: expected an integer, not {type(value).__name__}")
    return value


def read_result(observation: dict) -> int:
    current = _mapping(_mapping(observation, "observation").get("current"), "current")
    return _integer(current.get("result"), "current.result")


def selection_menu(observation: dict) -> dict:
    """Copy the issued contract before agent code can mutate the observation."""
    if read_result(observation) != -1:
        raise DecisionError("current.result: only ongoing decisions may be selected")
    current = observation["current"]
    seat = _integer(current.get("yourIndex"), "current.yourIndex")
    if seat not in (0, 1):
        raise DecisionError("current.yourIndex: expected seat 0 or 1")
    select = _mapping(observation.get("select"), "select")
    kind = _integer(select.get("type"), "select.type")
    minimum = _integer(select.get("minCount"), "select.minCount")
    maximum = _integer(select.get("maxCount"), "select.maxCount")
    if minimum < 0:
        raise DecisionError("select.minCount: must be nonnegative")
    if maximum < minimum:
        raise DecisionError("select.maxCount: must be at least minCount")
    options = select.get("option")
    if not isinstance(options, list):
        raise DecisionError("select.option: expected a list")
    if minimum > len(options):
        raise DecisionError("select.minCount: exceeds number of offered options")
    copied = []
    for index, option in enumerate(options):
        option = _mapping(option, f"select.option[{index}]")
        copied.append({"type": _integer(option.get("type"), f"select.option[{index}].type")})
    return {"current": {"result": -1, "yourIndex": seat},
            "select": {"type": kind, "minCount": minimum, "maxCount": maximum, "option": copied}}


def validate_decision(observation: dict, selected: list[int]) -> None:
    menu = selection_menu(observation)
    select = menu["select"]
    if not isinstance(selected, list):
        raise DecisionError("selected: expected a list of option indices")
    count = len(select["option"])
    for index in selected:
        _integer(index, "selected index")
        if not 0 <= index < count:
            raise DecisionError(f"selected index {index}: outside offered range 0..{count - 1}")
    if len(set(selected)) != len(selected):
        raise DecisionError("selected: duplicate option indices")
    maximum = min(select["maxCount"], count)
    if not select["minCount"] <= len(selected) <= maximum:
        raise DecisionError(f"selected: {len(selected)} choices, allowed {select['minCount']}..{maximum}")


def decision_record(observation: dict, selected: list[int], agent: str, game: int, step: int) -> dict:
    menu = selection_menu(observation)
    validate_decision(menu, selected)
    for name, value in (("game", game), ("step", step)):
        if _integer(value, name) < 0:
            raise DecisionError(f"{name}: must be nonnegative")
    if not isinstance(agent, str) or not agent.strip():
        raise DecisionError("agent: expected a nonempty name")
    return {"schema": SCHEMA, "game": game, "step": step, "agent": agent,
            "observation": menu, "selected": list(selected)}


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecisionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise DecisionError(f"invalid JSON numeric constant {value}")


def audit_lines(lines: Iterable[str], source: str) -> dict:
    agents: dict[str, Counter] = {}
    decisions = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = _mapping(json.loads(line, object_pairs_hook=_unique_fields,
                                        parse_constant=_invalid_constant), "record")
            if entry.get("schema") != SCHEMA:
                raise DecisionError(f"schema: expected {SCHEMA}")
            admitted = decision_record(entry.get("observation"), entry.get("selected"),
                                       entry.get("agent"), entry.get("game"), entry.get("step"))
        except json.JSONDecodeError as error:
            raise DecisionError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
        except RecursionError as error:
            raise DecisionError(f"{source}:{line_number}: JSON nesting exceeds parser limit") from error
        except DecisionError as error:
            raise DecisionError(f"{source}:{line_number}: {error}") from error
        counts = agents.setdefault(admitted["agent"], Counter(decisions=0, passes=0, selected_options=0))
        counts["decisions"] += 1
        counts["passes"] += int(not admitted["selected"])
        counts["selected_options"] += len(admitted["selected"])
        decisions += 1
    if decisions == 0:
        raise DecisionError(f"{source}: empty trace; no decisions to audit")
    return {"schema": "ptcg-decision-audit/v1", "decisions": decisions,
            "agents": {name: dict(counts) for name, counts in sorted(agents.items())},
            "scope": "selection-menu consistency only; no engine replay, authenticity or playing-strength proof"}
