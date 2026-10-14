# Inspect selections before debugging a whole match

A policy can return a duplicate, a negative index, the wrong number of choices or a response
for a malformed observer seat. This tool catches those at the agent/engine boundary and identifies
the first offending line in a recorded trace. It uses the same validator as the stdlib match harness.
It checks the supplied menu's field types, observer seat, ongoing-decision state, bounds, distinct
indices and index range. The official engine still checks card rules, dependencies between choices
and game transitions; the auditor does not execute those rules.

## Run without an engine or model

```bash
python3 demo/audit_trace.py demo/examples/decision-menu.synthetic.jsonl
```

This example contains **SYNTHETIC**, original hand-authored menus, not competition games. Option
type numbers are opaque declarations here, not a card database or rule simulator. The result is:

```json
{"agents":{"fixture_a":{"decisions":2,"passes":0,"selected_options":3},"fixture_b":{"decisions":1,"passes":1,"selected_options":0}},"decisions":3,"schema":"ptcg-decision-audit/v1","scope":"selection-menu consistency only; no engine replay, authenticity or playing-strength proof"}
```

`fixture_b` took the permitted zero-option response once. These counts do not measure strength.

```bash
python3 demo/audit_trace.py demo/examples/invalid-selection.synthetic.jsonl
# exit 1, no success summary:
# audit failed: demo/examples/invalid-selection.synthetic.jsonl:2: selected: duplicate option indices
```

Fix the producer so each chosen index appears once; do not alter the issued menu to admit a bad
response. Too few/many choices, booleans, floats, missing fields, malformed JSON, duplicate JSON
keys, NaN/Infinity, empty traces and unreadable files also fail. The input is read-only and parsed
one line at a time; the audit retains per-agent counters, not the whole trace.

## Capture your own issued menus

With an already authorized local engine installation:

```bash
PTCG_ENGINE_DIR=/path/to/sample_submission python3 demo/run_match.py --games 2 --trace decisions.jsonl
python3 demo/audit_trace.py decisions.jsonl
```

The output file must not exist. The harness copies the menu before invoking the agent, validates
the response against that copy and records game/step, agent, seat, bounds, option types and chosen
indices. Full hands, card/attack records, weights and unrelated observation fields are excluded.
This records an attempted response **before** engine processing; it does not prove acceptance or
match completion. Agent/I/O errors still release the engine handle, and a partial trace remains
diagnostic evidence. Default match output is unchanged.

This task did not run the real-engine command. Tests use original synthetic menus/test doubles.
No `BEST1_fixed` weights, corpus or final archive were recovered. No training, GPU benchmark or
historical admission gate ran. The original competition result and contribution boundaries stay.

## API and ownership

`demo/decision_audit.py` exposes `selection_menu`, `validate_decision`, `decision_record` and
`audit_lines`. Records declare schema `ptcg-decision/v1`, nonnegative integer `game`/`step`, nonempty
`agent`, `observation.current`/`observation.select` and `selected` list. Blank lines are ignored;
an input with no decisions fails. Records are checked independently: this does not authenticate
origin, infer transitions or detect omitted/reordered decisions in a game history.

The validator, capture path, CLI, examples, tests and these docs are Oscar's original MIT-owned
tooling. Team-owned imitation/report code and the competition-only engine keep their licenses.
A possible engineering contribution is: “Implemented validation and trace diagnostics for
engine-issued card-game decisions, rejecting malformed responses before dispatch.” This is local
component evidence pending root acceptance, not a new competition-performance claim.
