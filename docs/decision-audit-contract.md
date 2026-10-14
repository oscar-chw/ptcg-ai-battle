# Selection-menu audit contract

Status: acceptance specified before implementation,14October2026.

Input is an engine-issued observation and a proposed list of option indices. The observation
contains current.result=-1, current.yourIndex in{0,1}, and select.{type,minCount,maxCount,option}.
Each offered option declares an integer type. Bounds are nonnegative integers with minCount
<=maxCount and minCount<=number of options. Effective ceiling=min(maxCount,option count).
The response must be a list of distinct integer indices in range, with cardinality within
those bounds. Boolean values are not indices or bounds. A zero selection is legal iff the
minimum is zero. Validation fails with a field-specific error; it never clips or repairs a
response. The engine still arbitrates dependencies between choices and all card/game rules.

The match loop validates the state before choosing a seat and the action before battle_select;
an invalid response must leave battle_select uncalled. battle_finish runs on both normal and
exceptional exits. Default match output stays unchanged. Optional --trace refuses to overwrite
an existing output and records minimal selection menus, not full private hands or model weights.

Each JSONL record declares schema=ptcg-decision/v1, game/step nonnegative integer IDs, a nonempty
agent name, observation.current/observation.select and selected. A streaming audit uses the same
validator, reports counts by agent and stops on the first invalid record naming source and line.
Malformed/truncated JSON, duplicate JSON keys, wrong schema and an empty trace fail. Blank lines
are not decisions. Audit preserves caller input; no engine, cards, learning framework or network
is imported. It verifies menu/selection consistency only, not trace authenticity, transition
reproducibility, agent strength, or a competition submission.

Examples and tests are original hand-authored selection menus, labelled SYNTHETIC; they do not
contain engine files/card tables or represent real competition games. The engine-installed run
and historical teacher/training gates stay unrun by this task. No new scientific result or
public claim is approved by a successful local software audit.
