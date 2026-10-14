# Local PTCG decision-audit handoff — 14 October 2026

Implementation source: `29af384e88fee1b9f1d679d268f09c533ac1ef05`, branch
`codex/ptcg-decision-audit-20261014`, from preserved public base `45f2703`.
Acceptance-first commits: `adb6132`, `9f69f95`, `14e35d3`. No public push.

## Useful component delivered

The owned stdlib match harness now validates observer/selection-menu fields and agent response
indices before dispatch. It rejects duplicates, booleans, out-of-range indices and invalid choice
counts; it freezes the issued menu so agent mutation cannot relax that contract. Optional
`--trace` writes minimal decision menus to a new file, never overwriting previous evidence.
The independent stdlib audit reports the first invalid file/line and per-agent counters.
Startup, deck-refusal, action and trace-error exits run engine cleanup. JSON depth/format problems
become explicit diagnostic failures. All original scripts/check.sh/CI/test assertions stay intact.

## Runnable input and output

```bash
python3 demo/audit_trace.py demo/examples/decision-menu.synthetic.jsonl
python3 demo/audit_trace.py demo/examples/invalid-selection.synthetic.jsonl
```

The first exits0 with3 decisions: fixture_a2decisions/3selected options, fixture_b1decision/1pass.
The second exits1 with `:2: selected: duplicate option indices` and no success summary.
These are original SYNTHETIC menus, not engine game states or a competition-agent demonstration.
Full schema, capture instructions and caveats: [decision-audit.md](decision-audit.md).

## Verification

`plan gate decision-audit 7`: exit0,27focused demo tests. Repo Ruff: exit0.
Five mutation controls catch boolean coercion, relaxed index bounds, mutable menus, duplicate
JSON keys and nonstandard JSON constants; valid controls pass. The fresh source review found
two defects, reproduced as3failing subcases before repair; all27 focused cases then passed.

Final post-repair `plan gate decision-audit 8`: exit0 in21s;10groups ran,0failed,0skipped.
The same bounded reviewer re-read the two repairs and tests and found no remaining causal defects
in them; this remains source review rather than release certification.
The earlier10-group fullcheck passed before those final repairs and is retained as earlier
evidence only. `docs/decision-audit-evidence.json` records exact commands/outcomes and binds the
portable excerpt to locally retained raw output. Root owns independent acceptance/public release.

## Scope and holds

The menu declares legal individual options; the engine still decides coupled-choice legality,
card rules and state transitions. Traces record an attempted response before engine acceptance;
the audit does not authenticate origin, validate whole-game chronology or recover omitted rows.
Real-engine execution is NOT_RUN. No weights/corpus/final BEST1_fixed archive was recovered in
the bounded named-path check; this is not a new global artifact-absence claim. No training,
GPU job, performance measurement or historical teacher-strength/admission gate ran.

Existing team attribution, historical competition/rank records and simplified GPU prototype
versus full-game distinction are unchanged. New demo code/examples/audit docs are Oscar-owned
MIT tooling; team imitation/report content and competition-only engine retain their licenses.

Potential CV contribution, pending the normal claims review: implemented validation and trace
diagnostics for engine-issued card-game decisions, catching malformed responses before dispatch.
This is an engineering component claim, without model-strength or competition-performance uplift.
Website owner can reuse the declared JSONL input/output contract to expose actual validation;
no fake terminal or synthetic board should be presented as BEST1_fixed playing.
