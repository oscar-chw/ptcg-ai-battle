# CPU serving diagnostic: frozen local successor

Source revision: `bdaf344dba4503ab0d1205913a81de88aca345dd` on
`codex/ptcg-serving-diagnostic-20261014`. Preserved predecessor:
`2db4490ed2ee9ef8be032076df8799d6c0736470`; its separate root acceptance covers
only its original no-engine/numeric scope. Root independently accepted this
successor at the exact source revision above; [receipt excerpt](serving-diagnostic-root-acceptance.json)
binds the original receipt SHA256. No push or public release has occurred.

The [runnable diagnostic](serving-diagnostic.md) compares the actual Torch
reference with the existing NumPy forward, using identical seeded random
weights and encoded synthetic rows. JSON exposes graph flags, both score
vectors, earliest failing row/max-gap option, choice disagreements and hashes.
A matching-mode positive control must pass before a candidate verdict exists.
Changing the serving mode back to the reference recovers on the same fixture.

Seed0/eight-row witness: masked control maximum error `8.260142762839262e-8`.
Unmasked serving maximum error `0.03391529598635479`; all eight choices still
agree. Removing either mask or relations also changes scores. This illustrates
why argmax agreement alone can miss a changed function. It does not measure
competition performance. Same-unmasked agreement remains explicitly synthetic.

Original contribution is the bounded diagnostic, comparison/report layer,
control/recovery, fingerprints and structural menu validation. Existing
team-owned kernels, competition claims and licences are unchanged. Acceptance
was written/run red before implementation; the acceptance-only commit was
delayed until implementation existed uncommitted. The separately committed
criteria are `c8817e0` and `3a6a5f8`, not a retrospective claim of commit order.

Local evidence:

- Eight numerical cases PASS after a retained FAIL exposed implicit meta-device
  construction. Explicit CPU/float32 construction now restores caller defaults.
- The unchanged full gate PASS: ten groups, zero failed/skipped,21s after the
  final source/test change. It also collects seven new stdlib comparator cases.
- Ruff and `git diff --check` PASS. All25 predecessor acceptance/check/workflow
  files are byte-identical; all seven evaluated source hashes bind this revision.
- Three comparison mutations are rejected, with a valid control after restoration.
- [Evidence](serving-diagnostic-evidence.json), [source pins](serving-diagnostic-source-pins.json),
  [paired witnesses](serving-diagnostic-eval.json), [mutations](serving-diagnostic-mutations.json)
  and dated node9/node10 receipts retain scope and earlier outcomes.

Root read the complete diagnostic/tests/workflow and independently ran these
commands from a fresh immutable archive of the source revision:

```bash
python3 -m unittest discover -s demo/tests -p test_serving_contract.py
"$PTCG_PYTHON" -m unittest discover -s demo/tests_numeric
env -u PTCG_ENGINE_DIR PTCG_PYTHON="$PTCG_PYTHON" bash scripts/check.sh
```

Root used the already approved CPU numeric environment with the engine unset.
Seven stdlib tests, eight CPU tests and the unchanged ten-group full check passed
(full check21.53s); all34 source/check pins stayed unchanged, including25
predecessor acceptance files. Root independently repeated the matched/mismatching
CLI controls on identical input/weight hashes. This separate receipt accepts the
bounded synthetic CPU diagnostic; author checks above remain separately labelled.
The routing record is SHADOW with dispatch disabled and unverified runtime, not
authority to launch a differently configured reviewer or clear a release hold.

The API temporarily changes collator globals and supports separate CLI processes,
not concurrent calls in one process. Fixtures are synthetic encoded features,
not legal games. Real-engine games/replay, trained-model/corpus reproduction,
teacher/training/GPU work, performance measurement, remote CI and release remain
NOT_RUN. No older teacher, thermal, independent-release or publication gate is
cleared by this work.
