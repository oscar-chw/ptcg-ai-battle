# CPU serving diagnostic acceptance contract

The user is debugging a serving graph that can differ from its numerical reference while
still returning structurally valid choices. Input: explicit reference/serving configurations,
model/input seeds and1..64 synthetic feature rows. Output: stable JSON with scope/provenance,
input/weight/source hashes, runtime versions, comparison criterion, matching-mode positive
control, per-row scores/choices and first mismatch. Fixture weights are randomly initialized;
no optimizer or game engine runs.

The actual existing NumPy forward is compared with the independently executed Torch graph on
identical weights/rows. The configured reference controls masks/relations explicitly. The
matching-mode baseline must pass before a candidate can be classified; failure yields
CONTROL_FAILED, never a reassuring candidate verdict. Original score tolerance1e-4 and exact
choice-agreement criterion stay unchanged. Shapes and all numbers must be finite/consistent;
empty vectors, NaN/Infinity or missing rows are errors rather than passing comparisons.

CLI modes: masked,unmasked,no-state-mask,no-option-mask,no-relations for serving; masked or
unmasked for reference. Default is masked/masked. Identical configurations PASS; deliberate
wrong modes produce MISMATCH on the declared witness, showing first differing row and option
and a recovery command. Matching a deliberately unmasked reference must remain explicitly
labelled agreement with a synthetic configuration, not certification of trained behavior.

The normal report is deterministic within the recorded environment for the same seeds. All
reference globals changed by the fixture helper must be restored on success/error. Generated
offered-option menus validate choices through the existing decision audit; they remain synthetic
encoded-feature fixtures, not legal games. CLI invalid parameters fail before inference and
missing dependencies give an actionable environment error. Numerical mismatch exit1; usage/
setup errors exit2; successful control+candidate agreement exit0.

Acceptance: additive stdlib comparison/CLI-contract cases and CPU numerical differential cases,
positive/mismatching/same-unmasked controls, deterministic hashes/report, finite/shape failures,
first mismatch coordinates, menu-choice consistency, restoration on error, and unchanged full
repo checks. New CI documents the additional numerical command; remote CI stays NOT_RUN until
release. Independent source review and root acceptance are separate from author self-check.
