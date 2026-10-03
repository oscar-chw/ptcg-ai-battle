# Negative results

Each of these is a result that did not meet its bar, reported as recorded.

## 1. Spidops deck model missed its baseline: 0.5714 against 0.5943

The held-out top-1 gate for a deck model is the dense-scorer baseline, 0.5943
(`../imitation/training/train_ss.py`, module docstring and `BASELINE`). The Team Rocket's
Spidops model reached 0.5714, so it failed the gate and the deck was dropped
([team_results.md](team_results.md), table row "Team Rocket's Spidops"). The team traced
the failure to the data, not to training: the player being imitated won 6.2% of the 48
available episodes (45 losses), so the model faithfully reproduced a losing player
([team_results.md](team_results.md), "Imitate one strong player, not an archetype").

## 2. Older distillation line: the fidelity check failed, 0.376667 against a 0.55 target

An earlier repository (`kaggle-ptcg-model`, private) distilled a legacy teacher into a
public-feature model and gated it on same-state top-1 agreement with the teacher. Its
README records:

> Baseline V61 clean-public evidence failed the go/no-go gate: `top1_agreement=0.376667`, `p95_regret=0.831106`, failure=`top1<0.55`.

Later phases improved it but never passed. The best result it records is
`top1=0.496667` (`p95=0.662796`) after a support/clipping grid, "every grid member
failed only `top1<0.55`". The line was not carried forward. This metric is agreement with
a distillation teacher on a fixed set of decisions; it is not the validation top-1 in
section 1, and the two numbers should not be compared.

## 3. Search did not help: ISMCTS won 111/400 and 108/400

From [results.csv](results.csv), recomputed by
`python3 ../report/analyze_results.py` ([analysis_output.txt](analysis_output.txt)):

| experiment | wins / games | win rate, nominal Wilson 95% | opponent |
|---|---|---|---|
| `ismcts_same` | 111 / 400 | 27.75% [23.59, 32.33] | the same-deck frozen baseline |
| `ismcts_cross` | 108 / 400 | 27.00% [22.88, 31.55] | a cross-deck battery |
| `baseline` (for scale) | 198 / 400 | 49.50% [44.63, 54.38] | the corrected rule/search baseline against a cross-deck control |

The ISMCTS candidate was rejected on both. The three rows have different opponents, so
they are not pooled into one comparison (see `provenance.json`, scopes), and the
intervals are per-sample, not across opponents. The ISMCTS result is for that
implementation, not for search in general. A Gumbel candidate against the same-deck
baseline scored 204 / 400 = 51.00% [46.11, 55.87], which is no evidence of improvement
either way.
