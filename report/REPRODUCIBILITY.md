# Reproducing the aggregate statistics

Run with Python 3.8 or later and its standard library:

```sh
python3 report/analyze_results.py
```

The script reads `../results/results.csv` and `../results/provenance.json` regardless of the current directory. It performs no network requests, installation, game simulation or training. This capsule reproduces arithmetic from historical aggregate records; it does not reproduce the final competition agent, models, games, engine behavior or full evaluation environment.

`results.csv` uses one row per experiment/metric. Outcome counts, Brier class rates, timing medians and rollout counts remain separate. The baseline's first/second-seat rows are strata of the same 400 games, not additional observations. No comparison pools opponents or assumes Gumbel's official-engine shuffles were paired.

`provenance.json` records sanitized relative source labels, SHA256 digests of the inspected original records, field locators and scope. Original records are not redistributed. These digests identify source bytes but do not prove external authentication or make private source files available. The CSV digest detects changes to this exported snapshot. The package includes no raw replays, weights, engine code, personal identities, credentials or account/remote state.

The calculations are:

- Nominal two-sided 95% Wilson intervals for wins out of games, treating draws as non-wins. These intervals describe the samples; they are not dependency-adjusted uncertainty over all opponents or paired treatment effects.
- First-seat win rate minus second-seat win rate: **17.00 percentage points**, a descriptive difference without a causal interval.
- Multiclass sum Brier loss for the class-frequency constant predictor: `1 - sum(p_c ** 2)`, yielding **0.487022**. Published rates are rounded; the script permits a small explicit rounding tolerance. The 17,392 decision rows come from 185 games and must not be treated as 17,392 independent outcomes.
- The ratio of published timing medians, **3.814307209 / 2.546774833 = 1.497701x**. This is incremental speedup on the tested macOS arm64 workload. The additional 2x target failed. The 50-game/9,844-move parity result is a separate semantic check, not a confidence interval for speed or proof of strategic improvement.
- The rollout gate records **32/32 completed episodes, 5,513 decisions and zero illegal actions**. That is an instrumentation result, not completed policy learning or a universal correctness guarantee.

The script rejects missing/duplicate/unknown metrics, malformed units, invalid counts, inconsistent win/loss/draw totals, inconsistent seat strata, implausible probabilities, mismatching arithmetic and changes to the recorded CSV digest. An alternate CSV can be checked with `--results PATH`; contradictory counts are checked before the snapshot digest so their failure is explicit. Invalid inputs exit with status 2 and print `ERROR` to stderr.

Verification performed during preparation: the valid capsule exited 0 and printed the rounded statistics above. A temporary CSV outside the deliverable directory changed baseline wins from 198 to 401; it exited 2 with `baseline: wins + losses + draws must equal games`. The temporary file was removed. Source games and training were not rerun.
