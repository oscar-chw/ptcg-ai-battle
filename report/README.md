# report/: the case study

*Mechanics, Learning, and Reliability in PTCG*: an engineering case study in learning over
structured game state, preserving model semantics at deployment, and evaluating agents
under imperfect information. It accompanies the team's Strategy-track report for the
competition (submitted 13 September 2026).

![Policy architecture and evaluation boundaries](architecture.svg)

- [REPORT.md](REPORT.md): game strategy, policy design, experiments, limitations and
  primary references.
- [METHODS.md](METHODS.md): conditional invariance proofs, value semantics, evaluation
  design.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): what `analyze_results.py` recomputes, and what
  it does not.
- [LIMITATIONS.md](LIMITATIONS.md): what is reproducible here and what stays unestablished.
- [REFERENCES.md](REFERENCES.md): the literature, and what each source supports.
- [analyze_results.py](analyze_results.py): standard library only; reads
  [../results/results.csv](../results/results.csv) and
  [../results/provenance.json](../results/provenance.json).

## Evidence in one table

| Question | Observed | What it establishes |
|---|---|---|
| Can variable game decisions be represented coherently? | typed entity/option attention policy with instance pointers; 89,048 decisions from 926 games | a concrete representation and dataset; final-agent generalisation is a separate question |
| Does a faster engine preserve measured behaviour? | 1.497701x incremental speedup; 50 games matched all outcomes and 9,844 moves | a bounded optimisation on the tested macOS build and workload |
| Does more elaborate search help this agent? | ISMCTS won 111/400 against its same-deck baseline | this candidate failed its strength gate |
| Is the value threshold meaningful? | constant-predictor sum-Brier loss 0.487022 | a threshold below 0.5 can pass without predictive skill on this validation set |
| Are selected rollout faults detectable? | 32 completed games, 5,513 decisions; 5/5 mutations and 4/4 corrupted decisions detected | evidence for the measured instrumentation, not for a completed RL improvement |

The final Simulation result was 2043/6807, score 709.1. The experiments above are earlier
development branches; their results are not the final agent's win rates. The case study
claims no medal and no demonstrated state-of-the-art policy.

Reproduce the statistics with `python3 report/analyze_results.py` from the repository
root. Pokémon and associated names are third-party trademarks. This is an independent
participant report; it contains no official artwork or redistributed simulator assets.
