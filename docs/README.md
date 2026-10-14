# docs/

Where to find each detail behind the [README](../README.md).

## Understand it

| Page | What it answers |
|---|---|
| [DIAGRAMS.md](DIAGRAMS.md) | How do the parts connect? The numbered diagrams (overview, the false-green parity check, training to serving, the GPU prototype), each with the files it draws. |
| [decision-audit.md](decision-audit.md) | How can I validate an issued selection menu and locate an invalid agent response without the engine or weights? Runnable original synthetic examples and scope. |
| [serving-diagnostic.md](serving-diagnostic.md) | How can I inspect the first Torch/NumPy score mismatch, controls and recovery on seeded CPU features without pretending to reproduce the competition model? |
| [architecture.pdf](architecture.pdf) (source: [architecture.tex](architecture.tex)) | How does the set-transformer agent work? The team research lane's 9-page write-up, dated 2026-08-16. Every quantitative claim carries one of five evidence tags (theorem, practice, measured, speculative, unsourced), and a "Claims withdrawn" section records earlier claims that did not survive measurement. Build: `tectonic docs/architecture.tex` (or `pdflatex` twice); not re-run here, so the PDF is the original build. |

## Check the evidence

| Page | What it answers |
|---|---|
| [details.md](details.md) | What are all the results, including the negative ones? The full results table, the negative results, the serving-package scores whose cause was not recorded, the value head, and the long form of the design decisions, each with its source. |
| [gpu-prototype.md](gpu-prototype.md) | How fast is the GPU prototype of a simplified game loop, how was it checked, and what does it not show? |
| [decision-audit-contract.md](decision-audit-contract.md) | What menu/trace behavior does the auditor promise, and which behavior belongs to the engine? |
| [serving-diagnostic-contract.md](serving-diagnostic-contract.md) | Which CPU differential controls and scope limits must the diagnostic preserve? |
| [ptcg-tooling-handoff-2026-10-14.md](ptcg-tooling-handoff-2026-10-14.md) | Which decision-audit source and local evidence were frozen on 14 October? |
| [serving-diagnostic-handoff-2026-10-14.md](serving-diagnostic-handoff-2026-10-14.md) | Which synthetic CPU source did root independently accept, with what exclusions? |
| [licensing.md](licensing.md) | Which original parts are MIT, and which team/engine materials remain restricted? |

## Reference, elsewhere in the repository

| Page | What it answers |
|---|---|
| [report/REPORT.md](../report/REPORT.md) | The team's case study: corpus, evaluation, search, the value head. |
| [report/LIMITATIONS.md](../report/LIMITATIONS.md) | What cannot be reproduced or was never measured. |
| [results/negative_results.md](../results/negative_results.md) | The negative results in full. |
| [results/ppo_RESULTS.md](../results/ppo_RESULTS.md) | The PPO head-to-head records and what was still open. |
| [imitation/README.md](../imitation/README.md) | The model, the NumPy serving and the four packaging gates. |
| [ppo/README.md](../ppo/README.md) | The STOP head and the PPO design. |
| [engine/README.md](../engine/README.md) | The GPU prototype's raw records. |
| [demo/README.md](../demo/README.md) | Running a live match against the official engine. |
