# docs/

Where to find each detail behind the [README](../README.md).

## Understand it

| Page | What it answers |
|---|---|
| [DIAGRAMS.md](DIAGRAMS.md) | How do the parts connect? The numbered diagrams (overview, the false-green parity check, training to serving, the GPU prototype), each with the files it draws. |
| [architecture.pdf](architecture.pdf) (source: [architecture.tex](architecture.tex)) | How does the set-transformer agent work? The team research lane's 9-page write-up, dated 2026-08-16. Every quantitative claim carries one of five evidence tags (theorem, practice, measured, speculative, unsourced), and a "Claims withdrawn" section records earlier claims that did not survive measurement. Build: `tectonic docs/architecture.tex` (or `pdflatex` twice); not re-run here, so the PDF is the original build. |

## Check the evidence

| Page | What it answers |
|---|---|
| [details.md](details.md) | What are all the results, including the negative ones? The full results table, the negative results, the serving-package scores whose cause was not recorded, the value head, and the long form of the design decisions, each with its source. |
| [gpu-prototype.md](gpu-prototype.md) | How fast is the GPU prototype of a simplified game loop, how was it checked, and what does it not show? |

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
