# How fast is the GPU prototype, and what does it not show?

A batched Rust/GPU simulator of a **simplified slice** of the game loop (setup, draw, energy,
attacks with weakness and resistance, knock-outs, prizes, retreat, bench, three win conditions,
legal-action mask, observation): **~110M environment steps/s on an Apple M2 Max** (Metal, batch
262,144). Originally 110.14M, 4.8x the same slice on 12 CPU cores (22.82M); re-run on
2026-10-03, 109.47M, 4.4x (24.69M) ([raw output](../engine/results/gpubench-2026-10-03.txt)).

One parity test compares GPU output with the prototype's own Rust CPU reference (4,096 seeded
games over 48 steps, per [engine/README.md](../engine/README.md)); a second runs the CUDA source
as host C++ against the same reference, and a third checks the simulation progresses. All 3
pass; the committed output shows test names only
([output](../engine/results/parity-2026-10-03.txt)).

What it does not show: it is not rule-complete, not parity-tested against the official engine,
and never used for training; the code is a derivative of the competition-use-only engine and
stays private.

Details and how it is reproduced: [engine/README.md](../engine/README.md). Its three-backend
layout and the parity tests are drawn in
[DIAGRAMS.md](DIAGRAMS.md#4-gpu-prototype-and-its-parity-tests).
