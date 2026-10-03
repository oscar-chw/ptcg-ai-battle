# GPU engine port: numbers and design only

A separate local project ported a **slice** of the competition's game to Rust and ran it
batched on a GPU, to see how many environment steps per second a self-play learner could
be fed. **No code from that port is in this repository.** The official engine is licensed
for competition use only (`LicenseRef-PTCG-ABC-Competition-Use-Only`), and a port is a
derivative work of it, so the port stays local. What follows are the measured numbers and
the design. No result elsewhere in this repository depends on it.

## What was measured, and on what

Apple M2 Max (30-core integrated GPU, unified memory), Metal backend, one dispatch per
environment step, so a learner could look at the observation between steps. "Environment
step" means one agent decision for every game in the batch.

| batch (games) | env-steps/s | games/s |
|---:|---:|---:|
| 4,096 | 10.32M | 60.6k |
| 32,768 | 56.68M | 332.7k |
| **262,144** | **110.14M** | **572.2k** |
| 1,048,576 | 66.32M | 345.0k |

Peak: **110.14M env-steps/s and 572.2k complete games/s at batch 262,144.** Small batches
are dispatch-bound; the largest batch lost 40% of the peak because its state no longer fit
one storage binding (two shards) and the working set contended with the OS for unified
memory.

On the same machine the port's own CPU reference ran 3.19M env-steps/s on one core and
22.82M on 12 cores (rayon), so the GPU was 34.5x and 4.8x faster than those. Keeping
actions, observations and legal-action masks on the device, instead of copying them to the
host each step, was 6.2x faster (86.08M against 13.91M env-steps/s at batch 65,536).

Test status: **213 Rust tests passed** when the project was re-run on 2026-10-03.

Source of every number: `crates/ptcg-gpu/GPU_NOTES.md` in the local port (not published);
the test count is from that re-run.

## What the numbers do not mean

- **A partial game.** The port implemented setup and shuffle, drawing, energy attachment,
  attacks with weakness and resistance, knock-outs, prize taking, promotion, retreat,
  benching, the win conditions, the legal-action mask and the observation. It was not the
  full rule set, so 110.14M is not a throughput for the real engine. Its own notes
  estimate the full-rules cost as a projection and mark it unmeasured; that estimate is
  not repeated here.
- **Divergence is the expected limit.** One game per GPU lane means lanes in a warp can
  run different rule paths. The measured slice barely diverges, so these numbers do not
  price it in.
- **No CUDA number exists.** The CUDA kernel compiled as host C++ and matched the CPU
  reference, but no NVIDIA GPU was available, so CUDA throughput is unmeasured.
- **One machine.** A laptop-class integrated GPU; the notes treat the figures as a floor,
  not a forecast.

## Design

- **Three backends over one layout.** A Rust CPU implementation is the semantic ground
  truth; a WGSL/wgpu shader runs on Metal locally; a CUDA kernel is the intended
  production target. The GPU sources are assembled at run time from a generated layout
  prelude, so a layout change cannot reach one backend and miss another.
- **Fixed-size, allocation-free state**, split into a hot part touched every step and a
  cold part kept as a sparse side table: 5,156 bytes per game instead of 51,460.
- **Device-resident loop.** Observation, reward, done flag and legal-action mask are
  written by the kernel into device memory and exposed as raw pointers, so a learner can
  read them without a host round trip.
- **Parity before throughput.** The GPU and CUDA-source paths were checked against the CPU
  reference over 4,096 seeded games of 48 steps: state, legal mask, done flag and reward
  exact, observation within 1e-6. The CUDA benchmark runs that check before it prints
  any throughput number.
- **Workgroup size.** A sweep from 32 to 512 threads was flat within 7%, so the kernel was
  not occupancy-limited; 64 was used.
