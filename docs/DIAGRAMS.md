# Diagrams

Each diagram draws a mechanism this repository documents; every node is a file, function or
documented step, and every number is sourced in the README or the file named under it. The
README embeds diagrams 1 and 2.

*If a diagram and the code disagree, the code wins.*

1. [System overview](#1-system-overview)
2. [The false-green parity check, old and corrected](#2-the-false-green-parity-check-old-and-corrected)
3. [Training, gates and serving](#3-training-gates-and-serving)
4. [GPU prototype and its parity tests](#4-gpu-prototype-and-its-parity-tests)

## 1. System overview

How the parts connect: the engine feeds the imitation line that was submitted; PPO and search
are branches that were measured and never shipped; the verification suite checks both.

```mermaid
flowchart TB
  subgraph SIDE["Off the submitted path: measured only"]
    PPO["ppo/: PPO<br/>+ STOP head"]
    SEARCH["ISMCTS,<br/>Gumbel"]
    WIL["analyze_results.py<br/>Wilson 95%"]
    BRIER{{"value head vs<br/>constant"}}
  end
  ENG[("official cg engine,<br/>via PTCG_ENGINE_DIR")]
  subgraph IMIT["imitation/: the submitted line"]
    FEAT["featurize.py<br/>decision → tokens"]
    CORP[("corpus: 89,048<br/>decisions, 926 games")]
    MODEL["model_ss.py<br/>set transformer"]
    SERVE["main_v7._forward<br/>NumPy serving"]
    GATES{{"4 gates on the<br/>built package"}}
  end
  SUB["BEST1_fixed, score 709.1<br/>rank 2,043 of 6,807"]
  ENG -->|"obs + legal options"| FEAT
  FEAT -->|"token rows"| CORP
  CORP ==>|"cloning, split<br/>741/185 games"| MODEL
  MODEL ==>|"fp16 weights<br/>+ serve.* stamp"| SERVE
  FEAT -->|"live tokens"| SERVE
  SERVE ==>|".tar.gz"| GATES
  GATES ==>|"to Kaggle"| SUB
  MODEL -.->|"45M champion<br/>as frozen parent"| PPO
  PPO -.->|"0.8104 vs parent,<br/>240 games"| WIL
  SEARCH -.->|"111/400, 204/400<br/>vs same-deck"| WIL
  MODEL -.->|"earlier head:<br/>Brier 0.5771"| BRIER
  BRIER -.->|"worse than<br/>0.487022: unused"| SERVE

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
  class ENG ext
  class CORP data
  class FEAT,PPO,SEARCH,WIL step
  class MODEL,SERVE key
  class GATES,BRIER gate
  class SUB out
```

Where in the code: `imitation/training/` (featurize, model_ss, train_ss),
`imitation/serving/` (export_ss_numpy, main_v7), `imitation/gates/`, `ppo/ptcg_ppo/`,
`report/analyze_results.py`, `results/`. A component map, not one deployed agent:
`BEST1_fixed` was trained on its own 400-game corpus and its serving file is not recorded;
the corpus figures are from [REPORT.md](../report/REPORT.md) section 3.

## 2. The false-green parity check, old and corrected

The two parity checks, old above corrected: the only edge that differs is how the torch reference is
built, and that decides whether the wrong serving mode can be seen. A teammate's review of the
serving path raised the missing masks.

```mermaid
flowchart TB
  subgraph OLD["Old: gate_export_parity.py, not in repo"]
    direction TB
    W1[("trained weights")]
    T1["torch via collate:<br/>no PAD masks,<br/>no relations"]
    N1["NumPy main_v6:<br/>attends over PADs,<br/>no relations"]
    C1{{"argmax equal?"}}
    FG["false green: 1.0000;<br/>shipped 256.7, 183.1<br/>vs 800.5 (team's diagnosis)"]
  end
  subgraph NEW["Corrected: gate_compute_parity.py"]
    direction TB
    W2[("same weights<br/>+ run manifest")]
    T2["torch as trained:<br/>collate_fast masks<br/>+ _attach_relation"]
    N2["NumPy main_v7,<br/>same flags"]
    C2{{"argmax over real<br/>options, decidable rows"}}
    CAUGHT["wrong mode: FAIL<br/>trained mode: PASS"]
  end
  W1 -->|"weights"| T1
  W1 -->|"weights"| N1
  T1 -->|"same wrong way"| C1
  N1 -->|"unmasked scores"| C1
  C1 -->|"both wrong: PASS"| FG
  W2 -->|"weights + flags"| T2
  W2 -->|"weights + flags"| N2
  T2 ==>|"reference =<br/>trained function"| C2
  N2 -->|"served scores"| C2
  C2 ==>|"mismatch visible"| CAUGHT
  FG -.->|"teammate's review<br/>raised missing masks"| W2

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
  class W1,W2 data
  class N1,N2,T1 step
  class T2 key
  class C1,C2 gate
  class FG,CAUGHT out
```

Where in the code: [gate_compute_parity.py](../imitation/gates/gate_compute_parity.py)
(docstring names the old gate), `imitation/training/train_ss.py` (`collate`, `collate_fast`,
`_attach_relation`), [main_v7.py](../imitation/serving/main_v7.py); `main_v6` is not in this
repository. [demo/model_demo.py](../demo/model_demo.py) reproduces both rows on a SYNTHETIC
model.

## 3. Training, gates and serving

From training run to served decision: the run's manifest is required to build, the package is
gated as built, and the sandbox serves the mode the run trained with.

```mermaid
flowchart TB
  DATA[("frames.jsonl.gz")]
  TR["train_ss.py<br/>collate_fast + relations"]
  CK[("checkpoint .pt")]
  MAN[("manifest.json")]
  PA{{"package_arms.sh"}}
  PT["package_and_tar.sh"]
  EX["export_ss_numpy.py"]
  PROV["write_provenance.py"]
  TAR[("package .tar.gz")]
  ALL["gate_newarm_4gates.py"]
  PAR{{"compute parity"}}
  SEAM{{"seam"}}
  STAMP{{"serve stamp"}}
  REACH{{"reachability"}}
  AG["main_v7.agent<br/>(Kaggle, no torch)"]
  FWD["main_v7._forward"]
  ENG[("official engine")]
  DATA -->|"rows"| TR
  TR -->|"weights"| CK
  TR -->|"trained flags"| MAN
  CK -->|"path"| PA
  MAN -->|"refuses if absent"| PA
  PA ==>|"main_v7 +<br/>manifest"| PT
  PT ==>|"runs"| EX
  EX ==>|"fp16 weights,<br/>serve.* = flags"| TAR
  PT -->|"runs"| PROV
  PROV -->|"PROVENANCE.json"| TAR
  PT -->|"main.py sha256<br/>re-checked"| TAR
  TAR ==>|"built package"| ALL
  ALL -->|"read only if<br/>newer than package"| PAR
  ALL -->|"re-run"| SEAM
  ALL -->|"re-run"| STAMP
  ALL -->|"extract,<br/>import"| REACH
  CK -.->|"fp32 reference"| PAR
  ALL ==>|"PASS: 4 of 4"| AG
  AG ==>|"mode from<br/>serve.* stamp"| FWD
  FWD ==>|"argmax, legal<br/>options only"| ENG
  AG -.->|"on exception: lowest<br/>legal, FALLBACKS += 1"| ENG

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
  class DATA,CK,MAN,TAR data
  class TR,PT,EX,PROV,ALL step
  class PA,SEAM,STAMP,REACH,PAR gate
  class AG,FWD key
  class ENG ext
```

Where in the code: `imitation/training/train_ss.py`, `imitation/serving/` (package_arms.sh,
package_and_tar.sh, export_ss_numpy.py, write_provenance.py, main_v7.py), `imitation/gates/`
(gate_newarm_4gates.py and the four gates); planted-defect tests in `imitation/tests_torch/`.

## 4. GPU prototype and its parity tests

The prototype's three backends share one layout; the GPU path is checked against the port's own
CPU reference, and nothing connects it to the official engine except the port itself.

```mermaid
flowchart TB
  ENG[("official engine")]
  LAY["one generated layout<br/>prelude (private port)"]
  CPU["Rust CPU<br/>reference"]
  GPU["WGSL shader,<br/>one lane per game"]
  CUDA["CUDA kernel,<br/>unmeasured"]
  DEV[("device memory:<br/>obs, reward,<br/>done, mask")]
  subgraph TESTS["Parity tests: 3 of 3 pass, 2026-10-03"]
    P1{{"GPU vs CPU:<br/>4,096 games<br/>× 48 steps"}}
    P2{{"CUDA as host<br/>C++ vs CPU"}}
    P3{{"simulation<br/>progresses"}}
  end
  OUT["~110M env-steps/s,<br/>slice only, never<br/>used for training"]
  ENG -.->|"ported slice;<br/>no parity test"| LAY
  LAY -->|"layout"| CPU
  LAY -->|"layout"| GPU
  LAY -->|"layout"| CUDA
  GPU ==>|"every step,<br/>no host trip"| DEV
  DEV ==>|"batch 262,144"| OUT
  GPU -->|"state, mask, done,<br/>reward exact;<br/>obs within 1e-6"| P1
  CPU -->|"reference"| P1
  CUDA -->|"as host C++"| P2
  CPU -->|"reference"| P2

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
  class LAY,CPU,CUDA step
  class GPU key
  class DEV data
  class P1,P2,P3 gate
  class OUT out
  class ENG ext
```

Where in the code: not published (`crates/ptcg-gpu` in the private prototype); the records are
[engine/README.md](../engine/README.md) and
[engine/results/](../engine/results/gpubench-2026-10-03.txt).
