# Diagrams

Each diagram draws a mechanism this repository documents; every node is a file, function or
documented step, and every number is sourced in the README or the file named under it. The
README embeds diagrams 1 to 3.

*If a diagram and the code disagree, the code wins.*

1. [System overview](#1-system-overview)
2. [The false-green parity check, side by side](#2-the-false-green-parity-check-side-by-side)
3. [Training, gates and serving](#3-training-gates-and-serving)
4. [GPU prototype and its parity tests](#4-gpu-prototype-and-its-parity-tests)

## 1. System overview

How the parts connect: the engine feeds the imitation line that was submitted; PPO and search
are branches that were measured and never shipped; the verification suite checks both.

```mermaid
flowchart TB
  subgraph ENGINE["Official engine: not shipped"]
    ENG[("cg engine, read via<br/>PTCG_ENGINE_DIR")]
  end
  subgraph IMIT["imitation/: the submitted line"]
    FEAT["featurize.py<br/>decision → typed tokens"]
    CORP[("one preserved corpus<br/>89,048 decisions, 926 games")]
    MODEL["model_ss.py SixthSenseNet<br/>set transformer, torch"]
    SERVE["main_v7._forward<br/>NumPy, masks + relations"]
    GATES{{"4 gates on the<br/>built package"}}
  end
  SUB["Kaggle submission BEST1_fixed<br/>score 709.1,<br/>rank 2,043 of 6,807"]
  subgraph BRANCH["Branches never submitted"]
    PPO["ppo/: self-play PPO<br/>with a STOP head"]
    SEARCH["ISMCTS and Gumbel<br/>search baselines"]
  end
  subgraph VER["Verification: report/, results/"]
    WIL["analyze_results.py<br/>Wilson 95% intervals"]
    BRIER{{"value head vs<br/>constant predictor"}}
  end
  ENG -->|"observation, legal options"| FEAT
  FEAT -->|"token rows + label"| CORP
  CORP ==>|"behavioural cloning,<br/>split by game 741/185"| MODEL
  MODEL ==>|"export: fp16 weights<br/>+ serve.* stamp"| SERVE
  FEAT -->|"live tokens"| SERVE
  SERVE ==>|"package .tar.gz"| GATES
  GATES ==>|"tarball to Kaggle"| SUB
  MODEL -.->|"45M champion as<br/>frozen parent"| PPO
  PPO -.->|"0.8104 vs parent,<br/>240 games"| WIL
  SEARCH -.->|"111/400 and 204/400<br/>vs same-deck baseline"| WIL
  MODEL -.->|"earlier run's value head:<br/>Brier 0.5771"| BRIER
  BRIER -.->|"worse than constant 0.487022:<br/>head unused in serving"| SERVE

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

## 2. The false-green parity check, side by side

The two parity checks side by side: the only edge that differs is how the torch reference is
built, and that decides whether the wrong serving mode can be seen. A teammate's review of the
serving path raised the missing masks.

```mermaid
flowchart LR
  REV["teammate's review of<br/>the serving path"]
  subgraph OLD["Old check, gate_export_parity.py (not in repo)"]
    W1[("trained weights")]
    T1["torch side via collate:<br/>no PAD masks,<br/>no relation bias"]
    N1["NumPy main_v6:<br/>attends over PADs,<br/>no relation bias"]
    C1{{"argmax equal?"}}
  end
  subgraph NEW["Corrected check, gate_compute_parity.py"]
    W2[("same weights<br/>+ run manifest")]
    T2["torch side as trained:<br/>collate_fast PAD masks<br/>+ _attach_relation"]
    N2["NumPy main_v7._forward,<br/>same flags as training"]
    C2{{"argmax over real options,<br/>decidable rows only"}}
  end
  FG["false green, read 1.0000;<br/>shipped: 256.7, 183.1<br/>vs 800.5 (team's diagnosis)"]
  CAUGHT["wrong mode: FAIL<br/>trained mode: PASS"]
  W1 -->|"weights"| T1
  W1 -->|"weights"| N1
  T1 -->|"built the same wrong way"| C1
  N1 -->|"unmasked scores"| C1
  C1 -->|"both wrong alike: PASS"| FG
  W2 -->|"weights + flags"| T2
  W2 -->|"weights + flags"| N2
  T2 ==>|"reference = trained function"| C2
  N2 -->|"served scores"| C2
  C2 ==>|"mismatch is visible"| CAUGHT
  REV -.->|"raised the missing masks"| N1

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
  class REV ext
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
  subgraph TRAIN["Train: imitation/training, torch"]
    DATA[("frames.jsonl.gz rows")]
    TR["train_ss.py<br/>collate_fast + relations"]
    CK[("checkpoint .pt")]
    MAN[("run manifest.json")]
  end
  subgraph BUILD["Build: imitation/serving"]
    PA{{"package_arms.sh"}}
    PT["package_and_tar.sh"]
    EX["export_ss_numpy.py"]
    PROV["write_provenance.py"]
    TAR[("package .tar.gz")]
  end
  subgraph GATE["Gate: imitation/gates"]
    ALL["gate_newarm_4gates.py"]
    SEAM{{"seam"}}
    STAMP{{"serve stamp"}}
    REACH{{"reachability"}}
    PAR{{"compute parity"}}
  end
  subgraph SANDBOX["Serve: Kaggle sandbox, no torch"]
    AG["main_v7.agent"]
    FWD["main_v7._forward<br/>NumPy"]
  end
  ENG[("official engine")]
  DATA -->|"training rows"| TR
  TR -->|"weights"| CK
  TR -->|"flags it trained with"| MAN
  CK -->|"checkpoint path"| PA
  MAN -->|"required: refuses if absent"| PA
  PA ==>|"--main main_v7.py<br/>--run-manifest"| PT
  PT ==>|"runs export"| EX
  EX ==>|"weights.npz: fp16,<br/>serve.* = manifest flags"| TAR
  PT -->|"runs"| PROV
  PROV -->|"PROVENANCE.json"| TAR
  PT -->|"main.py sha256<br/>re-checked inside tar"| TAR
  TAR ==>|"built package"| ALL
  ALL -->|"re-run now"| SEAM
  ALL -->|"re-run now"| STAMP
  ALL -->|"extract, import featurizer"| REACH
  ALL -->|"read only if newer<br/>than the package"| PAR
  CK -.->|"fp32 torch reference"| PAR
  ALL ==>|"status PASS: 4 of 4"| AG
  AG ==>|"resolve_mode from<br/>serve.* stamp"| FWD
  FWD ==>|"argmax over legal options"| ENG
  AG -.->|"on exception: lowest legal,<br/>FALLBACKS += 1, logged"| ENG

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
  subgraph PRIV["Private prototype: code not published"]
    LAY["one generated<br/>layout prelude"]
    CPU["Rust CPU reference,<br/>the port's ground truth"]
    GPU["WGSL shader via wgpu,<br/>one lane per game"]
    CUDA["CUDA kernel source,<br/>throughput unmeasured"]
    DEV[("device memory: obs,<br/>reward, done, legal mask")]
  end
  subgraph TESTS["Parity tests: 3 of 3 pass, 2026-10-03"]
    P1{{"GPU vs CPU:<br/>4,096 games × 48 steps"}}
    P2{{"CUDA as host C++<br/>vs CPU"}}
    P3{{"simulation progresses"}}
  end
  ENG[("official engine")]
  OUT["~110M env-steps/s,<br/>simplified slice only,<br/>never used for training"]
  ENG -.->|"ported, slice only;<br/>never parity-tested"| PRIV
  LAY -->|"same layout"| CPU
  LAY -->|"same layout"| GPU
  LAY -->|"same layout"| CUDA
  GPU ==>|"writes every step,<br/>no host round trip"| DEV
  DEV ==>|"batch 262,144"| OUT
  GPU -->|"state, mask, done, reward exact;<br/>obs within 1e-6"| P1
  CPU -->|"reference output"| P1
  CUDA -->|"compiled as host C++"| P2
  CPU -->|"reference output"| P2

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
