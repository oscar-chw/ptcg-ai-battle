#!/usr/bin/env python3
"""The serving model making decisions without the engine, on a SYNTHETIC model and boards.

    PTCG_PYTHON=/path/to/python-with-numpy-and-torch bash scripts/demo.sh
    # or directly:  python demo/model_demo.py

No trained weights are published and real boards are engine-derived, so everything here
is SYNTHETIC: a tiny random set transformer (imitation/tests_torch/synthetic.py) and
random boards. What it shows is the mechanism, not playing strength:

1. the hand-written NumPy forward that ships to the sandbox scores the legal options of
   one board, next to torch's scores for the same weights;
2. the same weights served with the wrong serving mode (no padding masks, no relation
   bias) choose differently, which is the defect class behind the repository's lead;
3. the compute-parity check reads PASS for the trained mode and catches the wrong one.

Exit 0 only if the trained mode matches torch and the wrong mode is detected.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imitation" / "tests_torch"))

import numpy as np  # noqa: E402

import synthetic as syn  # noqa: E402  (puts training/, serving/, gates/ on sys.path)
import main_v7  # noqa: E402
from gate_compute_parity import _row_to_tok  # noqa: E402

TRAINED = {"mask_pad_state": True, "mask_pad_options": True, "relations": True}
WRONG = {"mask_pad_state": False, "mask_pad_options": False, "relations": False}
N_BOARDS = 64


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def numpy_logits(weights, row, mode):
    n = sum(1 for t in row["option_type"] if t >= 0)
    tok = _row_to_tok(row, syn.NS, syn.NO)
    out = main_v7._forward(tok, weights, layers=syn.LAYERS, heads=syn.HEADS, mode=mode)
    return np.asarray(out, dtype=np.float64)[:n]


def main():
    model = syn.make_model()
    weights = syn.numpy_weights(model)
    rows = syn.make_rows(N_BOARDS)
    syn.set_training_semantics(mask_state=True, mask_options=True, relations=True)
    ref = syn.torch_logits(model, rows)          # what the weights were trained to compute

    print("SYNTHETIC model and boards: no engine, no real weights, no strength claim.")
    # Show the first board on which the two serving modes choose differently.
    k = next(i for i, r in enumerate(rows)
             if numpy_logits(weights, r, WRONG).argmax() != ref[i].argmax())
    print(f"\nBoard {k}, {len(ref[k])} legal options; action probabilities:")
    p_torch = softmax(ref[k])
    p_np = softmax(numpy_logits(weights, rows[k], TRAINED))
    p_wrong = softmax(numpy_logits(weights, rows[k], WRONG))
    print("  option  torch (trained)  NumPy, trained mode  NumPy, wrong mode")
    for i in range(len(p_torch)):
        print(f"  {i:>6}  {p_torch[i]:>15.4f}  {p_np[i]:>19.4f}  {p_wrong[i]:>17.4f}")
    print(f"  chosen  {int(p_torch.argmax()):>15}  {int(p_np.argmax()):>19}  {int(p_wrong.argmax()):>17}")

    print(f"\nCompute parity over {N_BOARDS} SYNTHETIC boards, same weights:")
    ok = True
    for name, mode in (("trained mode", TRAINED), ("wrong mode  ", WRONG)):
        got = [numpy_logits(weights, r, mode) for r in rows]
        gap = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
        flips = sum(int(a.argmax() != b.argmax()) for a, b in zip(ref, got))
        passed = gap < 1e-4 and flips == 0
        print(f"  {name}  max logit gap {gap:.2e}, chosen action differs on "
              f"{flips}/{N_BOARDS} boards -> {'PASS' if passed else 'FAIL (detected)'}")
        ok &= passed if mode is TRAINED else not passed
    print("\nThe check passes the serving mode the weights were trained with and rejects the"
          "\nother. The team's records attribute two ladder scores of 256.7 and 183.1, against"
          "\na champion's 800.5, to this defect class: a parity check that collated torch the"
          "\nsame wrong way read green (imitation/gates/gate_compute_parity.py, docstring).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
