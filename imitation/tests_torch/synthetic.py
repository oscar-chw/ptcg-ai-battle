"""SYNTHETIC rows and a tiny model for the numeric tests. No engine data, no card text.

Row layout follows the corpus schema that train_ss.collate_fast reads. Every value
is random; card ids and numeric features carry no meaning.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "gates"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import model_ss  # noqa: E402
import train_ss  # noqa: E402

NS, NO = 24, 8                      # padded state / option widths of every row
NUM_W, OPT_NUM_W = 12, 10
LAYERS, HEADS = 2, 4
FAM_ACTIVE, FAM_PROMPT = train_ss.FAM_ACTIVE, train_ss.FAM_PROMPT


def make_model(seed=0):
    """A 2-layer d=32 net whose relation bias is non-zero, so edges matter."""
    torch.manual_seed(seed)
    model = model_ss.SixthSenseNet(
        d=32, layers=LAYERS, heads=HEADS, hidden=48, dropout=0.0,
        num_scale="log1p", num_w=NUM_W, opt_num_w=OPT_NUM_W)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.rel_bias.normal_(0.0, 1.0)
    return model.eval()


def make_rows(n, seed=1, ns=NS, no=NO, n_real_state=14, n_real_opt=5):
    """n synthetic decisions, right-padded: family 0 / option_type -1 mark PAD."""
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        n_state = rng.randint(n_real_state - 3, n_real_state)
        n_opt = rng.randint(2, n_real_opt)
        family = [rng.randint(1, model_ss.N_FAMILY - 1) for _ in range(n_state)]
        family[0], family[1], family[2] = FAM_ACTIVE, FAM_ACTIVE, FAM_PROMPT
        owner = [rng.randint(0, model_ss.N_OWNER - 1) for _ in range(n_state)]
        owner[0], owner[1] = 0, 1
        pad = ns - n_state
        optr = [[rng.randrange(n_state), rng.choice([-1, rng.randrange(n_state)])]
                for _ in range(n_opt)]
        rows.append({
            "family": family + [0] * pad,
            "owner": owner + [0] * pad,
            "card": [rng.randint(1, 60) for _ in range(n_state)] + [0] * pad,
            "card_type": [rng.randint(0, 7) for _ in range(n_state)] + [0] * pad,
            "energy_type": [rng.randint(0, 12) for _ in range(n_state)] + [0] * pad,
            "numeric": [[rng.uniform(-3, 40) for _ in range(NUM_W)]
                        for _ in range(n_state)] + [[0.0] * NUM_W] * pad,
            "option_type": [rng.randint(0, model_ss.N_OPTION_TYPE - 1)
                            for _ in range(n_opt)] + [-1] * (no - n_opt),
            "option_numeric": [[rng.uniform(-3, 40) for _ in range(OPT_NUM_W)]
                               for _ in range(n_opt)] + [[0.0] * OPT_NUM_W] * (no - n_opt),
            "option_ptr": optr + [[-1, -1]] * (no - n_opt),
            "option_attack": [rng.randint(0, 40) for _ in range(n_opt)] + [0] * (no - n_opt),
            "label": rng.randrange(n_opt),
            "wdl": rng.randint(0, 1),
            "episode": k, "step": 0,
        })
    return rows


def set_training_semantics(mask_state, mask_options, relations):
    """Put train_ss in the state a run with these flags trained in."""
    train_ss.TRUNCATE_SEQ = False
    train_ss.MASK_PAD_STATE = mask_state
    train_ss.MASK_PAD_OPTIONS = mask_options
    train_ss.BUILD_RELATIONS = relations
    train_ss.NUM_W_RUN, train_ss.OPT_NUM_W_RUN = NUM_W, OPT_NUM_W


def torch_logits(model, rows):
    """Policy logits over the real options, computed the way training computes them."""
    batch, _label, _wdl = train_ss.collate_fast(rows, torch.device("cpu"))
    train_ss._attach_relation(batch)
    with torch.inference_mode():
        out = model(batch)["policy"].numpy().astype(np.float64)
    return [out[i, :sum(1 for t in r["option_type"] if t >= 0)] for i, r in enumerate(rows)]


def numpy_weights(model):
    return {k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in model.state_dict().items()}
