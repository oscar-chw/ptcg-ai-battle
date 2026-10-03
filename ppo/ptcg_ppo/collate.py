"""One collate. Used by the rollout AND by the trainer. That is the whole point.

WHY THIS FILE EXISTS
--------------------
The 2026-08-08 run died because the rollout scored with ``train_ss.collate`` while
the trainer recomputed with ``train_ss.collate_fast``. Those two functions differ
on the mask flags and on truncation, so the two sides softmaxed over different
denominators -- and the importance ratio, which is supposed to measure how far the
policy moved, was instead measuring the difference between two collate functions.
Result: ``approx_kl 7.486`` at epoch 0, minibatch 0, before a single gradient step.

There is no configuration that fixes that. There is only one function.

So: this module is the sole path from featuriser tokens to a model batch. The
rollout calls it. The trainer calls it. There are no flags to disagree about, no
second implementation to drift, and the ratio-identity gate in ``objective.py``
therefore tests something real rather than testing two spellings of the same idea.

MASKING
-------
PAD tokens are masked out, for state and options alike. This is deliberate and it
is a change from the champion's serving path, which has no attention mask at all
and was measured putting 97.23% of its pooling attention on padding.

The consequence is stated rather than hidden: **an agent scored through this
collate is not bit-identical to the deployed v6 package.** That is fine here,
because the frozen parent in our arena is scored through this same function, so
the head-to-head stays a fair comparison of weights. It would NOT be fine to ship
a checkpoint trained here behind an unmasked serving path -- measured, that
divergence is 21.8% of decisions.

TRUNCATION
----------
Sequences are truncated to the real token count in the batch. Attention is
O(n^2) and the mean real length is ~57 of 256, so this is most of the compute --
but the reason it is here is correctness, not speed: it keeps the tensors honest
about what is real, and it is applied identically on both sides.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .objective import GateRefusal

# The featuriser pads state to 192 and options to 64. PAD is family id 0 on the
# state side; on the option side the ONLY record of padding is option_type < 0,
# which is destroyed by any clip -- so it must be read before anything else.
PAD_FAMILY = 0
OPTION_PAD_TYPE = -1


def real_counts(row: dict[str, Any]) -> tuple[int, int]:
    """(real state tokens, real options) for one row.

    Read from the data rather than from a stored count, because a stored count
    that disagrees with the arrays is exactly the failure this project keeps
    hitting -- and here the arrays are the ground truth.
    """
    family = np.asarray(row["family"])
    n_state = int((family > PAD_FAMILY).sum())
    option_type = np.asarray(row["option_type"])
    n_option = int((option_type > OPTION_PAD_TYPE).sum())
    if n_option < 1:
        raise GateRefusal("row has no legal options; the engine always offers one")
    if n_state < 1:
        raise GateRefusal("row has no real state tokens")
    return n_state, n_option


def collate(rows: list[dict[str, Any]], device: torch.device,
            numeric_width: int) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Featuriser rows -> the model's batch dict.

    Returns ``(batch, n_options)`` where ``n_options`` is the per-row count of real
    options, which the action space needs and which must never be re-derived
    downstream from a padded width.

    ``numeric_width`` is passed explicitly and comes from the CHECKPOINT
    (``num_proj.shape[1]``). It is not read from a module constant: ``model_ss.py``
    hardcodes ``NUM_W = 269`` under a comment claiming it must equal the
    featuriser's width, and the champion is 160-wide.
    """
    if not rows:
        raise GateRefusal("cannot collate an empty batch")

    counts = [real_counts(r) for r in rows]
    n_state = max(c[0] for c in counts)
    n_option = max(c[1] for c in counts)
    batch = len(rows)
    total = n_state + n_option

    fam = np.zeros((batch, n_state), dtype=np.int64)
    own = np.zeros((batch, n_state), dtype=np.int64)
    card = np.zeros((batch, n_state), dtype=np.int64)
    ctype = np.zeros((batch, n_state), dtype=np.int64)
    etype = np.zeros((batch, n_state), dtype=np.int64)
    num = np.zeros((batch, n_state, numeric_width), dtype=np.float32)

    otype = np.zeros((batch, n_option), dtype=np.int64)
    onum = np.zeros((batch, n_option, numeric_width), dtype=np.float32)
    optr = np.full((batch, n_option, 2), -1, dtype=np.int64)
    oatk = np.zeros((batch, n_option), dtype=np.int64)

    # ONE mask over [state ... options], which is the layout the backbone expects.
    mask = np.zeros((batch, total), dtype=bool)
    omask = np.zeros((batch, n_option), dtype=bool)
    n_options_out = np.zeros(batch, dtype=np.int64)

    for i, (row, (s, o)) in enumerate(zip(rows, counts)):
        fam[i, :s] = np.asarray(row["family"][:s], dtype=np.int64)
        own[i, :s] = np.asarray(row["owner"][:s], dtype=np.int64)
        card[i, :s] = np.asarray(row["card"][:s], dtype=np.int64)
        if "card_type" in row:
            ctype[i, :s] = np.asarray(row["card_type"][:s], dtype=np.int64)
        if "energy_type" in row:
            etype[i, :s] = np.asarray(row["energy_type"][:s], dtype=np.int64)
        num[i, :s] = np.asarray(row["numeric"][:s], dtype=np.float32)

        otype[i, :o] = np.asarray(row["option_type"][:o], dtype=np.int64)
        onum[i, :o] = np.asarray(row["option_numeric"][:o], dtype=np.float32)
        if "option_attack" in row:
            oatk[i, :o] = np.asarray(row["option_attack"][:o], dtype=np.int64)

        ptr = np.asarray(row["option_ptr"][:o], dtype=np.int64).reshape(o, 2)
        # A pointer into a truncated-away state slot is not a pointer.
        optr[i, :o] = np.where((0 <= ptr) & (ptr < s), ptr, -1)

        # THE MASK. Real state tokens, then real options. Never the padding.
        mask[i, :s] = True
        mask[i, n_state:n_state + o] = True
        omask[i, :o] = True
        n_options_out[i] = o

    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    tensors = {
        "family": to(fam), "owner": to(own), "card": to(card),
        "card_type": to(ctype), "energy_type": to(etype), "numeric": to(num),
        "option_type": to(otype), "option_numeric": to(onum),
        "option_ptr": to(optr), "option_attack": to(oatk),
        "mask": to(mask), "option_mask": to(omask),
        "relation": None,
    }
    return tensors, n_options_out


def assert_same_batch(left: dict[str, torch.Tensor],
                      right: dict[str, torch.Tensor]) -> None:
    """Refuse unless two batches are bit-identical.

    Used by the rollout/trainer seam check: the observation replayed at training
    time must be the observation that was scored at rollout time. Comparing the
    logits alone would not localise a mismatch; comparing the inputs does.
    """
    keys = sorted(k for k, v in left.items() if v is not None)
    other = sorted(k for k, v in right.items() if v is not None)
    if keys != other:
        raise GateRefusal(f"batch keys differ: {set(keys) ^ set(other)}")
    for key in keys:
        a, b = left[key], right[key]
        if a.shape != b.shape:
            raise GateRefusal(
                f"'{key}' shape {tuple(a.shape)} != {tuple(b.shape)}: the training "
                "batch is not the rollout batch, so the importance ratio would "
                "measure collation rather than policy drift."
            )
        if not bool(torch.equal(a, b)):
            n = int((a != b).sum())
            raise GateRefusal(
                f"'{key}' differs in {n} element(s) between rollout and training. "
                "This is the defect that produced approx_kl 7.486 at epoch 0 "
                "minibatch 0 on 2026-08-08."
            )
