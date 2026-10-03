"""collate: the single path from featuriser rows to a model batch (SYNTHETIC rows)."""
import numpy as np
import pytest
import torch

from ptcg_ppo.collate import assert_same_batch, collate, real_counts
from ptcg_ppo.objective import GateRefusal

W = 6                       # numeric width, as read from a checkpoint
NS, NO = 12, 6              # padded widths of the featuriser output


def row(n_state, n_option, seed=0, width=W):
    """One decision, right-padded to NS state slots and NO option slots."""
    rng = np.random.default_rng(seed)
    pad_s, pad_o = NS - n_state, NO - n_option
    return {
        "family": list(range(1, n_state + 1)) + [0] * pad_s,
        "owner": [1] * n_state + [0] * pad_s,
        "card": [5] * n_state + [0] * pad_s,
        "numeric": rng.normal(size=(NS, width)).tolist(),
        "option_type": [3] * n_option + [-1] * pad_o,
        "option_numeric": rng.normal(size=(NO, width)).tolist(),
        "option_ptr": [[0, n_state - 1]] * n_option + [[-1, -1]] * pad_o,
    }


def test_real_counts_come_from_the_arrays():
    assert real_counts(row(7, 3)) == (7, 3)


def test_a_row_without_options_or_state_is_refused():
    with pytest.raises(GateRefusal):
        real_counts(row(7, 0))
    with pytest.raises(GateRefusal):
        real_counts(row(0, 3))
    with pytest.raises(GateRefusal):
        collate([], torch.device("cpu"), W)


def test_batch_is_trimmed_to_the_real_counts_not_the_padded_width():
    batch, n_options = collate([row(7, 3), row(5, 4)], torch.device("cpu"), W)
    assert batch["family"].shape == (2, 7)           # max real state, not 12
    assert batch["option_type"].shape == (2, 4)      # max real options, not 6
    assert n_options.tolist() == [3, 4]
    assert batch["mask"].shape == (2, 7 + 4)


def test_mask_marks_real_tokens_and_never_padding():
    batch, _ = collate([row(7, 3), row(5, 4)], torch.device("cpu"), W)
    mask = batch["mask"].numpy()
    # row 1: 5 real state tokens then 4 real options, laid out after the 7 state slots
    assert mask[1].tolist() == [True] * 5 + [False] * 2 + [True] * 4
    assert batch["option_mask"].numpy().tolist() == [[True] * 3 + [False],
                                                     [True] * 4]


def test_a_pointer_outside_the_real_state_is_not_a_pointer():
    r = row(5, 2)
    r["option_ptr"][0] = [0, 9]          # slot 9 is padding for a 5-token state
    batch, _ = collate([r], torch.device("cpu"), W)
    assert batch["option_ptr"][0, 0].tolist() == [0, -1]


def test_rows_wider_than_the_checkpoint_fail_loudly():
    # The trap in the README: a 269-wide featuriser against a 160-wide checkpoint.
    with pytest.raises(ValueError):
        collate([row(5, 2, width=W + 3)], torch.device("cpu"), W)


def test_assert_same_batch_accepts_identical_and_names_a_one_element_difference():
    rows = [row(7, 3), row(5, 4)]
    a, _ = collate(rows, torch.device("cpu"), W)
    b, _ = collate(rows, torch.device("cpu"), W)
    assert_same_batch(a, b)
    b["numeric"][0, 0, 0] += 1e-3
    with pytest.raises(GateRefusal, match="numeric"):
        assert_same_batch(a, b)


def test_assert_same_batch_refuses_a_shape_mismatch():
    a, _ = collate([row(7, 3)], torch.device("cpu"), W)
    b, _ = collate([row(5, 3)], torch.device("cpu"), W)
    with pytest.raises(GateRefusal, match="shape"):
        assert_same_batch(a, b)
