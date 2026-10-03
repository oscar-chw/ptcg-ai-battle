"""Tests for architecture derivation and the STOP head."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ptcg_ppo.action import MAX_STOP_BIAS, Bounds, champion_selection, greedy_selection
from ptcg_ppo.objective import GateRefusal
from ptcg_ppo.policy import (
    StopHead,
    architecture_from_state_dict,
    assert_featurizer_width,
    calibrate_stop_bias,
    describe,
    new_parameters,
)


def fixed312_shapes(d=512, numeric=160, layers=12, head_dim=64, cards=1270):
    """The champion's actual tensor shapes, as read off weights.npz today."""
    state = {
        "num_proj.weight": torch.zeros(d, numeric),
        "num_proj.bias": torch.zeros(d),
        "card.weight": torch.zeros(cards, d),
    }
    for i in range(layers):
        state[f"blocks.{i}.attn.qkv.weight"] = torch.zeros(3 * d, d)
        state[f"blocks.{i}.attn.out.weight"] = torch.zeros(d, d)
        state[f"blocks.{i}.attn.k_norm.weight"] = torch.zeros(head_dim)
        state[f"blocks.{i}.attn.rel_bias"] = torch.zeros(8, 8)
    return state


# ---------------------------------------------------------------------------
# Architecture derivation
# ---------------------------------------------------------------------------


def test_derives_the_champions_architecture_from_tensors_alone():
    arch = architecture_from_state_dict(fixed312_shapes())
    assert (arch.d_model, arch.n_layers, arch.n_heads) == (512, 12, 8)
    assert arch.numeric_width == 160
    assert arch.card_vocab == 1270
    assert arch.head_dim == 64
    assert arch.n_relations == 8


def test_a_different_arm_derives_differently():
    """The branch README describes d384/12-head; that is a DIFFERENT model.

    This is why the architecture is read rather than recalled: two arms in the
    same tree disagree, and the trainer's own defaults match neither.
    """
    arch = architecture_from_state_dict(
        fixed312_shapes(d=384, numeric=269, head_dim=32)
    )
    assert (arch.d_model, arch.n_heads, arch.numeric_width) == (384, 12, 269)


def test_a_checkpoint_without_num_proj_refuses():
    with pytest.raises(GateRefusal, match="not a SixthSenseNet"):
        architecture_from_state_dict({"card.weight": torch.zeros(10, 4)})


def test_a_checkpoint_without_blocks_refuses():
    with pytest.raises(GateRefusal, match="no transformer blocks"):
        architecture_from_state_dict({
            "num_proj.weight": torch.zeros(8, 4),
            "card.weight": torch.zeros(10, 8),
        })


def test_an_indivisible_head_dim_refuses_rather_than_rounding():
    state = fixed312_shapes(d=512, head_dim=48)
    with pytest.raises(GateRefusal, match="not divisible"):
        architecture_from_state_dict(state)


def test_describe_names_every_derived_field():
    text = describe(architecture_from_state_dict(fixed312_shapes()))
    assert "d512" in text and "12L" in text and "8 heads" in text
    assert "numeric 160" in text


# ---------------------------------------------------------------------------
# GATE G-WIDTH
# ---------------------------------------------------------------------------


def test_matching_widths_pass():
    assert_featurizer_width(architecture_from_state_dict(fixed312_shapes()), 160)


def test_the_269_featurizer_against_the_160_champion_refuses():
    """The live mismatch: FIXED-312 is 160-wide, the workspace default is 269."""
    arch = architecture_from_state_dict(fixed312_shapes())
    with pytest.raises(GateRefusal, match="G-WIDTH FAILED"):
        assert_featurizer_width(arch, 269)


# ---------------------------------------------------------------------------
# The STOP head
# ---------------------------------------------------------------------------


def make_head(d=16, use_bias=True, init=-4.0):
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))
    return StopHead(d, mlp, stop_init_logit=init, use_bias=use_bias), mlp


def test_the_stop_head_adds_only_d_plus_nine_parameters():
    """521 new parameters on a 45.4M model: the parent is untouched."""
    head, mlp = make_head(d=512)
    added = sum(p.numel() for p in new_parameters(head))
    assert added == 512 + MAX_STOP_BIAS + 1 == 521
    # And it borrows the host MLP rather than owning a second one.
    assert head.policy_mlp is mlp


def test_the_stop_logit_depends_on_the_board():
    """A constant stop logit would be v8's lookup table with extra steps."""
    head, _ = make_head()
    states = torch.randn(32, 16)
    logits = head(states)[:, 0].detach()
    assert float(logits.std()) > 1e-6, "stop logit is not reading the board"


def test_the_per_k_bias_lets_the_count_matter():
    head, _ = make_head()
    head.stop_bias.data = torch.arange(MAX_STOP_BIAS + 1, dtype=torch.float32)
    out = head(torch.randn(4, 16))
    assert float((out[:, 3] - out[:, 0]).mean()) == pytest.approx(3.0, abs=1e-5)


def test_disabling_the_bias_makes_every_k_identical():
    head, _ = make_head(use_bias=False)
    out = head(torch.randn(4, 16))
    assert float((out[:, 0] - out[:, 5]).abs().max()) == 0.0


def test_an_init_below_the_saturation_floor_refuses():
    """A STOP that can never be sampled can never be credited."""
    with pytest.raises(GateRefusal, match="saturation floor"):
        make_head(init=-1e6)


def test_the_stop_head_carries_gradient():
    head, _ = make_head()
    head(torch.randn(8, 16)).sum().backward()
    assert head.stop_token.grad is not None
    assert float(head.stop_token.grad.abs().sum()) > 0.0
    assert float(head.stop_bias.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Calibration and GATE G-TIE
# ---------------------------------------------------------------------------


def test_calibration_puts_the_stop_logit_below_every_option():
    head, _ = make_head()
    states = torch.randn(64, 16)
    options = torch.randn(64, 6)
    valid = torch.ones(64, 6, dtype=torch.bool)
    report = calibrate_stop_bias(head, states, options, valid)
    assert report["n_states"] == 64
    stop = head(states)[:, 0]
    assert bool((stop < options.min(dim=-1).values).all())


def test_calibration_gives_greedy_a_held_out_tie():
    """G-TIE on states the calibration never saw -- a fit checked on itself is not
    a check."""
    head, _ = make_head()
    calibrate_stop_bias(head, torch.randn(128, 16), torch.randn(128, 6),
                        torch.ones(128, 6, dtype=torch.bool))
    held_out_states = torch.randn(200, 16)
    stop_logits = head(held_out_states).detach().numpy().astype(np.float64)
    rng = np.random.default_rng(4)
    for i in range(200):
        n = int(rng.integers(1, 7))
        lo = int(rng.integers(0, min(2, n) + 1))
        hi = int(rng.integers(max(lo, 1), max(lo, 1) + 3))
        bounds = Bounds(lo, hi, n)
        opts = rng.normal(size=n)
        ours = greedy_selection(opts, stop_logits[i], bounds)
        theirs = champion_selection(opts, bounds)
        assert sorted(ours) == sorted(theirs), (i, bounds, ours, theirs)


def test_calibration_ignores_masked_out_option_slots():
    """Padding slots hold garbage; calibrating against them would misplace the bias."""
    head, _ = make_head()
    options = torch.randn(32, 8)
    options[:, 4:] = -1e4                     # padding
    valid = torch.zeros(32, 8, dtype=torch.bool)
    valid[:, :4] = True
    report = calibrate_stop_bias(head, torch.randn(32, 16), options, valid)
    assert report["worst_gap"] > -1e3, "calibrated against padding"


def test_calibration_on_an_empty_sample_refuses():
    head, _ = make_head()
    with pytest.raises(GateRefusal, match="no legal options"):
        calibrate_stop_bias(head, torch.randn(4, 16), torch.randn(4, 3),
                            torch.zeros(4, 3, dtype=torch.bool))


def test_calibration_reports_its_margin_rather_than_assuming_it():
    head, _ = make_head()
    report = calibrate_stop_bias(head, torch.randn(16, 16), torch.randn(16, 5),
                                 torch.ones(16, 5, dtype=torch.bool), margin=7.0)
    assert report["margin"] == 7.0
    assert report["stop_bias"] == pytest.approx(report["worst_gap"] - 7.0)
