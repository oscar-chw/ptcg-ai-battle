"""Tests for the PPO objective and its gates.

THE CENTRAL TEST IN THIS FILE is `test_torch_replay_reproduces_the_numpy_sampler`.
The old run died because the training-time forward did not reproduce the rollout
forward, and the only check in place compared the stored logp against the stored
LOGITS -- validating arithmetic while blind to the seam. This file checks the
seam itself, in the direction that broke.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ptcg_ppo.action import (
    MAX_STOP_BIAS,
    NO_ACTION,
    STOP_TOKEN,
    Bounds,
    pack_substeps,
    sample_selection,
)
from ptcg_ppo.objective import (
    AdaptiveKL,
    GateRefusal,
    approx_kl_k1,
    approx_kl_k3,
    check_stop_init,
    ppo_losses,
    ratio_identity,
    replay_logp,
    should_abort,
)


def build_batch(decisions, width, n_sub):
    """Pack a list of (bounds, option_logits, stop_logits, substeps) for replay."""
    batch = len(decisions)
    option_logits = torch.zeros(batch, width, dtype=torch.float64)
    option_valid = torch.zeros(batch, width, dtype=torch.bool)
    stop_logits = torch.zeros(batch, MAX_STOP_BIAS + 1, dtype=torch.float64)
    min_count = torch.zeros(batch, dtype=torch.long)
    eff_max = torch.zeros(batch, dtype=torch.long)
    substeps = torch.full((batch, n_sub), NO_ACTION, dtype=torch.long)
    for i, (bounds, opts, stops, steps) in enumerate(decisions):
        option_logits[i, : bounds.n_options] = torch.tensor(opts[: bounds.n_options])
        option_valid[i, : bounds.n_options] = True
        stop_logits[i] = torch.tensor(stops)
        min_count[i] = bounds.min_count
        eff_max[i] = bounds.effective_max
        substeps[i] = torch.tensor(pack_substeps(steps, n_sub))
    return dict(option_logits=option_logits, option_valid=option_valid,
                stop_logits=stop_logits, min_count=min_count, eff_max=eff_max,
                substeps=substeps)


# ---------------------------------------------------------------------------
# The seam. This is the test the old run needed and did not have.
# ---------------------------------------------------------------------------


def test_torch_replay_reproduces_the_numpy_sampler():
    """Rollout logp (numpy) and training logp (torch, batched) must agree exactly.

    Padded rows sit beside unpadded ones, decisions of different widths share a
    batch, and STOP lands at different sub-steps -- all the ways a batched replay
    can quietly score a different distribution than the sampler did.
    """
    rng = np.random.default_rng(1234)
    decisions, expected = [], []
    for _ in range(200):
        n = int(rng.integers(1, 9))
        lo = int(rng.integers(0, min(3, n) + 1))
        hi = int(rng.integers(max(lo, 0), lo + 5))
        if lo > n or hi < lo:
            continue
        bounds = Bounds(lo, hi, n)
        opts = rng.normal(size=n) * 2.0
        stops = np.full(MAX_STOP_BIAS + 1, rng.normal())
        chosen, logp, steps = sample_selection(opts, stops, bounds,
                                               np.random.default_rng(int(rng.integers(1 << 30))))
        decisions.append((bounds, opts, stops, steps))
        expected.append(logp)

    width = max(b.n_options for b, *_ in decisions)
    n_sub = max(len(s) for *_, s in decisions)
    out = replay_logp(**build_batch(decisions, width, n_sub))
    got = (out["logp"] * out["valid"]).sum(-1).numpy()
    np.testing.assert_allclose(got, np.array(expected), atol=1e-12, rtol=0)


def test_replay_is_order_sensitive():
    """Negative control: the candidate set depends on WHAT WAS DRAWN BEFORE.

    A vectorised shortcut that ignored ordering would score a different
    distribution and still look plausible. Two permutations of the same selection
    must give different logps when the option logits differ.
    """
    bounds = Bounds(0, 2, 3)
    opts = np.array([3.0, -1.0, 0.0])
    stops = np.full(MAX_STOP_BIAS + 1, -2.0)
    # No trailing STOP: at k == effective_max the end is forced, and a forced
    # sub-step is never recorded. Writing one here is what the replay gate rejects.
    forward = build_batch([(bounds, opts, stops, [0, 1])], 3, 3)
    reverse = build_batch([(bounds, opts, stops, [1, 0])], 3, 3)
    a = float((replay_logp(**forward)["logp"]).sum())
    b = float((replay_logp(**reverse)["logp"]).sum())
    assert a != pytest.approx(b), "replay is ignoring draw order"


def test_replay_rejects_an_action_outside_the_candidate_set():
    """A stored action that the bounds forbid is a seam failure, not a rounding one."""
    bounds = Bounds(2, 3, 4)          # STOP illegal until two are taken
    opts = np.zeros(4)
    stops = np.full(MAX_STOP_BIAS + 1, 0.0)
    batch = build_batch([(bounds, opts, stops, [STOP_TOKEN])], 4, 2)
    with pytest.raises(GateRefusal, match="absent from the replayed candidate set"):
        replay_logp(**batch)


def test_replay_rejects_reselecting_the_same_option():
    bounds = Bounds(0, 3, 4)
    stops = np.full(MAX_STOP_BIAS + 1, -5.0)
    batch = build_batch([(bounds, np.zeros(4), stops, [1, 1, STOP_TOKEN])], 4, 3)
    with pytest.raises(GateRefusal):
        replay_logp(**batch)


def test_padding_contributes_nothing():
    """A padded row must not shift any reduction."""
    bounds = Bounds(0, 1, 2)
    stops = np.full(MAX_STOP_BIAS + 1, 0.0)
    tight = build_batch([(bounds, np.array([1.0, 0.0]), stops, [0])], 2, 1)
    padded = build_batch([(bounds, np.array([1.0, 0.0]), stops, [0])], 6, 5)
    a = replay_logp(**tight)
    b = replay_logp(**padded)
    assert float((a["logp"] * a["valid"]).sum()) == pytest.approx(
        float((b["logp"] * b["valid"]).sum()), abs=1e-12)
    assert int(a["valid"].sum()) == int(b["valid"].sum()) == 1


def test_replay_is_differentiable_to_the_stop_logit():
    """If no gradient reaches STOP, the cardinality head is decorative."""
    bounds = Bounds(0, 2, 3)
    stops = np.full(MAX_STOP_BIAS + 1, -1.0)
    batch = build_batch([(bounds, np.array([0.5, 0.2, 0.1]), stops,
                          [0, STOP_TOKEN])], 3, 3)
    batch["stop_logits"].requires_grad_(True)
    out = replay_logp(**batch)
    (out["logp"] * out["valid"]).sum().backward()
    grad = batch["stop_logits"].grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# GATE G-RATIO
# ---------------------------------------------------------------------------


def test_ratio_identity_passes_when_the_policy_has_not_moved():
    logp = torch.randn(8, 3, dtype=torch.float64)
    valid = torch.ones(8, 3, dtype=torch.bool)
    report = ratio_identity(logp, logp.clone(), valid)
    assert report["max_abs_logp_delta"] == 0.0
    assert report["max_ratio"] == pytest.approx(1.0)


def test_ratio_identity_refuses_on_the_old_runs_actual_numbers():
    """The old run reported ratio_max 125.2 at epoch 0 minibatch 0 and trained on."""
    old = torch.zeros(4, 2, dtype=torch.float64)
    new = old.clone()
    new[0, 0] = float(np.log(125.2))
    valid = torch.ones(4, 2, dtype=torch.bool)
    with pytest.raises(GateRefusal, match="G-RATIO FAILED"):
        ratio_identity(new, old, valid)


def test_ratio_identity_ignores_padded_substeps():
    """Padding must not be able to fail a gate, nor to hide a failure."""
    logp = torch.zeros(2, 3, dtype=torch.float64)
    old = logp.clone()
    valid = torch.zeros(2, 3, dtype=torch.bool)
    valid[:, 0] = True
    old[:, 2] = 99.0                       # garbage behind the mask
    ratio_identity(logp, old, valid)       # must not raise


# ---------------------------------------------------------------------------
# KL estimators and the abort
# ---------------------------------------------------------------------------


def test_k3_is_never_negative_and_k1_is():
    log_ratio = torch.linspace(-2.0, 2.0, 101, dtype=torch.float64)
    assert bool((approx_kl_k3(log_ratio) >= -1e-12).all())
    assert bool((approx_kl_k1(log_ratio) < 0).any()), "k1 must go negative"


def test_both_estimators_vanish_at_no_drift():
    zero = torch.zeros(5, dtype=torch.float64)
    assert float(approx_kl_k3(zero).abs().max()) == pytest.approx(0.0)
    assert float(approx_kl_k1(zero).abs().max()) == pytest.approx(0.0)


def test_k3_is_outlier_dominated_and_k1_is_not():
    """Why both are reported: a disagreement localises drift to the tail.

    One sample at ratio 32.8 among 99 unmoved ones. k3 carries the raw ratio, so
    the single outlier dominates its mean and it stays positive; k1 is linear in
    the log-ratio and reports a much smaller magnitude with the opposite sign.
    Seeing that split is how you tell "one decision moved a long way" from "every
    decision moved a little".
    """
    tail = torch.zeros(100, dtype=torch.float64)
    tail[0] = float(np.log(32.8))
    k3, k1 = float(approx_kl_k3(tail).mean()), float(approx_kl_k1(tail).mean())
    assert k3 > 0.0 > k1, (k3, k1)
    assert k3 > 5 * abs(k1), (k3, k1)


def test_abort_fires_at_one_and_a_half_times_target():
    assert not should_abort(0.02, 0.02)
    assert not should_abort(0.029, 0.02)
    assert should_abort(0.031, 0.02)
    assert should_abort(7.486, 0.02), "the old run's first minibatch must abort"


def test_abort_is_disabled_by_a_non_positive_target():
    """CleanRL and SB3 both ship target_kl=None. Off must be explicit, not implied."""
    assert not should_abort(1e9, 0.0)


# ---------------------------------------------------------------------------
# The adaptive KL controller
# ---------------------------------------------------------------------------


def test_adaptive_kl_raises_beta_when_drift_exceeds_target():
    controller = AdaptiveKL(beta=0.02, target=6.0)
    before = controller.beta
    controller.update(12.0)
    assert controller.beta > before


def test_adaptive_kl_lowers_beta_when_drift_is_small():
    controller = AdaptiveKL(beta=0.02, target=6.0)
    controller.update(0.5)
    assert controller.beta < 0.02


def test_adaptive_kl_step_is_bounded():
    """Ziegler clips the proportional error to +-0.2, so one step moves beta <=2%."""
    controller = AdaptiveKL(beta=0.02, target=6.0)
    controller.update(1e6)
    assert controller.beta == pytest.approx(0.02 * 1.02)


# ---------------------------------------------------------------------------
# The clipped surrogate
# ---------------------------------------------------------------------------


def kwargs(logp, old, ref, adv, valid, **over):
    base = dict(logp=logp, old_logp=old, ref_logp=ref,
                entropy=torch.ones_like(logp), norm_entropy=torch.ones_like(logp),
                advantage=adv, valid=valid)
    base.update(over)
    return base


def test_no_drift_gives_the_plain_policy_gradient():
    logp = torch.zeros(4, 2, dtype=torch.float64)
    adv = torch.tensor([1.0, -1.0, 2.0, 0.5], dtype=torch.float64)
    valid = torch.ones(4, 2, dtype=torch.bool)
    out = ppo_losses(**kwargs(logp, logp.clone(), logp.clone(), adv, valid),
                     entropy_weight=0.0, kl_weight=0.0)
    assert float(out["pg_loss"]) == pytest.approx(-float(adv.mean()))
    assert float(out["clipfrac"]) == 0.0
    assert float(out["approx_kl"]) == pytest.approx(0.0)


def test_clipping_caps_the_gain_on_a_positive_advantage():
    old = torch.zeros(1, 1, dtype=torch.float64)
    far = torch.full((1, 1), 1.0, dtype=torch.float64)     # ratio = e
    valid = torch.ones(1, 1, dtype=torch.bool)
    adv = torch.tensor([1.0], dtype=torch.float64)
    out = ppo_losses(**kwargs(far, old, old.clone(), adv, valid),
                     entropy_weight=0.0, kl_weight=0.0)
    assert float(out["pg_loss"]) == pytest.approx(-1.2), "clip must bind at 1+eps"


def test_the_reference_kl_pulls_toward_the_anchor():
    logp = torch.full((2, 1), 0.5, dtype=torch.float64)
    ref = torch.zeros(2, 1, dtype=torch.float64)
    valid = torch.ones(2, 1, dtype=torch.bool)
    adv = torch.zeros(2, dtype=torch.float64)
    out = ppo_losses(**kwargs(logp, logp.clone(), ref, adv, valid),
                     entropy_weight=0.0, kl_weight=1.0)
    assert float(out["ref_kl"]) > 0.0
    assert float(out["loss"]) == pytest.approx(float(out["ref_kl"]))


def test_reductions_exclude_padding():
    """A mean padded with zeros shrinks with the padding fraction -- a silent
    learning-rate rescale that varies batch to batch."""
    logp = torch.zeros(2, 4, dtype=torch.float64)
    valid = torch.zeros(2, 4, dtype=torch.bool)
    valid[:, 0] = True
    adv = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = ppo_losses(**kwargs(logp, logp.clone(), logp.clone(), adv, valid),
                     entropy_weight=0.0, kl_weight=0.0)
    assert float(out["pg_loss"]) == pytest.approx(-1.0)
    assert int(out["n_substeps"]) == 2


def test_an_empty_minibatch_refuses():
    logp = torch.zeros(2, 2, dtype=torch.float64)
    valid = torch.zeros(2, 2, dtype=torch.bool)
    with pytest.raises(GateRefusal):
        ppo_losses(**kwargs(logp, logp, logp, torch.zeros(2, dtype=torch.float64),
                            valid))


def test_stop_init_below_the_saturation_floor_refuses():
    check_stop_init(-12.0)
    with pytest.raises(GateRefusal, match="saturation floor"):
        check_stop_init(-1e9)
