"""PPO objective, KL diagnostics, and the gate the old run did not have.

THE FAILURE THIS MODULE IS SHAPED BY
------------------------------------
``runs/ppo-rloo-researched/metrics.jsonl``, first row -- minibatch 0 of epoch 0,
before a single gradient step::

    approx_kl 7.486    clipfrac 0.578    ratio_max 125.2
    max_old_logp_delta 9.2e-07

At minibatch 0 of epoch 0 the policy has not moved, so ``ratio`` must be exactly
1, ``approx_kl`` exactly 0 and ``clipfrac`` exactly 0. It was 7.486 -- 370x the
"there is a bug" threshold -- before any optimisation happened. **That run was
never a tuning failure. The ratio never measured policy drift at all.**

``max_old_logp_delta 9.2e-07`` is the trap that let it through: that check
compares the stored logp against a recomputation from the *stored logits*. It
validates logits->logp arithmetic and is structurally blind to model->logits.
``model.eval()`` was set and the checkpoint was the same one, so dropout is
excluded; what remains is that the observation re-collated at training time is
not the observation the rollout scored.

``ratio_identity`` below is the missing check, and it refuses rather than warns.

PER SUB-STEP, NOT PER DECISION
-------------------------------
The old objective took one ``logp`` per decision, defined as the joint
log-probability of ``k`` options drawn without replacement. The ratio is then
``exp(sum of k log-ratio terms)`` and per-term drift *compounds geometrically in
k* -- which is what ``ratio_max`` 32.8, then 125, then 373 looks like. The clip
range 0.2 was chosen for single-action ratios and does not mean the same thing
applied to a product of eight.

So the objective here operates on **flattened sub-steps**: each sub-step is one
categorical draw and gets its own ratio, its own clip and its own KL term. Every
sub-step of a decision carries that decision's advantage. This is the same
treatment the RLHF lineage gives per-token ratios in a sequence, and it is what
makes ``clip_eps = 0.2`` mean what every implementation intends by it.

TWO DIFFERENT KLs, AND THE OLD CONFIG CONSTRAINED NEITHER
----------------------------------------------------------
``approx_kl`` measures ``KL(pi || pi_old)`` -- drift within one update, the thing
the abort watches. ``kl_weight`` penalises ``KL(pi || pi_ref)`` -- drift from the
behaviour-cloned anchor across the whole run. Anchoring to the reference places
*no bound whatsoever* on within-update drift. The old run logged the first and
penalised the second, and had nothing constraining either: no abort, and a fixed
beta that a normalised advantage outweighs by ~50x.

Both are reported every minibatch, separately, and both are controlled: the first
by a hard per-minibatch abort, the second by Ziegler's adaptive controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .action import MAX_STOP_BIAS, NO_ACTION, STOP_TOKEN

# Anything below this and the softmax is saturated: no gradient reaches STOP, so
# a STOP that is never sampled can never be credited. Used to validate the init.
SATURATION_FLOOR = -30.0


class GateRefusal(RuntimeError):
    """A gate failed. Training must not proceed.

    Refusal rather than a warning is the whole point: the old run printed
    approx_kl 7.486 to a log and trained for 326 minibatches anyway.
    """


def replay_logp(
    option_logits: torch.Tensor,
    option_valid: torch.Tensor,
    stop_logits: torch.Tensor,
    min_count: torch.Tensor,
    eff_max: torch.Tensor,
    substeps: torch.Tensor,
    temperature: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiably replay stored selections, one sub-step at a time.

    Shapes, with B decisions, W padded options and T padded sub-steps::

        option_logits (B, W)   float   scores for every option slot
        option_valid  (B, W)   bool    True where the slot is a real legal option
        stop_logits   (B, K)   float   K = MAX_STOP_BIAS + 1, the stop logit per k
        min_count     (B,)     long    engine minCount
        eff_max       (B,)     long    min(maxCount, n_options)
        substeps      (B, T)   long    option index | STOP_TOKEN | NO_ACTION pad
        temperature   (B,)     float   sampling temperature in force at rollout

    Returns per-sub-step tensors of shape (B, T) plus their (B,) reductions.
    ``valid`` marks the sub-steps that actually happened; everything else is
    padding and must be excluded by the caller before any mean is taken.

    The replay is exact rather than approximate on purpose. The candidate set at
    sub-step t depends on what was drawn at 0..t-1, so a vectorised shortcut that
    ignored ordering would silently score a different distribution -- which is the
    same class of defect as the seam that broke the old run.
    """
    batch, width = option_logits.shape
    n_sub = substeps.shape[1]
    device = option_logits.device
    _check_replay_shapes(option_logits, option_valid, stop_logits, min_count,
                         eff_max, substeps)
    if temperature is None:
        temperature = torch.ones(batch, device=device, dtype=option_logits.dtype)

    taken = torch.zeros((batch, width), dtype=torch.bool, device=device)
    n_chosen = torch.zeros(batch, dtype=torch.long, device=device)
    rows = torch.arange(batch, device=device)

    logp = torch.zeros((batch, n_sub), dtype=option_logits.dtype, device=device)
    entropy = torch.zeros_like(logp)
    # Full per-sub-step log-probability vectors, kept so the reference KL can be
    # computed EXACTLY over the whole categorical rather than estimated from the
    # one sampled action. See exact_kl below for why that matters.
    dists = torch.full((batch, n_sub, width + 1), 0.0,
                       dtype=option_logits.dtype, device=device)
    dist_mask = torch.zeros((batch, n_sub, width + 1), dtype=torch.bool, device=device)
    n_cand = torch.zeros((batch, n_sub), dtype=torch.long, device=device)
    valid = substeps != NO_ACTION

    for step in range(n_sub):
        stored = substeps[:, step]
        active = stored != NO_ACTION
        if not bool(active.any()):
            break

        # -- candidate mask over [options ... STOP] -------------------------
        # A recorded sub-step always has n_chosen < eff_max: sample_selection
        # breaks out without recording once the cap is reached, because a forced
        # draw has probability 1 and would put a guaranteed ratio of exactly 1
        # into the statistics, diluting them.
        opt_ok = option_valid & ~taken
        stop_ok = (n_chosen >= min_count) & (n_chosen < eff_max)
        mask = torch.cat([opt_ok, stop_ok[:, None]], dim=1)

        # -- logits over the same layout ------------------------------------
        k_index = n_chosen.clamp(max=stop_logits.shape[1] - 1)
        stop_now = stop_logits[rows, k_index]
        logits = torch.cat([option_logits, stop_now[:, None]], dim=1)
        logits = logits / temperature[:, None]

        # Masking AT THE LOGITS. Huang & Ontanon: this zeroes the gradient at the
        # invalid entries and remains a valid policy gradient; renormalising after
        # the softmax does not.
        neg_inf = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg_inf)
        log_probs = torch.log_softmax(masked, dim=-1)

        # -- gather the stored choice ---------------------------------------
        # STOP is stored as a sentinel and mapped to the final slot here, so the
        # stored value never depends on the padded width it was packed into.
        picked = torch.where(stored == STOP_TOKEN,
                             torch.full_like(stored, width), stored)
        picked = picked.clamp(min=0)
        _assert_legal(mask, picked, active)

        step_logp = log_probs[rows, picked]
        probs = log_probs.exp()
        step_entropy = -(probs * log_probs.masked_fill(~mask, 0.0)).sum(-1)

        dists[:, step] = log_probs
        dist_mask[:, step] = mask & active[:, None]
        logp[:, step] = torch.where(active, step_logp, torch.zeros_like(step_logp))
        entropy[:, step] = torch.where(active, step_entropy,
                                       torch.zeros_like(step_entropy))
        n_cand[:, step] = torch.where(active, mask.sum(-1),
                                      torch.zeros_like(n_chosen))

        # -- advance ---------------------------------------------------------
        appended = active & (stored != STOP_TOKEN)
        if bool(appended.any()):
            idx = rows[appended]
            taken[idx, stored[appended]] = True
            n_chosen = n_chosen + appended.long()

    # Normalised entropy: the achievable entropy range scales with log(n_cand),
    # and n_cand varies from 1 to ~51 here. Reported so an entropy trend can be
    # read without confounding it with a shift in how many options are on offer.
    denom = n_cand.clamp(min=2).to(logp.dtype).log()
    norm_entropy = torch.where(n_cand > 1, entropy / denom,
                               torch.zeros_like(entropy))

    return {
        "logp": logp,
        "entropy": entropy,
        "norm_entropy": norm_entropy,
        "valid": valid,
        "n_candidates": n_cand,
        "log_probs": dists,
        "dist_mask": dist_mask,
        "logp_sum": (logp * valid).sum(-1),
        "entropy_sum": (entropy * valid).sum(-1),
    }


def exact_kl(log_p: torch.Tensor, log_q: torch.Tensor,
             mask: torch.Tensor, stop_scale: float = 1.0) -> torch.Tensor:
    """Exact KL(p || q) over the full masked categorical, per sub-step.

    WHY THIS REPLACES THE SINGLE-SAMPLE ESTIMATOR. The reference-KL term was being
    estimated from the ONE action that happened to be sampled:
    ``k3(logp - ref_logp)``. That estimator is unbiased but high-variance, and it
    only ever sees the sampled action -- so the anchor exerts no force on the rest
    of the distribution, which is most of what the supervised policy knows.

    Here the action set is a masked categorical of at most ~51 options, so the
    exact sum is cheap and strictly better:

        KL(p||q) = sum_a p(a) * (log p(a) - log q(a))

    computed over valid options only. AlphaStar's supervised-KL term is over the
    action distribution, not over a sample, and this matches that.

    Note the DIRECTION: KL(current || reference). This is the mode-seeking
    direction, and it is the one the RLHF lineage penalises.
    """
    p = log_p.exp() * mask
    diff = (log_p - log_q) * mask
    if stop_scale != 1.0:
        # PER-ACTION KL WEIGHTING, and the STOP column gets its own.
        #
        # The reference is the BC policy, which has P(STOP) ~ 0 BY CONSTRUCTION --
        # the champion has no STOP action at all, so the anchor's opinion about it
        # is not knowledge, it is an artefact of the action space we added. A
        # uniform beta therefore spends the anchor's whole budget holding down the
        # one action we are trying to free, while the columns where the reference
        # genuinely knows something get the same weight.
        #
        # A uniform KL coefficient is a design choice, not a law: it is measured to
        # make critical-decision probabilities revert to their initial levels.
        w = torch.ones_like(diff)
        w[..., -1] = stop_scale
        diff = diff * w
    return (p * diff).sum(-1)


def approx_kl_k1(log_ratio: torch.Tensor) -> torch.Tensor:
    """``-log r``. Unbiased, and NEGATIVE for about half the samples by design.

    Schulman's own note: it "has high-variance, as it's negative for half of the
    samples, whereas KL is always positive". A negative k1 next to a positive k3
    is normal and is not evidence of a bug. Spinning Up gates on this one.
    """
    return -log_ratio


def approx_kl_k3(log_ratio: torch.Tensor) -> torch.Tensor:
    """``(r - 1) - log r``. Unbiased and always >= 0, because ``log x <= x - 1``.

    Contains the raw ratio, so it is outlier-dominated: one sample at r = 32.8
    contributes 28.3 on its own. CleanRL and SB3 both gate on this one. Reported
    beside k1 so a disagreement localises drift to the tail rather than leaving
    it ambiguous.
    """
    return log_ratio.exp() - 1.0 - log_ratio


def ratio_identity(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    valid: torch.Tensor,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """GATE G-RATIO. The single highest-yield line in this project.

    At epoch 0, minibatch 0, the policy has not moved, so every ratio must be 1.
    Anything else means the training-time forward is not reproducing the forward
    that generated the data, and no amount of hyperparameter tuning will fix it.

    Returns a diagnostic dict. Raises ``GateRefusal`` when it fails, with the
    worst offenders named -- the old run's log said only "approx_kl 7.486", which
    is true and useless.
    """
    delta = (logp - old_logp).detach()
    delta = torch.where(valid, delta, torch.zeros_like(delta))
    worst = float(delta.abs().max()) if delta.numel() else 0.0
    ratio = delta.exp()
    report = {
        "max_abs_logp_delta": worst,
        "max_ratio": float(ratio[valid].max()) if bool(valid.any()) else 1.0,
        "min_ratio": float(ratio[valid].min()) if bool(valid.any()) else 1.0,
        "n_substeps": int(valid.sum()),
    }
    if worst > tolerance:
        offenders = delta.abs().flatten().topk(min(10, delta.numel())).indices
        rows = (offenders // delta.shape[1]).tolist()
        cols = (offenders % delta.shape[1]).tolist()
        raise GateRefusal(
            "G-RATIO FAILED: the training forward does not reproduce the rollout "
            f"forward. max |logp - old_logp| = {worst:.6e} > {tolerance:.0e} at "
            "epoch 0 minibatch 0, where the policy has not moved and every ratio "
            "must be exactly 1.\n"
            f"  worst (row, substep): {list(zip(rows, cols))}\n"
            "  This is a seam failure, not a tuning problem. Check, in order: "
            "(1) the featurizer width against num_proj.shape[1]; (2) the mask "
            "flags -- collate() and collate_fast() disagree on exactly these; "
            "(3) the option truncation convention; (4) model.eval() for dropout. "
            "The old run reported approx_kl 7.486 here and trained anyway."
        )
    return report


@dataclass
class AdaptiveKL:
    """Ziegler's log-space proportional controller for the reference-KL weight.

    A *fixed* beta is a nudge with no feedback: with advantages normalised to
    |A| ~ 1 and beta = 0.02, the reward term outweighs the penalty by roughly 50x
    and the policy simply walks away and keeps walking.

    Ziegler et al., verbatim::

        e_t    = clip((KL - KL_target) / KL_target, -0.2, 0.2)
        beta   = beta * (1 + K_beta * e_t)          K_beta = 0.1

    The default ``beta`` of 0.02 is InstructGPT's, whose Appendix E.7 sweep found
    "both 0 and 2 result in poor performance; the optimal value is around 0.01 and
    0.02". Note Stiennon's Table 9: the beta -> realised-KL map is strongly
    nonlinear (a 7x change in beta moved realised KL 10x), so this controller
    exists to target the *measured* KL rather than to trust the coefficient.
    """

    beta: float = 0.02
    target: float = 6.0
    horizon_gain: float = 0.1
    clip: float = 0.2
    history: list[float] = field(default_factory=list)

    def update(self, measured_kl: float) -> float:
        if self.target <= 0.0:
            return self.beta
        error = (measured_kl - self.target) / self.target
        error = max(-self.clip, min(self.clip, error))
        self.beta = self.beta * (1.0 + self.horizon_gain * error)
        self.history.append(self.beta)
        return self.beta


def ppo_losses(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    entropy: torch.Tensor,
    norm_entropy: torch.Tensor,
    advantage: torch.Tensor,
    valid: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    entropy_weight: float = 1e-3,
    kl_weight: float = 0.02,
    exact_ref_kl: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """The clipped surrogate, over flattened sub-steps.

    ``advantage`` is (B,) per decision and is broadcast across that decision's
    sub-steps: every sub-step of a decision shares the decision's credit, which is
    correct because they are one action as far as the environment is concerned.

    All reductions are over ``valid`` only. Padding a mean with zeros would shrink
    it by the padding fraction, which varies with the batch -- a slow, silent
    rescaling of the learning rate.
    """
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean mask")
    n_valid = valid.sum()
    if int(n_valid) == 0:
        raise GateRefusal("minibatch contains no valid sub-steps")

    adv = advantage[:, None].expand_as(logp)
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()

    # DECOUPLED CLIP ("clip-higher"). A symmetric epsilon puts a hard ceiling on
    # how fast a LOW-probability action can grow: at p=0.004 with eps=0.2 one
    # update reaches at most 0.0048, so ~20 consecutive fully-clipped positive
    # updates are needed to reach 0.15 -- and any sign cancellation resets it.
    # That is the STOP head's starvation, as arithmetic rather than intuition.
    # Raising only the UPPER bound lifts that ceiling while negative advantages
    # stay tightly clipped. Measured to help ALIGNED models specifically (a BC
    # clone is the structural analogue); base models barely clip at all.
    hi = clip_eps if clip_eps_high is None else clip_eps_high
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + hi) * adv
    pg = -torch.minimum(unclipped, clipped)
    pg_loss = _masked_mean(pg, valid)

    # KL to the frozen reference, per sub-step. The old code applied the k3 form
    # to a JOINT log-probability difference, so exp() of a compounded delta made
    # the anchor detonate exactly when it was most needed.
    if exact_ref_kl is not None:
        # Exact KL over the full masked categorical. Strictly better than the
        # single-sample estimator below: that one only ever sees the action that
        # happened to be drawn, so the anchor exerts no force on the rest of the
        # distribution -- which is most of what the supervised policy knows.
        ref_kl = _masked_mean(exact_ref_kl, valid)
    else:
        ref_delta = logp - ref_logp
        ref_kl = _masked_mean(approx_kl_k3(ref_delta), valid)

    entropy_mean = _masked_mean(entropy, valid)
    loss = pg_loss - entropy_weight * entropy_mean + kl_weight * ref_kl

    with torch.no_grad():
        k1 = _masked_mean(approx_kl_k1(log_ratio), valid)
        k3 = _masked_mean(approx_kl_k3(log_ratio), valid)
        clipfrac = _masked_mean(
            ((ratio > 1.0 + hi) | (ratio < 1.0 - clip_eps)).to(ratio.dtype), valid)
        ratio_max = ratio[valid].max()

    return {
        "loss": loss,
        "pg_loss": pg_loss,
        "ref_kl": ref_kl,
        "entropy": entropy_mean,
        "norm_entropy": _masked_mean(norm_entropy, valid),
        "approx_kl": k3,
        "approx_kl_k1": k1,
        "clipfrac": clipfrac,
        "ratio_max": ratio_max,
        "n_substeps": n_valid,
    }


def should_abort(approx_kl: float, target_kl: float, factor: float = 1.5) -> bool:
    """SB3's per-minibatch trust-region abort, with Spinning Up's philosophy.

    Clipping does not bound KL -- Spinning Up says it outright: the new policy
    "can still go farther than the clip_ratio says, but it doesn't help on the
    objective anymore". Clipping removes the *incentive* to diverge, not the
    *motion*, because the other samples in the minibatch keep moving the shared
    weights. Early stopping is the only actual constraint.

    Granularity matters: CleanRL breaks only at the end of an epoch, SB3 per
    minibatch. The old run reached minibatch 325 inside epoch 0, so an epoch-level
    check would not have fired once. Both ship this OFF by default, which is how
    it gets missed.
    """
    if target_kl <= 0.0:
        return False
    return approx_kl > factor * target_kl


def check_stop_init(stop_logit: float) -> None:
    """Refuse a STOP init that cannot learn.

    Below the saturation floor the softmax gradient to STOP underflows, so STOP is
    never sampled, never credited, and the cardinality head is decorative. This is
    the failure that would look exactly like "PPO changed nothing".
    """
    if stop_logit < SATURATION_FLOOR:
        raise GateRefusal(
            f"stop_init_logit {stop_logit} is below the saturation floor "
            f"{SATURATION_FLOOR}: STOP would never be sampled and could never be "
            "credited, so the cardinality head could not learn. Use a value low "
            "enough for greedy decoding to ignore it (gate G-TIE) but high enough "
            "to carry gradient."
        )


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (values * valid).sum() / valid.sum().clamp(min=1)


def _assert_legal(mask: torch.Tensor, picked: torch.Tensor,
                  active: torch.Tensor) -> None:
    rows = torch.arange(mask.shape[0], device=mask.device)
    legal = mask[rows, picked]
    bad = active & ~legal
    if bool(bad.any()):
        raise GateRefusal(
            f"stored action is absent from the replayed candidate set for "
            f"{int(bad.sum())} sub-steps. The candidate set is derived from the "
            "engine bounds and the draws before it, so this means the stored "
            "action and the stored bounds disagree -- a rollout/trainer seam "
            "failure, not a numerical one."
        )


def _check_replay_shapes(option_logits, option_valid, stop_logits, min_count,
                         eff_max, substeps) -> None:
    batch, width = option_logits.shape
    if option_valid.shape != (batch, width):
        raise ValueError(f"option_valid {tuple(option_valid.shape)} != "
                         f"{(batch, width)}")
    if option_valid.dtype != torch.bool:
        raise TypeError("option_valid must be boolean")
    if stop_logits.shape[0] != batch or stop_logits.shape[1] != MAX_STOP_BIAS + 1:
        raise ValueError(f"stop_logits {tuple(stop_logits.shape)} != "
                         f"{(batch, MAX_STOP_BIAS + 1)}")
    for name, tensor in (("min_count", min_count), ("eff_max", eff_max)):
        if tensor.shape != (batch,):
            raise ValueError(f"{name} {tuple(tensor.shape)} != {(batch,)}")
    if substeps.shape[0] != batch:
        raise ValueError(f"substeps batch {substeps.shape[0]} != {batch}")
    if bool((eff_max < min_count).any()):
        raise GateRefusal("eff_max < min_count: the engine bounds are impossible")
