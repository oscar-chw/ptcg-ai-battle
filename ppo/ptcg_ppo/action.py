"""The selection MDP — and the one thing the champion cannot do.

WHAT THIS REPLACES
------------------
``submissions/FIXED-312/main.py:242-268`` decides how many cards to take with a
constant::

    lo = max(1, sd.get("minCount") or 0)
    hi = max(lo, sd.get("maxCount") or 1)
    picked = [int(i) for i in order[:hi]]        # the top `hi` by logit, always

The network chooses *which*. A constant chooses *how many*. Measured on 6,219
held-out decisions where ``minCount < maxCount``:

    always maxCount (the champion)   54.93% exact-k
    always minCount (v7)             17.43% exact-k
    a bounds lookup table (v8)       56.33% exact-k

and the project's own note on the residual: "the count plainly depends on the
board rather than on the bounds -- a cardinality head is the real answer and this
table is not a substitute for one."

``tools/selfplay_rollout.py:821`` hard-wires the identical constant
(``count = min(maximum, n_options)``), so PPO run on that rollout would train the
defect in rather than out. That is the twin, and this module is the fix for both.

THE MODEL
---------
One decision becomes a short sequence of sub-steps::

    candidates at sub-step k  =  {options not yet chosen}  +  {STOP if k >= minCount}
    sample one; STOP ends the decision; otherwise append and continue
    forced to end when k == maxCount     (no STOP needed, nothing left to add)
    forced away from STOP when k < minCount

``k = 0`` is reachable whenever ``minCount == 0``. That is the pass the engine
offers on 13.99% of decisions and that the champion is structurally incapable of
taking.

ONE FORWARD PER DECISION, STILL
-------------------------------
The per-option logits do not change within a decision -- the board does not move
until the whole selection is submitted. So the model scores every option once,
and the sequential process is pure arithmetic on top. This matters twice: it
keeps rollout cost identical to the champion's, and it means the training-time
replay of a decision is a deterministic function of one logit vector, which is
what makes the ratio-identity gate (see ``objective.py``) meaningful.

A STATIC STOP LOGIT IS NOT A LIMITATION
---------------------------------------
With stop logit ``s`` and remaining option logits ``{l_i}``::

    P(stop) = exp(s) / (exp(s) + sum_remaining exp(l_i))

As options are consumed the denominator shrinks, so P(stop) *rises* on its own.
The induced policy is "take every option whose logit clears the stop threshold",
which is both a sensible inductive bias and directly interpretable. A learned
per-``k`` bias (``stop_bias``, 8 scalars) is added on top so the model can also
express a flat "after two, stop" -- the two mechanisms are complementary and
``--no-stop-bias`` ablates the second.

WHY THE INIT MATTERS MORE THAN IT LOOKS
---------------------------------------
``stop_init_logit`` is set low enough that greedy decoding never picks STOP
before ``maxCount``. The extended policy therefore plays the parent's moves
*exactly* -- gate G-TIE -- so the head-to-head comparison starts from a true tie
and every later divergence is attributable to training rather than to surgery.
Low, not ``-inf``: a saturated softmax has no gradient, and a STOP that can never
be sampled can never be credited. Under sampling it is explored at a small rate,
earns credit from outcomes, and learns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The largest k we build a learned per-step stop bias for. Measured caps: options
# reach 50, but maxCount over the corpus never exceeds 5 (the (0,5) cell is the
# widest observed). 8 is headroom, not a guess about the game.
MAX_STOP_BIAS = 8

# Sentinel written into the fixed-width action array for "no draw at this
# sub-step". -1 rather than 0 because 0 is a legal option index.
NO_ACTION = -1

# STOP is stored as its own sentinel rather than as the integer ``n_options``.
# ``n_options`` varies per decision, so storing it positionally would make a
# batched replay depend on the padded width -- and a stored action whose meaning
# shifts with the batch it lands in is the exact class of defect the ratio-identity
# gate exists to catch. -2 is stable under any repacking.
STOP_TOKEN = -2


class IllegalSelection(ValueError):
    """The engine's bounds and the stored action cannot both be true.

    Raised rather than repaired. A selection that does not satisfy the bounds the
    engine issued is a seam failure, and repairing it here would hide which side
    of the seam is wrong.
    """


@dataclass(frozen=True)
class Bounds:
    """The engine's ``minCount`` / ``maxCount`` for one decision.

    ``n_options`` is the number of legal options actually offered *after* the
    featuriser's OPTION_CAP truncation, because that is the set the policy scored.
    """

    min_count: int
    max_count: int
    n_options: int

    def __post_init__(self) -> None:
        if self.n_options < 0:
            raise IllegalSelection(f"negative option count {self.n_options}")
        if self.min_count < 0 or self.max_count < 0:
            raise IllegalSelection(
                f"negative bounds {self.min_count}..{self.max_count}"
            )
        if self.min_count > self.max_count:
            raise IllegalSelection(
                f"minCount {self.min_count} exceeds maxCount {self.max_count}"
            )
        if self.min_count > self.n_options:
            # The engine promised more selections than it offered options. This is
            # the engine contradicting itself; there is no correct action.
            raise IllegalSelection(
                f"minCount {self.min_count} exceeds {self.n_options} options"
            )

    @property
    def effective_max(self) -> int:
        """How many we may actually take, given only ``n_options`` exist."""
        return min(self.max_count, self.n_options)

    @property
    def pass_is_legal(self) -> bool:
        """True when the engine permits taking nothing at all."""
        return self.min_count == 0

    @property
    def cardinality_is_free(self) -> bool:
        """True when the count is a real decision rather than forced.

        This is the 3.10% slice the champion answers with a constant.
        """
        return self.min_count < self.effective_max


def stop_index(bounds: Bounds) -> int:
    """Index of the STOP pseudo-option: one past the real options."""
    return bounds.n_options


def candidate_mask(bounds: Bounds, chosen: np.ndarray, n_chosen: int) -> np.ndarray:
    """Legal candidates at the current sub-step, over width ``n_options + 1``.

    ``chosen`` is a boolean array over real options, True where already taken.
    The STOP slot is the last entry.
    """
    if n_chosen > bounds.effective_max:
        raise IllegalSelection(
            f"{n_chosen} already chosen exceeds max {bounds.effective_max}"
        )
    mask = np.empty(bounds.n_options + 1, dtype=bool)
    # Sampling is without replacement: an option already taken is not a candidate.
    mask[: bounds.n_options] = ~chosen[: bounds.n_options]
    if n_chosen >= bounds.effective_max:
        # Nothing left to add. Whether or not minCount is satisfied, the only
        # move is to end -- and it is satisfied, because effective_max >= min_count.
        mask[: bounds.n_options] = False
        mask[bounds.n_options] = True
    else:
        # STOP only once the floor is met. Below the floor the engine would reject
        # the action, so it is not a candidate and must not consume probability.
        mask[bounds.n_options] = n_chosen >= bounds.min_count
    if not mask.any():
        raise IllegalSelection(
            f"no legal candidate at k={n_chosen} for {bounds}"
        )
    return mask


def is_terminal_substep(bounds: Bounds, n_chosen: int) -> bool:
    """True when the only remaining candidate is STOP, so the draw is a formality.

    Used to skip storing sub-steps that carry no decision: a forced move has
    log-probability 0 and contributes nothing to the gradient, and storing it
    would put deterministic rows into the ratio statistics and dilute them.
    """
    return n_chosen >= bounds.effective_max


def sample_selection(
    option_logits: np.ndarray,
    stop_logits: np.ndarray,
    bounds: Bounds,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> tuple[list[int], float, list[int]]:
    """Draw one selection, returning ``(chosen, logp, substep_choices)``.

    ``option_logits`` has length >= ``n_options``; only the first ``n_options``
    are read. ``stop_logits`` has length ``MAX_STOP_BIAS + 1`` and supplies the
    stop logit for each ``k``.

    ``logp`` is the joint log-probability of the whole sequence *including the
    STOP draw*, over exactly the sub-steps recorded in ``substep_choices``.
    Forced sub-steps are excluded from both, because their probability is 1 and
    including them would add a guaranteed 0 to every ratio.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    chosen_flags = np.zeros(bounds.n_options + 1, dtype=bool)
    chosen: list[int] = []
    substeps: list[int] = []
    logp = 0.0
    stop = stop_index(bounds)

    while True:
        n_chosen = len(chosen)
        if is_terminal_substep(bounds, n_chosen):
            # Forced end. No draw, no probability mass, nothing recorded.
            break
        mask = candidate_mask(bounds, chosen_flags, n_chosen)
        logits = _assemble(option_logits, stop_logits, bounds, n_chosen)
        probs, logprobs = _masked_softmax(logits, mask, temperature)
        pick = int(rng.choice(probs.size, p=probs))
        logp += float(logprobs[pick])
        if pick == stop:
            substeps.append(STOP_TOKEN)
            break
        substeps.append(pick)
        chosen.append(pick)
        chosen_flags[pick] = True

    _validate(chosen, bounds)
    return chosen, logp, substeps


def greedy_selection(
    option_logits: np.ndarray,
    stop_logits: np.ndarray,
    bounds: Bounds,
) -> list[int]:
    """Deterministic decode — this is what replaces ``order[:hi]`` at serving.

    Repeatedly take the highest-scoring candidate; end at STOP. With
    ``stop_init_logit`` set low this reproduces the champion's selection exactly,
    which is what gate G-TIE asserts.
    """
    chosen_flags = np.zeros(bounds.n_options + 1, dtype=bool)
    chosen: list[int] = []
    stop = stop_index(bounds)

    while not is_terminal_substep(bounds, len(chosen)):
        mask = candidate_mask(bounds, chosen_flags, len(chosen))
        logits = _assemble(option_logits, stop_logits, bounds, len(chosen))
        # Masking at the logits, not after the softmax: Huang & Ontanon show this
        # zeroes the gradient at the invalid entries and stays a valid policy
        # gradient. Renormalising afterwards does not.
        masked = np.where(mask, logits, -np.inf)
        pick = int(np.argmax(masked))
        if pick == stop:
            break
        chosen.append(pick)
        chosen_flags[pick] = True

    _validate(chosen, bounds)
    return chosen


def champion_selection(option_logits: np.ndarray, bounds: Bounds) -> list[int]:
    """The champion's rule, reproduced exactly, as the control for gate G-TIE.

    Transcribed from ``submissions/FIXED-312/main.py:242-264``. Present so the tie
    check compares against the real thing rather than against a description of it.
    """
    lo = max(1, bounds.min_count)
    hi = max(lo, bounds.max_count or 1)
    order = list(np.argsort(-option_logits[: bounds.n_options], kind="stable"))
    picked = [int(i) for i in order[:hi]]
    return picked[: max(lo, min(hi, len(picked)))]


def _assemble(
    option_logits: np.ndarray,
    stop_logits: np.ndarray,
    bounds: Bounds,
    n_chosen: int,
) -> np.ndarray:
    """Concatenate the option logits with the stop logit for this ``k``."""
    out = np.empty(bounds.n_options + 1, dtype=np.float64)
    out[: bounds.n_options] = option_logits[: bounds.n_options]
    out[bounds.n_options] = stop_logits[min(n_chosen, len(stop_logits) - 1)]
    return out


def _masked_softmax(
    logits: np.ndarray, mask: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically stable masked softmax, returning ``(probs, logprobs)``."""
    scaled = np.where(mask, logits / temperature, -np.inf)
    peak = scaled.max()
    shifted = scaled - peak
    exp = np.where(mask, np.exp(shifted), 0.0)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise IllegalSelection("masked softmax has no probability mass")
    probs = exp / total
    logprobs = np.where(mask, shifted - np.log(total), -np.inf)
    # rng.choice demands an exactly-normalised vector; float error can leave it a
    # few ulps off. Renormalise the probabilities only -- logprobs stay exact.
    probs = probs / probs.sum()
    return probs, logprobs


def _validate(chosen: list[int], bounds: Bounds) -> None:
    """The bounds check the engine would apply, applied before the engine sees it."""
    if len(set(chosen)) != len(chosen):
        raise IllegalSelection(f"duplicate selection {chosen}")
    if any(i < 0 or i >= bounds.n_options for i in chosen):
        raise IllegalSelection(f"selection {chosen} outside 0..{bounds.n_options}")
    if not bounds.min_count <= len(chosen) <= bounds.effective_max:
        raise IllegalSelection(
            f"selected {len(chosen)}, engine allows "
            f"{bounds.min_count}..{bounds.effective_max}"
        )


def pack_substeps(substeps: list[int], width: int) -> np.ndarray:
    """Fixed-width array of sub-step choices, padded with ``NO_ACTION``.

    ``width`` must cover the longest sequence in the batch: ``effective_max + 1``
    (every option taken, then a STOP draw).
    """
    if len(substeps) > width:
        raise IllegalSelection(f"{len(substeps)} sub-steps exceeds width {width}")
    out = np.full(width, NO_ACTION, dtype=np.int64)
    out[: len(substeps)] = substeps
    return out
