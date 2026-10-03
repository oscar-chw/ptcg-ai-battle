"""Tests for the selection MDP.

Every test that asserts a fix works is paired with a NEGATIVE CONTROL that
reintroduces the defect and asserts the test goes red. A gate that has never been
seen to fail is not evidence -- the project has a green-gate-over-random-play
incident on record, and a green-gate-over-a-wrong-featurizer one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ptcg_ppo.action import (
    MAX_STOP_BIAS,
    NO_ACTION,
    STOP_TOKEN,
    Bounds,
    IllegalSelection,
    champion_selection,
    greedy_selection,
    pack_substeps,
    sample_selection,
)


def stop_at(value: float) -> np.ndarray:
    return np.full(MAX_STOP_BIAS + 1, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# The defect observed in play: it cannot stop, and it cannot pass.
# ---------------------------------------------------------------------------


def test_champion_cannot_pass_when_the_engine_offers_it():
    """Reproduces main.py:242 -- `lo = max(1, minCount)` on a pass-legal prompt.

    This is not a test of our code; it pins the behaviour we are replacing, so
    that if someone later "fixes" champion_selection the tie test stops lying.
    """
    bounds = Bounds(min_count=0, max_count=1, n_options=3)
    picked = champion_selection(np.array([0.1, 0.2, 0.3]), bounds)
    assert len(picked) == 1, "the champion is structurally unable to pass"


def test_champion_always_takes_max_count():
    """13.99% of decisions offer a pass; every maxCount>1 decision is over-taken."""
    bounds = Bounds(min_count=0, max_count=3, n_options=5)
    picked = champion_selection(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), bounds)
    assert picked == [0, 1, 2], "champion takes exactly maxCount, always"


def test_stop_is_reachable_when_min_count_is_zero():
    """The whole point: k = 0 must be a sampleable outcome."""
    bounds = Bounds(min_count=0, max_count=3, n_options=4)
    rng = np.random.default_rng(0)
    # Stop logit far above the options -> passing should dominate.
    chosen, logp, substeps = sample_selection(
        np.zeros(4), stop_at(10.0), bounds, rng
    )
    assert chosen == []
    assert substeps == [STOP_TOKEN]
    assert logp < 0.0 and np.isfinite(logp)


def test_stop_is_not_offered_below_min_count():
    """Below the floor the engine would reject the action, so STOP must not exist."""
    bounds = Bounds(min_count=2, max_count=3, n_options=4)
    rng = np.random.default_rng(1)
    for _ in range(50):
        chosen, _, _ = sample_selection(np.zeros(4), stop_at(50.0), bounds, rng)
        assert len(chosen) >= 2, "sampled below minCount despite a huge stop logit"


def test_cardinality_varies_with_the_stop_logit():
    """The count must be a decision, not a constant. This is the cardinality head."""
    bounds = Bounds(min_count=0, max_count=4, n_options=6)
    options = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    counts = {}
    for stop_logit in (-5.0, 0.85, 5.0):
        rng = np.random.default_rng(7)
        draws = [len(sample_selection(options, stop_at(stop_logit), bounds, rng)[0])
                 for _ in range(200)]
        counts[stop_logit] = float(np.mean(draws))
    assert counts[-5.0] > counts[0.85] > counts[5.0], counts
    assert counts[5.0] < 0.5, "a dominant stop logit should mostly pass"


# ---------------------------------------------------------------------------
# GATE G-TIE: at init the extended policy must play the parent's moves exactly.
# ---------------------------------------------------------------------------


def test_greedy_matches_the_champion_at_a_low_stop_init():
    """The comparison against the frozen original must start from a true tie."""
    rng = np.random.default_rng(11)
    for _ in range(300):
        n = int(rng.integers(1, 12))
        lo = int(rng.integers(0, 3))
        hi = int(rng.integers(max(lo, 1), max(lo, 1) + 4))
        if lo > n:
            continue
        bounds = Bounds(min_count=lo, max_count=hi, n_options=n)
        logits = rng.normal(size=n)
        ours = greedy_selection(logits, stop_at(-12.0), bounds)
        theirs = champion_selection(logits, bounds)
        assert sorted(ours) == sorted(theirs), (bounds, logits, ours, theirs)


def test_negative_control_a_high_stop_init_breaks_the_tie():
    """If G-TIE cannot fail, it is not a gate. Raise the stop logit and it must."""
    bounds = Bounds(min_count=0, max_count=3, n_options=4)
    logits = np.array([1.0, 0.9, 0.8, 0.7])
    ours = greedy_selection(logits, stop_at(5.0), bounds)
    theirs = champion_selection(logits, bounds)
    assert ours != theirs, "the tie gate cannot detect a divergent stop logit"


# ---------------------------------------------------------------------------
# Bounds handling: the engine's contract, enforced before the engine sees it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lo,hi,n",
    [(0, 1, 1), (0, 5, 3), (1, 1, 1), (2, 4, 9), (0, 0, 4), (3, 3, 3)],
)
def test_sampled_selections_always_satisfy_the_bounds(lo, hi, n):
    bounds = Bounds(min_count=lo, max_count=hi, n_options=n)
    rng = np.random.default_rng(3)
    for _ in range(120):
        chosen, _, _ = sample_selection(
            rng.normal(size=n), stop_at(rng.normal()), bounds, rng
        )
        assert lo <= len(chosen) <= min(hi, n)
        assert len(set(chosen)) == len(chosen)
        assert all(0 <= c < n for c in chosen)


def test_max_count_zero_means_take_nothing():
    """(0,0) is a real prompt shape and must not select anything."""
    bounds = Bounds(min_count=0, max_count=0, n_options=4)
    rng = np.random.default_rng(5)
    chosen, logp, substeps = sample_selection(np.zeros(4), stop_at(0.0), bounds, rng)
    assert chosen == []
    # Forced, so nothing is recorded and no probability mass is spent: a forced
    # draw would put a guaranteed ratio of 1 into the PPO statistics.
    assert substeps == []
    assert logp == 0.0


def test_impossible_bounds_raise_rather_than_repair():
    with pytest.raises(IllegalSelection):
        Bounds(min_count=3, max_count=2, n_options=5)
    with pytest.raises(IllegalSelection):
        Bounds(min_count=4, max_count=6, n_options=2)
    with pytest.raises(IllegalSelection):
        Bounds(min_count=-1, max_count=2, n_options=5)


def test_options_are_never_selected_twice():
    """Sampling is without replacement; a duplicate is an illegal action."""
    bounds = Bounds(min_count=3, max_count=3, n_options=3)
    rng = np.random.default_rng(9)
    for _ in range(100):
        chosen, _, _ = sample_selection(
            np.array([9.0, 0.0, -9.0]), stop_at(-20.0), bounds, rng
        )
        assert sorted(chosen) == [0, 1, 2]


def test_cardinality_is_free_flags_the_slice_that_matters():
    assert Bounds(0, 3, 5).cardinality_is_free
    assert not Bounds(2, 2, 5).cardinality_is_free
    # Capped by the options actually on offer, not by maxCount alone.
    assert not Bounds(1, 4, 1).cardinality_is_free


# ---------------------------------------------------------------------------
# logp correctness -- the number the whole update trusts.
# ---------------------------------------------------------------------------


def test_logp_is_the_probability_of_the_sequence_actually_drawn():
    """Recompute the joint by hand over the shrinking candidate set."""
    bounds = Bounds(min_count=0, max_count=2, n_options=3)
    options = np.array([1.0, 0.5, -0.5])
    stop = stop_at(-1.0)
    rng = np.random.default_rng(21)
    chosen, logp, substeps = sample_selection(options, stop, bounds, rng)

    expected = 0.0
    remaining = list(range(3))
    for k, pick in enumerate(substeps):
        cand = [options[i] for i in remaining] + [stop[k]]
        cand = np.array(cand)
        shifted = cand - cand.max()
        denom = np.log(np.exp(shifted).sum())
        if pick == STOP_TOKEN:
            expected += float(shifted[-1] - denom)
            break
        slot = remaining.index(pick)
        expected += float(shifted[slot] - denom)
        remaining.remove(pick)
    assert logp == pytest.approx(expected, abs=1e-12)
    assert len(chosen) == sum(1 for s in substeps if s != STOP_TOKEN)


def test_logp_is_a_valid_distribution_over_outcomes():
    """Probabilities over every reachable selection must sum to 1."""
    bounds = Bounds(min_count=0, max_count=2, n_options=3)
    options = np.array([0.3, -0.2, 1.1])
    stop = stop_at(0.4)
    rng = np.random.default_rng(33)
    seen: dict[tuple[int, ...], float] = {}
    for _ in range(40000):
        chosen, logp, _ = sample_selection(options, stop, bounds, rng)
        seen.setdefault(tuple(chosen), float(np.exp(logp)))
    assert sum(seen.values()) == pytest.approx(1.0, abs=1e-6), sorted(seen)


def test_temperature_flattens_the_distribution():
    bounds = Bounds(min_count=1, max_count=1, n_options=4)
    options = np.array([3.0, 0.0, 0.0, 0.0])
    hot = np.mean([sample_selection(options, stop_at(-20.0), bounds,
                                    np.random.default_rng(i), temperature=5.0)[0][0] == 0
                   for i in range(400)])
    cold = np.mean([sample_selection(options, stop_at(-20.0), bounds,
                                     np.random.default_rng(i), temperature=0.2)[0][0] == 0
                    for i in range(400)])
    assert cold > hot, (cold, hot)


def test_zero_temperature_is_rejected():
    bounds = Bounds(min_count=1, max_count=1, n_options=2)
    with pytest.raises(ValueError):
        sample_selection(np.zeros(2), stop_at(0.0), bounds,
                         np.random.default_rng(0), temperature=0.0)


# ---------------------------------------------------------------------------
# Packing -- the storage format the trainer replays from.
# ---------------------------------------------------------------------------


def test_pack_substeps_pads_with_no_action():
    packed = pack_substeps([2, STOP_TOKEN], 4)
    assert packed.tolist() == [2, STOP_TOKEN, NO_ACTION, NO_ACTION]


def test_pack_substeps_refuses_to_truncate():
    """Silent truncation would drop a real draw from the update."""
    with pytest.raises(IllegalSelection):
        pack_substeps([0, 1, 2], 2)


def test_stop_token_is_distinct_from_every_option_index():
    """STOP must not collide with a legal index under any width."""
    assert STOP_TOKEN < 0 and NO_ACTION < 0 and STOP_TOKEN != NO_ACTION
