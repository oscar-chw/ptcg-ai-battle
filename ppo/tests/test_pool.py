"""Tests for the opponent distribution."""

from __future__ import annotations

import numpy as np
import pytest

from ptcg_ppo.objective import GateRefusal
from ptcg_ppo.pool import (
    FORGOTTEN_THRESHOLD,
    MIN_GAMES_FOR_FORGOTTEN,
    Opponent,
    OpponentPool,
    f_hard,
    f_var,
)


def test_f_var_peaks_at_an_even_matchup():
    """A rating-proximate ladder is what f_var targets."""
    assert f_var(0.5) > f_var(0.2) > f_var(0.02)
    assert f_var(0.0) == 0.0 and f_var(1.0) == 0.0


def test_f_hard_spends_nothing_on_opponents_already_beaten():
    assert f_hard(1.0) == 0.0
    assert f_hard(0.0) > f_hard(0.5) > f_hard(0.9)


def test_the_two_weightings_disagree_where_it_matters():
    """f_hard sends games to opponents you lose to; f_var to even ones. This is
    the A/B the design flags as the first knob to sweep."""
    assert f_hard(0.1) > f_hard(0.5)
    assert f_var(0.1) < f_var(0.5)


def test_a_mix_that_does_not_sum_to_one_refuses():
    """A silently renormalised mix is a different experiment from the one written."""
    with pytest.raises(GateRefusal, match="sums to"):
        OpponentPool(parent_id="p", mix={"self": 0.5, "parent": 0.2})


def test_the_parent_is_always_a_member():
    pool = OpponentPool(parent_id="FIXED-312")
    assert "FIXED-312" in pool.members
    assert pool.members["FIXED-312"].kind == "parent"


def test_a_new_checkpoint_joins_at_the_best_existing_quality():
    """OpenAI Five: a fresh checkpoint is sampled at once, not after drifting up."""
    pool = OpponentPool(parent_id="p")
    pool.members["p"].quality = -3.0
    pool.add_checkpoint("iter-1", "/tmp/iter1.pt")
    assert pool.members["iter-1"].quality == -3.0


def test_adding_a_duplicate_refuses():
    pool = OpponentPool(parent_id="p")
    pool.add_checkpoint("iter-1", "/tmp/a.pt")
    with pytest.raises(GateRefusal):
        pool.add_checkpoint("iter-1", "/tmp/b.pt")


def test_sampling_follows_the_mix():
    pool = OpponentPool(parent_id="p", mix={"self": 0.5, "parent": 0.5})
    rng = np.random.default_rng(0)
    draws = [pool.sample(rng) for _ in range(4000)]
    share = draws.count("self") / len(draws)
    assert 0.46 < share < 0.54, share


def test_an_empty_category_reverts_to_self_play():
    """AlphaStar's rule for the forgotten slot, and the same logic for frozen."""
    pool = OpponentPool(parent_id="p", mix={"self": 0.0, "frozen": 1.0})
    rng = np.random.default_rng(0)
    assert {pool.sample(rng) for _ in range(50)} == {"self"}


def test_forgotten_needs_evidence_not_a_bad_streak():
    member = Opponent("x", "frozen", wins=0, games=MIN_GAMES_FOR_FORGOTTEN - 1)
    assert not member.is_forgotten, "a short streak is noise, not a trend"
    member.games = MIN_GAMES_FOR_FORGOTTEN
    assert member.is_forgotten


def test_an_unplayed_opponent_is_assumed_even():
    assert Opponent("x", "frozen").win_rate == 0.5


def test_quality_only_moves_when_the_current_agent_wins():
    """OpenAI Five's rule, verbatim: no update on a loss."""
    pool = OpponentPool(parent_id="p")
    pool.add_checkpoint("iter-1", "/tmp/a.pt")
    before = pool.members["iter-1"].quality
    pool.record("iter-1", current_agent_won=False)
    assert pool.members["iter-1"].quality == before
    pool.record("iter-1", current_agent_won=True)
    assert pool.members["iter-1"].quality < before


def test_recording_against_self_is_a_no_op():
    pool = OpponentPool(parent_id="p")
    pool.record("self", current_agent_won=True)   # must not raise


def test_recording_an_unknown_opponent_refuses():
    pool = OpponentPool(parent_id="p")
    with pytest.raises(GateRefusal, match="unknown opponent"):
        pool.record("ghost", current_agent_won=True)


def test_beaten_opponents_lose_sampling_share_under_f_var():
    """The pool must stop spending games on opponents it dominates."""
    pool = OpponentPool(parent_id="p", mix={"self": 0.0, "frozen": 1.0})
    pool.add_checkpoint("even", "/tmp/a.pt")
    pool.add_checkpoint("crushed", "/tmp/b.pt")
    pool.members["even"].games, pool.members["even"].wins = 100, 50
    pool.members["crushed"].games, pool.members["crushed"].wins = 100, 99
    weights = pool.category_weights("frozen")
    assert weights["even"] > weights["crushed"]


def test_a_degenerate_pool_falls_back_to_uniform_rather_than_dividing_by_zero():
    pool = OpponentPool(parent_id="p", mix={"self": 0.0, "frozen": 1.0})
    pool.add_checkpoint("a", "/tmp/a.pt")
    pool.add_checkpoint("b", "/tmp/b.pt")
    for name in ("a", "b"):
        pool.members[name].games, pool.members[name].wins = 40, 40
    weights = pool.category_weights("frozen")
    assert set(weights.values()) == {1.0}


def test_the_heuristic_is_not_in_the_pool():
    """It is the standing behavioural gate. Training on it destroys its value as a
    measurement: an opponent you train against measures memorisation, not
    generalisation."""
    pool = OpponentPool(parent_id="FIXED-312")
    pool.add_checkpoint("iter-1", "/tmp/a.pt")
    assert not any("heuristic" in name for name in pool.members)
    rng = np.random.default_rng(0)
    assert all("heuristic" not in pool.sample(rng) for _ in range(500))


def test_summary_reports_only_opponents_with_evidence():
    pool = OpponentPool(parent_id="p")
    pool.add_checkpoint("iter-1", "/tmp/a.pt")
    pool.record("iter-1", True)
    summary = pool.summary()
    assert "iter-1" in summary["win_rates"]
    assert "p" not in summary["win_rates"], "no games, no win rate"


def test_forgotten_threshold_is_the_documented_one():
    assert FORGOTTEN_THRESHOLD == 0.50
