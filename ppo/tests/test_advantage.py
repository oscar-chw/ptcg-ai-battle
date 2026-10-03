"""Tests for shaping, the leave-one-out baseline, and normalisation.

The load-bearing test here is `test_negative_control_an_unzeroed_terminal_is_caught`.
Potential-based shaping is only policy-invariant when the potential is zero at the
trajectory's terminal state, and our potential is MAXIMAL exactly at the winning
terminal state -- so getting this wrong pays the agent for reaching a board rather
than for winning, and every other metric stays healthy while it happens.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ptcg_ppo.advantage import (
    Episode,
    distinct_advantage_values,
    episode_returns,
    normalise,
    potential,
    rloo_advantages,
    shaped_episode,
)
from ptcg_ppo.objective import GateRefusal


def episode(potentials, reward=1.0, name="e0", **kw):
    return Episode(episode_id=name, seat=0, opponent_id="parent",
                   potentials=tuple(potentials), terminal_reward=reward, **kw)


# ---------------------------------------------------------------------------
# The potential itself
# ---------------------------------------------------------------------------


def test_potential_rises_as_i_close_on_the_win():
    """Both arguments count points STILL NEEDED, so lower is better for the owner."""
    level = potential(my_points_needed=3, their_points_needed=3)
    ahead = potential(my_points_needed=1, their_points_needed=3)
    behind = potential(my_points_needed=3, their_points_needed=1)
    assert behind < level < ahead
    assert level == 0.0


# ---------------------------------------------------------------------------
# GATE G-SHAPE: the telescoping identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [1.0, 0.99, 0.5])
def test_shaping_telescopes_to_minus_phi_of_the_start(gamma):
    """The shaping must contribute exactly -Phi(s_0), which is action-independent."""
    phis = [0.0, 1.0, 1.0, 2.0, 3.0]
    rewards = shaped_episode(episode(phis, reward=1.0), gamma=gamma)
    shaping = rewards.copy()
    shaping[-1] -= 1.0
    total = sum(gamma**t * shaping[t] for t in range(len(shaping)))
    assert total == pytest.approx(-phis[0], abs=1e-12)


def test_shaping_preserves_the_total_episode_reward_when_phi_starts_at_zero():
    """A game starting level (Phi(s_0) = 0) has shaped return == terminal return."""
    rewards = shaped_episode(episode([0.0, 1.0, 2.0], reward=1.0), gamma=1.0)
    assert float(rewards.sum()) == pytest.approx(1.0, abs=1e-12)


def test_negative_control_an_unzeroed_terminal_is_caught():
    """If G-SHAPE cannot fail, it is decoration.

    Rebuild the shaping the WRONG way -- carrying the final potential forward
    instead of zeroing it, which is the natural implementation and the one Grzes
    shows inverts the optimal policy -- and assert the gate rejects it.
    """
    phis = [0.0, 1.0, 5.0]
    gamma = 1.0
    wrong = np.array([gamma * phis[t + 1] - phis[t] for t in range(len(phis) - 1)]
                     + [0.0])          # last transition left unshaped: Phi(s_N)=Phi
    wrong[-1] += 1.0
    shaping = wrong.copy()
    shaping[-1] -= 1.0
    total = sum(gamma**t * shaping[t] for t in range(len(shaping)))
    assert total != pytest.approx(-phis[0], abs=1e-9), (
        "the wrong construction must NOT satisfy the identity, or the gate is blind"
    )
    # And the real path, on the same potentials, does satisfy it.
    right = shaped_episode(episode(phis, reward=1.0), gamma=gamma)
    ok = right.copy()
    ok[-1] -= 1.0
    assert sum(gamma**t * ok[t] for t in range(len(ok))) == pytest.approx(0.0, abs=1e-12)


def test_a_truncated_episode_is_zeroed_the_same_way():
    """Grzes: the condition applies to step-limit truncations, not only real ends."""
    rewards = shaped_episode(episode([0.0, 2.0, 4.0], reward=0.0, truncated=True))
    assert float(rewards.sum()) == pytest.approx(0.0, abs=1e-12)


def test_winning_from_behind_and_from_ahead_earn_the_same_total():
    """The shaping must not pay for the board, only for the result.

    Two wins, one cruising and one from a losing position. Their shaped returns
    differ only by -Phi(s_0), which no action can influence.
    """
    cruise = shaped_episode(episode([0.0, 1.0, 2.0, 3.0], reward=1.0), gamma=1.0)
    comeback = shaped_episode(episode([0.0, -2.0, -1.0, 2.0], reward=1.0), gamma=1.0)
    assert float(cruise.sum()) == pytest.approx(float(comeback.sum()), abs=1e-12)


def test_shaping_disabled_is_the_terminal_only_control_arm():
    rewards = shaped_episode(episode([0.0, 5.0, 9.0], reward=-1.0), enabled=False)
    assert rewards.tolist() == [0.0, 0.0, -1.0]


def test_gamma_outside_the_unit_interval_refuses():
    with pytest.raises(GateRefusal):
        shaped_episode(episode([0.0, 1.0]), gamma=1.5)
    with pytest.raises(GateRefusal):
        shaped_episode(episode([0.0, 1.0]), gamma=0.0)


def test_an_episode_with_no_decisions_refuses():
    with pytest.raises(GateRefusal):
        episode([])


def test_non_finite_potentials_refuse():
    with pytest.raises(GateRefusal):
        episode([0.0, float("nan")])


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def test_undiscounted_return_to_go_is_the_suffix_sum():
    out = episode_returns(np.array([1.0, 2.0, 3.0]), gamma=1.0)
    assert out.tolist() == [6.0, 5.0, 3.0]


def test_discounting_shrinks_distant_credit():
    out = episode_returns(np.array([0.0, 0.0, 1.0]), gamma=0.5)
    assert out.tolist() == [0.25, 0.5, 1.0]


def test_shaping_moves_credit_off_the_final_decision():
    """The whole point: an unshaped episode gives every decision the same return."""
    ep = episode([0.0, 1.0, 1.0, 3.0], reward=1.0)
    flat = episode_returns(shaped_episode(ep, enabled=False), gamma=1.0)
    shaped = episode_returns(shaped_episode(ep, enabled=True), gamma=1.0)
    assert len(set(flat.tolist())) == 1, "unshaped is constant within an episode"
    assert len(set(shaped.tolist())) > 1, "shaping must differentiate decisions"


# ---------------------------------------------------------------------------
# The leave-one-out baseline
# ---------------------------------------------------------------------------


def test_loo_advantages_are_centred_within_a_group():
    values = {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0}
    groups = dict.fromkeys(values, "g")
    adv = rloo_advantages(values, groups)
    assert math.fsum(adv.values()) == pytest.approx(0.0, abs=1e-12)


def test_loo_baseline_excludes_the_episode_itself():
    values = {"a": 3.0, "b": 0.0, "c": 0.0}
    adv = rloo_advantages(values, dict.fromkeys(values, "g"))
    assert adv["a"] == pytest.approx(3.0 - 0.0)
    assert adv["b"] == pytest.approx(0.0 - 1.5)


def test_groups_are_scored_independently():
    """A group is a set of exchangeable episodes; mixing them injects bias."""
    values = {"a": 10.0, "b": 12.0, "x": 0.0, "y": 2.0}
    groups = {"a": "hard", "b": "hard", "x": "easy", "y": "easy"}
    adv = rloo_advantages(values, groups)
    assert adv["a"] == pytest.approx(adv["x"]), "identical within-group standing"


def test_a_singleton_group_refuses_rather_than_falling_back():
    with pytest.raises(GateRefusal, match="singleton"):
        rloo_advantages({"a": 1.0, "b": 0.0}, {"a": "g1", "b": "g2"})


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalise_produces_zero_mean_unit_variance():
    out = normalise(np.array([1.0, 2.0, 3.0, 4.0]))
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(out.std()) == pytest.approx(1.0, abs=1e-12)


def test_a_homogeneous_minibatch_refuses_instead_of_dividing_by_epsilon():
    """The failure: `/(std + 1e-8)` on an all-win minibatch yields adv ~ 1e8."""
    with pytest.raises(GateRefusal, match="DECISION granularity"):
        normalise(np.ones(64))


def test_distinct_advantage_values_detects_the_degenerate_two_valued_case():
    """Two distinct values means shaping contributed nothing and credit is binary."""
    assert distinct_advantage_values(np.array([1.0, 1.0, -1.0, -1.0])) == 2
    assert distinct_advantage_values(np.array([0.1, 0.2, 0.3])) == 3
    assert distinct_advantage_values(np.array([])) == 0
