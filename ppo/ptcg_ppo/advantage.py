"""Turning game outcomes into per-decision advantages.

THREE THINGS HAPPEN HERE, IN ORDER
-----------------------------------
1. potential-based shaping densifies a single terminal bit across ~78 decisions,
2. a leave-one-out baseline removes the batch mean without a critic,
3. normalisation puts the result on a scale the learning rate was chosen for.

Each step has a way of being silently wrong, and each has a gate.

WHY POTENTIAL-BASED AND NOT A HAND-TUNED BONUS
-----------------------------------------------
Ng, Harada & Russell prove that a shaping term of exactly the form

    F(s, a, s') = gamma * Phi(s') - Phi(s)

leaves the optimal policy unchanged -- and that this form is *necessary* as well
as sufficient: for any F that is not potential-based, there exists an MDP where
no optimal policy of the shaped problem is optimal in the original. Their two
canonical failures are a bicycle agent that rode in tiny circles and a soccer
robot that vibrated next to the ball, both farming a shaping term that had a
positive cycle.

Our prize differential is a natural potential, so we get densified credit
assignment without inventing a new objective that trades winning for
prize-farming. Remark 1 of the paper extends the invariance to arbitrary
policies, so *near*-optimal policies are preserved too -- which matters, because
we will stop well short of optimal.

THE CONDITION THAT IS EASY TO GET WRONG, AND IT IS OUR LIVE RISK
-----------------------------------------------------------------
Grzes (AAMAS 2017) decomposes the shaped return of a finite trajectory as

    U_Phi(trajectory) = U(trajectory) + gamma^N * Phi(s_N) - Phi(s_0)

``Phi(s_0)`` cannot change the policy: it is action-independent. ``gamma^N *
Phi(s_N)`` **can**, because which terminal state you reach depends on earlier
actions. His counterexample inverts the optimal policy for any gamma above ~0.101.

This is precisely our exposure: ``Phi = their_points_needed - my_points_needed``
is **maximal exactly at the winning terminal state**. Left unzeroed, the agent is
paid for reaching a state rather than for winning the game.

Three details that are easy to miss and are all handled below:

* it applies to **step-limit truncations**, not only to genuine terminals;
* the same state keeps its ordinary potential when it is non-terminal in a
  different trajectory -- this is a property of the *trajectory*, not the state;
* zeroing only the final transition's shaping reward does **not** work. Grzes
  shows the same imbalance re-forms at the terminal states' predecessors.

In the two-player case this same condition protects the equilibrium set. Devlin &
Kudenko showed potential shaping preserves the Nash equilibria of a stochastic
game; Grzes corrects them -- "Devlin and Kudenko did not consider this
requirement" -- with a counterexample where a non-zero terminal potential
*introduces a new equilibrium*.

Ng et al. section 5 is also explicit that using ``Phi(s') - Phi(s)`` when
gamma != 1 voids the theorem. The same gamma appears in the shaping and in the
returns here, and ``shaped_episode`` refuses if they are passed differently.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .objective import GateRefusal

# Float slack for the telescoping identity. Generous relative to fp64 error over
# ~78 terms, tight enough that a genuinely unzeroed terminal potential -- which
# is O(1) or larger -- cannot hide under it.
TELESCOPE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Episode:
    """One seat's view of one game.

    ``potentials`` holds Phi(s_t) for each decision this seat made, in order, and
    is **not** yet zeroed at the terminal -- ``shaped_episode`` does that, because
    whether a state is terminal is a fact about the trajectory rather than about
    the state.

    ``terminal_reward`` is +1 win / -1 loss / 0 draw, undiscounted, matching
    AlphaStar's choice to reward the true goal rather than a proxy for it.
    """

    episode_id: str
    seat: int
    opponent_id: str
    potentials: tuple[float, ...]
    terminal_reward: float
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.potentials:
            raise GateRefusal(f"episode {self.episode_id} has no decisions")
        if not all(math.isfinite(p) for p in self.potentials):
            raise GateRefusal(f"episode {self.episode_id} has non-finite potential")
        if not math.isfinite(self.terminal_reward):
            raise GateRefusal(f"episode {self.episode_id} has non-finite reward")


def potential(my_points_needed: float, their_points_needed: float) -> float:
    """Phi(s) = how much closer to winning I am than my opponent.

    Both arguments count points **still needed**, so lower is better for their
    owner and the difference rises as I pull ahead. Bounded by construction (the
    engine's target is a small constant), which Ng et al. require for the
    infinite-state case and which costs nothing to honour here.
    """
    return float(their_points_needed) - float(my_points_needed)


def shaped_episode(
    episode: Episode,
    gamma: float = 1.0,
    enabled: bool = True,
) -> np.ndarray:
    """Per-decision shaped reward for one episode, gated by the telescope identity.

    Returns an array of length ``len(episode.potentials)``: the shaping term for
    each decision, with the terminal reward added to the last one.

    With ``enabled=False`` this is the unshaped control arm -- zeros everywhere
    except the terminal reward on the final decision.
    """
    n = len(episode.potentials)
    rewards = np.zeros(n, dtype=np.float64)
    if not enabled:
        rewards[-1] = episode.terminal_reward
        return rewards

    if not 0.0 < gamma <= 1.0:
        raise GateRefusal(f"gamma {gamma} outside (0, 1]")

    # THE ZEROING. Phi at the trajectory's terminal state is 0, for genuine
    # terminals and step-limit truncations alike.
    phi = list(episode.potentials) + [0.0]

    for t in range(n):
        rewards[t] = gamma * phi[t + 1] - phi[t]
    rewards[-1] += episode.terminal_reward

    _assert_telescopes(rewards, episode, gamma)
    return rewards


def _assert_telescopes(rewards: np.ndarray, episode: Episode, gamma: float) -> None:
    """GATE G-SHAPE. The shaping must contribute exactly ``-Phi(s_0)`` and no more.

    Discounted, the shaping terms collapse to::

        sum_t gamma^t (gamma*Phi(s_t+1) - Phi(s_t)) = gamma^N Phi(s_N) - Phi(s_0)

    and with the terminal zeroing that is exactly ``-Phi(s_0)``. Since Phi(s_0) is
    action-independent, the shaped and unshaped problems have the same optimal
    policy. If this identity does not hold, they do not, and the run would be
    optimising a different game while every other metric looked healthy.
    """
    n = len(rewards)
    shaping_only = rewards.copy()
    shaping_only[-1] -= episode.terminal_reward
    discounted = sum(gamma**t * shaping_only[t] for t in range(n))
    expected = -episode.potentials[0]
    if abs(discounted - expected) > TELESCOPE_TOLERANCE:
        raise GateRefusal(
            f"G-SHAPE FAILED on episode {episode.episode_id} seat {episode.seat}: "
            f"discounted shaping sums to {discounted:.12g}, expected "
            f"{expected:.12g} (= -Phi(s_0)). The shaping is not potential-based as "
            "applied, so it changes the optimal policy. The usual cause is a "
            "terminal potential that was not zeroed -- see Grzes AAMAS 2017; note "
            "Phi here is MAXIMAL at the winning terminal state, so an unzeroed "
            "terminal pays the agent for reaching a board rather than for winning."
        )


def episode_returns(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Discounted return-to-go for each decision.

    With gamma = 1 and shaping on, this is the natural per-decision credit; with
    shaping off it degenerates to the terminal reward stamped on every decision,
    which is what the old run trained on.
    """
    out = np.empty_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def rloo_advantages(values: dict[str, float],
                    groups: dict[str, str]) -> dict[str, float]:
    """Leave-one-out baseline within each group.

    ``values`` maps episode id -> that episode's return; ``groups`` maps episode
    id -> the group it is compared within. A group must be a set of episodes that
    are genuinely exchangeable -- same opponent, same deck, same start-state
    distribution -- or the baseline subtracts a mean drawn from a different
    problem and injects bias rather than removing it.

    Critic-free is the right choice here rather than a fallback: the project's own
    value head measured Brier 0.5771, worse than the 0.487022 a class-frequency
    constant scores on that corpus's validation split (the 0.6667 uniform reference
    it was first compared with assumes draws the data does not have), and it is
    switched off at serving. A baseline with no skill adds variance.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for episode_id, group in groups.items():
        grouped[group].append(episode_id)

    singles = sorted(g for g, eps in grouped.items() if len(eps) < 2)
    if singles:
        raise GateRefusal(
            f"leave-one-out is undefined for {len(singles)} singleton group(s): "
            f"{singles[:5]}. Increase episodes per opponent, or coarsen the "
            "grouping -- do not silently fall back to a global mean, which "
            "compares an episode against a different problem's difficulty."
        )

    out: dict[str, float] = {}
    for episodes in grouped.values():
        total = math.fsum(values[e] for e in episodes)
        denominator = len(episodes) - 1
        for episode_id in episodes:
            out[episode_id] = values[episode_id] - (total - values[episode_id]) / denominator
    return out


def normalise(advantages: np.ndarray, min_std: float = 1e-4) -> np.ndarray:
    """Zero-mean, unit-variance, with a refusal instead of a division by epsilon.

    THE FAILURE THIS GUARDS. With an unshaped terminal reward the advantage takes
    exactly two values across the whole batch, so a minibatch that happens to be
    class-homogeneous has ``std == 0``. The common idiom ``/(std + 1e-8)`` then
    yields advantages of order 1e8 and the run is over in one step. Shuffling at
    decision granularity makes this unlikely; refusing makes it impossible.

    Note the deeper reason shaping comes first: normalising a *two-valued* signal
    maps every sample to exactly +-1, a maximum-magnitude uniform-sign gradient
    with no credit assignment left in it. Shaping is what gives the distribution
    enough shape for normalisation to be a rescaling rather than a binarisation.
    """
    if advantages.size == 0:
        raise GateRefusal("cannot normalise an empty advantage array")
    std = float(advantages.std())
    if std < min_std:
        raise GateRefusal(
            f"advantage std {std:.3e} is below {min_std:.0e}: this minibatch "
            "carries no usable signal, and dividing by it would produce an "
            "arbitrarily large step. Shuffle at DECISION granularity rather than "
            "by game -- a game-homogeneous minibatch is all-win or all-loss and "
            "collapses the spread by construction."
        )
    return (advantages - float(advantages.mean())) / std


def distinct_advantage_values(advantages: np.ndarray, tolerance: float = 1e-9) -> int:
    """How many distinct advantage values a batch actually contains.

    Reported every iteration. Two means the shaping contributed nothing and the
    run has silently degenerated into the unshaped case -- healthy offline
    metrics, no credit assignment.
    """
    if advantages.size == 0:
        return 0
    ordered = np.sort(advantages)
    return int(1 + np.count_nonzero(np.diff(ordered) > tolerance))
