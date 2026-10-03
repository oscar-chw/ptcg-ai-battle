"""The opponent distribution.

Deployment is not self-play. It is a ladder of frozen submissions, matched at a
similar rating. Two published results shape the mix, and they pull in opposite
directions:

* **Pure best-response is brittle.** Johanson, Zinkevich & Bowling computed
  best responses to seven poker opponents and cross-played them: each beat its
  own target hard, and then *lost to opponents it was not fitted to*, including
  a weak one an approximate equilibrium beats comfortably. Verbatim: "best
  response is, in practice, a brittle computation, and can perform poorly when
  the model is wrong." The Restricted Nash Response is the correct object -- put
  mass ``p`` on the modelled field and ``1-p`` on free play -- and its
  exploitation/exploitability curve is steep near ``p = 1``, so conceding a
  little exploitation buys a large reduction in worst case.

* **Naive self-play is not weak, it is forgetful.** AlphaStar's ablation put
  naive self-play at 1519 Elo against pFSP+SP's 1540 -- 21 Elo -- while pure
  historical play scored 246-376 Elo *worse*. What separates them is retention:
  minimum win rate against all past selves, 46% vs 71%. So a pool buys memory,
  not peak strength, and the self-play fraction stays high.

Hence the default mix::

    35%  current self          generates transitive strength; do not go lower
    30%  own frozen history    PFSP-weighted over the full history
    25%  the frozen parent     the thing we must beat, and the field's stand-in
    10%  "forgotten"           opponents we have stopped beating; -> self if empty

f_var, NOT f_hard
-----------------
AlphaStar's default weighting is ``f_hard(x) = (1-x)^p``, which drives games
toward opponents you cannot yet beat -- "a smooth approximation of max-min
optimisation", chosen because they wanted to be unexploitable by *anyone*. We are
scored on expected result against a **rating-proximate** sample, which is what
their alternative ``f_var(x) = x(1-x)`` targets: opponents around your own level.
Train the matchmaking you are scored under.

That last step is reasoning, not a citation, and it is the first knob to sweep.

THE HEURISTIC IS DELIBERATELY ABSENT
-------------------------------------
``arena_checkpoints.py --b heuristic`` is the standing behavioural gate. An
opponent you train against stops measuring generalisation and starts measuring
memorisation. AlphaStar used fixed opponents as *Elo anchors* and for ablations
specifically to avoid multi-agent dynamics -- which only works while they are
held out. If scripted opponents are wanted in training, write a different set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .objective import GateRefusal

# Sampled below this and the "forgotten" designation is noise rather than a
# trend; a 20-game estimate has a standard error of 11 points.
MIN_GAMES_FOR_FORGOTTEN = 20

# Below this win rate against a pool member, we have stopped beating it.
FORGOTTEN_THRESHOLD = 0.50


@dataclass
class Opponent:
    """One frozen policy in the pool."""

    opponent_id: str
    kind: str                      # "self" | "frozen" | "parent"
    path: str | None = None
    wins: int = 0                  # games the CURRENT agent won against it
    games: int = 0
    quality: float = 0.0           # OpenAI Five's q_i

    @property
    def win_rate(self) -> float:
        """P[current agent beats this opponent]. 0.5 until there is evidence."""
        if self.games == 0:
            return 0.5
        return self.wins / self.games

    @property
    def is_forgotten(self) -> bool:
        return (self.games >= MIN_GAMES_FOR_FORGOTTEN
                and self.win_rate < FORGOTTEN_THRESHOLD)


def f_hard(x: float, power: float = 1.0) -> float:
    """AlphaStar's default: weight toward opponents you cannot beat.

    ``f_hard(1) = 0``, so no games are spent on opponents already beaten.
    Provided for the A/B against ``f_var``; not the default here.
    """
    return max(0.0, 1.0 - x) ** power


def f_var(x: float) -> float:
    """AlphaStar's alternative: weight toward opponents near your own level.

    Peaks at x = 0.5 and vanishes at both ends -- no games against opponents you
    always beat, and none against opponents you never beat. This is the shape
    that matches a rating-proximate ladder.
    """
    return max(0.0, x * (1.0 - x))


@dataclass
class OpponentPool:
    """Sampling and bookkeeping over the frozen opponents.

    ``mix`` is the per-episode probability of each category. It must sum to 1;
    the check is a refusal because a silently renormalised mix is a different
    experiment from the one written down.
    """

    parent_id: str
    mix: dict[str, float] = field(default_factory=lambda: {
        "self": 0.35, "frozen": 0.30, "parent": 0.25, "forgotten": 0.10,
    })
    weighting: str = "f_var"
    quality_eta: float = 0.01
    members: dict[str, Opponent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = math.fsum(self.mix.values())
        if abs(total - 1.0) > 1e-9:
            raise GateRefusal(f"opponent mix sums to {total}, not 1.0: {self.mix}")
        if any(v < 0 for v in self.mix.values()):
            raise GateRefusal(f"opponent mix has a negative share: {self.mix}")
        if self.weighting not in ("f_var", "f_hard", "uniform"):
            raise GateRefusal(f"unknown weighting {self.weighting!r}")
        self.members.setdefault(
            self.parent_id, Opponent(self.parent_id, "parent")
        )

    def add_checkpoint(self, opponent_id: str, path: str) -> None:
        """Freeze the current agent into the pool.

        New members join at the maximum existing quality, per OpenAI Five, so a
        fresh checkpoint is sampled immediately rather than waiting for its score
        to drift up from zero.
        """
        if opponent_id in self.members:
            raise GateRefusal(f"{opponent_id} is already in the pool")
        best = max((m.quality for m in self.members.values()), default=0.0)
        self.members[opponent_id] = Opponent(opponent_id, "frozen", path,
                                             quality=best)

    def category_weights(self, category: str) -> dict[str, float]:
        """Unnormalised sampling weight for each member of one category."""
        if category == "forgotten":
            candidates = [m for m in self.members.values() if m.is_forgotten]
        else:
            candidates = [m for m in self.members.values() if m.kind == category]
        if not candidates:
            return {}
        if self.weighting == "uniform":
            return {m.opponent_id: 1.0 for m in candidates}
        shape = f_var if self.weighting == "f_var" else f_hard
        weights = {m.opponent_id: shape(m.win_rate) for m in candidates}
        if math.fsum(weights.values()) <= 0.0:
            # Every candidate is at 0 or 1 win rate, so the shaped weight
            # vanishes everywhere. Fall back to uniform rather than dividing by
            # zero -- and say so, because it means the pool has stopped
            # discriminating and probably needs fresh members.
            return {m.opponent_id: 1.0 for m in candidates}
        return weights

    def sample(self, rng: np.random.Generator) -> str:
        """Draw one opponent id for one episode.

        ``"self"`` is returned literally: the caller mirrors the current weights.
        An empty category reverts its mass to self-play, matching AlphaStar's rule
        for the forgotten slot.
        """
        categories = list(self.mix)
        probs = np.array([self.mix[c] for c in categories], dtype=np.float64)
        category = str(rng.choice(categories, p=probs / probs.sum()))
        if category == "self":
            return "self"
        weights = self.category_weights(category)
        if not weights:
            return "self"
        ids = list(weights)
        w = np.array([weights[i] for i in ids], dtype=np.float64)
        return str(rng.choice(ids, p=w / w.sum()))

    def record(self, opponent_id: str, current_agent_won: bool) -> None:
        """Update one member after a game.

        OpenAI Five's quality rule: the score moves **only when the current agent
        wins**, and the ``1/(N p_i)`` normalisation makes frequently-sampled
        opponents decay proportionally faster, so the distribution self-balances.
        """
        if opponent_id == "self":
            return
        member = self.members.get(opponent_id)
        if member is None:
            raise GateRefusal(f"recorded a game against unknown opponent "
                              f"{opponent_id!r}")
        member.games += 1
        if not current_agent_won:
            return
        member.wins += 1
        n = len(self.members)
        share = self._sampling_share(opponent_id)
        if share > 0.0:
            member.quality -= self.quality_eta / (n * share)

    def _sampling_share(self, opponent_id: str) -> float:
        qualities = np.array([m.quality for m in self.members.values()])
        ids = list(self.members)
        shifted = qualities - qualities.max()
        exp = np.exp(shifted)
        total = exp.sum()
        if total <= 0.0:
            return 1.0 / max(len(ids), 1)
        return float(exp[ids.index(opponent_id)] / total)

    def summary(self) -> dict[str, object]:
        """What to log every iteration."""
        return {
            "members": len(self.members),
            "forgotten": sum(1 for m in self.members.values() if m.is_forgotten),
            "win_rates": {
                m.opponent_id: round(m.win_rate, 4)
                for m in sorted(self.members.values(), key=lambda x: x.opponent_id)
                if m.games > 0
            },
        }
