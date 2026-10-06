#!/usr/bin/env python3
"""Turn one engine decision into the token tensors the model consumes.

Token layout: see ../../docs/architecture.pdf, section 2. Every field is
read at the KIND the field-mapping gate proved it to be, which is where the
traps are:

  * `players[].active` and `current.stadium` are LISTS of <=1, not dicts
  * `select.effect` / `contextCard` are card REFERENCES, not scalars
  * `select.deck` is a LIST of card refs
  * `tools` may hold 2 (Rotom ex Multi Adapter) despite only 1 ever observed
  * `energies` is a list of EnergyType ints; `energyCards` the actual cards
  * skills carry no id in `cg.api` — the engine's own ids come from CardImpl.h
  * `current.result` is a constant -1 sentinel and is asserted, never embedded

Emits, per decision: token type/owner/slot ids, card and attack ids, a dense
numeric block, the option table with its pointer links, and the label.

ENGINE DEPENDENCY. This module reads the official engine's `cg.api` (enums,
card and attack tables) at import time and ships none of it. Point
PTCG_ENGINE_DIR at the directory that contains the `cg/` package (the Kaggle
sample submission); see demo/README.md. A packaged agent carries `cg/` beside
this file, so the variable is only needed for training and for the gates.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
if os.environ.get("PTCG_ENGINE_DIR"):
    sys.path.insert(0, os.environ["PTCG_ENGINE_DIR"])
try:
    from cg.api import (  # noqa: E402
        AreaType,
        CardType,
        EnergyType,
        LogType,
        OptionType,
        SelectContext,
        SelectType,
        SpecialConditionType,
        all_attack,
        all_card_data,
    )
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "featurize needs the official engine's `cg` package, which this "
        "repository does not ship (licence: competition use only). Set "
        "PTCG_ENGINE_DIR to the directory that contains cg/ -- see demo/README.md."
    ) from err
from deck_tracker import track  # noqa: E402
from strict_get import require_option_type  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
# Effect semantics from the engine source (a generator that is not part of this
# repository; the file it wrote is engine-derived and is not shipped). Card
# identity alone makes the model learn "Battle Cage kills my engine" from
# co-occurrence; the tags state it.
#
# THE ORDER IS PINNED HERE, NOT DERIVED FROM THE FILE. It used to be
# `sorted(tags observed in the json)`, which made the block's WIDTH a property
# of a file that is not part of this module. On a box where the json was absent
# the list came back empty, every tag slot silently became 0.0, and — now that
# blocks are appended after the tags — every later slot would have shifted
# index. A featurizer whose column meanings depend on an optional file is the
# 161.7 defect in a different costume, so a missing or disagreeing file is a
# hard failure instead.
EFFECT_TAG_NAMES = (
    "affects_both_players", "bench_damage", "checkup_effect", "coin_flip",
    "damage_prevent", "damage_reduction", "discard", "draw_cards",
    "energy_accel", "extra_tools", "heal", "knock_out_effect",
    "move_damage_counters", "once_per_turn", "prevent_damage_counter_place",
    "prize_manipulation", "retreat_free", "rule_box_condition", "search_deck",
    "special_condition", "switch",
)
N_EFFECT_TAG = len(EFFECT_TAG_NAMES)
TAG_INDEX = {name: i for i, name in enumerate(EFFECT_TAG_NAMES)}
_TAGS_PATH = ROOT / ".gate/marnie-sixthsense/effect_tags.json"
if not _TAGS_PATH.exists():
    raise RuntimeError(
        f"{_TAGS_PATH} is missing. {N_EFFECT_TAG} card-effect features would be "
        f"emitted as zeros and nothing downstream would notice — that is a "
        f"silent feature dropout, not a degraded mode. The file is derived "
        f"from the official engine and is not shipped here; supply your own "
        f"at that path."
    )
_tag_blob = json.loads(_TAGS_PATH.read_text())
CARD_TAGS = {int(k): v for k, v in _tag_blob["card_tags"].items()}
_observed_tags = sorted({t for v in CARD_TAGS.values() for t in v})
if _observed_tags != sorted(EFFECT_TAG_NAMES):
    raise RuntimeError(
        f"effect_tags.json carries tags {_observed_tags} but this featurizer "
        f"pins {sorted(EFFECT_TAG_NAMES)}. Re-pin EFFECT_TAG_NAMES and retrain: "
        f"a changed tag set silently re-maps every column after it."
    )


def effect_tag_vector(cid):
    vec = [0.0] * N_EFFECT_TAG
    for name in CARD_TAGS.get(cid or -1, ()):
        idx = TAG_INDEX.get(name)
        if idx is not None:
            vec[idx] = 1.0
    return vec
ATTACKS = {a.attackId: a for a in all_attack()}
N_ENERGY = len(EnergyType)
PAD, OOV = 0, 1  # reserved ids in every categorical table
CARD_BASE = 2  # card ids shift by this
ATTACK_BASE = 2
MAX_CTX = max(int(v) for v in SelectContext)
# Option types that carry area+index and therefore MUST resolve to a token.
# END/RETREAT/YES/NO/NUMBER carry no board reference by design.
POINTING_TYPES = {
    int(OptionType.CARD),
    int(OptionType.TOOL_CARD),
    int(OptionType.ENERGY_CARD),
    int(OptionType.ENERGY),
    int(OptionType.ATTACH),
    int(OptionType.EVOLVE),
    int(OptionType.ABILITY),
    int(OptionType.DISCARD),
    int(OptionType.PLAY),
    int(OptionType.ATTACK),
    int(OptionType.SKILL),
}

FAMILIES = [
    "PAD",
    "GLOBAL",
    "PLAYER",
    "ACTIVE",
    "BENCH",
    "HAND",
    "DISCARD_SUM",
    "UNSEEN_POOL",
    "STADIUM",
    "PRIZE",
    "LOOKING",
    "OPP_REVEALED",
    "PROMPT",
    "INTRA_TURN",
    "HISTORY",
    "OPTION",
    # One token per attached Tool and per SPECIAL Energy. The card block carried
    # only `has_tool` (a boolean over 5 distinct tools, all of which change
    # combat math) and an energy TYPE histogram, in which Grow Grass Energy is
    # identical to Basic {G} despite granting +20 HP, and Mist and Spiky Energy
    # both read as plain Colorless. Basic energies are omitted on purpose: the
    # type histogram already describes them exactly, so a token per basic energy
    # would triple occupancy for no information.
    "ATTACHED",
    # One token per DISTINCT card still unaccounted for, on top of the scalar
    # UNSEEN_POOL summary. The tracker already computes this composition and it
    # was being thrown away: the model knew "N cards remain" but not WHICH, so a
    # card sitting entirely in the deck had no token at all and therefore no
    # type, no energy type and no rules text. That is the information deck
    # searches, prize outs and "do I still have my second Boss's Orders" run on.
    # Affordable: measured p50 is 55 real tokens of a 192 cap, max 109, and a
    # 60-card list holds at most ~20 distinct cards.
    "UNSEEN_CARD",
    # The card the prompt is ABOUT, when it differs from the card that caused
    # the prompt. Both used to be raw card_tok floats inside PROMPT's numeric
    # block -- card 500 is not "twice" card 250 -- and PROMPT itself passed no
    # card id at all, so the token asking "choose a target" carried no type, no
    # energy type and no rules text for the card doing the asking.
    "CONTEXT_CARD",
]
FAM = {name: i for i, name in enumerate(FAMILIES)}
# Card SEMANTICS, as categorical tokens rather than numeric slots.
#
# WHY THIS EXISTS. cardType and energyType are in the catalog and were never
# passed to the model, so "is this a Supporter" and "is this a Fire Pokemon"
# were only learnable from the card-ID embedding -- i.e. memorised per card.
# 1040 of 1270 card rows are never trained, so an unseen card could not even
# be identified as a Supporter, let alone played correctly. These two tables
# have 8 and 13 rows and every row is common, so they are always well trained
# and they generalise to cards the corpus never contained.
#
# Categorical, not one-hot in `numeric`: the numeric block is 54/64 full, and
# family/owner already establish the embedding-table pattern for exactly this.
N_CARD_TYPE = 8      # 7 CardType members + 1 for "no card on this token"
N_ENERGY_TYPE = 13   # 12 EnergyType members + 1 for "no card"
CT_NONE, ET_NONE = 0, 0

CAP = 192
UNSEEN_CARD_CAP = 24   # 109 (measured max real tokens) + 24 stays well under CAP
OPTION_CAP = 64
# Shared pad width for EVERY family's numeric block and for the option table.
# 80 -> 160 -> 269. 269 is the MEASURED width of the widest real block (the card
# block; a per-family width census, not included here, printed it), not a round number:
# the card block grows retreat and checkup arithmetic, the two attacks priced by
# slot, the big attack's per-type shortfall, unused energy by type, the knockout
# race in both directions, the energy the owner can actually reach, weakness and
# resistance as types, the opposing board's worst hit, the engine's own legality
# for that board slot, the turn locks and whether this knockout ends the match.
# The option table is 172 wide and shares the tensor, so option rows carry 97
# columns of pad; that is the cost of one shared width, and the layout gate
# separates "pad" from "constant zero inside a real block" so the two are never
# confused again.
# model_ss.NUM_W / OPT_NUM_W must match; a mismatch is the defect that scored
# 161.7 against 767.5 with export parity reading 1.0000 throughout.
NUMERIC_WIDTH = 269
OPTION_NUMERIC_WIDTH = 269  # same tensor width; named so the seam gate can read it
N_BENCH_SLOT = 8    # bench positions 0..7; ACTIVE is a separate flag already
N_ENERGY_SLOT = 8   # attached-energy index within one Pokemon
N_TOOL_SLOT = 4     # attached-tool index; the catalog holds 5 distinct tools
N_CARD_TYPE_ONEHOT = len(CardType)          # 7
N_SPECIAL_CONDITION = len(SpecialConditionType)  # 5
N_SKILL_SLOT = 4    # which skill on the card an ABILITY/SKILL option activates
DEFICIT_CAP = 9     # energy shortfalls beyond this are equivalent in practice
HITS_CAP = 9        # "more than nine hits to knock out" is one situation
N_AREA = max(int(a) for a in AreaType) + 1   # AreaType ids run 1..12
DAMAGE_COUNTER = 10  # one damage counter
POISON_DAMAGE = 10  # one damage counter per checkup; the only status damage
                    # whose magnitude the API states without a coin flip
DRAW_LOOKAHEAD = 3  # turns of draw the hypergeometric energy odds cover
COLORLESS = int(EnergyType.COLORLESS)
RAINBOW = int(EnergyType.RAINBOW)
TEAM_ROCKET = int(EnergyType.TEAM_ROCKET)
# EnergyType comments: RAINBOW is "Every Types", TEAM_ROCKET is "PSYCHIC and
# DARKNESS". Their coverage sets are laminar ({t} subset {P,D} subset all), so
# paying least-flexible-first is provably optimal and the shortfall below is
# exact, not a heuristic.
TR_COVERS = frozenset({int(EnergyType.PSYCHIC), int(EnergyType.DARKNESS)})
# Set to a dict by a width-census script (not included) to record each family's real
# PRE-PAD width. None in every production path, so it costs one `is not None`.
# The pad hides how full a block is, and "how full is it" is the number whose
# absence let a 64-wide featurizer ship against 80-wide weights.
BLOCK_WIDTH_SINK: dict[str, int] | None = None
CHANCE_LOG_TYPES = {
    int(LogType.SHUFFLE),
    int(LogType.DRAW),
    int(LogType.DRAW_REVERSE),
    int(LogType.COIN),
}


def is_special_energy(cid: int | None) -> bool:
    """Anything the type histogram cannot already describe."""
    meta = CARDS.get(cid)
    name = getattr(meta, "name", "") or ""
    return bool(name) and not name.startswith("Basic ")


def card_tok(cid: int | None) -> int:
    if cid is None:
        return PAD
    return CARD_BASE + cid if cid in CARDS else OOV


def attack_tok(attack_id: int | None) -> int:
    if attack_id is None:
        return PAD
    return ATTACK_BASE + attack_id if attack_id in ATTACKS else OOV


def ctx_tok(value) -> int:
    """SelectContext grows during the competition — unseen values get OOV."""
    if value is None:
        return PAD
    return 2 + int(value) if 0 <= int(value) <= MAX_CTX else OOV


N_CTX_SLOTS = MAX_CTX + 3          # PAD, OOV, then 0..MAX_CTX
N_SELTYPE_SLOTS = 13               # PAD, OOV, then the 11 SelectType values


def ctx_onehot(value) -> list[float]:
    """One-hot over ctx_tok's own token space, so PAD and OOV stay distinct."""
    v = [0.0] * N_CTX_SLOTS
    t = ctx_tok(value)
    v[t if 0 <= t < N_CTX_SLOTS else OOV] = 1.0
    return v


def seltype_onehot(value) -> list[float]:
    v = [0.0] * N_SELTYPE_SLOTS
    if value is None:
        v[PAD] = 1.0
        return v
    t = 2 + int(value)
    v[t if 0 <= t < N_SELTYPE_SLOTS else OOV] = 1.0
    return v


def energy_vector(card: dict) -> list[int]:
    counts = [0] * N_ENERGY
    for e in (card or {}).get("energies") or []:
        if isinstance(e, int) and 0 <= e < N_ENERGY:
            counts[e] += 1
    return counts


# ---------------------------------------------------------------------------
# Attack readiness. WHY IT CHANGES A DECISION: until now a Pokemon's attack
# requirements were not a property of the Pokemon at all, only of a legal
# option that already carried an attackId. So "I am one Water away from
# attacking" existed as a feature ONLY on turns where the attack was already
# affordable, i.e. exactly when the number is uninteresting, and bench Pokemon
# — which generate no attack options — had no attack information whatsoever.
# `CardData.attacks` was in the catalog the whole time and read 0 times.
# ---------------------------------------------------------------------------
def attack_cost(attack_id: int | None) -> list[int]:
    cost = [0] * N_ENERGY
    atk = ATTACKS.get(attack_id)
    if atk:
        for e in atk.energies or []:
            if isinstance(e, int) and 0 <= e < N_ENERGY:
                cost[e] += 1
    return cost


def attack_damage(attack_id: int | None) -> int:
    atk = ATTACKS.get(attack_id)
    if not atk:
        return 0
    try:
        return int(str(atk.damage).strip() or 0)
    except ValueError:  # 0 of 1556 catalog attacks hit this; kept as a guard
        return 0


def energy_payment(
    have: list[int], cost: list[int]
) -> tuple[int, list[int], list[int]]:
    """Exact minimum extra energies needed to pay `cost` out of `have`, and the
    energies left over once it is paid.

    Typed requirements are paid from their own type first, then TEAM_ROCKET
    where it covers that type, then RAINBOW; COLORLESS is paid last out of
    whatever survives. Coverage sets are laminar so this greedy order is
    optimal — the total is the true minimum, never an estimate.

    The LEFTOVER pool is returned as well because "energy this Pokemon is
    holding that its attack does not need" is the exact quantity an energy-move
    or discard-cost decision runs on, and it was being computed and thrown away.
    """
    pool = list(have)
    short = [0] * N_ENERGY
    for t in range(N_ENERGY):
        if t == COLORLESS:
            continue
        need = cost[t]
        if need <= 0:
            continue
        for src in (t, TEAM_ROCKET if t in TR_COVERS else None, RAINBOW):
            if need <= 0 or src is None:
                continue
            take = min(need, pool[src])
            pool[src] -= take
            need -= take
        short[t] = need
    generic = cost[COLORLESS]
    short[COLORLESS] = max(0, generic - sum(pool))
    # spend the generic requirement out of the least flexible survivors first,
    # so what is left over is the most reusable energy — same laminar argument
    for src in [t for t in range(N_ENERGY) if t not in (RAINBOW, TEAM_ROCKET)] \
            + [TEAM_ROCKET, RAINBOW]:
        if generic <= 0:
            break
        take = min(generic, pool[src])
        pool[src] -= take
        generic -= take
    return sum(short), short, pool


def energy_shortfall(have: list[int], cost: list[int]) -> tuple[int, list[int]]:
    """Back-compatible view of energy_payment: (total short, per-type short)."""
    total, short, _pool = energy_payment(have, cost)
    return total, short


def prize_value(meta) -> float:
    """Prizes the opponent takes when this Pokemon is knocked out.

    cg.api states it exactly: megaEx is THREE, ex (excluding mega) is two,
    everything else one. This used to return 2.0 for megaEx, which understated
    the single most expensive knockout in the game by a whole prize.
    """
    if meta is None:
        return 0.0
    if getattr(meta, "megaEx", False):
        return 3.0
    if getattr(meta, "ex", False):
        return 2.0
    return 1.0


def hits_to_ko(hp: float, per_hit: float) -> float:
    """Attacks needed to knock something out, 0 when it can never happen."""
    if per_hit <= 0 or hp <= 0:
        return 0.0
    return float(min(HITS_CAP, math.ceil(hp / per_hit)))


def card_attack_ids(cid: int | None) -> list[int]:
    meta = CARDS.get(cid)
    return [a for a in (getattr(meta, "attacks", None) or []) if isinstance(a, int)]


def effective_damage(attack_id: int | None, attacker_meta, defender_meta) -> int:
    """Printed damage with the weakness multiplier applied.

    Weakness is x2 and decides most knockouts. Resistance magnitude is NOT in
    the catalog (CardData carries `resistance` as a type, never a value), so it
    is reported as a separate flag rather than guessed at -30 — the caller sees
    an exact upper bound plus the exact fact that it is resisted.
    """
    dmg = attack_damage(attack_id)
    if dmg and attacker_meta is not None and defender_meta is not None:
        if defender_meta.weakness is not None and \
                defender_meta.weakness == attacker_meta.energyType:
            dmg *= 2
    return dmg


@lru_cache(maxsize=16384)
def _best_affordable_damage(
    attacker_id: int | None, energies: tuple[int, ...], defender_id: int | None,
    extra: int | None = None,
) -> int:
    """Best damage `attacker_id` can deal to `defender_id` with `energies`
    (plus one hypothetical `extra` energy), counting only attacks it can pay for.

    Cached because it is the same question for every token in a decision: the
    opposing active's threat does not change from the point of view of my
    active, my bench or my hand, and the "what if they attach one more" scan
    asks it 12 more times per Pokemon.
    """
    att_meta = CARDS.get(attacker_id)
    def_meta = CARDS.get(defender_id)
    have = list(energies)
    if extra is not None and 0 <= extra < N_ENERGY:
        have[extra] += 1
    best = 0
    for aid in card_attack_ids(attacker_id):
        if energy_payment(have, attack_cost(aid))[0] == 0:
            best = max(best, effective_damage(aid, att_meta, def_meta))
    return best


def attack_profile(
    card: dict, defender: dict | None, supply: dict | None = None
) -> list[float]:
    """119 slots: what this Pokemon can do, what it needs to do more, and how
    the knockout race between it and the Pokemon opposite actually ends.

    Emitted for ACTIVE, BENCH and HAND alike. On a HAND token `have` is empty,
    so the shortfall is the full printed cost — which is exactly the question
    "what would it take to make this thing attack if I benched it".

    `supply` carries the energy the owner can actually reach (hand, unseen pool)
    so "one Water short" can be answered with "and I am holding a Water".
    """
    cid = (card or {}).get("id")
    meta = CARDS.get(cid)
    have = energy_vector(card)
    ids = card_attack_ids(cid)
    defender_meta = CARDS.get((defender or {}).get("id"))

    best_key, best_short_vec, best_short_id = None, [0] * N_ENERGY, None
    strongest_id, strongest_dmg = None, -1
    costs, n_affordable, best_afford_dmg = [], 0, 0
    max_dmg = 0
    for aid in ids:
        cost = attack_cost(aid)
        total, vec = energy_shortfall(have, cost)
        dmg = attack_damage(aid)
        costs.append(sum(cost))
        max_dmg = max(max_dmg, dmg)
        if total == 0:
            n_affordable += 1
            best_afford_dmg = max(best_afford_dmg, dmg)
        # tie-break on damage so "cheapest to reach" prefers the better attack
        key = (total, -dmg, aid)
        if best_key is None or key < best_key:
            best_key, best_short_vec, best_short_id = key, vec, aid
        if dmg > strongest_dmg:
            strongest_id, strongest_dmg = aid, dmg
    best_short = best_key[0] if best_key else 0

    def _one(aid: int | None) -> list[float]:
        cost = attack_cost(aid)
        total, _vec = energy_shortfall(have, cost)
        return [
            *cost,
            float(sum(cost)),
            float(attack_damage(aid)),
            float(min(total, DEFICIT_CAP)),
            1.0 if (aid is not None and total == 0) else 0.0,
        ]

    # matchup against the OPPOSING active: the prize-race arithmetic. Both
    # sides' energies and card ids are visible, so every number here is exact.
    def_hp = float((defender or {}).get("hp") or 0)
    my_hp = float(card.get("hp") or (meta.hp if meta else 0) or 0)
    out_now = out_any = 0
    for aid in ids:
        d = effective_damage(aid, meta, defender_meta)
        out_any = max(out_any, d)
        if energy_shortfall(have, attack_cost(aid))[0] == 0:
            out_now = max(out_now, d)
    incoming = 0
    incoming_plus_one = 0
    if defender is not None and defender_meta is not None:
        d_have = tuple(energy_vector(defender))
        d_id = defender.get("id")
        incoming = _best_affordable_damage(d_id, d_have, cid)
        # what they hit me for if they make their one attachment next turn.
        # Their hand is hidden, so the maximum over the 12 energy types is the
        # exact upper bound rather than a guess about what they hold.
        incoming_plus_one = max(
            (_best_affordable_damage(d_id, d_have, cid, t) for t in range(N_ENERGY)),
            default=0,
        )

    # --- the two attacks BY SLOT (18) ---------------------------------------
    # WHY IT CHANGES A DECISION: the cheapest-to-reach and highest-damage
    # profiles above collide whenever the cheaper attack also hits hardest, and
    # then the other attack was priced nowhere at all — the model could not see
    # that a Pokemon it is holding one energy short of has a second, affordable
    # attack. No card in the catalog has more than 2 attacks (measured over all
    # 1267 rows), so two fixed slots describe every Pokemon exactly.
    def _slot(k: int) -> list[float]:
        aid = ids[k] if k < len(ids) else None
        cost = attack_cost(aid)
        total, _short, _pool = energy_payment(have, cost)
        dmg_eff = effective_damage(aid, meta, defender_meta)
        return [
            1.0 if aid is not None else 0.0,
            float(sum(cost)),
            float(cost[COLORLESS]),
            float(sum(cost) - cost[COLORLESS]),
            float(attack_damage(aid)),
            float(dmg_eff),
            float(min(total, DEFICIT_CAP)),
            1.0 if (aid is not None and total == 0) else 0.0,
            1.0 if (dmg_eff and def_hp and dmg_eff >= def_hp) else 0.0,
        ]

    # --- what the BIG attack still needs, per type (12) ----------------------
    # The per-type deficit above is for the cheapest attack to reach. "What am I
    # missing for the attack that actually wins the game" is a different vector
    # and is the one a player builds their next three attachments around.
    _, strong_short, _ = energy_payment(have, attack_cost(strongest_id))
    # --- energy this Pokemon is NOT using (12) ------------------------------
    # Leftover after paying the cheapest reachable attack: exactly the energy an
    # energy-move effect can take away for free, and exactly what is wasted if
    # this Pokemon is knocked out.
    _, _, spare = energy_payment(have, attack_cost(best_short_id))

    # --- the knockout race, both directions (9) -----------------------------
    # hp and maxHp were given; "how many more swings does this take, and do I
    # get there first" was left as arithmetic over four tokens.
    h_them_now = hits_to_ko(def_hp, out_now)
    h_them_ready = hits_to_ko(def_hp, out_any)
    h_me = hits_to_ko(my_hp, incoming)
    race = [
        float(max(0.0, out_now - def_hp)),            # overkill if I attack now
        float(max(0.0, def_hp - out_now)) if def_hp else 0.0,  # damage still owed
        h_them_now,
        h_them_ready,
        float(max(0.0, incoming - my_hp)),            # their overkill on me
        h_me,
        1.0 if (my_hp and incoming < my_hp) else 0.0,  # I survive their best
        1.0 if (h_them_now and (not h_me or h_them_now <= h_me)) else 0.0,  # I KO first
        prize_value(defender_meta) - prize_value(meta),  # what the trade is worth
    ]

    # --- can I actually reach the energy I am short of (6) -------------------
    # A deficit only matters if the energy exists somewhere I can get it. My
    # hand is fully visible and the unseen pool is exact by card id, so both
    # answers are computable; they are zero on opponent tokens by construction
    # (their hand is None) and owner_is_self marks those rows.
    hand_e = list((supply or {}).get("hand_energy") or [0] * N_ENERGY)
    unseen_e = list((supply or {}).get("unseen_energy") or [0] * N_ENERGY)
    best_cost = attack_cost(best_short_id)
    hand_gain = hand_now = 0
    for t in range(N_ENERGY):
        if hand_e[t] <= 0:
            continue
        plus = list(have)
        plus[t] += 1
        after = energy_payment(plus, best_cost)[0]
        if after < best_short:
            hand_gain = 1
        if after == 0:
            hand_now = 1
    unseen_cover = 0
    for t in range(N_ENERGY):
        if unseen_e[t] <= 0:
            continue
        plus = list(have)
        plus[t] += 1
        if energy_payment(plus, best_cost)[0] < best_short:
            unseen_cover += unseen_e[t]

    return [
        float(len(ids)),                                   # n_attacks
        1.0 if n_affordable else 0.0,                      # can_attack_now
        float(n_affordable),
        float(min(costs) if costs else 0),                 # min_attack_cost
        float(max(costs) if costs else 0),                 # max_attack_cost
        float(max_dmg),                                    # max printed damage
        float(best_afford_dmg),
        float(min(best_short, DEFICIT_CAP)),               # attack_deficit
        *[float(min(v, DEFICIT_CAP)) for v in best_short_vec],   # per-type deficit
        *_one(best_short_id),                              # cheapest-to-reach attack
        *_one(strongest_id),                               # highest-damage attack
        float(out_now),                                    # best damage I can deal now
        float(out_any),                                    # ... if fully energised
        1.0 if (out_now and def_hp and out_now >= def_hp) else 0.0,   # KO now
        1.0 if (out_any and def_hp and out_any >= def_hp) else 0.0,   # KO when ready
        float(incoming),                                   # their best on me now
        1.0 if (incoming and my_hp and incoming >= my_hp) else 0.0,   # they KO me
        float(max(0.0, my_hp - incoming)),                 # my hp after their best
        1.0 if (defender_meta is not None and meta is not None
                and defender_meta.resistance is not None
                and defender_meta.resistance == meta.energyType) else 0.0,
        # --- appended blocks; see the comments where each is computed --------
        *_slot(0),                                         # attack slot 0 (9)
        *_slot(1),                                         # attack slot 1 (9)
        *[float(min(v, DEFICIT_CAP)) for v in strong_short],     # big-attack deficit (12)
        *[float(v) for v in spare],                        # unused energy by type (12)
        *race,                                             # knockout race (9)
        float(incoming_plus_one),                          # their best after 1 attach
        1.0 if (incoming_plus_one and my_hp
                and incoming_plus_one >= my_hp) else 0.0,  # ...and it kills me
        float(sum(hand_e)),                                # energy in my hand
        float(hand_gain),                                  # a hand energy shrinks it
        float(hand_now),                                   # a hand energy pays it off
        float(sum(1 for v in best_short_vec if v > 0)),     # how many types short
        float(min(unseen_cover, 60)),                      # copies left in deck+prize
        float(sum(spare)),                                 # spare energy total
    ]


def evolution_profile(
    card: dict, in_play: bool, hand_by_pre: dict[str, list[int]],
    board_names: set[str],
) -> list[float]:
    """6 slots. WHY IT CHANGES A DECISION: the evolution LOCK (appearThisTurn)
    was present but the CONJUNCTION with a matching card in hand was not, so
    deciding whether to bench a basic now meant joining one board token against
    24 hand tokens through attention with nothing marking the match.
    `CardData.evolvesFrom` names the pre-evolution and was read 0 times.
    """
    cid = (card or {}).get("id")
    meta = CARDS.get(cid)
    name = getattr(meta, "name", None)
    targets = hand_by_pre.get(name or "", []) if (name and in_play) else []
    # a Pokemon played this turn cannot evolve this turn
    unlocked = in_play and not card.get("appearThisTurn")
    gain = 0
    if unlocked and targets:
        base = float(card.get("maxHp") or (meta.hp if meta else 0) or 0)
        gain = max(
            (float(getattr(CARDS.get(t), "hp", 0) or 0) - base for t in targets),
            default=0,
        )
    return [
        1.0 if getattr(meta, "stage1", False) else 0.0,
        1.0 if getattr(meta, "stage2", False) else 0.0,
        1.0 if (unlocked and targets) else 0.0,        # can_evolve_now
        float(len(targets)),                           # evolution targets in hand
        float(max(0.0, gain)),                         # hp gained by evolving
        # for a HAND card: is the thing it evolves from actually in play
        1.0 if (getattr(meta, "evolvesFrom", None) in board_names) else 0.0,
    ]


def retreat_profile(
    card: dict, meta, status: tuple, retreat_used: bool, is_active: bool,
) -> list[float]:
    """6 slots. WHY IT CHANGES A DECISION: retreatCost was on the token as a raw
    number and the attached-energy total was on the same token, but the
    SUBTRACTION between them — "can this thing actually leave" — was not, and
    neither was the pair of rules that override it (Asleep and Paralyzed cannot
    retreat; one retreat per turn). A bench Pokemon's retreat cost is what you
    pay to bring it in, so this is emitted for the whole board, not the active.
    """
    cost = float(getattr(meta, "retreatCost", 0) or 0)
    have = float(sum(energy_vector(card)))
    st = (tuple(status) + (0,) * N_SPECIAL_CONDITION)[:N_SPECIAL_CONDITION]
    # STATUS_FIELDS order: asleep, burned, confused, paralyzed, poisoned
    locked = bool(is_active and (st[0] or st[3]))
    can_pay = have >= cost
    return [
        float(max(0.0, cost - have)),                       # retreat_deficit
        1.0 if can_pay else 0.0,                            # can pay the cost
        1.0 if locked else 0.0,                             # asleep/paralyzed
        1.0 if retreat_used else 0.0,                       # once-per-turn lock
        1.0 if (is_active and can_pay and not locked
                and not retreat_used) else 0.0,             # can_retreat_now
        1.0 if cost == 0 else 0.0,                          # free retreat
    ]


def checkup_profile(
    card: dict, status: tuple, is_active: bool
) -> list[float]:
    """5 slots. WHY IT CHANGES A DECISION: a poisoned Pokemon on 10 HP is dead
    at the checkup whatever else happens, and that is the difference between
    retreating it and attacking with it. The status flags were emitted, the HP
    was emitted, and the one-line consequence was not. Only POISON has a damage
    magnitude the API states without a coin flip, so only poison is priced —
    burn is left as its flag rather than guessed at.
    """
    st = (tuple(status) + (0,) * N_SPECIAL_CONDITION)[:N_SPECIAL_CONDITION]
    asleep, _burned, _confused, paralyzed, poisoned = st
    hp = float(card.get("hp") or 0)
    dmg = float(POISON_DAMAGE) if (is_active and poisoned) else 0.0
    return [
        dmg,                                                # damage at checkup
        float(max(0.0, hp - dmg)),                          # hp after it
        1.0 if (dmg and hp and hp <= dmg) else 0.0,         # dies at checkup
        1.0 if (is_active and (asleep or paralyzed)) else 0.0,  # cannot act
        float(sum(st)) if is_active else 0.0,               # how many conditions
    ]


def attached_tag_vector(card: dict) -> list[float]:
    """21 slots: the effect tags of everything ATTACHED to this Pokemon.

    WHY IT CHANGES A DECISION: tools and special energies are their own tokens,
    so "this Pokemon is carrying damage_prevent" required a board-to-board join
    that the attention mask does not permit. All five tools in the catalog
    change combat math, and Grow Grass / Mist / Spiky Energy change it too.
    """
    vec = [0.0] * N_EFFECT_TAG
    for item in [*(card.get("tools") or []), *(card.get("energyCards") or [])]:
        cid = item.get("id") if isinstance(item, dict) else item
        if not isinstance(cid, int):
            continue
        for i, v in enumerate(effect_tag_vector(cid)):
            if v:
                vec[i] = 1.0
    return vec


def board_threat(card: dict, board: list[dict]) -> list[float]:
    """2 slots: the worst hit this Pokemon can take from the opposing board as a
    whole, not just from the Pokemon currently opposite it.

    WHY IT CHANGES A DECISION: the incoming-damage number above is against the
    defending active only, so a bench Pokemon that dies the moment they switch
    reads as perfectly safe. Both boards are fully visible, so the maximum is
    exact.
    """
    cid = (card or {}).get("id")
    hp = float((card or {}).get("hp") or 0)
    worst = 0
    for other in board:
        if not isinstance(other, dict):
            continue
        worst = max(worst, _best_affordable_damage(
            other.get("id"), tuple(energy_vector(other)), cid))
    return [float(worst), 1.0 if (worst and hp and worst >= hp) else 0.0]


def option_semantics(
    opt: dict, kind: int, source_cid: int | None, dest_card: dict | None,
    target_card: dict | None, ctx: dict,
) -> list[float]:
    """138 slots of per-option-type meaning (the option ROW is 172: these plus
    the index one-hots and attack_features). Everything below is computed from
    the catalog and the observation; nothing is estimated.

    WHY IT CHANGES A DECISION, by type:
      ATTACH (19.6% of options) had ZERO features — every attach in a turn was
        the same zero vector distinguished only by two pointer embeddings. That
        is the failure that shipped: "played on the ladder without attaching
        energy". The one number that decides it — does this attachment make an
        attack affordable on THIS destination — is computed here.
      RETREAT (4.4%) had nothing at all: no numeric, no pointer, no card id.
        Retreating a free-retreat Pokemon and a 3-cost one were identical rows.
      CARD (33.4%) had zero features and 55% of its pointers collapse onto one
        shared zone token, so "switch to the 60hp with two energy" and "switch
        to the fresh 190hp ex" were told apart by card-id embedding alone — and
        1040 of 1270 card rows are never trained.
      ATTACK carried printed damage but never the KO: weakness sat on a
        different token as a bare boolean and nothing multiplied it.
      SPECIAL_CONDITION options were byte-identical across POISON/BURN/SLEEP/
        PARALYZE/CONFUSE — sleep is a coin-flip lock, confusion is a
        damage-yourself risk, and they are worth different amounts.
      ABILITY: two abilities on the same card produced identical rows.
    """
    active = ctx["active"]
    defender = ctx["defender"]
    unseen = ctx["unseen"]
    my_prizes = float(ctx.get("my_prizes") or 0)
    opp_prizes = float(ctx.get("opp_prizes") or 0)
    src_meta = CARDS.get(source_cid)

    # -- ATTACH: the energy being moved, and what it unlocks on the target ----
    is_attach = kind == int(OptionType.ATTACH)
    e_onehot = [0.0] * N_ENERGY
    if is_attach and src_meta is not None and src_meta.energyType is not None:
        et = int(src_meta.energyType)
        if 0 <= et < N_ENERGY:
            e_onehot[et] = 1.0
    before = after = [0.0] * 4
    if is_attach and isinstance(dest_card, dict):
        base = attack_profile(dest_card, defender)
        hypo = dict(dest_card)
        hypo["energies"] = [
            *(dest_card.get("energies") or []),
            *([int(src_meta.energyType)] if src_meta is not None
              and src_meta.energyType is not None else []),
        ]
        post = attack_profile(hypo, defender)
        # attack_profile slots: 1 can_attack_now, 6 best_affordable_damage,
        # 7 attack_deficit, 62-ish KO flags -> read by index, kept adjacent.
        before = [base[1], base[6], base[7], base[54]]
        after = [post[1], post[6], post[7], post[54]]

    # -- ATTACK: the knockout arithmetic -------------------------------------
    attack_id = opt.get("attackId")
    def_meta = CARDS.get((defender or {}).get("id"))
    eff = effective_damage(attack_id, CARDS.get((active or {}).get("id")), def_meta)
    def_hp = float((defender or {}).get("hp") or 0)
    printed = attack_damage(attack_id)

    # -- RETREAT --------------------------------------------------------------
    act_meta = CARDS.get((active or {}).get("id"))
    r_cost = float(getattr(act_meta, "retreatCost", 0) or 0)
    r_have = float(sum(energy_vector(active))) if active else 0.0

    # -- the referent card, for every option that names one --------------------
    ref = target_card if isinstance(target_card, dict) else None
    ref_cid = (ref or {}).get("id", source_cid)
    ref_meta = CARDS.get(ref_cid)
    ct = [0.0] * N_CARD_TYPE_ONEHOT
    if ref_meta is not None and 0 <= int(ref_meta.cardType) < N_CARD_TYPE_ONEHOT:
        ct[int(ref_meta.cardType)] = 1.0
    ret = [0.0] * N_ENERGY
    if ref_meta is not None and ref_meta.energyType is not None \
            and 0 <= int(ref_meta.energyType) < N_ENERGY:
        ret[int(ref_meta.energyType)] = 1.0
    ref_ev = energy_vector(ref) if ref else [0] * N_ENERGY

    sc = [0.0] * N_SPECIAL_CONDITION
    sct = opt.get("specialConditionType")
    if sct is not None and 0 <= int(sct) < N_SPECIAL_CONDITION:
        sc[int(sct)] = 1.0

    is_supporter = ref_meta is not None and int(ref_meta.cardType) == int(CardType.SUPPORTER)
    is_stadium = ref_meta is not None and int(ref_meta.cardType) == int(CardType.STADIUM)

    return [
        # ATTACH (19)
        1.0 if is_attach else 0.0,
        *e_onehot,
        *before,
        *after,
        1.0 if (is_attach and before[2] > 0 and after[2] == 0) else 0.0,  # unlocks
        float(max(0.0, after[1] - before[1])),                           # dmg unlocked
        # ATTACK / KO (8)
        float(printed),
        float(eff),
        def_hp,
        1.0 if (attack_id is not None and eff and def_hp and eff >= def_hp) else 0.0,
        float(max(0.0, eff - def_hp)),                                   # overkill
        prize_value(def_meta),                                           # prizes on KO
        1.0 if (attack_id in ATTACKS and printed == 0) else 0.0,         # status attack
        1.0 if (def_meta is not None and act_meta is not None
                and def_meta.resistance is not None
                and def_meta.resistance == act_meta.energyType) else 0.0,
        # RETREAT (5)
        1.0 if kind == int(OptionType.RETREAT) else 0.0,
        r_cost,
        r_have,
        float(max(0.0, r_cost - r_have)),
        1.0 if ctx["retreated"] else 0.0,
        # referent card (14 + 7 + 12 = 33)
        float((ref or {}).get("hp") or 0),
        float((ref or {}).get("maxHp") or 0),
        float(max(0, ((ref or {}).get("maxHp") or 0) - ((ref or {}).get("hp") or 0))),
        float(sum(ref_ev)),
        float(getattr(ref_meta, "retreatCost", 0) or 0),
        float(getattr(ref_meta, "hp", 0) or 0),
        1.0 if getattr(ref_meta, "ex", False) else 0.0,
        1.0 if getattr(ref_meta, "basic", False) else 0.0,
        1.0 if getattr(ref_meta, "stage1", False) else 0.0,
        1.0 if getattr(ref_meta, "stage2", False) else 0.0,
        prize_value(ref_meta),
        float(unseen.get(ref_cid, 0)) if ref_cid is not None else 0.0,
        1.0 if (ref or {}).get("appearThisTurn") else 0.0,
        float(len((ref or {}).get("tools") or [])),
        *ct,
        *ret,
        # what the referent DOES: the tag vocabulary existed and never reached
        # an option, so search_deck / energy_accel / heal / switch were invisible
        # on exactly the rows where the model has to choose between them. (21)
        *effect_tag_vector(ref_cid),
        # SPECIAL_CONDITION (5) + ability ambiguity (1) + play locks (3)
        *sc,
        # NOT a skill one-hot: cg.api Option has no skillIndex, and two abilities
        # on one card produce options with the same area, index, serial and
        # cardId. Which ability an ABILITY option activates is genuinely not in
        # the observation, so the count is emitted instead — it at least tells
        # the model the row it is looking at is ambiguous.
        float(len(getattr(src_meta, "skills", None) or [])),
        1.0 if (is_supporter and ctx["supporter_played"]) else 0.0,
        1.0 if (is_stadium and ctx["stadium_played"]) else 0.0,
        1.0 if (is_attach and ctx["energy_attached"]) else 0.0,
        # --- WHICH ZONE this option reaches into (12) ------------------------
        # 55% of CARD pointers collapse onto one shared zone token, so the
        # pointer embedding cannot say whether the row means "the card in my
        # discard" or "the card on their bench". `area` says it exactly and was
        # never emitted. AreaType starts at DECK=1, so there is no slot 0 — a
        # column that can never be set is exactly what this file refuses to emit.
        *[1.0 if opt.get("area") == a else 0.0 for a in range(1, N_AREA)],
        # --- the prize race, on every option row (5) ------------------------
        # An option is chosen against the score, not in a vacuum: the same trade
        # is correct at 6-6 and losing at 1-2. These live on GLOBAL, which the
        # option table cannot attend to.
        float(my_prizes),
        float(opp_prizes),
        float(opp_prizes - my_prizes),
        float(ctx.get("my_kos") or 0),
        float(ctx.get("opp_kos") or 0),
        # --- does THIS action end the match (2) ------------------------------
        1.0 if (attack_id is not None and eff and def_hp and eff >= def_hp
                and 0 < my_prizes <= prize_value(def_meta)) else 0.0,
        float(max(0.0, my_prizes - (prize_value(def_meta)
              if (attack_id is not None and eff and def_hp and eff >= def_hp)
              else 0.0))),
        # --- how far off a knockout this attack is (2) -----------------------
        hits_to_ko(def_hp, float(eff)),
        float(max(0.0, def_hp - eff)) if (attack_id is not None and def_hp) else 0.0,
        # No retreat-legality flags here: the engine only ever OFFERS a legal
        # retreat, so "can afford it" is constant 1 on every RETREAT row and
        # "asleep or paralyzed" is constant 0 on every one. Both measured dead
        # across 94,154 option rows. The cost/have/deficit triple above carries
        # the part that varies.
        # --- room and resources for a PLAY (2) -------------------------------
        float(ctx.get("bench_open") or 0),
        float(ctx.get("hand_count") or 0),
        # --- what an EVOLVE actually buys (2) --------------------------------
        # hp gained and damage carried over are the two numbers that decide
        # whether evolving saves the Pokemon or wastes the card.
        float(max(0.0, float(getattr(ref_meta, "hp", 0) or 0)
                  - float((dest_card or {}).get("maxHp") or 0)))
        if kind == int(OptionType.EVOLVE) else 0.0,
        float(max(0, ((dest_card or {}).get("maxHp") or 0)
                  - ((dest_card or {}).get("hp") or 0)))
        if kind == int(OptionType.EVOLVE) else 0.0,
        # --- the referent's special conditions (5) ---------------------------
        # Status is a PlayerState fact about that player's ACTIVE, so it is
        # exact for an option that points at an active and zero elsewhere.
        *_referent_status(opt, ctx),
        # --- what the referent would DO if it were promoted (4) --------------
        # This is the switch decision: "put in the one that can attack", which
        # needed the referent's energies, its attack costs and my defender all
        # joined across three tokens.
        *_referent_threat(ref, ref_cid, ctx),
        # --- the attach destination after the attachment (2) -----------------
        float(len((dest_card or {}).get("energies") or []) + 1)
        if is_attach and isinstance(dest_card, dict) else 0.0,
        hits_to_ko(def_hp, float(after[1])) if is_attach else 0.0,
    ]


def _referent_status(opt: dict, ctx: dict) -> list[float]:
    """5 slots: the special conditions on the Pokemon this option points at."""
    if opt.get("area") != int(AreaType.ACTIVE):
        return [0.0] * N_SPECIAL_CONDITION
    # DEFAULT-OK: unreachable, and asserted rather than assumed. The only
    # caller is option_semantics (:950), whose only caller passes opt_ctx
    # (:1994), which always sets "seat" (:1929). A defaulted seat would be a
    # WHOLE-CORPUS mislabel -- the 52%-of-teacher-games bug -- so the chain is
    # pinned here: if a second caller ever appears without a seat, this raises
    # instead of silently scoring every option against seat 0.
    if "seat" not in ctx:
        raise KeyError(
            "_referent_status: ctx has no 'seat'. Defaulting it to 0 scores "
            "every option against player 0 and mislabels the whole corpus.")
    seat = ctx["seat"]
    mine = opt.get("playerIndex", seat) == seat
    st = (ctx.get("my_status") if mine else ctx.get("opp_status")) or ()
    st = tuple(st) + (0,) * N_SPECIAL_CONDITION
    return [float(v) for v in st[:N_SPECIAL_CONDITION]]


def _referent_threat(ref: dict | None, ref_cid, ctx: dict) -> list[float]:
    """4 slots: what the referent Pokemon can do to the Pokemon opposite it."""
    if not isinstance(ref, dict) or ref_cid is None:
        return [0.0] * 4
    have = tuple(energy_vector(ref))
    defender = ctx["defender"]
    dmg = _best_affordable_damage(ref_cid, have, (defender or {}).get("id"))
    def_hp = float((defender or {}).get("hp") or 0)
    costs = [sum(attack_cost(a)) for a in card_attack_ids(ref_cid)]
    short = min(
        (energy_payment(list(have), attack_cost(a))[0]
         for a in card_attack_ids(ref_cid)), default=DEFICIT_CAP)
    return [
        1.0 if dmg > 0 else 0.0,
        float(min(short, DEFICIT_CAP)),
        float(dmg),
        1.0 if (dmg and def_hp and dmg >= def_hp) else 0.0,
    ] if costs else [0.0, 0.0, 0.0, 0.0]


def card_numerics(
    card: dict, owner_is_self: bool, slot: int, unseen: dict[int, int],
    defender_type=None, defender: dict | None = None, in_play: bool = False,
    hand_by_pre: dict[str, list[int]] | None = None,
    board_names: set[str] | None = None, status: tuple = (),
    ctx: dict | None = None, key: tuple | None = None,
) -> list[float]:
    """The card block, 269 slots:
        42 base + 21 effect tags + 119 attack readiness + 6 evolution
        + 5 status + 3 counts
        + 6 retreat + 5 checkup + 21 attached-card effects
        + 24 weakness/resistance types + 2 whole-board threat
        + 6 engine legality + 6 turn locks/playability + 3 prize decisiveness.

    `ctx` carries the once-per-decision facts a Pokemon token needs but cannot
    see from inside itself: the turn locks, the prize counts, the energy the
    owner can reach, the opposing board, and which options the engine is
    offering against this exact board slot. `key` is this token's
    (playerIndex, area, index) so the option index can be read.
    """
    cid = card.get("id")
    meta = CARDS.get(cid)
    hp, max_hp = card.get("hp") or 0, card.get("maxHp") or 0
    dmg = max(0, max_hp - hp)
    ev = energy_vector(card)
    flags = (
        [
            1.0 if getattr(meta, f, False) else 0.0
            for f in ("ex", "megaEx", "tera", "aceSpec", "basic")
        ]
        if meta
        else [0.0] * 5
    )
    return [
        hp,
        max_hp,
        dmg,
        (dmg / max_hp) if max_hp else 0.0,
        min(dmg // 10, 20),
        (meta.retreatCost if meta else 0),
        1.0 if card.get("appearThisTurn") else 0.0,
        1.0 if slot < 0 else 0.0,  # is_active
        # one binary fact per bench position, not a magnitude
        *[1.0 if (0 <= slot == i) else 0.0 for i in range(N_BENCH_SLOT)],
        *ev,
        sum(ev),
        1.0 if (card.get("tools") or []) else 0.0,
        len(card.get("preEvolution") or []),
        *flags,
        1.0 if owner_is_self else 0.0,
        unseen.get(cid, 0),
        # Weakness is x2 damage and decides most knockouts; the catalog carries
        # it and the block was dropping it entirely.
        1.0 if (meta and defender_type is not None
                and meta.weakness == defender_type) else 0.0,
        1.0 if (meta and defender_type is not None
                and meta.resistance == defender_type) else 0.0,
        # prizes this card yields when knocked out: megaEx 3, ex 2, else 1.
        prize_value(meta) if meta else 1.0,
        # remaining HP, i.e. damage still needed to knock it out. Present as hp
        # already, but the KO-relevant framing costs one slot and removes a
        # subtraction the policy would otherwise have to learn.
        float(hp),
        *effect_tag_vector(cid),
        # --- attack readiness (119) -----------------------------------------
        # 38.7% of decisions have a positive energy shortfall and had zero
        # representation of it; bench Pokemon had no attack features at all.
        *attack_profile(
            card, defender, (ctx or {}).get("supply") if owner_is_self else None
        ),
        # --- evolution (6) ---------------------------------------------------
        *evolution_profile(
            card, in_play, hand_by_pre or {}, board_names or set()
        ),
        # --- status, bound to the Pokemon rather than the player (5) ---------
        # PlayerState carries these five and the API defines them as facts about
        # the ACTIVE Pokemon ("Active Pokémon is Poisoned"). Reading them off the
        # PLAYER token forced a cross-token join to answer "is the thing I am
        # deciding about asleep", which is the whole reason to retreat it.
        *[
            1.0 if (in_play and slot < 0 and s) else 0.0
            for s in (tuple(status) + (0,) * N_SPECIAL_CONDITION)[:N_SPECIAL_CONDITION]
        ],
        # --- counts the boolean/histogram forms lose (3) ---------------------
        # `tools` may hold 2 (Rotom ex Multi Adapter) and slot 29 is a boolean.
        float(len(card.get("tools") or [])),
        float(sum(1 for c in (card.get("energyCards") or [])
                  if isinstance(c, dict) and is_special_energy(c.get("id")))),
        # catalog HP: a HAND Pokemon carries no hp/maxHp (cg.api Card has only
        # id/serial/playerIndex), so slots 0 and 1 are 0.0 for every hand card
        # and "how big is the thing I am about to bench" was unanswerable.
        float(getattr(meta, "hp", 0) or 0),
        # --- retreat arithmetic (6) ------------------------------------------
        *retreat_profile(
            card, meta, status, bool((ctx or {}).get("retreat_used")),
            in_play and slot < 0,
        ),
        # --- checkup arithmetic (5) ------------------------------------------
        *checkup_profile(card, status, in_play and slot < 0),
        # --- what is attached, as effects rather than as a count (21) --------
        *attached_tag_vector(card),
        # --- weakness and resistance as TYPES (24) ---------------------------
        # WHY IT CHANGES A DECISION: the two booleans above answer only "is it
        # weak to the Pokemon standing opposite right now". A bench Pokemon is
        # benched for a future defender, and "this dies to Fire" is the reason
        # you keep it back. The catalog carries both types and neither reached
        # the model. Most columns here are rare by construction — an unusual
        # weakness type is a rare VALUE, not dead wiring.
        *[1.0 if (meta is not None and meta.weakness is not None
                  and int(meta.weakness) == t) else 0.0 for t in range(N_ENERGY)],
        *[1.0 if (meta is not None and meta.resistance is not None
                  and int(meta.resistance) == t) else 0.0 for t in range(N_ENERGY)],
        # --- the worst hit the whole opposing board can land on this (2) -----
        *(board_threat(card, (ctx or {}).get("their_board") or [])
          if in_play else [0.0, 0.0]),
        # --- which options the engine is offering against this slot (6) ------
        # WHY IT CHANGES A DECISION: legality is decided by the engine and the
        # model was left to re-derive it from board state. The option list is
        # right there in the same observation; binding it back onto the board
        # token says "this Pokemon can use its ability RIGHT NOW", which is the
        # once-per-turn fact cg.api never exposes as a field on the Pokemon.
        *option_presence((ctx or {}).get("opt_index") or {}, key),
        # --- the once-per-turn locks, on the token they constrain (6) --------
        # These four live on GLOBAL, one token away, so "this Supporter in my
        # hand is dead until next turn" was a join instead of a fact.
        1.0 if (ctx or {}).get("supporter_played") else 0.0,
        1.0 if (ctx or {}).get("stadium_played") else 0.0,
        1.0 if (ctx or {}).get("energy_attached") else 0.0,
        1.0 if (ctx or {}).get("retreat_used") else 0.0,
        *playability(meta, ctx, in_play, owner_is_self),
        # --- does this Pokemon decide the game (3) ---------------------------
        # prize_value above says what the knockout is worth; whether that value
        # is the LAST prize the taker needs is the difference between a good
        # trade and the end of the match.
        float(len(getattr(meta, "skills", None) or [])),
        # prizes the player who would take this knockout still needs: if the
        # card is mine the taker is my opponent, and vice versa.
        float(_taker_prizes(ctx, owner_is_self)),
        1.0 if (meta is not None and in_play
                and 0 < _taker_prizes(ctx, owner_is_self) <= prize_value(meta)
                ) else 0.0,                       # this knockout ends the game
    ]


def _taker_prizes(ctx: dict | None, owner_is_self: bool) -> float:
    """Prizes still owed to whoever knocks this Pokemon out."""
    ctx = ctx or {}
    return float(ctx.get("opp_prizes" if owner_is_self else "my_prizes") or 0)


def option_presence(index: dict, key: tuple | None) -> list[float]:
    """6 slots: the engine's own legality, bound to the board slot it concerns."""
    entry = index.get(key) if key is not None else None
    if not entry:
        return [0.0] * 6
    return [
        1.0 if entry.get("attack") else 0.0,
        1.0 if entry.get("ability") else 0.0,
        1.0 if entry.get("attach_dest") else 0.0,
        1.0 if entry.get("evolve_dest") else 0.0,
        1.0 if entry.get("total") else 0.0,
        float(min(entry.get("total", 0), 16)),
    ]


def playability(meta, ctx: dict | None, in_play: bool, mine: bool) -> list[float]:
    """2 slots: can this hand card be played AT ALL this turn, and is a
    once-per-turn lock the only thing stopping it.

    WHY IT CHANGES A DECISION: a hand full of Supporters after the Supporter has
    been used is a hand with no plays in it, and the model had to reach the
    GLOBAL token and the card-type table to notice.
    """
    if meta is None or in_play or not mine:
        return [0.0, 0.0]
    ctx = ctx or {}
    ct = int(meta.cardType)
    locked = (
        (ct == int(CardType.SUPPORTER) and bool(ctx.get("supporter_played")))
        or (ct == int(CardType.STADIUM) and bool(ctx.get("stadium_played")))
        or (ct in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY))
            and bool(ctx.get("energy_attached")))
        or (ct == int(CardType.POKEMON) and getattr(meta, "basic", False)
            and not ctx.get("bench_open"))
    )
    return [0.0 if locked else 1.0, 1.0 if locked else 0.0]


def attack_features(attack_id: int, attacker: dict | None) -> list[float]:
    """18 per-attack features; affordability computed, never learned."""
    atk = ATTACKS.get(attack_id)
    cost = [0] * N_ENERGY
    if atk:
        for e in atk.energies or []:
            if isinstance(e, int) and 0 <= e < N_ENERGY:
                cost[e] += 1
    have = energy_vector(attacker) if attacker else [0] * N_ENERGY
    colorless = int(EnergyType.COLORLESS)
    need = sum(cost)
    generic = cost[colorless]
    # RAINBOW and TEAM_ROCKET energy pay typed costs; only energy_payment knows that.
    affordable = energy_payment(have, cost)[0] == 0
    damage = 0
    if atk:
        try:
            damage = int(str(atk.damage).strip() or 0)
        except ValueError:
            damage = 0
    return [
        *cost,
        need,
        generic,
        damage,
        1.0 if affordable else 0.0,
        sum(have),
        1.0 if atk else 0.0,
    ]


def energy_supply_of(cards) -> list[int]:
    """Per-type count of the ENERGY CARDS in a zone (hand, or the unseen pool).

    An energy card's type is the catalog's energyType, so this is exact for
    basics and for the special energies whose catalog type says what they pay.
    """
    counts = [0] * N_ENERGY
    for item in cards or []:
        cid = item.get("id") if isinstance(item, dict) else item
        meta = CARDS.get(cid)
        if meta is None or int(meta.cardType) not in (
            int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)
        ):
            continue
        et = meta.energyType
        if et is not None and 0 <= int(et) < N_ENERGY:
            counts[int(et)] += 1
    return counts


def kos_needed(board: list, prizes: int) -> float:
    """Fewest knockouts on the CURRENT board that take the remaining prizes.

    Exact for the board as it stands (ex is 2, megaEx is 3), and a lower bound
    once the board runs out — every further knockout is worth at least one
    prize. This is the number that decides whether to trade: "they need two
    knockouts, I need one" is the whole game and it was nowhere in the tensor.
    """
    vals = sorted(
        (prize_value(CARDS.get(c.get("id")))
         for c in board if isinstance(c, dict)), reverse=True)
    need, n = float(prizes), 0
    for v in vals:
        if need <= 0:
            break
        need -= v
        n += 1
    if need > 0:
        n += int(math.ceil(need))
    return float(min(n, 12))


def draw_odds(good: int, pool: int, k: int) -> float:
    """P(at least one of `good` cards in the next `k` draws from `pool`).

    Exact by exchangeability: the unseen pool is deck+prize and any of its
    orderings is equally likely, so the hypergeometric answer is the real
    probability, not a model of one.
    """
    if pool <= 0 or good <= 0 or k <= 0:
        return 0.0
    p_none = 1.0
    for i in range(min(k, pool)):
        bad = pool - good - i
        if bad <= 0:
            return 1.0
        p_none *= bad / (pool - i)
    return 1.0 - p_none


def build_tokens(
    obs: dict,
    seat: int,
    deck: Counter,
    intra_turn: list[dict],
    actions_since_chance: int = 0,
) -> dict:
    cur, sel = obs["current"], obs["select"]
    assert cur.get("result") == -1, f"result leak canary fired: {cur.get('result')}"
    state = track(cur, seat, deck, sel)
    unseen = state["unseen"]
    me = (cur.get("players") or [])[seat]
    opp = (cur.get("players") or [])[1 - seat]

    fam, owner, cards, attacks, nums = [], [], [], [], []
    ctypes, etypes = [], []

    def push(
        family: str,
        own: int,
        cid: int | None,
        numeric: list[float],
        attack_id: int | None = None,
    ) -> int:
        if len(numeric) > NUMERIC_WIDTH:
            raise ValueError(
                f"{family} numeric width {len(numeric)} exceeds {NUMERIC_WIDTH}"
            )
        if BLOCK_WIDTH_SINK is not None:
            BLOCK_WIDTH_SINK[family] = max(
                # DEFAULT-OK: max() accumulator seed, not a data lookup. An
                # unseen family has no recorded width yet, and max(0, n) == n.
                BLOCK_WIDTH_SINK.get(family, 0), len(numeric)
            )
        fam.append(FAM[family])
        owner.append(own)
        cards.append(card_tok(cid))
        attacks.append(attack_tok(attack_id))
        # +1 so index 0 stays "this token carries no card" (PAD, GLOBAL, PROMPT...)
        meta_ct = CARDS.get(cid) if cid is not None else None
        ctypes.append(int(meta_ct.cardType) + 1 if meta_ct is not None else CT_NONE)
        etypes.append(int(meta_ct.energyType) + 1 if meta_ct is not None else ET_NONE)
        nums.append([*numeric, *([0.0] * (NUMERIC_WIDTH - len(numeric)))])
        return len(fam) - 1

    # index from (playerIndex, area, index) -> token, so options can point
    referent: dict[tuple[int, int, int], int] = {}
    referent_card: dict[tuple[int, int, int], int | None] = {}
    # the card OBJECT, not just its id: an option's numeric block needs the
    # referent's hp, damage and attached energies, and an id cannot supply them
    referent_obj: dict[tuple[int, int, int], dict | None] = {}
    serial_referent: dict[int, int] = {}

    def register(key: tuple[int, int, int], token: int, card: dict | None) -> None:
        referent[key] = token
        referent_card[key] = card.get("id") if isinstance(card, dict) else None
        referent_obj[key] = card if isinstance(card, dict) else None
        if isinstance(card, dict) and isinstance(card.get("serial"), int):
            serial_referent[card["serial"]] = token

    # weakness/resistance are only meaningful against a specific defender, so
    # each side is scored against the OPPOSING active's energy type. Without
    # this the two features were computed but always 0.
    def _active_card(p):
        for c in (p.get("active") or [])[:1]:
            if isinstance(c, dict):
                return c
        return None

    def _active_type(p):
        c = _active_card(p)
        meta = CARDS.get(c.get("id")) if isinstance(c, dict) else None
        return meta.energyType if meta is not None else None

    my_type, opp_type = _active_type(me), _active_type(opp)
    my_active, opp_active = _active_card(me), _active_card(opp)

    # --- the prize race, arithmetic done (5 new slots on GLOBAL) -------------
    # WHY IT CHANGES A DECISION: both prize counts were emitted and their
    # DIFFERENCE was not, and "how many knockouts does each side still need"
    # — which depends on the ex/megaEx values sitting on the board, not on the
    # counts — was not computed anywhere. That number decides whether you take
    # a trade or refuse it.
    my_prizes = len(me.get("prize") or [])
    opp_prizes = len(opp.get("prize") or [])
    my_board_cards = [
        c for c in [*(me.get("active") or []), *(me.get("bench") or [])]
        if isinstance(c, dict)
    ]
    opp_board_cards = [
        c for c in [*(opp.get("active") or []), *(opp.get("bench") or [])]
        if isinstance(c, dict)
    ]
    # I take my prizes by knocking out THEIR Pokemon, so my race is measured
    # against their board.
    my_kos = kos_needed(opp_board_cards, my_prizes)
    opp_kos = kos_needed(my_board_cards, opp_prizes)

    option_counts = Counter(require_option_type(act) for act in intra_turn)
    push(
        "GLOBAL",
        2,
        None,
        [
            cur.get("turn") or 0,
            cur.get("turnActionCount") or 0,
            cur.get("firstPlayer") or 0,
            seat,
            1.0 if cur.get("retreated") else 0.0,
            1.0 if cur.get("stadiumPlayed") else 0.0,
            1.0 if cur.get("supporterPlayed") else 0.0,
            1.0 if cur.get("energyAttached") else 0.0,
            len(cur.get("looking") or []),
            len(me.get("prize") or []),
            len(opp.get("prize") or []),
            len(intra_turn),
            actions_since_chance,
            *[option_counts[i] for i in range(len(OptionType))],
            # --- prize race (5) ----------------------------------------------
            float(opp_prizes - my_prizes),          # + means I am ahead
            my_kos,
            opp_kos,
            float(opp_kos - my_kos),                # + means I am winning the race
            1.0 if my_kos <= 1 else 0.0,            # one knockout from winning
        ],
    )
    for who, player in ((1, me), (0, opp)):
        _board = my_board_cards if who == 1 else opp_board_cards
        # what this player's WHOLE board can do to the Pokemon opposite it: the
        # "what if they switch" number, and the "do I have a better attacker on
        # the bench" number. Both boards and both energy sets are visible, so
        # every entry is exact.
        _target = opp_active if who == 1 else my_active
        _target_hp = float((_target or {}).get("hp") or 0)
        _dmg = [
            _best_affordable_damage(
                c.get("id"), tuple(energy_vector(c)), (_target or {}).get("id"))
            for c in _board
        ]
        _hand = player.get("hand")
        _hand_visible = isinstance(_hand, list)
        _hand_cards = [c for c in (_hand or []) if isinstance(c, dict)]
        _hand_meta = [CARDS.get(c.get("id")) for c in _hand_cards]
        _hand_by_type = [0.0] * N_CARD_TYPE_ONEHOT
        for _m in _hand_meta:
            if _m is not None and 0 <= int(_m.cardType) < N_CARD_TYPE_ONEHOT:
                _hand_by_type[int(_m.cardType)] += 1
        _hand_energy = energy_supply_of(_hand_cards)
        # Ability holders on THIS player's board. `skills` is the pool's Ability
        # field; a Pokemon with no skills cannot be hit by a symmetric
        # ability-targeting effect and must not be counted.
        _ability_holders = sum(
            1 for c in _board
            if getattr(CARDS.get(c.get("id")), "skills", None)
        )
        # The board OPPOSITE this player. Both boards are public zones -- the
        # existing comment above states "Both boards and both energy sets are
        # visible, so every entry is exact" -- so this is not redaction-bounded
        # the way a hand is. The flag is kept as an explicit emptiness marker so
        # "no board" and "a board with no Ability holders" stay distinguishable.
        _opp_board_raw = opp_board_cards if who == 1 else my_board_cards
        _opp_board_visible = bool(_opp_board_raw)
        _opp_ability_holders = sum(
            1 for c in _opp_board_raw
            if getattr(CARDS.get(c.get("id")), "skills", None)
        )
        _act = my_active if who == 1 else opp_active
        _board_checkup = [
            checkup_profile(
                c,
                [1.0 if c.get(s) else 0.0 for s in
                 ("asleep", "burned", "confused", "paralyzed", "poisoned")],
                c is _act,
            )
            for c in _board
        ]
        _board_checkup_damage = sum(p[0] for p in _board_checkup)
        _board_checkup_deaths = sum(p[2] for p in _board_checkup)
        # A symmetric per-checkup source anywhere in play taxes EVERY Ability
        # holder on BOTH boards, so it is counted across both sides. Identified
        # by tag, not by card id: "checkup_effect AND affects_both_players" is
        # the class, and the class is what survives a new card pool.
        _symmetric_sources = sum(
            1 for c in (my_board_cards + opp_board_cards)
            if {"checkup_effect", "affects_both_players"}
            <= set(CARD_TAGS.get(c.get("id")) or ())
        )
        _own_exposure = float(_ability_holders * _symmetric_sources * DAMAGE_COUNTER)
        _marginal_bench_cost = float(_symmetric_sources * DAMAGE_COUNTER)
        push(
            "PLAYER",
            who,
            None,
            [
                player.get("deckCount") or 0,
                player.get("handCount") or 0,
                player.get("benchMax") or 0,
                len(player.get("bench") or []),
                len(player.get("prize") or []),
                *[
                    1.0 if player.get(c) else 0.0
                    for c in ("asleep", "burned", "confused", "paralyzed", "poisoned")
                ],
                # --- board aggregates (7) ------------------------------------
                # benchMax and len(bench) were both present and their DIFFERENCE
                # was not, so "do I have room to bench this" was a subtraction
                # inside one token. The rest are the prize-race denominators:
                # how much material is on the table and how much of it is worth
                # two prizes.
                float(max(0, (player.get("benchMax") or 0)
                          - len(player.get("bench") or []))),   # bench_open
                float(len(_board)),
                float(sum(sum(energy_vector(c)) for c in _board)),
                float(sum(c.get("hp") or 0 for c in _board)),
                float(sum(c.get("maxHp") or 0 for c in _board)),
                float(sum(1 for c in _board
                          if getattr(CARDS.get(c.get("id")), "ex", False))),
                float(sum(1 for c in _board if c.get("appearThisTurn"))),
                # --- board offence, computed (5) -----------------------------
                # "my best attacker is on the bench" is a switch decision and it
                # required comparing every board token against every other.
                float(max(_dmg) if _dmg else 0.0),        # best damage on board
                float(sum(1 for d in _dmg if d > 0)),     # who can attack at all
                float(sum(1 for d in _dmg
                          if d and _target_hp and d >= _target_hp)),  # who KOs
                float(sum(prize_value(CARDS.get(c.get("id"))) for c in _board)),
                float(max((prize_value(CARDS.get(c.get("id"))) for c in _board),
                          default=0.0)),
                # --- prize race, this player's side (3) ----------------------
                float(my_prizes if who == 1 else opp_prizes),
                float(my_kos if who == 1 else opp_kos),
                float((opp_kos - my_kos) if who == 1 else (my_kos - opp_kos)),
                # --- hand composition (21) -----------------------------------
                # WHY IT CHANGES A DECISION: handCount was emitted and its
                # CONTENTS were only reachable as up to 24 separate tokens, so
                # "do I still hold an energy to attach" was an attention sum.
                # The opponent's hand is None in the API, so it is emitted as a
                # visibility flag plus zeros rather than as a fake zero count.
                1.0 if _hand_visible else 0.0,
                float(len(_hand_cards)),
                *_hand_by_type,
                float(sum(_hand_energy)),
                *[float(v) for v in _hand_energy],
                # --- symmetric-effect exposure (6) ---------------------------
                # own_ability_holders is the quantity the teachers demonstrably
                # act on (decline-to-bench 1.8/4.7/13.6/38.3% by this count,
                # n=2,318). checkup_profile models poison only, so a board-wide
                # per-checkup effect was invisible. Counted as a CLASS -- any
                # Ability holder -- not as a named card, so it survives a new
                # card pool.
                float(_ability_holders),
                float(_opp_ability_holders),
                1.0 if _opp_board_visible else 0.0,
                # what one MORE benched Ability holder would add to my standing
                # exposure: the counterfactual behind the decline, stated rather
                # than left to be inferred from a count the model must compare
                # against the opponent's board.
                # how many symmetric per-checkup sources are on the table
                float(_symmetric_sources),
                # my standing bleed per checkup, and what one more benched
                # Ability holder would add to it. Zero when no source is in play,
                # which is why this is not a restatement of the holder count.
                _own_exposure,
                _marginal_bench_cost,
            ],
        )

    # name -> hand card ids that evolve from it, so "this Pokemon can evolve
    # RIGHT NOW" becomes one flat feature instead of a 24-way attention join.
    hand_by_pre: dict[str, list[int]] = {}
    for _c in me.get("hand") or []:
        _m = CARDS.get(_c.get("id")) if isinstance(_c, dict) else None
        _pre = getattr(_m, "evolvesFrom", None)
        if _pre:
            hand_by_pre.setdefault(_pre, []).append(_m.cardId)
    my_board_names = {
        n for n in (
            getattr(CARDS.get(c.get("id")), "name", None)
            for c in [*(me.get("active") or []), *(me.get("bench") or [])]
            if isinstance(c, dict)
        ) if n
    }
    STATUS_FIELDS = ("asleep", "burned", "confused", "paralyzed", "poisoned")
    my_status = tuple(1 if me.get(f) else 0 for f in STATUS_FIELDS)
    opp_status = tuple(1 if opp.get(f) else 0 for f in STATUS_FIELDS)

    # --- the engine's own legality, indexed by the board slot it concerns ----
    # WHY IT CHANGES A DECISION: cg.api has no "ability already used" field on a
    # Pokemon, so once-per-turn ability state is not directly observable — but
    # the engine only offers LEGAL options, so the presence of an ABILITY option
    # against this slot IS the answer, exactly, and it was being thrown away
    # after the option row was built.
    opt_index: dict[tuple, dict] = {}

    def _mark(k: tuple, field: str) -> None:
        if k[1] is None or k[2] is None:
            return
        entry = opt_index.setdefault(k, {"total": 0})
        entry[field] = entry.get(field, 0) + 1
        entry["total"] += 1

    for _o in sel.get("option") or []:
        _k = int(_o.get("type") or -1)
        _p = _o.get("playerIndex", seat)
        if _k == int(OptionType.ATTACK):
            _mark((seat, int(AreaType.ACTIVE), 0), "attack")
        elif _k == int(OptionType.PLAY):
            _mark((seat, int(AreaType.HAND), _o.get("index")), "play")
        elif _k == int(OptionType.ABILITY):
            _mark((_p, _o.get("area"), _o.get("index")), "ability")
        else:
            _mark((_p, _o.get("area"), _o.get("index")), "other")
        if _k in (int(OptionType.ATTACH), int(OptionType.EVOLVE)):
            _mark(
                (_p, _o.get("inPlayArea"), _o.get("inPlayIndex")),
                "attach_dest" if _k == int(OptionType.ATTACH) else "evolve_dest",
            )

    # energy I can actually reach, so a shortfall can be answered rather than
    # only stated: the hand is fully visible and the unseen pool is exact.
    unseen_energy = [0] * N_ENERGY
    for _cid, _n in unseen.items():
        _m = CARDS.get(_cid)
        if _m is None or int(_m.cardType) not in (
            int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)
        ):
            continue
        if _m.energyType is not None and 0 <= int(_m.energyType) < N_ENERGY:
            unseen_energy[int(_m.energyType)] += _n
    board_ctx = {
        "supply": {
            "hand_energy": energy_supply_of(me.get("hand") or []),
            "unseen_energy": unseen_energy,
        },
        "opt_index": opt_index,
        "supporter_played": bool(cur.get("supporterPlayed")),
        "stadium_played": bool(cur.get("stadiumPlayed")),
        "energy_attached": bool(cur.get("energyAttached")),
        "retreat_used": bool(cur.get("retreated")),
        "bench_open": max(0, (me.get("benchMax") or 0) - len(me.get("bench") or [])),
        "my_prizes": my_prizes,
        "opp_prizes": opp_prizes,
    }

    def push_attached(card: dict, own: int, slot: int) -> None:
        """Identity of each Tool and special Energy, bound to its holder.

        The holder is identified the same way a BENCH token identifies itself —
        owner plus slot plus the holder's card token — so attention can bind the
        pair without needing board-to-board edges, which stay disabled.
        """
        holder = card_tok(card.get("id"))
        for attached, kind in (
            ((card.get("tools") or []), "tool"),
            ((card.get("energyCards") or []), "energy"),
        ):
            for item in attached:
                cid = item.get("id") if isinstance(item, dict) else item
                if not isinstance(cid, int):
                    continue
                if kind == "energy" and not is_special_energy(cid):
                    continue
                push("ATTACHED", own, cid,
                     [1.0 if kind == "tool" else 0.0,
                      1.0 if kind == "energy" else 0.0,
                      1.0 if slot < 0 else 0.0,
                      max(slot, 0),
                      holder])

    for pidx, player in ((seat, me), (1 - seat, opp)):
        own = 1 if pidx == seat else 0
        defender = opp_type if own == 1 else my_type
        # the Pokemon on the other side of the board, for the KO arithmetic
        def_card = opp_active if own == 1 else my_active
        status = my_status if own == 1 else opp_status
        # evolution targets are only readable in MY hand (opponent hand is None)
        pre = hand_by_pre if own == 1 else {}
        # the board on the far side of THIS Pokemon, for the "what if they
        # switch" threat number
        side_ctx = {
            **board_ctx,
            "their_board": opp_board_cards if own == 1 else my_board_cards,
        }
        for card in (player.get("active") or [])[:1]:  # LIST of <=1
            if isinstance(card, dict):
                key = (pidx, int(AreaType.ACTIVE), 0)
                t = push(
                    "ACTIVE", own, card["id"],
                    card_numerics(card, own == 1, -1, unseen, defender,
                                  defender=def_card, in_play=True,
                                  hand_by_pre=pre, board_names=my_board_names,
                                  status=status, ctx=side_ctx, key=key)
                )
                register(key, t, card)
                push_attached(card, own, -1)
        for i, card in enumerate((player.get("bench") or [])[:8]):
            if isinstance(card, dict):
                push_attached(card, own, i)
                key = (pidx, int(AreaType.BENCH), i)
                t = push(
                    "BENCH", own, card["id"],
                    card_numerics(card, own == 1, i, unseen, defender,
                                  defender=def_card, in_play=True,
                                  hand_by_pre=pre, board_names=my_board_names,
                                  status=status, ctx=side_ctx, key=key)
                )
                register(key, t, card)

    hand_ctx = {**board_ctx, "their_board": opp_board_cards}
    for i, card in enumerate((me.get("hand") or [])[:24]):
        if isinstance(card, dict):
            key = (seat, int(AreaType.HAND), i)
            t = push("HAND", 1, card["id"],
                     card_numerics(card, True, i, unseen, opp_type,
                                   defender=opp_active, in_play=False,
                                   hand_by_pre=hand_by_pre,
                                   board_names=my_board_names,
                                   ctx=hand_ctx, key=key))
            register(key, t, card)
    for i, card in enumerate((cur.get("looking") or [])[:8]):
        if isinstance(card, dict):
            pidx = card.get("playerIndex", seat)
            t = push(
                "LOOKING",
                1 if pidx == seat else 0,
                card["id"],
                [0.0],
            )
            register((pidx, int(AreaType.LOOKING), i), t, card)
    stadium_token = -1
    for card in (cur.get("stadium") or [])[:1]:  # LIST of <=1
        if isinstance(card, dict):
            stadium_token = push(
                "STADIUM",
                1 if card.get("playerIndex") == seat else 0,
                card["id"],
                [0.0],
            )
            register(
                (card.get("playerIndex", seat), int(AreaType.STADIUM), 0),
                stadium_token,
                card,
            )
            # Stadium abilities are available to either player; ABILITY options
            # omit playerIndex, so both actor-relative keys point to this token.
            register((seat, int(AreaType.STADIUM), 0), stadium_token, card)
            register((1 - seat, int(AreaType.STADIUM), 0), stadium_token, card)
    if stadium_token < 0:
        stadium_token = push("STADIUM", 2, None, [0.0])

    # --- what is still in the deck, BY KIND (7 + 2) --------------------------
    # WHY IT CHANGES A DECISION: "can I keep attaching for the next three turns"
    # is governed by how many energy cards are left, and that aggregate existed
    # only as one UNSEEN_CARD token per distinct card — an attention sum the
    # model had to learn. Each count is exact: the tracker's unseen multiset
    # keyed by the catalog's cardType.
    _by_type = [0.0] * N_CARD_TYPE_ONEHOT
    _unseen_basics = 0
    for _cid, _n in unseen.items():
        _m = CARDS.get(_cid)
        if _m is None:
            continue
        _t = int(_m.cardType)
        if 0 <= _t < N_CARD_TYPE_ONEHOT:
            _by_type[_t] += _n
        if getattr(_m, "basic", False):
            _unseen_basics += _n
    _pool = float(state["deck_size"]) + float(state["prize_size"])
    _energy_left = _by_type[int(CardType.BASIC_ENERGY)] + _by_type[int(CardType.SPECIAL_ENERGY)]
    unseen_token = push(
        "UNSEEN_POOL",
        1,
        None,
        [
            state["deck_size"], state["prize_size"], sum(unseen.values()),
            *_by_type,
            float(_unseen_basics),
            # exact fraction of the unseen pool that is energy: the draw odds
            _energy_left / _pool if _pool else 0.0,
            # --- draw odds over a real horizon (3) ---------------------------
            # WHY IT CHANGES A DECISION: one turn is one draw, so "will I have
            # an energy in time" is a hypergeometric over the next few draws,
            # not a ratio. Exact by exchangeability of the unseen pool.
            *[draw_odds(int(_energy_left), int(_pool), k)
              for k in range(1, DRAW_LOOKAHEAD + 1)],
            # --- what is left, by kind and by energy type (14) ---------------
            # UNSEEN_CARD tokens truncate at 24 distinct cards, so the counts
            # themselves have to be stated numerically or the tail is lost.
            float(_by_type[int(CardType.SUPPORTER)]
                  + _by_type[int(CardType.ITEM)]),      # outs that draw/search
            float(len([1 for _n in unseen.values() if _n > 0])),  # distinct left
            *[float(v) for v in unseen_energy],
        ],
    )
    # deterministic order: commonest first, card id breaking ties, so the same
    # state always tokenises identically
    for _cid, _n in sorted(unseen.items(), key=lambda kv: (-kv[1], kv[0]))[:UNSEEN_CARD_CAP]:
        if _n > 0:
            push("UNSEEN_CARD", 1, _cid, [float(_n)])
    for i, ref in enumerate(sel.get("deck") or []):
        if isinstance(ref, dict):
            pidx = ref.get("playerIndex", seat)
            register((pidx, int(AreaType.DECK), i), unseen_token, ref)
            register((seat, int(AreaType.DECK), i), unseen_token, ref)

    for pidx, who, player in ((seat, 1, me), (1 - seat, 0, opp)):
        # Exact composition, not a scalar: "3 Munkidori discarded" and "3 Poffin
        # discarded" were previously the identical input [3].
        dcount = Counter(
            c["id"] for c in (player.get("discard") or []) if isinstance(c, dict)
        )
        # --- discard by kind (9) ---------------------------------------------
        # WHY IT CHANGES A DECISION: the per-card counts above are keyed to MY
        # 60-card list, so the opponent's discard — the only public record of
        # what their deck is doing — collapsed to a single length. Energy in the
        # discard is also the resource every recovery card plays for.
        _dis_by_type = [0.0] * N_CARD_TYPE_ONEHOT
        for _c in (player.get("discard") or []):
            _m = CARDS.get(_c.get("id")) if isinstance(_c, dict) else None
            if _m is not None and 0 <= int(_m.cardType) < N_CARD_TYPE_ONEHOT:
                _dis_by_type[int(_m.cardType)] += 1
        discard_token = push(
            "DISCARD_SUM", who, None,
            [len(player.get("discard") or []),
             *[dcount.get(cid, 0) for cid in sorted(deck)],
             *_dis_by_type,
             _dis_by_type[int(CardType.BASIC_ENERGY)]
             + _dis_by_type[int(CardType.SPECIAL_ENERGY)],
             float(sum(1 for _c in (player.get("discard") or [])
                       if getattr(CARDS.get(
                           _c.get("id") if isinstance(_c, dict) else None),
                           "ex", False))),
             ],
        )
        for i, card in enumerate(player.get("discard") or []):
            register(
                (pidx, int(AreaType.DISCARD), i),
                discard_token,
                card if isinstance(card, dict) else None,
            )
        prize_token = push("PRIZE", who, None, [len(player.get("prize") or [])])
        for i, card in enumerate(player.get("prize") or []):
            register(
                (pidx, int(AreaType.PRIZE), i),
                prize_token,
                card if isinstance(card, dict) else None,
            )

    stats: dict[str, int] = {}
    revealed: dict[int, dict] = {}
    for card in [
        *(opp.get("active") or []),
        *(opp.get("bench") or []),
        *(opp.get("discard") or []),
        *(cur.get("stadium") or []),
        *(cur.get("looking") or []),
    ]:
        if (
            isinstance(card, dict)
            and card.get("playerIndex") == 1 - seat
            and isinstance(card.get("serial"), int)
        ):
            revealed[card["serial"]] = card
    revealed_list = list(revealed.values())
    stats["opp_revealed_truncated"] = max(0, len(revealed_list) - 48)
    for card in revealed_list[:48]:
        push("OPP_REVEALED", 0, card.get("id"),
             [float(card.get("hp") or 0), float(card.get("maxHp") or 0),
              1.0 if card.get("energyCards") else 0.0])

    effect = sel.get("effect")
    context_card = sel.get("contextCard")
    effect_cid = effect.get("id") if isinstance(effect, dict) else None
    context_cid = context_card.get("id") if isinstance(context_card, dict) else None
    push(
        "PROMPT",
        1,
        # the EFFECT card is the one that caused this prompt, so the prompt now
        # inherits its semantics: an ability that says "search your deck" is the
        # explanation of the choice being asked for
        effect_cid,
        [
            int(sel.get("type") or 0),
            ctx_tok(sel.get("context")),
            sel.get("minCount") or 0,
            sel.get("maxCount") or 0,
            sel.get("remainDamageCounter") or 0,
            sel.get("remainEnergyCost") or 0,
            len(sel.get("deck") or []),
            # the same two fields as ONE-HOT. The scalars above are kept so a
            # checkpoint trained before this patch is unaffected: its learned
            # columns keep their meaning and these are new columns that were
            # zero. Removing the scalars would have shifted every later column
            # and silently invalidated the warm start.
            *ctx_onehot(sel.get("context")),
            *seltype_onehot(sel.get("type")),
        ],
    )
    if context_cid is not None and context_cid != effect_cid:
        push("CONTEXT_CARD", 1, context_cid, [1.0])

    for j, act in enumerate(intra_turn[-40:]):
        push(
            "INTRA_TURN",
            1,
            act.get("card"),
            [
                require_option_type(act),
                card_tok(act.get("card")),
                j,
                act.get("actions_since_chance", 0),
            ],
        )

    logs = (obs.get("logs") or [])[-32:]
    for start in range(0, len(logs), 8):
        chunk = logs[start : start + 8]
        last = chunk[-1] if chunk else {}
        push(
            "HISTORY",
            2,
            None,
            [
                len(chunk),
                int(last.get("type") or 0) if isinstance(last, dict) else 0,
            ],
        )

    # ---- options: the action space, each linked to what it points at ----
    options = sel["option"]
    options_truncated = max(0, len(options) - OPTION_CAP)
    if options_truncated:
        # Never raise here: main.py catches exceptions and falls back to a random
        # legal move, so a raise degrades the agent to random play with no signal.
        # Deterministic truncation keeps the best-known options and is counted.
        options = options[:OPTION_CAP]
    opt_type, opt_ptr, opt_num, opt_positions = [], [], [], []
    unresolved_local = [0]
    opt_attack: list[int] = []
    unresolved_by_kind: Counter = Counter()
    valid_option = {int(v) for v in OptionType}
    valid_select = {int(v) for v in SelectType}
    valid_context = {int(v) for v in SelectContext}
    valid_area = {int(v) for v in AreaType}
    valid_special = {int(v) for v in SpecialConditionType}
    categorical_oov = int(sel.get("type") not in valid_select)
    if sel.get("context") is not None:
        categorical_oov += int(int(sel["context"]) not in valid_context)
    option_kind_counts: Counter = Counter()
    active = (me.get("active") or [None])[0]
    opt_ctx = {
        "active": active if isinstance(active, dict) else None,
        "defender": opp_active,
        "unseen": unseen,
        "retreated": bool(cur.get("retreated")),
        "supporter_played": bool(cur.get("supporterPlayed")),
        "stadium_played": bool(cur.get("stadiumPlayed")),
        "energy_attached": bool(cur.get("energyAttached")),
        "seat": seat,
        "my_status": my_status,
        "opp_status": opp_status,
        "my_prizes": my_prizes,
        "opp_prizes": opp_prizes,
        "my_kos": my_kos,
        "opp_kos": opp_kos,
        "bench_open": board_ctx["bench_open"],
        "hand_count": len(me.get("hand") or []),
    }
    for opt in options:
        kind = opt.get("type")
        kind_int = int(kind) if kind is not None else -1
        option_kind_counts[kind_int] += 1
        categorical_oov += int(kind_int not in valid_option)
        area, idx = opt.get("area"), opt.get("index")
        pidx = opt.get("playerIndex", seat)
        target = referent.get((pidx, area, idx), -1)
        source_cid = referent_card.get((pidx, area, idx))
        target_card = referent_obj.get((pidx, area, idx))
        if kind == int(OptionType.PLAY):
            target = referent.get((seat, int(AreaType.HAND), idx), -1)
            source_cid = referent_card.get((seat, int(AreaType.HAND), idx))
            target_card = referent_obj.get((seat, int(AreaType.HAND), idx))
        elif kind == int(OptionType.ATTACK):
            target = referent.get((seat, int(AreaType.ACTIVE), 0), -1)
            source_cid = active.get("id") if isinstance(active, dict) else None
            target_card = active if isinstance(active, dict) else None
        elif kind == int(OptionType.SKILL):
            target = serial_referent.get(opt.get("serial"), -1)
            source_cid = opt.get("cardId")
            target_card = None
        dest_key = (pidx, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        dest = referent.get(dest_key, -1)
        dest_card = referent_obj.get(dest_key)
        attack_id = opt.get("attackId")
        if kind in POINTING_TYPES and target < 0:
            unresolved_local[0] += 1
            unresolved_by_kind[f"{OptionType(kind).name}:source"] += 1
        if kind in {int(OptionType.ATTACH), int(OptionType.EVOLVE)} and dest < 0:
            unresolved_local[0] += 1
            unresolved_by_kind[f"{OptionType(kind).name}:destination"] += 1
        for field in ("area", "inPlayArea"):
            if opt.get(field) is not None:
                categorical_oov += int(int(opt[field]) not in valid_area)
        if opt.get("specialConditionType") is not None:
            categorical_oov += int(
                int(opt["specialConditionType"]) not in valid_special
            )
        categorical_oov += int(attack_id is not None and attack_id not in ATTACKS)
        numeric = [
            opt.get("number") or 0,
            opt.get("count") or 0,
            # presence flags + one-hot position: an index is a LABEL, not a
            # magnitude, and the raw form told the model index 3 > index 1
            1.0 if opt.get("energyIndex") is not None else 0.0,
            *[1.0 if opt.get("energyIndex") == i else 0.0 for i in range(N_ENERGY_SLOT)],
            1.0 if opt.get("toolIndex") is not None else 0.0,
            *[1.0 if opt.get("toolIndex") == i else 0.0 for i in range(N_TOOL_SLOT)],
            *(
                attack_features(attack_id, active)
                if attack_id is not None
                else [0.0] * 18
            ),
            # 138 slots of per-option-type meaning; see option_semantics().
            *option_semantics(
                opt, kind_int, source_cid, dest_card, target_card, opt_ctx
            ),
        ]
        opt_type.append(kind_int)
        opt_ptr.append([target, dest])
        opt_num.append([*numeric, *([0.0] * (NUMERIC_WIDTH - len(numeric)))])
        opt_positions.append(push("OPTION", 1, source_cid, numeric, attack_id))
        # per-option attack identity: two attacks with the same cost and
        # damage are otherwise indistinguishable to the model
        opt_attack.append(attack_tok(attack_id))

    actual_tokens = len(fam)
    if actual_tokens > CAP:
        raise ValueError(f"decision has {actual_tokens} tokens > {CAP}")
    categorical_oov += sum(1 for token in cards if token == OOV)
    categorical_oov += sum(1 for token in attacks if token == OOV)
    for ref in [sel.get("effect"), sel.get("contextCard"), *(sel.get("deck") or [])]:
        if isinstance(ref, dict):
            categorical_oov += int(card_tok(ref.get("id")) == OOV)
    for player in (me, opp):
        for card in [
            *(player.get("active") or []),
            *(player.get("bench") or []),
            *(player.get("hand") or []),
        ]:
            if isinstance(card, dict):
                categorical_oov += sum(
                    1
                    for energy in card.get("energies") or []
                    if not isinstance(energy, int) or not 0 <= energy < N_ENERGY
                )

    token_padding = CAP - actual_tokens
    fam.extend([FAM["PAD"]] * token_padding)
    owner.extend([2] * token_padding)
    cards.extend([PAD] * token_padding)
    attacks.extend([PAD] * token_padding)
    ctypes.extend([CT_NONE] * token_padding)
    etypes.extend([ET_NONE] * token_padding)
    nums.extend([[0.0] * NUMERIC_WIDTH for _ in range(token_padding)])
    opt_attack.extend([0] * (OPTION_CAP - len(opt_attack)))
    option_padding = OPTION_CAP - len(opt_type)
    opt_type.extend([-1] * option_padding)
    opt_ptr.extend([[-1, -1] for _ in range(option_padding)])
    opt_num.extend([[0.0] * NUMERIC_WIDTH for _ in range(option_padding)])
    return {
        "family": fam,
        "owner": owner,
        "card": cards,
        "attack": attacks,
        "card_type": ctypes,
        "energy_type": etypes,
        "numeric": nums,
        "option_type": opt_type,
        "option_ptr": opt_ptr,
        "option_attack": opt_attack,
        "option_numeric": opt_num,
        "option_positions": opt_positions,
        "attention_mask": [1] * actual_tokens + [0] * token_padding,
        "option_mask": [1] * len(options) + [0] * option_padding,
        "unresolved_pointers": unresolved_local[0],
        "unresolved_by_kind": unresolved_by_kind,
        "categorical_oov": categorical_oov,
        "option_kind_counts": option_kind_counts,
        "n_options": len(options),
        "n_tokens": actual_tokens,
        "unseen_total": sum(unseen.values()),
    }

