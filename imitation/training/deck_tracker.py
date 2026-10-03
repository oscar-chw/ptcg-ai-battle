#!/usr/bin/env python3
"""Exact per-zone accounting of our own 60 cards, and the unseen pool.

Our deck is a closed 60-card list, so every card is in exactly one zone at any
moment.  The zones we can see (hand, discard, active, bench, everything attached
to them, and cards an effect is showing us) are read straight from the
observation; whatever is left is the **unseen pool** — deck union prize, exact by
card id, with the split known only by size.

This is deliberately NOT a log-replay accumulator.  Card location is not
monotonic (measured: 961 HAND->DECK and 95 DISCARD->HAND moves in 40 games), so
subtracting "cards seen so far" from 60 drifts the moment Lillie's Determination
shuffles a hand back in.  Reading current zones each decision is immune to that
by construction.

The catch is that attachments are easy to miss: energy cards, tools and
pre-evolution cards are *inside* a Pokémon object, not in a zone list.  Miss them
and the pool silently over-counts.  Hence the invariant, checked on every single
decision:

    sum(all visible zones) + unseen == 60
"""
from __future__ import annotations

from collections import Counter

# Zones a card of ours can be in.  DECK and PRIZE are the hidden pair.
VISIBLE_ZONES = ("hand", "discard", "active", "bench", "attached", "looking",
                 "stadium")
# Not a zone: a card mid-resolution, tracked separately (see zone_counts).
IN_FLIGHT = "in_flight"


def card_ids(obj) -> list[int]:
    """Every card id inside one in-play Pokémon object, including attachments."""
    out = [obj["id"]]
    for key in ("energyCards", "tools", "preEvolution"):
        for sub in (obj.get(key) or []):
            if isinstance(sub, dict) and sub.get("id") is not None:
                out.append(sub["id"])
    return out


def zone_counts(cur: dict, seat: int, sel: dict | None = None) -> dict[str, Counter]:
    """Count our own cards by id, per visible zone."""
    zones = {z: Counter() for z in (*VISIBLE_ZONES, IN_FLIGHT)}
    player = (cur.get("players") or [])[seat]

    for card in (player.get("hand") or []):
        if isinstance(card, dict):
            zones["hand"][card["id"]] += 1
    for card in (player.get("discard") or []):
        if isinstance(card, dict):
            zones["discard"][card["id"]] += 1

    for card in (player.get("active") or []):
        if isinstance(card, dict):
            head, *attached = card_ids(card)
            zones["active"][head] += 1
            zones["attached"].update(attached)
    for card in (player.get("bench") or []):
        if isinstance(card, dict):
            head, *attached = card_ids(card)
            zones["bench"][head] += 1
            zones["attached"].update(attached)

    # cards an effect is currently showing us are ours only if we own them
    for card in (cur.get("looking") or []):
        if isinstance(card, dict) and card.get("playerIndex") == seat:
            zones["looking"][card["id"]] += 1
    for card in (cur.get("stadium") or []):
        if isinstance(card, dict) and card.get("playerIndex") == seat:
            zones["stadium"][card["id"]] += 1

    # A card mid-resolution — played from hand, effect executing, not yet in the
    # discard — lives in `select.effect` / `select.contextCard` and in no zone
    # list at all.  It is only unaccounted when it is NOT already on the board
    # (an ability's source Pokemon appears here too), so dedup by serial.
    if sel:
        seen_serials = set()
        for key in ("hand", "discard"):
            for card in (((cur.get("players") or [])[seat]).get(key) or []):
                if isinstance(card, dict):
                    seen_serials.add(card.get("serial"))
        for key in ("active", "bench"):
            for card in (((cur.get("players") or [])[seat]).get(key) or []):
                if isinstance(card, dict):
                    seen_serials.add(card.get("serial"))
                    for sub_key in ("energyCards", "tools", "preEvolution"):
                        for sub in (card.get(sub_key) or []):
                            if isinstance(sub, dict):
                                seen_serials.add(sub.get("serial"))
        for key in ("looking",):
            for card in (cur.get(key) or []):
                if isinstance(card, dict):
                    seen_serials.add(card.get("serial"))
        for card in (cur.get("stadium") or []):
            if isinstance(card, dict):
                seen_serials.add(card.get("serial"))
        # NOTE: deliberately NOT counted as a zone. A card mid-resolution is in
        # `select.effect`/`contextCard`, and whether the engine still counts it
        # in deckCount depends on the effect: a resolving Trainer has left every
        # zone, while an ATTACH_FROM energy is still inside deckCount. Rather
        # than model each case, `track()` reconciles against the engine's own
        # counts and attributes the residual to this card.
        for key in ("effect", "contextCard"):
            card = sel.get(key)
            if (isinstance(card, dict) and card.get("playerIndex") == seat
                    and card.get("serial") not in seen_serials):
                zones["in_flight"][card["id"]] += 1
                seen_serials.add(card.get("serial"))
    return zones


def track(cur: dict, seat: int, deck: Counter, sel: dict | None = None) -> dict:
    """Zone counts plus the exact unseen pool for one decision."""
    zones = zone_counts(cur, seat, sel)
    visible = Counter()
    for name, counter in zones.items():
        if name != IN_FLIGHT:
            visible.update(counter)
    unseen = Counter({cid: deck[cid] - visible.get(cid, 0) for cid in deck})
    unseen = Counter({k: v for k, v in unseen.items() if v > 0})
    player = (cur.get("players") or [])[seat]
    deck_size = player.get("deckCount") or 0
    prize_size = sum(1 for _ in (player.get("prize") or []))

    # Reconcile against the engine's authoritative pool size. Any residual must
    # be exactly the in-flight card, +-1; anything else is a real defect.
    authoritative = deck_size + prize_size
    residual = sum(unseen.values()) - authoritative
    in_flight = zones[IN_FLIGHT]
    reconciled = True
    adjusted = 0
    if residual:
        if abs(residual) == 1 and in_flight:
            cid = next(iter(in_flight))
            unseen[cid] = unseen.get(cid, 0) - residual
            if unseen[cid] <= 0:
                unseen.pop(cid, None)
            adjusted = residual
        else:
            reconciled = False
    return {
        "zones": {z: dict(c) for z, c in zones.items()},
        "visible_total": sum(visible.values()),
        "unseen": dict(unseen),
        "unseen_total": sum(unseen.values()),
        "deck_size": deck_size,
        "prize_size": prize_size,
        "over_count": {cid: -(deck[cid] - visible.get(cid, 0))
                       for cid in deck if visible.get(cid, 0) > deck[cid]},
        "in_flight": dict(in_flight),
        "residual": residual,
        "reconciled": reconciled,
        # a card moved out of the pool because the engine no longer counts it
        "adjusted": adjusted,
    }
