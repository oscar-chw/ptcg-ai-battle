"""deck_tracker: visible zones + unseen pool must always sum to the deck (SYNTHETIC boards).

The invariant the module states is `sum(visible zones) + unseen == deck size`. The
boards below are hand-made observation dicts with invented card ids; no engine data.
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from deck_tracker import card_ids, track  # noqa: E402

DECK = Counter({1: 10, 2: 4, 3: 4, 4: 2})  # 20 cards, ids invented


def card(cid, serial, owner=0):
    return {"id": cid, "serial": serial, "playerIndex": owner}


def pokemon(cid, serial, energy=(), tools=(), pre=()):
    return {"id": cid, "serial": serial,
            "energyCards": [card(*e) for e in energy],
            "tools": [card(*t) for t in tools],
            "preEvolution": [card(*p) for p in pre]}


def board(deck_count, prize_count, **mine):
    me = {"hand": [], "discard": [], "active": [], "bench": [],
          "deckCount": deck_count, "prize": [None] * prize_count}
    me.update(mine)
    return {"players": [me, {"hand": None, "discard": [], "active": [], "bench": [],
                             "deckCount": 20, "prize": [None] * 6}]}


class DeckTracker(unittest.TestCase):
    def test_card_ids_includes_every_attachment(self):
        mon = pokemon(2, 50, energy=[(1, 51)], tools=[(3, 52)], pre=[(4, 53)])
        self.assertEqual(sorted(card_ids(mon)), [1, 2, 3, 4])

    def test_unseen_pool_is_the_deck_minus_what_is_visible(self):
        cur = board(deck_count=10, prize_count=6,
                    hand=[card(1, 1), card(1, 2)],
                    active=[pokemon(2, 3, energy=[(1, 4)])])
        out = track(cur, 0, DECK)
        self.assertEqual(out["visible_total"], 4)          # 2 hand + active + its energy
        self.assertEqual(out["unseen_total"], 16)
        self.assertEqual(out["unseen"][1], 7)              # 10 - 2 in hand - 1 attached
        self.assertTrue(out["reconciled"])
        self.assertEqual(out["visible_total"] + out["unseen_total"], sum(DECK.values()))

    def test_attached_cards_are_not_missed(self):
        # The failure the module exists to prevent: an energy inside a Pokemon
        # object that the pool silently over-counts.
        cur = board(deck_count=14, prize_count=6, active=[pokemon(2, 1, energy=[(1, 2)])])
        self.assertEqual(track(cur, 0, DECK)["zones"]["attached"], {1: 1})

    def test_a_pool_that_disagrees_with_the_engine_is_not_reconciled(self):
        cur = board(deck_count=3, prize_count=6, hand=[card(1, 1)])  # engine says 9
        self.assertFalse(track(cur, 0, DECK)["reconciled"])

    def test_over_count_is_reported(self):
        cur = board(deck_count=0, prize_count=0, hand=[card(4, i) for i in range(3)])
        self.assertEqual(track(cur, 0, DECK)["over_count"], {4: 1})


if __name__ == "__main__":
    unittest.main()
