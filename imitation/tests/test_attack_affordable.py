"""attack_features' `affordable` must agree with energy_payment on RAINBOW / TEAM_ROCKET.

featurize needs the official engine's `cg` package at import, which this repository does not
ship, and it also reads an engine-derived effect_tags.json. The test imports it against a
minimal stand-in `cg.api` (enums and two catalog tables) and an empty tag file, then removes
the stand-ins again, so it needs neither the engine nor torch.
"""
import enum
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

TRAINING = str(Path(__file__).resolve().parents[1] / "training")


_TAGS_NAMES = ("affects_both_players", "bench_damage", "checkup_effect", "coin_flip",
               "damage_prevent", "damage_reduction", "discard", "draw_cards", "energy_accel",
               "extra_tools", "heal", "knock_out_effect", "move_damage_counters", "once_per_turn",
               "prevent_damage_counter_place", "prize_manipulation", "retreat_free",
               "rule_box_condition", "search_deck", "special_condition", "switch")
_TAGS = json.dumps({"card_tags": {"1": list(_TAGS_NAMES)}})


def _enum(name, members):
    return enum.IntEnum(name, members, start=0)


def _load_featurize():
    api = types.ModuleType("cg.api")
    api.EnergyType = _enum("EnergyType", "GRASS FIRE WATER LIGHTNING PSYCHIC FIGHTING DARKNESS "
                                         "METAL DRAGON COLORLESS RAINBOW TEAM_ROCKET")
    api.AreaType = _enum("AreaType", "ACTIVE BENCH DECK DISCARD HAND LOOKING PRIZE STADIUM")
    api.CardType = _enum("CardType", "POKEMON ITEM SUPPORTER STADIUM BASIC_ENERGY SPECIAL_ENERGY TOOL")
    api.LogType = _enum("LogType", "COIN DRAW DRAW_REVERSE SHUFFLE")
    api.OptionType = _enum("OptionType", "ABILITY ATTACH ATTACK CARD DISCARD ENERGY ENERGY_CARD "
                                         "EVOLVE NUMBER PLAY RETREAT SKILL TOOL_CARD")
    api.SelectContext = _enum("SelectContext", "NONE")
    api.SelectType = _enum("SelectType", "MAIN")
    api.SpecialConditionType = _enum("SpecialConditionType", "POISONED BURNED ASLEEP PARALYZED CONFUSED")
    E = api.EnergyType
    api.all_card_data = lambda: []
    api.all_attack = lambda: [
        types.SimpleNamespace(attackId=1, damage="30", energies=[int(E.FIRE)]),
        types.SimpleNamespace(attackId=2, damage="10", energies=[int(E.FIRE), int(E.COLORLESS)]),
        types.SimpleNamespace(attackId=3, damage="10", energies=[int(E.PSYCHIC)]),
    ]
    pkg = types.ModuleType("cg")
    pkg.api = api
    sys.modules["cg"], sys.modules["cg.api"] = pkg, api
    sys.path.insert(0, TRAINING)
    real_exists, real_read = Path.exists, Path.read_text
    is_tags = lambda p: p.name == "effect_tags.json"  # noqa: E731
    try:
        with mock.patch.object(Path, "exists", lambda p: True if is_tags(p) else real_exists(p)), \
             mock.patch.object(Path, "read_text",
                               lambda p, *a, **k: _TAGS if is_tags(p) else real_read(p, *a, **k)):
            import featurize
        return featurize
    finally:
        sys.path.remove(TRAINING)
        for name in ("cg", "cg.api", "featurize"):
            sys.modules.pop(name, None)


class AttackAffordableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = _load_featurize()

    def affordable(self, attack_id, held):
        attacker = {"energies": [e for e, n in held.items() for _ in range(n)]}
        # layout: cost (N_ENERGY), need, generic, damage, affordable flag
        return self.f.attack_features(attack_id, attacker)[self.f.N_ENERGY + 3]

    def test_rainbow_energy_pays_a_typed_cost(self):
        self.assertEqual(self.affordable(1, {self.f.RAINBOW: 1}), 1.0)

    def test_team_rocket_pays_psychic_but_not_fire(self):
        self.assertEqual(self.affordable(3, {self.f.TEAM_ROCKET: 1}), 1.0)
        self.assertEqual(self.affordable(1, {self.f.TEAM_ROCKET: 1}), 0.0)

    def test_plain_shortfalls_stay_unaffordable(self):
        water = int(self.f.EnergyType.WATER)
        self.assertEqual(self.affordable(1, {water: 2}), 0.0)
        self.assertEqual(self.affordable(2, {int(self.f.EnergyType.FIRE): 1}), 0.0)
        self.assertEqual(self.affordable(2, {int(self.f.EnergyType.FIRE): 1, water: 1}), 1.0)


if __name__ == "__main__":
    unittest.main()
