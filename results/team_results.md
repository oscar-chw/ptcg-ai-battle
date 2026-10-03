# Team repository results (extract)

Verbatim extract of the "Current results" section of the team's private repository
README (the section as it stood when the repository was last paused, 2026-08-09). Only
the relative links were turned into plain text, because the files they point to are not
in this repository. Scores here are validation top-1 and mid-competition ladder scores;
the final standing is in [final_standing.json](final_standing.json), and the two are not
the same quantity.

## Current results

Best submission on the ladder scored **790.8** (multi-task loss) against 699.8
for pure imitation and 631.1 for the previous agent. Four decks have been
trained; every one exports to NumPy with **parity 1.0000** against its Torch
model, verified over 400 decisions each.

| deck | best val top-1 | note |
|---|---|---|
| Marnie/Froslass | 0.7785 | flagship, 89k training rows |
| Alakazam | 0.7024 | only 156 games — biggest upside |
| Mega Lopunny | 0.6332 | demonstrator mixing, see below |
| Team Rocket's Spidops | 0.5714 | **FAILED** the 0.5943 baseline; deck dropped |

## The three things that actually matter

**1. Train in fp32, not bf16.** Four of four `--amp` arms diverged (epochs 170,
290, 515, 520) with pre-clip `grad_norm` ~1e5 despite gradient clipping at 1.0.
Zero of four fp32 arms did; one ran 2,956 epochs. Two earlier explanations —
dropout, then the auxiliary loss — were both wrong. Use
`tools/launch_teacher_arm.sh`, which encodes the
corrected recipe.

**2. Imitate one strong player, not an archetype.** The harvester keeps an
episode when the *stronger* seat scored ≥1150, but the seat we imitate is chosen
by deck match — so the target can be the weaker player. Measured win rate of what
we were actually copying:

| corpus | episodes | win rate |
|---|---|---|
| gold-alakazam | 190 | 84.2% (82% one player) |
| sixth-sense | 926 | 57.9% (single player) |
| gold-lopunny | 227 | 56.8% (**five** players) |
| gold-spidops | 48 | **6.2%** — 45 losses |

Spidops trained 2,956 epochs to faithfully reproduce a player who loses 94% of
the time. Lopunny is the instructive case: it has *more* rows than Alakazam but a
worse gap (+0.284 vs +0.192), because five demonstrators give the same board
several conflicting "correct" moves. `tools/teacher_census.py` ranks all 782
players in the harvest by winning games; `tools/build_player_corpus.py` builds a
single teacher's wins.

**3. We are overfitting, not grokking.** The gap tracks corpus size almost
monotonically (89k rows → +0.091; 2.9k rows → +0.309). Spidops ran far past any
plausible transition and validation peaked at epoch 955 then fell. The levers
that work are more data and better labels, not more epochs.
