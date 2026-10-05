# demo/: a local match between two agents

`run_match.py` plays two small agents against each other on the official engine and prints
win counts with intervals. It is a smoke test of the whole loop (engine, observation,
legal-option response, result), and a place to plug in your own agent.

- `random`: a uniformly random legal response.
- `greedy`: sets up first (evolve, attach energy, play a card, use an ability), then attacks
  with the most printed damage, then ends the turn; outside the main phase it takes the first
  `maxCount` options, preferring YES.

Neither is a competition agent, and the demo says nothing about how strong `BEST1_fixed` was.
The trained models need weights that are not published.

## The engine is not included

The official engine is licensed for competition use only
(`LicenseRef-PTCG-ABC-Competition-Use-Only`), so this repository cannot ship it and the demo
loads it from a directory you supply.

1. Join the Pokémon TCG AI Battle competition on Kaggle and accept its rules.
2. Download the competition data from the **Data** tab (or
   `kaggle competitions download -c pokemon-tcg-ai-battle`) and unzip it.
3. Find the folder that contains `cg/`, `deck.csv` and `main.py`. In the unzipped data it is
   `sample_submission/sample_submission`.
4. Run, with Python 3.10 or newer (the engine's `cg/api.py` uses 3.10 syntax):

```bash
PTCG_ENGINE_DIR=/path/to/sample_submission/sample_submission python3 demo/run_match.py --games 1000
```

Both seats use the `deck.csv` from that folder, so the match is a mirror and only the agent
differs. Seats alternate game by game. `--seed` seeds the random agent only; the engine's
shuffles are not seeded, so counts vary slightly between runs.

If `PTCG_ENGINE_DIR` is unset the script prints these steps to stderr and exits with
status 2; if the folder has no `cg/` it says so and exits 2 as well.

## Example output

One run of the command above on an Apple M-series Mac. The 2.5 s is an observation from that
single run, not a benchmark: there is no repeated timing, seed or hardware record behind it,
and nothing in this repository depends on it.

```
1000 games, seats alternated, mirror deck
  greedy   920/1000 =  92.0%  [90.2, 93.5] nominal Wilson 95%
  random    80/1000 =   8.0%  [6.5, 9.8] nominal Wilson 95%
  draws or timeouts: 0
  2.5 s. The engine's shuffles are not seeded here, so counts vary slightly between runs.
```

## The model demo, no engine

`model_demo.py` needs numpy and torch (`PTCG_PYTHON=... bash scripts/demo.sh` runs it). On a
tiny SYNTHETIC model and SYNTHETIC boards it prints the NumPy serving forward's action
probabilities beside torch's, then three parity checks over 64 boards: a check whose torch
reference is built the same wrong way as the serving code reads a false PASS; the correct
check FAILs the wrong serving mode and PASSes the trained one. It exits 1 if any row comes
out otherwise. No real weights or engine data are involved, so it says
nothing about playing strength.

## Tests

`python3 -m unittest discover -s demo/tests` needs no engine: the match loop and both agents
run against a small test double with the same call surface as the engine's `cg.game`
(`battle_start`, `battle_select`, `battle_finish`) and plain-dict observations.
