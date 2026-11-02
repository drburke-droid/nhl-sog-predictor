# What actually wins Monopoly? (video research)

Research and simulations for a video on the best Monopoly strategy: which named strategies win (Railroad
Tycoon, Boardwalk Dreamer, buy-everything, three houses, jail campers...), which colour set is best, how the
rules people actually play change things, and what a genetic algorithm finds when it designs its own strategy.

- `reports/` - the research report and video plan.
- `research_notes/Monopoly winning strategies/` - the notes behind it: published board maths, simulation and AI
  studies, champion strategy, rules and history, the videos already made, and our own simulation results.
- `sim/` - our simulator and every result.

## sim/

| File | What it is |
|---|---|
| `monopoly.cpp` | The game engine (C++): standard US board, current card decks, official rules (auctions, even building, 32 houses / 12 hotels, mortgages, bankruptcy), trading (set-for-set swaps and cash buyouts), house-rule switches. ~25 million turns a second on 4 cores. |
| `markov.py` | Exact long-run square probabilities (Markov chain). Reproduces Truman Collins' published leave-jail table to two decimals; for the stay-in-jail policy it gives 9.39% in jail against Collins' 9.46%. |
| `experiments.py` | Every experiment; results cached as JSON in `results/<experiment>/`. |
| `evolve.py` | Genetic algorithm (stage 1: beat three normal players; stage 2: arms race against its own champion). |
| `analyze.py` | Builds `results/summary.md` (all the tables) and `results/summary.json`. |
| `charts.py` | Draws `results/charts/*.png`. |
| `results/sample_games/*.jsonl` | Turn-by-turn event logs of single games (for animating a real game). |

Rebuild and rerun:

```
cd monopoly/sim
g++ -O2 -std=c++17 -pthread -o monopoly monopoly.cpp
python3 markov.py
python3 experiments.py length duel table skip h2h players shark   # about an hour on 4 cores
python3 evolve.py stage1 25 && python3 evolve.py stage2 3
python3 experiments.py ablation ecology
python3 analyze.py && python3 charts.py
```

Strategy specs: `--strats` takes one spec per player separated by `|`. A spec is a preset name
(`everything`, `nothing`, `railroads`, `darkblue`, `orange`, `prudent`, `threehouses`, `houselock`, `jailbird`,
`smartjail`, `stubborn`, `shark`, `allin`, `champion`, or any colour) optionally followed by `;key=value`
overrides (`buy=all-green`, `reserve=300`, `build=3`, `jail=stay`, `trade=none`, ...). See `parseStrat` in
`monopoly.cpp`.
