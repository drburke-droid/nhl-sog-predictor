# Win now or tear it down? (`rebuild/`)

The theory: teams with rabid fan bases and high valuations can't stomach a rebuild, so their owners chase wins every
year, trading picks and prospects; teams that can tear down reach higher peaks later, at the cost of lower lows. Tested
on the NHL, NBA, MLB and NFL, 2000-01 to 2024-25: 124 franchises, 3,026 seasons.

## Run it

```
python fetch.py      # Sports Reference pages (NHL, NBA, MLB) and nflverse files, cached in data/ (~15 min, once)
python build.py      # -> ../media/rebuild/seasons.csv, drafts.csv, nfl_trades.csv, teams.csv
python analyse.py    # -> ../media/rebuild/results.json, team_level.csv, events.csv (~2 min)
```

Valuations, metro populations and World Series winners come from Wikipedia pages saved in `data/val/`,
`data/metro/` and `data/champions/` (Forbes lists of the most valuable NHL, NBA, NFL and MLB teams; US metropolitan
statistical areas, 2025 estimates; Canadian census metropolitan areas, 2021 census; list of World Series champions).

## Data

* **Seasons** (`build.py`): win share as the league ranks it (NHL points %, NFL ties as half a win), roster age (NHL
  and NBA team pages; MLB batters' and pitchers' ages averaged; none for the NFL), playoffs, titles, and each season's
  rank in its league as a percentile (0 the worst team, 1 the best). Franchises are followed through moves
  (`teams.py`): Thrashers to Jets, SuperSonics to Thunder, Expos to Nationals, Coyotes to Utah, and so on.
* **Owners and fans**: Forbes value (the latest edition on Wikipedia's per-league lists: NHL and NBA 2025, NFL and MLB
  2026), the home metro's population, and the *premium*, value beyond what the market predicts (the residual of log
  value on log population within the league), the closest thing here to a fan base's pull.
* **Win now**: roster age against the league that season; own first-round picks made per draft (NHL, NBA, NFL drafts
  2000-2025); NFL first-round picks traded away, net (nflverse trades since 2002).

## Findings

* **Valuable teams bottom out less, because they are better.** Value against bottom-three finishes: Spearman -0.29
  (p 0.003); the fan premium -0.33 (p 0.001, every league negative); market size alone -0.07. With each franchise's
  average rank held fixed, all three vanish (-0.07 to -0.08), and value is unrelated to the spread of a team's results
  (+0.03): rich teams are shifted up (more top finishes and titles, fewer bottoms), not steadier.
* **Win-now signs, mostly from market size.** Older rosters: value +0.55, market +0.48, premium +0.28; at equal
  quality the market's +0.46 survives and the premium's does not. Own first-round picks: value -0.26 (p 0.02), -0.21
  at equal quality (p 0.05). NFL picks traded: nothing (+0.15, 29 teams).
* **Tearing down buys lower lows, not higher highs.** From mediocrity, the teams that fell to the bottom three within
  two seasons against those that stayed in the middle, over the next ten: the same peak (best rank 0.81 against 0.82),
  a much lower low (0.02 against 0.13) and a lower average (0.39 against 0.48). Titles: 16 % against 11 %, not
  significant; over eight definitions +4 to +11 points, significant in about half. The NBA is where it shows most (+5
  to +20 points; 18 % against 6 % in the main definition). Against every other path from mediocrity, tearing down is
  worse (peak -0.04).
* **Champions had not torn down more than anyone else.** 47 % of the 58 champions since 2010 had a bottom-three finish
  in the ten seasons before, against 50 % of all teams (NHL 69 % against 50 %, p 0.14; NBA 47 % against 55 %; MLB 47 %
  against 52 %; NFL 27 % against 44 %).

## Caveats

* Values are today's, after the seasons they are compared with; winning raises value, so value cannot be read as the
  cause. Market size is the one measure winning does not move.
* The premium measures value beyond market size, which is more than fan passion (stadium deals, ownership, brand).
* MLB picks could not be traded before 2012 and prospects traded are not measured, so MLB's win-now signs rest on age.

## The video

`../media/rebuild/win_now_or_tear_it_down.mp4` (1920x1080, 30 fps, 3 min 25.5 s, 6,165 frames, captions and an original score):
the light-speed film's look, every number read from `results.json` and the tables beside it.

```
python video.py teardown 14 27   # stills of a section (out/teardown_014.0.png ...)
python assemble.py render        # render the ten sections (4 at a time), join them with cross-fades, add the score
```

`look.py` (palette, glow, the season grid, chart helpers, chunked rendering), `video.py` (the sections: hook, theory,
stories, richer, shifted, winnow, teardown, titles, verdict, outro), `music.py` (the score, synthesized with
`../flights/score.py`), `assemble.py`. `VOICEOVER.md` has a narration script timed to the captions, for a voice over the
top.
