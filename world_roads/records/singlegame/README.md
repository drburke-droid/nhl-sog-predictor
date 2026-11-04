# Single-day outliers: the greatest-seasons method, one game at a time

The same question as `../outliers/`, but every entry is one performance on one day: each season's best single
game per player, compared with the best single game of everyone else in the 15 seasons before and after. Athletics
and swimming need no change (a season best is already one day). Team sports switch from season totals to games.

| File | What it does |
|---|---|
| `build_data.py` | `nba`, `nhl`, `mlb`, `nfl`: every regular-season game reduced to each player's best game per season (`../cache/singlegame/`); `merge`: the model's input lists (`data/`); `check`: 14 record games every build must reproduce |
| `score.py` | The locked model (E3_main_env: ±15, curved trend, league adjustment with elasticities estimated from these lists, peer exclusion, ties), three variants, and the 16-version robustness check |
| `data/` | `sg_categories.csv`, `sg_season_lists.csv` (top ten per category-season, with game dates) |
| `outputs/` | Rankings, spec ensembles and score scales per variant; elasticities; best season per category |

## Sources

| League | Source | Seasons used | Validation |
|---|---|---|---|
| NBA | Eoin Moore, *Historical NBA Data and Player Box Scores* (CC0), mirrored on Hugging Face as `CoachCheezy/nba-stats` | 1955–56 to 2023–24 (box scores before 1955 are incomplete, e.g. 18 of 1946–47's games; 2024–25 is partial) | Wilt 100, Kobe 81, Wilt 55 rebounds, Skiles 30 assists |
| NHL | Hugging Face `RentoSaijo/NHL_DB`: every goal with scorer and assists, from NHL.com game data | 1917–18 to 2025–26; complete every season | Malone 7 goals, Sittler 10 points, Malone's 39-goal season |
| MLB | Retrosheet via Chadwick Bureau retrosplits (daybyday) | 1901–2025, complete | Whiten and Bottomley 12 RBI, Gehrig 4 HR, Stennett 7 hits, Wood and Clemens 20 strikeouts |
| NFL | nflverse weekly player stats | 1999–2025 (no openly downloadable game-level source before 1999) | C. Johnson 329, A. Peterson 296 |

All downloads were sequential with pauses and a descriptive User-Agent, from hosts whose robots.txt allows them.
The NHL stats API and Basketball-/Hockey-/Pro-Football-Reference were not used (their robots rules or terms
restrict automated access).

## The whole-number problem, and the three variants

In single games most team stats are small whole numbers. In most seasons everyone else's best game is exactly
3 home runs, 4 goals, or 6 hits, so the usual give-or-take is close to zero and one extra unit looks like six
steps. Johnny Burnett's 9 hits (1932, in an 18-inning game) and Mike Schmidt's 4 home runs (1976, 10 innings)
score 7.3 and 6.3 for that reason.

- **all**: every category as scored. The whole-number artefacts reach #3 and #4.
- **corrected**: a half-unit continuity correction for team-sport counts. Burnett still reaches #4.
- **big** (recommended): only team stats whose season-best game is usually 15 or more: NBA points, rebounds and
  assists; pitcher strikeouts; NFL passing, rushing and receiving yards.

## Clear of the field (eligibility rule, November 2026)

The score is distance above the curve, but a day only makes the ranking if it also beats every other mark on its own
graph (the comparison window, in the athlete's own units) by at least one step (`score.CLEAR_STEPS`). Without it, a
mark could rank high for being early while later rivals in the same window matched or beat it. Of 2,797 scored days,
45 pass. `outputs/sg_beaten_big.csv` lists what the rule removed from the old top 100, with the mark that came within
a step: from the old top ten, Ledecky's 8:04.79 (2016; McIntosh 8:05.07 in 2025, 0.12 steps) and Oldfield's 22.86 m
(1975; Barnes 23.12 m in 1990, beaten). The swimming supersuit seasons (2008-09) and list cut-offs are left out of the
check. The rule is applied in every version of the 16-version ensemble, on that version's own window.

Top ten (big, with the rule): Beamon 7.80, Wilt Chamberlain's 100 points 7.40, Železný 5.95, Kobe Bryant's 81 points
4.88, Rojas 4.69, Bubka 4.45, Marchand 4.34 (2025: one-sided window), Vetter 4.30, Gretchen Walsh's 54.33 100 m
butterfly 4.17 (2026: one-sided), Bob Cousy's 28 assists 4.10 (1959: one-sided, box scores start in 1955-56).
Chamberlain, Železný and Bryant are in the top ten in all 16 versions; Beamon in 10 (with a 20-season window Robert
Emmiyan's 8.86 m in 1987 comes within 0.34-0.43 steps; with a straight 15-season trend Carl Lewis's 8.79 m in 1983
comes within 0.96); places 5-10 range from 3 to 12 of 16.

## Rebuilding the inputs

`data/`, `../cache/` and `../followup/data/` are not in the repository. To rebuild what the `big` variant needs:
`python build_data.py nba mlb nfl` (sources above), `python build_data.py merge`; the athletics and swimming season
lists are `../followup/data/v2/`, rebuilt from `../followup/handoff/original_project/inputs/` (top ten per season, with
the v2 corrections that need no download cache: wind-prefixed names, seasons after 2026, a swimmer listed twice, the
transliteration-aware name key), which reproduces every published athletics and swimming score. League rates
(`../followup/data/context/*_league.csv`): NBA from Basketball-Reference's league per-game table (one page; early box
scores undercount rebounds and assists), MLB and NFL as per-team-game totals from the same game data. Without the NHL
lists the elasticity pooling cannot be redone, so run `python score.py big --stored-gamma` (the published values in
`outputs/sg_gamma.csv`). Rebuilt this way, the published top 100 (before the rule) comes back in the same order, 94 of
100 scores to the published three decimals and the rest within 0.01 (MLB and NFL league rates), and the 16-version
counts exactly. The `all` and `corrected` outputs need the NHL lists and were not regenerated with the rule.

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may
contact Retrosheet at www.retrosheet.org.
