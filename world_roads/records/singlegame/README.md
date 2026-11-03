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

Recommended top ten (big): Beamon 7.80, Wilt Chamberlain's 100 points 7.40, Železný 5.95, Kobe Bryant's 81 points
4.88, Ledecky 4.86, Rojas 4.69, Oldfield 4.54 (a professional-circuit mark), Bubka 4.45, Marchand 4.34 (2025:
one-sided window), Vetter 4.30. The top four stay in the top ten under 15 or 16 of the 16 versions; places 5–10
are unstable (2 to 12 of 16).

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may
contact Retrosheet at www.retrosheet.org.
