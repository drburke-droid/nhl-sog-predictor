# Fact check: every number on screen

Every number in the film is written by `facts.py` into `facts.json`; `film.py` reads it from there and types none
of its own, except the record marks in the card headers. This table maps each on-screen claim to its value and
source, and to where a primary record can be checked.

**Status key**

- **data**: matches the project data, and the data matches the source named. Not yet compared with a second,
  independent record.
- **data + web**: also confirmed against an outside page during this build (linked).
- **flag**: worth checking against a primary record before publishing.

The production pack (`SOURCE_FACT_CHECKS.md`) asks for every record to be checked against a primary source before
release. Its ten headline marks are the first column of the countdown below. Its requirement that both data sets
agree on all ten marks holds.

## Locked ranking (E3_main_env)

The scores come from `followup/outputs/rankings_sensitivity.csv`, column `E3_main_env_z`. They match
`FINAL_RANKING_LOCKED.csv` in the pack.

| Rank | Season | Mark | Score | "Without league adjustment" (A2_main) |
|---|---|---|---|---|
| 10 | Ichiro Suzuki, 2004 | 262 hits | 5.27 | 5.10 |
| 9 | Babe Ruth, 1920 | 54 home runs | 5.56 | 3.52 |
| 8 | Dennis Rodman, 1992–93 | 18.3 rebounds a game | 5.68 | 5.27 |
| 7 | Stephen Curry, 2015–16 | 402 three-pointers | 5.69 | 4.27 |
| 6 | Dazzy Vance, 1924 | 262 strikeouts | 5.83 | 3.48 |
| 5 | Jan Železný, 1996 | 98.48 m javelin | 5.95 | 5.95 (athletics: no league adjustment) |
| 4 | Wayne Gretzky, 1985–86 | 163 assists | 6.47 | 7.53 |
| 3 | Joe DiMaggio, 1941 | 56-game hitting streak | 7.01 | 7.01 (no league adjustment) |
| 2 | Wilt Chamberlain, 1961–62 | 50.4 points a game | 7.05 | 7.66 |
| 1 | Bob Beamon, 1968 | 8.90 m long jump | 7.80 | 7.80 (no league adjustment) |

## Card lines

| Card | On screen | Value from | Status |
|---|---|---|---|
| 10 | Broke George Sisler's record of 257, set in 1920 | `previous_record` (season lists, Baseball-Reference) | data |
| 10 | Next best in 2004: Juan Pierre, 221 | `runner_up` | data |
| 9 | His own record the year before: 29 | `previous_record` (1919) | data |
| 9 | Next best in 1920: George Sisler, 19; the average team hit 39 | `runner_up`; `extras.mlb_1920_avg_team_hr` = 39.4 (league HR per team-game 0.255 × 154.25 games) | data |
| 8 | Next best that season: Shaquille O'Neal, 13.9 | `runner_up` | data |
| 8 | His 18.7 the season before scores lower (1991–92: 5.35, 1992–93: 5.68) | `rankings_sensitivity.csv` | data |
| 8 | Led the league in rebounding seven seasons running, 1991–92 to 1997–98 | `season_lists_v2.csv`, rank 1 in nba_rpg 1992–1998 | data |
| 7 | Previous record his own: 286, the season before; next best Klay Thompson, 276 | `previous_record`, `runner_up` | data |
| 6 | Next best in either league: Walter Johnson, 158 | `runner_up` | data |
| 6 | About 2.7 strikeouts per team per game | `extras.mlb_1924_so_per_team_game` = 2.69 | data |
| 5 | Javelin redesigned in 1986; 98.48 m still the world record | category starts 1986; no better mark through 2026 in the data | data |
| 5 | Next best in 1996: Raymond Hecht, 92.28 m | `runner_up` | data |
| 5 | Broke his own record of 95.66 m, set in 1993 | `previous_record` | data |
| 4 | Next best in assists: Mario Lemieux, 93 | `runner_up` | data |
| 4 | Assists alone topped every other player's points; Lemieux had 141 | `extras.nhl_1986_points_leaders` (Gretzky 215, Lemieux 141, Coffey 138) | data |
| 3 | Record before: 44 games, Willie Keeler, 1897 | `previous_record` | data |
| 3 | Longest by anyone else in 1941: 24 games (Cecil Travis, 16 June–14 July) | computed from Retrosheet game logs (`cache/streaks.csv`). The same computation gives Ted Williams 23 games from 15 May to 7 June, which matches the usual figure. No outside page found that states Travis's 24. | **flag** |
| 3 | Closest since: Pete Rose's 44, in 1978 | `extras.longest_streak_since_1941` | data |
| 2 | Next best: Walt Bellamy, 31.6, 18.8 points behind | `runner_up` | data |
| 2 | 48.5 minutes a game, overtimes included (80 games) | `extras.wilt_minutes_per_game` (Basketball-Reference) | data |
| 2 | Teams averaged 118.8 points | `extras.nba_points_per_team_game_1962` (`followup/data/context/nba_league.csv`) | data |
| 1 | 55 cm beyond the world record (8.35 m, Ralph Boston 1965; equalled by Igor Ter-Ovanesyan 1967) | `previous_record` | data |
| 1 | Next best in 1968: 8.28 m | `runner_up` (Igor Ter-Ovanesyan) | data + web: [elitetrackr](https://elitetrackr.com/athletes/igor-ter-ovanesyan-22971) lists his 8.28 on 17 Aug 1968. Check the 1968 ATFS or Track & Field News year list. |
| 1 | Stood until Mike Powell's 8.95 m in 1991 | `next_record` | data |
| 1 | 2,240 m altitude, +2.0 m/s tailwind | Mexico City; World Athletics record entry | data |

**League-environment insets** are drawn from `league_env` in `facts.json`, which comes from the per-category
league rates in `followup/data/context/`. Team-sport charts show each value as a share of an average team's total
(value ÷ league rate per team-game ÷ games). That is an intuitive version of the adjustment. The score itself uses
the estimated elasticity (`gamma_estimates.csv`), and the card says so.

## Method section

| On screen | Value | Status |
|---|---|---|
| 74 categories, 6 sports | `robustness.n_categories` | data |
| Ledecky 2016, women's 800 m freestyle, 8:04.79 | `method_example` | data |
| "4.86 units above her era", ranks 12th | `method_example.z`, `rank_e3` | data |

## Women's throws section

| On screen | Value | Status |
|---|---|---|
| Shot put record 22.63 m (Natalya Lisovskaya, 1987) | `womens_throws.ath_W_SP` | data |
| No 22 m throw since 1988; the best this decade is 21.14 m (2026) | season lists through 2 Nov 2026 | data; **flag** the 2026 mark, since the season list is from the 2026 scrape |
| Best discus since 2000 is 3.28 m short of 76.80 (73.52 m, 2025) | `womens_throws.ath_W_DT.best_since_2000` | data |
| 1980s: 18 shot puts of 22 m or more (by 3 women); 50 discus throws of 72 m or more by 11 women | all-time lists (alltime-athletics.com), counted by `facts.py` | data |
| Record holders' best seasons score 2.4 (shot, Lisovskaya 1988) and 3.2 (discus, Reinsch 1988) | `rankings_sensitivity.csv` | data |
| The top ten starts at 5.27 | rank 10 | data |
| Random out-of-competition testing began in athletics in the late 1980s | Historical context, phrased without naming anyone. IAAF out-of-competition testing was introduced around 1989–1991. | context; keep the wording general |

These events were not excluded from the model. They were scored like every other event. The best women's throw is
Anita Włodarczyk's 2016 hammer at #51 (z 3.57), then Reinsch's 1988 discus at #62 and Valerie Adams's 2011 shot put
at #75.

## Closing section

| On screen | Value | Status |
|---|---|---|
| Four seasons in the top ten under all 16 specifications (windows 7–20, straight or curved, with or without league adjustment) | `spec_ensemble.csv` → `robustness.in_all_16` | data |
| "The other six places change hands often" | model-averaged bootstrap: next best is 0.32 (Walsh) | data |
| All-seasons rarity: Gretzky #20, DiMaggio #1 | `robustness.all_seasons_rank` (B1_allseasons) | data |
| Without the league adjustment: Marino 1984 #9, Ledecky 2016 #10 | `robustness.no_env_rank` (A2_main) | data |
| 4,940 season-leading performances | `robustness.n_scored` | data |

## Wording kept out on purpose (from the pack)

- No "once every N seasons" or return periods anywhere, including the women's throws. Those use observed gaps
  instead.
- No "proved", "greatest athlete" or "statistically certain".
- The film never says the adjustment accounts for minutes, possessions, ice time or innings.
- "50.4 today = 39.7" does not appear.
