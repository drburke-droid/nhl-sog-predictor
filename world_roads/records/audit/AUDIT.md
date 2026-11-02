# Audit brief: "The ten greatest seasons" and "Out of Reach"

This package holds the analysis behind a planned video on the ten greatest seasons in sport, and behind the
web page *Out of Reach* that charts world records and league records against the field. It is written for
an independent auditor: everything needed to check the ranking is in here, and `audit_recompute.py`
rebuilds the published top 10 from plain CSV files with numpy alone.

Data were read in November 2026. Season labels: NBA and NHL seasons are shown as "1961–62" on the page and
stored as the year they end (1962) in the CSVs.

## 1. What was asked

The brief, in the author's words: "We need find complete data for all big 4 sports going back as far as
possible. The big four are an integral part of this. Add swimming. Remove events that are too niche. This
will eventually be a video on top 10 greatest sports seasons of all time but utilizing our metric of 15
years before and after."

"Our metric" came from an earlier step: how far a performance stands above what the rest of the world was
doing, expressed as "once in N seasons" (a return period). The 15-years-before-and-after version measures
every season against its own era.

## 2. Package contents

| Path | What it is |
|---|---|
| `AUDIT.md` | This brief |
| `CHATGPT_PROMPT.md` | A suggested prompt for the audit |
| `audit_recompute.py` | Standalone recomputation of the ranking from `inputs/` (numpy + standard library) |
| `inputs/categories.csv` | The 74 categories: what is measured, the performance transform, schedules, seasons left out, cut-offs, the record |
| `inputs/season_lists.csv` | Each season's top ten in every category (44,952 rows): raw value, performance, games per team, short-season flag, raw and display name |
| `inputs/record_lines.csv` | Every step of every record line on the page |
| `inputs/pipeline_era.csv` | The pipeline's own score for every season-leading performance at windows 7, 10, 15 and 20 (18,561 rows) |
| `outputs/greatest_seasons.csv` | The published ranking: top 200 athlete-seasons with ranks at the other windows |
| `outputs/era_adjusted.csv` | Every season-leading performance scored at ±15 |
| `outputs/allsports.json` | Everything the page draws for the 74 categories |
| `outputs/hitting_streaks.csv` | Longest hitting streak per player-season, 1897–2025 (from Retrosheet game logs) |
| `outputs/deep_summary.csv`, `seasons.csv`, `deep_report.json` | The athletics record analysis ("Out of Reach") |
| `outputs/wr_progression.csv`, `olympic_top8.csv`, `summary.csv`, `games.csv` | The first film, *The Record and the Rest* (record lines and Olympic finals) |
| `outputs/report.html` | The published page (open in a browser) |
| `outputs/logs/` | The printed output of the final build and the football validation |
| `code/` | The full pipeline (fetching, parsing, measures, page) |
| `docs/README.md`, `docs/ACCURACY.md` | Project readme; accuracy disclosure for the first film |

The pipeline's download cache (raw HTML/JSON, several hundred MB) is not included. The code in `code/` will
not run without it; `audit_recompute.py` does not need it.

## 3. Quick check (2 minutes)

```
python audit_recompute.py              # expect: 4659 matched, largest |z difference| 0.00e+00, and the top 10 below
python audit_recompute.py --window 10  # the same at other windows
python audit_recompute.py --flat       # sensitivity: no trend line
python audit_recompute.py --no-shift   # sensitivity: no separate level for swimming's suit seasons
```

## 4. Data, by sport

| Sport | Categories | Seasons | Source |
|---|---|---|---|
| Athletics | 30: men's and women's 100, 200, 400, 800, 1500 m, sprint hurdles, 400 m hurdles, high jump, pole vault, long jump, triple jump, shot, discus, javelin, hammer | performances scored from 1960; lists from the 1960s (javelins from the 1986 and 1999 redesigns) | Peter Larsson's all-time lists (alltime-athletics.com; about 164,000 legal performances: fully automatic timing, legal wind, indoor marks included for jumps and throws), filled below their cut-offs by World Athletics' season lists |
| Swimming | 22: men's and women's 50, 100, 200, 400, 800, 1500 m freestyle; 100 m backstroke, breaststroke, butterfly; 200 and 400 m individual medley (long course) | 1994–2026 (a season counts once at least 20 swimmers are listed: 1994 in most events, about 2000 in the men's 800 m and women's 1500 m) | World Aquatics world rankings (best times, top 100 per season); record lines from Wikipedia's long-course progressions |
| Baseball (MLB) | 9: hitting streak, RBI, home runs, batting average, hits, stolen bases (from 1898), strikeouts, wins, ERA | 1876–2026 | Baseball-Reference leaders: the National League alone 1876–1900 (its own leader pages), AL + NL from 1901. Left out: American Association, Union Association, Players' League, Federal League, Negro Leagues (see 7.13). Hitting streaks from Retrosheet game logs, 1897–2025 |
| Basketball (NBA) | 4: points, rebounds and assists per game; three-pointers made | 1947–2026 (rebounds from 1951, threes from 1980) | Basketball-Reference season leaders |
| Hockey (NHL) | 3: goals, assists, points | 1918–2026 | Hockey-Reference season leaders |
| Football (NFL) | 6: passing yards, touchdown passes, rushing yards, receiving yards, interceptions (from 1940), sacks (from 1982) | 1932–2025 | Pro Football Archives team pages summed per player (NFL 1932–98 and AFL 1960–69, pooled); nflverse season stats from 1999 |

Left out as too niche (the brief asked for this): the 5000 m and 10,000 m in athletics; the 200 m
backstroke, breaststroke and butterfly in swimming; and the non-Olympic 50 m backstroke, breaststroke and
butterfly.

Schedules (games per team, for scaling totals): MLB 1876–1900 from the NL standings (mean of W+L; 64 in 1876
rising to about 130–150 in the 1890s), then 140/154/162 by era; NHL by era (22 games in 1918 to 84); NBA 80
or 82 (50, 66, 72 in shortened seasons); NFL from the season standings 1932–69 (mean of W+L+T; 1960 pools
NFL and AFL at 12.8), then 14/16/17.

Seasons cut short are left out of every total-based category and its comparison windows: MLB 1918, 1981,
1994, 2020; NBA 1999, 2012, 2020 (threes only); NHL 1995, 2013, 2020, 2021; NFL 1982. Rate categories
(batting average, ERA, per-game averages) keep them.

## 5. The measure, exactly

For every season-leading performance x in season y of a category (only the season's leader is scored):

1. **The era.** For each season s from y−15 to y+15 that is in the data and not cut short: the best
   performance by anyone other than the performer. Names are compared after removing accents, case and
   punctuation, and the performer's own seasons are left out entirely. Totals are scaled to the performer's
   schedule: value × games(y) ÷ games(s). In athletics, a season in which nobody else is listed above the
   all-time list's cut-off takes the cut-off mark. A window needs at least 8 seasons with real values.
2. **Performance.** Speed (distance ÷ time) for running and swimming, the mark for jumps and throws, the stat
   for leagues, minus the ERA (so bigger is always better).
3. **The trend.** Least squares through those points: perf_s = a + b·(s − y), plus, in swimming only, a
   level-shift term c·[s ∈ {2008, 2009}] for the full-body polyurethane suits (banned from 2010). The term
   is used only when the window contains a suit season and at least 8 non-suit seasons.
4. **The score.** Baseline = a (+ c if y is itself a suit season). Residuals r_s; sd = √(Σr² ÷ (n − k)),
   k = number of fitted parameters. **z = (x − baseline) ÷ sd.**
5. **"Once in N seasons".** A Gumbel distribution fitted by moments to the residuals
   (β = s·√6/π and μ = mean − 0.5772·β, where s is the sample standard deviation of the residuals);
   N = 1 ÷ (1 − F(x − baseline)), F(t) = exp(−exp(−(t − μ)/β)).
   This is reported alongside z; the ranking uses z.
6. **One entry per athlete-season.** If an athlete led several categories in the same season, the season is
   entered once, under the category where it scores highest (the others are listed as "also led").
   4,037 athlete-seasons in all.
7. **The top ten.** Ranked by z; the page's top ten keeps each athlete's best season (repeat seasons by the
   same athlete are noted, not listed).
8. **Robustness.** The whole ranking is recomputed with 7, 10 and 20 seasons either side.

Why a trend line: over 31 seasons most sports move a lot (the passing game, the three-point shot, the
javelin redesign). Measured against a flat average, a season is scored partly on where the trend happened to
be. With a flat window, Beamon's 8.90 m (1968) scored 3.2 and fell out of the list because long jumping
rose steadily through his window. The flat version is available as `--flat` for comparison.

Why the swimming suit term: without it, the "greatest season of its era" in 11 of the 22 swimming events
was a 2008–09 suit swim. The straight trend cannot see a two-season equipment jump.

## 6. Results

### The published top ten (±15 seasons)

| # | Athlete, season | Category | Value | z | Once in … | Rank among all athlete-seasons at ±7 / ±10 / ±15 / ±20 |
|---|---|---|---|---|---|---|
| 1 | Wilt Chamberlain, 1961–62 | NBA points per game | 50.4 | 7.73 | 42,727 | 4 / 4 / 1 / 1 |
| 2 | Wayne Gretzky, 1985–86 | NHL assists | 163 | 7.50 | 31,764 | 3 / 2 / 2 / 3 |
| 3 | Joe DiMaggio, 1941 | MLB hitting streak | 56 | 7.20 | 21,286 | 7 / 6 / 3 / 2 |
| 4 | Ichiro Suzuki, 2004 | MLB hits | 262 | 5.18 | 1,529 | 17 / 12 / 4 / 7 |
| 5 | Bob Beamon, 1968 | Men's long jump | 8.90 m | 5.02 | 1,291 | 2 / 1 / 6 / 5 |
| 6 | Jan Železný, 1996 | Men's javelin | 98.48 m | 4.99 | 1,216 | 11 / 13 / 7 / 8 |
| 7 | Bob Feller, 1946 | MLB strikeouts | 348 | 4.71 | 833 | 24 / 28 / 9 / 23 |
| 8 | Dennis Rodman, 1991–92 | NBA rebounds per game | 18.7 | 4.50 | 628 | 15 / 9 / 10 / 71 |
| 9 | Katie Ledecky, 2016 | Women's 800 m freestyle | 8:04.79 | 4.47 | 706 | 10 / 37 / 11 / 10 |
| 10 | Pete Rose, 1978 | MLB hitting streak | 44 | 4.46 | 597 | 13 / 41 / 12 / 6 |

Next, within 0.15 of tenth place: Yulimar Rojas 2022 (triple jump 15.74 m, 4.36), Dan Marino 1984 (48 TD
passes, 4.35), Léon Marchand 2025 (200 m IM 1:52.69, 4.34), Maury Wills 1962 (104 stolen bases, 4.34).
Chamberlain 1962–63 (5.14) and Gretzky 1984–85 (4.84) would also be in the top ten if repeat seasons counted.

### The same ranking under other settings (one entry per athlete; from `audit_recompute.py`)

| Setting | Top ten |
|---|---|
| ±7 | Walsh 2026 (n=8), Beamon, Gretzky, Chamberlain, Marino 1984, Mahuchikh 2024, DiMaggio, Perkins 1994 (n=8), Ledecky 2016, Železný |
| ±10 | Beamon, Gretzky, Walsh 2026, Chamberlain, DiMaggio, Marino 1984, Wills 1962, Rodman, Ichiro, Železný |
| ±20 | Chamberlain, DiMaggio, Gretzky, Beamon, Rose 1978, Ichiro, Železný, Rojas 2022, Ledecky 2016, Bolt 2009 |
| ±15, flat (no trend) | DiMaggio, Gretzky, Chamberlain, Ichiro, Walsh 2026, Feller, Rose, Vetter 2020, Marchand 2025, Železný (Beamon not in the top 12) |
| ±15, no suit term | as published, except Ledecky 2016 drops out and Rojas 2022 enters at 10th |

Stable under every window: Chamberlain, Gretzky, DiMaggio and Beamon stay in the top ten.

### Side checks shown on the page (scored at ±15)

| Check | z | Rank among all athlete-seasons |
|---|---|---|
| Beamon 1968 minus 31 cm (largest published estimate of altitude and wind help) = 8.59 m | 2.32 | about 226 |
| Florence Griffith Joyner's official 10.49 (1988; wind reading questioned; the ranking uses her 10.61) | 5.73 | about 4 |
| Babe Ruth's 54 home runs, 1920 | 3.31 | about 70 |
| Don Hutson's 1,211 receiving yards, 1942 (next best that season: 571) | 2.88 | about 114 |
| Usain Bolt 9.58, 2009 | 4.17 | 18 |

## 7. Known limitations and judgment calls (where to push)

1. **The trend line matters.** Flat versus trended windows reorder the list (Beamon in or out). The trended
   version is defensible (sports move over 31 years) but it is a modelling choice the brief did not specify.
2. **One-sided windows at the ends of the data.** The first seasons of a sport (MLB 1876, NFL 1932, NBA 1947,
   swimming 1994) and the last few (2024–26) have only one side; the trend is extrapolated. Their scores are
   less certain and move most with the window (Walsh 2026: #1 at ±7, #107 at ±20). At ±7, Walsh and Perkins
   are scored on only 8 seasons.
3. **Only season leaders are scored.** A season that was second to an even greater one in the same year is
   never scored, and a close two-person duel lowers the leader's score, because the runner-up is "everyone
   else's best".
4. **z across different kinds of statistics.** Averages, totals, maxima (hitting streak = the longest run)
   and times have different tail shapes; comparing z across them assumes the scatter around the trend is
   comparable. "Once in N" uses a Gumbel fit, which can order seasons differently from z (Ledecky's N is
   above Rodman's although her z is lower).
5. **Category counts differ by sport.** Athletics has 30 categories, swimming 22, MLB 9, NFL 6, NBA 4, NHL
   3. More categories give more chances for an extreme season (a multiple-comparisons effect); per-athlete
   deduplication softens but does not remove it.
6. **Swimming suit term.** It moves Ledecky 2016 in or out of the top ten. It is a single shared level for
   2008 and 2009, fitted from the rest of the field in those seasons.
7. **Name matching.** The performer's own seasons are excluded by normalised name. Two different people with
   the same name would wrongly exclude each other; one person spelled two ways would wrongly count as
   "someone else". Swimmers are mapped to the record lists' spellings (Leon → Léon Marchand, Cesar Cielo
   Filho → César Cielo).
8. **Schedule scaling is linear in games.** It treats 1,211 yards in 11 games like 1,321 in 12. It does not
   adjust for roster size, pace or rule changes.
9. **Altitude, wind, equipment, doping.** Athletics marks are taken as ratified or listed; Beamon's altitude
   and tailwind are not adjusted (a side check shows the effect). No judgment about doping is made anywhere.
10. **Athletics before 1960 is not scored**; the all-time lists are not deep enough before then. Lists are
    complete above their cut-off; below it, World Athletics' season lists fill in.
11. **Quirky seasons the measure cannot see**, such as Ed Williamson's 27 home runs in 1884 (Chicago's Lake
    Front Park counted balls over its very short fences as home runs that season only), or wartime depletion
    (1942–45).
12. **Football data**: pre-1999 totals are summed from team pages. Every season's passing-yard,
    TD-pass, rushing and receiving leader 1932–98 matches Wikipedia's leader lists, except Paul Moss in
    1933 (383 yards here, 283 on Wikipedia; not resolved, and it does not affect the top of the list). Interception leaders were
    spot-checked (1940, 1948, 1950, 1952, 1961, 1966, 1975, 1981, 1998, 2005). Sacks before 1982 are
    unofficial and not used.
13. **Negro Leagues.** MLB recognises them (1920–48), and Baseball-Reference's combined leaderboards include
    them. They are excluded here because their short schedules make season leaderboards hard to compare; for
    batting average and ERA in 1920–48 the AL and NL leader pages are used. Including them would change some
    records (Josh Gibson's .466 in 1943) and could change era scores in those years.
14. **Pooling.** AL + NL are pooled from 1901, and NFL + AFL in 1960–69, as one field per season.
15. **Shortened seasons** are excluded only for totals; rates keep them (e.g. batting average in the
    60-game 2020 season).
16. **The 2026 hitting-streak data** are not yet published by Retrosheet, so streaks end in 2025.

## 8. The other parts of the page

* **Every record, one chart each.** Gold: the record line. Blue: the median of each season's top ten
  (athletics and swimming: one mark per athlete), smoothed over five seasons. The number under each chart is
  how often today's best would match the record: a Gumbel fitted by moments to the leading marks of the last
  15 seasons (2012–26; NFL and streaks 2011–25), totals scaled to today's schedule.
* **Out of Reach (athletics).** Records grouped as within reach (once in under 35 seasons at today's level),
  era record (set by 2000 in an event whose depth peaked before 2006 and has not recovered) or lone outlier.
  See `outputs/deep_summary.csv` and `code/deep.py`.
* **The first film** (*The Record and the Rest*) has its own disclosure in `docs/ACCURACY.md`.

## 9. Validation already done

* `audit_recompute.py` reproduces all 4,659 scores at ±15 (and every score at ±7, ±10, ±20) exactly, and
  the published `greatest_seasons.csv`.
* Football leaders: see 7.12 (`outputs/logs/nfl_validation.txt`).
* Swimming: lists that came back empty were refetched; every event has a full list in every season from
  2005 (gaps before: men's 1500 m 1997, women's 1500 m 2002 and 2004). In all 22
  events the median of the top ten was slower in 2010 than in 2009, which is consistent with the suit ban.
* Hitting streaks: DiMaggio 56 (1941), Rose 44 (1978), Keeler 44 (1897) match the published record lists.

## 10. Sources and notices

alltime-athletics.com (Peter Larsson); World Athletics season lists; World Aquatics world rankings;
Wikipedia record progressions; Baseball-Reference, Basketball-Reference, Hockey-Reference; Pro Football
Archives; nflverse; Retrosheet.

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested
parties may contact Retrosheet at www.retrosheet.org.
