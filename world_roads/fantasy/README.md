# Fantasy football's most valuable players since 2000 (`fantasy/`)

This is a bar-chart race of career fantasy value in a 12-team PPR league, 2000 to 2025. Value is points over a
replacement player, so quarterbacks don't take over the board just by throwing for 4,000 yards every year. It's built
like the best-paid athletes race (`paid/`): the same look, the same kind of synthesized score, and captions found in
the data.

* **Film:** `../media/fantasy/fantasy_vorp_race.mp4` (1920x1080, 30 fps, 2 min 10 s, with an original synthesized
  score).
* **Data:**
  * `../media/fantasy/seasons.csv`: every QB, RB, WR, TE and K season, with stat line, fantasy points, rank at his
    position, replacement level, VORP and career totals.
  * `replacement.csv`: the replacement player, his points and the position leader, for each season and position.
  * `weekly.csv`: the weekly points and the share of the season's VORP each week added, for every player with a season
    above replacement.

## Why VORP and not career points

By career PPR points since 2000, the top six are all quarterbacks:

| | Pos | PPR points | VORP |
|---|---|---:|---:|
| Tom Brady | QB | 5,944 | 1,400 |
| Drew Brees | QB | 5,194 | 1,205 |
| Aaron Rodgers | QB | 5,057 | 1,222 |
| Peyton Manning | QB | 4,221 | 1,353 |
| Matthew Stafford | QB | 4,059 | 537 |
| Ben Roethlisberger | QB | 4,023 | 380 |

A 12-team league starts only twelve quarterbacks, and even the 16th-best scores 170-270 points. The points a
quarterback scores over that line are what make him worth drafting. Ben Roethlisberger's 4,023 points are worth 380
over replacement; Tiki Barber's 2,102 are worth 1,077.

## Choosing the replacement ranks

The first cut used QB13, RB36, WR36 and TE13. That gave running backs 57% of the board over the race and 6 of the
final top 12. Same-rank baselines for RB and WR were the cause. RB scoring falls off faster than WR scoring, and a PPR
league starts more receivers than backs, so the 36th RB is a much weaker player than the 36th WR.

"Share of the board" below is the share of top-12 places by position at the 26 season ends:

| Ranks | QB | RB | WR | TE | Final top 12 |
|---|---:|---:|---:|---:|---|
| QB13 RB36 WR36 TE13 (first cut) | 4% | 57% | 30% | 9% | 6 RB, 3 WR, 2 TE, 1 QB |
| QB20 RB36 WR36 TE14 K13 | 21% | 48% | 22% | 9% | 5 RB, 4 QB, 2 TE, 1 WR; Brady (1,963) is No. 2 |
| QB20 RB30 WR42 TE14 K13 | 22% | 30% | 40% | 9% | 4 QB, 4 WR, 2 RB, 2 TE; Brady (1,963) finishes No. 1 |
| **QB16 RB30 WR42 TE14 K13 (used)** | **16%** | **32%** | **43%** | **9%** | **4 QB, 4 WR, 2 RB, 2 TE** |
| QB14 RB30 WR42 TE14 K13 | 10% | 34% | 46% | 10% | 6 WR, 2 RB, 2 TE, 2 QB |

* **The RB and WR ranks are what fix the running backs.** A deeper QB rank alone doesn't: at QB20 with RB36 and WR36,
  RBs still hold nearly half the board, and quarterbacks gain at the receivers' expense.
* **The QB rank is the most sensitive.** Quarterbacks 13 to 20 are bunched together. From QB16 to QB20 the line drops by
  29 points a season on average, and that adds about 560 to a 20-season career like Brady's. At QB20, the four great
  quarterbacks take four of the top five places.
* **The ranks used come from a lineup:** a QB, 2 RBs, 3 WRs, a TE, a flex (RB, WR or TE) and a K, twelve teams.
  * RB30 and WR42: the 24 RB and 36 WR starters, and about half the 12 flex spots each;
  * TE14: the 12 starters, and the odd flex;
  * QB16: the 12 starters, and the bye-week and injury fill-ins teams start along the way;
  * K13: the 12 starters, and the next one.
* **Kickers count, but never come close.** Justin Tucker leads them with 299, which ranks 186th; the best kicker season
  is worth 70. The legend leaves them out because no kicker comes within 15 places of the board.

`python compare.py` reproduces this table. To try other ranks, change `REPL` in `vorp.py` and rerun the steps below.

## Method

* **Scoring.** Standard PPR, regular season only:
  * 1 point per reception;
  * 1 per 10 rushing or receiving yards;
  * 1 per 25 passing yards;
  * 6 per rushing, receiving or special-teams TD, and 4 per passing TD;
  * 2 per two-point conversion;
  * −2 per interception or lost fumble.

  `vorp.py` recomputes this from the box-score columns and asserts that it matches nflverse's `fantasy_points_ppr` on
  every row.
* **Kickers:** 3 points per field goal up to 39 yards, 4 from 40 to 49, 5 from 50 on, 1 per extra point, and −1 per
  missed or blocked field goal or extra point.
* **Replacement level.** For each season and position, replacement is the season total of the Nth-best player: QB16,
  RB30, WR42, TE14, K13 (see above).
* **Season VORP.** A season's VORP is the player's points minus that line, and **never below zero**. A season under
  replacement is one a manager would have filled off waivers. Career VORP is the sum of those seasons.
* **Season totals, not per game.** A player who misses half the season scores half the points. That's what happened to
  the team that drafted him.
* **Positions** come from each season's nflverse roster, so a receiver who became a running back counts as each in turn.
  Fullbacks count as running backs. Team defences are left out.
* **Counting from 2000.** Seasons before 2000 don't count. Marshall Faulk's 1998-99, Randy Moss's, Peyton Manning's,
  Marvin Harrison's, Terrell Owens's and Tony Gonzalez's early years are not in their totals.
* **Within a season**, the VORP accrues week by week in step with the player's points: each week adds its share of his
  season's (non-negative) points. So each season's board is exact at its end, and a player hurt in Week 3 stops
  climbing in Week 3.

Replacement levels, season points, averaged by era:

| | QB16 | RB30 | WR42 | TE14 | K13 |
|---|---:|---:|---:|---:|---:|
| 2000-09 | 185 | 149 | 149 | 107 | 114 |
| 2010-17 | 236 | 142 | 159 | 139 | 127 |
| 2018-25 | 242 | 156 | 157 | 135 | 128 |

## Findings

<!-- FINDINGS -->
* **LaDainian Tomlinson, by a distance.** He has 1,834 points over replacement from 2001 to 2011, and is the only
  player past 1,500.
  * No. 1 changed hands three times in 13 months. Priest Holmes passed Marshall Faulk in Week 7 of 2004. Marvin
    Harrison, the only receiver ever to lead, passed Holmes in Week 14. Tomlinson passed Harrison in Week 3 of 2005 and
    has held it since.
* **Tony Gonzalez is No. 2** with 1,420, ahead of Tom Brady (1,400).
* **Christian McCaffrey is the top active player.** He climbed to No. 6 (1,321) in Week 12 of 2025, 513 behind
  Tomlinson, 1.6 times his best season.
* **The top 12 after 2025:**
  * 4 QBs: Brady, Manning, Rodgers, Brees;
  * 4 WRs: Fitzgerald, Owens, Moss, Davante Adams;
  * 2 RBs: Tomlinson, McCaffrey;
  * 2 TEs: Gonzalez, Kelce.
* **Quarterbacks build slowly.** Peyton Manning's 55-touchdown 2013 made him the first QB in the top 3. Brady passed
  him as the top QB in the last week of 2021, his 22nd season.
* **Larry Fitzgerald** passed Terrell Owens as the top receiver in Week 9 of 2017.
* **The most valuable season** is Tomlinson's 2006 (31 TDs, 481 PPR points, 330 over replacement). McCaffrey's 2019 is
  next (318), then Faulk's 2000 (311).
* **Best season at each other position:**
  * WR: Cooper Kupp 2021 (283);
  * QB: Brady 2007 (213);
  * TE: Gronkowski 2011 (181), with Kelce's 2022 a hair behind.

## The film

* **The board.** It shows the twelve players with the most career VORP, one per team in the league. Each bar shows:
  * on the left, the player and his position and team (his current team while he's playing; afterwards, the team he
    played the most games for);
  * on the right, his career VORP and what this season has added so far.

  Once a player's last season is over, his bar fades and shows his career's span.
* **Pacing.** Seasons don't all take the same time. A season runs 2 to 12 beats of the score (1.4 to 8.4 s), depending
  on how much value the players on the board added in it and how often they passed each other (no more than about four
  passes a second where the cap allows). The crowded early 2000s run longest; the quietest seasons pass in 1.4 s. Within a season, the weeks take the first 80% and the final standings hold for the rest.
  Between weeks, values move in a straight line, and a pass takes about a third of a second.
* **Replacement line.** Under the board, a line gives that season's replacement levels.
* **Intro.** The title card explains the replacement ranks and the lineup they come from.
* **Colour** encodes the position: QB magenta, RB blue, WR yellow, TE green. These four were validated together for
  colour-blind separation on this background. The one warning (yellow against green under protanopia) is covered
  because every bar also names its position in text.
* **Captions** come from the data (`captions.py`), and none is written by hand. They mark:
  * each change at No. 1;
  * the most valuable season, and the one closest to it;
  * the first player to the highest 500-point milestone (1,500);
  * the first quarterback in the top 3;
  * each new No. 2 after the leader retires;
  * the last change of the top QB and the top WR;
  * the top active player's climb in the final season.

  Rules for still seasons and the first new name after them are kept. They don't fire with these ranks.
* **The score** is synthesized from scratch with the instruments of `../flights/score.py`:
  * 85.7 BPM, and every season starts on a beat;
  * its energy follows each season's value added on the board;
  * it rises a whole step when Tomlinson takes No. 1 for good;
  * a bell marks each caption.

## Run it

```
python vorp.py fetch     # nflverse weekly player stats and season rosters, 2000-2025 -> data/ (about 200 MB)
python vorp.py           # -> ../media/fantasy/seasons.csv, replacement.csv, weekly.csv
python compare.py        # how the replacement ranks shape the board (the README's table)
python captions.py       # -> captions.json, summary.json (from the data)
python video.py timeline # seconds per season
python video.py still 2000.5 2006.99 2026.5     # stills (t in seasons; 2026.5 is the closing summary)
python assemble.py render 4                     # picture (4 processes, about 2 min), score, mux -> ../media/fantasy/
```

Needs numpy, pandas, Pillow, numba and scipy (for the score), ffmpeg with libx264, and the Inter font.

Sources: [nflverse-data](https://github.com/nflverse/nflverse-data) releases `stats_player` (weekly player stats) and
`rosters` (season rosters).
