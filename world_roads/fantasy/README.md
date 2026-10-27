# Fantasy football's most valuable players since 2000 (`fantasy/`)

This is a bar-chart race of career fantasy value in a 12-team PPR league, 2000 to 2025. Value is points over a
replacement player, so quarterbacks don't take over the board just by throwing for 4,000 yards every year. It's built
like the best-paid athletes race (`paid/`): the same look, the same kind of synthesized score, and captions found in
the data.

* **Film:** `../media/fantasy/fantasy_vorp_race.mp4` (1920x1080, 30 fps, 2 min 17 s, with an original synthesized
  score).
* **Data:**
  * `../media/fantasy/seasons.csv`: every QB, RB, WR, TE and K season, with stat line, fantasy points, rank at his
    position, replacement level, VORP and career totals.
  * `replacement.csv`: the replacement player, his points and the position leader, for each season and position.
  * `weekly.csv`: each game's points and the VORP the film credits to it, for every player with a season above
    replacement.

## Why VORP and not career points

By career PPR points since 2000, the top six are all quarterbacks:

| | Pos | PPR points | VORP |
|---|---|---:|---:|
| Tom Brady | QB | 5,944 | 1,428 |
| Drew Brees | QB | 5,194 | 1,330 |
| Aaron Rodgers | QB | 5,057 | 1,266 |
| Peyton Manning | QB | 4,221 | 1,275 |
| Matthew Stafford | QB | 4,059 | 548 |
| Ben Roethlisberger | QB | 4,023 | 606 |

A 12-team league starts only twelve quarterbacks, and even the 16th-best scores 170-270 points. The points a
quarterback scores over that line are what make him worth drafting. Ben Roethlisberger's 4,023 points are worth 606
over replacement; Tiki Barber's 2,102 are worth 1,094.

## Choosing the replacement ranks

The first cut used QB13, RB36, WR36 and TE13. That gave running backs well over half the board across the race and 6
or 7 of the final top 12. Same-rank baselines for RB and WR were the cause. RB scoring falls off faster than WR scoring, and a PPR
league starts more receivers than backs, so the 36th RB is a much weaker player than the 36th WR.

"Share of the board" below is the share of top-12 places by position at the 26 season ends. VORP is counted as in the
film (by the game, with seasons below replacement negative):

| Ranks | QB | RB | WR | TE | Final top 12 |
|---|---:|---:|---:|---:|---|
| QB13 RB36 WR36 TE13 (first cut) | 1% | 65% | 27% | 8% | 7 RB, 3 WR, 2 TE |
| QB20 RB36 WR36 TE14 K13 | 21% | 51% | 21% | 7% | 5 RB, 4 QB, 2 TE, 1 WR; Brady (2,017) is No. 2 |
| QB20 RB30 WR42 TE14 K13 | 23% | 29% | 39% | 8% | 4 QB, 4 WR, 2 RB, 2 TE; Brady (2,017) finishes No. 1 |
| **QB16 RB30 WR42 TE14 K13 (used)** | **15%** | **33%** | **43%** | **9%** | **4 QB, 4 WR, 2 RB, 2 TE** |
| QB14 RB30 WR42 TE14 K13 | 9% | 36% | 46% | 10% | 7 WR, 2 RB, 2 TE, 1 QB |

* **The RB and WR ranks are what fix the running backs.** A deeper QB rank alone doesn't: at QB20 with RB36 and WR36,
  RBs still hold nearly half the board, and quarterbacks gain at the receivers' expense.
* **The QB rank is the most sensitive.** Quarterbacks 13 to 20 are bunched together. From QB16 to QB20 the line drops by
  29 points a season on average, and that adds about 590 to a 20-season career like Brady's. At QB20, the four great
  quarterbacks take four of the top five places.
* **The ranks used come from a lineup:** a QB, 2 RBs, 3 WRs, a TE, a flex (RB, WR or TE) and a K, twelve teams.
  * RB30 and WR42: the 24 RB and 36 WR starters, and about half the 12 flex spots each;
  * TE14: the 12 starters, and the odd flex;
  * QB16: the 12 starters, and the bye-week and injury fill-ins teams start along the way;
  * K13: the 12 starters, and the next one.
* **Kickers count, but never come close.** Justin Tucker leads them with 299, which ranks 185th; the best kicker season
  is worth 71. The legend leaves them out because no kicker comes within 15 places of the board.

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
* **Season VORP, by the game.** A season is worth the player's points minus replacement's share of the games he played.
  That share is the line over the season's 16 games (17 from 2021). A full season is worth his total minus the line; a
  season of 8 games, his total minus half the line. Career VORP is the sum of his seasons.
  * **Playing below replacement counts negative.** Manning's 2015 (10 games, 91 points) costs 78. Moss's 2012 with the
    49ers costs 68. Fitzgerald's 2020 costs 33.
  * **A missed game counts nothing.** His manager starts a replacement that week. Rodgers' 2023, which lasted four snaps
    of one game, costs 14.
  * **Why not season totals with no floor:** that charges injuries and backup years as if they were bad play. Rodgers
    would fall from 1,266 to 311, charged for his three backup seasons and his injury-shortened ones. Brady would drop
    from 1,428 to 1,002 and McCaffrey from 1,469 to 1,122.
* **Positions** come from each season's nflverse roster, so a receiver who became a running back counts as each in turn.
  Fullbacks count as running backs. Team defences are left out.
* **Counting from 2000.** Seasons before 2000 don't count. Marshall Faulk's 1998-99, Randy Moss's, Peyton Manning's,
  Marvin Harrison's, Terrell Owens's and Tony Gonzalez's early years are not in their totals.
* **Within a season**, the film spreads its VORP evenly over the games he played. Each season's board is exact at its end,
  a player hurt in Week 3 stops moving in Week 3, and a bad season's bar shrinks game by game. Counting each game's own
  points over replacement instead made close bars swap back and forth week to week.

Replacement levels, season points, averaged by era:

| | QB16 | RB30 | WR42 | TE14 | K13 |
|---|---:|---:|---:|---:|---:|
| 2000-09 | 185 | 149 | 149 | 107 | 114 |
| 2010-17 | 236 | 142 | 159 | 139 | 127 |
| 2018-25 | 242 | 156 | 157 | 135 | 128 |

## Findings

<!-- FINDINGS -->
* **LaDainian Tomlinson, by a distance.** He has 1,877 points over replacement from 2001 to 2011, and is the only
  player past 1,500.
  * Marshall Faulk led from 2000 until Priest Holmes passed him in Week 4 of 2005.
  * Tomlinson passed Holmes five weeks later and has held No. 1 since.
* **Christian McCaffrey is No. 2.** He passed Tony Gonzalez in Week 16 of 2025 to finish with 1,469. That's 409 behind
  Tomlinson, 1.3 times his best season. Gonzalez had held No. 2 since Week 17 of 2013.
* **The top 12 after 2025:**
  * 2 RBs: Tomlinson, McCaffrey;
  * 2 TEs: Gonzalez, Kelce;
  * 4 WRs: Owens, Fitzgerald, Davante Adams, Antonio Brown;
  * 4 QBs: Brady, Brees, Manning, Rodgers.
* **Quarterbacks build slowly.** The highest any quarterback has stood is No. 4: Peyton Manning, in Week 7 of his
  55-touchdown 2013. Brady passed Brees as the top QB in Week 12 of 2021.
* **Terrell Owens** passed Marvin Harrison as the top receiver in Week 13 of 2007.
* **Bad seasons are rare at the top.** Among the final top 12, 13 seasons fall below replacement, worth −348 together.
  The worst are Manning's 2015 (−78), Antonio Brown's rookie 2010 (−47) and Davante Adams's rookie 2014 (−46).
* **The most valuable season** is Faulk's 2000: 26 TDs in 14 games and 329.7 over replacement. Tomlinson's 2006 (31
  TDs, 16 games) is 0.1 behind at 329.6, and McCaffrey's 2019 is next (318).
* **Best season at each other position:**
  * WR: Cooper Kupp 2021 (283);
  * QB: Brady 2007 (213);
  * TE: Gronkowski 2011 (181), with Kelce's 2022 a hair behind.

## The film

* **The board.** It shows the twelve players with the most career VORP, one per team in the league. Each bar shows:
  * on the left, the player and his position and team (his current team while he's playing; afterwards, the team he
    played the most games for);
  * on the right, his career VORP and what this season has added (or taken away) so far.

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
  * each change at No. 1 after the first season that lasts four weeks or more;
  * the most valuable season, and the one 0.1 behind it;
  * the first player to the highest 500-point milestone (1,500);
  * the highest any quarterback climbs;
  * each new No. 2 after the leader retires;
  * the last change of the top QB and the top WR;
  * the top active player's climb in the final season, unless another caption already marks it.

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
