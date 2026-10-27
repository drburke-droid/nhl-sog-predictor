# Fantasy football's most valuable players since 2000 (`fantasy/`)

This is a bar-chart race of career fantasy value in a 12-team PPR league, 2000 to 2025. Value is points over a
replacement player, so quarterbacks don't take over the board just by throwing for 4,000 yards every year. It's built
like the best-paid athletes race (`paid/`): the same look, the same kind of synthesized score, and captions found in
the data.

* **Film:** `../media/fantasy/fantasy_vorp_race.mp4` (1920x1080, 30 fps, about 2 min 6 s, with an original synthesized
  score).
* **Data:**
  * `../media/fantasy/seasons.csv`: every skill player's season, with stat line, PPR points, rank at his position,
    replacement level, VORP and career totals.
  * `replacement.csv`: the replacement player, his points and the position leader, for each season and position.
  * `weekly.csv`: the weekly points and the share of the season's VORP each week added, for every player with a season
    above replacement.

## Why VORP and not career points

By career PPR points since 2000, the top six are all quarterbacks:

| | Pos | PPR points | VORP |
|---|---|---:|---:|
| Tom Brady | QB | 5,944 | 1,059 |
| Drew Brees | QB | 5,194 | 971 |
| Aaron Rodgers | QB | 5,057 | 1,055 |
| Peyton Manning | QB | 4,221 | 1,119 |
| Matthew Stafford | QB | 4,059 | 423 |
| Ben Roethlisberger | QB | 4,023 | 236 |

A 12-team league starts only twelve quarterbacks, and the 13th-best still scores 200-280 points. The points a quarterback
scores over that line are what make him worth drafting. Ben Roethlisberger's 4,023 points are worth 236 over
replacement; Tiki Barber's 2,102 are worth 1,244.

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
* **Replacement level.** For each season and position, replacement is the season total of the Nth-best player:
  * **QB13 and TE13:** a 12-team league starts twelve of each, so the 13th is the best one left on waivers;
  * **RB36 and WR36:** two starters, a flex share and a bench spot per team, so three each.
* **Season VORP.** A season's VORP is the player's points minus that line, and **never below zero**. A season under
  replacement is one a manager would have filled off waivers. Career VORP is the sum of those seasons.
* **Season totals, not per game.** A player who misses half the season scores half the points. That's what happened to
  the team that drafted him.
* **Positions** come from each season's nflverse roster, so a receiver who became a running back counts as each in turn.
  Fullbacks count as running backs. Kickers and defences are left out.
* **Counting from 2000.** Seasons before 2000 don't count. Marshall Faulk's 1998-99, Randy Moss's, Peyton Manning's,
  Marvin Harrison's, Terrell Owens's and Tony Gonzalez's early years are not in their totals.
* **Within a season**, the VORP accrues week by week in step with the player's points: each week adds its share of his
  season's (non-negative) points. So each season's board is exact at its end, and a player hurt in Week 3 stops
  climbing in Week 3.

Replacement levels, season PPR points, averaged by era:

| | QB13 | RB36 | WR36 | TE13 |
|---|---:|---:|---:|---:|
| 2000-09 | 202 | 128 | 158 | 110 |
| 2010-17 | 249 | 127 | 169 | 143 |
| 2018-25 | 258 | 135 | 172 | 141 |

## Findings

<!-- FINDINGS -->
* **LaDainian Tomlinson, by a distance.** He has 2,023 points over replacement from 2001 to 2011. He passed Marshall
  Faulk for No. 1 in Week 3 of 2005 and has held it since. No one else has reached 1,500.
* **Christian McCaffrey is the only active player close.** He has 1,428, and passed Tony Gonzalez for No. 2 in Week 13 of
  2025. Tomlinson's lead is 595, 1.8 times McCaffrey's best season (329 in 2019).
* **The top 12 after 2025:**
  * 6 RBs: Tomlinson, McCaffrey, Barber, Peterson, Kamara, Forte;
  * 3 WRs: Fitzgerald, Owens, Moss;
  * 2 TEs: Gonzalez, Kelce;
  * 1 QB: Peyton Manning, No. 12 with 1,119.

  Brady (1,059) and Rodgers (1,055) are just off the board.
* **Tied for the most valuable season since 2000.** Marshall Faulk's 2000 (26 TDs, 460 PPR points) and Tomlinson's 2006
  (31 TDs, 481) are both exactly 351.7 over replacement. McCaffrey's 2019 is third (329).
* **Best season at each position:**
  * QB: Brady 2007 (199);
  * WR: Cooper Kupp 2021 (271);
  * TE: Kelce 2022 (177), with Gronkowski's 2011 a hair behind.
* **The workhorse era.** The early 2000s produced 28 running-back seasons worth 200 or more over replacement. 2010-17
  produced 13, and 2018-25 produced 21.
* **The board stands still from 2020 to 2022.** Everyone in the top 12 has retired or finishes below replacement.
  Travis Kelce (Week 7 of 2023) is the first new name since Antonio Brown in 2018, and McCaffrey follows.

## The film

* **The board.** It shows the twelve players with the most career VORP, one per team in the league. Each bar shows:
  * on the left, the player and his position and team (his current team while he's playing; afterwards, the team he
    played the most games for);
  * on the right, his career VORP and what this season has added so far.

  Once a player's last season is over, his bar fades and shows his career's span.
* **Pacing.** Seasons don't all take the same time. A season runs 2 to 12 beats of the score (1.4 to 8.4 s), depending
  on how much value the players on the board added in it and how often they passed each other (no more than about four
  passes a second where the cap allows). The crowded early 2000s run longest; the seasons in which the board stands
  still pass in 1.4 s. Within a season, the weeks take the first 80% and the final standings hold for the rest.
  Between weeks, values move in a straight line, and a pass takes about a third of a second.
* **Replacement line.** Under the board, a line gives that season's replacement levels.
* **Colour** encodes the position: QB magenta, RB blue, WR yellow, TE green. These four were validated together for
  colour-blind separation on this background. The one warning (yellow against green under protanopia) is covered
  because every bar also names its position in text.
* **Captions** come from the data (`captions.py`), and none is written by hand. They mark:
  * the most valuable season and the one that ties it;
  * the last change at No. 1;
  * the first player to 2,000;
  * the first quarterback to finish a season on the board;
  * the still seasons;
  * the first new name after them;
  * each new No. 2 after the leader retires.
* **The score** is synthesized from scratch with the instruments of `../flights/score.py`:
  * 85.7 BPM, and every season starts on a beat;
  * its energy follows each season's value added on the board;
  * it rises a whole step when Kelce breaks the still years;
  * a bell marks each caption.

## Run it

```
python vorp.py fetch     # nflverse weekly player stats and season rosters, 2000-2025 -> data/ (about 200 MB)
python vorp.py           # -> ../media/fantasy/seasons.csv, replacement.csv, weekly.csv
python captions.py       # -> captions.json, summary.json (from the data)
python video.py timeline # seconds per season
python video.py still 2000.5 2006.99 2026.5     # stills (t in seasons; 2026.5 is the closing summary)
python assemble.py render 4                     # picture (4 processes, about 10 min), score, mux -> ../media/fantasy/
```

Needs numpy, pandas, Pillow, numba and scipy (for the score), ffmpeg with libx264, and the Inter font.

Sources: [nflverse-data](https://github.com/nflverse/nflverse-data) releases `stats_player` (weekly player stats) and
`rosters` (season rosters).
