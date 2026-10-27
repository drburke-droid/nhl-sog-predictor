# Fantasy football's most valuable seasons since 1980 (`fantasy/`)

This is a bar-chart race of the best single seasons in fantasy football, 1980 to 2025. Value is points over a
replacement player in a 12-team PPR league. The board is a record book that builds as the years go by: it holds the 12
most valuable seasons so far. While a season is played, its bars grow toward their totals, and a great one climbs onto
the board and stays until better seasons push it off. It's built like the best-paid athletes race (`paid/`): the same
look, the same kind of synthesized score, and captions found in the data.

* **Film:** `../media/fantasy/fantasy_best_seasons.mp4` (1920x1080, 30 fps, 3 min 19 s, with an original synthesized
  score). It opens with a 42-second explainer of the method in plain words.
* **Data:**
  * `../media/fantasy/seasons.csv`: every QB, RB, WR and TE season 1980-2025, with stat line, PPR points, games, rank
    at his position, replacement level, VORP and rank in its season.
  * `replacement.csv`: the replacement player and the position leader for each season and position.
  * `weekly.csv`: from 1999, each game's points and the VORP the film credits to it, for each season's top 13.

An earlier version raced career VORP from 2000 instead. It is in this branch's history, at commit `c52e12a`.

## Findings

<!-- FINDINGS -->
* **The record:** Marshall Faulk's 2000 (26 TDs in 14 games), worth 329.7 points over replacement. LaDainian
  Tomlinson's 2006 (31 TDs) finished 0.1 behind, and Christian McCaffrey's 2019 is third (318).
* **How the record moved.** Each of these took No. 1:

  | Season | VORP |
  |---|---:|
  | Billy Sims 1980 | 174 |
  | William Andrews 1981 | 184 |
  | Eric Dickerson 1983 | 202 |
  | James Wilder 1984 (85 catches) | 245 |
  | Emmitt Smith 1992 | 254 |
  | Emmitt Smith 1995 | 269 |
  | Marshall Faulk 2000 | 330 |

  Dickerson's 2,105-yard 1984 is worth 178: in PPR, Wilder's 85 catches outscored it.
* **The final board:**
  * Faulk 2000 and 2001;
  * Tomlinson 2006 and 2003;
  * McCaffrey 2019 and 2025;
  * Priest Holmes 2002 and 2003;
  * Cooper Kupp 2021;
  * Emmitt Smith 1995;
  * Jerry Rice 1995;
  * Steven Jackson 2006.
* **The board's make-up.** It holds 10 RB seasons and 2 WR seasons. Seven are from the 2000s, two from the 1990s, one
  from the 2010s and two from the 2020s.
* **Best season at each other position:**
  * WR: Kupp 2021 (283), No. 8 all-time;
  * QB: Dan Marino 1984 (5,084 yards, 48 TD, 221), No. 3 all-time when it ended;
  * TE: Todd Christensen 1983 (92 catches, 192), No. 2 when it ended.

  The 1980s board had room for them; the 2000s' running backs pushed them off.
* **Short seasons.** The 1982 strike cut the season to 9 games. Wes Chandler's 1982 still reached the board: 1,032
  yards in 8 games, worth 153.
* **Running backs dominate single seasons** more than they did careers. With these ranks they hold 74% of the board's
  places across the race (see below).

## Method

* **Sources.**
  * **1980-1998:** Pro Football Reference's season fantasy tables, from the copy in
    [fantasydatapros/data](https://github.com/fantasydatapros/data). That copy holds season totals only, and its repo
    carries no licence file. Pro Football Reference itself refuses requests from this machine (a Cloudflare
    challenge).
  * **1999-2025:** nflverse's weekly player stats, with each season's roster position.
  * **Overlap check:** for 2006 and 2019 the two sources agree within 1.2-1.4 points a season among the top 60. The
    differences are the two-point conversions and return TDs that the season tables lack.
* **Scoring.** Standard PPR, regular season only:
  * 1 point per reception;
  * 1 per 10 rushing or receiving yards;
  * 1 per 25 passing yards;
  * 6 per rushing or receiving TD, and 4 per passing TD;
  * −2 per interception or lost fumble;
  * from 1999 only: 6 per special-teams TD and 2 per two-point conversion.

  Fumbles lost are recorded from 1994. For 1980-1993 they're estimated at 54.5% of fumbles, the league's share lost in
  1994-1998.
* **Replacement level.** For each season and position, replacement is the season total of the Nth-best player. The ranks
  come from a lineup of a QB, 2 RBs, 3 WRs, a TE and a flex (RB, WR or TE), twelve teams:
  * **RB30 and WR42:** the 24 RB and 36 WR starters, and about half the 12 flex spots each;
  * **TE14:** the 12 starters, and the odd flex;
  * **QB16:** the 12 starters, and the bye-week and injury fill-ins teams start along the way.
* **Kickers** are left out. There's no kicking data before 1999 here, and from 1999 the best kicker season is worth 71,
  far below the board.
* **VORP, by the game.** A season is worth the player's points minus replacement's share of the games he played. That
  share is the line over the season's games: 16; 9 in 1982; 15 in 1987; 17 from 2021.
  * A full season is worth his total minus the line; a season of 14 games, his total minus 14/16 of it.
  * A game he misses counts nothing, because his manager starts a replacement that week.
  * Faulk's 2000 edges Tomlinson's 2006 because Faulk did it in 14 games.
  * The 17th game adds about 6% to a full season's value; the seasons aren't scaled to 16 games.
* **Within a season**, the bars grow toward their totals. From 1999 they grow evenly over the games the player played,
  week by week. Before 1999 only season totals exist, so they grow evenly over the season's weeks.

Replacement levels, season PPR points, averaged by era:

| | QB16 | RB30 | WR42 | TE14 |
|---|---:|---:|---:|---:|
| 1980s | 141 | 141 | 112 | 89 |
| 1990s | 151 | 132 | 139 | 87 |
| 2000s | 185 | 149 | 149 | 107 |
| 2010s | 237 | 145 | 158 | 134 |
| 2020-25 | 242 | 157 | 158 | 141 |

## Choosing the replacement ranks

The ranks came out of the career version. There, RB36 and WR36 gave running backs well over half the board, and
RB30 / WR42 brought the positions into balance. Single seasons lean further toward running backs: the best RB seasons
stand further above replacement than anyone else's.

"Share of the board" below is the share of all-time top-12 places by position at the 46 season ends:

| Ranks | QB | RB | WR | TE | Final top 12 |
|---|---:|---:|---:|---:|---|
| QB13 RB36 WR36 TE13 | 1% | 91% | 7% | 2% | 12 RB |
| QB20 RB36 WR36 TE14 | 4% | 88% | 6% | 2% | 12 RB |
| **QB16 RB30 WR42 TE14 (used)** | **2%** | **74%** | **22%** | **2%** | **10 RB, 2 WR** |
| QB16 RB24 WR48 TE14 | 3% | 46% | 49% | 2% | 7 RB, 5 WR |
| QB24 RB24 WR48 TE20 | 21% | 39% | 35% | 4% | 7 RB, 3 QB, 2 WR |

`python compare.py` reproduces this table. To try other ranks, change `REPL` in `vorp.py` and rerun the steps below.

## The film

* **The explainer (42 s).** Most viewers won't know what value over replacement is, so the film opens with six short
  scenes:
  1. **Title.**
  2. **The question.** Who had the best season ever? Total points can't settle it. In 2020 Josh Allen outscored Derrick
     Henry, 396 to 333, yet Henry did more to win fantasy leagues.
  3. **The answer.** The same two bars split into what the best free player at the position scored and what the
     player added. The best free QB scored 261 and the best free RB 158, so Henry added 175 and Allen 135. That margin
     is what the race measures.
  4. **Who the free player is.** A 12-team league starts a QB, 2 RBs, 3 WRs, a TE and a flex. Once every team's
     starters are taken, the best player left is about the 16th QB, 30th RB, 42nd WR and 14th TE. Each season is
     measured against that year's free players.
  5. **Missed games count zero.** 300 points in 12 games beats 300 in 16.
  6. **How to read the chart.** Colours, the board of 12, outlined bars for the season being played, and the row
     under the board.

  The example is picked from the data (`captions.py`): the season's top-scoring QB who outscored one of its top three
  RBs, both playing every game, while the RB was worth at least 30 more over replacement. Of those, the widest
  outscoring wins.
* **The board.** It shows the twelve most valuable seasons so far. Each bar shows:
  * on the left, the player, his position, team and year;
  * on the right, the value, and once the season is over, its stat line.

  The current season's bars are outlined.
* **The chaser.** Under the board, while the season's best so far isn't on the board, a row shows it with the all-time
  place it would hold. In the quiet years, that row is the one that moves.
* **Pacing.** Seasons don't all take the same time. A season runs 2 to 12 beats of the score (1.4 to 8.4 s), depending
  on how much value its seasons put on the board and how often bars passed each other. The years when the board fills
  and the record years run longest; a season that changes nothing passes in 1.4 s. Within a season, the weeks take the
  first 80% and the final standings hold for the rest.
* **Replacement line.** Under the board, a line gives that season's replacement levels.
* **Colour** encodes the position: QB magenta, RB blue, WR yellow, TE green. These four were validated together for
  colour-blind separation on this background. The one warning (yellow against green under protanopia) is covered
  because every bar also names its position in text.
* **Captions** come from the data (`captions.py`), and none is written by hand. They mark:
  * each new No. 1;
  * the season 0.1 short of the record;
  * the best QB, WR and TE seasons, with the place each held when it ended;
  * the 1982 strike season;
  * the first 17-game season;
  * the last season's best and its place.
* **The score** is synthesized from scratch with the instruments of `../flights/score.py`:
  * 85.7 BPM, and every season starts on a beat;
  * a pad and a light arpeggio under the explainer, with a soft bell at each scene;
  * its energy follows how much value each season puts on the board;
  * it rises a whole step when Faulk's 2000 takes No. 1;
  * a bell marks each caption, and an impact each new No. 1.

## Run it

```
python vorp.py fetch     # season tables 1980-1998, nflverse weekly stats and rosters 1999-2025 -> data/ (about 210 MB)
python vorp.py           # -> ../media/fantasy/seasons.csv, replacement.csv, weekly.csv
python compare.py        # how the replacement ranks shape the board (the table above)
python captions.py       # -> captions.json, summary.json (from the data)
python video.py timeline # seconds per season
python video.py still 1984.95 2006.99 2026.5    # stills (t in seasons; 2026.5 is the closing summary)
python video.py still 1979.2 1979.45 1979.63    # explainer stills (1979.x: that fraction of the explainer)
python assemble.py render 4                     # picture (4 processes, about 4 min), score, mux -> ../media/fantasy/
```

Needs numpy, pandas, Pillow, numba and scipy (for the score), ffmpeg with libx264, and the Inter font.
