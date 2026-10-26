# Win now or tear it down? (`rebuild/`)

The theory: teams with rabid fan bases and high valuations can't stomach a rebuild, so their owners chase wins every
year, trading picks and prospects; teams that can tear down reach higher peaks later, at the cost of lower lows. Tested
on the NHL, NBA, MLB and NFL, 2000-01 to 2024-25: 124 franchises, 3,026 seasons. The second pass adds the levers
themselves, league by league: payroll, free agents, and trades (star out and prospects in, or the reverse).

## Run it

```
python fetch.py      # Sports Reference pages (NHL, NBA, MLB) and nflverse files, cached in data/ (~15 min, once)
python build.py      # -> ../media/rebuild/seasons.csv, drafts.csv, nfl_trades.csv, teams.csv
python analyse.py    # -> ../media/rebuild/results.json, team_level.csv, events.csv (~2 min)
python moves.py      # -> players.csv, moves.csv, payroll.csv, team_moves.csv (the second pass; fetch.py ~65 min more)
python horizon.py    # -> trade_horizon.csv, team_horizon.csv (trades by time horizon and control)
python levers.py     # -> levers.json, lever_events.csv, horizon_events.csv (~3 min)
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

### Payroll, free agents and trades (`moves.py`)

| | Payroll | Trades | Free agents | Player value |
|---|---|---|---|---|
| NHL | 2014-2025, rebuilt from every contract's cap hit (CapWages) and who played where | 2001-2025 (NHL Trade Tracker) | 2010-2025 (CapWages) | point shares; goalies scaled to skaters |
| NBA | 2001-2025, team salary tables | 2001-2025 (Basketball Reference transactions) | 2001-2025 | win shares |
| MLB | 2001-2025, team salaries (Baseball Reference; r = 0.97 against the Lahman salaries, 2001-2016) | 2001-2021 (Retrosheet) | 2001-2021 (Retrosheet) | wins above replacement |
| NFL | 2013-2024, cap charges (OverTheCap via nflverse) | 2002-2024 (nflverse) | 2013-2024 (OverTheCap) | cap share (no production measure on hand) |

* **Stars.** A player's value over his league's star line, the top 5 % of all player-seasons (1.0 = an all-star-level
  season: 8.3 win shares, 8.3 point shares, 4.0 WAR, 5.2 % of the NFL cap). A traded player is judged by his season
  before (mid-season, also by what he had done for the team trading him); a free agent by his season before (in the
  NFL, by what his new deal pays). Free agents count only when they sign from another team.
* **Future assets.** A first-round pick 1, a second 0.35, later picks 0.1, a prospect (23 or younger, 25 in MLB, 24 in
  the NFL, with under a quarter-star of track record, or a minor leaguer never in the majors) 0.5.
* **Payroll** as a share of the league's average that season.
* A **sell** season: half a star or more traded away, net, and half a future asset or more back; a **buy** season the
  reverse. This first classification is kept in `levers.json`, but the trade findings below use the next one.

### Trades by time horizon (`horizon.py`)

Trading away a big salary is not selling: teams dump salary, or send it out to make a trade work under the rules. And a
team like the Rays trades cheap, cost-controlled stars before they get expensive, often for players on short-term
salaries packaged with young prospects. So every trade is valued again by when its value arrives and how long the team
keeps it (NBA, NHL, MLB; the NFL has no production measure here and stays on the first classification):

* **Production, not pay.** A player's projected stars: his last three seasons weighted 5-4-3 (a shortened season
  scaled to a full one), carried forward by his league's aging curve (fitted on every player and season, 0 for a
  player out of the league). A salary filler who doesn't produce counts for nothing, whatever he earns.
* **Control.** A veteran counts only while he stayed with the team that got him: a rental for the rest of the season,
  a player with four years of control for four. The realized tenure stands in for the contract.
* **Prospects and picks.** What traded prospects of that age (busts included) and picks in that round went on to give,
  season by season.
* **Now and later.** Now: the season of the move (what is left of it) and the next; later: the four after that.
* **Salary**, as a share of the league's average payroll (NBA 2001-2025, NHL cap hits 2010-2025, MLB 2001-2016), to
  tell a dump from a sale.
* A window from mediocrity (the next two seasons, ranked within the league): **tore down**, the fifth that gave up the
  most present value and came out ahead in future value; **dumped**, gave up as much with no future back and shed a
  tenth of an average payroll; **shed**, the same without the salary (stars walking, deals for little); **went for
  it**, the fifth that added the most present value and paid in future value; **added**, added as much without losing
  future value (mostly free agents); **neither**, the middle of both.

Checks: the Rays' 2014 David Price deal, a sale by stars last season, reads as a gain now and later (+0.6 stars now,
+1.3 later: Smyly, Franklin and Adames under control against a season and a half of Price); their 2012 Shields and
Davis deal as present for future (-1.0 now, +0.6 later); Chicago's 2017 Jimmy Butler deal the same (-1.4, +1.4);
Brooklyn's 2013 Garnett and Pierce deal as paying the future (-2.2 later, three first-round picks).

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

## Findings: payroll, free agents and trades, league by league (`levers.py`)

The leagues do differ, the way their money rules say they should: baseball has no cap, basketball a soft one with a
tax, hockey and football hard caps.

* **How much money matters.** Payrolls spread far wider in MLB (a standard deviation of 42 % of the league's average
  payroll, the median season) than in the NBA (14 %), NHL (11 %) or NFL (10 %). Payroll against finish: MLB 0.43, NBA
  0.34, NFL 0.30, NHL 0.27 (Spearman, team-seasons); within a franchise, its richer years against its poorer ones, MLB
  0.32, NBA 0.32, NFL 0.28, the NHL 0.11 (not significant). Stars signed in free agency, net, go with a better season
  than the one before in every league (MLB 0.35, NFL 0.24, NHL 0.21, NBA 0.13); stars traded in over the summer, net,
  a little in the NBA (0.15), MLB (0.10) and NHL (0.08), not in the NFL.
* **Who pulls the levers: in baseball, money; in basketball, the market.** In MLB a franchise's value goes with its
  payroll (+0.88; +0.79 at equal average rank) and with signing stars (+0.71; +0.45), and in trades the valuable clubs
  take on salary while the poorer ones shed it (+0.58; +0.46 at equal rank, p 0.009). Counted by stars last season,
  rich MLB clubs also seemed to sell less and buy more; counted by horizon and control they trade present for future
  about as often as anyone (-0.07 at equal rank): that gap was salary. In the NBA the big markets sign the stars
  (market +0.58 at equal rank, value +0.66, both p < 0.001) and trade away more first-round picks (value +0.38, p
  0.04); payroll follows value (+0.45, p 0.01), and the valuable teams send out more salary than they take back in
  trades (-0.40, p 0.03), the luxury tax at work. In the NHL, teams with a fan premium tear down less often (-0.41, p
  0.03); otherwise valuable teams spend, sign and trade like everyone else. In the NFL valuable teams trade away more
  first-round picks (+0.47, p 0.008) and trade for more expensive players (+0.55, p 0.002; pay, not play).
* **After a bad season (bottom third), the next two: rich against poor (each league's top and bottom third by
  value).** MLB: the rich signed 1.1 stars' worth of free agents against none (-0.1: the poor sign players who were
  below replacement; CI +0.6 to +2.0) and added more present value in all (1.2 stars against 0.4, CI -0.1 to +1.8),
  but tore down about as often (15 % against 14 %). Counted by stars last season, the poor seemed to sell far more (60 against 38 %); much of that was salary. NBA: the rich signed more (1.7 stars against 1.3, CI +0.1 to +0.8). NHL:
  the rich tore down less (3 % against 13 %, CI -22 to +3 points). The rich answer a bad year by signing, in baseball
  and basketball, rather than by refusing to sell.
* **Tearing down, counted by horizon: a higher peak in the NBA, not elsewhere.** Few of the windows the first
  classification called selling were teardowns: in MLB 6 of 33 (8 were salary dumps), the NBA 6 of 22, the NHL 6 of
  32. From mediocrity, against the middle, over the ten seasons after: NBA teardowns reached a higher peak (best rank
  0.89 against 0.80; +0.12 with the starting rank, the slide into it and roster age held fixed, CI +0.02 to +0.20) at
  the cost of a lower low (0.03 against 0.08), titles 14 % against 12 %. That is the theory's shape, in the league
  where one player changes a team; over four definitions (fifths or thirds, eight or ten seasons) the peak gap stays
  +0.06 to +0.09, though not significant on its own in any. NHL teardowns: a lower peak (0.73 against 0.86; -0.09 to
  -0.14 over the four, significant in one) and a lower low. MLB: a slightly lower peak (-0.02 to -0.06, not
  significant) and titles 11 % against 20 %, while going for it (adding the most present value, paying in future
  value) reached a higher peak (0.93 against 0.85, CI +0.01 to +0.16; +0.04 to +0.09 over the four, significant in
  two) and average; salary dumps won no title in 12 windows against 20 % for the middle (CI -35 to -8 points). The
  groups are small: 8-14 teams each in the main definition, 14-26 in thirds.
* **Champions pay, and buy at the end.** 24 of the 25 MLB champions since 2001 had a payroll above the season's median
  (the exception: the 2003 Marlins, at 72 % of the league average; average standing 0.77, 0 the cheapest team and 1
  the dearest; p < 0.0001); NBA champions 18 of 25 (0.70, p 0.04); in the capped leagues, about half: NHL 7 of 12
  since 2014 (0.66), NFL 7 of 12 since 2013 (0.59). In the three seasons up to the title, MLB champions added more
  present value than almost every other team (11 of 12 above the season's median, p 0.006) and NHL champions gave up
  more future value than almost everyone (13 of 14 at or below the median, p 0.002: picks and prospects spent at the
  deadline); NBA champions neither. In the seasons t-10 to t-3 champions had not torn down more than anyone (standing
  0.46-0.57 in present value moved).
* **So, league by league.** MLB: the rich spend, sign and absorb salary while the poor shed it; spending wins (24 of
  25 champions paid above the median), going for it lifts the peak, and tearing down does not. NBA: the market teams
  sign the stars, and a genuine teardown (present value given up for future, not a dump) is the one path in any league
  with a higher peak, as bottoming out was in the first pass, at the price of a lower low. NHL and NFL: the caps
  squeeze payrolls to within a tenth of each other and valuable clubs spend no more than anyone (NHL +0.33, NFL -0.12,
  neither significant); NHL teardowns peak lower, and NHL champions spend their future at the deadline.

## Caveats

* Values are today's, after the seasons they are compared with; winning raises value, so value cannot be read as the
  cause. Market size is the one measure winning does not move.
* The premium measures value beyond market size, which is more than fan passion (stadium deals, ownership, brand).
* MLB picks could not be traded before 2012 and prospects traded are not measured, so MLB's win-now signs rest on age.
  (The second pass measures them: prospects in Retrosheet's trades, by age and track record.)
* Second pass: the first classification judged a player by his last season, so a star back from injury was undervalued
  and a one-year wonder overvalued, and it ignored age, control and salary (hence the second). NFL value is pay, not
  play (no season-by-season production measure was reachable): a traded player's cap share, a free agent's new deal.
  NHL free agents (2010 on) and payroll (2014 on) are shorter series, so early NHL windows rest on trades alone; MLB's
  moves stop in 2021 (Retrosheet), its payroll does not, and its player salaries stop in 2016, so later MLB dumps are
  not flagged.
* By horizon: a veteran's tenure with the team that got him is known only afterwards; it stands in for his contract,
  but a team that flips him again cuts it short. The prospect and pick averages come from the same seasons they are
  used on. Value is production, not production net of cost, so the method says what a trade did to a team's present
  and future, not who won it. A team that tears down is often one already sliding with an aging roster; the adjusted
  gaps hold the starting rank, the slide into it and roster age fixed, not everything.

## The video

The video tells the first pass (it predates payroll, free agents and trades).

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
