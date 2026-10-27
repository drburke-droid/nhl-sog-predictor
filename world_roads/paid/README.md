# The best-paid athletes, 1913-2025, in 2025 dollars (`paid/`)

Two bar-chart races of players in MLB, the NBA, the NFL and the NHL, every salary converted to 2025 dollars, so Babe
Ruth's $80,000 of 1930 and Dak Prescott's $60 million of 2025 stand on one scale:

* **The yearly race**: the ten best-paid each year since 1913. Film: `../media/paid/best_paid_athletes.mp4`.
* **The career race**: the ten biggest careers to date each year, every season's pay in 2025 dollars added up, retired
  players keeping their totals until someone passes them. Film: `../media/paid/best_paid_careers.mp4`.

Both 1920x1080, 30 fps, about 2 min 54 s, with an original synthesized score.
Data: `../media/paid/salaries.csv` (every salary found, both dollars), `top.csv` (the 15 best-paid each year),
`coverage.csv` (salaries found per league and year, and the CPI factor), `careers.csv` (every season counted in a
career, documented or estimated, with running totals), `career_top.csv` (the 40 biggest careers to date each year).

<!-- FINDINGS -->

## What counts as a salary

Each league's pay is reported a different way, and the race takes each league's own convention:

| League | Figure | Source | Seasons |
|---|---|---|---|
| MLB | the season's salary | Baseball Reference player pages (for the early years Michael Haupert's research of the Hall of Fame's contract cards), the Lahman database for 1985-2016 | 1913-2025 |
| NBA | the season's salary (a season counts for the year it ends) | Basketball Reference team pages, every roster | 1984-85 to 2024-25 |
| NFL | the average per year of the contract the player was playing under | OverTheCap's contract histories (nflverse), for the seasons the player was on that team | 1983-2025 |
| NHL | the cap hit, the contract's average per year | CapWages | 2009-10 to 2024-25 |
| all four, before those | the documented contracts of the stars | `curated.csv`: each row with its source's URL and a quote | 1913 on |

* **Deferred money** counts at its present value, as Baseball Reference lists it: Shohei Ohtani's $70 million a year,
  $68 million of it paid from 2034 to 2043, is $28.2 million a year.
* **An NFL contract counts from the season it was signed**, as OverTheCap's average per year and the NFL's
  "highest-paid" rankings report it: Patrick Mahomes's extension of July 2020 ($45 million a year) counts for 2020,
  though its new money began in 2022. (The career race uses the cash actually paid each season instead.)
* **Bonuses** are spread over the contract, as the NFL's and NHL's averages already do. Sergei Fedorov's 1997-98
  offer sheet paid $28 million that season, nearly all of it bonus. It counts as up to $38 million over six years,
  $6.3 million a year. NFL cash-paid figures, which put a whole signing bonus in one season, are not used.
* **Rival leagues count**: the AFL, AAFC, WFL and USFL in football, the ABA and its predecessors in basketball, the WHA in
  hockey. Each bar names its circuit (Joe Namath "Jets · AFL").
* **Not counted**: endorsements, boxing purses, prize money (golf, tennis, racing) and leagues outside North America.
  The race is about what teams paid their players.

## Inflation

The Bureau of Labor Statistics' consumer price index for urban consumers (CPI-U), annual averages, from the Federal
Reserve Bank of Minneapolis' table (1913: 9.9; 1930: 16.7; 2025: 321.9). A salary in year y is worth
salary × CPI(2025) / CPI(y) in 2025 dollars; the factor shows under the year counter ("$1 in 1930 = $19.28 in 2025").

## How the documented contracts were chosen

The NBA's, NFL's and NHL's systematic series start in the 1980s or later. Before that, the race uses the stars'
contracts as published. Three research passes turned up 247 figures, each with its source and a quote. They are kept,
conflicts included, in `research/`. `curate.py` picks one figure per player and season:

* a contract document, a contemporary wire or newspaper report, or a biography beats the 2010 APBR forum compilation
  of star contracts, which lists terms without sources; the compilation is used where nothing else turns up;
* a contract's average is spread over the seasons it covered;
* a figure only one secondary source reports, with no contract terms, is left out (Ernie DiGregorio's $500,000 of
  1973-74);
* a season no source covers is left empty rather than guessed.

So before the 1980s the NBA, NFL and NHL boards hold only stars whose contracts were reported. That is the group that
could make a top ten, but a well-paid player whose deal went unreported is missing.

## Coverage

<!-- COVERAGE -->

## The career race

A career's earnings are the sum of its seasons' pay, each converted with its own year's CPI (`lifetime.py`). The pay of
a season is the yearly race's figure, with two differences, because a career adds up the money as it was paid:

* **NFL: cash paid** that season (OverTheCap's season histories, 1983 on: salary and every bonus in the season it was
  paid), not the contract's average per year.
* **Careers begin at their first season**, before 1913 too: Baseball Reference's salaries back to the 1870s, converted
  with the Minneapolis Fed's CPI estimates for 1800-1912 (spliced to the CPI-U at 1913). So the 1913 board holds Honus
  Wagner's and Nap Lajoie's whole careers, not one season of them.
* NHL cap hits count from 2005-06 (the yearly race starts them in 2009-10, when every contract is in CapWages; a career
  needs the long contracts already running before then).

**Whole careers before the leagues' series.** The documented contracts the yearly race uses were researched for the
peak seasons; a career needs every season. A second research pass collected season-by-season pay for the stars who
could reach a career top ten, each figure with its source and a quote: `research/careers_nba.csv` (150 seasons),
`careers_nfl.csv` (100), `careers_nhl.csv` (321), and `spans_*.csv`, each player's seasons and the ones he sat out or
wasn't paid for. `curate.py` takes them after the yearly selection, which keeps its figures unless the research found
a better source:

* corrected: Joe Namath 1965-67 ($75,000 a season: Phil Berger's 1969 biography itemizes the famous $427,000, of which
  $25,000 a season and a $200,000 bonus were his pay and $127,000 went to a lawyer, a car and scouting jobs for
  relatives), Warren Moon 1991-92 (the NFLPA survey), Rick Barry 1970-71 ($165,000 with the Nets), Walt Frazier
  1974-75, Bob McAdoo 1980-81, Eric Lindros 1997-98 ($7.5 million under his revised deal), Phil Esposito 1976-79, Wayne
  Gretzky 1996-98;
* removed: Bobby Orr's 1977-78 salary, which he refused (he sat the season out and "never cashed a Chicago paycheque");
* dropped: a magazine writer's own estimate (Jerry West 1965-66), a superseded contract schedule (Gretzky 1986-87);
* salaries reported in Canadian dollars converted at the Federal Reserve's H.10 rates (the average of the two calendar
  years a season spans; 1971 on).

**Seasons with no documented pay.** A season a player played between two documented ones is estimated: his salary
rises (or falls) by the same percentage each season from one documented salary to the next. A straight line would put
too much in the seasons just before a big raise (O.J. Simpson's 1973-75, between his $53,750 rookie salary and the
$833,333 of 1976). MLB seasons played come from the Lahman database's appearances; in the other leagues the seasons
between two documented ones are taken as played, except those the research found a player sat out, the years between
his stints, and the 2004-05 lockout. Seasons before a player's first documented salary or after his last are left
out, not guessed, so a total can only be too low, never padded. Every estimated season is marked in `careers.csv`, and
a career more than a fifth of whose money is estimated carries a "≈" on its bar.

**Limits.** Some careers before the series remain thin: the research ran out of searches before Johnny Unitas, Fran
Tarkenton, Terry Bradshaw and Jim Brown, none of whom reached a career top ten on what was found. Career pay excludes
the same things as the yearly race (endorsements, other sports, leagues outside North America).

<!-- CAREER FINDINGS -->

## The film

* The ten best-paid of each year, ranked by salary in 2025 dollars. A bar shows the salary in 2025 dollars and, under
  it, the figure in its own year's money.
* Between two years the salaries move in a straight line, and the board re-sorts as they do. A bar passes another
  when its pay does; the passes happen over the middle of the year, so each year's board stands exactly as ranked on
  the year itself. A player arriving rises from under the board; one leaving sinks under it. A year takes 1.4 s.
* The scale follows the leader. While it is small, a dashed line marks today's MLB minimum salary ($760,000). By that
  line, the best-paid stars of the 1920s and 1930s earned, in today's money, about what a rookie on the minimum makes
  now.
* Colour encodes the league: MLB blue, NBA orange, NFL green. These three hues were checked for colour-blind
  separation on this background, and the NHL is a neutral grey. Every bar also names its league in text, so no
  reading depends on colour.
* Captions come from the data (`captions.py`): in the yearly race the first salary on record to reach $100,000, $1
  million, $10 million, $25 million and $50 million, the record in 2025 dollars, the long reigns at No. 1, and two
  documented moments, each checked against the data before use; in the career race each new leader, the first careers
  past $10 million, $100 million, $250 million and $500 million in 2025 dollars, and the long reigns.
* In the career race a retired player's bar names his league and seasons ("MLB · 1914–1935") instead of a team, and
  under its figure, his seasons and the money as it was paid.
* The score is synthesized from scratch with the instruments of `../flights/score.py`: 85.7 BPM, one year every two
  beats; it rises a whole step when free agency arrives, and its energy follows the leader's pay on a log scale.

## Run it

```
python fetch.py a        # CPI table, Baseball Reference value pages 1913-2000, Basketball Reference team pages 1985-2000
python candidates.py     # whose Baseball Reference salary tables to read -> data/candidates.txt
python fetch.py b        # those pages (~2,300; about 2.5 hours at Sports Reference's limit), and any NBA pages left
python fetch.py c        # the CPI estimates for 1800-1912, the Federal Reserve's US$/C$ rates since 1971
python curate.py         # research/*.csv -> curated.csv (one documented figure per player and season)
python salaries.py       # -> ../media/paid/salaries.csv, top.csv, coverage.csv
python lifetime.py       # -> ../media/paid/careers.csv, career_top.csv
python captions.py       # -> captions.json (milestones and documented moments, from the data)
python captions.py --career                # -> captions_career.json
python assemble.py render                  # render, score, mux -> ../media/paid/best_paid_athletes.mp4
python assemble.py --career render         # -> ../media/paid/best_paid_careers.mp4
```

`python video.py [--career] still 1930 1998.5` renders stills.

The rebuild study's cached pages (`../rebuild/data`: Basketball Reference team pages 2001-2025, Baseball Reference value
pages 2001-2025, the Lahman package, nflverse contracts, CapWages signings) are reused, not fetched again. Sports
Reference asks for no more than 20 requests a minute: `fetch.py` starts requests at least 3.5 s apart across all of
its threads and never asks for a cached page again.
