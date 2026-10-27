# The best-paid athletes, 1913-2025, in 2025 dollars (`paid/`)

A bar-chart race of the ten best-paid players in MLB, the NBA, the NFL and the NHL each year since 1913, every salary
converted to 2025 dollars, so Babe Ruth's $80,000 of 1930 and Dak Prescott's $60 million of 2025 stand on one scale.

## Run it

```
python fetch.py a        # CPI table, Baseball Reference value pages 1913-2000, Basketball Reference team pages 1985-2000
python candidates.py     # whose Baseball Reference salary tables to read -> data/candidates.txt
python fetch.py b        # those pages (~2,300; about 2.5 hours at Sports Reference's limit), and any NBA pages left
python salaries.py       # -> ../media/paid/salaries.csv, top.csv, coverage.csv
python captions.py       # -> captions.json (milestones and documented moments, from the data)
python video.py render   # -> out/race.mp4 (python video.py still 1930 1998.5 for stills)
python assemble.py       # score, mux -> ../media/paid/best_paid_athletes.mp4
```

The rebuild study's cached pages (`../rebuild/data`: Basketball Reference team pages 2001-2025, Baseball Reference value
pages 2001-2025, the Lahman package, nflverse contracts, CapWages signings) are reused, not fetched again.
