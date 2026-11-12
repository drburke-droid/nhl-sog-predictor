# The biggest companies in history, in sand (working title)

A Sandglass film (the style of `ancestry_sand/` and `mortality_sand/`; the style guide is the `sandglass`
skill on branch `claude/sweet-faraday-uk5z62`): every company that was ever among the world's most
valuable, from the Dutch East India Company in 1602 to today, as coloured sand on a dark table.
The bigger the company, the more sand. Sand pours in as it grows and drains as it shrinks; companies that
die stay on the table as empty husks, so the film ends as a record of 400 years of capitalism.

Status: design and data. Nothing has been rendered in Blender yet; `preview.py` draws a flat top-down
version to check the layout, the clock and the camera.

## Decisions

**Scale: today's dollars, one fixed grain.** One grain = $10 million in today's money (`story.GRAIN_USD_M`),
the same for the whole film. Values are converted with `money.py`: into US dollars at that date's exchange
rate, then adjusted for US consumer prices to today; before 1800, adjusted for UK prices to 1800 first.
(We considered measuring each company as a share of the economy, which makes the old companies much bigger,
and chose plain inflation-adjusted dollars because viewers understand it without explanation.)

**The camera carries the scale.** The first companies are tiny next to today's (the VOC was worth a few
hundred grains at most; today's leaders are worth hundreds of thousands), so the film opens close on the
first grains in Amsterdam, where a few dozen grains fill the screen, and pulls back as money grows. It only
ever pulls back, so the husks stay in view. A pile's width and height grow with the cube root of its sand,
so a company 1,000 times bigger is only 10 times wider and taller, which keeps everything on one table.
(This departs from the house style's still camera in the middle act; the subject needs it.)

**Opening: the VOC in 1602.** The Bazacle mill company of Toulouse (from 1372) has the oldest share prices
anyone has reconstructed, but in today's dollars it would be worth less than one grain, so it is a line of
narration, not a vessel.

**Districts by home market**, laid out like a map: New York on the left; London, Amsterdam and Paris (and
the rest of Europe) in the middle; Riyadh; Tokyo, Seoul, Shanghai/Hong Kong and Taipei on the right.
Colour is industry (`story.INDUSTRIES`, 13 colours). So the viewer sees which industries win and also
capital's centre moving: Amsterdam, London, New York, Tokyo in 1989, and back. Inside a district the oldest
companies stand at the back, so the husks become a ghost city behind today's mountains.

**Rivers: the market, not a closed system.** Market value is not conserved: when a stock falls, the money
does not go to another company, it is gone. So:
- canals run between all the islands, carrying a stream of pale, uncoloured sand; the stream is thin while
  there is little money on the table and widens as the total grows;
- sand lifts from the canal into a growing company (taking its colour as it lands) and drains back into
  the canal when it shrinks (losing its colour); the canals flow toward the front edge of the table;
- coloured sand flies directly from one company to another only for a real deal (merger, purchase,
  break-up, spin-off). A direct arc always means a transaction.

**Mergers and splits.** A vessel follows one legal company; a rename only changes its plaque.
- Bought or merged: its sand arcs into the survivor and its island becomes a husk.
- Broken up (Standard Oil 1911, AT&T 1984): the parent keeps its own share (Jersey Standard became Exxon)
  and the rest pours into the children by their first values. Exxon and Mobil pour back together in 1999;
  SBC, one of AT&T's children, buys AT&T in 2005 and takes its name.
- Failed, nationalised or wound up (VOC 1799, East India Company 1874, Penn Central 1970): the sand drains
  into the canal.
- A company appears only once its shares trade (Saudi Aramco in 2019, not before).

**Husks.** A dead company keeps a dark pedestal, its name and years, and a tide mark on the table at the
widest its pile ever was.

**Who is on the table:** every listed company that was among the world's 10 most valuable in any year, plus
the parents and children needed for its mergers and splits.

## Files

| File | What it does |
|---|---|
| `research/SCHEMA.md` | the format the research for each era is written in |
| `research/a_to1825/` ... `d_1990_2026/` | companies, values (nominal, with sources), deals, top-10 lists, notes |
| `money.py` | any value, date and currency to today's US dollars; `python3 money.py` prints checks |
| `data/` | price and exchange-rate series (sources in `money.py`), and the merged film data |
| `build_data.py` | merges the eras into `data/companies.csv`, `values_today.csv`, `events.csv` |
| `story.py` | grains per company at any date; the grain size; the industry colours |
| `clock.py` | seconds of film per year (fast through quiet centuries, slow for bubbles and the last 40 years) |
| `layout.py` | districts, islands and canals |
| `preview.py` | the flat top-down preview: stills, contact sheet, low-res video |

```
python3 build_data.py --report
python3 preview.py info
python3 preview.py still 1602.3 1720.5 1911.5 1989.9 2026.8
python3 preview.py sheet
python3 preview.py video --size 960x540 --fps 24
```
