# The biggest companies in history, in sand (working title)

A Sandglass film (the style of `ancestry_sand/` and `mortality_sand/`; the style guide is the `sandglass`
skill on branch `claude/sweet-faraday-uk5z62`): every company that was ever among the world's most
valuable, from the Dutch East India Company in 1602 to today, as coloured sand on a dark table.
The bigger the company, the more sand. Sand pours in as it grows and drains as it shrinks; companies that
die stay on the table as empty husks, so the film ends as a record of 400 years of capitalism.

Status: data gathered and merged (183 companies, about 16,600 values, 205 deals and failures); the clock, layout
and camera checked in a flat top-down preview (`preview.py`, about 4:24). Nothing has been rendered in Blender.

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

**The table: a city of islands growing outward from the VOC.** The VOC stands at the centre. Every other
company gets an island in its home market's district, a wedge pointing the way that market lies from Amsterdam
(New York to the west, London west-south-west, Paris south-west, Johannesburg south, Riyadh and the rest of
Europe south-east, Moscow, Shanghai and Tokyo to the north-east). Islands are placed in the order companies
first appear, each as close to the centre as its wedge allows, so the city grows in rings like Amsterdam's
canals: the husks of the old companies form the old core, today's giants the outer rings. The camera stays
centred and pulls straight back, so everything earlier stays in view. (A first version laid the districts
out as rectangles in a grid; the early companies ended up in a long thin strip along the back and the camera
had to frame a sliver. The rings fixed that.) Colour is industry (`story.INDUSTRIES`, 13 colours).

**The water is the market, not a closed system.** Market value is not conserved: when a stock falls, the money
does not go to another company, it is gone. So:
- the islands stand in water carrying pale, uncoloured sand that drifts outward from the centre to the edge of
  the table; it is sparse while there is little money on the table and denser as the total grows;
- sand lifts from the water into a growing company (taking its colour as it lands) and drains back when it
  shrinks (losing its colour);
- coloured sand flies directly from one company to another only for a real deal (merger, purchase, break-up,
  spin-off). A direct arc always means a transaction.

**Mergers and splits.** A vessel follows one legal company; a rename only changes its plaque.
- Bought or merged: its sand arcs into the survivor and its island becomes a husk.
- Broken up (Standard Oil 1911, AT&T 1984): the parent keeps its own share (Jersey Standard became Exxon)
  and the rest pours into the children by their first values. Exxon and Mobil pour back together in 1999;
  SBC, one of AT&T's children, buys AT&T in 2005 and takes its name.
- Failed, nationalised or wound up (VOC 1799, East India Company 1874, Penn Central 1970): the sand drains
  into the canal.
- A company appears only once its shares trade (Saudi Aramco in 2019, not before).

**Husks.** A dead company keeps a dark island, its name and years, and a tide mark at the widest its pile ever
was. Islands grow with the widest their pile has been so far (they never show a company's future size).

**Gaps in the data.** Between two known values a company moves at a steady rate, except across a gap of three
years or more where its home market has an index: there it follows the index, tilted to meet both known values
exactly (`story.bridge`). So a company with no 1930s figures still falls in 1930-32 the way its market did.
Indices: the Bank of England's spliced UK share index, monthly from 1709 (its 1709-1823 part is Larry Neal's
average of the Bank of England, East India and South Sea shares), for London, and for Paris, Amsterdam and the
rest of Europe before 1914; Shiller's S&P composite, monthly from 1871, for New York. This is an estimate and
the description must say so.

**Who is on the table:** every listed company that was among the world's 10 most valuable in any year, plus
the parents and children needed for its mergers and splits.

## Files

| File | What it does |
|---|---|
| `research/SCHEMA.md` | the format the research for each era is written in |
| `research/a_to1825/` ... `d_1990_2026/` | first pass by era: companies, values (nominal, with sources), deals, top-10 lists, notes |
| `research/c2_gapfill/` | year-end values 1926-1990 for the US leaders (checked against the CRSP top 10 for 1939, 1949, 1959) |
| `research/e_lives_pre1925/`, `f_lives_post1925/` | each company's whole listed life: early years and afterlives to its end or today |
| `research/g_editor_fixes/` | corrections made while merging, each with its source |
| `research/worklist_*.csv` | the coverage checks that drove the completion passes |
| `money.py` | any value, date and currency to today's US dollars; `python3 money.py` prints checks |
| `data/` | price, exchange-rate and share-index series (sources in `money.py`, `story.py`), and the merged film data |
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

## Decisions taken while merging

- **Bazacle** (Toulouse, from 1372): narration only; at most a fraction of one grain.
- **Royal Dutch Shell is one island** in Amsterdam: Royal Dutch Petroleum's value / 0.6 stands for the group after
  the 1907 union (it owned 60%); Shell Transport's own island ends at the union.
- **US Steel, 2001:** the island follows the steel company, not the legal heir (USX became Marathon Oil, with about
  85% of the value); the drop is marked as a spin-off.
- **Imperial Tobacco** (taken over by Hanson in 1986 for about GBP 2.5bn) and **Imperial Brands** (demerged 1996)
  are two islands.
- **GM** ends in its 2009 bankruptcy; the GM listed in 2010 is a new island. **Dow and du Pont** merge into
  DowDuPont (2017), which splits into Dow, Corteva and DuPont de Nemours (2019).
- **Saudi Aramco** counts at its full market value although only about 1.5% of its shares trade, as every market-cap
  list does. **SpaceX** is in: it listed on 12 June 2026 and closed its first day at about $2.1 trillion.
- Rows a later pass found wrong are dropped with the reason (`build_data.DROP`).

## Still open

- London prices 1721-1825 (East India Company, South Sea Company, Bank of England): the best source, Larry Neal's
  Course of the Exchange data at esfdb.websites.bbk.ac.uk, is blocked by this environment's network allowlist.
  Until then these years follow the Bank of England's UK index (which is built from those three stocks).
- Thin spots: Standard Oil 1912-25, AT&T before 1922, the Banque de France, French railways after 1900, non-US
  companies 1926-1980, IBM before 1954, 1926-27 for most US companies. `python3 build_data.py --report` and the
  gap check in the worklists show where.
