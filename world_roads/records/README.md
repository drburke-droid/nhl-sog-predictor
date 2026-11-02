# The Record and the Rest

A 7:56 data film: every world record in 34 Olympic athletics events (men's and women's sprints, hurdles,
middle and long distance, jumps and throws), against the top eight of every Olympic final since 1896.

It shows how much each event's record has improved since the first Games (men: since 1912, when records
became official), how much the Olympic field has improved, and how close the field gets to the record.
The finished video is `videos/athletics/the_record_and_the_rest_7m56s.mp4`. YouTube copy is in
`videos/copy/records_YOUTUBE.md`, and method and sources are in `ACCURACY.md`.

**Findings**
* The middle of the men's Olympic final ran, jumped and threw at 95.7% of the world record in 1912 and
  95.9% in 2024.
* The record gained more than the field in 22 of 34 events, but rarely by much.
* The widest gaps are in the women's throws: 85% of the 1987–88 records in the discus and shot put.
  Seven women's records date from 1983–88; one men's record does.

| File | What it does |
|---|---|
| `fetch.py` | Downloads and caches the Olympedia result pages and the Wikipedia progression pages (`cache/`, not committed) |
| `wikitables.py` | Reads Wikipedia tables, keeping the section and row shading (ratified, not ratified, rescinded) |
| `build_records.py` | Builds the world-record lines → `tables/wr_progression.csv` (log of every row kept and dropped: `tables/build_records.txt`) |
| `parse_olympics.py` | Reads each Olympic final's top eight → `tables/olympic_top8.csv` |
| `analysis.py` | Computes the baselines, improvements and gaps → `tables/summary.csv`, `tables/games.csv` |
| `look.py` | Colours, type and the parallel frame renderer (matplotlib → ffmpeg) |
| `film.py` | The sections: the race, the explainer, 34 event cards, the comparisons, the close |
| `music.py` | The synthesized score (felt piano and pad, borrowed from `../vernier/music.py`) |
| `assemble.py` | Crossfades the sections, lays in the score, writes `out/records_final.mp4` |
| `thumbs.py` | Thumbnails A/B/C |
| `render_all.sh` | Everything after the data: `./render_all.sh` (1080p) or `./render_all.sh 2` (4K) |

**Out of Reach**, the follow-up analysis: which records sit far above today's best, whether one athlete or
a whole era put them there, and which events have plateaued (with the men's long jump as the spotlight).
Built on Peter Larsson's all-time lists (alltime-athletics.com, about 164,000 legal performances); the page
is `report.html`, built from `report_page.html` by `report_data.py`.

| File | What it does |
|---|---|
| `alltime.py` | Parses the cached all-time lists (main and indoor sections; wind-aided lists as context) → `cache/alltime_parsed.csv` |
| `deep.py` | Season leads, season depth, record premiums, Gumbel return periods, depth peaks → `tables/seasons.csv`, `tables/deep_summary.csv` |
| `report_data.py` | Groups the records (era record, lone outlier, within reach), builds `tables/deep_report.json` and `report.html` |

The page also charts every athletics event and 20 records from baseball, basketball, hockey and football
(the record against the middle of each season's top ten, and how often today's best would match the
record), and ranks the greatest performance of all time adjusted for its era (Bob Beamon's 8.90 m, 1968).

| File | What it does |
|---|---|
| `sports_fetch.py` | Fetches Baseball-, Basketball- and Hockey-Reference leaders, Wikipedia's NFL leaders, nflverse season stats, World Athletics season lists, and Retrosheet game logs (hitting streaks → `tables/hitting_streaks.csv`) |
| `bigfour.py` | One shape for all 54 categories: smoothed medians, return periods at today's level (counting stats scaled to today's schedule), era-adjusted rarity of every season-leading performance → `tables/allsports.json`, `tables/era_adjusted.csv` |

The hitting streaks use Retrosheet data: the information used here was obtained free of charge from and is
copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

Data tables are rebuilt with:

```
python fetch.py olympics && python fetch.py records
python build_records.py > tables/build_records.txt && python parse_olympics.py && python analysis.py
python fetch.py alltime && python alltime.py && python deep.py
for j in wa sref sref_leagues nflwiki nflverse streaks; do python sports_fetch.py $j; done
python bigfour.py && python report_data.py
```
