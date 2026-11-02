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

Data tables are rebuilt with:

```
python fetch.py olympics && python fetch.py records
python build_records.py > tables/build_records.txt && python parse_olympics.py && python analysis.py
```
