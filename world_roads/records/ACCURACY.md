# Accuracy disclosure: "The Record and the Rest"

The film compares two things for 34 Olympic athletics events (17 for men, 17 for women):

* **the record**: the world record over time, set at any meet;
* **the field**: how the top eight did in every Olympic final, 1896–2024.

Every number on screen comes from the tables in `tables/`. Those tables are built by the scripts in this
folder from cached copies of the sources (section 6).

## 1. Sources

| What | Source | Fetched |
|---|---|---|
| World-record progressions | English Wikipedia's record-progression pages, one per event. These pages follow World Athletics' (IAAF's) own progression lists and mark records that were not ratified, or ratified and later rescinded | 2 November 2026, via the MediaWiki parse API |
| Olympic results | [Olympedia](https://www.olympedia.org) result pages for every athletics event of every Summer Games, 1896–2024 (808 events, of which 34 series are used) | 2 November 2026 |
| Records set in 2026 | Cross-checked against news reports, because they are newer than any published list | November 2026 |

The 2026 records in the film:
* Ja'Kobe Tharp, 110 m hurdles, 12.75, NCAA Championships, Eugene, 10 June 2026. Some reports
  describe it as pending ratification.
* Masai Russell, 100 m hurdles, 12.09, Zürich, 27 August 2026. Reported as pending ratification.
* Alison dos Santos, 400 m hurdles, 45.80, Zürich, 27 August 2026. Wikipedia marks it pending.
* Armand Duplantis, pole vault, 6.31 m, Uppsala, 12 March 2026: his 15th world record.

## 2. Definitions

* **Performance.** Running events use speed (distance ÷ time); jumps and throws use distance or height.
  So "+10% better" means the same in every event: 10% faster, 10% farther or 10% higher.
* **The record at a Games.** The world record standing on the day of that Games' final. A record set in
  the final itself counts, as Duplantis's 6.25 m and McLaughlin-Levrone's 50.37 did in Paris.
* **The field.** The median of the top eight marks in the Olympic final. When fewer marks were recorded
  it is the median of those, with at least four required; sprint finals before 1964 had six runners.
  Runners who did not finish or were disqualified have no mark. Runners who finished injured keep their
  mark, for example Asafa Powell (11.99, 2012), Hasely Crawford (1:19.6, 1976) and Marie-Josée Ta
  Lou-Smith (13.84, 2024). The median is used so that these finishes barely move it. On the charts, the
  column of dots joins only the marks near the median, and a runner who pulled up stays a lone dot.
* **Baseline ("% better than …").** For men it is the first Games from 1912 on: the IAAF was founded
  in 1912 and began ratifying records. For women it is the first Games the event was held. In each
  case the event must have both a ratified record and a recorded field, so some events start later:
  * men's 200 m from 1952 (the first record on a curve dates from 1951);
  * men's 400 m hurdles and hammer from 1920;
  * men's javelin from 1988 and women's from 2000 (both javelins were redesigned; section 3);
  * women's 100 m hurdles from 1972.
* **Gap.** The field's performance as a percentage of the record's. 100% would mean the median finalist
  matched the record.
* **"To Paris 2024."** Every percentage compares the record standing at the 2024 Games with the 2024
  field. Records set after August 2024 appear on the left-hand charts, the record-age chart and in the
  captions, but not in the percentages.

## 3. How the record lines were built (`build_records.py`)

For each event the script reads the tables of the progression page, as configured in `CONFIG`. It parses
each mark and date, and drops rows that the page's own legend marks as **not ratified** or **ratified and
later rescinded**.

Examples of rows dropped:
* Ben Johnson's 9.83 (1987) and 9.79 (1988);
* Tim Montgomery's 9.78 (2002) and Justin Gatlin's 9.77 (2006);
* Brian Oldfield's 22.86 m shot put (1975, a professional meet).

Every drop is listed in `tables/build_records.txt`. The line itself:

* **Before the first ratified record**, the line follows the page's best known marks. It is drawn
  lighter, and the line restarts at the first ratified record, so baselines use official records. For
  example, the men's 100 m baseline is Donald Lippincott's ratified 10.6 (6 July 1912), not the
  unratified 10.5s run that spring.
* **Hand and electronic timing.** In the sprints and hurdles, the line follows hand-timed records until
  the first electronically timed record, then electronic records only. That is 1968 for the men's and
  women's 100 m and the men's 200 m and 400 m; later for the others. On the charts the hand-timed part is
  lighter. The line steps back at the switch (for example from 9.9 to 9.95), as the IAAF's own lists do.
* **Olympic finals are timed the same way as the record they are compared with.** Olympedia gives hand
  and automatic times side by side for 1932–1968, and the analysis uses one timing per final:
  * automatic times when every finisher has one and the record line is electronic by then;
  * otherwise hand times.
* **Javelins.** The men's javelin was redesigned in 1986 and the women's in 1999, and the records
  restarted. Each card shows the old javelin's record line in faint gold for context. Percentages use
  only the new implement. Men's marks set in 1990–91 with serrated-tail javelins, later annulled, are
  excluded.
* **Men's hammer.** That page gives its key in prose: blue rows were ratified, and the others were never
  submitted or not ratified (`LEGEND` in the script).
* **Shading.** Row shades that differ slightly from the legend's colour are snapped to the nearest
  legend colour. One row was affected (Myrtle Cook's 12.0, women's 100 m, 1928, ratified).

**Not included:**
* the marathon (records were "world bests" until 2004, and course lengths varied at early Games);
* the steeplechase (early Olympic distances varied, and the women's event dates only from 2008);
* the women's 80 m hurdles (a different event from the 100 m hurdles);
* combined events and walks.

## 4. How the Olympic field was read (`parse_olympics.py`)

* Placings 1–8 from each event's summary table. Ties ("=3") count. Athletes eliminated in earlier rounds,
  and AC, DNF, DNS and DQ entries, do not.
* **Marks.** Taken from the final. Where an event had one round, or where the summary has no marks, they
  come from the final's own table.
  * Early finals often record only margins ("at 2 metres"); these are not times and are left empty.
  * Olympedia's estimated times ("est") are kept.
* **Early field events.** In several early Games the qualifying round counted towards the final
  standings, so athletes placed 4th to 8th threw only in qualifying. There each athlete's best mark from
  both rounds is used, as the standings were. This applies only when placed athletes lack a final mark.
* Olympedia's women's sprint hurdles ("100 / 80 metres Hurdles") are split by year: 100 m from 1972.
* **Plausibility sweep.** Every mark was checked against plausible ranges and the record. It caught one
  parsing error, which was fixed: the 1952 women's high jump had read a "misses" column. Everything else
  it flagged is genuine: the 1896 hurdles (17.6), John Taylor's fall in the 1924 400 m (1:07.0 est) and
  Crawford's injured 200 m (1976).
* **Spot checks** against the published results, all matching: Paris 2024 men's 100 m (Lyles 9.784 to
  Seville 9.91), men's 800 m, men's 1500 m and women's shot put; Tokyo 2020 men's 100 m (six finishers,
  after a DNF and a DQ); the 1964 and 1968 sprints, hand and automatic.
* Disqualifications for doping appear as Olympedia shows them: medals and places reallocated, the
  disqualified athlete without a mark.

## 5. What the comparison can and cannot say

* **The record is the best performance by anyone, ever; the field is one competition.** A gap is
  expected. What the film follows is whether that gap grows, shrinks or holds.
* **Olympic distance finals are often tactical**, so they run well below record pace (for example
  Rio 2016's 1500 m, won in 3:50.00). The median is not a measure of what finalists *could* run.
* **Context the numbers do not remove:**
  * altitude in Mexico City (1968);
  * the boycotts of 1980 and 1984. The women's throws dip in 1984 because most of the world's best
    throwers stayed home.
  * hand timing before the 1970s;
  * changes in tracks, poles, shoes and implements.
* **Jumps and throws scale with the square of speed.** Distance grows with the square of take-off or
  release speed, so their percentage gains run roughly twice those of running events. The film never
  compares a jump's percentage with a run's on the same axis.
* **Drug testing.** Most of the 1980s records were set before random out-of-competition drug testing
  became routine in the late 1980s and early 1990s. The film states this as context only: no individual
  is accused of anything, and every record shown stands in World Athletics' books.
* **Summary statistics.** The "median of all events" lines count the Games at which at least a third of
  the events were held. The men's figures quoted in the close (95.7% in 1912, 95.9% in 2024) are those
  medians.

## 6. The numbers

| | Event | From | Record then → at Paris 2024 | Field then → 2024 | Record gain | Field gain | Field as % of record, then → 2024 |
|---|---|---:|---|---|---:|---:|---|
| Men | 100 metres | 1912 | 10.60 → 9.58 | 10.90 → 9.84 | +10.6% | +10.8% | 97.2% → 97.4% |
| Men | 200 metres | 1952 | 20.60 → 19.19 | 20.90 → 20.00 | +7.3% | +4.5% | 98.6% → 95.9% |
| Men | 400 metres | 1912 | 47.80 → 43.03 | 48.40 → 43.83 | +11.1% | +10.4% | 98.8% → 98.2% |
| Men | 800 metres | 1912 | 1:51.90 → 1:40.91 | 1:52.40 → 1:41.88 | +10.9% | +10.3% | 99.6% → 99.1% |
| Men | 1500 metres | 1912 | 3:55.80 → 3:26.00 | 3:57.20 → 3:28.84 | +14.5% | +13.6% | 99.4% → 98.6% |
| Men | 5000 metres | 1912 | 14:36.60 → 12:35.36 | 15:08.70 → 13:15.30 | +16.1% | +14.3% | 96.5% → 95.0% |
| Men | 10,000 metres | 1912 | 30:58.80 → 26:11.00 | 32:21.80 → 26:43.90 | +18.3% | +21.1% | 95.7% → 97.9% |
| Men | 110 m hurdles | 1912 | 15.00 → 12.80 | 15.30 → 13.21 | +17.2% | +15.9% | 98.0% → 96.9% |
| Men | 400 m hurdles | 1920 | 54.00 → 45.94 | 54.80 → 47.76 | +17.5% | +14.7% | 98.5% → 96.2% |
| Men | High jump | 1912 | 2.00 m → 2.45 m | 1.87 m → 2.33 m | +22.5% | +24.3% | 93.5% → 94.9% |
| Men | Pole vault | 1912 | 4.02 m → 6.25 m | 3.80 m → 5.88 m | +55.5% | +54.6% | 94.5% → 94.0% |
| Men | Long jump | 1912 | 7.61 m → 8.95 m | 7.00 m → 8.17 m | +17.6% | +16.7% | 92.0% → 91.3% |
| Men | Triple jump | 1912 | 15.52 m → 18.29 m | 14.12 m → 17.55 m | +17.8% | +24.4% | 90.9% → 96.0% |
| Men | Shot put | 1912 | 15.54 m → 23.56 m | 13.39 m → 21.87 m | +51.6% | +63.3% | 86.2% → 92.8% |
| Men | Discus | 1912 | 47.58 m → 74.35 m | 41.67 m → 67.97 m | +56.3% | +63.1% | 87.6% → 91.4% |
| Men | Javelin | 1988 | 87.66 m → 98.48 m | 82.52 m → 88.11 m | +12.3% | +6.8% | 94.1% → 89.5% |
| Men | Hammer | 1920 | 57.77 m → 86.74 m | 47.29 m → 78.99 m | +50.1% | +67.0% | 81.9% → 91.1% |
| Women | 100 metres | 1928 | 12.00 → 10.49 | 12.30 → 10.96 | +14.4% | +12.2% | 97.6% → 95.7% |
| Women | 200 metres | 1948 | 23.60 → 21.34 | 25.25 → 22.23 | +10.6% | +13.6% | 93.5% → 96.0% |
| Women | 400 metres | 1964 | 51.90 → 47.60 | 54.50 → 49.28 | +9.0% | +10.6% | 95.2% → 96.6% |
| Women | 800 metres | 1928 | 2:16.80 → 1:53.28 | 2:21.90 → 1:57.92 | +20.8% | +20.3% | 96.4% → 96.1% |
| Women | 1500 metres | 1972 | 4:01.40 → 3:49.04 | 4:04.46 → 3:53.06 | +5.4% | +4.9% | 98.7% → 98.3% |
| Women | 5000 metres | 1996 | 14:36.45 → 14:00.21 | 15:11.08 → 14:31.93 | +4.3% | +4.5% | 96.2% → 96.4% |
| Women | 10,000 metres | 1988 | 30:13.74 → 28:54.14 | 31:32.39 → 30:44.81 | +4.6% | +2.6% | 95.8% → 94.0% |
| Women | 100 m hurdles | 1972 | 12.59 → 12.12 | 13.07 → 12.43 | +3.9% | +5.2% | 96.3% → 97.5% |
| Women | 400 m hurdles | 1984 | 53.58 → 50.37 | 55.42 → 52.48 | +6.4% | +5.6% | 96.7% → 96.0% |
| Women | High jump | 1928 | 1.59 m → 2.10 m | 1.50 m → 1.95 m | +31.7% | +30.4% | 93.7% → 92.9% |
| Women | Pole vault | 2000 | 4.63 m → 5.06 m | 4.47 m → 4.80 m | +9.3% | +7.3% | 96.7% → 94.9% |
| Women | Long jump | 1948 | 6.25 m → 7.52 m | 5.56 m → 6.79 m | +20.3% | +22.1% | 88.9% → 90.2% |
| Women | Triple jump | 1996 | 15.50 m → 15.74 m | 14.46 m → 14.63 m | +1.5% | +1.1% | 93.3% → 92.9% |
| Women | Shot put | 1948 | 14.59 m → 22.63 m | 12.95 m → 19.27 m | +55.1% | +48.8% | 88.8% → 85.2% |
| Women | Discus | 1928 | 39.62 m → 76.80 m | 35.71 m → 65.21 m | +93.8% | +82.6% | 90.1% → 84.9% |
| Women | Javelin | 2000 | 69.48 m → 72.28 m | 64.05 m → 63.20 m | +4.0% | −1.3% | 92.2% → 87.4% |
| Women | Hammer | 2000 | 76.07 m → 82.98 m | 68.14 m → 74.14 m | +9.1% | +8.8% | 89.6% → 89.3% |

The record gained more than the field in 10 of 17 men's events and 12 of 17 women's.

The median gap across events (`analysis.py`; a dash where fewer than a third of the events were held):

| | 1912 | 1936 | 1960 | 1980 | 2000 | 2024 |
|---|---:|---:|---:|---:|---:|---:|
| Men, all events | 95.7% | 95.7% | 96.7% | 95.8% | 95.1% | 95.9% |
| Women, all events | – | – | 92.7% | 97.1% | 94.5% | 94.9% |

## 7. Rebuilding

```
python fetch.py olympics && python fetch.py records    # cached under cache/ (not committed)
python build_records.py > tables/build_records.txt && python parse_olympics.py && python analysis.py
./render_all.sh            # frames (matplotlib) -> sections -> score -> out/records_final.mp4
```

Requirements: Python 3 with matplotlib, numpy, scipy, beautifulsoup4 and lxml; ffmpeg with libx264; the
Inter font. The score borrows the felt-piano, pad and room models from `../vernier/music.py`.
