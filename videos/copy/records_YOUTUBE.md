# YouTube copy: the records film ("The Record and the Rest")

Every world record in 34 Olympic athletics events, set against the top eight of every Olympic final since
1896. How much the record improved, how much the field improved, and how close the field gets.

**Upload `athletics/the_record_and_the_rest_7m56s.mp4`:**
* 7:56, 1920×1080, 30 fps, H.264;
* on-screen captions, no narration;
* felt-piano score at −16 LUFS.

To rebuild it, run `./render_all.sh` in `world_roads/records`.

**A/B/C test.**

| | Thumbnail | Title |
|---|---|---|
| A | `records_thumb_A.png`: the 1896 champion 20 m behind Bolt, "20 metres" | 130 Years of World Records vs. Everyone Else |
| B | `records_thumb_B.png`: the men's field line, flat under the record, "96%. For 112 years." | The Olympic Field Has Stayed Within 4% of the World Record for 112 Years |
| C | `records_thumb_C.png`: the women's discus record line flat since 1988 above the field, "Untouched since 1988" | The World Records the Olympic Field Can't Catch |

**Description**

```
Run Usain Bolt's 9.58 against the first Olympic 100 m champion, Tom Burke (12.0, Athens 1896), at their average speeds: when Bolt finishes, Burke is still 20 metres back.

But a world record is one person, on one day. What about everyone else?

We followed 34 Olympic events, men's and women's, from the sprints to the hammer. For each we drew the world record over time, set anywhere, and the top eight of every Olympic final since 1896. Then we measured both against the first Games: how much faster, farther or higher the record has got, and how much the field has.

What we found:
• The middle of the men's Olympic final ran, jumped and threw at 95.7% of the world record in 1912, and 95.9% in Paris 2024. The record moves; the field follows.
• The record gained more in 22 of 34 events, but rarely by much. The field gained more in the men's throws and triple jump, the men's 10,000 m, and the women's 200 m and 400 m.
• The tightest event: the Paris 800 m field ran at 99.1% of David Rudisha's record speed.
• The widest gaps are in the women's throws: the Paris discus field threw 84.9% of the 1988 record, and the shot put field 85.2% of the 1987 record. Seven women's world records date from 1983–88; one men's record does.
• The women's javelin is the only event whose field went backwards: the Paris median threw shorter than Sydney 2000's.
• New in 2026: Ja'Kobe Tharp's 12.75 (110 m hurdles), Masai Russell's 12.09 (100 m hurdles), Alison dos Santos's 45.80 (400 m hurdles) and Armand Duplantis's 6.31 m, his 15th pole vault record.

How we measured: performance is speed for running events and distance or height for jumps and throws, so +10% means the same everywhere. The field is the median of each Olympic final's top eight; the record is the world record standing on the day of that final. Men's comparisons start at Stockholm 1912, when the IAAF began ratifying records; women's at each event's first Games. Hand-timed and electronic marks are never mixed in one comparison. Jumps and throws scale with the square of speed, so their percentages run about twice as large as running events'.

0:00 One person, one day
0:29 How to read the charts
1:09 Sprints
2:05 Hurdles
2:44 Middle and long distance
3:58 Jumps
5:12 Throws
6:27 Every event: the record against the field
6:53 How close does the field get?
7:22 How old the records are
7:34 The record and the rest

How it's made: every frame is drawn in code (Python, matplotlib, ffmpeg). Method, sources, every number and every judgement call: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/records/ACCURACY.md

Sources: world-record progressions from World Athletics' lists, via Wikipedia's record-progression pages (November 2026), cross-checked against news reports for the 2026 records; Olympic results from Olympedia (olympedia.org).

Music: an original felt-piano score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 One person, one day
0:29 How to read the charts
1:09 Sprints
2:05 Hurdles
2:44 Middle and long distance
3:58 Jumps
5:12 Throws
6:27 Every event: the record against the field
6:53 How close does the field get?
7:22 How old the records are
7:34 The record and the rest
```

**Tags**

```
world records, athletics, track and field, Olympics, Olympic history, world record progression, Usain Bolt, Mondo Duplantis, Bob Beamon, Florence Griffith Joyner, sports statistics, data visualization, data video, 100 metres, long jump, pole vault, shot put, discus, javelin, hammer throw, hurdles
```
