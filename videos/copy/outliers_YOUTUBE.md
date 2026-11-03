# YouTube copy: "The 10 Biggest Statistical Outliers in Sports History"

A countdown of the ten season-leading performances that stood furthest above everyone else of their era. It
covers 74 categories in athletics, swimming, baseball, basketball, hockey and football. The ranking is the locked
E3_main_env model from the greatest-seasons project.

**Upload `sports/biggest_statistical_outliers_9m46s.mp4`:**
* 9:46, 1920×1080, 30 fps, H.264.
* On-screen captions and no narration. A voice-over script timed to the cut is in
  `world_roads/records/outliers/VO_SCRIPT.md`.
* Felt-piano score at −16 LUFS.

To rebuild it, run `./render_all.sh` and then `python assemble.py` in `world_roads/records/outliers`.

**A/B/C test**

| | Thumbnail | Title |
|---|---|---|
| A | `outliers_thumb_A.png`: Beamon's 8.90 m, one gold dot far above every other season's best, "The biggest outlier in sports history?" | The 10 Biggest Statistical Outliers in Sports History |
| B | `outliers_thumb_B.png`: 8.90 m · 50.4 · 56 · 163, "Which stood furthest above its era?" | Beamon, Wilt, DiMaggio, Gretzky: Who Stood Furthest Above Their Era? |
| C | `outliers_thumb_C.png`: Wilt's 50.4 above every other scorer of his era, "Next best: 31.6" | 10 Seasons That Broke the Curve |

**Description**

```
Bob Beamon's 8.90 m. Wilt Chamberlain's 50.4 points a game. Joe DiMaggio's 56-game hitting streak. Wayne Gretzky's 163 assists. Which of these stood furthest above everyone else of its time?

We took every season's leader in 74 categories across six sports (athletics, swimming, baseball, basketball, hockey and football) and compared each one with the best of everyone else in the 15 seasons before and after, around a curved trend. For team sports we also allowed for the league's scoring level. The score is how many "typical gaps" a performance sits above its era.

The countdown:
10. Ichiro Suzuki, 2004: 262 hits
9. Babe Ruth, 1920: 54 home runs
8. Dennis Rodman, 1992–93: 18.3 rebounds a game
7. Stephen Curry, 2015–16: 402 three-pointers
6. Dazzy Vance, 1924: 262 strikeouts
5. Jan Železný, 1996: 98.48 m javelin
4. Wayne Gretzky, 1985–86: 163 assists
3. Joe DiMaggio, 1941: 56-game hitting streak
2. Wilt Chamberlain, 1961–62: 50.4 points a game
1. Bob Beamon, 1968: 8.90 m long jump

What this is not: a ranking of the greatest athletes, or a probability. It measures how far one result stood from everyone else of its time, and the order depends on the method. Four seasons (Beamon, Chamberlain, DiMaggio and Gretzky) were in the top ten under all 16 versions of the method we tested; their order was not. Places 5 to 10 change hands under small changes. Count an athlete's own other seasons in the comparison and Gretzky falls to 20th while DiMaggio rises to 1st. Drop the league adjustment and Dan Marino's 1984 and Katie Ledecky's 2016 enter the top ten.

Notes: seasons are named by the year they end. Beamon's entry is a single jump and DiMaggio's a streak, not season totals. The league adjustment allows for each league's scoring rate; it does not standardise player minutes, possessions, ice time or innings. A correction for the number of teams was tested and left out as unreliable.

Why the women's throws aren't on the list: by one measure the women's shot put and discus records are the most extreme numbers in the dataset, but the whole 1980s field threw marks like them. Measured against their own era, the record holders' best seasons score 2.4 and 3.2; the top ten starts at 5.27. Random out-of-competition drug testing began in athletics in the late 1980s. This video does not single anyone out.

0:00 Four numbers
0:23 How the list was built
1:41 #10 Ichiro Suzuki
2:17 #9 Babe Ruth
2:52 #8 Dennis Rodman
3:27 #7 Stephen Curry
4:03 #6 Dazzy Vance
4:38 #5 Jan Železný
5:14 #4 Wayne Gretzky
5:49 #3 Joe DiMaggio
6:24 #2 Wilt Chamberlain
7:00 #1 Bob Beamon
7:35 Why the women's throws aren't on this list
8:33 How sure is this order?
9:30 Sources

How it's made: every frame is drawn in code (Python, matplotlib, ffmpeg). Data, model, tests and two rounds of independent review: https://github.com/drburke-droid/nhl-sog-predictor/tree/master/world_roads/records/followup

Sources: Baseball-Reference, Basketball-Reference, Hockey-Reference, Pro Football Archives, nflverse, World Athletics, World Aquatics, alltime-athletics.com (Peter Larsson), Retrosheet.

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

Music: an original felt-piano score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 Four numbers
0:23 How the list was built
1:41 #10 Ichiro Suzuki
2:17 #9 Babe Ruth
2:52 #8 Dennis Rodman
3:27 #7 Stephen Curry
4:03 #6 Dazzy Vance
4:38 #5 Jan Železný
5:14 #4 Wayne Gretzky
5:49 #3 Joe DiMaggio
6:24 #2 Wilt Chamberlain
7:00 #1 Bob Beamon
7:35 Why the women's throws aren't on this list
8:33 How sure is this order?
9:30 Sources
```

**Pinned comment**

```
How the score works, in one line: each season's leader is compared with the best of everyone else in the 15 seasons either side, around a curved trend; the score is how far above that trend it sits, in units of the usual scatter. It's a statistical distance, not a probability, and the order depends on choices we show in the video (8:33). Which definition would you choose?
```

**Tags**

```
greatest seasons, sports statistics, statistical outliers, Bob Beamon, Wilt Chamberlain, Joe DiMaggio, Wayne Gretzky, Babe Ruth, Stephen Curry, Dennis Rodman, Ichiro Suzuki, Jan Zelezny, Dazzy Vance, Katie Ledecky, sports history, data visualization, data video, NBA history, NHL history, MLB history, track and field
```

**Before publishing**

- The GitHub link points at `master`. The work is on the branch `claude/wizardly-cannon-vkoo2e` until it is merged.
- The production pack asks for every record to be checked against a primary source. That checklist, with the value
  each was checked against, is `world_roads/records/outliers/FACT_CHECK.md`.
