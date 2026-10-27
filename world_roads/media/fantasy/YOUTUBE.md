# YouTube copy: the greatest fantasy seasons

A bar-chart race of the most valuable single seasons in fantasy football, 1980-2025. The value is points over
replacement in a 12-team PPR league, explained in plain words in the first 39 seconds.

**Upload `fantasy_best_seasons.mp4`:**
* 3:16, 1920×1080, 30 fps, H.264;
* on-screen text, no narration;
* original synthesized score at about −14 LUFS.

**A/B/C test.** Each thumbnail is 1280×720 and built by `../../fantasy/thumbs.py` from the same data as the film.

| | Thumbnail | Title |
|---|---|---|
| A | `fantasy_thumb_A.png`: Faulk '00 and Tomlinson '06, 329.7 and 329.6, "0.1 POINTS APART" | The Greatest Fantasy Football Season Ever Was Decided by 0.1 Points |
| B | `fantasy_thumb_B.png`: "WHO'S #1?" over five famous seasons with their values hidden | Ranking the Greatest Fantasy Football Seasons Since 1980 |
| C | `fantasy_thumb_C.png`: "FANTASY POINTS LIE", Josh Allen's 396 points against Derrick Henry's 333, split into the free player's share and what each added | Fantasy Points Lie: The Real Greatest Fantasy Seasons Since 1980 |

* **A** gives the answer's hook (two famous seasons a tenth of a point apart) without saying which one wins.
* **B** is the plain curiosity gap.
* **C** sells the idea behind the ranking.

**Description**

```
Who had the greatest fantasy football season ever? Total points can't tell you: quarterbacks score the most, but a good one is easy to find on waivers. So every season since 1980 is ranked by what a player scored beyond the best free player at his position, in a 12-team PPR league. In 2020 Josh Allen outscored Derrick Henry 396 to 333, yet Henry added 175 points over a free running back and Allen 135 over a free quarterback.

The board keeps the 12 most valuable seasons so far. Watch each season's bars grow as it's played, and a great one climb onto the board and stay there until something better knocks it off. The top two end up a tenth of a point apart.

How it's counted
• PPR scoring: 1 per catch, 1 per 10 rushing or receiving yards, 1 per 25 passing yards, 6 per TD (4 passing), −2 per interception or lost fumble. Regular season only.
• The "free player" each season is about the 16th-best QB, 30th RB, 42nd WR and 14th TE: what's left once a 12-team league has filled its starting lineups (QB, 2 RB, 3 WR, TE, flex).
• Counted by the game: a game a player misses counts nothing, because his manager starts a free player that week.

Data: Pro Football Reference season tables 1980–1998 (via fantasydatapros/data on GitHub) and nflverse weekly player stats 1999–2025. Fumbles lost are estimated before 1994; two-point conversions and return TDs count from 1999. Kickers aren't included.

Chapters
0:00 The greatest fantasy season ever?
0:13 Points over replacement, explained
0:23 Who is the free player? How to read the race
0:39 1980s
1:33 1990s
2:08 2000s
2:35 2010s
2:52 2020s

Music: original, synthesized for this video.
```

**Notes on the upload.**
* **Chapters.** YouTube needs each chapter to last at least 10 s, so the title and the "why total points lie" scenes
  share the first chapter. The closing summary (3:07, 9 s) is part of the 2020s chapter.
* **Pinned comment.** A good one is the full top 12 from the closing board, as a list, with the question "Which season
  is missing?" The full table is in `seasons.csv`.
