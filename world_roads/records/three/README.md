# Three companion films

Three countdowns that use the season film's look and machinery (`../outliers/film.py`, `scenes.py`, `orchestra.py`):

| Film | Title | Entries | Model |
|---|---|---|---|
| A | The 10 Biggest Single-Game Outliers in Team Sports | Wilt's 100 … Derrick Thomas's 7 sacks | `../singlegame/topk.py` (REVIEW_A.md) |
| B | The 10 Biggest Single Performances in Athletics and Swimming | Beamon … Bolt | `../singlegame/review_B.py` (REVIEW_B.md) |
| C | The 5 Biggest Season-Long Outliers in Team Sports | Wilt, DiMaggio, Gretzky, Rodman, Ichiro | `../outliers/review_C.py` (REVIEW_C.md) |

All three use the clear-of-the-field rule: an entry counts only if it beats every other entry on its own graph by at
least one step. Film C's five are the seasons in the top ten of at least 12 of the 16 versions of the method; it also
says why the rest of the top ten by score missed. Films A and B are the top ten by score.

| File | What it does |
|---|---|
| `facts_a.py`, `facts_b.py`, `facts_c.py` | Every number each film shows, into `facts_<film>.json` (the films read only these) |
| `afilm.py`, `bfilm.py`, `cfilm.py` | The sections (hook, cards, how the score works, the rules, how sure the list is, close, credits) |
| `facts_intro.py`, `intro.py`, `opening.py` | The opening of each film (why performances can't be compared across eras as they stand, and how the score allows for it): world records since 1950, the same number in different eras, step 1 (every performance converted to its season's scoring level: hockey's 1980s), step 2 (compared with the best of everyone else in the 15 seasons either side; the score in standard deviations), and in film B the women's throws of the 1980s. `opening.py` voices the lines first and times every scene, caption and animation to them |
| `common.py` | The shared board ("16 ways to rank"), closing line, credits pages, the score's timeline and the assembly |
| `photos3.py` | Portraits from Wikimedia Commons for the athletes the earlier films didn't need (free licences only; GFDL-only files are passed over) |
| `VO_A.md`, `VO_B.md`, `VO_C.md` | The narration, timed to each cut |
| `render.sh` | Renders a film's missing sections: `./render.sh a` |
| `copy3.py` | YouTube copy (titles, description, chapters, full photo credits) into `../../../videos/copy/` |

New charts (in `../outliers/scenes.py`): **TopK** for film A (each season's five best games by anyone else as dots, the
curve through them, and the game measured from half a unit below its count, since a count of 7 stands for 6.5 to 7.5),
**Track** for Bolt (the pool chart on a running track), and a drawn football icon.

## Build

```
python facts_a.py; python facts_b.py; python facts_c.py; python facts_intro.py
ELEVENLABS_API_KEY=... python opening.py   # opening_<film>.json and the opening's cues in VO_<film>.md
python photos3.py fetch; python photos3.py crop         # portraits (minute-spaced image requests)
ELEVENLABS_API_KEY=... python -c "import sys; sys.path.insert(0, '../outliers'); import narrate; narrate.build('VO_C.md', 'out_c', 419.4)"
./render.sh c && python cfilm.py assemble                # out_c/film_c_final.mp4 and film_c_final_voiced.mp4
```

The voiced copy puts the narration over the score as a steady bed about 18 dB under the voice, with no ducking (a slow level-rider and a peak limiter keep the music's own swells and hits even), normalised to −16 LUFS. To remix only the sound: `python <film>film.py voiced`.
Mark Whiten, Derrick Thomas and Gretchen Walsh have no freely licensed photo, so their cards run without one.
