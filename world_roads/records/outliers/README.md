# The 10 Biggest Statistical Outliers in Sports History

A 10:07 countdown film of the locked greatest-seasons ranking (model `E3_main_env`, from `../followup/`). It
opens with a 30-second hook (all 4,940 scores pour into a pile; ten break away), goes straight into the
countdown, then explains the score (grading on a curve, three fairness rules), the women's throws, and how stable the
order is.

The finished video is `videos/sports/biggest_statistical_outliers_10m07s.mp4`. YouTube copy (titles, description,
chapters, tags, pinned comment) is `videos/copy/outliers_YOUTUBE.md`, with thumbnails in `videos/sports/thumbnails/`.

## Files

| File | What it does |
|---|---|
| `facts.py` | Computes every on-screen number from the locked follow-up outputs and data, and writes `facts.json` |
| `facts.json` | The only data the film reads |
| `film.py` | The sections: hook; ten cards; how the score works (grading on a curve, three rules); women's throws; close; sources and photo credits |
| `scenes.py` | Each card's look: team or country colours, the toned cut-out portrait, and the sport-specific chart (pucks, ball bins, bats, K's, game tiles, basketballs on court lines, the javelin sector, the long-jump pit), always with the curve and the step ruler |
| `photos.py` | Portraits from Wikimedia Commons (free licences only; robots.txt-compliant pages, 5 s apart): `fetch`, `alternates`, `crop` |
| `photos/` | The cropped portraits, `cut/` (backgrounds removed with rembg), `credits.json` and `CREDITS.md` (author, licence, link, changes) |
| `score.py` | The music: felt piano and pad in D major, one bar per 3 s, synthesized with `../music.py` and `../vernier/music.py` |
| `assemble.py` | Crossfades the sections (0.6 s), lays in the score (EBU R128, −16 LUFS), writes `out/outliers_final.mp4` |
| `render_all.sh` | Renders every section that isn't already in `out/` |
| `thumbs.py` | Thumbnails A/B/C |
| `VO_SCRIPT.md` | Optional narration, timed to the cut |
| `FACT_CHECK.md` | Each on-screen claim, its value and source, and what still needs a primary-source check |

## Build

```
python facts.py           # facts.json (needs ../followup outputs from run_all.py)
python photos.py crop     # photos/ (after photos.py fetch)
./render_all.sh           # out/<section>.mp4, 1080p 30 fps; about 18,200 frames
python assemble.py        # out/outliers_final.mp4
python thumbs.py          # ../../../videos/sports/thumbnails/
python film.py stills     # review stills in out/stills/
```

## Editorial choices

- **Straight to the list.** The hook carries the scale (a typical leader scores about 0; only 12 of 4,940 reach 5)
  and card #10 labels its own chart (a bin, the curve, the ticks). The longer explanation comes after #1.
- **Team colours.** Each card uses the colours of the team or country the athlete is most associated with in that
  season: Mariners, Yankees, Pistons, Warriors (Curry and Wilt: the same franchise), Brooklyn Robins (an approximation
  of a 1924 palette), Czech Republic, Oilers, United States. Portraits are toned in those colours.
- **Sport-specific charts are the same data.** Every icon stack is drawn to scale from zero, so the stack heights are
  the values; the curve and the ticks are the model's (`ruler` in `facts.json`).
- **How the score is explained.** It's grading on a curve. A class test comes first: a 95 is 5 steps above a
  class averaging 70 (give or take 5), but 1 step above a class averaging 90. Then the same idea in sport, with
  Ledecky's 2016 swim: the class is everyone else's best in the 15 seasons either side, the curve is the class
  average, and one step is the usual give-or-take. Three rules follow (inflation for league scoring; rivals, not
  yourself; one entry each), then the scale of all 4,940 scores (a typical leader scores about 0, only 12 reach 5).
- **Every card shows its ruler** in the sport's own units: the curve, the size of one step, tick marks up to the
  athlete, and the arithmetic, e.g. (262 − 218.9) ÷ 8.18 = 5.27. For team sports the other seasons are converted
  into that season's league rate (`ruler` in `facts.json`), which is exactly the locked model's comparison, rescaled.
- **The order is the locked ranking**, with each card's score taken from `rankings_sensitivity.csv` and not
  recomputed.
- **The women's throws were not excluded.** They were scored like every other event. Their record holders' best
  seasons score 2.4 (shot put) and 3.2 (discus) because the whole 1980s field threw marks like them. The segment
  shows that with observed facts (the last 22 m shot put was in 1988; there were 50 discus throws of 72 m or more
  by 11 women in the 1980s). It uses no return periods and names no one in connection with drug testing.
- **The production pack's wording rules are followed:** no "proved", "greatest athlete", "once every N seasons" or
  "statistically certain"; Beamon is one jump and DiMaggio a streak; seasons are named by their ending year; and the
  adjustment is never said to cover minutes, possessions, ice time or innings.
- **The close keeps three things separate:** the specification ensemble (four seasons in the top ten under all 16
  specifications), resampling ("places 5 to 10 change hands often") and changes of question (all-seasons rarity, no
  league adjustment).

Sources: Baseball-Reference, Basketball-Reference, Hockey-Reference, Pro Football Archives, nflverse, World
Athletics, World Aquatics, alltime-athletics.com (Peter Larsson), Retrosheet.

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may
contact Retrosheet at www.retrosheet.org.
