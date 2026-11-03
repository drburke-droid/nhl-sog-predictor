# The 10 Biggest Statistical Outliers in Sports History

A 9:46 countdown film of the locked greatest-seasons ranking (model `E3_main_env`, from `../followup/`). It
covers the ten season-leading performances that stood furthest above everyone else of their era, a method
explainer, a segment on why the women's throws aren't on the list, and a closing look at how stable the order is.

The finished video is `videos/sports/biggest_statistical_outliers_9m46s.mp4`. YouTube copy (titles, description,
chapters, tags, pinned comment) is `videos/copy/outliers_YOUTUBE.md`, with thumbnails in `videos/sports/thumbnails/`.

## Files

| File | What it does |
|---|---|
| `facts.py` | Computes every on-screen number from the locked follow-up outputs and data, and writes `facts.json` |
| `facts.json` | The only data the film reads |
| `film.py` | The sections: cold open, method (Ledecky 2016 as the worked example), ten cards, women's throws, close, sources |
| `score.py` | The music: felt piano and pad in D major, one bar per 3 s, synthesized with `../music.py` and `../vernier/music.py` |
| `assemble.py` | Crossfades the sections (0.6 s), lays in the score (EBU R128, −16 LUFS), writes `out/outliers_final.mp4` |
| `render_all.sh` | Renders every section that isn't already in `out/` |
| `thumbs.py` | Thumbnails A/B/C |
| `VO_SCRIPT.md` | Optional narration, timed to the cut |
| `FACT_CHECK.md` | Each on-screen claim, its value and source, and what still needs a primary-source check |

## Build

```
python facts.py           # facts.json (needs ../followup outputs from run_all.py)
./render_all.sh           # out/<section>.mp4, 1080p 30 fps; about 17,600 frames
python assemble.py        # out/outliers_final.mp4
python thumbs.py          # ../../../videos/sports/thumbnails/
python film.py stills     # review stills in out/stills/
```

## Editorial choices

- **The order is the locked ranking**, with each card's score taken from `rankings_sensitivity.csv` and not
  recomputed.
- **Team-sport charts use an intuitive unit:** each season's best as a share of an average team's total. The score
  is the model's own, which uses an estimated elasticity rather than a straight share, and each card says so.
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
