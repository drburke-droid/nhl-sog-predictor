# The 10 Biggest Single-Day Outliers in Sports History

A 9:49 companion to `../../outliers/`. It uses the same method and the same look, but every entry is one game, race,
jump or throw. The data and model are in `..` (`build_data.py`, `score.py`, variant `big`); this folder makes the film.

| File | What it does |
|---|---|
| `facts_sg.py` | Computes every on-screen number into `facts_sg.json`: the top ten (rulers in the day's own units, next best, records before and after, game details), the hook's distribution, the Sittler segment, both lists for the close |
| `sdfilm.py` | The sections. It reuses `../../outliers/film.py` and `scenes.py` (hook, cards, explainer) with this film's facts, palettes and photos, and adds the Sittler segment, the season-against-day close and the end card |
| `photos_sg.py` | Portraits for the athletes the season film didn't need, from Wikimedia Commons (Gretchen Walsh has no free photo, so her card has none): a polite fetch (curl, minute-spaced image requests, standard 500 px thumbnails), 4:5 crops, rembg cut-outs, `photos/credits.json` |
| `sd_render_all.sh`, `sd_assemble.py` | Render every section; join them with the orchestral score (`../../outliers/orchestra.py`, finale at 33 s) |
| `VO_SCRIPT.md` | Optional narration, timed to the cut |

New chart types (in `../../outliers/scenes.py`):
- **Pool:** where everyone else would be when the champion touches the wall, used for Walsh and Marchand.
- **Vault:** crossbars at each season's best height, used for Bubka.
- **Shot:** the shot-put sector with landing dents (no longer used: Oldfield left the list).
- **Pit and javelin:** generalized from the season film, used for Rojas, Beamon, Železný and Vetter.
- **Basketball stacks:** five points a ball for Wilt and Kobe; two assists a ball for Cousy, whose #10 card also
  explains how to read the charts.

Editorial choices: only team stats that come in big numbers are ranked; the Sittler segment explains why. A day must
also stand at least one step clear of every other mark on its own graph (fourth rule in "fair"; see `../README.md`), which
takes out Ledecky's 2016 800 m and Oldfield's 1975 shot put. Marchand's 2025 and Walsh's 2026 have only past seasons to
compare with; Cousy's 1959 has three seasons before it (box scores start in 1955-56). Cousy's 28 was matched by Guy
Rodgers in 1963, when teams averaged more assists; converted to 1959's rate that is about 25, and the card says so.
