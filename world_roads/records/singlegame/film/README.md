# The 10 Biggest Single-Day Outliers in Sports History

A 9:49 companion to `../../outliers/`. It uses the same method and the same look, but every entry is one game, race,
jump or throw. The data and model are in `..` (`build_data.py`, `score.py`, variant `big`); this folder makes the film.

| File | What it does |
|---|---|
| `facts_sg.py` | Computes every on-screen number into `facts_sg.json`: the top ten (rulers in the day's own units, next best, records before and after, game details), the hook's distribution, the Sittler segment, both lists for the close |
| `sdfilm.py` | The sections. It reuses `../../outliers/film.py` and `scenes.py` (hook, cards, explainer) with this film's facts, palettes and photos, and adds the Sittler segment, the season-against-day close and the end card |
| `photos_sg.py` | Portraits for the six new athletes from Wikimedia Commons: a polite fetch (curl, minute-spaced image requests, standard 500 px thumbnails), 4:5 crops, rembg cut-outs, `photos/credits.json` |
| `sd_render_all.sh`, `sd_assemble.py` | Render every section; join them with the orchestral score (`../../outliers/orchestra.py`, finale at 33 s) |
| `VO_SCRIPT.md` | Optional narration, timed to the cut |

New chart types (in `../../outliers/scenes.py`):
- **Pool:** where everyone else would be when the champion touches the wall, used for Ledecky and Marchand.
- **Vault:** crossbars at each season's best height, used for Bubka.
- **Shot:** the shot-put sector with landing dents, used for Oldfield.
- **Pit and javelin:** generalized from the season film, used for Rojas, Beamon, Železný and Vetter.
- **Basketball stacks:** five points a ball, used for Wilt and Kobe.

Editorial choices: only team stats that come in big numbers are ranked; the Sittler segment explains why. Oldfield's
22.86 m is a professional-circuit mark, never ratified, and the card says so. Marchand's 2025 has only past seasons to
compare with.
