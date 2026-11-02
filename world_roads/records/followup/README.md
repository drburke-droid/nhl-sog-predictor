# Greatest seasons: follow-up audit and improved analysis

This is an opt-in analysis that runs alongside the original. It answers the handoff in `handoff/CLAUDE_TASK.md`.
The original method, data and published ranking are preserved and reproduced exactly; nothing here
overwrites them.

## Read first

| File | What it is |
|---|---|
| `audit_followup.md` | Ranked findings: evidence, error or judgment call, effect on the top ten, fix |
| `top10_robustness.md` | The top ten under every model, with bootstrap ranges, the specification ensemble and figures |
| `model_specification.md` | The equations, estimands, assumptions and limits of every model |
| `outputs/sources_and_corrections.csv` | Every data repair, verified fact and unresolved warning, with sources and score effects |
| `outputs/rankings_sensitivity.csv` | Every athlete-season in any variant (8,315 rows, 86 columns): the original score and every labelled variant's z and ranks |

## Commands

You need Python 3.10+, numpy, scipy (for the tail calibration only) and matplotlib (for the figures only).
Run these from this folder (`followup/`):

```
python tests/test_reproduction.py     # 15 checks: original scores to 1e-9, published ranking, handoff numbers, invariances
python analysis/cv_window.py v2       # window/trend cross-validation          -> outputs/cv_window_trend.csv
python analysis/env_field_tests.py v2 # environment elasticities, field size, NBA denominators, Babe Dye
python analysis/beta_diagnostics.py   # fit, ties and coverage of the rank-tail beta
python analysis/simulate.py           # field-size simulations (about 10 s)
python analysis/run_all.py            # every variant, calibration, ensemble, bootstrap (B=300; 20-30 min)
python analysis/run_all.py --quick    #   the same with B=40, about 2 min
python analysis/corrections_log.py    # outputs/sources_and_corrections.csv
python analysis/plots.py              # figures/*.png
python analysis/report.py             # top10_robustness.md
```

`make_all.sh` runs them in order. Every script uses paths relative to its own location.

## Layout

```
data/original/     the inputs exactly as audited (checksums match handoff/FILE_MANIFEST_SHA256.txt)
data/v2/           corrected, ID-keyed season lists (top 40 per season) and the log of what was applied
data/context/      league context: NBA, NHL, MLB, NFL per-team-game rates, team counts, estimated paces,
                   and NBA/NHL/NFL player-season lines (Basketball-, Hockey-, Baseball-Reference,
                   Pro Football Archives, nflverse)
analysis/          gsmodel.py (the model), context.py, calibrate.py and the experiment scripts above
build/             cache-dependent builders (need the original project's download cache, not shipped):
                   fetch_context.py, parse_context.py, build_v2_inputs.py
handoff/           the handoff package as received (scripts and outputs), runnable unchanged
outputs/           every generated table
figures/           every generated figure
tests/             test_reproduction.py
```

## Season convention

Seasons are stored as the calendar year in which they end: 1961–62 is 1962 and 1985–86 is 1986. Tables label them
the same way.

## Model keys (columns of rankings_sensitivity.csv)

| Key | Model |
|---|---|
| `A0_original` | The published method and data |
| `A1_corrected` | Corrected data (stable IDs, logged repairs), every tied leader scored |
| `A1_w7`, `A1_w10`, `A1_w20`, `A1_flat`, `A1_huber`, `A1_noshift` | Window, trend and suit-term sensitivity |
| `A2_main` | **Main peer-dominance model**: corrected data, ±15, quadratic trend (cross-validation choice within ±15) |
| `A2_w7` | ±7 quadratic, the best cross-validated window overall |
| `A2_twosided` | Main model, ranking only seasons with at least 5 comparison seasons on each side |
| `B1_allseasons`, `B2_allseasons_main` | **All-seasons rarity**: the athlete's other seasons stay in the comparison |
| `E1_env_ratio`, `E2_env_power` | Big four divided by the league rate (elasticity 1, or estimated and pooled) |
| `E3_main_env` | **Main model plus the estimated environment** (big four; individual sports unchanged) |
| `F_ratio_field25/50/100`, `_normal`, `_estimated` | Field-size offsets (handoff model extended to MLB/NFL; normal-tail; estimated) |
| `D1_depth` | Athletics and swimming relative to the season's depth (median of ranks 2–10) |
| `T1_spikes` | Athletics running events with a separate 2020–26 level (spike technology) |
| `A2_main_catstd`, `*_cal` | Category-standardised and tail-calibrated versions |

Each key has three columns: `_z` (score), `_rank_all` (rank among all athlete-seasons) and `_rank_page` (rank with
each athlete's best season only; blank for an athlete's other seasons).

## Sources

- **Basketball-Reference:** `leagues/NBA_stats_per_game.html`, `leagues/{BAA,NBA}_{year}.html`,
  `leagues/{BAA,NBA}_{year}_per_game.html`.
- **Hockey-Reference:** `leagues/stats.html`, `leagues/NHL_{year}_skaters.html`.
- **Baseball-Reference:** `leagues/{AL,NL}/{bat,pitch}.shtml`.
- **Football:** Pro Football Archives team pages and nflverse player stats.
- **Athletics:** alltime-athletics.com and the World Athletics season lists.
- **Swimming:** the World Aquatics rankings API.
- **Hitting streaks:** Retrosheet. The information used here was obtained free of charge from and is copyrighted by
  Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

All pages were fetched in November 2026 under each site's robots.txt, at least 3.5 s apart.
