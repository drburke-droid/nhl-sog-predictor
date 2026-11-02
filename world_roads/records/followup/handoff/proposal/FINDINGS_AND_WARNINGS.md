# Prior findings, status and guardrails

Date: November 2, 2026. All scores below are from the supplied data and exploratory scripts and should be independently reproduced and stress tested.

## Reproduced original

- `original_project/audit_recompute.py` reproduced 4,659 scored category-seasons and the published top ranking. Example z: Wilt 1961-62 PPG = 7.731; Gretzky 1985-86 assists = 7.500; DiMaggio 1941 hitting streak = 7.197.
- This confirms internal numerical agreement, not correctness of underlying data or inferential validity.

## Preliminary NBA/NHL adjustments

- Scoring era normalization: NBA PPG / NBA points per team-game; NHL goals/assists/points / NHL goals per team-game, throughout the entire local window. Does NOT normalize NBA rebounds, assists or three-pointers. Refit original z model.
- Full field-size sensitivity: estimate upper-tail scale from ranks 2-10, smoothed in time; add beta * ln(N_reference / N_season), where references are 30 NBA and 32 NHL teams. No adjustment made to MLB/NFL/individual sports.

| Subject | Original z | Scoring only z | Scoring + full field z |
|---|---:|---:|---:|
| Wilt Chamberlain 1961-62 | 7.731 | 4.478 | 6.328 |
| Wayne Gretzky 1985-86 | 7.500 | 5.195 | 5.169 |
| Joe DiMaggio 1941 | 7.197 | 7.197 (unchanged) | 7.197 (unchanged) |
| Babe Dye 1924-25 | see CSV | 3.770, rank 31 | 5.278, rank 3 |

- Alternate normalization for NBA points via league-wide FGA + 0.44*FTA (shot-opportunity proxy) returned Wilt z approximately 5.753, without field adjustment. It is not true possession normalization; it should not be represented as such.
- The exact league-size model is highly sensitive to field-effect strength and category/event era. At 50% field adjustment: Wilt 5.474, Gretzky 5.185, Babe Dye 4.937. At 100%: Wilt 6.328, Gretzky 5.169, Babe Dye 5.278. Rising and then falling response for Babe Dye beyond 100% illustrates that model-induced SD changes make effects nonlinear.
- 1985-86 NHL ~3.97 team goals/game vs 2025-26 ~3.08. 1961-62 NBA ~118.8 team points/game vs 2025-26 ~115.6. These are league averages, not individual pace-adjusted projections.

## Evidence of data hazards

- Three rows have future/unrealistic athletics years relative to a November 2026 snapshot: `ath_M_PV,2027,Ernest John Obiena`; `ath_W_PV,2109,Lucy Bryan`; `ath_W_HT,2202,Janee' Kassanavoid`. See `exploration/scan_data_quality.py`; verify original source and fix via a logged correction, not a guess.
- Athlete name prefixes containing measurement/wind notes (such as `+1,7 Simon Ehammer`) and aliases may cause self-exclusion failures. Inspect `exploration/input_inspection_notes.txt` and `exploration/possible_name_collisions.txt`. Similar strings are candidate issues, NOT definitive duplicate identities.
- Player identity must use stable IDs where possible. Similar names need not be the same person (e.g., Joe DiMaggio vs Dom DiMaggio).
- The field-size script's comment originally called its tail estimate "Theil-Sen" but the actual fitting uses `np.polyfit` (OLS) per season and rolling median smoothing. Corrected comment in portable version; still not robust per-season regression.
- Field-size script uses estimated NHL team counts for early years. Year-by-year independent validation is required.
- The project has only the top ten for most categories. It does NOT contain complete qualified competitor distributions, which limits identification of field-size effects.
- The NHL denominator applies goals/team-game to assists. A proper adjustment requires assist credits or opportunities (not assumed proportional to goals in every era).
- NBA pace figures depend on whether one is comparing league or Philadelphia team pace. An earlier summary used both ~126.2 and ~131.1 possessions/48: verify exact source/season/definition and do not conflate. Wilt also averaged more than 48 minutes per game, so per-36 and fixed-minutes alternatives answer different questions.
- NHL early franchises plus team-scoring adjustment may create implausible extrapolations. The relative standing of Babe Dye is a conspicuous stress test, not evidence that Babe Dye is definitively third.
- Unchanged MLB, NFL, swimming and athletics scores in the adjusted scenarios are NOT equally era-corrected. DiMaggio's unchanged z is a control in this experiment, not proof of an unassailable top rank.

## Reproducibility, uncertainty and presentation

- Approximate full combined ranking (note only NBA/NHL adjusted): DiMaggio, Chamberlain, Babe Dye, Ichiro, Gretzky, Beamon, Železný, Feller, Ledecky, Pete Rose. Do NOT publish as canonical.
- A window +/-7 versus +/-20 can materially move Gretzky's score even with the combined model. A single 15-year window is a design choice.
- Original peer-exclusion strongly favors athletes who repeat domination: removing only the tested season (and retaining other seasons) previously dropped Gretzky approximately 7.50 -> 3.56, Wilt approximately 7.73 -> 4.43, with little change for Beamon. Recompute this experiment from supplied baseline before using those numbers.
- Gumbel once-in-N values extrapolate beyond the observed 15-year windows and should be accompanied by tail diagnostics, model uncertainty and clear explanation.
- Do not claim interval estimates until a full documented resampling model is fitted.

## Files

- `exploration/greatest_seasons_adjusted_scenarios.csv`: full ranking by scenario.
- `exploration/greatest_seasons_adjusted_top20.csv`: original top-20 by selected scenarios.
- `exploration/era_field_sensitivity.csv`: raw, era, field-only, full combined results.
- `exploration/era_field_robustness.txt`: field strength/window sensitivity.
- `original_project/inputs/categories.csv` and `season_lists.csv`: authoritative project input snapshot for the audit.
