# Response to the round-2 independent review

Review received 2–3 November 2026 (`INDEPENDENT_REVIEW_FOR_CLAUDE.md` with `repair_comparison.py`). Every point
is listed below with what changed, where, and the result.

Nothing in `data/original/` or `data/v2/` was overwritten to force a result. The original method and the published
ranking still reproduce exactly (`tests/test_reproduction.py`, 15 checks). New checks are in `tests/test_round2.py`.

## Critical issue 1: original and corrected IDs were not joined (agreed; fixed)

**The problem.** `run_all.py` keyed original-data rows by normalised name and corrected rows by stable ID, so the
sensitivity table never put both scores in one row: 8,315 rows, none with both.

**What changed.**
- **The crosswalk.** `analysis/run_all.py: crosswalk()` maps every original (category, season, rank) row to the stable
  ID carried by `data/v2/season_lists_v2.csv`, the reviewer's approach.
- **Ranking versus joining.** `ranks()` ranks each variant with its own identities, so the original still reproduces
  the published ranking. Only the join key is mapped.
- **Every variant** goes through the same join, not just four.
- **New columns.** `rankings_sensitivity.csv` gains `athlete_id` and `id_status` (matched, collapsed,
  removed_in_v2, original_only, corrected_only), plus `one_sided_window_main` and `incomplete_season`.
- **The log.** `outputs/identity_crosswalk_log.csv` lists the collapses and removals.
- **A guard.** The run asserts that more than 4,000 athlete-seasons carry both scores.

**Result.**
- **The counts match the reviewer's.** 4,281 rows; 4,032 with original and corrected scores; 2 collapses (Hank/Henry
  Aaron 1956, Bobby Witt/Bobby Witt Jr. 2025); 3 records removed in v2 (Ernie Calverley 1947 and 1948 NBA assists,
  Ernest John Obiena 2027).
- **The scores match too.** Across the 4,280 athlete-seasons in the reviewer's `comparison_join_fixed.csv`, every
  original and corrected z agrees. The one extra row here (Robert Hayes 1964) comes from a variant outside the
  reviewer's four.

**Also found.** The collapses led to a wider check: 9 baseball players appear under two display names on different
leader boards (for example Joe/"Shoeless" Joe Jackson, Cal Ripken/Cal Ripken Jr.). In the original this split a
player into two athlete-seasons only where both names led a category in one season; exclusion within a category was
not affected. They are logged as `ALIAS` rows in `sources_and_corrections.csv`.

**Tests.** `test_round2.py` checks:
- more than 4,000 rows carry both scores;
- each of the four headline seasons has exactly one row with both scores;
- the collapse and removal sets are exactly as above;
- there is one row per (sport, stable ID, season).

## Critical issue 2: the calibration was labelled GPD but computed an exponential tail (agreed; fixed both ways)

**Labels now match the code.** The calibrated rankings use the pooled-exponential tail. Every `*_cal` label in
`variants.csv`, `model_specification.md` section 7, `calibrate.py`'s docstring and `top10_robustness.md` now says
"pooled-EXPONENTIAL tail (sensitivity model); GPD shapes are diagnostics only".

**A real GPD alternative.**
- **Fit.** `calibrate(..., family="gpd")` fits a GPD with a bounded, partially pooled shape and a leave-one-out
  scale.
- **Support check.** A leader beyond the fitted upper endpoint is flagged "beyond fitted support" rather than given a
  silent p = 0.

**Validation: leave-one-out tail coverage** (`outputs/calibration_tail_coverage.csv`, main model, all leaders).

| t | Expected leaders with p ≤ t | Exponential | GPD |
|---|---|---|---|
| 0.05 | 247 | 278 | 308 |
| 0.01 | 49 | 44 | 127 |
| 0.002 | 9.9 | 8 | 52 |
| 0.001 | 4.9 | 5 | 32 |

- **The exponential tail is calibrated** to t = 0.001.
- **The GPD is anti-conservative** by up to 6.5 times. It also puts Beamon beyond its support at every threshold,
  and DiMaggio for q ≥ 0.75, so it is rejected as a calibrator.
- **Below 0.001** the data can't check either model.

**Threshold and family sensitivity** (`outputs/calibration_sensitivity.csv`). Every model's top 12 seasons are
recalibrated at q = 0.60, 0.65, 0.70, 0.75, 0.80 and 0.85 with both families. The exponential values reproduce the
reviewer's table to within 1%; `test_round2.py` checks all 24 cells.

**Intervals.** Each top-30 season's p has a 90% interval from resampling its category's other leaders
(`calibrated_*.csv`). The table and documents show the threshold range next to it, because the threshold range is
wider. Beamon: resampling interval 4.0e-5 to 8.4e-5; threshold range 1.2e-5 to 1.3e-4.

**Expected counts.**
- **E = p × N,** where N is the number of scored records of that model, with tied co-leaders counted: 4,940 for
  `E3_main_env`, 4,943 for `A2_main`. This fixes the earlier "4,943 category-seasons" for E3.
- **Poisson P(≥1) is printed beside E:** Beamon 0.27, DiMaggio 0.47.

**Claim withdrawn.** "Only Beamon and DiMaggio survive the multiple-chances check" is gone. The documents now say E
below 1 is not a significance threshold, that the independent-Poisson reading is illustrative because scores share
athletes, seasons and categories, and that Beamon and DiMaggio are merely the hardest seasons to explain as the
expected extremes of a ~4,900-score search.

**Empirical null** (`outputs/empirical_null_top30.csv`, `calibrate.empirical_null`).
- **How it's built.** Ordinary, non-target comparison seasons are withheld with their neighbours and scored like
  leaders, about 119,000 draws from 4,660 distinct category-seasons. Draws held by the athlete being calibrated are
  excluded.
- **Its limit.** It can only bound p down to about 1/4,660, so it is shown as a model-light reference beside the tail
  model, not as a replacement.
- **Not attempted.** A full-pipeline null simulation of all 74 categories.

## High-priority cautions

**1. Cross-validation target mismatch (agreed; redone).**
- **New script.** `analysis/cv_comparator.py` predicts the candidate-specific comparator, the best non-candidate
  performance in the target season, with the block {y−1, y, y+1} withheld.
- **Held-out test.** Categories are split once (seeded, stratified by sport) into a selection half and a test half.
- **Interval.** The improvement over the original line has a two-level bootstrap interval (categories, then 5-season
  blocks).
- **Result.** Within ±15 the selection half again chooses `quad_safe`. On the held-out test half it beats the
  original straight line by 0.057 CRPS units (90%: 0.025 to 0.100).
- **The old script** (`cv_window.py`) is marked superseded.

**2. ±15 against ±7 (agreed).** The documents now call ±15 an editorial constraint from the brief. On the new
cross-validation, ±10 `quad_safe` is best on the selection half and ±7 `quad` on the test half; both are reported.

**3. Coverage (agreed).** The candidate-specific 90% intervals cover 78–83%. The documents call z a standardization,
not a calibrated normal quantile.

**4. The bootstrap is conditional (agreed; relabelled, enlarged, model-averaged).**
- **Columns.** `top10_frequency_in_refits` with `monte_carlo_se`, from B = 1,000 for the main models (500 for A1).
- **Wording.** Documents and figures say "frequency among refits, conditional on the model".
- **Model-averaged bootstrap** (`bootstrap_model_averaged.csv`). 800 draws, each picking one of the 16 pre-declared
  specifications at random, then a block resample.
- **No more "robust" for the top four.** The specification-ensemble statement (in the top ten under all 16
  specifications) is kept separate from the resampling frequencies, which are lower. Results are below.

**5. Equal-category weighting (agreed; wording fixed, all three schemes reported).**
- `cv_comparator_scores.csv` reports CRPS per prediction, per category and per sport.
- All three rank ±15 `quad_safe` above ±15 `ols`.
- The incorrect "all categories count equally" line in `cv_window.py` is corrected.

**6. Incomplete 2026 season (agreed).** Swimming 2026 is flagged `incomplete_season` ("calendar year to 2 Nov 2026")
in the sensitivity and calibration tables. The one-sided-window flag already marks every 2026 season. Walsh 2026 is
called out in `audit_followup.md` finding 8.

**7. 4,943 against 4,940 (agreed).** E3 scores 4,940 records and `A2_main` 4,943; each table reports its own N, and
ties count as separate records. `test_round2.py` checks N = 4,940 for E3.

## Recommended order

1. **Identity mapping:** done, with tests.
2. **Exponential versus GPD, with sensitivity and intervals:** done, with tests.
3. **Cross-validation on the comparator, with uncertainty in the improvement:** done.
4. **Bootstrap labels, more draws and model uncertainty:** done.
5. **Per-player opportunity research:** not done in this round. It stays first on the acquisition plan
   (`audit_followup.md`), and the team-count correction stays out of every headline model.
6. **Reader-facing output:** `audit_followup.md` ("For a documentary or website audience") now separates original
   peer dominance, corrected peer dominance, environment sensitivity, all-season rarity and model uncertainty, and
   does not declare a single greatest season.

## Resampling results (B = 1,000; model-averaged 800)

Top-ten frequency among refits, conditional on each model. MC SE is at most ±0.016 at B = 1,000 and ±0.018 at 800.

| Season | `A2_main` | `E3_main_env` | `A1_corrected` (B = 500) | Model-averaged | Specs with season in top ten (of 16) |
|---|---|---|---|---|---|
| Wilt 1961–62 | 0.98 | 0.88 | 1.00 | 0.81 | 16 |
| Gretzky 1985–86 | 0.92 | 0.61 | 0.93 | 0.67 | 16 |
| DiMaggio 1941 | 0.84 | 0.80 | 0.93 | 0.56 | 16 |
| Beamon 1968 | 0.76 | 0.74 | 0.44 | 0.77 | 16 |
| Best of the rest | Železný 0.43 | Vance 0.41 | Ledecky 0.31, Ichiro 0.30 | Walsh 0.32 | Železný 11 |

**The reviewer's point holds.** Gretzky at 0.61 under E3, DiMaggio at 0.56 once the specification varies, and Beamon
at 0.44 under the original straight-line trend are not "robust under uncertainty". The documents now say the four are
clearly separated from the rest but not settled.
