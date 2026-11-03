# Independent review of greatest_seasons_followup.zip
Date: 2026-11-02

## Verdict
The follow-up is a substantial statistical improvement. The reported 15 reproduction tests pass when run locally, including the four original windows, the top 200, and the earlier handoff's era/field examples. The strongest methodological correction is the rejection of a franchise-count multiplier as a headline adjustment. The revised approach sensibly separates peer dominance, all-season rarity, era-relative dominance, and speculation about field size.

However, two implementation/documentation defects need repair before the new rankings and rarity estimates are released. Further important uncertainty statements should be softened.

## Critical issue 1: Rankings sensitivity table fails to align original and corrected athlete IDs

`analysis/run_all.py`: the `ranks()` dictionaries use keys `(sport, id, year)`. In `Data('original')`, `id` is a normalized athlete name; in `Data('v2')`, it is a persistent source ID. No identity crosswalk is applied before unioning these dictionaries. The result is a duplicated legacy/corrected record for most athletes.

Measured in the supplied `outputs/rankings_sensitivity.csv`:
- 8,315 rows; 4,037 rows with `A0_original_z`, 4,277 with `A1_corrected_z`.
- ZERO rows have both scores. The four headline athlete-seasons each appear twice.
- This misleads direct score comparisons, can break downstream plots/merges, and unnecessarily inflates the file.

A reproducible repair is in `repair_comparison.py`. It maps each original `(category, year, rank)` record to the corrected stable ID carried in `data/v2/season_lists_v2.csv`, then joins by `(sport, stable ID, year)`.

The fixed four-scenario output has 4,280 unique athlete-seasons, of which 4,032 have both original and corrected scores. Of the 4,037 original athlete-seasons, two pairs collapse to single corrected identity: Hank/ Henry Aaron (1956) and Bobby Witt/ Bobby Witt Jr. (2025); three original-season entries are intentionally removed from v2 (two Ernie Calverley NBA assists seasons and Ernest John Obiena 2027). This is why the fixed table has 4,035 distinct original-identity keys before joining, not 4,037. A total of 245 corrected athlete-seasons are genuinely new.

Action: implement the explicit original-to-v2 mapping inside `run_all.py`, flag ID collapses and removals, and assert that the resulting comparison has >4,000 rows with both original and corrected scores. Never overwrite source inputs to force joins.

## Critical issue 2: tail-calibration model and documentation contradict one another

`model_specification.md` section 7 and the variant labels in `outputs/variants.csv` claim the *calibrated rankings* fit a generalized Pareto distribution with partially pooled shape `xi`, i.e. tail factor `(1 + xi*y/sigma)^(-1/xi)`.

But `analysis/calibrate.py:calibrate()` actually computes calibrated p by the exponential formula:

`p = (k/n) * exp(-(z-u)/exp(ls))`

It calculates GPD shapes as a diagnostic (`gpd_shape`, `xi_hat`, `xi_pooled`) and never applies them to these p-values. The leading docstring of that function correctly calls it an exponential tail. This is a substantive mismatch, not simply terminology, because the estimated GPD shapes are negative on average and could dramatically alter extreme tail probabilities.

Action: either (a) correctly relabel all variant keys/reports as 'pooled exponential tail with GPD shape diagnostics', OR (b) actually fit, validate and use a GPD with uncertainty intervals, checking support and out-of-sample tail coverage. Until then the `*_cal` rankings and 'expected occurrence' claims should explicitly say 'exponential sensitivity model', not 'GPD' or settled rarity.

### Tail sensitivity check

Recomputed E3 main environment scores from `data/v2` and `outputs/gamma_estimates.csv` and reproduced the exponential calibration while varying the category tail threshold `q`. The reported `q=0.75` p-values matched the packaged results.

| q | Bob Beamon p | Joe DiMaggio p | Wilt p | Gretzky p |
|---|---:|---:|---:|---:|
| 0.60 | 0.000125 | 0.000182 | 0.000653 | 0.00113 |
| 0.65 | 0.000104 | 0.000168 | 0.000634 | 0.00115 |
| 0.70 | 0.0000842 | 0.000146 | 0.000442 | 0.000773 |
| 0.75 | 0.0000634 | 0.000127 | 0.000383 | 0.000716 |
| 0.80 | 0.0000227 | 0.0000895 | 0.000424 | 0.000858 |
| 0.85 | 0.0000117 | 0.000134 | 0.000456 | 0.000756 |

Beamon's estimated p changes more than 10-fold across these defensible thresholds, even before model-family uncertainty. At q=.75 the package's expected-count calculation returns Beamon ~0.31 and DiMaggio ~0.63 across ~4,940 tests. Under a Poisson interpretation, these correspond to roughly 27% and 47% probability of at least one such event, respectively. An 'expected count below one' is not a conventional significance threshold or a guarantee that either season uniquely 'survives' multiple-testing correction. Scores share athletes/seasons/categories, so independent-Poisson is illustrative, not an exact family-wise test.

Action: present uncertainty intervals and threshold/model sensitivity before any cross-sport tail-probability headline; ideally use full-pipeline null simulations or a valid empirical null calibration.

## High-priority methodological cautions

1. **Cross-validation target mismatch.** `analysis/cv_window.py` cross-validates *whole-season maxima* against maxima in surrounding seasons. The intended peer-dominance score compares with the best *non-target athlete* in each season, which has a different distribution, particularly for serial dominators. A lower CRPS for raw maxima need not optimize the actual peer-exclusion statistic. Add CV at the level of the candidate-specific comparator series, withholding a whole season or temporal block at a time, not merely refitting maxima. Preserve an independent test set for trend selection.
2. **A2_main vs ±7.** The best CV model reported is ±7 quadratic (0.739), but the declared main variant remains ±15 quadratic (0.785) to honor the original brief. That is a reasonable editorial constraint, but call it a constraint, not the globally optimal method.
3. **Coverage.** The CV normal intervals cover only 85–88% of nominal 90%. Treat the z-values as standardizations, not calibrated standard-normal tail probabilities.
4. **Bootstrap is conditional.** The reported `P(top ten)` is a proportion among 300 refits to bootstrapped comparator seasons, not an inferential probability of the 'true historical top ten'. A 0.57 estimate for Gretzky under E3 is not consistent with calling the top four universally robust under *uncertainty*. Distinguish 'all 16 selected specifications' from 'bootstrap top-ten frequency'. For 300 trials a probability near 0.57 has Monte Carlo standard error of ~0.029, excluding model uncertainty.
5. **Equal category weighting claim.** Dividing CRPS by each category's median year-to-year change equalizes *units*, but aggregating individual season predictions still gives longer-history categories more weight. Independent recalculation under equal category and equal sport weights still favors ±15 quadratic over ±15 linear, and ±7 quadratic over ±15 quadratic, so this wording error does not change the model winner. The diagnostics should nevertheless report all three weighting schemes.
6. **Partial season and edge effects.** The 2026 swimming 'season' is not necessarily complete on November 2, 2026. Do not conflate an incomplete calendar-year record with a finished season. The extreme Walsh rank spread is an especially good reason to flag incomplete and one-sided comparisons.
7. **Small inconsistency:** `audit_followup.md` describes 4,943 category-season opportunities; `G.score(v2, Spec(ties='all', trend='quad_safe', env='power'))` produces 4,940 scored category leader records locally. The calibration's reported Beamon count is consistent with multiplying p by ~4,940. Correct the report to the actual quantity and specify whether ties count as extra tests.

## Recommended implementation order

1. Repair identity mapping in the sensitivity output. Add tests asserting nonzero cross-version alignment and explicit handling for identity merges and removed records.
2. Fix the exponential-versus-GPD specification mismatch and run threshold/model-family calibration sensitivity tests with tail uncertainty intervals.
3. Redesign the CRPS evaluation to predict the candidate-specific exclusion comparator, and quantify uncertainty in improvement versus the original trend (category-level temporal blocks).
4. Re-run bootstrap with the documented target, clear labels, and enough draws to stabilize estimates; include differing trend/window/era choices as model uncertainty.
5. Continue per-player opportunity research (NBA minutes, MLB PA/batting streaks, NHL TOI limits) rather than applying a poorly identified team-count correction.
6. Final output for readers: original peer dominance, corrected peer dominance, environment sensitivity, all-season rarity, model uncertainty, and well-labelled limitations. Avoid declaring a unique numerical 'greatest season'.

## Independent verifications

- All 15 supplied reproduction checks passed locally.
- The fixed comparison pairs original and revised results for Wilt, Gretzky, DiMaggio and Beamon correctly.
- The source figure for Philadelphia Warriors' 1961–62 team pace is 131.1 on Basketball-Reference (https://www.basketball-reference.com/teams/PHW/1962.html); the team's pace and the mean of all teams' pace are different quantities.
- Sports Reference explicitly describes pre-1973–74 pace as estimated from league-average assumptions: https://www.sports-reference.com/blog/2013/12/estimating-pace-and-per-possession-ratings-1951-1973/

This review audits the methodology and outputs supplied in the follow-up ZIP. It is not a comprehensive independent verification of all 163 claimed historical source corrections.
