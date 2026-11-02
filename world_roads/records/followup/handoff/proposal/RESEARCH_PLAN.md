# Proposed statistical research plan

## 1. Define the estimands first

The term "greatest season" is underspecified. Keep at least two separate estimands:

A. Peer dominance: exceptional performance relative to contemporaries, with the focal athlete's other years removed from the comparator. This is what the original analysis principally targets.

B. Historical one-season rarity: exceptional performance among all other seasons, retaining the focal athlete's other years (excluding the particular tested season). This is not the same question; earlier tests found huge differences for serial dominators.

A third optional estimand is percentile dominance relative to all qualified competitors in the target year. This directly incorporates eligible-pool size but requires full player-season distributions or appropriately modeled truncation.

## 2. Potential statistical architecture

Let Y_{i,t,c} be a direction-adjusted performance for competitor i, season t and category c. Separate opportunity/exposure O_{i,t,c} from skill/relative output where meaningful:

- NBA PPG: estimated possessions while on court and minutes, with per-75-possession points plus usage/efficiency and team-point-share checks; do not replace a per-game record with a per-36 number without declaring a changed estimand. Early estimated pace adds measurement error.
- NBA RPG/APG: rebound chances, assisted-basket opportunities, role, playing time and pace. Three pointers: three-point attempts/usage and changing shot availability; raw makes should remain a separate, historically literal target.
- NHL goals/assists/points: goals per team game is a crude proxy, especially for assists. Use league goals, assist credits per goal, player time on ice where available, team scoring, role and schedule. Account for changes in goalie equipment, expansion, rules and power plays where documented.
- MLB: plate appearances or innings, league run environment and park effects, team games, integrated/expansion-era coverage, strike zone and ball changes; hitting streaks require a distinct at-bat/plate-appearance opportunity model.
- NFL: team passing/rushing volume, attempts/targets/snaps, league games, roster sizes, passing efficiency, historical rule changes and team count; adapt categories individually.
- Athletics/swimming: event participation depth, world-list completeness, national access, equipment/rule changes, timing and pool/track effects. More entrants do not automatically mean the per-year top mark should rise by log entrants.

## 3. Field-size model, not merely franchise ratio

The original reference uses one season leader (after excluding the subject) in each year. The expected extreme grows with the number of independent eligible competitors. However, players are not independent and identically distributed, and team expansion changes talent concentration and opponent quality.

Preferred empirical path: gather qualified full player distributions and eligibility counts by category-year; compare equal upper quantiles within year, e.g. 99th percentile, with equal qualification rules. Sensitivity: compare top-k where k scales as roster/player pool, not fixed top ten.

Second path: estimate number of effective competitors N_eff and a category/year tail distribution using censored/truncated likelihood on available top-K ranks. Calibrate tail choice using held-out seasons and predictive checks; propagate uncertainty in N_eff, tail shape and threshold. Compare generalized Pareto, Gumbel/GEV, empirical quantiles and stable parametric/hierarchical models. Do not infer exact N_eff from team counts without uncertainty.

Current experimental formula: x_adjusted = x_normalized + lambda*beta_{c,t}*ln(N_reference/N_t). Here beta is estimated by an OLS fit of top-ranked season entries 2-10 against -ln(rank-0.5) and median-smoothed across +/-7 years. This is a scenario assumption, not a justified correction. In particular, adding the same shift to every top player in a year does not establish equivalence of full distributions and can create large artifacts in short early leagues.

## 4. Estimate conditional trends cautiously

Original: local straight line plus suit adjustment, +/-15 years. Test LOESS/robust local regression with robust scale, change-point methods and one-sided edge windows. Cross-validation should tune smoothing bandwidth rather than choosing it to make a preferred athlete win. Test whether the series of season maxima is approximately stationary after adjustment; include autocorrelation and rule/equipment changes. Do not overinterpret a category with 8 real comparator seasons as equally precise as a category with 31.

## 5. Cross-category, cross-sport comparison

A z-score of one category's fitted annual maxima need not have equal upper-tail rarity across categories with different skew, sample sizes, history lengths, ties and truncation. Calibrate every category's score through posterior/predictive extreme-tail probabilities under an explicitly chosen model; use annual maxima and the full field where available. Control multiple chances: a sport represented by 30 athletics categories has more opportunities to enter the top ten than a sport with 3 NHL categories. Run equal-sport-weight or per-category competition sensitivity. Treat the Gumbel return period as a model-based extrapolation, not observed empirical frequency.

## 6. Distinguish ranking and information uncertainty

For every candidate, output: original score; opportunity/era-adjusted score; full or partial field adjustment range; rank range; sensitivity to +/-window and exclusion rules; quality of underlying measurement; uncertainty (posterior/bootstrap if estimable). Use seasonal or temporal block resampling and model-averaging to propagate specification uncertainty. A single bootstrap from independent regression residuals will understate temporal and data-generation uncertainty.

## 7. Proposed study phases

Phase 1, reproduce and clean: input checks, name IDs, improbable years, rank ties, schedules, shortened seasons, coverage diagnostics and exact score reproduction.

Phase 2, targeted NHL/NBA: use independently sourced scoring/opportunity series, qualified player distributions, real league counts and team/roster size, and test plausible expansion mechanisms rather than assuming one. Reevaluate Wilt, Gretzky, Babe Dye, Rodman, Curry and other affected competitors.

Phase 3, other sports: comparable adjustments on MLB/NFL, then athletics/swimming participation and equipment; itemize where reliable measurements cannot be recovered.

Phase 4, uncertainty and presentation: stable vs unstable top-ten membership, rank intervals, decision table and transparent script for a documentary audience.

## 8. Core model checks

- Pure change of measurement units should not alter z within a homogeneous category. Era adjustment must work through time-varying denominators, not simple constant rescaling.
- Reordering rows in input must not change the season leader selected; ties require explicit rules.
- Top-K fitting should not become spuriously precise as the league grows while K remains fixed.
- A simulation with identical individual ability but varying N should exhibit little systematic rank bias after adjustment. A simulation with expansion-driven opponent dilution should identify that confounder instead of blindly adding a positive correction.
- Missingness, athlete aliasing and 2027/2109/2202 years must not leak into historical reference windows.
- Explicitly test rate (points per game) vs season-total categories for schedule effects and incomplete participation.
