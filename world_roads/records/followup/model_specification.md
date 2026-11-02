# Model specification

This document gives the exact equations behind every score in `outputs/rankings_sensitivity.csv`. The variant
keys in brackets match its column names. All models are implemented in `analysis/gsmodel.py`
(`Spec`, `score`, `fit_baseline`, `transform`), `analysis/context.py` and `analysis/calibrate.py`.

## 1. Notation and data

- **Indices.** c is a category (for example NBA points per game, or women's 800 m freestyle) and s or t is a season. A
  season is stored as the calendar year in which it ends, so 1985–86 is stored as 1986.
- **Performance.** Y_{i,t,c} is the direction-adjusted performance of competitor i, where bigger is always better:
  - running and swimming: speed, d / time;
  - jumps and throws: the mark;
  - leagues: the statistic itself;
  - ERA: minus the ERA, or League ERA / ERA in the environment models.
- **List(t,c).** The published top of season t, in rank order. Athletics and swimming give each athlete's season best;
  leagues give the qualified leaders.
- **Identity.**
  - Original: identity is the normalised name.
  - Corrected data (`data/v2`) uses stable IDs: Reference IDs for MLB, NBA and NHL; Pro Football Archives and nflverse
    IDs linked across 1998/1999; Retrosheet IDs mapped to Baseball-Reference; name plus a birth-date and nation
    cluster for athletics; World Aquatics personIds clustered into people for swimming. The repairs are listed in
    `outputs/sources_and_corrections.csv`.
- **Short seasons.** K_c is the set of seasons excluded as cut short (totals only). In the corrected data it also holds
  seasons recorded under a different definition (NBA assists 1947–48).
- **Schedule.** G_t is the number of games per team, used for totals.

## 2. Estimands

**A. Peer dominance** (the original question). How far a season-leading performance stands above the best of
everyone else in the surrounding seasons, with all of the performer's own seasons removed from the comparison.

**B. All-seasons rarity** [`B1_allseasons`, `B2_allseasons_main`]. How far it stands above every other season in the
window, the performer's other seasons included. Only the tested season's own leader is replaced by that season's
runner-up. A serial dominator is compared with his own other seasons.

**A′. Environment-relative dominance** [`E*`]. Estimand A after dividing every value by the league's
per-team-game rate of the same statistic, raised to an elasticity γ. It asks how dominant the season was relative
to the scoring or opportunity environment. This is a changed estimand. For example, 50.4 points per game becomes
0.424 of a league team-game's points.

## 3. The score

For a season leader i in season y of category c, with window w:

**Comparators.** For each season s in W(y) = {y−w, …, y+w}, where s is a season of c and s ∉ K_c:

    C_s = max_{j ∈ List(s,c), j ≠ i} Y_{j,s,c}                (estimand A)
    C_s = max_{j ∈ List(s,c)} Y_{j,s,c} if s ≠ y; the runner-up if s = y   (estimand B)

Further rules:
- **Totals** are moved to the performer's schedule: C_s ← C_s · G_y / G_s.
- **Athletics.** A season with nobody other than i above the all-time list's cut-off takes the cut-off value.
  Athletics performances are scored from 1960.
- **Minimum.** At least 8 comparison seasons must have a real value.

**Baseline.** A regression of the comparators on time:

    C_s = f(s − y; θ) + δ · 1[s ∈ S_c] + ε_s

where S_c = {2008, 2009} for swimming (the full-body suits) and is empty otherwise. δ is fitted only if the window has
a suit season and at least 8 other seasons.

| trend | f(u; θ) | fitted by | scale σ̂ |
|---|---|---|---|
| `ols` (original) | a + b·u | least squares | √(SSR / (n − k)) |
| `flat` | a | least squares | √(SSR / (n − k)) |
| `quad` | a + b·u + q·u² | least squares | √(SSR / (n − k)) |
| `quad_safe` (main) | `quad` if at least 5 comparison seasons lie on each side of y, else `ols` | least squares | √(SSR / (n − k)) |
| `huber` | a + b·u | Huber M-estimate (c = 1.345, IRLS) | 1.4826 · MAD of residuals |

**Score:**

    z = (x − f(0; θ̂) − δ̂ · 1[y ∈ S_c]) / σ̂ ,   x = Y_{i,y,c}

**Ties.** The original scores only the first-listed leader. The corrected variants score every leader tied for the
season's best (`ties="all"`).

**Edge flag.** A score is flagged one-sided when fewer than 5 comparison seasons lie on one side of y. Variant
`A2_twosided` ranks only two-sided scores.

**Ranking.**
1. Each athlete-season enters once, under the category where its z is highest.
2. Athlete-seasons are ranked by z.
3. The page's top ten keeps each athlete's best season.

**"Once in N seasons".** This is the original's Gumbel return period, kept only for continuity:

    N = 1 / (1 − F(x − baseline))

where F is a Gumbel fitted by moments to the window's residuals. It extrapolates far beyond 15–40 observations. It
is not an observed frequency and is not used for ranking.

## 4. Choosing the window and trend

**Procedure** (`analysis/cv_window.py`).
1. For every category and season s, predict the season's best mark M_s (totals per team-game) from the other seasons
   in the window, with s itself left out.
2. Score the predictive N(f(0), σ̂²) by the continuous ranked probability score:

       CRPS = σ̂ [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ]

3. Divide by the category's median absolute season-to-season change, so that categories count equally.

**Rule, fixed before looking at any athlete.** Keep the brief's ±15 window and use the trend with the lowest mean
CRPS within it. That is `quad_safe` (0.785, against 0.858 for the original `ols`). The best overall is ±7
`quad_safe` (0.739), which is reported as variant `A2_w7`.

**Coverage.** Every trended model under-covers its nominal 90% interval (85–88%). Season maxima are heavier-tailed
than normal, which is why section 7 exists.

## 5. Scoring and opportunity environment (`analysis/context.py`)

**Denominators.** E_{s,c} is the league's per-team-game rate of the same statistic, from the Reference league
tables, Pro Football Archives and nflverse (`data/context/*_league.csv`):

| sport | categories and denominators |
|---|---|
| NBA | points, rebounds, assists and threes per team-game |
| NHL | goals per team-game; assists and points per team-game (goals × assists-per-goal, summed from every skater line) |
| MLB | HR, H, RBI, SB and pitcher SO per team-game; league BA; league ERA |
| NFL | passing yards, TD passes, rushing yards, receiving yards, interceptions and sacks per team-game |

**Not adjusted:**
- **MLB wins:** no denominator separates opportunity (starts, rotation size) from ability.
- **Hitting streaks:** these need a per-game hit-probability model.
- **Athletics and swimming:** a league rate does not apply.

**Transformation:**

    Y' = Y / E_{s,c}^{γ_c}          (ERA: Y' = E_{s,c}^{γ_c} / ERA)

**Elasticity.** γ_c = 1 in `E1_env_ratio`. In `E2_env_power` and `E3_main_env` it is estimated:

    Δ log F_{s,c} = α_c + γ_c Δ log E_{s,c} + u_s ,     F = median of ranks 2–10 (totals per team-game)

First differences remove slow trends; the leader is excluded from F.

**Pooling.** The γ̂_c are pooled across the 20 categories by DerSimonian–Laird random effects:

    w_c = 1/se_c² ,  τ² = max(0, (Q − (K − 1)) / (Σw − Σw²/Σw))
    μ = Σ (γ̂_c / (se_c² + τ²)) / Σ 1/(se_c² + τ²)
    γ_c* = (γ̂_c/se_c² + μ/τ²) / (1/se_c² + 1/τ²)

**Identification and limits.**
- **What identifies γ:** short-run co-movement of the field level with the league rate.
- **The rate is not purely external.** It is partly mechanical, because the league rate includes the players being
  compared.
- **Per-player opportunity isn't used.** No variant uses possessions, minutes, plate appearances or ice time for
  individual players: they are missing for the early decades, and no such values were invented.

**NBA points, three denominators** (`outputs/nba_points_denominator_effect.csv`, figure 3):
- league points per team-game;
- the mean of Basketball-Reference's *estimated* team paces (from 1950–51; the league table publishes pace only from
  1973–74);
- shot opportunities, FGA + 0.44 FTA.

These answer different questions and are reported side by side, not chosen between.

## 6. Field size (`analysis/context.py`, `analysis/env_field_tests.py`, `analysis/simulate.py`)

**The handoff's model** [`F_ratio_field25/50/100`], here extended from NBA and NHL to MLB and NFL:

    Y'' = Y' + λ · β_{c,s} · ln(N_ref / N_s)

- **N_s** is the number of teams, from the Reference league data; N_ref is 30 for the NBA and MLB and 32 for the NHL
  and NFL.
- **β_{c,s}** is the OLS slope of ranks 2–10 (on the Y' scale) against −ln(rank − 0.5) per season, clipped at 0, then
  taken as the median over seasons within ±7.
- **λ** is 0.25, 0.5 or 1.

**Normal-tail variant** [`F_ratio_field100_normal`]. Replaces β·ln(N_ref/N) with σ(√(2 ln N_ref) − √(2 ln N)), where
σ = β√(2 ln N).

**Estimated variant** [`F_ratio_field_estimated`].

    Δ C^{(2)}_{s,c} / sd_c = α + g_sport · Δ ln N_s + e_s

C^{(2)} is the season's second-best value on the Y' scale. g is pooled within sport. The offset applied is
g_sport · sd_c · ln(N_ref/N_s).

**What is and is not identified.**
- **Expansion seasons do identify a reduced-form shift.** It is near zero for the NBA (−1.0 ± 0.7 sd per unit
  ln N) and NFL (+0.2 ± 0.6), and positive for the NHL (+2.1 ± 0.5) and MLB (+1.7 ± 0.4).
- **They don't identify why.** More independent competitors and diluted talent both predict the same sign.
- **They don't identify the size of the talent pool.** That is what the β·ln N model assumes it knows.
- **The simulations** (`outputs/sim_*.csv`) show that the offset:
  - over-corrects when talent really is "more draws";
  - creates a small-league bias when the league is "the best of a fixed pool";
  - overstates the gain by 39–53% when tails are normal-like rather than Gumbel.

Field size is therefore reported only as sensitivity, never in the main model.

## 7. Cross-category tail calibration (`analysis/calibrate.py`)

**Why.** z is not equally rare in every category: hitting-streak leaders, for example, are often more than 3 sd above
trend.

**Fit.** For each category c, take the leaders' z above the category's 75th percentile u_c (at least 6 exceedances)
and fit a generalized Pareto distribution by maximum likelihood:

    P(Z − u_c > y | Z > u_c) = (1 + ξ_c y / σ_c)^(−1/ξ_c)

**Pooling.** The shape ξ_c is partially pooled across categories with the same DerSimonian–Laird estimator as in
section 5, using the standard errors from the numerical Hessian. The scale σ_c is then refitted with ξ_c* fixed,
leaving out the leader being calibrated.

**Output:**

    p = (k_c/n_c) (1 + ξ_c* (z − u_c)/σ_c)^(−1/ξ_c*)   for z > u_c ;   p = empirical exceedance share otherwise

The calibrated score [`*_cal`] is −log10 p.

**Multiple chances.** `expected_count` = p × (number of category-seasons) estimates how many seasons this rare the
whole dataset would hold by chance. It addresses the fact that 30 athletics categories give more chances than 3 NHL
categories.

**What p is.** A model-based probability within the category, with large uncertainty at the extremes, not an
observed frequency.

## 8. Uncertainty (`analysis/calibrate.py: bootstrap`)

**Moving-block bootstrap of comparison seasons.**
- **Blocks:** length 3, circular, drawn per category over its whole history, so neighbouring seasons stay together.
- **Replicates:** B = 300.
- **Each replicate:** reweight every window's comparison seasons by how often they were drawn, refit by weighted
  least squares, rescore every leader and rerank.
- **Output:** P(top ten) and the 5–95% range of each athlete's page rank. A rank beyond 200 is recorded as 201.

**What it covers.** Estimation uncertainty in the baseline and scale, with temporal dependence.

**What it does not cover:**
- measurement error in the leader's own value;
- data errors;
- the choice of model.

Model choice is covered separately by the **specification ensemble** (`outputs/spec_ensemble.csv`). This is the share
of 16 pre-declared specifications that put each athlete in the top ten: windows 7, 10, 15 and 20, trends `ols` and
`quad_safe`, and environment none or estimated, all on corrected data under estimand A.

## 9. Assumptions and limitations (all models)

1. **Leaders only.** Only season leaders are scored, so a great runner-up season is invisible. Under estimand A a
   close duel lowers the leader's score.
2. **The comparator is the best of the rest, not the full field.** Field depth enters only through the scatter of
   those bests. There are no qualified-player distributions for MLB, and the NBA and NHL player tables were fetched
   for context but not used to rebuild the comparator.
3. **Ends of the data.** Windows at either end are one-sided, and the trend is extrapolated there.
4. **Stationarity within a window.** Within a window, the residuals are treated as exchangeable after the trend
   (and the suit term). Rule changes inside a window are absorbed only through the trend, and through the
   environment in `E*`.
5. **Schedule scaling is linear in games.**
6. **Equipment and setting are not adjusted.** Altitude, wind beyond legality, equipment other than the swimming
   suits (spikes, poles, the ball) and doping are not modelled. Athletics needs participation-depth and technology
   treatment, which is not attempted beyond the suit term and the depth diagnostics of the earlier report.
7. **Identity rules are heuristics where no ID exists** (athletics, swimming). Unresolved cases are listed.
8. **Category coverage differs by sport**: 30 athletics, 22 swimming, 9 MLB, 6 NFL, 4 NBA and 3 NHL categories.
   Calibration and the expected count soften this; the category list itself is a design choice.
