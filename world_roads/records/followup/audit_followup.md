# Follow-up audit: greatest seasons

Date: 2 November 2026. Seasons are labelled by the year they end, so 1985–86 is 1986. "z" is standard deviations
above the comparison baseline; "#" is the page rank, with each athlete's best season only.

Every number below comes from `outputs/` and can be regenerated with `make_all.sh`. Each finding is labelled:
- **Fact:** verified against a source.
- **Reproduction:** recomputed and matched.
- **Judgment call:** a modelling choice with no single right answer.
- **Error:** a defect that should be fixed.
- **Counterfactual:** a simulation or what-if.

## 0. Reproduction (reproduction)

- **The original scores.** `tests/test_reproduction.py` reproduces all 18,561 pipeline scores at windows 7, 10, 15
  and 20 (largest difference 5e-13) and the published top 200.
- **The handoff's numbers, run unchanged.**
  - Original: Wilt 7.731, Gretzky 7.500, DiMaggio 7.197.
  - Scoring era: Wilt 4.478, Gretzky 5.195.
  - Scoring plus full field: Wilt 6.328, Gretzky 5.169, Babe Dye 5.278.
- **The all-seasons figures** the handoff asked to recompute are confirmed: Gretzky 3.56 and Wilt 4.43.
- **No numerical error in the handoff's scripts.** The problems below are in data and design, plus three wrong NHL
  team counts.

**Round 2** (independent peer review, 2–3 November 2026). Two defects in this follow-up were found and fixed:
- **The sensitivity table did not join** original and corrected scores.
- **The calibration was labelled GPD** but computed an exponential tail.

The cross-validation now targets the comparator the score actually uses, the bootstrap is reported as a conditional
frequency, and several claims below have been narrowed. Point by point: `REVIEW_RESPONSE.md`.

## Ranked findings (largest effect on the top ten first)

### 1. The estimand decides more than any adjustment (judgment call)

| | Peer dominance (own seasons excluded) | All-seasons rarity (own other seasons kept) |
|---|---|---|
| Wilt Chamberlain 1961–62 | 7.73 (#1) | 4.43 (#4) |
| Wayne Gretzky 1985–86 | 7.50 (#2) | 3.56 (#20) |
| Joe DiMaggio 1941 | 7.20 (#3) | 7.35 (#1) |
| Bob Beamon 1968 | 5.02 (#5) | 5.02 (#2) |

- **Why serial dominators fall.** Their own other seasons become the bar. Gretzky's 163 assists is measured against
  his own 120–135-assist seasons.
- **Which question is right.** Both are legitimate. "Most dominant over his rivals" is peer dominance; "the most
  unrepeatable single season" is all-seasons rarity.
- **Fix:** name the estimand on screen. Report both (`A*` and `B*` variants).

### 2. Field-size offset: not identified, and it creates artifacts (error, as used for correction)

**The offset.** β·ln(N_ref/N) is added to every season, with β taken from ranks 2–10. Babe Dye 1924–25 shows what
it does:

| Model | z | Rank |
|---|---|---|
| Original | 0.85 | #601 |
| Goals ÷ league goals per team-game | 3.77 | #36 |
| Plus 50% field offset | 5.11 | #7 |
| Plus 100% field offset | 5.50 | #4 (handoff: 5.278, #3) |

**Why Dye rises.**
- **Mechanism.** The offset lifts the 3–4-team seasons before 1925 more than his own 6-team season. That steepens
  the comparison line and shrinks its scatter from 1.34 to 0.87. His own margin over the baseline actually falls
  (5.04 to 4.78).
- **His rise comes from the denominator, not from his season** (`outputs/babe_dye_decomposition.csv`,
  `figures/fig4_babe_dye.png`).

**The tail estimate is not like-for-like across eras** (`outputs/beta_diagnostics.csv`).
- **Different parts of the distribution.** In NHL leagues of six or fewer teams, ranks 2–10 reach down to the top
  6.8% of skaters. Today they cover the top 1%. A slope from a small league describes a different part of the
  distribution.
- **Ties.** Integer counts tie 2–3 neighbouring ranks out of 8 in the NHL.
- **One-value sensitivity.** Dropping rank 2 moves β by 10–20%.
- **Normal-like tails.** The β·ln N rule overstates the gain from 60 to 640 players by 53%, and from 120 by 39%
  (`outputs/sim_beta_estimator.csv`).

**The data can't tell why expansions matter** (`outputs/field_size_test.csv`). The shift in the second-best value
at expansion seasons, in standard deviations per unit of ln(teams):

| Sport | Shift | Reading |
|---|---|---|
| NBA | −1.0 ± 0.7 | not detectable |
| NFL | +0.2 ± 0.6 | not detectable |
| NHL | +2.1 ± 0.5 | positive |
| MLB | +1.7 ± 0.4 | positive |

A positive shift fits both "more independent competitors" and "diluted talent", so it does not identify the
mechanism the offset assumes.

**Simulations** (counterfactual; `outputs/sim_field_worlds.csv`). Share of the top ten from small-league seasons,
where 33% would be no bias:

| World | Original metric | With the offset |
|---|---|---|
| Expansion means more independent draws | 29% | 36% (over-corrects) |
| Expansion spreads a fixed talent pool thinner | 31% | 37% (creates a small-league bias) |

**The handoff's NHL team counts** are wrong in 3 seasons: 1918–19 should be 3, 1931–32 should be 8 and 1934–35
should be 9.

**Fix:**
- Keep the field-size offset as a labelled sensitivity only, never in a headline ranking.
- An empirically estimated expansion effect (`F_ratio_field_estimated`) gives Dye 2.55 (#110) and Wilt 4.04 (#23).
  It is shown, not recommended, because its mechanism is unidentified.

### 3. The scoring environment: supported in most leagues, weakest for NBA points (judgment call with evidence)

**Elasticity.** Does the field (median of ranks 2–10) move with the league's rate of the same statistic? Measured
on first differences of logs (`outputs/env_elasticity.csv`, `figures/fig5_elasticity.png`):
- **Close to proportional:** NHL goals, assists and points (0.93–1.01); MLB HR, H, RBI, SO and BA (0.85–1.03);
  NFL yardage (1.1–1.3); NBA assists and threes (0.85–1.14). These explain 35–70% of the field's year-to-year
  changes.
- **Weak:** NBA points 0.65 ± 0.23 (it explains 10% of the variation), NBA rebounds 0.37 ± 0.23, NFL interceptions
  and sacks.

**Wilt's z for 1961–62 depends on the denominator** (`outputs/nba_points_denominator_effect.csv`,
`figures/fig3_nba_denominators.png`):

| Denominator | Wilt 1961–62 z |
|---|---|
| None | 7.73 |
| Mean of estimated team paces | 6.05 |
| Shot opportunities, FGA + 0.44 FTA | 5.75 |
| League points per team-game | 4.48 |

He stays first in NBA points under every one.

**The two pace figures (fact).** Basketball-Reference publishes no league pace before 1973–74, the first season
turnovers and offensive rebounds were recorded.
- **126.2** is the unweighted mean of the nine teams' *estimated* paces for 1961–62.
- **131.1** is the Philadelphia Warriors' own estimated pace; the range across teams is 122.9–131.1.

They answer different questions: league environment versus Wilt's own team, which his play helped speed up. No
per-player possession figures exist for 1961–62, and none were invented.

**With the elasticity estimated** (partially pooled, `E3_main_env`):

| Season | z | Rank |
|---|---|---|
| Wilt 1961–62 | 7.05 | #2 |
| Gretzky 1985–86 | 6.47 | #4 |
| Babe Ruth 1920 (home runs) | 5.56 | #9 |
| Stephen Curry 2015–16 (threes) | 5.69 | #7 |

**Fix:**
- Treat environment-relative dominance as its own estimand.
- Estimate the elasticity rather than impose 1.
- Show the NBA denominators side by side.

### 4. The ±15 straight line is not the best baseline (judgment call, decided by cross-validation)

**Cross-validation on the quantity the score uses** (`analysis/cv_comparator.py`, revised in round 2).
- **Target.** For each scored candidate, predict the best non-candidate performance in the target season (the
  runner-up) with the block {y−1, y, y+1} withheld.
- **Split.** Categories are split into a selection half and a test half.
- **Weighting.** Results are reported per prediction, per category and per sport.

| Model | Selection half (equal category) | Test half (equal category) |
|---|---|---|
| Original: straight line, ±15 | 0.870 | 0.987 |
| Quadratic, straight near the data's ends, ±15 (main) | 0.857 | 0.931 |
| Same, ±10 (best overall on the selection half) | 0.842 | 0.910 |
| Quadratic, ±7 | 0.844 | 0.890 |

- **Main trend confirmed out of sample.** The main trend's improvement over the original line on the held-out
  categories is 0.057 CRPS units (90% interval 0.025 to 0.100, two-level bootstrap over categories and 5-season
  blocks).
- **Weighting doesn't change the answer.** All three weightings rank it above the straight line.
- **±15 is an editorial constraint** from the brief, not the best window. ±10 and ±7 predict better.
- **Coverage.** The 90% intervals cover only 78–83%. z is a standardization, not a calibrated normal quantile.
- **The earlier season-maxima version** (`cv_window.py`) chose the same trend but had the wrong target.

**Main model (`A2_main`).**

| Season | Original | Main model |
|---|---|---|
| Beamon | 5.02 | 7.80 (#1) |
| Wilt | 7.73 | 7.66 |
| Gretzky | 7.50 | 7.53 |
| DiMaggio | 7.20 | 7.01 |
| Pete Rose | #10 | #17 (leaves the top ten) |
| Dan Marino | #12 | #9 (enters) |

Beamon rises because the long-jump comparators were flat at about 8.2–8.35 m from 1960 to 1978 and only rose in
1979–83. A straight line through that overstates the scatter.

**Fix:** adopt the cross-validated trend within the brief's window, and report ±10 and ±7 as sensitivities.

### 5. Calibrating across categories is a sensitivity model; "once in N" is not a frequency (error in presentation)

**What z is.** The 90% intervals cover only 78–83%, and z behaves differently from one category to another. The
original's Gumbel "once in 42,727 seasons" extrapolates from about 30 residuals and is not an observed frequency.

**The calibration actually used.** Round 2 found the documents described a GPD while the code computed an exponential
tail; the labels now match the code. For each category, an exponential tail is fitted above the 75th percentile of
leader z, with its scale partially pooled across categories, leaving the leader out. GPD shapes are diagnostics only.

**Checks** (`outputs/calibration_tail_coverage.csv`, `outputs/calibration_sensitivity.csv`).
- **Leave-one-out coverage.** If the model is calibrated, about t × N leaders should get p ≤ t. For the exponential
  tail, the observed counts track that down to t = 0.001: 5 against 4.9 expected; at t = 0.002, 8 against 9.9.
- **The GPD fails it.** It gives 32 against 4.9 at t = 0.001, and calls Beamon (and, for q ≥ 0.75, DiMaggio)
  impossible. It is rejected as a calibrator.
- **Below p ≈ 0.001** no model can be checked against these data.
- **Threshold sensitivity.** Beamon's exponential p ranges from 1.2e-5 to 1.3e-4 for q between 0.60 and 0.85. These
  values reproduce the reviewer's independent table exactly.

**Main model with environment** (N = 4,940 scored records, tied co-leaders counted):

| Season | p (exponential) | 90% interval (resampling) | E = p × N | Poisson P(≥1) |
|---|---|---|---|---|
| Beamon 1968 | 6.3e-5 | 4.0e-5 – 8.4e-5 | 0.31 | 0.27 |
| DiMaggio 1941 | 1.3e-4 | 5.8e-5 – 2.3e-4 | 0.63 | 0.47 |
| Wilt 1961–62 | 3.8e-4 | 2.1e-4 – 6.0e-4 | 1.9 | 0.85 |
| Železný 1996 | 6.9e-4 | 3.6e-4 – 1.2e-3 | 3.4 | 0.97 |
| Gretzky 1985–86 | 7.2e-4 | 4.1e-4 – 1.1e-3 | 3.5 | 0.97 |

**How to read this.**
- **E below 1 is not a significance test.** Under an independent-Poisson reading, about one chance in four (Beamon)
  and one in two (DiMaggio) of at least one season this extreme somewhere in the dataset.
- **The scores aren't independent.** They share athletes, seasons and categories, so this is illustrative, not a
  family-wise test.
- **What it does show.** Beamon and DiMaggio are the hardest seasons to explain as the expected extremes of a
  ~4,900-score search. The others are within what such a search produces.

**Empirical null** (`outputs/empirical_null_top30.csv`). Withheld ordinary comparison seasons are scored like leaders.
It can only bound p down to about 1/4,660, and its upper tail contains the other great seasons seen from other
athletes' windows.

**Fix:**
- Replace "once in N seasons" with a labelled, model-based probability, given with its threshold and model range.
- Don't present it as an observed frequency or a significance level.

### 6. Data defects, verified against the source pages (error)

Full list: `outputs/sources_and_corrections.csv`, 172 rows with source, original and revised values, status and
score effect.

| Defect | Detail | Effect |
|---|---|---|
| **Impossible dates** | Three marks are dated 2027, 2109 and 2202 in the source lists themselves. They are quarantined, not re-dated. | The 2027 one created a phantom men's pole-vault season: removing it raises Duplantis 2025 from 3.02 to 4.04 (#98 to #20) and moves 14 other pole-vault scores. |
| **Wind readings in names** | A parser bug (a decimal comma in "+1,7") put wind readings into 15 athlete names. | It also created 19 duplicate athletes in deeper lists. |
| **Duplicate swimmers** | World Aquatics lists some swimmers under two IDs. | 11 merged on evidence; 4 same-name pairs kept as different people (for example Sophie Taylor of GBR and of AUS, both in 2013–14). |
| **Name-based identity** | Different people were treated as one: two Adrian Petersons, two Eddie Johnsons, Mike Dunleavy Sr. and Jr., Syl Apps Sr. and Jr., Wallace Spearmon Sr. and Jr., two Calvin Smiths. | Stable IDs fix it. Crosswalks keep one person as one across source changes (167 football players across 1998/1999; Retrosheet to Baseball-Reference IDs). |
| **NBA assists 1946–48** | Recorded under a different definition: 6.9–7.0 per team-game, against 18.6 in 1948–49. | Dividing by them made Ernie Calverley 1946–47 the second-greatest season of all (5.97). Those seasons are now excluded. |
| **Ties** | 237 ties for first were decided by row order. | Every tied leader is now scored (285 more scores). |
| **One player under two names** | Found in round 2: 9 baseball players appear under two display names on different leader boards (Hank/Henry Aaron, Joe/"Shoeless" Joe Jackson, Cal Ripken/Cal Ripken Jr., Bobby Witt/Bobby Witt Jr. and others). | The original split one player into two athlete-seasons where both names led a category in the same season (Aaron 1956, Witt 2025). Exclusion within a category was unaffected. Stable IDs fix it. |

**Effect on the published top ten:** none. The changes are concentrated in men's pole vault and a few others.

**Unresolved:** 46 athletics same-name cases with inconsistent birth dates (kept as one person, as in the
original), 2 swimming merges based on a missing birth date, and Paul Moss 1933 (383 vs 283 yards).

### 7. Four seasons separate from the rest; places 5–10 are not settled (fact, from resampling, conditional on the models)

**Specification ensemble** (point estimates; `outputs/spec_ensemble.csv`). Of 16 pre-declared specifications:
- Beamon, Gretzky, Wilt and DiMaggio are in the top ten in all 16.
- Železný 11, Ichiro 10, Rodman 9, Walsh 8 (a one-sided, possibly incomplete season), Marino and Ledecky 7.

**Resampling** (a separate statement). Each refit block-resamples the comparison seasons (blocks of 3) and reranks.
The figure is the share of refits in which the athlete's best season is in the top ten.
- It is **conditional on the model**, not a probability that the season is truly top ten.
- The Monte Carlo SE is at most ±0.016.
- The model-averaged column also draws one of the 16 specifications in each of 800 draws.

| Season | Main model (B = 1,000) | Main + environment (B = 1,000) | Model-averaged (800) |
|---|---|---|---|
| Wilt 1961–62 | 0.98 | 0.88 | 0.81 |
| Gretzky 1985–86 | 0.92 | 0.61 | 0.67 |
| DiMaggio 1941 | 0.84 | 0.80 | 0.56 |
| Beamon 1968 | 0.76 | 0.74 | 0.77 |
| Next highest | Železný 0.43 | Vance 0.41, Železný 0.39 | Walsh 0.32, Ledecky 0.30 |

**Reading.**
- **These four are clearly separated from everyone else**, but not settled in absolute terms. Under the environment
  model, or once the specification is allowed to vary, one of them (Gretzky, DiMaggio) misses the top ten in roughly
  a third to almost half of the draws.
- **Places 5–10 are not determined** by these data.

### 8. One-sided windows at the ends of the data (judgment call)

**Recent and earliest seasons are compared with one side only.** Gretchen Walsh 2026 is #1 at ±7 (8 comparison
seasons) but #107 at ±20. Ross Barnes 1876 and Tim Keefe 1880 sit at the other end.

**Incomplete seasons.** On 2 November 2026 the 2026 swimming "season" is a calendar year to date: the long-course
season may not be complete. It applies to Walsh, Douglass, McEvoy, Steenbergen and Liebmann 2026.

**Fix:**
- Flag one-sided scores (`one_sided_window_main`) and possibly incomplete seasons (`incomplete_season`) in
  `rankings_sensitivity.csv` and the calibrated tables.
- `A2_twosided` ranks only seasons with at least 5 comparison seasons on each side. Its top ten equals the main
  model's.

### 9. Robust (Huber) trend with a median-absolute-deviation scale inflates z (judgment call: don't use)

**What happens.** The scale shrinks, giving Duplantis 2025 9.88 and DiMaggio 9.50. Its 90% intervals cover only
82%, so it is reported but not recommended.

### 10. Athletics and swimming need their own treatment, and the early data are thin (judgment call, limitation)

**Depth-relative scores (`D1_depth`).** Each mark is divided by the median of that season's ranks 2–10, with 2020
left out because pandemic-thinned fields made it meaningless.
- Włodarczyk 2016 rises to #2 and Vetter 2021 to #5.
- Beamon can't be scored: the all-time list holds only four 1968 long jumpers above its cut-off, and one per season
  in the early 1960s.

**Equipment.** Swimming's full-body suits are handled by the existing 2008–09 level term. A separate 2020–26 level
for track spikes (`T1_spikes`) leaves the top ten unchanged.

**No franchise multiplier** is applied to individual sports.

## What the data cannot settle (acquisition plan, in priority order)

1. **NBA per-player opportunity.** Minutes from 1951–52 are in `data/context/nba_players.csv` but not yet used.
   Possessions before 1973–74 can only be estimated (no turnovers or offensive rebounds). Next steps:
   - per-36 and share-of-team-points estimands, from the fetched player tables;
   - play-by-play-based possession estimates where they exist.
2. **Qualified player distributions** for MLB (Baseball-Reference season batting and pitching tables, about 300
   pages) and for athletics before 2001 (World Athletics/ATFS annual lists). These would allow percentile-based
   comparators that don't depend on league size.
3. **Hitting-streak opportunity model.** Per-game hit probability from at-bats and batting average (Retrosheet game
   logs are already used for streaks), giving the probability of a k-game streak for comparable hitters.
4. **NHL ice time** before 1998–99 isn't recorded. Historical team-level power-play opportunities exist only from
   the 1960s on, and assist-crediting practice changed over the 1920s–40s (assists per goal: 0.42 in 1918, 0.63 in
   1925, 1.4 by 1944).
5. **Independent identity checks** for the 48 unresolved name cases (World Athletics athlete IDs; World Aquatics
   person pages).

## For a documentary or website audience

**What holds up best in this analysis** (to say with the method named, not as a final verdict):
- **Four seasons stand furthest above their eras** on the "dominance over rivals" question: Wilt Chamberlain 1961–62,
  Wayne Gretzky 1985–86, Joe DiMaggio 1941 and Bob Beamon 1968.
  - They are the only seasons in the top ten under all 16 specifications tested.
  - When the comparison seasons are resampled, each is in the top ten in 56–98% of refits under the main models and
    the model-averaged bootstrap. Under the original straight-line trend Beamon falls to 44%. No other season reaches
    45% in any of these.
- **Within their sports, in every model tested:** Wilt's 50.4 points a game is the most dominant NBA scoring season,
  and Gretzky's 163 assists the most dominant NHL assist season. DiMaggio's 56 games barely moves under any adjustment,
  because no league denominator applies to a hitting streak.
- **Beamon and DiMaggio are the hardest to explain as chance.** Under a calibrated tail model they are the hardest
  seasons to explain as the expected extremes of a ~4,900-score search. That is a model-based statement with a wide
  range (Beamon's probability varies ten-fold with the tail threshold), not a proof that they "beat the odds".

**What depends on the method**, to present with the method named:
- **The order of the top four.** It depends on the trend form and the environment adjustment.
- **Places 5–10.** No season there is reliably separated: their resampled rank ranges overlap heavily.
- **"Measured against his own other seasons"** (all-seasons rarity) demotes Gretzky to about #20 and Wilt to #4.
- **Adjusting for scoring environments** lifts Babe Ruth 1920 and Stephen Curry 2015–16 toward the top ten and
  lowers Gretzky.
- **League-size "corrections"** can make a 1920s hockey season look historic. That says more about the correction
  than about the season.
- **2026 swimming seasons** are compared with one side only and may be incomplete.

**Avoid:**
- "Once in 42,727 seasons" and similar figures. They are model extrapolations, not observed frequencies.
- Claims of statistical significance from expected counts below one.
- A single definitive top-ten order, or a single "greatest season".
