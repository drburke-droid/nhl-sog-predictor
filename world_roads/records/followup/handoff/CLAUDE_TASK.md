# Task for Claude: Critical continuation of the Greatest Seasons audit

You are a statistical researcher specializing in extreme values, sports analytics, causal reasoning, hierarchical models, robust inference, and reproducible computational research.

Context: The goal is to evaluate which athlete-seasons are most exceptional versus their contemporaries across Athletics, Swimming, MLB, NBA, NHL and NFL. The original score is a z-score of a season leader relative to a +/-15-year local regression baseline using the best eligible non-self performer in each comparison season. The original results ARE computationally reproducible. The new request is to test whether changing league size and changing scoring environments make cross-era comparisons unfair, particularly 1961-62 Wilt Chamberlain and 1985-86 Wayne Gretzky, without introducing larger biases in the process.

Instructions:

1. Read `CLAUDE_START_HERE.md`, `proposal/RESEARCH_PLAN.md`, `proposal/FINDINGS_AND_WARNINGS.md` and `original_project/AUDIT.md` before making methodological changes.
2. Run the supplied reproduction and validation scripts. Confirm original z and exploratory sensitivity estimates; document any mismatch. Do NOT use the earlier numbers as ground truth if you detect an implementation error.
3. Audit the experimental field-size adjustment critically. The rank-2-to-10 tail slope is ordinary least squares on `-log(rank-0.5)`, smoothed with a rolling median, not a validated extreme-value fit. Test model fit and the validity of adding a common beta*log(N_ref/N_season) offset. Look for bias from ties, rank truncation, zero slopes, player self-dependence, data scarcity, and very small early leagues.
4. Audit the scoring-environment transformation. Test points-per-team-game versus pace, opportunities, minutes, share of team scoring, offensive role, assists per goal, and historical stat definitions. When possible, separate team opportunity from individual ability. Specifically investigate the inconsistent 1961-62 NBA pace figures (126.2 for the league versus 131.1 estimated for Philadelphia) and distinguish the denominators instead of silently choosing one.
5. Extend at least preliminary standardized sensitivity analysis to MLB and NFL; athletics and swimming need distinct participation-depth and technology/event-rule treatment, not a franchise multiplier. Do not call a two-sport correction an all-sport adjustment.
6. Test field-size choices 0%, 25%, 50%, 100%; windows +/-7, +/-10, +/-15, +/-20; flat and trend; include/exclude athlete's other seasons; evaluate robust trend fitting and category standardization. Avoid choosing the setting that yields a preferred winner.
7. Develop an identifiable improved metric if the data permit. Prefer qualified player-level distributions, eligible player counts, percentile-equivalent ranks, multilevel partial pooling of category-specific tail shape, and simulation-based calibration. For scarce historical data, give honest ranges or partial identification rather than contrived point precision.
8. Validate data flags before changing source files: impossible seasons, player identity aliases, duplicated rankings, cut-short flags and denominator anomalies. Produce a source-specific correction log with evidence and record-level impact. Never silently overwrite the original data.
9. Quantify uncertainty using an appropriate block/season/bootstrap or posterior predictive strategy that preserves temporal dependence, plus model/specification uncertainty. Report rank inclusion probabilities or uncertainty ranges, not just a single z point estimate. Do not interpret the uncalibrated Gumbel `once in N` as an empirically validated return period.
10. Produce an actionable summary for a documentary or website audience separating robust takeaways from methodology-sensitive rankings.

Required outputs:

- `audit_followup.md`: concise ranked findings with evidence, error versus judgment call, likely effect on top ten, and concrete fixes.
- `model_specification.md`: explicit equations, assumptions, estimands and limitations for at least two alternatives (peer dominance and all-seasons rarity).
- `sources_and_corrections.csv`: individual data repairs with URLs and original/revised values where independently validated, and unresolved warnings flagged.
- `rankings_sensitivity.csv`: all sports, all athlete-seasons, original score plus clearly labelled alternative estimates/ranks.
- `top10_robustness.md`: tables and plots comparing current top ten with model variants; show uncertainty or intervals where estimable.
- Reproducible code with relative paths, minimal dependencies, scripts to regenerate figures and outputs, automated tests against original scores, and a `README.md` with exact commands.

Acceptance criteria:

- Distinguish verified empirical facts, numerical reproduction, assumptions, and illustrative counterfactuals.
- Preserve original files and baseline. The improved model must run as an opt-in parallel analysis.
- Use sources that can be checked. Season-year conventions are explicit (e.g. 1985-86 stored as 1986).
- Clearly label which sports and categories were adjusted, and where required inputs remain unavailable.
- Address why Babe Dye moves from roughly 31st to 3rd with a full field-size model. Treat this as a red flag rather than proof of a corrected ranking.
- No invented per-player possession data, league participation totals, or artificial confidence intervals.
- If full data acquisition is infeasible, complete the strongest defensible partial analysis and provide a prioritized acquisition plan instead of stopping.

Focus on statistical validity and auditability, not preserving an appealing existing top-ten narrative.
