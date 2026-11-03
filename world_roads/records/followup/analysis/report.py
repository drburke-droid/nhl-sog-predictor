"""Write top10_robustness.md from the outputs (tables are generated; the short text around them is fixed).

    python analysis/report.py       # from followup/, after run_all.py and plots.py
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.dirname(HERE)
OUT = os.path.join(F, "outputs")


def read(n):
    return list(csv.DictReader(open(os.path.join(OUT, n), encoding="utf-8")))


def table(rows, cols, heads):
    s = "| " + " | ".join(heads) + " |\n|" + "|".join("---" for _ in heads) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return s


def main():
    V = {r["key"]: r for r in read("variants.csv")}
    top = read("top10_by_variant.csv")
    by = {}
    for r in top:
        by.setdefault(r["variant"], []).append(r)
    show = ["A0_original", "A1_corrected", "A2_main", "E3_main_env", "A2_main_cal", "E3_main_env_cal"]
    heads = ["#"] + ["Original" if k == "A0_original" else k for k in show]
    rows = []
    for i in range(10):
        row = {"#": i + 1}
        for k in show:
            r = by.get(k, [])[i] if len(by.get(k, [])) > i else None
            row[k] = f"{r['athlete']} {r['season']} ({float(r['score']):.2f})" if r else ""
        rows.append(row)
    t1 = table(rows, ["#"] + show, heads)
    more = ["A1_w7", "A1_w10", "A1_w20", "A1_flat", "A2_w7", "A2_twosided", "B1_allseasons", "B2_allseasons_main",
            "E1_env_ratio", "E2_env_power", "F_ratio_field50", "F_ratio_field100", "F_ratio_field_estimated"]
    rows2 = []
    for i in range(10):
        row = {"#": i + 1}
        for k in more:
            r = by.get(k, [])[i] if len(by.get(k, [])) > i else None
            row[k] = f"{r['athlete']} {r['season']}" if r else ""
        rows2.append(row)
    t2 = table(rows2, ["#"] + more, ["#"] + more)
    legend = "\n".join(f"- `{k}`: {V[k]['label']}" for k in show + more if k in V)
    boots = {}
    for k in ("E3_main_env", "A2_main", "A1_corrected"):
        p = os.path.join(OUT, f"bootstrap_{k}.csv")
        if os.path.exists(p):
            boots[k] = read(f"bootstrap_{k}.csv")[:20]
    tb = ""
    for k, rs in boots.items():
        tb += f"\n**{k}** ({V[k]['label']})\n\n" + table(
            rs, ["athlete", "season", "sport", "category", "z_point", "z_p05", "z_p95", "top10_frequency_in_refits",
                 "monte_carlo_se", "rank_median", "rank_p05", "rank_p95"],
            ["Athlete", "Season", "Sport", "Category", "z", "z 5%", "z 95%", "Top-10 frequency in refits", "MC SE",
             "median rank", "rank 5%", "rank 95%"])
    ma = read("bootstrap_model_averaged.csv")[:20] if os.path.exists(os.path.join(OUT, "bootstrap_model_averaged.csv")) else []
    tma = table(ma, ["athlete", "sport", "top10_frequency_over_specs_and_refits", "monte_carlo_se", "rank_median", "rank_p05", "rank_p95"],
                ["Athlete", "Sport", "Top-10 frequency (specs x refits)", "MC SE", "median rank", "rank 5%", "rank 95%"]) if ma else ""
    ens = read("spec_ensemble.csv")[:20]
    te = table(ens, ["athlete", "sport", "specs_in_top10", "specs", "share"], ["Athlete", "Sport", "Specs with athlete in top 10", "Specs", "Share"])
    cal = read("calibrated_E3_main_env.csv")[:15]
    tc = table(cal, ["athlete", "season", "sport", "category", "z", "p_exponential_tail", "p_90pct_interval",
                     "expected_count_E_equals_p_times_N", "poisson_P_at_least_one", "one_sided_window", "incomplete_season"],
               ["Athlete", "Season", "Sport", "Category", "z", "p (exponential tail)", "p 90% interval",
                "E = p x N", "Poisson P(>=1)", "One-sided", "Incomplete season"])
    sens = [r for r in read("calibration_sensitivity.csv") if r["model"] == "E3_main_env"]
    leaders4 = [("Bob Beamon", "1968"), ("Joe DiMaggio", "1941"), ("Wilt Chamberlain", "1962"), ("Wayne Gretzky", "1986")]
    srows = []
    for q in ("0.6", "0.65", "0.7", "0.75", "0.8", "0.85"):
        for fam in ("exponential", "gpd"):
            row = {"q": q, "family": fam}
            for nm, yr in leaders4:
                r = next((x for x in sens if x["athlete"] == nm and x["season"] == yr and x["q"] == q and x["family"] == fam), None)
                row[nm] = r["p"] if r else ""
            srows.append(row)
    tsens = table(srows, ["q", "family"] + [n for n, _ in leaders4], ["q", "Tail family"] + [f"{n} {y}" for n, y in leaders4])
    cov = [r for r in read("calibration_tail_coverage.csv") if r["model"] == "A2_main"]
    tcov = table(cov, ["family", "t", "expected_count", "observed_count"], ["Tail family", "t", "Expected leaders with p <= t", "Observed (leave-one-out)"])
    nul = read("empirical_null_top30.csv")[:12]
    tnul = table(nul, ["athlete", "season", "category", "z", "null_category_seasons_at_or_above_z_other_athletes", "distinct_null_category_seasons", "examples"],
                 ["Athlete", "Season", "Category", "z (main)", "Null category-seasons >= z (other athletes)", "Null category-seasons", "Examples"])
    dec = json.load(open(os.path.join(OUT, "cv_comparator_decision.json")))
    cvs = read("cv_comparator_scores.csv")[:10]
    tcv = table(cvs, ["window", "trend", "crps_per_prediction", "crps_equal_category", "crps_equal_sport", "coverage_90",
                      "selection_half_equal_category", "test_half_equal_category"],
                ["Window", "Trend", "CRPS per prediction", "CRPS equal category", "CRPS equal sport", "90% coverage",
                 "Selection half", "Test half"])
    meta = json.load(open(os.path.join(OUT, "run_meta.json")))
    md = f"""# Top-ten robustness

Generated by `analysis/report.py` from `outputs/`. Seasons are labelled by the year they end (1985-86 = 1986).
Scores are z (standard deviations above the comparison baseline) unless the column says calibrated (-log10 p).

The ranking keeps **each athlete's best season**. "Rank" in the bootstrap tables is that page rank.

## 1. The current top ten against the improved models

{t1}
- **Original** is the published ranking. It is reproduced exactly by `tests/test_reproduction.py`.
- **A1_corrected** changes only the data: stable IDs, logged repairs, and scoring every tied leader.
- **A2_main** is the main peer-dominance model: corrected data, ±15 seasons, and a quadratic trend chosen by
  cross-validation within the brief's ±15 window.
- **E3_main_env** adds a scoring/opportunity environment adjustment for the big four. Its elasticities are estimated
  and partially pooled.
- **_cal** columns rank by a sensitivity model: a pooled-exponential category tail, -log10 P(a season leader of the
  category exceeds z). This is not a GPD fit (section 5).

## 2. Every other variant (top ten names)

{t2}
{legend}

![Rank of each athlete's best season under each model](figures/fig1_rank_heatmap.png)

## 3. Resampling uncertainty

These come from a moving-block bootstrap of comparison seasons (blocks of {meta['block_length']} seasons,
B = {meta['B']} refits for the main models, {meta.get('B_A1', '')} for A1). Each refit reweights the comparison seasons,
refits every window and reranks.

- **What the "top-10 frequency" is.** The share of refits in which the athlete's best season ranks in the top ten,
  *conditional on the model*. It is not a probability that the season is truly a top-ten season. The Monte Carlo
  SE is sqrt(f(1−f)/B).
- **What it covers.** Estimation uncertainty in the baseline and scale. It does not cover data errors or model choice.
  The model-averaged table below adds specification choice.
- **Columns.** "z 5%/95%" is the interval of the athlete's best-season score; the ranks are page ranks.
{tb}
**Model-averaged.** Each of {meta.get('B_model_averaged', '')} draws picks one of the {meta.get('specs_in_ensemble', 16)}
pre-declared specifications at random (window 7/10/15/20 × trend ols/quad_safe × environment none/estimated), then one
block resample.

{tma}
![Bootstrap rank ranges](figures/fig2_bootstrap_E3_main_env.png)

![Bootstrap rank ranges, no environment adjustment](figures/fig2_bootstrap_A2_main.png)

## 4. Specification ensemble

This is the share of {ens[0]['specs'] if ens else 16} pre-declared specifications that put the athlete in the top
ten. The specifications cross windows 7, 10, 15 and 20, trends `ols` and `quad_safe`, and environment none or
estimated, all on corrected data under estimand A.

{te}
## 5. Calibrated rarity across categories: a sensitivity model

- **p** is P(a season leader of this category exceeds z) under a **pooled-exponential category tail** fitted above the
  75th percentile, with the leader left out of its own fit. GPD shapes are diagnostics only (see below).
- **E = p × N.** N is the number of scored records in that model (4,940 for the main model with environment); every
  tied co-leader counts as a record.
- **Poisson P(≥1) = 1 − e^−E.** This is illustrative: scores share athletes, seasons and categories, so the records
  are not independent and this is not a family-wise test. An E below 1 is not a significance threshold.
- **These are extrapolations of a tail model**, not observed frequencies. Below p ≈ 0.001 the model can't be
  checked against the data.

{tc}
**Threshold and tail family** (p for the four headline seasons, main model with environment).
- **The GPD rows** use a pooled shape that is negative, which implies a bounded tail. "Beyond fitted support" means
  the GPD says the observed season is impossible: evidence against that fit, not a probability of zero.
- **Beamon's exponential p** varies more than ten-fold across thresholds.

{tsens}
**Leave-one-out check of the tail model** (main model, all 4,943 leaders).
- **What it tests.** If the model is calibrated, about t × N leaders should get p ≤ t.
- **Result.** The exponential tail tracks this down to t = 0.001. The GPD is anti-conservative by up to 6.5 times.

{tcov}
**Model-light reference (empirical null).**
- **How it's built.** Withheld non-target comparison seasons are predicted like leaders, counted as distinct
  category-seasons. The great seasons appear here as other athletes' comparators; their own are excluded when they
  are the leader being calibrated.
- **Its limit.** It can only bound p down to about 1 in the number of distinct category-seasons.

{tnul}
## 6. Choosing the trend: cross-validation on the quantity the score uses

- **Target.** The candidate's own comparator, the best non-candidate performance in the target season, predicted
  with the block {{y−1, y, y+1}} withheld.
- **Rule.** Within the brief's ±15 window (an editorial constraint), choose the trend with the lowest
  equal-category CRPS on a selection half of the categories, then measure it on the test half.
- **Chosen within ±15:** {dec['chosen_within_15'][1]}.
- **Test-half improvement over the original straight line:**
  {dec['test_improvement_chosen_vs_ols15']['mean']:.3f} CRPS units (90%:
  {dec['test_improvement_chosen_vs_ols15']['p05']:.3f} to {dec['test_improvement_chosen_vs_ols15']['p95']:.3f}).
- **Best overall on the selection half:** ±{dec['best_overall_on_selection'][0]} {dec['best_overall_on_selection'][1]}.
- **Coverage.** The 90% intervals cover only 78–83%, so z is a standardization, not a calibrated normal quantile.

{tcv}
## 7. Context figures

![NBA scoring-environment denominators](figures/fig3_nba_denominators.png)

![Babe Dye and the field offset](figures/fig4_babe_dye.png)

![Environment elasticities](figures/fig5_elasticity.png)
"""
    open(os.path.join(F, "top10_robustness.md"), "w", encoding="utf-8").write(md)
    print("top10_robustness.md written")


if __name__ == "__main__":
    main()
