"""Checks added for the round-2 peer review (run after analysis/run_all.py and analysis/cv_comparator.py).

    python tests/test_round2.py        # from followup/; exits non-zero on any failure

1. Identity alignment: rankings_sensitivity.csv joins original and corrected scores on stable IDs; more than
   4,000 athlete-seasons carry both; the four headline seasons appear once each with both scores; the two
   original-name collapses (Hank/Henry Aaron 1956, Bobby Witt/Bobby Witt Jr. 2025) and the three records
   removed in v2 (Ernie Calverley 1947 and 1948 NBA assists, Ernest John Obiena 2027) are flagged.
2. Calibration: the exponential tail reproduces the reviewer's independent threshold table (Beamon, DiMaggio,
   Wilt, Gretzky at q = 0.60 ... 0.85) within 1%; variant labels say "EXPONENTIAL" and never claim a GPD fit;
   expected counts use the number of scored records of that model.
3. Cross-validation: the documented decision (quad_safe within +/-15) and a positive test-half improvement.
4. Bootstrap outputs are labelled as frequencies with Monte Carlo SE.
"""
import csv, json, os, sys

F = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(F, "analysis"))
import gsmodel as G                                            # noqa: E402
import context as X                                            # noqa: E402
import calibrate as K                                          # noqa: E402

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def read(name):
    return list(csv.DictReader(open(os.path.join(F, "outputs", name), encoding="utf-8")))


def main():
    S = read("rankings_sensitivity.csv")
    both = [r for r in S if r["A0_original_z"] and r["A1_corrected_z"]]
    check(len(both) > 4000, f"{len(both)} athlete-seasons have original and corrected scores (> 4,000)")
    for nm, yr in (("Wilt Chamberlain", "1962"), ("Wayne Gretzky", "1986"), ("Joe DiMaggio", "1941"), ("Bob Beamon", "1968")):
        rows = [r for r in S if r["athlete"] == nm and r["season"] == yr]
        check(len(rows) == 1 and rows[0]["A0_original_z"] and rows[0]["A1_corrected_z"],
              f"{nm} {yr}: one row with both scores ({len(rows)} row(s))")
    col = {(r["athlete_id"], r["season"]) for r in S if r["id_status"] == "collapsed"}
    check(col == {("ref:aaronha01", "1956"), ("ref:wittbo02", "2025")}, f"collapsed identities flagged: {sorted(col)}")
    rem = {(r["category"], r["season"]) for r in S if r["id_status"] == "removed_in_v2"}
    check({("nba_apg", "1947"), ("nba_apg", "1948"), ("ath_M_PV", "2027")} <= rem | {(c, s) for c, s in rem},
          f"records removed in v2 flagged: {sorted(rem)}")
    check(len({(r["sport"], r["athlete_id"], r["season"]) for r in S}) == len(S), "one row per (sport, stable ID, season)")

    # calibration
    D2 = G.Data("v2")
    gam = {r["category"]: float(r["gamma_pooled"]) for r in read("gamma_estimates.csv")}
    recs = G.score(D2, G.Spec(ties="all", trend="quad_safe", env="power"), ctx=X.make(D2, env="power", gamma=gam))
    keys = [("ath_M_LJ", 1968), ("mlb_streak", 1941), ("nba_ppg", 1962), ("nhl_assists", 1986)]
    review = {0.60: [0.000125, 0.000182, 0.000653, 0.00113], 0.65: [0.000104, 0.000168, 0.000634, 0.00115],
              0.70: [0.0000842, 0.000146, 0.000442, 0.000773], 0.75: [0.0000634, 0.000127, 0.000383, 0.000716],
              0.80: [0.0000227, 0.0000895, 0.000424, 0.000858], 0.85: [0.0000117, 0.000134, 0.000456, 0.000756]}
    worst = 0.0
    for q, vals in review.items():
        cal, summ = K.calibrate(recs, q=q, family="exponential")
        d = {(r["cat"], r["year"]): r["p_cal"] for r in cal}
        worst = max(worst, max(abs(d[k] - v) / v for k, v in zip(keys, vals)))
    check(worst < 0.01, f"exponential calibration matches the reviewer's table (largest relative difference {worst:.4f})")
    check(summ["n_scored"] == len(recs) == 4940, f"expected counts use N = scored records ({summ['n_scored']})")
    calrows = read("calibrated_E3_main_env.csv")
    check(all(r["scored_records_N"] == "4940" for r in calrows), "calibrated_E3_main_env.csv reports N = 4,940")
    labels = {r["key"]: r["label"] for r in read("variants.csv")}
    cal_labels = [v for k, v in labels.items() if k.endswith("_cal")]
    check(cal_labels and all("EXPONENTIAL" in v and "GPD shapes are diagnostics only" in v for v in cal_labels),
          "every *_cal label names the exponential tail and calls GPD a diagnostic")
    cov = [r for r in read("calibration_tail_coverage.csv") if r["model"] == "A2_main" and r["family"] == "exponential"]
    ratio = max(int(r["observed_count"]) / float(r["expected_count"]) for r in cov if float(r["t"]) >= 0.002)
    check(ratio < 1.5, f"exponential tail: leave-one-out exceedances within 1.5x of expected for t >= 0.002 (max ratio {ratio:.2f})")

    # cross-validation decision
    dec = json.load(open(os.path.join(F, "outputs", "cv_comparator_decision.json")))
    check(dec["chosen_within_15"] == [15, "quad_safe"], f"CV choice within +/-15: {dec['chosen_within_15']}")
    imp = dec["test_improvement_chosen_vs_ols15"]
    check(imp["p05"] > 0, f"test-half improvement over the original line: {imp['mean']:.3f} (90%: {imp['p05']:.3f} to {imp['p95']:.3f})")

    # bootstrap labelling
    b = read("bootstrap_A2_main.csv")
    check("top10_frequency_in_refits" in b[0] and "monte_carlo_se" in b[0], "bootstrap tables report a frequency and its MC SE")
    print("\n" + ("ALL PASSED" if not fails else f"{len(fails)} FAILED"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
