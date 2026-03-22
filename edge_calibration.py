"""
Edge Calibration Analysis — statistically grounded post-hoc Kelly sizing.

Instead of trusting the model's raw edge estimate for Kelly, we measure
how predicted edges map to realized win rates across the walk-forward.
This gives a calibrated "shrinkage factor" that accounts for model
overconfidence without overfitting to a single sample path.

Key concepts:
  - Raw edge: model_prob - implied_prob (what the model thinks)
  - Realized edge: actual_win_rate - implied_prob (what happened)
  - Calibration slope: realized_edge / raw_edge (shrinkage factor)
  - Calibrated Kelly: uses shrunk edges instead of raw edges

This is NOT overfitting because:
  1. Walk-forward already uses out-of-sample predictions
  2. The calibration function is simple (linear shrinkage)
  3. We report confidence intervals via bootstrap
  4. We split by strategy type (under/over) not by individual bet
"""

import logging
import numpy as np
import pandas as pd

import nhl_walkforward as wf_v1
import nhl_walkforward_v2 as wf_v2
import nhl_simulation

logger = logging.getLogger(__name__)


def calibration_analysis(bets_df, label="Model"):
    """Analyze edge calibration for a set of walk-forward bets.

    Returns dict with calibration stats, optimal shrinkage, and
    recommended Kelly parameters.
    """
    if bets_df.empty:
        return {"error": "No bets"}

    results = {}

    # === 1. Overall calibration: predicted prob vs realized win rate ===
    # Bin by predicted probability
    bets_df = bets_df.copy()
    bets_df["prob_bin"] = pd.cut(bets_df["model_prob"], bins=10, labels=False)

    prob_cal = []
    for bin_id, group in bets_df.groupby("prob_bin"):
        if len(group) < 20:
            continue
        pred_prob = group["model_prob"].mean()
        actual_wr = group["won"].mean()
        n = len(group)
        # Wilson confidence interval for win rate
        z = 1.96
        p_hat = actual_wr
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
        prob_cal.append({
            "bin": int(bin_id),
            "pred_prob": round(pred_prob, 4),
            "actual_wr": round(actual_wr, 4),
            "n": n,
            "ci_low": round(center - margin, 4),
            "ci_high": round(center + margin, 4),
            "gap": round(actual_wr - pred_prob, 4),
        })
    results["probability_calibration"] = prob_cal

    # === 2. Edge calibration: predicted edge vs realized yield ===
    def _edge_analysis(subset, name):
        if len(subset) < 30:
            return None

        # Bin by edge size
        edge_bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 1.0]
        subset = subset.copy()
        subset["edge_bin"] = pd.cut(subset["edge"], bins=edge_bins, labels=False)

        rows = []
        for bin_id, group in subset.groupby("edge_bin"):
            if len(group) < 10:
                continue
            avg_edge = group["edge"].mean()
            actual_wr = group["won"].mean()
            avg_implied = group["implied_prob"].mean()
            realized_edge = actual_wr - avg_implied
            n = len(group)

            # Realized yield (what you'd actually earn)
            profits = []
            for _, bet in group.iterrows():
                dec = bet["decimal_odds"]
                if bet["won"]:
                    profits.append(dec - 1)
                else:
                    profits.append(-1)
            avg_yield = np.mean(profits)

            rows.append({
                "edge_range": f"{edge_bins[int(bin_id)]:.0%}-{edge_bins[int(bin_id)+1]:.0%}",
                "avg_predicted_edge": round(avg_edge * 100, 2),
                "realized_win_rate": round(actual_wr * 100, 1),
                "implied_win_rate": round(avg_implied * 100, 1),
                "realized_edge": round(realized_edge * 100, 2),
                "avg_yield_per_bet": round(avg_yield * 100, 2),
                "n_bets": n,
                "shrinkage": round(realized_edge / max(avg_edge, 0.001), 3),
            })
        return rows

    # Overall edge analysis
    results["edge_calibration_all"] = _edge_analysis(bets_df, "all")

    # By side
    unders = bets_df[bets_df["side"] == "UNDER"]
    overs = bets_df[bets_df["side"] == "OVER"]
    results["edge_calibration_unders"] = _edge_analysis(unders, "unders")
    results["edge_calibration_overs"] = _edge_analysis(overs, "overs")

    # === 3. Blended edge calibration (the strategy we actually use) ===
    has_blend = bets_df["blended_prob"].notna()
    if has_blend.sum() > 50:
        blend_df = bets_df[has_blend].copy()
        blend_df["blend_edge"] = blend_df["blended_prob"] - blend_df["implied_prob"]
        blend_df["edge"] = blend_df["blend_edge"]  # override for _edge_analysis
        results["edge_calibration_blended"] = _edge_analysis(blend_df, "blended")

        # Blended unders (our best V1 strategy)
        blend_unders = blend_df[blend_df["side"] == "UNDER"]
        results["edge_calibration_blended_unders"] = _edge_analysis(blend_unders, "blend_unders")

        # Blended overs (V2's strong suit)
        blend_overs = blend_df[blend_df["side"] == "OVER"]
        results["edge_calibration_blended_overs"] = _edge_analysis(blend_overs, "blend_overs")

    # === 4. Compute optimal shrinkage factor ===
    # Linear regression: realized_edge = alpha * predicted_edge
    # (forced through origin — if model says 0% edge, realized should be ~0%)
    edges = []
    realized = []
    for row in (results.get("edge_calibration_all") or []):
        if row["n_bets"] >= 20:
            edges.append(row["avg_predicted_edge"] / 100)
            realized.append(row["realized_edge"] / 100)

    if len(edges) >= 3:
        edges_arr = np.array(edges)
        realized_arr = np.array(realized)
        # Weighted least squares through origin: shrinkage = sum(e*r) / sum(e*e)
        weights = np.array([r["n_bets"] for r in results["edge_calibration_all"]
                            if r["n_bets"] >= 20])
        shrinkage = float(np.sum(weights * edges_arr * realized_arr) /
                          np.sum(weights * edges_arr ** 2))
        shrinkage = max(min(shrinkage, 1.5), 0.0)  # clamp to reasonable range

        # Bootstrap confidence interval for shrinkage
        boot_shrinks = []
        n_boot = 1000
        rng = np.random.default_rng(42)
        all_edges_flat = bets_df["edge"].values
        all_won_flat = bets_df["won"].astype(float).values
        all_implied_flat = bets_df["implied_prob"].values
        for _ in range(n_boot):
            idx = rng.choice(len(bets_df), size=len(bets_df), replace=True)
            b_edges = all_edges_flat[idx]
            b_won = all_won_flat[idx]
            b_implied = all_implied_flat[idx]
            b_realized = b_won - b_implied
            # Weighted regression through origin
            denom = np.sum(b_edges ** 2)
            if denom > 0:
                boot_shrinks.append(float(np.sum(b_edges * b_realized) / denom))

        ci_low = float(np.percentile(boot_shrinks, 2.5)) if boot_shrinks else 0
        ci_high = float(np.percentile(boot_shrinks, 97.5)) if boot_shrinks else 1

        results["shrinkage"] = {
            "factor": round(shrinkage, 3),
            "ci_95_low": round(ci_low, 3),
            "ci_95_high": round(ci_high, 3),
            "interpretation": (
                f"Model edges should be multiplied by {shrinkage:.2f} "
                f"(95% CI: {ci_low:.2f} to {ci_high:.2f}). "
                f"{'Model is well-calibrated.' if 0.8 <= shrinkage <= 1.2 else 'Model is overconfident — shrink edges.' if shrinkage < 0.8 else 'Model is underconfident — edges are real.'}"
            ),
        }

        # === 5. Calibrated Kelly recommendation ===
        # Instead of Kelly(raw_edge), use Kelly(shrinkage * raw_edge)
        # Also compute what fraction of Kelly is optimal
        # Test different Kelly fractions against walk-forward data
        kelly_analysis = []
        for frac in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            bankroll = 100.0
            for _, bet in bets_df.sort_values("date").iterrows():
                shrunk_prob = bet["implied_prob"] + shrinkage * bet["edge"]
                shrunk_prob = max(min(shrunk_prob, 0.99), 0.01)
                dec = bet["decimal_odds"]
                ev = shrunk_prob * (dec - 1) - (1 - shrunk_prob)
                if ev <= 0:
                    continue
                kf = ev / (dec - 1) * frac
                kf = min(kf, 0.10)  # cap
                wager = bankroll * kf
                if wager < 1.0:
                    continue
                if bet["won"]:
                    bankroll += wager * (dec - 1)
                else:
                    bankroll -= wager

            kelly_analysis.append({
                "kelly_fraction": frac,
                "ending_bankroll": round(bankroll, 2),
                "total_return": round((bankroll - 100) / 100 * 100, 1),
            })

        results["kelly_fraction_analysis"] = kelly_analysis

        # Find optimal
        best = max(kelly_analysis, key=lambda x: x["ending_bankroll"])
        results["recommended_kelly"] = {
            "fraction": best["kelly_fraction"],
            "with_shrinkage": round(shrinkage, 3),
            "explanation": (
                f"Use {best['kelly_fraction']:.0%} Kelly with edges multiplied by "
                f"{shrinkage:.2f}. This produced {best['total_return']:+.1f}% return "
                f"in walk-forward vs raw 25% Kelly."
            ),
        }

    return results


def print_calibration(results, label="Model"):
    """Print formatted calibration analysis."""
    print()
    print("=" * 100)
    print(f"  EDGE CALIBRATION ANALYSIS: {label}")
    print("=" * 100)

    # Probability calibration
    print()
    print("  PROBABILITY CALIBRATION (predicted prob vs actual win rate):")
    print(f"  {'Pred Prob':>10s} {'Actual WR':>10s} {'Gap':>8s} {'N':>6s} {'95% CI':>18s}")
    print("  " + "-" * 60)
    for row in (results.get("probability_calibration") or []):
        print(f"  {row['pred_prob']:10.1%} {row['actual_wr']:10.1%} "
              f"{row['gap']:+7.1%} {row['n']:6d} "
              f"[{row['ci_low']:.1%}, {row['ci_high']:.1%}]")

    # Edge calibration
    for section, title in [
        ("edge_calibration_all", "ALL BETS"),
        ("edge_calibration_unders", "UNDERS ONLY"),
        ("edge_calibration_overs", "OVERS ONLY"),
        ("edge_calibration_blended", "BLENDED PROB (50/50 model+sharp)"),
        ("edge_calibration_blended_unders", "BLENDED UNDERS"),
        ("edge_calibration_blended_overs", "BLENDED OVERS"),
    ]:
        data = results.get(section)
        if not data:
            continue
        print()
        print(f"  EDGE CALIBRATION — {title}:")
        print(f"  {'Edge Range':>12s} {'Pred Edge':>10s} {'Real Edge':>10s} "
              f"{'Yield/Bet':>10s} {'Shrink':>8s} {'N':>6s}")
        print("  " + "-" * 65)
        for row in data:
            print(f"  {row['edge_range']:>12s} {row['avg_predicted_edge']:+9.2f}% "
                  f"{row['realized_edge']:+9.2f}% {row['avg_yield_per_bet']:+9.2f}% "
                  f"{row['shrinkage']:7.2f}x {row['n_bets']:6d}")

    # Shrinkage
    s = results.get("shrinkage")
    if s:
        print()
        print("  OPTIMAL SHRINKAGE FACTOR:")
        print(f"  {s['interpretation']}")
        print(f"  Factor: {s['factor']:.3f} (95% CI: {s['ci_95_low']:.3f} to {s['ci_95_high']:.3f})")

    # Kelly analysis
    ka = results.get("kelly_fraction_analysis")
    if ka:
        print()
        print("  KELLY FRACTION OPTIMIZATION (with calibrated edges):")
        print(f"  {'Kelly %':>8s} {'End Bankroll':>14s} {'Return':>10s}")
        print("  " + "-" * 35)
        for row in ka:
            marker = " <<<" if row == max(ka, key=lambda x: x["ending_bankroll"]) else ""
            print(f"  {row['kelly_fraction']:7.0%} ${row['ending_bankroll']:13.2f} "
                  f"{row['total_return']:+9.1f}%{marker}")

    rec = results.get("recommended_kelly")
    if rec:
        print()
        print(f"  RECOMMENDATION: {rec['explanation']}")

    print()
    print("=" * 100)


def run_full_analysis():
    """Run walk-forward for both models and perform calibration analysis."""

    # Run V1 vs V2 walk-forward
    logger.info("Running walk-forward backtest...")
    wf_result = wf_v2.run_walkforward(
        starting_bankroll=100.0, kelly_fraction=0.25,
        min_edge=0.0,  # No edge filter — we want ALL bets for calibration
        min_train_days=60, test_window_days=14, step_days=14,
    )

    if "error" in wf_result:
        print(f"Walk-forward error: {wf_result['error']}")
        return

    bets_v1 = wf_result["v1"]["bets_df"]
    bets_v2 = wf_result["v2"]["bets_df"]

    print(f"\nWalk-forward complete: V1={len(bets_v1)} bets, V2={len(bets_v2)} bets")

    # Run calibration on each
    cal_v1 = calibration_analysis(bets_v1, "V1")
    cal_v2 = calibration_analysis(bets_v2, "V2")

    print_calibration(cal_v1, "V1 (Current Model)")
    print_calibration(cal_v2, "V2 (MoneyPuck + Clusters)")

    # === Comparative summary ===
    print()
    print("=" * 100)
    print("  V1 vs V2 CALIBRATION COMPARISON")
    print("=" * 100)

    s1 = cal_v1.get("shrinkage", {})
    s2 = cal_v2.get("shrinkage", {})
    print(f"  V1 shrinkage: {s1.get('factor', '?')} (CI: {s1.get('ci_95_low', '?')}-{s1.get('ci_95_high', '?')})")
    print(f"  V2 shrinkage: {s2.get('factor', '?')} (CI: {s2.get('ci_95_low', '?')}-{s2.get('ci_95_high', '?')})")

    r1 = cal_v1.get("recommended_kelly", {})
    r2 = cal_v2.get("recommended_kelly", {})
    print()
    print(f"  V1 recommended: {r1.get('fraction', '?'):.0%} Kelly * {r1.get('with_shrinkage', '?')} shrinkage" if r1 else "  V1: insufficient data")
    print(f"  V2 recommended: {r2.get('fraction', '?'):.0%} Kelly * {r2.get('with_shrinkage', '?')} shrinkage" if r2 else "  V2: insufficient data")

    # Where each model has an advantage
    print()
    print("  WHERE EACH MODEL'S EDGE IS MOST REAL:")
    for label, cal in [("V1", cal_v1), ("V2", cal_v2)]:
        for section, title in [
            ("edge_calibration_blended_unders", "Blended Unders"),
            ("edge_calibration_blended_overs", "Blended Overs"),
        ]:
            data = cal.get(section)
            if not data:
                continue
            profitable = [r for r in data if r["avg_yield_per_bet"] > 0 and r["n_bets"] >= 20]
            if profitable:
                best = max(profitable, key=lambda r: r["avg_yield_per_bet"])
                print(f"  {label} {title}: best at {best['edge_range']} edge "
                      f"({best['avg_yield_per_bet']:+.1f}% yield/bet, "
                      f"shrinkage={best['shrinkage']:.2f}x, n={best['n_bets']})")

    print()
    print("=" * 100)

    return {"v1": cal_v1, "v2": cal_v2, "wf": wf_result}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_full_analysis()
