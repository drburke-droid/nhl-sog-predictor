"""
Side Bias Deep Dive — Sprint 3.

Diagnoses and corrects over/under asymmetry at the model level,
not just the staking layer. If the model systematically over-predicts,
unders look artificially good even without real edge.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import evaluation as ev

logger = logging.getLogger(__name__)


def side_residual_report(bets_df):
    """Compute residual diagnostics by side."""
    if bets_df.empty or "side" not in bets_df.columns:
        return {}

    results = {}
    for side in ["OVER", "UNDER"]:
        sub = bets_df[bets_df["side"] == side]
        if len(sub) < 10:
            continue

        probs = sub["model_prob"].values
        outcomes = sub["won"].astype(float).values
        edges = sub["edge"].values

        # Calibration
        slope, intercept = ev.calibration_slope_intercept(probs, outcomes)

        # Average residual (predicted prob - actual hit rate)
        avg_residual = float(probs.mean() - outcomes.mean())

        # Edge inflation: average edge vs realized yield
        dec_odds = sub["decimal_odds"].values
        profits = np.where(outcomes, dec_odds - 1, -1)
        realized_yield = float(np.mean(profits))

        results[side] = {
            "n": len(sub),
            "avg_model_prob": round(float(probs.mean()), 4),
            "actual_hit_rate": round(float(outcomes.mean()), 4),
            "avg_residual": round(avg_residual, 4),
            "brier": round(ev.brier_score(probs, outcomes), 4),
            "calibration_slope": round(slope, 4),
            "calibration_intercept": round(intercept, 4),
            "avg_edge": round(float(edges.mean()) * 100, 2),
            "realized_yield": round(realized_yield * 100, 2),
            "edge_inflation": round(float(edges.mean()) * 100 - realized_yield * 100, 2),
        }

    return results


def fit_side_specific_calibration(bets_df, prob_col="model_prob",
                                   outcome_col="won"):
    """Fit separate isotonic calibration for overs and unders.

    Returns dict of side -> IsotonicRegression.
    """
    calibrators = {}
    for side in ["OVER", "UNDER"]:
        sub = bets_df[bets_df["side"] == side]
        if len(sub) < 50:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(sub[prob_col].values, sub[outcome_col].astype(float).values)
        calibrators[side] = iso
    return calibrators


def compare_side_adjustment_methods(bets_df):
    """Compare three approaches to side asymmetry:

    1. No adjustment (baseline)
    2. Side-specific calibration
    3. Side-specific edge threshold

    Returns comparison dict.
    """
    if bets_df.empty:
        return {}

    results = {}

    # 1. Baseline (no adjustment)
    probs = bets_df["model_prob"].values
    outcomes = bets_df["won"].astype(float).values
    dec_odds = bets_df["decimal_odds"].values
    profits = np.where(outcomes, dec_odds - 1, -1)
    results["baseline"] = {
        "brier": round(ev.brier_score(probs, outcomes), 4),
        "yield": round(float(np.mean(profits)) * 100, 2),
    }

    # 2. Side-specific calibration
    calibrators = fit_side_specific_calibration(bets_df)
    if calibrators:
        cal_probs = probs.copy()
        for side, iso in calibrators.items():
            mask = bets_df["side"] == side
            cal_probs[mask.values] = iso.predict(probs[mask.values])

        results["side_calibrated"] = {
            "brier": round(ev.brier_score(cal_probs, outcomes), 4),
            "yield": round(float(np.mean(profits)) * 100, 2),  # yield unchanged (same bets)
            "brier_improvement": round(
                ev.brier_score(probs, outcomes) - ev.brier_score(cal_probs, outcomes), 4),
        }

    # 3. Side-specific threshold (8% for overs, 5% for unders)
    mask_over = (bets_df["side"] == "OVER") & (bets_df["edge"] >= 0.08)
    mask_under = (bets_df["side"] == "UNDER") & (bets_df["edge"] >= 0.05)
    filtered = bets_df[mask_over | mask_under]
    if len(filtered) >= 20:
        f_probs = filtered["model_prob"].values
        f_out = filtered["won"].astype(float).values
        f_dec = filtered["decimal_odds"].values
        f_profits = np.where(f_out, f_dec - 1, -1)
        results["side_threshold"] = {
            "n_bets": len(filtered),
            "brier": round(ev.brier_score(f_probs, f_out), 4),
            "yield": round(float(np.mean(f_profits)) * 100, 2),
        }

    return results


def print_side_bias_report(residuals, comparison):
    """Print formatted side bias report."""
    print(f"\n{'=' * 80}")
    print("  SIDE BIAS DEEP DIVE")
    print(f"{'=' * 80}")

    if residuals:
        print(f"\n  {'Side':8s} {'N':>6s} {'AvgProb':>8s} {'HitRate':>8s} "
              f"{'Residual':>9s} {'Brier':>7s} {'CalSlope':>9s} "
              f"{'AvgEdge':>8s} {'Yield':>7s} {'Inflation':>10s}")
        print("  " + "-" * 78)
        for side, r in residuals.items():
            print(f"  {side:8s} {r['n']:6d} {r['avg_model_prob']:8.4f} "
                  f"{r['actual_hit_rate']:8.4f} {r['avg_residual']:+8.4f} "
                  f"{r['brier']:7.4f} {r['calibration_slope']:9.4f} "
                  f"{r['avg_edge']:+7.1f}% {r['realized_yield']:+6.1f}% "
                  f"{r['edge_inflation']:+9.1f}%")

    if comparison:
        print(f"\n  ADJUSTMENT COMPARISON:")
        for method, c in comparison.items():
            parts = f"  {method:20s}: Brier={c.get('brier', 'N/A')}"
            if "yield" in c:
                parts += f", Yield={c['yield']:+.1f}%"
            if "brier_improvement" in c:
                parts += f", Brier improvement={c['brier_improvement']:+.4f}"
            if "n_bets" in c:
                parts += f" ({c['n_bets']} bets)"
            print(parts)

    print(f"{'=' * 80}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import nhl_walkforward as wf
    result = wf.run_walkforward(min_edge=0.03)
    bets_df = result.get("bets_df", pd.DataFrame())

    if not bets_df.empty:
        ev_plus = bets_df[bets_df["ev"] > 0]
        residuals = side_residual_report(ev_plus)
        comparison = compare_side_adjustment_methods(ev_plus)
        print_side_bias_report(residuals, comparison)
