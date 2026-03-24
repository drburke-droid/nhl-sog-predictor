"""
Stress Testing Suite — Sprint 3.

Tests whether the framework survives perturbations:
- Remove top edges
- Add prediction noise
- Widen vig assumptions
- Remove feature families

If the system is fragile, stress tests reveal it.
"""

import logging

import numpy as np
import pandas as pd

import evaluation as ev

logger = logging.getLogger(__name__)


def _compute_yield(bets_df):
    """Flat-stake yield from bets DataFrame."""
    if bets_df.empty:
        return 0.0
    dec = bets_df["decimal_odds"].values
    won = bets_df["won"].astype(float).values
    profits = np.where(won, dec - 1, -1)
    return float(np.mean(profits)) * 100


def _compute_drawdown(bets_df):
    """Max drawdown % from bets DataFrame."""
    if bets_df.empty:
        return 0.0
    dec = bets_df["decimal_odds"].values
    won = bets_df["won"].astype(float).values
    profits = np.where(won, dec - 1, -1)
    cum = np.cumsum(profits) + 100  # start at $100
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(dd.max()) * 100


def stress_remove_top_edges(bets_df, pct=0.10):
    """Remove top X% of highest-edge bets and recompute."""
    if bets_df.empty:
        return {}
    n_remove = max(1, int(len(bets_df) * pct))
    threshold = bets_df["edge"].nlargest(n_remove).iloc[-1]
    stressed = bets_df[bets_df["edge"] < threshold].copy()
    return {
        "name": f"Remove top {pct:.0%} edges",
        "removed": len(bets_df) - len(stressed),
        "remaining": len(stressed),
        "yield": round(_compute_yield(stressed), 2),
        "max_dd": round(_compute_drawdown(stressed), 1),
    }


def stress_add_prediction_noise(bets_df, noise_std=0.03, seed=42):
    """Add Gaussian noise to model probabilities and recompute edges."""
    if bets_df.empty:
        return {}
    rng = np.random.default_rng(seed)
    stressed = bets_df.copy()
    noise = rng.normal(0, noise_std, len(stressed))
    stressed["model_prob"] = np.clip(stressed["model_prob"] + noise, 0.01, 0.99)
    stressed["edge"] = stressed["model_prob"] - stressed["implied_prob"]
    # Re-filter: only keep positive edge
    stressed = stressed[stressed["edge"] > 0]
    return {
        "name": f"Add noise (std={noise_std})",
        "remaining": len(stressed),
        "yield": round(_compute_yield(stressed), 2),
        "max_dd": round(_compute_drawdown(stressed), 1),
    }


def stress_price_worsening(bets_df, vig_increase=0.02):
    """Simulate worse odds by reducing decimal payout."""
    if bets_df.empty:
        return {}
    stressed = bets_df.copy()
    # Reduce decimal odds (simulate higher vig)
    stressed["decimal_odds"] = stressed["decimal_odds"] - vig_increase
    stressed["decimal_odds"] = stressed["decimal_odds"].clip(lower=1.01)
    return {
        "name": f"Vig increase +{vig_increase:.0%}",
        "remaining": len(stressed),
        "yield": round(_compute_yield(stressed), 2),
        "max_dd": round(_compute_drawdown(stressed), 1),
    }


def stress_test_summary(bets_df):
    """Run full stress test suite on a filtered set of bets."""
    if bets_df.empty:
        return []

    baseline_yield = round(_compute_yield(bets_df), 2)
    baseline_dd = round(_compute_drawdown(bets_df), 1)

    tests = [
        {"name": "BASELINE", "remaining": len(bets_df),
         "yield": baseline_yield, "max_dd": baseline_dd},
        stress_remove_top_edges(bets_df, 0.10),
        stress_remove_top_edges(bets_df, 0.20),
        stress_add_prediction_noise(bets_df, 0.02),
        stress_add_prediction_noise(bets_df, 0.05),
        stress_price_worsening(bets_df, 0.02),
        stress_price_worsening(bets_df, 0.05),
    ]

    # Add degradation metrics
    for t in tests:
        if t.get("yield") is not None and t["name"] != "BASELINE":
            t["yield_delta"] = round(t["yield"] - baseline_yield, 2)
            t["survives"] = t["yield"] > 0
        else:
            t["yield_delta"] = 0
            t["survives"] = True

    return tests


def print_stress_tests(tests):
    """Print formatted stress test results."""
    print(f"\n{'=' * 80}")
    print("  STRESS TEST RESULTS")
    print(f"{'=' * 80}")

    print(f"\n  {'Test':35s} {'Bets':>5s} {'Yield':>7s} {'Delta':>7s} "
          f"{'MaxDD':>6s} {'Survives':>9s}")
    print("  " + "-" * 72)

    for t in tests:
        surv = "YES" if t.get("survives", True) else "**NO**"
        delta = f"{t.get('yield_delta', 0):+6.1f}%" if t["name"] != "BASELINE" else "  ---"
        print(f"  {t['name']:35s} {t.get('remaining', 0):5d} "
              f"{t.get('yield', 0):+6.1f}% {delta} "
              f"{t.get('max_dd', 0):5.1f}% {surv:>9s}")

    failed = sum(1 for t in tests if not t.get("survives", True))
    print(f"\n  {len(tests) - 1} stress tests, {failed} failures")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import nhl_walkforward as wf
    result = wf.run_walkforward(min_edge=0.03)
    bets_df = result.get("bets_df", pd.DataFrame())

    if not bets_df.empty:
        # Production filter
        prod = bets_df[
            (bets_df["side"] == "UNDER") &
            (bets_df["has_soft"] == True) &
            (bets_df["edge"] >= 0.05)
        ]
        tests = stress_test_summary(prod)
        print_stress_tests(tests)
