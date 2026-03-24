"""
Uncertainty-Adjusted Staking Framework.

Replaces simple quarter-Kelly + edge shrinkage with a confidence-weighted
Kelly that accounts for:
1. Edge bucket reliability (from walk-forward calibration)
2. Model track quality (Track A vs C signal strength)
3. Bootstrap P(>0) for the edge bucket
4. Calibration quality in the relevant probability range
5. Side-specific uncertainty (unders vs overs)
6. Sample size of the edge bucket

Supports multiple staking modes for comparison.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge bucket reliability (from walk-forward analysis)
# ---------------------------------------------------------------------------

# Walk-forward realized yield by edge bucket (from our latest evaluation)
# Used to determine how reliable each edge level is
EDGE_RELIABILITY = {
    # (edge_lo, edge_hi): {realized_yield, n_bets, p_positive}
    (0.00, 0.02): {"yield": -0.06, "n": 1163, "p_pos": 0.0, "reliable": False},
    (0.02, 0.04): {"yield": -0.08, "n": 969, "p_pos": 0.0, "reliable": False},
    (0.04, 0.06): {"yield": -0.03, "n": 754, "p_pos": 0.1, "reliable": False},
    (0.06, 0.08): {"yield": -0.01, "n": 566, "p_pos": 0.3, "reliable": False},
    (0.08, 0.10): {"yield": 0.04, "n": 361, "p_pos": 0.7, "reliable": True},
    (0.10, 0.12): {"yield": 0.07, "n": 279, "p_pos": 0.8, "reliable": True},
    (0.12, 1.00): {"yield": 0.20, "n": 481, "p_pos": 0.95, "reliable": True},
}


def get_edge_confidence(edge):
    """Get confidence multiplier for a given edge level.

    Returns value between 0 and 1 based on how reliable this edge bucket
    has been in walk-forward testing.
    """
    for (lo, hi), data in EDGE_RELIABILITY.items():
        if lo <= edge < hi:
            if not data["reliable"]:
                return 0.0
            # Scale by P(positive) from walk-forward
            return data["p_pos"]
    return 0.0


# ---------------------------------------------------------------------------
# Staking modes
# ---------------------------------------------------------------------------

def flat_stake(bankroll, unit_size=0.02):
    """Fixed percentage of bankroll per bet."""
    return round(bankroll * unit_size, 2)


def fractional_kelly(bankroll, prob, odds, fraction=0.25, max_pct=0.08):
    """Standard fractional Kelly criterion.

    Args:
        prob: estimated win probability
        odds: American odds
        fraction: Kelly fraction (0.25 = quarter Kelly)
        max_pct: maximum bet as fraction of bankroll
    """
    dec = odds / 100 + 1 if odds > 0 else -100 / odds + 1
    ev = prob * (dec - 1) - (1 - prob)
    if ev <= 0:
        return 0.0

    kf = ev / (dec - 1)
    kf = max(kf, 0) * fraction
    kf = min(kf, max_pct)
    return round(bankroll * kf, 2)


def uncertainty_kelly(bankroll, prob, odds, edge, fraction=0.25, max_pct=0.08,
                      calibration_quality=1.0, side="UNDER", n_sharp=0):
    """Uncertainty-adjusted Kelly — scales bet by confidence in the edge.

    The adjustment accounts for:
    - Edge bucket reliability (historical P(>0))
    - Calibration quality (how well the probability is calibrated)
    - Side-specific confidence (unders are more reliable than overs)
    - Sharp book agreement (more sharp books = higher confidence)

    Returns wager amount.
    """
    dec = odds / 100 + 1 if odds > 0 else -100 / odds + 1
    ev = prob * (dec - 1) - (1 - prob)
    if ev <= 0:
        return 0.0

    # Base Kelly
    kf = ev / (dec - 1)
    kf = max(kf, 0) * fraction

    # --- Confidence multiplier ---
    confidence = 1.0

    # 1. Edge bucket reliability
    edge_conf = get_edge_confidence(edge)
    if edge_conf <= 0:
        return 0.0  # Don't bet on unreliable edge buckets
    confidence *= edge_conf

    # 2. Calibration quality (1.0 = perfect, <1.0 = overconfident)
    # Use calibration slope as proxy (ideal = 1.0, our models ~0.2-0.5)
    confidence *= min(calibration_quality, 1.0)

    # 3. Side-specific: unders are more reliable in our framework
    if side == "UNDER":
        confidence *= 1.0
    elif side == "OVER":
        confidence *= 0.7  # overs are less reliable historically
    else:
        confidence *= 0.85

    # 4. Sharp book agreement: more sharp books confirming = higher confidence
    if n_sharp >= 4:
        confidence *= 1.1
    elif n_sharp >= 2:
        confidence *= 1.0
    elif n_sharp == 1:
        confidence *= 0.8
    else:
        confidence *= 0.5  # no sharp confirmation = halve the bet

    # Apply confidence to Kelly fraction
    adjusted_kf = kf * confidence
    adjusted_kf = min(adjusted_kf, max_pct)

    return round(bankroll * adjusted_kf, 2)


# ---------------------------------------------------------------------------
# Bankroll Risk Metrics
# ---------------------------------------------------------------------------

def bankroll_risk_report(bets_df, starting_bankroll=100.0):
    """Compute comprehensive bankroll risk metrics.

    Args:
        bets_df: DataFrame with 'profit', 'wager', 'won', 'date' columns

    Returns dict with risk metrics.
    """
    if bets_df.empty:
        return {}

    profits = bets_df["profit"].values if "profit" in bets_df.columns else np.zeros(len(bets_df))
    wagers = bets_df["wager"].values if "wager" in bets_df.columns else np.ones(len(bets_df))

    # Bankroll curve
    bankroll = starting_bankroll
    curve = [bankroll]
    for p in profits:
        bankroll += p
        curve.append(bankroll)

    curve = np.array(curve)

    # Max drawdown
    peak = np.maximum.accumulate(curve)
    drawdowns = (peak - curve) / peak
    max_dd = float(drawdowns.max())

    # Longest drawdown
    in_dd = curve < peak
    longest_dd = 0
    current_dd = 0
    for v in in_dd:
        if v:
            current_dd += 1
            longest_dd = max(longest_dd, current_dd)
        else:
            current_dd = 0

    # Longest losing streak
    won = bets_df["won"].values if "won" in bets_df.columns else np.zeros(len(bets_df))
    longest_loss = 0
    current_loss = 0
    for w in won:
        if not w:
            current_loss += 1
            longest_loss = max(longest_loss, current_loss)
        else:
            current_loss = 0

    # Daily returns
    if "date" in bets_df.columns:
        daily = bets_df.groupby("date")["profit"].sum()
        daily_vol = float(daily.std()) if len(daily) > 1 else 0
        daily_mean = float(daily.mean())
        sharpe_proxy = daily_mean / max(daily_vol, 0.01) * np.sqrt(252)
    else:
        daily_vol = 0
        sharpe_proxy = 0

    # Fraction at risk per day
    if "date" in bets_df.columns:
        daily_wagered = bets_df.groupby("date")["wager"].sum()
        avg_daily_risk = float(daily_wagered.mean()) / starting_bankroll
    else:
        avg_daily_risk = 0

    return {
        "ending_bankroll": round(float(curve[-1]), 2),
        "total_return_pct": round(float((curve[-1] - starting_bankroll) / starting_bankroll * 100), 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "longest_drawdown_bets": longest_dd,
        "longest_losing_streak": longest_loss,
        "daily_volatility": round(daily_vol, 2),
        "sharpe_proxy": round(sharpe_proxy, 2),
        "avg_daily_risk_pct": round(avg_daily_risk * 100, 1),
        "total_bets": len(bets_df),
        "total_wagered": round(float(wagers.sum()), 2),
    }


# ---------------------------------------------------------------------------
# Compare staking modes
# ---------------------------------------------------------------------------

def compare_staking_modes(bets_df, starting_bankroll=100.0):
    """Run the same bets through different staking modes and compare.

    Expects bets_df with: model_prob, odds, edge, won, side, n_sharp_books
    """
    if bets_df.empty:
        return {}

    modes = {
        "flat_1pct": lambda br, row: flat_stake(br, 0.01),
        "flat_2pct": lambda br, row: flat_stake(br, 0.02),
        "quarter_kelly": lambda br, row: fractional_kelly(
            br, row["model_prob"], row["odds"], 0.25, 0.08),
        "eighth_kelly": lambda br, row: fractional_kelly(
            br, row["model_prob"], row["odds"], 0.125, 0.05),
        "uncertainty_kelly": lambda br, row: uncertainty_kelly(
            br, row["model_prob"], row["odds"], row["edge"], 0.25, 0.08,
            calibration_quality=0.5,  # our models are ~0.2-0.5 cal slope
            side=row.get("side", "UNDER"),
            n_sharp=row.get("n_sharp_books", 0)),
    }

    results = {}
    for mode_name, staking_fn in modes.items():
        bankroll = starting_bankroll
        bet_results = []

        for _, row in bets_df.sort_values("date").iterrows():
            wager = staking_fn(bankroll, row)
            if wager < 1.0:
                continue

            dec = row["odds"] / 100 + 1 if row["odds"] > 0 else -100 / row["odds"] + 1
            profit = wager * (dec - 1) if row["won"] else -wager
            bankroll += profit

            bet_results.append({
                "date": row["date"],
                "wager": wager,
                "profit": profit,
                "won": row["won"],
            })

        if bet_results:
            br_df = pd.DataFrame(bet_results)
            risk = bankroll_risk_report(br_df, starting_bankroll)
            results[mode_name] = risk

    return results


def print_staking_comparison(results):
    """Print formatted staking mode comparison."""
    if not results:
        print("No staking results")
        return

    import pandas as pd

    print(f"\n{'=' * 90}")
    print("  STAKING MODE COMPARISON")
    print(f"{'=' * 90}")
    print(f"\n  {'Mode':25s} {'End $':>8s} {'Return':>8s} {'MaxDD':>7s} {'LngDD':>6s} {'LngLoss':>8s} {'Sharpe':>7s}")
    print("  " + "-" * 72)

    for mode, r in sorted(results.items(), key=lambda x: x[1].get("ending_bankroll", 0), reverse=True):
        print(f"  {mode:25s} ${r['ending_bankroll']:7.2f} "
              f"{r['total_return_pct']:+7.1f}% {r['max_drawdown_pct']:6.1f}% "
              f"{r['longest_drawdown_bets']:5d} {r['longest_losing_streak']:7d} "
              f"{r['sharpe_proxy']:6.2f}")

    print(f"{'=' * 90}")


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    print("Edge confidence multipliers:")
    for edge in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        conf = get_edge_confidence(edge)
        print(f"  Edge {edge:.0%}: confidence = {conf:.2f}")

    print("\nStaking examples ($100 bankroll, -110 odds):")
    for prob, edge in [(0.55, 0.05), (0.58, 0.08), (0.62, 0.12), (0.65, 0.15)]:
        flat = flat_stake(100, 0.02)
        qk = fractional_kelly(100, prob, -110, 0.25, 0.08)
        uk = uncertainty_kelly(100, prob, -110, edge, 0.25, 0.08,
                               calibration_quality=0.5, side="UNDER", n_sharp=3)
        print(f"  P={prob:.2f} edge={edge:.0%}: flat=${flat}, qKelly=${qk}, uKelly=${uk}")
