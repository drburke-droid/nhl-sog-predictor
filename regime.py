"""
Regime Detection and Adaptive Calibration — Sprint 3.

Detects environmental drift (scoring changes, market shifts) and
flags when the current environment differs from the training
environment. Optionally shortens calibration window during drift.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_environment_state(bets_df, window_days=30):
    """Compute rolling environment summaries.

    Tracks: scoring rate, prediction residuals, edge distribution,
    win rates, and market vig shifts.
    """
    if bets_df.empty or "date" not in bets_df.columns:
        return []

    df = bets_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    min_date = df["date"].min()
    max_date = df["date"].max()

    states = []
    current = min_date + pd.Timedelta(days=window_days)

    while current <= max_date:
        window = df[(df["date"] > current - pd.Timedelta(days=window_days)) &
                    (df["date"] <= current)]
        if len(window) < 20:
            current += pd.Timedelta(days=7)
            continue

        edges = window["edge"].values
        probs = window["model_prob"].values
        implied = window["implied_prob"].values
        won = window["won"].astype(float).values

        # Market vig proxy: avg(model_prob + implied_prob) should be ~1 for fair market
        avg_vig = float((probs + implied).mean()) - 1.0

        # Prediction residual drift
        residual = float((probs - won).mean())

        states.append({
            "date": current.strftime("%Y-%m-%d"),
            "n_bets": len(window),
            "avg_edge": round(float(edges.mean()) * 100, 2),
            "win_rate": round(float(won.mean()) * 100, 1),
            "avg_model_prob": round(float(probs.mean()), 4),
            "avg_implied_prob": round(float(implied.mean()), 4),
            "residual_drift": round(residual, 4),
            "vig_proxy": round(avg_vig, 4),
        })

        current += pd.Timedelta(days=7)

    return states


def detect_regime_shift(states, drift_threshold=0.03, vig_threshold=0.02):
    """Detect regime shifts from environment state series.

    Flags periods where:
    - residual drift exceeds threshold (model becoming miscalibrated)
    - vig changes significantly (market structure shift)
    - win rate deviates sharply from model probability
    """
    if len(states) < 3:
        return []

    alerts = []
    recent = states[-1] if states else {}
    baseline_residual = np.mean([s["residual_drift"] for s in states[:3]])
    baseline_vig = np.mean([s["vig_proxy"] for s in states[:3]])

    for s in states:
        flags = []

        if abs(s["residual_drift"] - baseline_residual) > drift_threshold:
            flags.append(f"residual drift {s['residual_drift']:+.4f} "
                        f"(baseline {baseline_residual:+.4f})")

        if abs(s["vig_proxy"] - baseline_vig) > vig_threshold:
            flags.append(f"vig shift {s['vig_proxy']:+.4f} "
                        f"(baseline {baseline_vig:+.4f})")

        wr_diff = s["win_rate"] / 100 - s["avg_model_prob"]
        if abs(wr_diff) > 0.08:
            flags.append(f"win rate deviation {wr_diff:+.3f}")

        if flags:
            alerts.append({
                "date": s["date"],
                "flags": flags,
                "severity": "HIGH" if len(flags) >= 2 else "MODERATE",
            })

    return alerts


def print_regime_report(states, alerts):
    """Print formatted regime detection report."""
    print(f"\n{'=' * 80}")
    print("  REGIME DETECTION REPORT")
    print(f"{'=' * 80}")

    if states:
        print(f"\n  {'Date':>12s} {'Bets':>5s} {'Edge':>6s} {'WR%':>6s} "
              f"{'ModelP':>7s} {'ImplP':>7s} {'Drift':>7s} {'Vig':>6s}")
        print("  " + "-" * 60)
        for s in states:
            print(f"  {s['date']:>12s} {s['n_bets']:5d} {s['avg_edge']:+5.1f}% "
                  f"{s['win_rate']:5.1f}% {s['avg_model_prob']:7.4f} "
                  f"{s['avg_implied_prob']:7.4f} {s['residual_drift']:+6.4f} "
                  f"{s['vig_proxy']:+5.4f}")

    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    [{a['severity']}] {a['date']}: {'; '.join(a['flags'])}")
    else:
        print(f"\n  No regime shift alerts detected.")

    print(f"{'=' * 80}")
