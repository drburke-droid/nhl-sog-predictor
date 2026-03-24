"""
Tail Calibration Analysis — Sprint 3.

Evaluates distribution model calibration in the tails where
betting value often lives. Mean/center calibration can look
fine while tail probabilities remain wrong.
"""

import logging

import numpy as np
import pandas as pd

import distribution_model as dist_model

logger = logging.getLogger(__name__)


def tail_probability_report(actuals, mean_preds, variance_preds):
    """Evaluate calibration at tail thresholds.

    For SOG count data, tails are where prop betting happens:
    - P(X <= 1): low-end tail (under 1.5)
    - P(X >= 4): high-end tail (over 3.5)
    - P(X >= 5): extreme high (over 4.5)
    """
    actuals = np.asarray(actuals, dtype=int)

    tails = {
        "P(X <= 1)": {"pred_fn": lambda m, v: dist_model.negbin_prob_under(m, v, 1.5),
                       "actual_fn": lambda a: (a <= 1).astype(float)},
        "P(X >= 3)": {"pred_fn": lambda m, v: dist_model.negbin_prob_over(m, v, 2.5),
                       "actual_fn": lambda a: (a >= 3).astype(float)},
        "P(X >= 4)": {"pred_fn": lambda m, v: dist_model.negbin_prob_over(m, v, 3.5),
                       "actual_fn": lambda a: (a >= 4).astype(float)},
        "P(X >= 5)": {"pred_fn": lambda m, v: dist_model.negbin_prob_over(m, v, 4.5),
                       "actual_fn": lambda a: (a >= 5).astype(float)},
        "P(X >= 6)": {"pred_fn": lambda m, v: dist_model.negbin_prob_over(m, v, 5.5),
                       "actual_fn": lambda a: (a >= 6).astype(float)},
    }

    results = []
    for name, spec in tails.items():
        pred_probs = np.array([
            spec["pred_fn"](m, v)
            for m, v in zip(mean_preds, variance_preds)
        ])
        actual_flags = spec["actual_fn"](actuals)

        pred_rate = float(pred_probs.mean())
        actual_rate = float(actual_flags.mean())
        brier = float(np.mean((pred_probs - actual_flags) ** 2))

        results.append({
            "tail": name,
            "n": len(actuals),
            "predicted_rate": round(pred_rate, 4),
            "actual_rate": round(actual_rate, 4),
            "gap": round(actual_rate - pred_rate, 4),
            "brier": round(brier, 4),
        })

    return results


def line_family_tail_report(bets_df, lines=None):
    """Evaluate calibration by prop line family.

    For each line, compare model P(over) vs actual over rate.
    """
    if bets_df.empty:
        return []

    if lines is None:
        lines = sorted(bets_df["line"].unique())

    results = []
    for line in lines:
        overs = bets_df[(bets_df["line"] == line) & (bets_df["side"] == "OVER")]
        unders = bets_df[(bets_df["line"] == line) & (bets_df["side"] == "UNDER")]

        for side_label, sub in [("OVER", overs), ("UNDER", unders)]:
            if len(sub) < 10:
                continue

            probs = sub["model_prob"].values
            outcomes = sub["won"].astype(float).values
            brier = float(np.mean((probs - outcomes) ** 2))

            results.append({
                "line": line,
                "side": side_label,
                "n": len(sub),
                "avg_model_prob": round(float(probs.mean()), 4),
                "actual_hit_rate": round(float(outcomes.mean()), 4),
                "gap": round(float(outcomes.mean() - probs.mean()), 4),
                "brier": round(brier, 4),
            })

    return results


def print_tail_report(tail_results, line_results=None):
    """Print formatted tail calibration report."""
    print(f"\n{'=' * 70}")
    print("  TAIL CALIBRATION REPORT")
    print(f"{'=' * 70}")

    if tail_results:
        print(f"\n  {'Tail':12s} {'N':>6s} {'Predicted':>10s} {'Actual':>8s} "
              f"{'Gap':>8s} {'Brier':>7s}")
        print("  " + "-" * 55)
        for r in tail_results:
            print(f"  {r['tail']:12s} {r['n']:6d} {r['predicted_rate']:10.4f} "
                  f"{r['actual_rate']:8.4f} {r['gap']:+7.4f} {r['brier']:7.4f}")

    if line_results:
        print(f"\n  BY LINE FAMILY:")
        print(f"  {'Line':>5s} {'Side':6s} {'N':>5s} {'AvgProb':>8s} "
              f"{'HitRate':>8s} {'Gap':>8s} {'Brier':>7s}")
        print("  " + "-" * 52)
        for r in line_results:
            print(f"  {r['line']:5.1f} {r['side']:6s} {r['n']:5d} "
                  f"{r['avg_model_prob']:8.4f} {r['actual_hit_rate']:8.4f} "
                  f"{r['gap']:+7.4f} {r['brier']:7.4f}")

    print(f"{'=' * 70}")
