"""
Multi-Cutoff Locked-Forward Evaluation — Sprint 3.

Runs locked_forward.run_locked_forward() from multiple cutoff dates
spanning the season to prove the edge survives across different
starting points and market regimes.
"""

import logging

import numpy as np
import pandas as pd

import locked_forward
import evaluation as ev

logger = logging.getLogger(__name__)

# Default cutoffs spanning the season
DEFAULT_CUTOFFS = [
    "2025-12-01",   # early season (~3 months training)
    "2026-01-01",   # mid season (~4 months training)
    "2026-01-15",   # pre all-star
    "2026-02-01",   # post all-star / trade deadline window
]


def run_multi_locked(cutoffs=None, min_edge=0.05):
    """Run locked-forward from multiple cutoff dates.

    Returns dict with per-cutoff results and consensus summary.
    """
    if cutoffs is None:
        cutoffs = DEFAULT_CUTOFFS

    results = []

    for cutoff in cutoffs:
        logger.info("=" * 60)
        logger.info("Locked-forward cutoff: %s", cutoff)
        logger.info("=" * 60)

        r = locked_forward.run_locked_forward(
            train_cutoff=cutoff, min_edge=min_edge)

        if "error" in r:
            logger.warning("Cutoff %s failed: %s", cutoff, r["error"])
            results.append({
                "cutoff": cutoff, "error": r["error"],
            })
            continue

        prod = r.get("production", {})
        boot = prod.get("bootstrap", {})

        results.append({
            "cutoff": cutoff,
            "train_size": r["train_size"],
            "test_period": r["test_period"],
            "n_bets": prod.get("n_bets", 0),
            "yield_pct": prod.get("yield", 0),
            "win_rate": prod.get("win_rate", 0),
            "brier": prod.get("brier_score"),
            "cal_slope": prod.get("calibration_slope"),
            "ci_low": boot.get("ci_low", 0),
            "ci_high": boot.get("ci_high", 0),
            "p_positive": boot.get("p_positive", 0),
            "report": prod,
        })

    # Consensus summary
    valid = [r for r in results if "error" not in r and r["n_bets"] > 0]

    consensus = {}
    if valid:
        yields = [r["yield_pct"] for r in valid]
        p_pos = [r["p_positive"] for r in valid]
        briers = [r["brier"] for r in valid if r["brier"] is not None]

        consensus = {
            "n_cutoffs": len(valid),
            "median_yield": round(float(np.median(yields)), 2),
            "min_yield": round(float(np.min(yields)), 2),
            "max_yield": round(float(np.max(yields)), 2),
            "worst_p_positive": round(float(np.min(p_pos)), 3),
            "avg_brier": round(float(np.mean(briers)), 4) if briers else None,
            "all_positive": all(y > 0 for y in yields),
        }

    return {
        "cutoffs": results,
        "consensus": consensus,
    }


def print_multi_locked(result):
    """Print formatted multi-locked-forward summary."""
    print("\n" + "=" * 100)
    print("  MULTI-CUTOFF LOCKED-FORWARD EVALUATION")
    print("=" * 100)

    cutoffs = result["cutoffs"]

    print(f"\n  {'Cutoff':>12s} {'Train':>7s} {'Test Period':>26s} "
          f"{'Bets':>5s} {'Yield':>7s} {'WR%':>6s} {'Brier':>7s} "
          f"{'CalSlope':>9s} {'CI_Low':>7s} {'CI_High':>8s} {'P>0':>6s}")
    print("  " + "-" * 96)

    for r in cutoffs:
        if "error" in r:
            print(f"  {r['cutoff']:>12s}  ** ERROR: {r['error']} **")
            continue

        print(f"  {r['cutoff']:>12s} {r['train_size']:7d} {r['test_period']:>26s} "
              f"{r['n_bets']:5d} {r['yield_pct']:+6.1f}% {r['win_rate']:5.1f}% "
              f"{r.get('brier', 0) or 0:7.4f} {r.get('cal_slope', 0) or 0:9.4f} "
              f"{r['ci_low']:+6.1f}% {r['ci_high']:+7.1f}% {r['p_positive']:5.3f}")

    # Consensus
    c = result.get("consensus", {})
    if c:
        print(f"\n  CONSENSUS ({c['n_cutoffs']} cutoffs):")
        print(f"    Median yield: {c['median_yield']:+.1f}%")
        print(f"    Range: {c['min_yield']:+.1f}% to {c['max_yield']:+.1f}%")
        print(f"    Worst P(>0): {c['worst_p_positive']:.3f}")
        if c.get("avg_brier"):
            print(f"    Avg Brier: {c['avg_brier']:.4f}")
        verdict = "PASS" if c["all_positive"] and c["worst_p_positive"] >= 0.6 else "NEEDS REVIEW"
        print(f"    Verdict: {verdict}")

    print("=" * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_multi_locked()
    print_multi_locked(result)
