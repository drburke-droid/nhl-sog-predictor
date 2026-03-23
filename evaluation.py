"""
Production-Grade Evaluation Framework.

Unified metrics for all models across Track A (pure), Track B (market-only),
Track C (hybrid). Implements:
- Brier score, log loss, ECE
- Calibration slope/intercept
- Bootstrap ROI confidence intervals
- Edge bucket monotonicity
- Rolling stability reports
- Side-specific diagnostics
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probability Metrics
# ---------------------------------------------------------------------------

def brier_score(probs, outcomes):
    """Brier score: mean squared error of probability predictions. Lower = better."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(probs, outcomes, eps=1e-7):
    """Binary log loss. Lower = better."""
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def expected_calibration_error(probs, outcomes, n_bins=10):
    """Expected Calibration Error: weighted average of |predicted - actual| per bin."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_pred = probs[mask].mean()
        avg_actual = outcomes[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_pred - avg_actual)
    return float(ece)


def calibration_slope_intercept(probs, outcomes):
    """Fit logistic calibration: log-odds(outcome) = a * log-odds(pred) + b.

    Perfect calibration: slope=1, intercept=0.
    Slope < 1: overconfident. Slope > 1: underconfident.
    """
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    outcomes = np.asarray(outcomes, dtype=float)
    log_odds = np.log(probs / (1 - probs))
    # Simple linear regression of outcomes on log-odds
    from numpy.polynomial.polynomial import polyfit
    try:
        intercept, slope = polyfit(log_odds, outcomes, 1)
        return float(slope), float(intercept)
    except Exception:
        return 1.0, 0.0


def calibration_table(probs, outcomes, n_bins=10):
    """Calibration table: predicted vs actual by decile bin."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bins.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "avg_pred": round(float(probs[mask].mean()), 4),
            "avg_actual": round(float(outcomes[mask].mean()), 4),
            "gap": round(float(outcomes[mask].mean() - probs[mask].mean()), 4),
        })
    return bins


# ---------------------------------------------------------------------------
# Isotonic Calibration
# ---------------------------------------------------------------------------

def fit_isotonic_calibration(probs_train, outcomes_train):
    """Fit isotonic regression calibrator on training data."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(probs_train, outcomes_train)
    return iso


def apply_calibration(iso, probs):
    """Apply fitted isotonic calibration to new probabilities."""
    return iso.predict(np.asarray(probs, dtype=float))


# ---------------------------------------------------------------------------
# Bootstrap ROI
# ---------------------------------------------------------------------------

def bootstrap_roi(profits, n_boot=2000, seed=42):
    """Bootstrap confidence interval for ROI.

    Args:
        profits: array of per-bet profits (positive = win, negative = loss)

    Returns dict with mean, median, ci_low, ci_high, p_positive.
    """
    profits = np.asarray(profits, dtype=float)
    if len(profits) == 0:
        return {"mean": 0, "median": 0, "ci_low": 0, "ci_high": 0, "p_positive": 0}

    rng = np.random.default_rng(seed)
    total_wagered = np.sum(np.abs(profits))  # approximate
    rois = []
    for _ in range(n_boot):
        sample = rng.choice(profits, size=len(profits), replace=True)
        roi = np.sum(sample) / max(np.sum(np.where(sample < 0, -sample, sample * 0.5 + abs(sample) * 0.5)), 1)
        rois.append(roi)

    rois = np.array(rois)
    return {
        "mean": round(float(np.mean(rois)) * 100, 2),
        "median": round(float(np.median(rois)) * 100, 2),
        "ci_low": round(float(np.percentile(rois, 2.5)) * 100, 2),
        "ci_high": round(float(np.percentile(rois, 97.5)) * 100, 2),
        "p_positive": round(float(np.mean(rois > 0)), 3),
    }


def bootstrap_yield(wagers, profits, n_boot=2000, seed=42):
    """Bootstrap yield (profit / wagered) with confidence intervals."""
    wagers = np.asarray(wagers, dtype=float)
    profits = np.asarray(profits, dtype=float)
    if len(profits) == 0 or np.sum(wagers) == 0:
        return {"mean": 0, "median": 0, "ci_low": 0, "ci_high": 0, "p_positive": 0}

    rng = np.random.default_rng(seed)
    yields = []
    n = len(profits)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        w = wagers[idx].sum()
        p = profits[idx].sum()
        yields.append(p / max(w, 1))

    yields = np.array(yields)
    return {
        "mean": round(float(np.mean(yields)) * 100, 2),
        "median": round(float(np.median(yields)) * 100, 2),
        "ci_low": round(float(np.percentile(yields, 2.5)) * 100, 2),
        "ci_high": round(float(np.percentile(yields, 97.5)) * 100, 2),
        "p_positive": round(float(np.mean(yields > 0)), 3),
    }


# ---------------------------------------------------------------------------
# Edge Bucket Monotonicity
# ---------------------------------------------------------------------------

def edge_monotonicity_report(edges, outcomes, odds, buckets=None):
    """Check if larger predicted edges produce better realized results.

    Returns list of bucket dicts + monotonicity flag.
    """
    edges = np.asarray(edges, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    odds = np.asarray(odds, dtype=float)

    if buckets is None:
        buckets = [(0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08),
                   (0.08, 0.10), (0.10, 0.12), (0.12, 1.0)]

    results = []
    for lo, hi in buckets:
        mask = (edges >= lo) & (edges < hi)
        if mask.sum() < 5:
            continue

        sub_outcomes = outcomes[mask]
        sub_odds = odds[mask]

        # Compute yield from actual odds
        profits = []
        for won, o in zip(sub_outcomes, sub_odds):
            dec = o / 100 + 1 if o > 0 else -100 / o + 1
            profits.append(dec - 1 if won else -1)

        avg_yield = np.mean(profits)
        wr = sub_outcomes.mean()

        results.append({
            "edge_range": f"{lo:.0%}-{hi:.0%}",
            "n": int(mask.sum()),
            "avg_edge": round(float(edges[mask].mean()) * 100, 2),
            "win_rate": round(float(wr) * 100, 1),
            "yield": round(float(avg_yield) * 100, 2),
            "se": round(float(np.std(profits) / max(np.sqrt(mask.sum()), 1)) * 100, 2),
        })

    # Check monotonicity: are yields generally increasing with edge?
    yields = [r["yield"] for r in results if r["n"] >= 10]
    monotonic = all(yields[i] <= yields[i + 1] for i in range(len(yields) - 1)) if len(yields) >= 3 else None

    return {
        "buckets": results,
        "is_monotonic": monotonic,
        "n_buckets_with_data": len(results),
    }


# ---------------------------------------------------------------------------
# Rolling Stability
# ---------------------------------------------------------------------------

def rolling_stability(dates, outcomes, profits, window_days=30):
    """Compute performance by rolling time windows.

    Returns list of window dicts with ROI, win rate, bet count.
    """
    dates = pd.to_datetime(dates)
    outcomes = np.asarray(outcomes, dtype=float)
    profits = np.asarray(profits, dtype=float)

    df = pd.DataFrame({"date": dates, "won": outcomes, "profit": profits})
    df = df.sort_values("date")

    windows = []
    start = df["date"].min()
    end = df["date"].max()
    current = start

    while current <= end:
        window_end = current + pd.Timedelta(days=window_days)
        mask = (df["date"] >= current) & (df["date"] < window_end)
        sub = df[mask]

        if len(sub) >= 5:
            total_profit = sub["profit"].sum()
            total_wagered = sub["profit"].abs().sum()  # approximate
            windows.append({
                "start": current.strftime("%Y-%m-%d"),
                "end": window_end.strftime("%Y-%m-%d"),
                "bets": len(sub),
                "win_rate": round(float(sub["won"].mean()) * 100, 1),
                "profit": round(float(total_profit), 2),
                "yield": round(float(total_profit / max(total_wagered, 1)) * 100, 1),
            })

        current += pd.Timedelta(days=window_days)

    # Flag if profitability depends on 1-2 windows
    if len(windows) >= 3:
        profits_by_window = [w["profit"] for w in windows]
        total = sum(profits_by_window)
        if total > 0:
            max_window = max(profits_by_window)
            concentrated = max_window / total > 0.60
        else:
            concentrated = True
    else:
        concentrated = None

    return {
        "windows": windows,
        "concentrated_risk": concentrated,
    }


# ---------------------------------------------------------------------------
# Side Bias Diagnostics
# ---------------------------------------------------------------------------

def side_diagnostics(sides, edges, outcomes, odds, probs=None):
    """Compute diagnostics split by bet side (OVER/UNDER).

    Returns dict per side with counts, hit rates, ROI, avg edge, etc.
    """
    sides = np.asarray(sides)
    edges = np.asarray(edges, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    odds = np.asarray(odds, dtype=float)

    result = {}
    for side_val in ["OVER", "UNDER"]:
        mask = sides == side_val
        if mask.sum() == 0:
            continue

        sub_out = outcomes[mask]
        sub_edges = edges[mask]
        sub_odds = odds[mask]

        profits = []
        wagers = []
        for won, o in zip(sub_out, sub_odds):
            dec = o / 100 + 1 if o > 0 else -100 / o + 1
            profits.append(dec - 1 if won else -1)
            wagers.append(1.0)

        avg_yield = np.mean(profits)
        boot = bootstrap_yield(np.array(wagers), np.array(profits))

        entry = {
            "n": int(mask.sum()),
            "win_rate": round(float(sub_out.mean()) * 100, 1),
            "avg_edge": round(float(sub_edges.mean()) * 100, 2),
            "avg_odds": round(float(sub_odds.mean()), 0),
            "yield": round(float(avg_yield) * 100, 2),
            "bootstrap": boot,
        }

        if probs is not None:
            sub_probs = np.asarray(probs)[mask]
            entry["avg_prob"] = round(float(sub_probs.mean()), 4)
            entry["brier"] = round(brier_score(sub_probs, sub_out), 4)

        result[side_val] = entry

    return result


# ---------------------------------------------------------------------------
# Unified Report Generator
# ---------------------------------------------------------------------------

def generate_full_report(bets_df, label="Model"):
    """Generate comprehensive evaluation report from a walk-forward bets DataFrame.

    Expected columns: date, model_prob, implied_prob, edge, ev, won, odds,
                      side (OVER/UNDER), decimal_odds
    """
    report = {"label": label, "n_bets": len(bets_df)}

    if bets_df.empty:
        return report

    b = bets_df
    probs = b["model_prob"].values if "model_prob" in b.columns else None
    outcomes = b["won"].astype(float).values
    odds = b["odds"].values if "odds" in b.columns else np.full(len(b), -110)
    edges = b["edge"].values if "edge" in b.columns else np.zeros(len(b))

    # Profits (for bootstrap)
    dec_odds = b["decimal_odds"].values if "decimal_odds" in b.columns else np.where(
        odds > 0, odds / 100 + 1, -100 / odds + 1)
    profits = np.where(outcomes, dec_odds - 1, -1)
    wagers = np.ones(len(b))

    # --- Probability Metrics ---
    if probs is not None:
        report["brier_score"] = round(brier_score(probs, outcomes), 4)
        report["log_loss"] = round(log_loss(probs, outcomes), 4)
        report["ece"] = round(expected_calibration_error(probs, outcomes), 4)
        slope, intercept = calibration_slope_intercept(probs, outcomes)
        report["calibration_slope"] = round(slope, 4)
        report["calibration_intercept"] = round(intercept, 4)
        report["calibration_table"] = calibration_table(probs, outcomes)

    # --- Betting Metrics ---
    report["win_rate"] = round(float(outcomes.mean()) * 100, 1)
    report["yield"] = round(float(np.mean(profits)) * 100, 2)
    report["total_profit"] = round(float(np.sum(profits)), 2)

    # --- Bootstrap ---
    report["bootstrap"] = bootstrap_yield(wagers, profits)

    # --- Edge Monotonicity ---
    report["edge_monotonicity"] = edge_monotonicity_report(edges, outcomes, odds)

    # --- Rolling Stability ---
    if "date" in b.columns:
        report["stability"] = rolling_stability(b["date"].values, outcomes, profits)

    # --- Side Diagnostics ---
    if "side" in b.columns:
        report["side_diagnostics"] = side_diagnostics(
            b["side"].values, edges, outcomes, odds, probs)

    return report


def print_report(report):
    """Print formatted evaluation report."""
    print(f"\n{'=' * 90}")
    print(f"  EVALUATION: {report['label']} ({report['n_bets']} bets)")
    print(f"{'=' * 90}")

    if report["n_bets"] == 0:
        print("  No bets")
        return

    # Probability metrics
    for key in ["brier_score", "log_loss", "ece", "calibration_slope", "calibration_intercept"]:
        if key in report:
            print(f"  {key}: {report[key]}")

    print(f"  Win rate: {report['win_rate']}%")
    print(f"  Yield: {report['yield']}%")

    # Bootstrap
    boot = report.get("bootstrap", {})
    if boot:
        print(f"  Bootstrap yield: {boot['mean']}% "
              f"(95% CI: {boot['ci_low']}% to {boot['ci_high']}%) "
              f"P(>0): {boot['p_positive']}")

    # Calibration table
    cal = report.get("calibration_table", [])
    if cal:
        print(f"\n  Calibration:")
        print(f"  {'Bin':>10s} {'N':>5s} {'Pred':>7s} {'Actual':>7s} {'Gap':>7s}")
        for c in cal:
            print(f"  {c['bin']:>10s} {c['n']:5d} {c['avg_pred']:7.3f} {c['avg_actual']:7.3f} {c['gap']:+7.3f}")

    # Edge monotonicity
    em = report.get("edge_monotonicity", {})
    if em.get("buckets"):
        print(f"\n  Edge Monotonicity (monotonic={em.get('is_monotonic')}):")
        print(f"  {'Range':>10s} {'N':>5s} {'AvgEdge':>8s} {'WR':>6s} {'Yield':>8s} {'SE':>6s}")
        for r in em["buckets"]:
            print(f"  {r['edge_range']:>10s} {r['n']:5d} {r['avg_edge']:+7.1f}% "
                  f"{r['win_rate']:5.1f}% {r['yield']:+7.1f}% {r['se']:5.1f}%")

    # Stability
    stab = report.get("stability", {})
    if stab.get("windows"):
        print(f"\n  Rolling Stability (concentrated={stab.get('concentrated_risk')}):")
        for w in stab["windows"]:
            print(f"  {w['start']} - {w['end']}: {w['bets']:3d} bets, "
                  f"{w['win_rate']:5.1f}% WR, {w['yield']:+6.1f}% yield")

    # Side diagnostics
    sd = report.get("side_diagnostics", {})
    if sd:
        print(f"\n  Side Diagnostics:")
        for side, d in sd.items():
            boot_s = d.get("bootstrap", {})
            print(f"  {side}: {d['n']} bets, {d['win_rate']}% WR, yield {d['yield']}%, "
                  f"avg edge {d['avg_edge']}%, P(>0) {boot_s.get('p_positive', '?')}")

    print(f"{'=' * 90}")
