"""
MLB Pitcher Strikeout Model — Comprehensive Evaluation Module.

Benchmarks, segmented diagnostics, calibration analysis, and tail diagnostics.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, brier_score_loss
from datetime import timedelta

logger = logging.getLogger(__name__)


def _mean_bias_error(actual, predicted):
    return float(np.mean(predicted - actual))


def _segment_metrics(actual, predicted, label=""):
    """Compute metrics for a segment."""
    n = len(actual)
    if n == 0:
        return {"n": 0, "label": label}
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    bias = _mean_bias_error(actual, predicted)
    return {
        "label": label, "n": n,
        "mae": round(mae, 3), "rmse": round(rmse, 3),
        "bias": round(bias, 3),
        "avg_actual": round(float(np.mean(actual)), 2),
        "avg_predicted": round(float(np.mean(predicted)), 2),
    }


def compute_baselines(df):
    """Compute baseline predictions on test dataframe.

    df must have columns: strikeouts, season_avg_k, roll_3_k, roll_5_k,
    baseline_bf, baseline_k_rate, opp_k_rate, batters_faced

    Returns dict mapping baseline name -> array of predictions.
    """
    baselines = {}
    # 1. Season average K
    baselines["season_avg"] = df["season_avg_k"].values
    # 2. Rolling 5
    baselines["rolling_5"] = df["roll_5_k"].values
    # 3. Rolling 3
    baselines["rolling_3"] = df["roll_3_k"].values
    # 4. BF x season K rate
    baselines["bf_x_k_rate"] = (df["baseline_bf"] * df["baseline_k_rate"]).values
    # 5. Opponent K rate only (assume league avg BF ~25, multiply by opp K rate)
    baselines["opp_k_only"] = (25.0 * df["opp_k_rate"]).values
    return baselines


def evaluate_baselines(actual, baselines):
    """Evaluate all baselines against actual."""
    results = {}
    for name, preds in baselines.items():
        results[name] = _segment_metrics(actual, preds, label=name)
    return results


def evaluate_segments(actual, predicted, df):
    """Segmented evaluation by multiple dimensions."""
    segments = {}

    # By pitcher talent (using baseline_k_rate * baseline_bf as proxy for expected K)
    expected_k = df["baseline_bf"].values * df["baseline_k_rate"].values
    for lo, hi, label in [(0, 4, "low_k"), (4, 6, "mid_k"), (6, 8, "high_k"), (8, 99, "elite_k")]:
        mask = (expected_k >= lo) & (expected_k < hi)
        segments[f"talent_{label}"] = _segment_metrics(actual[mask], predicted[mask], label)

    # By BF bucket
    bf = df["batters_faced"].values
    for lo, hi, label in [(0, 20, "low_bf"), (20, 26, "mid_bf"), (26, 99, "high_bf")]:
        mask = (bf >= lo) & (bf < hi)
        segments[f"opp_{label}"] = _segment_metrics(actual[mask], predicted[mask], label)

    # By prop line buckets (based on predicted)
    for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
        near = (predicted >= line - 0.75) & (predicted <= line + 0.75)
        segments[f"near_line_{line}"] = _segment_metrics(actual[near], predicted[near], f"near_{line}")

    # Home vs away
    is_home = df["is_home"].values.astype(bool)
    segments["home"] = _segment_metrics(actual[is_home], predicted[is_home], "home")
    segments["away"] = _segment_metrics(actual[~is_home], predicted[~is_home], "away")

    return segments


def evaluate_tails(actual, predicted):
    """Tail diagnostics by predicted K range."""
    tails = {}
    for lo, hi, label in [(0, 3, "<3"), (3, 5, "3-5"), (5, 7, "5-7"), (7, 9, "7-9"), (9, 99, "9+")]:
        mask = (predicted >= lo) & (predicted < hi)
        tails[label] = _segment_metrics(actual[mask], predicted[mask], label)
    return tails


def evaluate_calibration(actual, predicted, sim_func=None):
    """Calibration analysis for prop thresholds.

    If sim_func is provided, it should take (pred_bf, pred_kbf) and return
    simulation dict with P_over_X.X keys. Otherwise uses simple threshold comparison.

    For simple calibration without simulation:
    - Bin predictions into ranges
    - Compare predicted mean vs actual mean per bin
    """
    calibration = {}

    # Simple prediction calibration (binned)
    bins = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 15)]
    bin_cal = []
    for lo, hi in bins:
        mask = (predicted >= lo) & (predicted < hi)
        n = mask.sum()
        if n > 0:
            bin_cal.append({
                "bin": f"{lo}-{hi}", "n": int(n),
                "avg_predicted": round(float(predicted[mask].mean()), 2),
                "avg_actual": round(float(actual[mask].mean()), 2),
                "bias": round(float(predicted[mask].mean() - actual[mask].mean()), 3),
            })
    calibration["prediction_bins"] = bin_cal

    # Threshold calibration
    threshold_cal = []
    for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
        actual_over = (actual > line).astype(float)
        predicted_over = (predicted > line).astype(float)

        # Model "probability" from simple threshold (binary)
        model_pct = float(predicted_over.mean())
        actual_pct = float(actual_over.mean())

        # Brier score (using binary prediction as probability - rough)
        brier = brier_score_loss(actual_over, predicted_over)

        # Accuracy
        correct = (predicted_over == actual_over).mean()

        threshold_cal.append({
            "line": line,
            "n": len(actual),
            "model_over_pct": round(model_pct, 3),
            "actual_over_pct": round(actual_pct, 3),
            "accuracy": round(float(correct), 3),
            "brier_score": round(float(brier), 4),
        })
    calibration["thresholds"] = threshold_cal

    return calibration


def run_evaluation() -> dict:
    """Run comprehensive model evaluation.

    Returns a dict with all evaluation results.
    """
    import mlb_model

    logger.info("Building feature matrix for evaluation...")
    df = mlb_model._build_feature_dataframe()

    if df.empty or len(df) < 50:
        return {"error": "Not enough data for evaluation"}

    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - timedelta(days=14)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff].copy()

    if len(test_df) < 10:
        return {"error": "Not enough test data"}

    actual = test_df["strikeouts"].values

    # Model predictions (BF x K/BF)
    bf_pred = test_df["baseline_bf"].values + mlb_model._model_bf.predict(test_df[mlb_model.BF_FEATURES].values)
    bf_pred = np.clip(bf_pred, 10, 40)
    kbf_pred = test_df["baseline_k_rate"].values + mlb_model._model_kbf.predict(test_df[mlb_model.KBF_FEATURES].values)
    kbf_pred = np.clip(kbf_pred, 0.0, 0.60)
    model_pred = bf_pred * kbf_pred
    model_pred = np.maximum(model_pred, 0.0)

    # Reconstruct approximate baselines from available columns.
    # _build_feature_dataframe() does not store rolling K averages as separate
    # columns; they are used internally to compute the baseline fields.  The
    # baselines we CAN compute from the stored columns are:
    #   weighted_baseline = baseline_bf * baseline_k_rate
    #   opp_k_baseline    = 25 * opp_k_rate
    test_df["season_avg_k_approx"] = test_df["baseline_bf"] * test_df["baseline_k_rate"]

    results = {}

    # 1. Global metrics
    mae = mean_absolute_error(actual, model_pred)
    rmse = np.sqrt(mean_squared_error(actual, model_pred))
    r2 = r2_score(actual, model_pred) if len(actual) > 1 else 0
    bias = _mean_bias_error(actual, model_pred)

    results["global"] = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r_squared": round(r2, 3),
        "mean_bias": round(bias, 3),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "holdout_period": f"{cutoff.strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
    }

    # 2. Baseline comparisons
    baseline_preds = {
        "weighted_baseline": (test_df["baseline_bf"] * test_df["baseline_k_rate"]).values,
        "opp_k_baseline": (25.0 * test_df["opp_k_rate"]).values,
    }

    baseline_results = {}
    for name, preds in baseline_preds.items():
        baseline_results[name] = _segment_metrics(actual, preds, name)
    baseline_results["model"] = _segment_metrics(actual, model_pred, "model")
    results["baselines"] = baseline_results

    # 3. Segmented evaluation
    results["segments"] = evaluate_segments(actual, model_pred, test_df)

    # 4. Tail diagnostics
    results["tails"] = evaluate_tails(actual, model_pred)

    # 5. Calibration
    results["calibration"] = evaluate_calibration(actual, model_pred)

    # 6. Monte Carlo calibration (if simulation module available)
    try:
        import mlb_simulation

        # Run simulation for each test sample and compute Brier scores
        sim_over_probs = {line: [] for line in [3.5, 4.5, 5.5, 6.5, 7.5]}
        actual_overs = {line: [] for line in [3.5, 4.5, 5.5, 6.5, 7.5]}

        for i in range(len(test_df)):
            sim = mlb_simulation.simulate_strikeouts(
                bf_pred[i], kbf_pred[i],
                bf_std=bf_pred[i] * 0.18,
                kbf_std=kbf_pred[i] * 0.20,
                n_sims=2000, seed=42 + i
            )
            for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
                sim_over_probs[line].append(sim[f"P_over_{line:.1f}"])
                actual_overs[line].append(1.0 if actual[i] > line else 0.0)

        sim_brier = {}
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            probs = np.array(sim_over_probs[line])
            actuals = np.array(actual_overs[line])
            sim_brier[f"over_{line}"] = {
                "brier_score": round(float(brier_score_loss(actuals, probs)), 4),
                "avg_model_prob": round(float(probs.mean()), 3),
                "actual_rate": round(float(actuals.mean()), 3),
            }
        results["simulation_brier"] = sim_brier

    except Exception as e:
        logger.warning("Simulation calibration skipped: %s", e)

    logger.info("Evaluation complete: MAE=%.3f RMSE=%.3f R²=%.3f", mae, rmse, r2)
    return results


def print_evaluation(results: dict):
    """Pretty-print evaluation results."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    g = results.get("global", {})
    print("=" * 70)
    print("MLB PITCHER STRIKEOUT MODEL — EVALUATION REPORT")
    print("=" * 70)
    print(f"\nHoldout: {g.get('holdout_period', 'N/A')}")
    print(f"Train: {g.get('train_samples', 0)} | Test: {g.get('test_samples', 0)}")
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 35)
    for k in ["mae", "rmse", "r_squared", "mean_bias"]:
        print(f"  {k:<23} {g.get(k, 'N/A'):>10}")

    # Baselines
    bl = results.get("baselines", {})
    if bl:
        print(f"\n{'Baseline Comparison':<25} {'MAE':>8} {'RMSE':>8} {'Bias':>8}")
        print("-" * 50)
        for name, m in bl.items():
            print(f"  {name:<23} {m.get('mae', 'N/A'):>8} {m.get('rmse', 'N/A'):>8} {m.get('bias', 'N/A'):>8}")

    # Segments
    segs = results.get("segments", {})
    if segs:
        print(f"\n{'Segment':<25} {'N':>5} {'MAE':>8} {'Bias':>8} {'AvgAct':>8} {'AvgPrd':>8}")
        print("-" * 65)
        for name, m in segs.items():
            if m.get("n", 0) == 0:
                continue
            print(f"  {name:<23} {m['n']:>5} {m.get('mae', 'N/A'):>8} {m.get('bias', 'N/A'):>8} {m.get('avg_actual', 'N/A'):>8} {m.get('avg_predicted', 'N/A'):>8}")

    # Tails
    tails = results.get("tails", {})
    if tails:
        print(f"\n{'Tail Range':<15} {'N':>5} {'MAE':>8} {'Bias':>8} {'AvgAct':>8} {'AvgPrd':>8}")
        print("-" * 55)
        for name, m in tails.items():
            if m.get("n", 0) == 0:
                continue
            print(f"  {name:<13} {m['n']:>5} {m.get('mae', 'N/A'):>8} {m.get('bias', 'N/A'):>8} {m.get('avg_actual', 'N/A'):>8} {m.get('avg_predicted', 'N/A'):>8}")

    # Calibration
    cal = results.get("calibration", {})
    if cal.get("thresholds"):
        print(f"\n{'Line':<8} {'ModelOver%':>10} {'ActualOver%':>12} {'Accuracy':>10} {'Brier':>8}")
        print("-" * 50)
        for t in cal["thresholds"]:
            print(f"  {t['line']:<6} {t['model_over_pct']:>10} {t['actual_over_pct']:>12} {t['accuracy']:>10} {t['brier_score']:>8}")

    # Simulation Brier
    sb = results.get("simulation_brier", {})
    if sb:
        print(f"\n{'SimLine':<12} {'Brier':>8} {'ModelProb':>10} {'ActualRate':>12}")
        print("-" * 45)
        for name, m in sb.items():
            print(f"  {name:<10} {m['brier_score']:>8} {m['avg_model_prob']:>10} {m['actual_rate']:>12}")

    print("\n" + "=" * 70)
