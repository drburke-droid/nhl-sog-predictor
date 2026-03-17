"""
MLB Model Performance Audit & Deployment Decision Framework.

Runs automatically after each training cycle to evaluate the model,
detect weaknesses, and determine deployment readiness.
"""

import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def run_audit() -> dict:
    """Run full 13-section audit against current trained model.

    Returns audit report dict with all section results and final deployment decision.
    """
    import mlb_model
    import mlb_simulation

    metrics = mlb_model.get_model_metrics()
    if not metrics or "error" in metrics:
        return {"error": "No trained model available", "model_readiness": "not_ready"}

    # Build test set
    logger.info("Building feature matrix for audit...")
    df = mlb_model._build_feature_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - timedelta(days=14)
    test_df = df[df["date"] > cutoff].copy()
    actual = test_df["strikeouts"].values

    # Model predictions
    bf_pred = test_df["baseline_bf"].values + mlb_model._model_bf.predict(test_df[mlb_model.BF_FEATURES].values)
    bf_pred = np.clip(bf_pred, 10, 40)
    kbf_pred = test_df["baseline_k_rate"].values + mlb_model._model_kbf.predict(test_df[mlb_model.KBF_FEATURES].values)
    kbf_pred = np.clip(kbf_pred, 0.0, 0.60)
    model_pred = bf_pred * kbf_pred
    model_pred = np.maximum(model_pred, 0.0)

    # Run simulations for probability checks
    sim_results = []
    for i in range(len(test_df)):
        pid = int(test_df.iloc[i]["pitcher_id"])
        bf_std = mlb_model._pitcher_bf_std.get(pid, bf_pred[i] * 0.18)
        kbf_std = mlb_model._pitcher_kbf_std.get(pid, kbf_pred[i] * 0.20)
        sim = mlb_simulation.simulate_strikeouts(
            bf_pred[i], kbf_pred[i], bf_std=bf_std, kbf_std=kbf_std,
            n_sims=5000, seed=42 + i,
        )
        sim_results.append(sim)

    report = {"holdout_samples": len(test_df)}

    # ===================================================================
    # SECTION 1 — Prediction Signal Validation
    # ===================================================================
    baselines = metrics.get("baselines", {})
    model_mae = metrics["mae"]
    season_avg_mae = baselines.get("season_avg", model_mae)

    mae_improvement = season_avg_mae - model_mae
    mae_improvement_pct = mae_improvement / season_avg_mae if season_avg_mae > 0 else 0

    if mae_improvement_pct >= 0.03:
        prediction_signal = "strong"
    elif mae_improvement_pct >= 0.01:
        prediction_signal = "moderate"
    else:
        prediction_signal = "weak"

    report["section_1_prediction_signal"] = {
        "model_mae": model_mae,
        "season_avg_mae": season_avg_mae,
        "mae_improvement": round(mae_improvement, 4),
        "mae_improvement_pct": round(mae_improvement_pct, 4),
        "prediction_signal": prediction_signal,
    }

    # ===================================================================
    # SECTION 2 — Feature Sanity Validation
    # ===================================================================
    valid_categories = {
        # Pitcher strikeout skill
        "baseline_k_rate", "k_minus_bb_rate", "csw_rate", "whiff_rate",
        "two_strike_putaway_rate", "first_pitch_strike_rate", "zone_contact_rate",
        "pitcher_cv", "rolling_velocity", "ff_whiff", "sl_whiff", "ch_whiff", "cu_whiff",
        "velo_gap", "pitch_entropy", "breaking_usage",
        # Opponent strikeout metrics
        "opp_k_rate", "opp_contact_rate", "opp_chase_rate",
        "matchup_whiff_rate", "matchup_k_rate", "matchup_contact_rate",
        "ff_x_lineup_whiff", "sl_x_lineup_whiff", "ch_x_lineup_whiff", "cu_x_lineup_whiff",
        # Pitch efficiency
        "ff_usage", "sl_usage", "ch_usage", "cu_usage",
        "zone_rate", "chase_rate", "tto_k_decay",
        # Pitch count / opportunity
        "baseline_bf", "days_rest", "pitches_last", "innings_last",
        "avg_pitch_count", "bf_trend", "season_bb_rate",
        "pitches_per_bf", "pitches_per_ip", "rolling_walk_rate", "rolling_whip",
        "rolling_3_pc",
        # Context
        "is_home", "park_k_factor",
        # Market / odds-derived
        "market_k_line", "implied_team_win_prob", "game_total_line", "team_moneyline",
    }

    kbf_top = list(metrics.get("top_kbf_features", {}).keys())
    bf_top = list(metrics.get("top_bf_features", {}).keys())
    all_top = kbf_top + bf_top

    valid_count = sum(1 for f in all_top if f in valid_categories)
    valid_pct = valid_count / max(len(all_top), 1)

    feature_logic_status = "valid" if valid_pct >= 0.70 else "suspicious"

    report["section_2_feature_sanity"] = {
        "total_top_features": len(all_top),
        "valid_features": valid_count,
        "valid_pct": round(valid_pct, 3),
        "feature_logic_status": feature_logic_status,
        "unrecognized_features": [f for f in all_top if f not in valid_categories],
    }

    # ===================================================================
    # SECTION 3 — Baseline Dominance Check
    # ===================================================================
    baseline_names = ["season_avg", "weighted_baseline", "rolling_5", "rolling_3", "opp_k_only"]
    baseline_outperformance = 0
    baseline_comparisons = {}
    for name in baseline_names:
        b_mae = baselines.get(name)
        if b_mae is not None:
            beats = model_mae < b_mae
            baseline_comparisons[name] = {"mae": b_mae, "model_beats": beats}
            if beats:
                baseline_outperformance += 1

    baseline_status = "pass" if baseline_outperformance == len(baseline_comparisons) else "fail"

    report["section_3_baseline_dominance"] = {
        "baselines_tested": len(baseline_comparisons),
        "baselines_beaten": baseline_outperformance,
        "baseline_status": baseline_status,
        "comparisons": baseline_comparisons,
    }

    # ===================================================================
    # SECTION 4 — Probability Calibration Validation
    # ===================================================================
    calibration_results = {}
    calibration_flags = []

    for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
        probs = np.array([s[f"P_over_{line:.1f}"] for s in sim_results])
        actuals = (actual > line).astype(float)
        brier = brier_score_loss(actuals, probs)
        accuracy = float(((probs >= 0.5) == (actuals == 1)).mean())
        cal_error = abs(float(probs.mean()) - float(actuals.mean()))

        if cal_error <= 0.03:
            cal_grade = "excellent"
        elif cal_error <= 0.05:
            cal_grade = "acceptable"
        else:
            cal_grade = "recalibration_required"
            calibration_flags.append(line)

        calibration_results[f"over_{line}"] = {
            "brier_score": round(brier, 4),
            "accuracy": round(accuracy, 3),
            "model_prob_avg": round(float(probs.mean()), 4),
            "actual_rate": round(float(actuals.mean()), 4),
            "calibration_error": round(cal_error, 4),
            "calibration_grade": cal_grade,
        }

    overall_cal_status = "acceptable" if len(calibration_flags) <= 1 else "needs_work"

    report["section_4_calibration"] = {
        "lines": calibration_results,
        "lines_needing_recalibration": calibration_flags,
        "overall_calibration_status": overall_cal_status,
        "recommended_methods": ["isotonic regression", "logistic calibration"] if calibration_flags else [],
    }

    # ===================================================================
    # SECTION 5 — Middle Probability Bias Detection
    # ===================================================================
    mid_bias_results = {}
    for line in [4.5, 5.5, 6.5]:
        probs = np.array([s[f"P_over_{line:.1f}"] for s in sim_results])
        actuals = (actual > line).astype(float)
        mask = (probs >= 0.35) & (probs <= 0.55)

        if mask.sum() >= 10:
            pred_rate = float(probs[mask].mean())
            actual_rate = float(actuals[mask].mean())
            bias = actual_rate - pred_rate

            if bias > 0.05:
                bias_label = "conservative"
            elif bias < -0.05:
                bias_label = "aggressive"
            else:
                bias_label = "balanced"

            mid_bias_results[f"over_{line}"] = {
                "n": int(mask.sum()),
                "predicted_rate": round(pred_rate, 4),
                "actual_rate": round(actual_rate, 4),
                "bias": round(bias, 4),
                "bias_label": bias_label,
            }

    edge_zone_flagged = any(r["bias_label"] == "conservative" for r in mid_bias_results.values())

    report["section_5_middle_bias"] = {
        "results": mid_bias_results,
        "edge_zone_flagged": edge_zone_flagged,
        "action": "Middle probability range shows conservative bias — potential edge zone for over bets" if edge_zone_flagged else "No systematic middle-range bias detected",
    }

    # ===================================================================
    # SECTION 6 — High Strikeout Pitcher Bias
    # ===================================================================
    baseline_k = test_df["baseline_bf"].values * test_df["baseline_k_rate"].values
    high_k_mask = baseline_k >= 6
    high_k_count = int(high_k_mask.sum())

    if high_k_count > 0:
        mean_bias_high_k = float(np.mean(model_pred[high_k_mask] - actual[high_k_mask]))
        high_k_flag = abs(mean_bias_high_k) > 0.25
    else:
        mean_bias_high_k = 0.0
        high_k_flag = False

    report["section_6_high_k_bias"] = {
        "high_k_pitchers": high_k_count,
        "mean_bias": round(mean_bias_high_k, 3),
        "highK_bias_flag": high_k_flag,
        "action": "Investigate opportunity model and innings limits for high-K pitchers" if high_k_flag else "High-K bias within acceptable range",
    }

    # ===================================================================
    # SECTION 7 — Extreme Outcome Handling
    # ===================================================================
    errors = np.abs(model_pred - actual)
    p95_error = float(np.percentile(errors, 95))

    distribution_model_status = "weak" if p95_error > 5 else "acceptable"

    report["section_7_extreme_outcomes"] = {
        "p95_error": round(p95_error, 2),
        "max_error": round(float(errors.max()), 2),
        "distribution_model_status": distribution_model_status,
        "note": "Single extreme events (e.g., 12K vs 5.1K predicted) are ignored — focus on 95th percentile",
    }

    # ===================================================================
    # SECTION 8 — Extreme Line Evaluation
    # ===================================================================
    extreme_line_results = {}
    for line in [6.5, 7.5]:
        key = f"over_{line}"
        cr = calibration_results.get(key, {})
        brier = cr.get("brier_score", 1.0)
        prob_quality = "good" if brier <= 0.20 else "weak"
        extreme_line_results[key] = {
            "brier_score": brier,
            "calibration_error": cr.get("calibration_error", 0),
            "probability_quality": prob_quality,
        }

    report["section_8_extreme_lines"] = extreme_line_results

    # ===================================================================
    # SECTION 9 — Sample Size Check
    # ===================================================================
    n = len(test_df)
    if n >= 1000:
        sample_status = "strong"
    elif n >= 400:
        sample_status = "moderate"
    else:
        sample_status = "limited"

    report["section_9_sample_size"] = {
        "holdout_size": n,
        "sample_status": sample_status,
        "action": "Continue collecting data before trusting tail probabilities" if sample_status == "limited" else "Sample size adequate for current evaluation",
    }

    # ===================================================================
    # SECTION 10 — Architecture Verification
    # ===================================================================
    has_bf_model = mlb_model._model_bf is not None
    has_kbf_model = mlb_model._model_kbf is not None
    has_simulation = True
    try:
        import mlb_simulation
    except ImportError:
        has_simulation = False

    all_components = has_bf_model and has_kbf_model and has_simulation
    architecture_status = "correct" if all_components else "incorrect"

    report["section_10_architecture"] = {
        "opportunity_model": has_bf_model,
        "k_rate_model": has_kbf_model,
        "distribution_simulation_layer": has_simulation,
        "architecture_status": architecture_status,
    }

    # ===================================================================
    # SECTION 11 — Market Validation Requirement
    # ===================================================================
    report["section_11_market_validation"] = {
        "status": "ready",
        "note": "Market comparison layer implemented. Pass market_line, market_over_odds, market_under_odds to predict_pitcher() to compute edge and EV.",
        "edge_threshold_for_bet": 0.05,
        "implementation": "mlb_simulation.simulate_with_market() and mlb_simulation.compute_edge()",
    }

    # ===================================================================
    # SECTION 12 — Deployment Decision Logic
    # ===================================================================
    conditions = {
        "prediction_signal_strong": prediction_signal == "strong",
        "feature_logic_valid": feature_logic_status == "valid",
        "baseline_status_pass": baseline_status == "pass",
        "calibration_acceptable": overall_cal_status == "acceptable",
        "architecture_correct": architecture_status == "correct",
    }

    all_pass = all(conditions.values())
    model_readiness = "ready_for_testing" if all_pass else "not_ready"

    failing_conditions = [k for k, v in conditions.items() if not v]

    report["section_12_deployment_decision"] = {
        "conditions": conditions,
        "all_conditions_met": all_pass,
        "model_readiness": model_readiness,
        "failing_conditions": failing_conditions,
    }

    # ===================================================================
    # SECTION 13 — Testing Protocol
    # ===================================================================
    report["section_13_testing_protocol"] = {
        "phase_1": {
            "name": "Paper trading",
            "duration": "2-4 weeks",
            "track": ["model_probability", "market_odds", "closing_odds", "outcome"],
            "evaluate": ["closing_line_value", "ROI", "calibration_drift"],
        },
        "phase_2": {
            "name": "Small stakes",
            "rule": "Bet only if edge >= 6%",
        },
        "phase_3": {
            "name": "Refinement priorities",
            "priorities": [],
        },
    }

    # Populate refinement priorities based on audit findings
    priorities = []
    if high_k_flag:
        priorities.append("High-K pitcher bias correction")
    if edge_zone_flagged:
        priorities.append("Middle probability recalibration to capture edge zone")
    if calibration_flags:
        priorities.append(f"Recalibrate lines: {calibration_flags}")
    if sample_status == "limited":
        priorities.append("Collect more holdout data for robust tail evaluation")
    priorities.append("Variance modeling refinement")
    priorities.append("Lineup matchup features (when projected lineups available)")
    priorities.append("Closing line value tracking")
    report["section_13_testing_protocol"]["phase_3"]["priorities"] = priorities

    # ===================================================================
    # Corrective Actions (if any validation fails)
    # ===================================================================
    corrective_actions = []
    if prediction_signal == "weak":
        corrective_actions.append("Model signal too weak — review feature engineering or consider additional data sources")
    if feature_logic_status == "suspicious":
        corrective_actions.append("Top features don't match expected categories — check for data leakage or irrelevant features")
    if baseline_status == "fail":
        corrective_actions.append("Model fails to beat all baselines — review model architecture or hyperparameters")
    if overall_cal_status == "needs_work":
        corrective_actions.append(f"Calibration issues on lines {calibration_flags} — apply isotonic regression or logistic calibration")
    if high_k_flag:
        corrective_actions.append(f"High-K pitcher bias of {mean_bias_high_k:.3f} — investigate BF model ceiling for aces")
    if distribution_model_status == "weak":
        corrective_actions.append("95th percentile error too high — review variance estimation and simulation parameters")
    if architecture_status == "incorrect":
        corrective_actions.append("Missing architecture components — ensure all 3 layers are implemented")

    report["corrective_actions"] = corrective_actions if corrective_actions else ["None — all validations passed"]

    report["model_readiness"] = model_readiness
    return report


def print_audit(report: dict):
    """Print formatted audit report."""
    if "error" in report:
        print(f"AUDIT ERROR: {report['error']}")
        return

    print("=" * 72)
    print("  MLB PITCHER STRIKEOUT MODEL — PERFORMANCE AUDIT REPORT")
    print("=" * 72)
    print(f"  Holdout samples: {report['holdout_samples']}")
    print()

    # Section 1
    s1 = report["section_1_prediction_signal"]
    print("SECTION 1 — Prediction Signal Validation")
    print(f"  Model MAE:         {s1['model_mae']}")
    print(f"  Season Avg MAE:    {s1['season_avg_mae']}")
    print(f"  Improvement:       {s1['mae_improvement']:.4f} ({s1['mae_improvement_pct']:.1%})")
    sig = s1["prediction_signal"]
    marker = "+++" if sig == "strong" else ("++" if sig == "moderate" else "+")
    print(f"  SIGNAL:            {sig.upper()} {marker}")
    print()

    # Section 2
    s2 = report["section_2_feature_sanity"]
    print("SECTION 2 — Feature Sanity Validation")
    print(f"  Top features checked: {s2['total_top_features']}")
    print(f"  Valid:                {s2['valid_features']} ({s2['valid_pct']:.0%})")
    print(f"  STATUS:               {s2['feature_logic_status'].upper()}")
    if s2["unrecognized_features"]:
        print(f"  Unrecognized:         {s2['unrecognized_features']}")
    print()

    # Section 3
    s3 = report["section_3_baseline_dominance"]
    print("SECTION 3 — Baseline Dominance Check")
    for name, comp in s3["comparisons"].items():
        check = "BEAT" if comp["model_beats"] else "LOST"
        print(f"  vs {name:<22} {comp['mae']:.3f}  [{check}]")
    print(f"  STATUS: {s3['baseline_status'].upper()} ({s3['baselines_beaten']}/{s3['baselines_tested']})")
    print()

    # Section 4
    s4 = report["section_4_calibration"]
    print("SECTION 4 — Probability Calibration Validation")
    print(f"  {'Line':<10} {'Brier':>8} {'CalErr':>8} {'Grade':<22}")
    print("  " + "-" * 50)
    for key, cr in s4["lines"].items():
        print(f"  {key:<10} {cr['brier_score']:>8.4f} {cr['calibration_error']:>7.1%} {cr['calibration_grade']:<22}")
    print(f"  OVERALL: {s4['overall_calibration_status'].upper()}")
    if s4["lines_needing_recalibration"]:
        print(f"  Lines needing recalibration: {s4['lines_needing_recalibration']}")
        print(f"  Recommended: {s4['recommended_methods']}")
    print()

    # Section 5
    s5 = report["section_5_middle_bias"]
    print("SECTION 5 — Middle Probability Bias Detection (35-55% range)")
    for key, r in s5["results"].items():
        print(f"  {key:<10} n={r['n']:<4} pred={r['predicted_rate']:.3f}  actual={r['actual_rate']:.3f}  bias={r['bias']:+.4f}  [{r['bias_label'].upper()}]")
    if s5["edge_zone_flagged"]:
        print(f"  >>> EDGE ZONE FLAGGED: {s5['action']}")
    else:
        print(f"  {s5['action']}")
    print()

    # Section 6
    s6 = report["section_6_high_k_bias"]
    print("SECTION 6 — High Strikeout Pitcher Bias (baseline K >= 6)")
    print(f"  Sample:    {s6['high_k_pitchers']} starts")
    print(f"  Mean bias: {s6['mean_bias']:+.3f}")
    flag = "FLAGGED" if s6["highK_bias_flag"] else "OK"
    print(f"  STATUS:    {flag}")
    print(f"  {s6['action']}")
    print()

    # Section 7
    s7 = report["section_7_extreme_outcomes"]
    print("SECTION 7 — Extreme Outcome Handling")
    print(f"  95th pctl error: {s7['p95_error']}")
    print(f"  Max error:       {s7['max_error']}")
    print(f"  STATUS:          {s7['distribution_model_status'].upper()}")
    print()

    # Section 8
    s8 = report["section_8_extreme_lines"]
    print("SECTION 8 — Extreme Line Evaluation (6.5, 7.5)")
    for key, r in s8.items():
        print(f"  {key:<10} Brier={r['brier_score']:.4f}  CalErr={r['calibration_error']:.4f}  [{r['probability_quality'].upper()}]")
    print()

    # Section 9
    s9 = report["section_9_sample_size"]
    print("SECTION 9 — Sample Size Check")
    print(f"  Holdout:  {s9['holdout_size']} starts")
    print(f"  STATUS:   {s9['sample_status'].upper()}")
    print(f"  {s9['action']}")
    print()

    # Section 10
    s10 = report["section_10_architecture"]
    print("SECTION 10 — Architecture Verification")
    print(f"  Opportunity model (BF):     {'YES' if s10['opportunity_model'] else 'NO'}")
    print(f"  K rate model (K/BF):        {'YES' if s10['k_rate_model'] else 'NO'}")
    print(f"  Distribution simulation:    {'YES' if s10['distribution_simulation_layer'] else 'NO'}")
    print(f"  STATUS: {s10['architecture_status'].upper()}")
    print()

    # Section 11
    s11 = report["section_11_market_validation"]
    print("SECTION 11 — Market Validation Requirement")
    print(f"  Status: {s11['status'].upper()}")
    print(f"  Edge threshold: {s11['edge_threshold_for_bet']:.0%}")
    print(f"  {s11['note']}")
    print()

    # Section 12
    s12 = report["section_12_deployment_decision"]
    print("SECTION 12 — Deployment Decision")
    print("  Conditions:")
    for cond, passed in s12["conditions"].items():
        mark = "PASS" if passed else "FAIL"
        print(f"    {cond:<35} [{mark}]")
    print()
    readiness = s12["model_readiness"]
    if readiness == "ready_for_testing":
        print(f"  >>> MODEL READINESS: {readiness.upper()}")
    else:
        print(f"  >>> MODEL READINESS: {readiness.upper()}")
        print(f"  Failing: {s12['failing_conditions']}")
    print()

    # Section 13
    s13 = report["section_13_testing_protocol"]
    print("SECTION 13 — Testing Protocol")
    print(f"  Phase 1: {s13['phase_1']['name']} ({s13['phase_1']['duration']})")
    print(f"  Phase 2: {s13['phase_2']['name']} — {s13['phase_2']['rule']}")
    print(f"  Phase 3: Refinement priorities:")
    for p in s13["phase_3"]["priorities"]:
        print(f"    - {p}")
    print()

    # Corrective Actions
    print("CORRECTIVE ACTIONS")
    for a in report["corrective_actions"]:
        print(f"  - {a}")

    print()
    print("=" * 72)
    print(f"  FINAL VERDICT: {report['model_readiness'].upper()}")
    print("=" * 72)
