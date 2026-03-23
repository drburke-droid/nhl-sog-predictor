"""
Locked-Forward Evaluation — strictest out-of-sample test.

1. Train model ONCE on data before cutoff
2. Freeze all hyperparameters, thresholds, calibration
3. Run forward on ALL unseen data without retuning
4. No expanding window — single frozen model

This tests whether the strategy survives deployment without continuous retraining.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import model as nhl_model
import data_collector
import nhl_odds_collector
import nhl_simulation
import nhl_walkforward as wf_v1
import evaluation as ev

logger = logging.getLogger(__name__)


def run_locked_forward(train_cutoff="2026-02-01", min_edge=0.05):
    """Train once, freeze, score all future data.

    Args:
        train_cutoff: date string — train on everything before this, test on everything after
        min_edge: fixed edge threshold (not tuned)
    """
    logger.info("Building feature matrix...")
    df = nhl_model._build_feature_dataframe()
    if df.empty:
        return {"error": "No data"}

    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp(train_cutoff)

    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    logger.info("Locked-forward: train %d rows (to %s), test %d rows (%s to %s)",
                len(train), train_cutoff, len(test),
                test["date"].min().date(), test["date"].max().date())

    if len(train) < 500 or len(test) < 100:
        return {"error": "Insufficient data"}

    # --- TRAIN ONCE (frozen) ---
    feat_cols = [c for c in nhl_model.FEATURE_COLS if c != "sog_prop_line"]
    avail = [f for f in feat_cols if f in train.columns]
    weights = nhl_model._compute_sample_weights(train)

    xgb_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
    )

    fwd_train = train[train["position_group"] == "F"]
    def_train = train[train["position_group"] == "D"]
    fwd_w = weights[train["position_group"] == "F"]
    def_w = weights[train["position_group"] == "D"]

    m_fwd = XGBRegressor(**xgb_params)
    m_fwd.fit(fwd_train[avail].values, fwd_train["sog_residual"].values, sample_weight=fwd_w)

    m_def = XGBRegressor(**xgb_params)
    m_def.fit(def_train[avail].values, def_train["sog_residual"].values, sample_weight=def_w)

    # Variance ratios from training only
    var_ratio_map = {}
    for pid, grp in train.groupby("player_id"):
        shots = grp["shots"].values
        if len(shots) >= 10 and shots.mean() > 0:
            var_ratio_map[int(pid)] = float(shots.var() / shots.mean())

    # Calibration from training only
    # Predict on train to build calibration curve
    train_fwd = train[train["position_group"] == "F"].copy()
    train_def = train[train["position_group"] == "D"].copy()

    train_pred_fwd = np.maximum(train_fwd["baseline_sog"].values +
                                m_fwd.predict(train_fwd[avail].values), 0)
    train_pred_def = np.maximum(train_def["baseline_sog"].values +
                                m_def.predict(train_def[avail].values), 0)

    logger.info("Frozen model trained. Scoring test set...")

    # --- LOAD ODDS (frozen — from data as it existed) ---
    prop_lines = wf_v1._load_prop_lookup()
    per_book_props = wf_v1._load_per_book_props()

    conn = data_collector.get_db()
    pid_to_name = {int(r["player_id"]): wf_v1._normalize_name(r["player_name"])
                   for r in conn.execute("SELECT player_id, player_name FROM player_game_stats").fetchall()}
    conn.close()

    # --- SCORE TEST SET (no retraining) ---
    all_bets = []

    test_fwd = test[test["position_group"] == "F"].copy()
    test_def = test[test["position_group"] == "D"].copy()

    for subset, mdl, label in [(test_fwd, m_fwd, "F"), (test_def, m_def, "D")]:
        if len(subset) == 0:
            continue

        cols = [c for c in avail if c in subset.columns]
        pred_res = mdl.predict(subset[cols].values)
        pred_sog = np.maximum(subset["baseline_sog"].values + pred_res, 0.0)

        for i in range(len(subset)):
            row = subset.iloc[i]
            ps = float(pred_sog[i])
            actual = int(row["shots"])
            pid = int(row["player_id"])
            pname = row.get("player_name", "") or pid_to_name.get(pid, "")
            gd = row["date"].strftime("%Y-%m-%d")
            vr = var_ratio_map.get(pid, 1.0)

            if not pname:
                continue

            sim = nhl_simulation.simulate_sog(
                pred_sog=ps, var_ratio=vr, model_std=0.3, n_sims=10000,
                seed=42 + i)

            ini_k, last_k = wf_v1._name_match_key(str(pname))

            for line in nhl_simulation.PROP_LINES:
                lk = (ini_k, last_k, gd, line)
                if lk not in prop_lines:
                    continue
                pl = prop_lines[lk]

                model_p_over = sim.get(f"P_over_{line}", 0)
                model_p_under = 1.0 - model_p_over

                sharp = wf_v1._compute_sharp_consensus(per_book_props, ini_k, last_k, gd, line)
                soft = wf_v1._get_soft_book_prices(per_book_props, ini_k, last_k, gd, line)

                for side, price_key, model_p, won_fn in [
                    ("OVER", "over_price", model_p_over, lambda s, l: s > l),
                    ("UNDER", "under_price", model_p_under, lambda s, l: s <= l),
                ]:
                    price = pl.get(price_key)
                    if price is None or model_p <= 0.01:
                        continue

                    implied = nhl_simulation.american_to_implied_prob(price)
                    decimal_odds = nhl_simulation.american_to_decimal(price)

                    # Blended prob (50/50 model + sharp)
                    blended = model_p
                    if sharp:
                        sp = sharp["sharp_prob_over"] if side == "OVER" else sharp["sharp_prob_under"]
                        blended = 0.5 * model_p + 0.5 * sp

                    # Edge vs soft book
                    soft_price = None
                    soft_implied = None
                    if soft:
                        sp_key = "soft_over_price" if side == "OVER" else "soft_under_price"
                        soft_price = soft.get(sp_key)
                        if soft_price:
                            soft_implied = nhl_simulation.american_to_implied_prob(soft_price)

                    edge = blended - (soft_implied or implied)
                    ev_val = blended * (decimal_odds - 1) - (1 - blended)

                    all_bets.append({
                        "date": gd,
                        "side": side,
                        "odds": price,
                        "decimal_odds": round(decimal_odds, 4),
                        "model_prob": round(blended, 4),
                        "implied_prob": round(implied, 4),
                        "edge": round(edge, 4),
                        "ev": round(ev_val, 4),
                        "won": won_fn(actual, line),
                        "has_soft": soft is not None,
                        "soft_price": soft_price,
                        "soft_implied": round(soft_implied, 4) if soft_implied else None,
                        "line": line,
                    })

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

    if bets_df.empty:
        return {"error": "No bets"}

    # --- FIXED PRODUCTION FILTER (not tuned) ---
    is_under = bets_df["side"] == "UNDER"
    has_soft = bets_df["has_soft"] == True
    edge_ok = bets_df["edge"] >= min_edge

    prod = bets_df[is_under & has_soft & edge_ok].copy()

    # Reports
    report_all = ev.generate_full_report(bets_df[bets_df["ev"] > 0], "Locked: All +EV")
    report_prod = ev.generate_full_report(prod, f"Locked: Production (unders {min_edge:.0%}+ vs BMG)")

    return {
        "train_cutoff": train_cutoff,
        "train_size": len(train),
        "test_size": len(test),
        "test_period": f"{test['date'].min().date()} to {test['date'].max().date()}",
        "all_ev": report_all,
        "production": report_prod,
        "bets_df": bets_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Test with Feb 1 cutoff — trains on Oct-Jan, tests on Feb-Mar
    result = run_locked_forward(train_cutoff="2026-02-01")

    print(f"\nTrain cutoff: {result['train_cutoff']}")
    print(f"Train: {result['train_size']} rows, Test: {result['test_size']} rows")
    print(f"Test period: {result['test_period']}")

    ev.print_report(result["all_ev"])
    ev.print_report(result["production"])

    # Also test with Jan 1 cutoff — longer forward period
    result2 = run_locked_forward(train_cutoff="2026-01-15")
    print(f"\nTrain cutoff: {result2['train_cutoff']}")
    print(f"Test period: {result2['test_period']}")
    ev.print_report(result2["production"])
