"""
MLB Meta Model — Gating layer that predicts when the main model beats the market.

Implements the full spec: dataset construction, logistic + XGBoost classifiers,
calibration, walk-forward backtest, and comparison against ungated strategies.
"""

import logging
import sqlite3
import math
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    precision_score, recall_score, mean_absolute_error,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

import mlb_model
import mlb_simulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1-3: Build meta-model dataset from walk-forward predictions
# ---------------------------------------------------------------------------

def _train_window_model(train_df):
    """Train BF + KBF models on a training window."""
    weights = mlb_model._compute_sample_weights(train_df)
    model_bf = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=8,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
    )
    model_bf.fit(train_df[mlb_model.BF_FEATURES].values, train_df["bf_residual"].values,
                 sample_weight=weights)
    model_kbf = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=8,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
    )
    model_kbf.fit(train_df[mlb_model.KBF_FEATURES].values, train_df["kbf_residual"].values,
                  sample_weight=weights)

    # Pitcher variance
    train_bf_resid = train_df["bf_residual"].values - model_bf.predict(train_df[mlb_model.BF_FEATURES].values)
    train_kbf_resid = train_df["kbf_residual"].values - model_kbf.predict(train_df[mlb_model.KBF_FEATURES].values)
    _tv = train_df[["pitcher_id"]].copy()
    _tv["bf_r"] = train_bf_resid
    _tv["kbf_r"] = train_kbf_resid
    g_bf_std = float(np.std(train_bf_resid))
    g_kbf_std = float(np.std(train_kbf_resid))
    bf_std_map, kbf_std_map = {}, {}
    for pid, grp in _tv.groupby("pitcher_id"):
        n = len(grp)
        shrink = min(n / 15.0, 1.0) if n >= 5 else 0.0
        bf_std_map[pid] = shrink * float(grp["bf_r"].std()) + (1 - shrink) * g_bf_std
        kbf_std_map[pid] = shrink * float(grp["kbf_r"].std()) + (1 - shrink) * g_kbf_std
    return model_bf, model_kbf, bf_std_map, kbf_std_map


def _ml_to_prob(ml):
    if ml is None or ml == 0:
        return 0.50
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)


def build_meta_dataset(min_train_days=60, test_window_days=14, step_days=14,
                       bookmaker="draftkings"):
    """Build the full meta-model dataset using walk-forward windows.

    For each window: train main model, predict on test set, match with
    sportsbook odds, generate meta-features for every candidate bet,
    label with model_beats_market target.
    """
    logger.info("Building full feature matrix...")
    df = mlb_model._build_feature_dataframe()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    season_start = df["date"].min()
    season_end = df["date"].max()
    logger.info("Season: %s to %s (%d starts)", season_start.date(), season_end.date(), len(df))

    # Load prop odds
    conn = sqlite3.connect("mlb_data.db")
    conn.row_factory = sqlite3.Row
    props_df = pd.read_sql_query(
        "SELECT pitcher_name, game_date, line, over_under, price, bookmaker "
        "FROM mlb_pitcher_props WHERE bookmaker = ?",
        conn, params=(bookmaker,),
    )
    # Consensus lines (all books)
    consensus_df = pd.read_sql_query(
        "SELECT pitcher_name, game_date, AVG(line) as consensus_line, "
        "COUNT(DISTINCT bookmaker) as num_books "
        "FROM mlb_pitcher_props WHERE over_under = 'Over' "
        "GROUP BY pitcher_name, game_date", conn,
    )
    consensus_map = {}
    for _, r in consensus_df.iterrows():
        consensus_map[(mlb_model._normalize_name(r["pitcher_name"]), r["game_date"])] = {
            "consensus_line": float(r["consensus_line"]),
            "num_books": int(r["num_books"]),
        }

    # Pitcher name lookup
    name_rows = conn.execute(
        "SELECT pitcher_id, date, pitcher_name FROM mlb_pitcher_game_stats"
    ).fetchall()
    pid_date_to_name = {}
    for r in name_rows:
        pid_date_to_name[(r["pitcher_id"], r["date"])] = mlb_model._normalize_name(r["pitcher_name"])
    conn.close()

    # Build prop lookup: (norm_name, date, line) -> {over_price, under_price}
    prop_lines = {}
    for _, r in props_df.iterrows():
        key = (mlb_model._normalize_name(r["pitcher_name"]), r["game_date"], r["line"])
        if key not in prop_lines:
            prop_lines[key] = {}
        if r["over_under"] == "Over":
            prop_lines[key]["over_price"] = int(r["price"])
        else:
            prop_lines[key]["under_price"] = int(r["price"])

    logger.info("Prop lines: %d, Consensus entries: %d", len(prop_lines), len(consensus_map))

    # Walk-forward loop to build meta-dataset
    all_rows = []
    first_test_start = season_start + timedelta(days=min_train_days)
    current_test_start = first_test_start
    window_num = 0

    while current_test_start < season_end:
        window_num += 1
        current_test_end = current_test_start + timedelta(days=test_window_days)
        if current_test_end > season_end:
            current_test_end = season_end

        train_df = df[df["date"] < current_test_start].copy()
        test_df = df[(df["date"] >= current_test_start) & (df["date"] <= current_test_end)].copy()

        if len(train_df) < 50 or len(test_df) == 0:
            current_test_start += timedelta(days=step_days)
            continue

        logger.info("Window %d: train %d, test %d (%s - %s)",
                     window_num, len(train_df), len(test_df),
                     test_df["date"].min().date(), test_df["date"].max().date())

        model_bf, model_kbf, bf_std_map, kbf_std_map = _train_window_model(train_df)

        # Predict
        bf_raw = model_bf.predict(test_df[mlb_model.BF_FEATURES].values)
        bf_pred = test_df["baseline_bf"].values + np.nan_to_num(bf_raw, nan=0.0)
        bf_pred = np.clip(bf_pred, 10, 40)
        kbf_raw = model_kbf.predict(test_df[mlb_model.KBF_FEATURES].values)
        kbf_pred = test_df["baseline_k_rate"].values + np.nan_to_num(kbf_raw, nan=0.0)
        kbf_pred = np.clip(kbf_pred, 0.01, 0.55)
        pred_k = bf_pred * kbf_pred

        # Simulate
        sim_results = []
        for i in range(len(test_df)):
            pid = int(test_df.iloc[i]["pitcher_id"])
            bf_s = bf_std_map.get(pid, bf_pred[i] * 0.18)
            kbf_s = kbf_std_map.get(pid, kbf_pred[i] * 0.20)
            if np.isnan(bf_s) or bf_s <= 0:
                bf_s = bf_pred[i] * 0.18
            if np.isnan(kbf_s) or kbf_s <= 0:
                kbf_s = kbf_pred[i] * 0.20
            sim = mlb_simulation.simulate_strikeouts(
                bf_pred[i], kbf_pred[i], bf_std=bf_s, kbf_std=kbf_s,
                n_sims=10000, seed=42 + window_num * 1000 + i,
            )
            sim_results.append(sim)

        # Build meta-rows for each candidate bet
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            sim = sim_results[i]
            actual_k = int(row["strikeouts"])
            gd = row["date"].strftime("%Y-%m-%d")
            pid = int(row["pitcher_id"])
            pname_norm = pid_date_to_name.get((pid, gd), "")
            if not pname_norm:
                continue

            model_error = abs(pred_k[i] - actual_k)
            mk_line = row.get("market_k_line", 0)
            market_error = abs(mk_line - actual_k)
            model_beats_market = 1 if model_error < market_error else 0

            # Consensus info
            cons = consensus_map.get((pname_norm, gd), {})
            consensus_line = cons.get("consensus_line", mk_line)
            num_books = cons.get("num_books", 1)

            # Pitcher tier
            bk = row.get("baseline_k_rate", 0.22) * row.get("baseline_bf", 25)
            if bk < 4:
                pitcher_tier = "low_k"
            elif bk < 6:
                pitcher_tier = "mid_k"
            else:
                pitcher_tier = "high_k"

            # Predicted K bucket
            pk = pred_k[i]
            if pk < 3:
                pk_bucket = "under_3"
            elif pk < 5:
                pk_bucket = "3_to_5"
            elif pk < 7:
                pk_bucket = "5_to_7"
            else:
                pk_bucket = "7_plus"

            # Base meta features (shared across lines)
            base = {
                "window": window_num,
                "game_date": gd,
                "pitcher": pname_norm,
                "pitcher_id": pid,
                "actual_k": actual_k,
                "model_pred_k": round(pred_k[i], 3),
                "model_pred_bf": round(bf_pred[i], 2),
                "model_pred_k_per_bf": round(kbf_pred[i], 4),
                "model_std_k": sim.get("std_k", 0),
                "model_median_k": sim.get("median_k", round(pred_k[i])),
                "model_prob_over_3_5": sim.get("P_over_3.5", 0),
                "model_prob_over_4_5": sim.get("P_over_4.5", 0),
                "model_prob_over_5_5": sim.get("P_over_5.5", 0),
                "model_prob_over_6_5": sim.get("P_over_6.5", 0),
                "model_prob_over_7_5": sim.get("P_over_7.5", 0),
                "model_error": round(model_error, 3),
                "market_error": round(market_error, 3),
                "model_beats_market_global": model_beats_market,
                "model_minus_market": round(pred_k[i] - mk_line, 3),
                "abs_model_minus_market": round(abs(pred_k[i] - mk_line), 3),
                "consensus_line": consensus_line,
                "num_books": num_books,
                # Pitcher context
                "baseline_k_rate": row.get("baseline_k_rate", 0),
                "baseline_bf": row.get("baseline_bf", 25),
                "rolling_velocity": row.get("rolling_velocity", 93),
                "rolling_whip": row.get("rolling_whip", 1.3),
                "season_bb_rate": row.get("season_bb_rate", 0.08),
                "pitches_per_bf": row.get("pitches_per_bf", 4.0),
                "pitches_per_ip": row.get("pitches_per_ip", 16.0),
                "days_rest": row.get("days_rest", 4),
                "pitcher_cv": row.get("pitcher_cv", 1.0),
                "k_minus_bb_rate": row.get("k_minus_bb_rate", 0),
                "first_pitch_strike_rate": row.get("first_pitch_strike_rate", 0.6),
                "tto_k_decay": row.get("tto_k_decay", 0),
                "avg_pitch_count": row.get("avg_pitch_count", 90),
                "pitches_last": row.get("pitches_last", 90),
                # Opponent
                "opp_k_rate": row.get("opp_k_rate", 0.22),
                "opp_contact_rate": row.get("opp_contact_rate", 0.75),
                "opp_chase_rate": row.get("opp_chase_rate", 0.30),
                "matchup_k_rate": row.get("matchup_k_rate", 0),
                "matchup_contact_rate": row.get("matchup_contact_rate", 0),
                "matchup_whiff_rate": row.get("matchup_whiff_rate", 0),
                # Game context
                "is_home": row.get("is_home", 0),
                "park_k_factor": row.get("park_k_factor", 1.0),
                "game_total_line": row.get("game_total_line", 8.5),
                "team_moneyline": row.get("team_moneyline", 0),
                "implied_team_win_prob": row.get("implied_team_win_prob", 0.5),
                # Regime
                "pitcher_tier": pitcher_tier,
                "predicted_k_bucket": pk_bucket,
                "train_size": len(train_df),
            }

            # Generate one row per pitcher-line-side
            for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
                lk = (pname_norm, gd, line)
                if lk not in prop_lines:
                    continue
                pl = prop_lines[lk]

                model_p_over = sim.get(f"P_over_{line}", 0)
                model_p_under = 1.0 - model_p_over

                for side, price_key, model_p in [
                    ("OVER", "over_price", model_p_over),
                    ("UNDER", "under_price", model_p_under),
                ]:
                    price = pl.get(price_key)
                    if price is None or model_p <= 0:
                        continue

                    if price < 0:
                        implied = abs(price) / (abs(price) + 100)
                        decimal_odds = 1 + 100 / abs(price)
                    else:
                        implied = 100 / (price + 100)
                        decimal_odds = 1 + price / 100

                    edge = model_p - implied
                    ev = model_p * (decimal_odds - 1) - (1 - model_p)

                    # Bet outcome
                    if side == "OVER":
                        won = actual_k > line
                    else:
                        won = actual_k <= line

                    # Target: model beat market for THIS bet context
                    # Use global model_beats_market (did our pred_k get closer?)
                    target = model_beats_market

                    # Market line bucket
                    ml_bucket = f"{line}"

                    # Edge bucket
                    if edge < 0.03:
                        edge_bucket = "0_to_0.03"
                    elif edge < 0.06:
                        edge_bucket = "0.03_to_0.06"
                    elif edge < 0.10:
                        edge_bucket = "0.06_to_0.10"
                    else:
                        edge_bucket = "0.10_plus"

                    # Confidence bucket
                    if model_p < 0.40:
                        conf_bucket = "low"
                    elif model_p < 0.55:
                        conf_bucket = "mid"
                    elif model_p < 0.70:
                        conf_bucket = "high"
                    else:
                        conf_bucket = "very_high"

                    # Vig estimate
                    over_p = pl.get("over_price")
                    under_p = pl.get("under_price")
                    vig = 0.0
                    if over_p is not None and under_p is not None:
                        ip_o = _ml_to_prob(over_p)
                        ip_u = _ml_to_prob(under_p)
                        vig = ip_o + ip_u - 1.0

                    meta_row = {
                        **base,
                        "market_line": line,
                        "bet_side": side,
                        "market_odds": price,
                        "market_implied_prob": round(implied, 4),
                        "decimal_odds": round(decimal_odds, 4),
                        "model_prob_selected_side": round(model_p, 4),
                        "edge_selected_side": round(edge, 4),
                        "ev_selected_side": round(ev, 4),
                        "vig_estimate": round(vig, 4),
                        "price_asymmetry": round(abs(_ml_to_prob(pl.get("over_price", -110)) -
                                                     _ml_to_prob(pl.get("under_price", -110))), 4),
                        # Derived
                        "line_is_low": 1 if line in [3.5, 4.5] else 0,
                        "line_is_high": 1 if line >= 5.5 else 0,
                        "low_line_under_flag": 1 if line in [3.5, 4.5] and side == "UNDER" else 0,
                        "high_line_over_flag": 1 if line >= 5.5 and side == "OVER" else 0,
                        "market_line_bucket": ml_bucket,
                        "edge_bucket": edge_bucket,
                        "model_confidence_bucket": conf_bucket,
                        "bet_side_num": 1 if side == "OVER" else 0,
                        # Outcome
                        "bet_won": int(won),
                        "bet_profit_unit": round(decimal_odds - 1, 4) if won else -1.0,
                        "target": target,
                    }
                    all_rows.append(meta_row)

        current_test_start += timedelta(days=step_days)

    meta_df = pd.DataFrame(all_rows)
    logger.info("Meta dataset built: %d rows, %d windows", len(meta_df), window_num)
    return meta_df


# ---------------------------------------------------------------------------
# Section 4-6: Meta-model features and training
# ---------------------------------------------------------------------------

META_FEATURES = [
    # Model output features (A)
    "model_pred_k", "model_pred_bf", "model_pred_k_per_bf",
    "model_std_k", "model_median_k",
    "model_prob_over_3_5", "model_prob_over_4_5", "model_prob_over_5_5",
    "model_prob_over_6_5", "model_prob_over_7_5",
    "model_prob_selected_side", "edge_selected_side",
    "abs_model_minus_market", "model_minus_market",
    # Market features (B)
    "market_line", "market_implied_prob", "market_odds",
    "vig_estimate", "price_asymmetry", "consensus_line", "num_books",
    "line_is_low", "line_is_high",
    # Pitcher context (C)
    "baseline_k_rate", "baseline_bf", "days_rest", "pitcher_cv",
    "rolling_velocity", "rolling_whip", "season_bb_rate",
    "pitches_per_bf", "pitches_per_ip", "tto_k_decay",
    "first_pitch_strike_rate", "k_minus_bb_rate",
    "avg_pitch_count", "pitches_last",
    # Opponent/matchup (D)
    "opp_k_rate", "opp_contact_rate", "opp_chase_rate",
    "matchup_k_rate", "matchup_contact_rate", "matchup_whiff_rate",
    # Game context (E)
    "is_home", "park_k_factor", "game_total_line",
    "team_moneyline", "implied_team_win_prob",
    # Regime features (F)
    "bet_side_num", "low_line_under_flag", "high_line_over_flag",
]


def train_meta_model(train_data, model_type="xgb"):
    """Train meta-model classifier.

    Args:
        train_data: DataFrame with META_FEATURES + 'target' column
        model_type: 'logistic' or 'xgb'

    Returns (model, calibrator) tuple.
    """
    X = train_data[META_FEATURES].values
    y = train_data["target"].values

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        model.fit(X, y)
    else:
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X, y)

    # Calibrate using isotonic regression on training predictions
    raw_probs = model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y)

    return model, calibrator


def predict_meta(model, calibrator, data):
    """Get calibrated meta-model predictions."""
    X = data[META_FEATURES].values
    raw = model.predict_proba(X)[:, 1]
    calibrated = calibrator.predict(raw)
    return calibrated


# ---------------------------------------------------------------------------
# Section 7-8, 12: Walk-forward backtest with meta-model gating
# ---------------------------------------------------------------------------

def run_meta_walkforward(starting_bankroll=100.0, kelly_fraction=0.25,
                         max_kelly_pct=0.10, min_wager=1.0,
                         model_type="xgb", meta_min_windows=2):
    """Full walk-forward backtest with meta-model gating.

    For each window:
    1. Train main model on past data
    2. Generate predictions
    3. Build meta features
    4. Train meta-model on ALL previous windows (temporal safety)
    5. Score candidate bets
    6. Compare gated vs ungated vs rules-based strategies
    """
    # Build the full meta dataset first
    meta_df = build_meta_dataset()
    if meta_df.empty:
        return {"error": "No meta data"}

    windows = sorted(meta_df["window"].unique())
    logger.info("Windows available: %s", windows)

    # Strategies to track
    strategies = {
        "all_positive_ev": {"bankroll": starting_bankroll, "bets": []},
        "rules_under_low": {"bankroll": starting_bankroll, "bets": []},
        "meta_logistic": {"bankroll": starting_bankroll, "bets": []},
        "meta_xgb": {"bankroll": starting_bankroll, "bets": []},
    }

    # Threshold grid for meta gating
    edge_thresholds = [0.03, 0.05, 0.06, 0.08, 0.10]
    conf_thresholds = [0.55, 0.60, 0.65, 0.70]
    threshold_results = {}
    for et in edge_thresholds:
        for ct in conf_thresholds:
            threshold_results[(et, ct)] = {"bankroll": starting_bankroll, "bets": 0,
                                           "wins": 0, "profit": 0.0, "wagered": 0.0}

    meta_model_log = None
    meta_cal_log = None
    meta_model_xgb = None
    meta_cal_xgb = None

    window_reports = []

    for w_idx, w in enumerate(windows):
        w_data = meta_df[meta_df["window"] == w].copy()
        past_data = meta_df[meta_df["window"] < w].copy()

        # Only +EV candidates
        candidates = w_data[w_data["edge_selected_side"] > 0].copy()

        if len(candidates) == 0:
            continue

        # Train meta-models if we have enough past data
        if len(past_data) > 100 and w_idx >= meta_min_windows:
            past_pos_ev = past_data[past_data["edge_selected_side"] > 0]
            if len(past_pos_ev) > 50:
                try:
                    meta_model_log, meta_cal_log = train_meta_model(past_pos_ev, "logistic")
                    meta_model_xgb, meta_cal_xgb = train_meta_model(past_pos_ev, "xgb")
                    candidates["meta_prob_log"] = predict_meta(meta_model_log, meta_cal_log, candidates)
                    candidates["meta_prob_xgb"] = predict_meta(meta_model_xgb, meta_cal_xgb, candidates)
                except Exception as e:
                    logger.warning("Meta-model training failed for window %d: %s", w, e)
                    candidates["meta_prob_log"] = 0.5
                    candidates["meta_prob_xgb"] = 0.5
            else:
                candidates["meta_prob_log"] = 0.5
                candidates["meta_prob_xgb"] = 0.5
        else:
            candidates["meta_prob_log"] = 0.5
            candidates["meta_prob_xgb"] = 0.5

        # Process each candidate bet under different strategies
        for _, b in candidates.iterrows():
            edge = b["edge_selected_side"]
            won = bool(b["bet_won"])
            dec_odds = b["decimal_odds"]
            meta_log = b["meta_prob_log"]
            meta_xgb = b["meta_prob_xgb"]
            is_under_low = b["low_line_under_flag"] == 1

            for sname, strat in strategies.items():
                place = False
                if sname == "all_positive_ev":
                    place = edge > 0
                elif sname == "rules_under_low":
                    place = is_under_low and edge > 0
                elif sname == "meta_logistic":
                    place = edge >= 0.05 and meta_log >= 0.55
                elif sname == "meta_xgb":
                    place = edge >= 0.05 and meta_xgb >= 0.55

                if place:
                    kf = max(0, edge / (dec_odds - 1)) * kelly_fraction
                    kf = min(kf, max_kelly_pct)
                    wager = round(strat["bankroll"] * kf, 2)
                    if wager < min_wager:
                        continue
                    profit = round(wager * (dec_odds - 1), 2) if won else -wager
                    strat["bankroll"] = round(strat["bankroll"] + profit, 2)
                    strat["bets"].append({
                        "window": w, "date": b["game_date"],
                        "pitcher": b["pitcher"], "line": b["market_line"],
                        "side": b["bet_side"], "edge": edge,
                        "meta_log": meta_log, "meta_xgb": meta_xgb,
                        "wager": wager, "won": won, "profit": profit,
                        "bankroll": strat["bankroll"],
                    })

            # Threshold grid (use XGBoost meta)
            for (et, ct), tr in threshold_results.items():
                if edge >= et and meta_xgb >= ct:
                    kf = max(0, edge / (dec_odds - 1)) * kelly_fraction
                    kf = min(kf, max_kelly_pct)
                    wager = round(tr["bankroll"] * kf, 2)
                    if wager < min_wager:
                        continue
                    profit = round(wager * (dec_odds - 1), 2) if won else -wager
                    tr["bankroll"] = round(tr["bankroll"] + profit, 2)
                    tr["bets"] += 1
                    tr["wins"] += int(won)
                    tr["profit"] += profit
                    tr["wagered"] += wager

        # Window report
        wr = {"window": w}
        for sname in strategies:
            wb = [b for b in strategies[sname]["bets"] if b["window"] == w]
            wr[f"{sname}_bets"] = len(wb)
            wr[f"{sname}_wins"] = sum(1 for b in wb if b["won"])
            wr[f"{sname}_profit"] = round(sum(b["profit"] for b in wb), 2)
            wr[f"{sname}_bankroll"] = strategies[sname]["bankroll"]
        window_reports.append(wr)

    # Compile final results
    result = {
        "strategies": {},
        "threshold_grid": {},
        "window_reports": window_reports,
    }

    for sname, strat in strategies.items():
        bets_list = strat["bets"]
        if not bets_list:
            result["strategies"][sname] = {
                "ending_bankroll": strat["bankroll"], "bets": 0,
            }
            continue
        bdf = pd.DataFrame(bets_list)
        wins = bdf["won"].sum()
        wagered = bdf["wager"].sum()
        profit = strat["bankroll"] - starting_bankroll

        # Drawdown
        running = [starting_bankroll]
        for p in bdf["profit"].values:
            running.append(running[-1] + p)
        peak, max_dd = starting_bankroll, 0
        for v in running:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        result["strategies"][sname] = {
            "ending_bankroll": round(strat["bankroll"], 2),
            "profit": round(profit, 2),
            "roi_pct": round(profit / starting_bankroll * 100, 1),
            "total_bets": len(bdf),
            "wins": int(wins),
            "win_rate": round(wins / len(bdf) * 100, 1),
            "total_wagered": round(wagered, 2),
            "yield_pct": round(profit / max(wagered, 1) * 100, 1),
            "avg_edge": round(bdf["edge"].mean() * 100, 1),
            "max_drawdown_pct": round(max_dd * 100, 1),
            "peak_bankroll": round(max(running), 2),
        }

    # Threshold grid
    for (et, ct), tr in threshold_results.items():
        profit = tr["bankroll"] - starting_bankroll
        key = f"edge>={et:.2f}_conf>={ct:.2f}"
        result["threshold_grid"][key] = {
            "ending_bankroll": round(tr["bankroll"], 2),
            "profit": round(profit, 2),
            "bets": tr["bets"],
            "wins": tr["wins"],
            "win_rate": round(tr["wins"] / max(tr["bets"], 1) * 100, 1),
            "wagered": round(tr["wagered"], 2),
            "yield_pct": round(profit / max(tr["wagered"], 1) * 100, 1),
        }

    # Meta-model classification metrics (on last window's candidates)
    if meta_model_xgb is not None and len(meta_df) > 0:
        last_w = windows[-1]
        eval_data = meta_df[meta_df["window"] == last_w]
        eval_pos = eval_data[eval_data["edge_selected_side"] > 0]
        if len(eval_pos) > 10:
            probs = predict_meta(meta_model_xgb, meta_cal_xgb, eval_pos)
            y_true = eval_pos["target"].values
            try:
                result["meta_classification"] = {
                    "auc": round(roc_auc_score(y_true, probs), 3),
                    "log_loss": round(log_loss(y_true, probs), 3),
                    "brier_score": round(brier_score_loss(y_true, probs), 3),
                    "mean_pred": round(float(np.mean(probs)), 3),
                    "mean_actual": round(float(np.mean(y_true)), 3),
                    "calibration_error": round(abs(float(np.mean(probs)) - float(np.mean(y_true))), 3),
                    "n_eval": len(eval_pos),
                }
            except Exception:
                pass

        # Feature importance
        imp = meta_model_xgb.feature_importances_
        feat_imp = sorted(zip(META_FEATURES, imp), key=lambda x: x[1], reverse=True)
        result["meta_feature_importance"] = [
            {"feature": f, "importance": round(float(v), 4)} for f, v in feat_imp[:20]
        ]

    result["meta_df"] = meta_df
    return result


def print_meta_report(result):
    """Print formatted meta-model report."""
    print("=" * 80)
    print("  META-MODEL GATING LAYER — WALK-FORWARD BACKTEST REPORT")
    print("=" * 80)

    # Strategy comparison
    print()
    print("  STRATEGY COMPARISON ($100 starting bankroll):")
    print(f"  {'Strategy':30s} {'Bank':>8s} {'P&L':>9s} {'ROI':>7s} {'Bets':>5s} "
          f"{'W%':>6s} {'Yield':>7s} {'MaxDD':>6s}")
    print("  " + "-" * 82)
    for sname, s in result["strategies"].items():
        if s.get("total_bets", 0) == 0:
            print(f"  {sname:30s} ${s['ending_bankroll']:7.2f}     (no bets)")
            continue
        print(f"  {sname:30s} ${s['ending_bankroll']:7.2f} ${s['profit']:+8.2f} "
              f"{s['roi_pct']:+6.1f}% {s['total_bets']:5d} "
              f"{s['win_rate']:5.1f}% {s['yield_pct']:+6.1f}% {s['max_drawdown_pct']:5.1f}%")

    # Meta classification metrics
    if "meta_classification" in result:
        mc = result["meta_classification"]
        print()
        print("  META-MODEL CLASSIFICATION METRICS (last window):")
        print(f"    AUC:               {mc['auc']}")
        print(f"    Log Loss:          {mc['log_loss']}")
        print(f"    Brier Score:       {mc['brier_score']}")
        print(f"    Calibration Error: {mc['calibration_error']}")
        print(f"    Eval samples:      {mc['n_eval']}")

    # Feature importance
    if "meta_feature_importance" in result:
        print()
        print("  META-MODEL TOP 20 FEATURES:")
        for i, fi in enumerate(result["meta_feature_importance"], 1):
            print(f"    {i:2d}. {fi['feature']:35s} {fi['importance']:.4f}")

    # Threshold grid
    print()
    print("  THRESHOLD GRID (XGBoost meta, edge x confidence):")
    print(f"  {'Thresholds':30s} {'Bank':>8s} {'Bets':>5s} {'W%':>6s} {'Yield':>7s}")
    print("  " + "-" * 60)
    for key, tr in sorted(result["threshold_grid"].items()):
        if tr["bets"] > 0:
            print(f"  {key:30s} ${tr['ending_bankroll']:7.2f} {tr['bets']:5d} "
                  f"{tr['win_rate']:5.1f}% {tr['yield_pct']:+6.1f}%")

    # Per-window breakdown
    print()
    print("  PER-WINDOW BREAKDOWN:")
    print(f"  {'Win':>4s}  {'AllEV':>18s}  {'Rules':>18s}  {'MetaXGB':>18s}")
    print(f"  {'':4s}  {'bets P&L  bank':>18s}  {'bets P&L  bank':>18s}  {'bets P&L  bank':>18s}")
    print("  " + "-" * 65)
    for wr in result["window_reports"]:
        def _fmt(prefix):
            b = wr.get(f"{prefix}_bets", 0)
            p = wr.get(f"{prefix}_profit", 0)
            bk = wr.get(f"{prefix}_bankroll", 0)
            return f"{b:4d} ${p:+7.2f} ${bk:7.2f}"
        print(f"  {wr['window']:4d}  {_fmt('all_positive_ev')}  {_fmt('rules_under_low')}  {_fmt('meta_xgb')}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_meta_walkforward(starting_bankroll=100.0, kelly_fraction=0.25)
    print_meta_report(result)
