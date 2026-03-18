"""
MLB Walk-Forward Backtest — Rolling window with sharp-vs-soft strategy evaluation.

Trains on expanding window, tests on next 2 weeks, slides forward.
No data leakage — model never sees future data in any window.

Matches model predictions with actual sportsbook pitcher K props from
multiple bookmakers, simulates distributions, and evaluates many betting
strategies including sharp-book confirmation and soft-book targeting.

Sharp books: FanDuel, DraftKings, BetOnlineAg
Soft books:  BetMGM, William Hill (US)
"""

import logging
import sqlite3
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import mlb_model
import mlb_simulation
import mlb_odds_collector

logger = logging.getLogger(__name__)


def _train_window_model(train_df):
    """Train BF + KBF models on a training window. Returns (model_bf, model_kbf, bf_std_map, kbf_std_map)."""
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

    # Pitcher-specific variance
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
        if n >= 5:
            shrink = min(n / 15.0, 1.0)
            bf_std_map[pid] = shrink * float(grp["bf_r"].std()) + (1 - shrink) * g_bf_std
            kbf_std_map[pid] = shrink * float(grp["kbf_r"].std()) + (1 - shrink) * g_kbf_std
        else:
            bf_std_map[pid] = g_bf_std
            kbf_std_map[pid] = g_kbf_std

    return model_bf, model_kbf, bf_std_map, kbf_std_map


def _american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds < 0:
        return 1 + 100 / abs(odds)
    elif odds > 0:
        return 1 + odds / 100
    return 2.0


def _american_to_implied(odds):
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def run_walkforward(starting_bankroll=100.0, kelly_fraction=0.25, max_kelly_pct=0.10,
                    min_edge=0.0, min_wager=1.0,
                    min_train_days=60, test_window_days=14, step_days=14):
    """Run walk-forward backtest across entire 2024 season with strategy matrix.

    Collects ALL candidate bets from all bookmakers, then evaluates many
    strategy variants including sharp confirmation and soft book targeting.

    Args:
        min_train_days: Minimum training window before first test (days from season start)
        test_window_days: Size of each test window in days
        step_days: How far to slide the window each iteration
    """
    # Build full feature matrix once
    logger.info("Building full feature matrix...")
    df = mlb_model._build_feature_dataframe()
    if df.empty:
        return {"error": "No data"}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    season_start = df["date"].min()
    season_end = df["date"].max()
    logger.info("Season: %s to %s (%d starts)", season_start.date(), season_end.date(), len(df))

    # Load per-bookmaker props (all books, not just one)
    per_book_props = mlb_odds_collector.load_per_book_props_bulk()
    logger.info("Per-book props loaded: %d entries", len(per_book_props))

    # Load sharp consensus data
    sharp_map = mlb_odds_collector.load_sharp_consensus_bulk()
    logger.info("Sharp consensus loaded: %d entries", len(sharp_map))

    # Build a consolidated prop lookup across all books (consensus odds)
    conn = sqlite3.connect("mlb_data.db")
    conn.row_factory = sqlite3.Row
    props_df = pd.read_sql_query(
        "SELECT pitcher_name, game_date, line, over_under, price, bookmaker "
        "FROM mlb_pitcher_props",
        conn,
    )

    # Pitcher name lookup: (pitcher_id, date) -> normalized name
    name_rows = conn.execute(
        "SELECT pitcher_id, date, pitcher_name FROM mlb_pitcher_game_stats"
    ).fetchall()
    pid_date_to_name = {}
    for r in name_rows:
        pid_date_to_name[(r["pitcher_id"], r["date"])] = mlb_model._normalize_name(r["pitcher_name"])
    conn.close()

    # Build consensus prop lookup (average across all books)
    prop_lines = {}  # (pname_norm, game_date, line) -> {over_price, under_price}
    for _, r in props_df.iterrows():
        key = (mlb_model._normalize_name(r["pitcher_name"]), r["game_date"], r["line"])
        if key not in prop_lines:
            prop_lines[key] = {"over_prices": [], "under_prices": []}
        if r["over_under"] == "Over":
            prop_lines[key]["over_prices"].append(int(r["price"]))
        else:
            prop_lines[key]["under_prices"].append(int(r["price"]))

    # Average the prices for consensus
    consensus_props = {}
    for key, prices in prop_lines.items():
        entry = {}
        if prices["over_prices"]:
            entry["over_price"] = int(round(sum(prices["over_prices"]) / len(prices["over_prices"])))
        if prices["under_prices"]:
            entry["under_price"] = int(round(sum(prices["under_prices"]) / len(prices["under_prices"])))
        if entry:
            consensus_props[key] = entry

    logger.info("Consensus prop lines loaded: %d", len(consensus_props))

    # Walk-forward loop
    all_bets = []
    window_results = []

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

        # Train on this window
        logger.info("Window %d: train %s-%s (%d), test %s-%s (%d)",
                     window_num,
                     train_df["date"].min().date(), train_df["date"].max().date(), len(train_df),
                     test_df["date"].min().date(), test_df["date"].max().date(), len(test_df))

        model_bf, model_kbf, bf_std_map, kbf_std_map = _train_window_model(train_df)

        # Predict on test
        bf_pred = test_df["baseline_bf"].values + model_bf.predict(test_df[mlb_model.BF_FEATURES].values)
        bf_pred = np.clip(bf_pred, 10, 40)
        kbf_pred = test_df["baseline_k_rate"].values + model_kbf.predict(test_df[mlb_model.KBF_FEATURES].values)
        kbf_pred = np.clip(kbf_pred, 0.0, 0.60)
        pred_k = bf_pred * kbf_pred
        actual_k = test_df["strikeouts"].values

        window_mae = mean_absolute_error(actual_k, pred_k)
        market_lines = test_df["market_k_line"].values
        market_mae = mean_absolute_error(actual_k, market_lines)

        # Simulate for each test start
        sim_results = []
        for i in range(len(test_df)):
            pid = int(test_df.iloc[i]["pitcher_id"])
            bf_s = bf_std_map.get(pid, bf_pred[i] * 0.18)
            kbf_s = kbf_std_map.get(pid, kbf_pred[i] * 0.20)
            sim = mlb_simulation.simulate_strikeouts(
                bf_pred[i], kbf_pred[i], bf_std=bf_s, kbf_std=kbf_s,
                n_sims=10000, seed=42 + window_num * 1000 + i,
            )
            sim_results.append(sim)

        # Collect ALL candidate bets (strategies will filter later)
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            sim = sim_results[i]
            ak = int(row["strikeouts"])
            gd = row["date"].strftime("%Y-%m-%d")
            pid = int(row["pitcher_id"])
            pname_norm = pid_date_to_name.get((pid, gd), "")
            if not pname_norm:
                continue

            for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
                # Check if any book has this line for this pitcher
                lk = (pname_norm, gd, line)
                if lk not in consensus_props:
                    continue
                cp = consensus_props[lk]

                model_p_over = sim.get(f"P_over_{line}", 0)
                model_p_under = 1.0 - model_p_over

                # Sharp consensus for this pitcher-date
                sharp_entry = sharp_map.get((gd, pname_norm))

                # Soft book prices (try BetMGM first, then William Hill)
                soft_prices = None
                for soft_book in mlb_odds_collector.SOFT_BOOKS:
                    sbk = (gd, pname_norm, line, soft_book)
                    sp = per_book_props.get(sbk)
                    if sp and sp.get("over_price") is not None:
                        soft_prices = sp
                        break

                for side, price_key, model_p, won_fn in [
                    ("OVER", "over_price", model_p_over, lambda k, l: k > l),
                    ("UNDER", "under_price", model_p_under, lambda k, l: k <= l),
                ]:
                    price = cp.get(price_key)
                    if price is None or model_p <= 0.01:
                        continue

                    implied = _american_to_implied(price)
                    decimal_odds = _american_to_decimal(price)
                    edge = model_p - implied
                    ev = model_p * (decimal_odds - 1) - (1 - model_p)

                    # Sharp book signal
                    sharp_prob = None
                    sharp_edge = None
                    n_sharp = 0
                    sharp_agrees = False
                    if sharp_entry and sharp_entry["line"] == line:
                        sp = (sharp_entry["sharp_prob_over"] if side == "OVER"
                              else sharp_entry["sharp_prob_under"])
                        sharp_prob = round(sp, 4)
                        sharp_edge = round(sp - implied, 4)
                        n_sharp = sharp_entry["n_sharp_books"]
                        sharp_agrees = (
                            (side == "OVER" and sp > 0.5)
                            or (side == "UNDER" and sp < 0.5)
                        )

                    # Blended prob: 50/50 model + sharp
                    blended_prob = None
                    if sharp_prob is not None:
                        blended_prob = round(0.5 * model_p + 0.5 * sharp_prob, 4)

                    # Soft book prices
                    soft_price = None
                    soft_implied = None
                    soft_edge = None
                    has_soft = False
                    if soft_prices:
                        has_soft = True
                        sp_key = ("over_price" if side == "OVER" else "under_price")
                        soft_price = soft_prices.get(sp_key)
                        if soft_price is not None:
                            soft_implied = round(_american_to_implied(soft_price), 4)
                            soft_edge = round(model_p - soft_implied, 4)

                    won = won_fn(ak, line)

                    all_bets.append({
                        "window": window_num,
                        "date": gd,
                        "pitcher": pname_norm,
                        "pitcher_id": pid,
                        "line": line,
                        "side": side,
                        "odds": price,
                        "model_prob": round(model_p, 4),
                        "implied_prob": round(implied, 4),
                        "edge": round(edge, 4),
                        "ev": round(ev, 4),
                        "decimal_odds": round(decimal_odds, 4),
                        "pred_k": round(float(pred_k[i]), 2),
                        "actual_k": ak,
                        "won": won,
                        "train_size": len(train_df),
                        "market_k_line": float(row["market_k_line"]),
                        # Sharp book consensus
                        "sharp_prob": sharp_prob,
                        "sharp_edge": sharp_edge,
                        "n_sharp_books": n_sharp,
                        "sharp_agrees": sharp_agrees,
                        # Blended prob: 50/50 model + sharp
                        "blended_prob": blended_prob,
                        # Soft book prices
                        "has_soft": has_soft,
                        "soft_price": soft_price,
                        "soft_implied": soft_implied,
                        "soft_edge": soft_edge,
                    })

        window_results.append({
            "window": window_num,
            "train_start": train_df["date"].min().strftime("%Y-%m-%d"),
            "train_end": train_df["date"].max().strftime("%Y-%m-%d"),
            "test_start": test_df["date"].min().strftime("%Y-%m-%d"),
            "test_end": test_df["date"].max().strftime("%Y-%m-%d"),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "model_mae": round(window_mae, 3),
            "market_mae": round(market_mae, 3),
            "mae_edge": round(market_mae - window_mae, 3),
        })

        current_test_start += timedelta(days=step_days)

    # Compile all bets
    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

    if bets_df.empty:
        logger.warning("No bets generated — check odds data coverage")
        return {"error": "No bets", "windows": window_results}

    logger.info("Total candidate bets: %d", len(bets_df))

    # =====================================================================
    # STRATEGY EVALUATION
    # =====================================================================
    strategies = _evaluate_strategies(bets_df, starting_bankroll,
                                      kelly_fraction, max_kelly_pct, min_wager)

    return {
        "period": f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
        "total_starts": len(df),
        "windows": window_results,
        "total_candidate_bets": len(bets_df),
        "strategies": strategies,
        "bets_df": bets_df,
    }


def _simulate_strategy(bets_df, starting_bankroll, kelly_fraction,
                        max_kelly_pct, min_wager, min_edge=0.0,
                        use_soft_odds=False, use_sharp_prob=False,
                        use_blended_prob=False):
    """Simulate P&L for a filtered set of bets using Kelly sizing.

    If use_soft_odds=True, uses soft book (BetMGM) odds for payouts.
    If use_sharp_prob=True, uses sharp book consensus probability for edge calc.
    If use_blended_prob=True, uses 50/50 blend of model + sharp prob.
    """
    bets_df = bets_df.sort_values("date").reset_index(drop=True)

    bankroll = starting_bankroll
    results = []

    for _, bet in bets_df.iterrows():
        # Determine probability estimate
        if use_blended_prob and bet.get("blended_prob") is not None:
            prob = bet["blended_prob"]
        elif use_sharp_prob and bet.get("sharp_prob") is not None:
            prob = bet["sharp_prob"]
        else:
            prob = bet["model_prob"]

        # Determine which odds to bet into
        if use_soft_odds:
            if bet.get("soft_price") is None:
                continue
            dec_odds = _american_to_decimal(bet["soft_price"])
            imp = _american_to_implied(bet["soft_price"])
        else:
            dec_odds = bet["decimal_odds"]
            imp = bet["implied_prob"]

        edge_val = prob - imp
        ev_val = prob * (dec_odds - 1) - (1 - prob)

        if edge_val <= min_edge or ev_val <= 0:
            continue

        # Kelly sizing
        kf = ev_val / (dec_odds - 1) if dec_odds > 1 else 0
        kf = max(kf, 0) * kelly_fraction
        kf = min(kf, max_kelly_pct)

        wager = round(bankroll * kf, 2)
        if wager < min_wager:
            continue

        won = bet["won"]
        profit = round(wager * (dec_odds - 1), 2) if won else -wager
        bankroll = round(bankroll + profit, 2)

        results.append({
            "wager": wager, "won": won, "profit": profit,
            "bankroll": bankroll, "edge": edge_val,
        })

    if not results:
        return None

    results_df = pd.DataFrame(results)
    total_wagered = results_df["wager"].sum()
    total_profit = bankroll - starting_bankroll

    # Drawdown
    running = [starting_bankroll]
    for p in results_df["profit"]:
        running.append(running[-1] + p)
    peak = starting_bankroll
    max_dd = 0
    for v in running:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "bets": len(results_df),
        "wins": int(results_df["won"].sum()),
        "win_rate": round(results_df["won"].mean() * 100, 1),
        "profit": round(total_profit, 2),
        "wagered": round(total_wagered, 2),
        "yield_pct": round(total_profit / max(total_wagered, 1) * 100, 1),
        "ending_bankroll": round(bankroll, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "avg_edge": round(results_df["edge"].mean() * 100, 1),
        "bankroll_curve": running,
    }


def _evaluate_strategies(bets_df, starting_bankroll, kelly_fraction,
                          max_kelly_pct, min_wager):
    """Evaluate many betting strategies to find profitable niches."""
    strategies = {}

    def _run(name, mask, min_edge=0.0, use_soft_odds=False,
             use_sharp_prob=False, use_blended_prob=False):
        subset = bets_df[mask].copy()
        if len(subset) == 0:
            return
        result = _simulate_strategy(
            subset, starting_bankroll, kelly_fraction,
            max_kelly_pct, min_wager, min_edge, use_soft_odds,
            use_sharp_prob, use_blended_prob,
        )
        if result:
            strategies[name] = result

    ev_plus = bets_df["ev"] > 0

    # === BASELINE STRATEGIES (consensus odds from all books) ===
    _run("all_ev_plus", ev_plus)

    # By side
    _run("overs_only", ev_plus & (bets_df["side"] == "OVER"))
    _run("unders_only", ev_plus & (bets_df["side"] == "UNDER"))

    # By line
    for line in sorted(bets_df["line"].unique()):
        _run(f"line_{line}", ev_plus & (bets_df["line"] == line))

    # By side + line
    for line in sorted(bets_df["line"].unique()):
        _run(f"over_{line}", ev_plus & (bets_df["side"] == "OVER") & (bets_df["line"] == line))
        _run(f"under_{line}", ev_plus & (bets_df["side"] == "UNDER") & (bets_df["line"] == line))

    # Under combos (MLB K props)
    for lines in [(4.5,), (5.5,), (4.5, 5.5), (3.5, 4.5), (5.5, 6.5), (4.5, 5.5, 6.5)]:
        name = "under_" + "+".join(str(l) for l in lines)
        _run(name, ev_plus & (bets_df["side"] == "UNDER") & bets_df["line"].isin(lines))

    # Over combos
    for lines in [(4.5,), (5.5,), (6.5,), (4.5, 5.5), (5.5, 6.5), (6.5, 7.5)]:
        name = "over_" + "+".join(str(l) for l in lines)
        _run(name, ev_plus & (bets_df["side"] == "OVER") & bets_df["line"].isin(lines))

    # By edge threshold
    for edge_pct in [3, 5, 8]:
        edge_thresh = edge_pct / 100.0
        _run(f"all_edge_{edge_pct}pct", ev_plus, min_edge=edge_thresh)
        _run(f"unders_edge_{edge_pct}pct",
             ev_plus & (bets_df["side"] == "UNDER"), min_edge=edge_thresh)
        _run(f"overs_edge_{edge_pct}pct",
             ev_plus & (bets_df["side"] == "OVER"), min_edge=edge_thresh)

    # High-K pitcher filter (market line >= 6.5 = elite strikeout pitchers)
    high_k = bets_df["market_k_line"] >= 6.5
    _run("high_k_all", ev_plus & high_k)
    _run("high_k_overs", ev_plus & high_k & (bets_df["side"] == "OVER"))
    _run("high_k_unders", ev_plus & high_k & (bets_df["side"] == "UNDER"))

    # Low-K pitcher filter (market line <= 4.5)
    low_k = bets_df["market_k_line"] <= 4.5
    _run("low_k_all", ev_plus & low_k)
    _run("low_k_overs", ev_plus & low_k & (bets_df["side"] == "OVER"))
    _run("low_k_unders", ev_plus & low_k & (bets_df["side"] == "UNDER"))

    # =================================================================
    # SHARP-VS-SOFT META STRATEGIES
    # =================================================================
    has_sharp = bets_df["sharp_prob"].notna()
    has_soft_col = bets_df["has_soft"] == True  # noqa: E712
    has_both = has_sharp & has_soft_col
    sharp_agrees = bets_df["sharp_agrees"] == True  # noqa: E712

    # -- Soft book baseline: model +EV vs BetMGM odds --
    soft_ev = has_soft_col & (bets_df["soft_edge"].fillna(0) > 0)
    _run("BMG_all_ev_plus", soft_ev, use_soft_odds=True)
    _run("BMG_overs", soft_ev & (bets_df["side"] == "OVER"), use_soft_odds=True)
    _run("BMG_unders", soft_ev & (bets_df["side"] == "UNDER"), use_soft_odds=True)

    # By line on soft book
    for line in sorted(bets_df["line"].unique()):
        _run(f"BMG_over_{line}",
             soft_ev & (bets_df["side"] == "OVER") & (bets_df["line"] == line),
             use_soft_odds=True)
        _run(f"BMG_under_{line}",
             soft_ev & (bets_df["side"] == "UNDER") & (bets_df["line"] == line),
             use_soft_odds=True)

    # Soft book edge thresholds
    for edge_pct in [3, 5, 8]:
        et = edge_pct / 100.0
        _run(f"BMG_all_edge_{edge_pct}pct", soft_ev, min_edge=et,
             use_soft_odds=True)
        _run(f"BMG_unders_edge_{edge_pct}pct",
             soft_ev & (bets_df["side"] == "UNDER"),
             min_edge=et, use_soft_odds=True)
        _run(f"BMG_overs_edge_{edge_pct}pct",
             soft_ev & (bets_df["side"] == "OVER"),
             min_edge=et, use_soft_odds=True)

    # -- SHARP CONFIRMS: model +EV AND sharp books agree on direction --
    sharp_confirm = soft_ev & has_both & sharp_agrees
    _run("BMG+sharp_all", sharp_confirm, use_soft_odds=True)
    _run("BMG+sharp_overs",
         sharp_confirm & (bets_df["side"] == "OVER"), use_soft_odds=True)
    _run("BMG+sharp_unders",
         sharp_confirm & (bets_df["side"] == "UNDER"), use_soft_odds=True)

    for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
        _run(f"BMG+sharp_over_{line}",
             sharp_confirm & (bets_df["side"] == "OVER") & (bets_df["line"] == line),
             use_soft_odds=True)
        _run(f"BMG+sharp_under_{line}",
             sharp_confirm & (bets_df["side"] == "UNDER") & (bets_df["line"] == line),
             use_soft_odds=True)

    for edge_pct in [3, 5, 8]:
        et = edge_pct / 100.0
        _run(f"BMG+sharp_all_{edge_pct}pct", sharp_confirm,
             min_edge=et, use_soft_odds=True)
        _run(f"BMG+sharp_unders_{edge_pct}pct",
             sharp_confirm & (bets_df["side"] == "UNDER"),
             min_edge=et, use_soft_odds=True)
        _run(f"BMG+sharp_overs_{edge_pct}pct",
             sharp_confirm & (bets_df["side"] == "OVER"),
             min_edge=et, use_soft_odds=True)

    # N sharp books filter with confirmation
    for n_min in [2, 3]:
        n_mask = sharp_confirm & (bets_df["n_sharp_books"] >= n_min)
        _run(f"BMG+sharp_{n_min}books",
             n_mask, use_soft_odds=True)
        _run(f"BMG+sharp_{n_min}books_unders",
             n_mask & (bets_df["side"] == "UNDER"), use_soft_odds=True)
        _run(f"BMG+sharp_{n_min}books_overs",
             n_mask & (bets_df["side"] == "OVER"), use_soft_odds=True)

    # High-K with sharp confirmation
    _run("BMG+sharp_highK_unders",
         sharp_confirm & high_k & (bets_df["side"] == "UNDER"), use_soft_odds=True)
    _run("BMG+sharp_highK_overs",
         sharp_confirm & high_k & (bets_df["side"] == "OVER"), use_soft_odds=True)
    _run("BMG+sharp_lowK_unders",
         sharp_confirm & low_k & (bets_df["side"] == "UNDER"), use_soft_odds=True)
    _run("BMG+sharp_lowK_overs",
         sharp_confirm & low_k & (bets_df["side"] == "OVER"), use_soft_odds=True)

    # -- SHARP DISAGREES: model says +EV but sharp leans other way --
    sharp_disagree = soft_ev & has_both & (~sharp_agrees)
    _run("BMG_sharp_disagrees", sharp_disagree, use_soft_odds=True)
    _run("BMG_sharp_disagrees_unders",
         sharp_disagree & (bets_df["side"] == "UNDER"), use_soft_odds=True)
    _run("BMG_sharp_disagrees_overs",
         sharp_disagree & (bets_df["side"] == "OVER"), use_soft_odds=True)

    # -- BLENDED PROB: 50/50 model + sharp for sizing --
    has_blended = bets_df["blended_prob"].notna()
    blend_soft = has_soft_col & has_blended
    _run("BMG_blend_all", blend_soft, use_soft_odds=True,
         use_blended_prob=True)
    _run("BMG_blend_overs",
         blend_soft & (bets_df["side"] == "OVER"),
         use_soft_odds=True, use_blended_prob=True)
    _run("BMG_blend_unders",
         blend_soft & (bets_df["side"] == "UNDER"),
         use_soft_odds=True, use_blended_prob=True)

    for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
        _run(f"BMG_blend_over_{line}",
             blend_soft & (bets_df["side"] == "OVER") & (bets_df["line"] == line),
             use_soft_odds=True, use_blended_prob=True)
        _run(f"BMG_blend_under_{line}",
             blend_soft & (bets_df["side"] == "UNDER") & (bets_df["line"] == line),
             use_soft_odds=True, use_blended_prob=True)

    # Blended + edge thresholds
    for edge_pct in [3, 5, 8]:
        et = edge_pct / 100.0
        _run(f"BMG_blend_all_{edge_pct}pct", blend_soft,
             min_edge=et, use_soft_odds=True, use_blended_prob=True)
        _run(f"BMG_blend_unders_{edge_pct}pct",
             blend_soft & (bets_df["side"] == "UNDER"),
             min_edge=et, use_soft_odds=True, use_blended_prob=True)
        _run(f"BMG_blend_overs_{edge_pct}pct",
             blend_soft & (bets_df["side"] == "OVER"),
             min_edge=et, use_soft_odds=True, use_blended_prob=True)

    # -- BMG LINE COMBOS --
    for lines in [(4.5, 5.5), (5.5, 6.5), (4.5, 5.5, 6.5), (6.5, 7.5)]:
        lbl = "+".join(str(l) for l in lines)
        _run(f"BMG_under_{lbl}",
             soft_ev & (bets_df["side"] == "UNDER") & bets_df["line"].isin(lines),
             use_soft_odds=True)
        _run(f"BMG+sharp_under_{lbl}",
             sharp_confirm & (bets_df["side"] == "UNDER") & bets_df["line"].isin(lines),
             use_soft_odds=True)
        _run(f"BMG_over_{lbl}",
             soft_ev & (bets_df["side"] == "OVER") & bets_df["line"].isin(lines),
             use_soft_odds=True)
        _run(f"BMG+sharp_over_{lbl}",
             sharp_confirm & (bets_df["side"] == "OVER") & bets_df["line"].isin(lines),
             use_soft_odds=True)

    # -- High-K pitchers at soft book --
    _run("BMG_highK_unders",
         soft_ev & (bets_df["side"] == "UNDER") & high_k,
         use_soft_odds=True)
    _run("BMG_highK_overs",
         soft_ev & (bets_df["side"] == "OVER") & high_k,
         use_soft_odds=True)
    _run("BMG+sharp_highK_all",
         sharp_confirm & high_k, use_soft_odds=True)

    return strategies


def print_walkforward(result):
    """Print formatted walk-forward results with strategy sections."""
    if "error" in result:
        print(f"Walk-forward error: {result['error']}")
        if "windows" in result:
            print(f"  Windows completed: {len(result['windows'])}")
        return

    print("=" * 100)
    print("  MLB PITCHER K WALK-FORWARD BACKTEST: Sharp-vs-Soft Meta Analysis")
    print("=" * 100)
    print(f"  Period:              {result['period']}")
    print(f"  Pitcher starts:      {result['total_starts']}")
    print(f"  Walk-forward windows: {len(result['windows'])}")
    print(f"  Candidate bets:      {result['total_candidate_bets']}")

    # Windows
    print()
    print("  WALK-FORWARD WINDOWS:")
    print(f"  {'Win':>4s}  {'Train Period':24s} {'Trn':>5s}  {'Test Period':24s} {'Tst':>4s}  "
          f"{'MAE':>5s} {'MktMAE':>6s} {'Edge':>5s}")
    print("  " + "-" * 95)
    for w in result["windows"]:
        print(f"  {w['window']:4d}  {w['train_start']} - {w['train_end']}  {w['train_size']:5d}  "
              f"{w['test_start']} - {w['test_end']}  {w['test_size']:4d}  "
              f"{w['model_mae']:5.3f} {w['market_mae']:6.3f} {w['mae_edge']:+5.3f}")

    strategies = result.get("strategies", {})
    if not strategies:
        print("\n  No strategies to display.")
        print("=" * 100)
        return

    def _print_section(title, prefix_filter=None, keys=None, min_bets=5):
        """Print a section of strategies."""
        if keys:
            items = [(k, strategies[k]) for k in keys if k in strategies]
        elif prefix_filter:
            items = [(k, v) for k, v in strategies.items()
                     if k.startswith(prefix_filter)]
        else:
            items = list(strategies.items())
        items = [(k, v) for k, v in items if v["bets"] >= min_bets]
        if not items:
            return
        items.sort(key=lambda x: x[1]["yield_pct"], reverse=True)
        print()
        print(f"  {title}:")
        print(f"  {'Strategy':40s} {'Bets':>5s} {'Wins':>5s} {'W%':>6s} "
              f"{'Profit':>9s} {'Yield':>7s} {'MaxDD':>7s} {'AvgEdge':>8s}")
        print("  " + "-" * 97)
        for name, s in items:
            print(f"  {name:40s} {s['bets']:5d} {s['wins']:5d} "
                  f"{s['win_rate']:5.1f}% ${s['profit']:+8.2f} "
                  f"{s['yield_pct']:+6.1f}% {s['max_drawdown_pct']:6.1f}% "
                  f"{s['avg_edge']:7.1f}%")

    # ---- BASELINE STRATEGIES (consensus odds) ----
    _print_section("BASELINE STRATEGIES (consensus odds)",
                   keys=["all_ev_plus", "overs_only", "unders_only",
                          "high_k_all", "high_k_overs", "high_k_unders",
                          "low_k_all", "low_k_overs", "low_k_unders"])

    # By line
    _print_section("BY LINE",
                   keys=[f"line_{l}" for l in [3.5, 4.5, 5.5, 6.5, 7.5]])

    _print_section("BY SIDE + LINE",
                   keys=[f"{s}_{l}" for l in [3.5, 4.5, 5.5, 6.5, 7.5]
                         for s in ["over", "under"]])

    # By edge threshold
    _print_section("BY EDGE THRESHOLD",
                   keys=[f"all_edge_{e}pct" for e in [3, 5, 8]]
                   + [f"unders_edge_{e}pct" for e in [3, 5, 8]]
                   + [f"overs_edge_{e}pct" for e in [3, 5, 8]])

    # ---- BETMGM STRATEGIES (model edge vs soft book) ----
    _print_section("BETMGM: Model Edge vs Soft Book",
                   keys=["BMG_all_ev_plus", "BMG_overs", "BMG_unders"]
                   + [f"BMG_{s}_{l}" for l in [3.5, 4.5, 5.5, 6.5, 7.5]
                      for s in ["over", "under"]]
                   + [f"BMG_all_edge_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG_unders_edge_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG_overs_edge_{e}pct" for e in [3, 5, 8]])

    # ---- SHARP CONFIRMS MODEL (dual confirmation) ----
    _print_section("BMG + SHARP CONFIRMS (model +EV AND sharp agrees on direction)",
                   keys=["BMG+sharp_all", "BMG+sharp_overs", "BMG+sharp_unders"]
                   + [f"BMG+sharp_{s}_{l}" for l in [3.5, 4.5, 5.5, 6.5, 7.5]
                      for s in ["over", "under"]]
                   + [f"BMG+sharp_all_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG+sharp_unders_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG+sharp_overs_{e}pct" for e in [3, 5, 8]]
                   + ["BMG+sharp_highK_unders", "BMG+sharp_highK_overs",
                      "BMG+sharp_lowK_unders", "BMG+sharp_lowK_overs"]
                   + [f"BMG+sharp_{n}books{s}"
                      for n in [2, 3] for s in ["", "_unders", "_overs"]])

    # ---- SHARP DISAGREES ----
    _print_section("SHARP DISAGREES (model +EV but sharp leans other direction)",
                   keys=["BMG_sharp_disagrees", "BMG_sharp_disagrees_overs",
                         "BMG_sharp_disagrees_unders"])

    # ---- BLENDED PROB (50/50 model + sharp) ----
    _print_section("BLENDED PROB (50/50 model+sharp, bet on BMG)",
                   prefix_filter="BMG_blend_")

    # ---- BMG LINE COMBOS ----
    _print_section("BMG LINE COMBOS",
                   keys=[f"BMG_{p}_{c}" for c in ["4.5+5.5", "5.5+6.5", "4.5+5.5+6.5", "6.5+7.5"]
                         for p in ["under", "over"]]
                   + [f"BMG+sharp_{p}_{c}" for c in ["4.5+5.5", "5.5+6.5", "4.5+5.5+6.5", "6.5+7.5"]
                      for p in ["under", "over"]]
                   + ["BMG_highK_unders", "BMG_highK_overs", "BMG+sharp_highK_all"])

    # ---- FULL LEADERBOARD ----
    ranked = sorted(strategies.items(),
                    key=lambda x: x[1]["yield_pct"], reverse=True)
    profitable = [
        (n, s) for n, s in ranked
        if s["bets"] >= 10 and s["yield_pct"] > 0
    ]

    print()
    print("  TOP 10 STRATEGIES (min 10 bets, positive yield):")
    print("  " + "-" * 97)
    if profitable:
        for name, s in profitable[:10]:
            soft_tag = " [BMG]" if name.startswith(("BMG", "soft_")) else ""
            print(f"    >>> {name}{soft_tag}: {s['bets']} bets, "
                  f"{s['win_rate']:.1f}% wins, yield {s['yield_pct']:+.1f}%, "
                  f"drawdown {s['max_drawdown_pct']:.1f}%")
    else:
        print("    No strategies with 10+ bets and positive yield found.")

    # ---- SHARP CONFIRMATION VALUE ----
    print()
    print("  SHARP CONFIRMATION VALUE:")
    print("  " + "-" * 97)
    for label, key_model, key_confirm, key_disagree in [
        ("All +EV", "BMG_all_ev_plus", "BMG+sharp_all", "BMG_sharp_disagrees"),
        ("Overs", "BMG_overs", "BMG+sharp_overs", "BMG_sharp_disagrees_overs"),
        ("Unders", "BMG_unders", "BMG+sharp_unders", "BMG_sharp_disagrees_unders"),
    ]:
        parts = [f"  {label:10s}"]
        for lbl, key in [("Model@BMG", key_model),
                          ("Sharp confirms", key_confirm),
                          ("Sharp disagrees", key_disagree)]:
            s = strategies.get(key)
            if s:
                parts.append(f"{lbl}: {s['bets']:4d} bets, yield {s['yield_pct']:+.1f}%")
        if len(parts) > 1:
            print("  |  ".join(parts))

    # ---- MONTHLY PERFORMANCE (for deployment timing) ----
    bets_df = result.get("bets_df")
    if bets_df is not None and len(bets_df) > 0:
        # Quick monthly summary on the best baseline strategy
        best_baseline = strategies.get("all_ev_plus")
        if best_baseline:
            ev_bets = bets_df[bets_df["ev"] > 0].copy()
            if len(ev_bets) > 0:
                ev_bets["month"] = pd.to_datetime(ev_bets["date"]).dt.to_period("M")
                print()
                print("  MONTHLY PERFORMANCE (all +EV baseline):")
                print(f"  {'Month':10s} {'Bets':>5s} {'Wins':>5s} {'W%':>6s}")
                print("  " + "-" * 35)
                for month, mg in ev_bets.groupby("month"):
                    mw = mg["won"].sum()
                    print(f"  {str(month):10s} {len(mg):5d} {int(mw):5d} {mw/len(mg)*100:5.1f}%")

    print()
    print("=" * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_walkforward(
        starting_bankroll=100.0,
        kelly_fraction=0.25,
        min_train_days=60,      # Start testing after ~2 months of season data
        test_window_days=14,    # 2-week test windows
        step_days=14,           # Non-overlapping windows
    )
    print_walkforward(result)
