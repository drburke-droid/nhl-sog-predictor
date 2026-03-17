"""
MLB Walk-Forward Backtest — Rolling window across full 2024 season.

Trains on expanding window, tests on next 2 weeks, slides forward.
No data leakage — model never sees future data in any window.
Also tracks cumulative performance by month to show when the model
becomes trustworthy (informs 2025 deployment timing).
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


def run_walkforward(starting_bankroll=100.0, kelly_fraction=0.25, max_kelly_pct=0.10,
                    min_edge=0.0, min_wager=1.0, bookmaker="draftkings",
                    min_train_days=60, test_window_days=14, step_days=14):
    """Run walk-forward backtest across entire 2024 season.

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

    # Load prop odds
    conn = sqlite3.connect("mlb_data.db")
    conn.row_factory = sqlite3.Row
    props_df = pd.read_sql_query(
        "SELECT pitcher_name, game_date, line, over_under, price "
        "FROM mlb_pitcher_props WHERE bookmaker = ?",
        conn, params=(bookmaker,),
    )

    # Pitcher name lookup
    name_rows = conn.execute(
        "SELECT pitcher_id, date, pitcher_name FROM mlb_pitcher_game_stats"
    ).fetchall()
    pid_date_to_name = {}
    for r in name_rows:
        pid_date_to_name[(r["pitcher_id"], r["date"])] = mlb_model._normalize_name(r["pitcher_name"])
    conn.close()

    # Build prop lookup
    prop_lines = {}
    for _, r in props_df.iterrows():
        key = (mlb_model._normalize_name(r["pitcher_name"]), r["game_date"], r["line"])
        if key not in prop_lines:
            prop_lines[key] = {}
        if r["over_under"] == "Over":
            prop_lines[key]["over_price"] = int(r["price"])
        else:
            prop_lines[key]["under_price"] = int(r["price"])

    logger.info("Prop lines loaded: %d", len(prop_lines))

    # Walk-forward loop
    bankroll = starting_bankroll
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
        logger.info("Window %d: train %s-%s (%d), test %s-%s (%d), bankroll $%.2f",
                     window_num,
                     train_df["date"].min().date(), train_df["date"].max().date(), len(train_df),
                     test_df["date"].min().date(), test_df["date"].max().date(), len(test_df),
                     bankroll)

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

        # Place bets
        window_bets = 0
        window_wins = 0
        window_profit = 0.0
        bankroll_at_window_start = bankroll

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
                lk = (pname_norm, gd, line)
                if lk not in prop_lines:
                    continue
                pl = prop_lines[lk]

                model_p_over = sim.get(f"P_over_{line}", 0)
                model_p_under = 1.0 - model_p_over

                for side, price_key, model_p, won_fn in [
                    ("OVER", "over_price", model_p_over, lambda k, l: k > l),
                    ("UNDER", "under_price", model_p_under, lambda k, l: k <= l),
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

                    if edge > min_edge and ev > 0:
                        kf = (model_p * (decimal_odds - 1) - (1 - model_p)) / (decimal_odds - 1)
                        kf = max(kf, 0) * kelly_fraction
                        kf = min(kf, max_kelly_pct)

                        wager = round(bankroll * kf, 2)
                        if wager < min_wager:
                            continue

                        won = won_fn(ak, line)
                        profit = round(wager * (decimal_odds - 1), 2) if won else -wager
                        bankroll = round(bankroll + profit, 2)

                        window_bets += 1
                        if won:
                            window_wins += 1
                        window_profit += profit

                        all_bets.append({
                            "window": window_num,
                            "date": gd, "pitcher": pname_norm, "line": line,
                            "side": side, "odds": price,
                            "model_prob": round(model_p, 3),
                            "implied_prob": round(implied, 3),
                            "edge": round(edge, 3),
                            "ev": round(ev, 3),
                            "kelly_pct": round(kf * 100, 2),
                            "wager": wager, "actual_k": ak,
                            "won": won, "profit": round(profit, 2),
                            "bankroll": bankroll,
                            "train_size": len(train_df),
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
            "bets": window_bets,
            "wins": window_wins,
            "win_rate": round(window_wins / max(window_bets, 1) * 100, 1),
            "profit": round(window_profit, 2),
            "bankroll": round(bankroll, 2),
            "window_roi": round(window_profit / max(bankroll_at_window_start, 1) * 100, 1),
        })

        current_test_start += timedelta(days=step_days)

    # Compile results
    bets_df = pd.DataFrame(all_bets)
    total_profit = bankroll - starting_bankroll

    # Drawdown
    running = [starting_bankroll]
    if len(bets_df) > 0:
        for p in bets_df["profit"].values:
            running.append(running[-1] + p)
    peak = starting_bankroll
    max_dd = 0
    for v in running:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Monthly breakdown
    monthly = {}
    if len(bets_df) > 0:
        bets_df["month"] = pd.to_datetime(bets_df["date"]).dt.to_period("M")
        for month, mg in bets_df.groupby("month"):
            mw = mg["won"].sum()
            monthly[str(month)] = {
                "bets": len(mg),
                "wins": int(mw),
                "win_rate": round(mw / len(mg) * 100, 1),
                "profit": round(mg["profit"].sum(), 2),
                "wagered": round(mg["wager"].sum(), 2),
                "yield_pct": round(mg["profit"].sum() / max(mg["wager"].sum(), 1) * 100, 1),
                "avg_edge": round(mg["edge"].mean() * 100, 1),
                "avg_train_size": int(mg["train_size"].mean()),
            }

    result = {
        "period": f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": bankroll,
        "profit": round(total_profit, 2),
        "roi_pct": round(total_profit / starting_bankroll * 100, 1),
        "total_bets": len(bets_df),
        "wins": int(bets_df["won"].sum()) if len(bets_df) > 0 else 0,
        "losses": len(bets_df) - int(bets_df["won"].sum()) if len(bets_df) > 0 else 0,
        "win_rate": round(bets_df["won"].mean() * 100, 1) if len(bets_df) > 0 else 0,
        "total_wagered": round(bets_df["wager"].sum(), 2) if len(bets_df) > 0 else 0,
        "yield_pct": round(total_profit / max(bets_df["wager"].sum(), 1) * 100, 1) if len(bets_df) > 0 else 0,
        "avg_wager": round(bets_df["wager"].mean(), 2) if len(bets_df) > 0 else 0,
        "avg_edge": round(bets_df["edge"].mean() * 100, 1) if len(bets_df) > 0 else 0,
        "max_drawdown_pct": round(max_dd * 100, 1),
        "peak_bankroll": round(max(running), 2),
        "trough_bankroll": round(min(running), 2),
        "windows": window_results,
        "monthly": monthly,
        "bets": bets_df,
        "bankroll_curve": running,
    }

    return result


def print_walkforward(result):
    """Print formatted walk-forward results."""
    bets_df = result["bets"]

    print("=" * 80)
    print("  WALK-FORWARD BACKTEST: Quarter Kelly, Full 2024 Season")
    print("=" * 80)
    print(f"  Period:              {result['period']}")
    print(f"  Starting Bankroll:   ${result['starting_bankroll']:.2f}")
    print(f"  Ending Bankroll:     ${result['ending_bankroll']:.2f}")
    print(f"  Profit/Loss:         ${result['profit']:+.2f}")
    print(f"  ROI:                 {result['roi_pct']:+.1f}%")
    print()
    print(f"  Total Bets:          {result['total_bets']}")
    print(f"  Wins:                {result['wins']} ({result['win_rate']:.1f}%)")
    print(f"  Losses:              {result['losses']} ({100-result['win_rate']:.1f}%)")
    print(f"  Total Wagered:       ${result['total_wagered']:.2f}")
    print(f"  Yield:               {result['yield_pct']:+.1f}%")
    print(f"  Avg Wager:           ${result['avg_wager']:.2f}")
    print(f"  Avg Edge:            {result['avg_edge']:.1f}%")
    print()
    print(f"  Max Drawdown:        {result['max_drawdown_pct']:.1f}%")
    print(f"  Peak Bankroll:       ${result['peak_bankroll']:.2f}")
    print(f"  Trough Bankroll:     ${result['trough_bankroll']:.2f}")

    # Windows
    print()
    print("  WALK-FORWARD WINDOWS:")
    print(f"  {'Win':>4s}  {'Train Period':24s} {'Trn':>5s}  {'Test Period':24s} {'Tst':>4s}  "
          f"{'MAE':>5s} {'MktMAE':>6s} {'Edge':>5s}  {'Bets':>4s} {'W%':>5s}  {'P&L':>8s}  {'Bank':>8s}")
    print("  " + "-" * 115)
    for w in result["windows"]:
        print(f"  {w['window']:4d}  {w['train_start']} - {w['train_end']}  {w['train_size']:5d}  "
              f"{w['test_start']} - {w['test_end']}  {w['test_size']:4d}  "
              f"{w['model_mae']:5.3f} {w['market_mae']:6.3f} {w['mae_edge']:+5.3f}  "
              f"{w['bets']:4d} {w['win_rate']:5.1f}  ${w['profit']:+7.2f}  ${w['bankroll']:7.2f}")

    # Monthly performance
    if result["monthly"]:
        print()
        print("  MONTHLY PERFORMANCE (key for 2025 deployment timing):")
        print(f"  {'Month':10s} {'Bets':>5s} {'Wins':>5s} {'W%':>6s} {'Profit':>9s} {'Wagered':>9s} "
              f"{'Yield':>7s} {'AvgEdge':>8s} {'TrainN':>7s}")
        print("  " + "-" * 75)
        cumulative_profit = 0
        for month, m in sorted(result["monthly"].items()):
            cumulative_profit += m["profit"]
            print(f"  {month:10s} {m['bets']:5d} {m['wins']:5d} {m['win_rate']:5.1f}% "
                  f"${m['profit']:+8.2f} ${m['wagered']:8.2f} {m['yield_pct']:+6.1f}% "
                  f"{m['avg_edge']:7.1f}% {m['avg_train_size']:7d}")

        print()
        print("  INTERPRETATION FOR 2025 DEPLOYMENT:")
        print("  Look for months where:")
        print("    - Yield is consistently positive")
        print("    - Win rate is above 53%")
        print("    - Training size is large enough (1000+ starts)")
        print("    - MAE edge vs market is positive")

    # By side
    if len(bets_df) > 0:
        print()
        for side in ["OVER", "UNDER"]:
            sb = bets_df[bets_df["side"] == side]
            if len(sb) > 0:
                sw = sb["won"].sum()
                print(f"  {side}S:  {len(sb)} bets, {sw} wins ({sw/len(sb)*100:.1f}%), "
                      f"profit ${sb['profit'].sum():+.2f}, yield {sb['profit'].sum()/sb['wager'].sum()*100:+.1f}%")

        # By line
        print()
        print("  BY LINE:")
        for line in sorted(bets_df["line"].unique()):
            lb = bets_df[bets_df["line"] == line]
            lw = lb["won"].sum()
            print(f"    {line}:  {len(lb)} bets, {lw} wins ({lw/len(lb)*100:.1f}%), "
                  f"profit ${lb['profit'].sum():+.2f}, yield {lb['profit'].sum()/lb['wager'].sum()*100:+.1f}%")

    print()
    print("=" * 80)


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
