"""
MLB Pitcher K Prop Backtest — Quarter Kelly on Holdout Sample.

Uses model predictions, Monte Carlo simulations, and actual DraftKings odds
to simulate a betting season on the holdout period.
"""

import logging
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

import mlb_model
import mlb_simulation

logger = logging.getLogger(__name__)


def run_backtest(starting_bankroll=100.0, kelly_fraction=0.25, max_kelly_pct=0.10,
                 min_edge=0.0, min_wager=1.0, bookmaker="draftkings"):
    """Run full backtest on holdout sample.

    Args:
        starting_bankroll: Starting bankroll in dollars
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
        max_kelly_pct: Maximum bet as fraction of bankroll
        min_edge: Minimum edge required to place bet (0.0 = any +EV)
        min_wager: Minimum wager size in dollars
        bookmaker: Which sportsbook odds to use

    Returns dict with full backtest results.
    """
    mlb_model.load_model()

    # Build holdout set
    df = mlb_model._build_feature_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=14)
    test_df = df[df["date"] > cutoff].copy()

    # Model predictions
    bf_pred = test_df["baseline_bf"].values + mlb_model._model_bf.predict(
        test_df[mlb_model.BF_FEATURES].values
    )
    bf_pred = np.clip(bf_pred, 10, 40)
    kbf_pred = test_df["baseline_k_rate"].values + mlb_model._model_kbf.predict(
        test_df[mlb_model.KBF_FEATURES].values
    )
    kbf_pred = np.clip(kbf_pred, 0.0, 0.60)
    test_df["pred_k"] = bf_pred * kbf_pred

    # Run simulations
    logger.info("Running simulations for %d holdout starts...", len(test_df))
    sim_results = []
    for i in range(len(test_df)):
        pid = int(test_df.iloc[i]["pitcher_id"])
        bf_std = mlb_model._pitcher_bf_std.get(pid, bf_pred[i] * 0.18)
        kbf_std = mlb_model._pitcher_kbf_std.get(pid, kbf_pred[i] * 0.20)
        sim = mlb_simulation.simulate_strikeouts(
            bf_pred[i], kbf_pred[i], bf_std=bf_std, kbf_std=kbf_std,
            n_sims=10000, seed=42 + i,
        )
        sim_results.append(sim)

    # Load actual prop odds
    conn = sqlite3.connect("mlb_data.db")
    conn.row_factory = sqlite3.Row
    props_df = pd.read_sql_query(
        "SELECT pitcher_name, game_date, line, over_under, price "
        "FROM mlb_pitcher_props WHERE bookmaker = ?",
        conn, params=(bookmaker,),
    )

    # Also get pitcher names for the holdout pitcher_ids
    pid_to_name = {}
    for _, row in test_df.iterrows():
        pid = int(row["pitcher_id"])
        gd = row["date"].strftime("%Y-%m-%d")
        nr = conn.execute(
            "SELECT pitcher_name FROM mlb_pitcher_game_stats "
            "WHERE pitcher_id=? AND date=? LIMIT 1", (pid, gd)
        ).fetchone()
        if nr:
            pid_to_name[(pid, gd)] = mlb_model._normalize_name(nr["pitcher_name"])
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

    logger.info("Prop lines loaded: %d", len(prop_lines))

    # --- Backtest ---
    bankroll = starting_bankroll
    bets = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        sim = sim_results[i]
        actual_k = int(row["strikeouts"])
        gd = row["date"].strftime("%Y-%m-%d")
        pid = int(row["pitcher_id"])
        pname_norm = pid_to_name.get((pid, gd), "")

        if not pname_norm:
            continue

        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            lk = (pname_norm, gd, line)
            if lk not in prop_lines:
                continue
            pl = prop_lines[lk]

            model_p_over = sim.get(f"P_over_{line}", 0)
            model_p_under = 1.0 - model_p_over

            # Check OVER bet
            over_price = pl.get("over_price")
            if over_price is not None and model_p_over > 0:
                if over_price < 0:
                    implied = abs(over_price) / (abs(over_price) + 100)
                    decimal_odds = 1 + 100 / abs(over_price)
                else:
                    implied = 100 / (over_price + 100)
                    decimal_odds = 1 + over_price / 100

                edge = model_p_over - implied
                ev = model_p_over * (decimal_odds - 1) - (1 - model_p_over)

                if edge > min_edge and ev > 0:
                    kf = (model_p_over * (decimal_odds - 1) - (1 - model_p_over)) / (decimal_odds - 1)
                    kf = max(kf, 0) * kelly_fraction
                    kf = min(kf, max_kelly_pct)

                    wager = round(bankroll * kf, 2)
                    if wager < min_wager:
                        continue

                    won = actual_k > line
                    profit = round(wager * (decimal_odds - 1), 2) if won else -wager
                    bankroll = round(bankroll + profit, 2)

                    bets.append({
                        "date": gd, "pitcher": pname_norm, "line": line,
                        "side": "OVER", "odds": over_price,
                        "model_prob": round(model_p_over, 3),
                        "implied_prob": round(implied, 3),
                        "edge": round(edge, 3),
                        "ev": round(ev, 3),
                        "kelly_pct": round(kf * 100, 2),
                        "wager": wager, "actual_k": actual_k,
                        "won": won, "profit": profit,
                        "bankroll": bankroll,
                    })

            # Check UNDER bet
            under_price = pl.get("under_price")
            if under_price is not None and model_p_under > 0:
                if under_price < 0:
                    implied = abs(under_price) / (abs(under_price) + 100)
                    decimal_odds = 1 + 100 / abs(under_price)
                else:
                    implied = 100 / (under_price + 100)
                    decimal_odds = 1 + under_price / 100

                edge = model_p_under - implied
                ev = model_p_under * (decimal_odds - 1) - (1 - model_p_under)

                if edge > min_edge and ev > 0:
                    kf = (model_p_under * (decimal_odds - 1) - (1 - model_p_under)) / (decimal_odds - 1)
                    kf = max(kf, 0) * kelly_fraction
                    kf = min(kf, max_kelly_pct)

                    wager = round(bankroll * kf, 2)
                    if wager < min_wager:
                        continue

                    won = actual_k <= line
                    profit = round(wager * (decimal_odds - 1), 2) if won else -wager
                    bankroll = round(bankroll + profit, 2)

                    bets.append({
                        "date": gd, "pitcher": pname_norm, "line": line,
                        "side": "UNDER", "odds": under_price,
                        "model_prob": round(model_p_under, 3),
                        "implied_prob": round(implied, 3),
                        "edge": round(edge, 3),
                        "ev": round(ev, 3),
                        "kelly_pct": round(kf * 100, 2),
                        "wager": wager, "actual_k": actual_k,
                        "won": won, "profit": profit,
                        "bankroll": bankroll,
                    })

    bets_df = pd.DataFrame(bets)
    total_profit = bankroll - starting_bankroll

    # Drawdown calculation
    running = [starting_bankroll]
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

    result = {
        "period": f"{test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d}",
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": bankroll,
        "profit": round(total_profit, 2),
        "roi_pct": round(total_profit / starting_bankroll * 100, 1),
        "total_bets": len(bets_df),
        "wins": int(bets_df["won"].sum()) if len(bets_df) > 0 else 0,
        "losses": int((~bets_df["won"]).sum()) if len(bets_df) > 0 else 0,
        "win_rate": round(bets_df["won"].mean() * 100, 1) if len(bets_df) > 0 else 0,
        "total_wagered": round(bets_df["wager"].sum(), 2) if len(bets_df) > 0 else 0,
        "yield_pct": round(total_profit / bets_df["wager"].sum() * 100, 1) if len(bets_df) > 0 and bets_df["wager"].sum() > 0 else 0,
        "avg_wager": round(bets_df["wager"].mean(), 2) if len(bets_df) > 0 else 0,
        "avg_edge": round(bets_df["edge"].mean() * 100, 1) if len(bets_df) > 0 else 0,
        "avg_kelly_pct": round(bets_df["kelly_pct"].mean(), 2) if len(bets_df) > 0 else 0,
        "max_drawdown_pct": round(max_dd * 100, 1),
        "peak_bankroll": round(max(running), 2),
        "trough_bankroll": round(min(running), 2),
        "bets": bets_df,
        "bankroll_curve": running,
    }

    return result


def print_backtest(result):
    """Print formatted backtest results."""
    bets_df = result["bets"]

    print("=" * 70)
    print("  HOLDOUT BACKTEST: Quarter Kelly Criterion")
    print("=" * 70)
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
    print(f"  Avg Kelly %:         {result['avg_kelly_pct']:.2f}%")
    print()
    print(f"  Max Drawdown:        {result['max_drawdown_pct']:.1f}%")
    print(f"  Peak Bankroll:       ${result['peak_bankroll']:.2f}")
    print(f"  Trough Bankroll:     ${result['trough_bankroll']:.2f}")

    if len(bets_df) == 0:
        print("\n  No bets placed.")
        return

    # By side
    print()
    for side in ["OVER", "UNDER"]:
        sb = bets_df[bets_df["side"] == side]
        if len(sb) > 0:
            sw = sb["won"].sum()
            print(f"  {side}S:  {len(sb)} bets, {sw} wins ({sw/len(sb)*100:.1f}%), "
                  f"profit ${sb['profit'].sum():+.2f}, avg edge {sb['edge'].mean()*100:.1f}%")

    # By line
    print()
    print("  By Line:")
    for line in sorted(bets_df["line"].unique()):
        lb = bets_df[bets_df["line"] == line]
        lw = lb["won"].sum()
        print(f"    {line}:  {len(lb)} bets, {lw} wins ({lw/len(lb)*100:.1f}%), "
              f"profit ${lb['profit'].sum():+.2f}, avg edge {lb['edge'].mean()*100:.1f}%")

    # Daily P&L
    print()
    print("  Daily P&L:")
    for dt, dg in bets_df.groupby("date"):
        dw = dg["won"].sum()
        print(f"    {dt}: {len(dg)} bets, {dw}W/{len(dg)-dw}L, "
              f"P&L ${dg['profit'].sum():+.2f}, bankroll ${dg.iloc[-1]['bankroll']:.2f}")

    # Sample bets
    print()
    print("  Sample Bets (first 20):")
    hdr = (f"  {'Date':12s} {'Pitcher':22s} {'Line':>5s} {'Side':>5s} {'Odds':>6s} "
           f"{'Edge':>6s} {'Wager':>7s} {'ActK':>5s} {'W/L':>4s} {'P&L':>8s} {'Bank':>8s}")
    print(hdr)
    print("  " + "-" * len(hdr.strip()))
    for _, b in bets_df.head(20).iterrows():
        wl = "W" if b["won"] else "L"
        print(f"  {b['date']:12s} {b['pitcher']:22s} {b['line']:5.1f} {b['side']:>5s} "
              f"{b['odds']:+6d} {b['edge']*100:5.1f}% ${b['wager']:6.2f} "
              f"{b['actual_k']:5d} {wl:>4s} ${b['profit']:+7.2f} ${b['bankroll']:7.2f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_backtest(starting_bankroll=100.0, kelly_fraction=0.25)
    print_backtest(result)
