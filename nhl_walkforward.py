"""
NHL SOG Walk-Forward Backtest — Rolling window with strategy evaluation.

Trains on expanding window, tests on next 2 weeks, slides forward.
No data leakage — model never sees future data in any window.

Matches model predictions with actual sportsbook SOG props, simulates
distributions, and evaluates many betting strategies to find exploitable
patterns (like the MLB finding that unders-3.5 was the profitable niche).

IMPORTANT: sog_prop_line is excluded from model features during walk-forward
to avoid circular reasoning (model can't learn from the line it bets against).
"""

import logging
import sqlite3
from datetime import timedelta
from collections import Counter

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import model as nhl_model
import data_collector
import nhl_odds_collector
import nhl_simulation

logger = logging.getLogger(__name__)

# Features for walk-forward: exclude sog_prop_line to avoid circularity
WF_FEATURE_COLS = [c for c in nhl_model.FEATURE_COLS if c != "sog_prop_line"]


def _normalize_name(name: str) -> str:
    """Normalize player name for matching between model and odds."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.strip().lower()


def _name_match_key(name: str) -> tuple:
    """Extract (first_initial, last_name) for cross-source matching.

    NHL API uses abbreviated names ('A. Laferriere'),
    Odds API uses full names ('Alex Laferriere').
    Matching on first initial + last name gives 99% coverage.
    """
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.strip().lower().replace(".", "")
    parts = name.split()
    if len(parts) >= 2:
        initial = parts[0][0]
        last = " ".join(parts[1:])
        return (initial, last)
    return ("", name)


def _train_window_model(train_df):
    """Train forward + defense XGBoost models on a training window.

    Returns (model_fwd, model_def, var_ratio_map).
    """
    weights = nhl_model._compute_sample_weights(train_df)
    fwd = train_df[train_df["position_group"] == "F"]
    dmen = train_df[train_df["position_group"] == "D"]

    xgb_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
    )

    model_fwd = None
    if len(fwd) >= 50:
        fwd_w = weights[train_df["position_group"] == "F"]
        model_fwd = XGBRegressor(**xgb_params)
        model_fwd.fit(fwd[WF_FEATURE_COLS].values,
                       fwd["sog_residual"].values,
                       sample_weight=fwd_w)

    model_def = None
    if len(dmen) >= 50:
        def_w = weights[train_df["position_group"] == "D"]
        model_def = XGBRegressor(**xgb_params)
        model_def.fit(dmen[WF_FEATURE_COLS].values,
                       dmen["sog_residual"].values,
                       sample_weight=def_w)

    # Player-specific variance/mean ratio from training data
    var_ratio_map = {}
    for pid, grp in train_df.groupby("player_id"):
        shots = grp["shots"].values
        if len(shots) >= 10:
            mean_s = shots.mean()
            if mean_s > 0:
                var_ratio_map[int(pid)] = float(shots.var() / mean_s)

    return model_fwd, model_def, var_ratio_map


def _load_prop_lookup():
    """Load all player SOG props from DB into a fast lookup dict.

    Returns dict of (initial, last_name, game_date, line) -> {over_price, under_price}
    Keyed by first initial + last name to bridge NHL API abbreviated names
    ('A. Laferriere') with Odds API full names ('Alex Laferriere').
    """
    conn = nhl_odds_collector.get_db()
    nhl_odds_collector.create_odds_tables(conn)
    rows = conn.execute(
        "SELECT player_name, game_date, line, over_under, price "
        "FROM nhl_player_props"
    ).fetchall()
    conn.close()

    prop_lines = {}
    for r in rows:
        ini, last = _name_match_key(r["player_name"])
        key = (ini, last, r["game_date"], r["line"])
        if key not in prop_lines:
            prop_lines[key] = {}
        if r["over_under"] == "Over":
            prop_lines[key]["over_price"] = int(r["price"])
        else:
            prop_lines[key]["under_price"] = int(r["price"])

    return prop_lines


def run_walkforward(starting_bankroll=100.0, kelly_fraction=0.25,
                    max_kelly_pct=0.10, min_edge=0.0, min_wager=1.0,
                    min_train_days=60, test_window_days=14, step_days=14):
    """Run walk-forward backtest across all available data with odds.

    Returns dict with full results including per-window metrics,
    all candidate bets, strategy evaluations, and bankroll curves.
    """
    # Build full feature matrix once
    logger.info("Building full feature matrix...")
    df = nhl_model._build_feature_dataframe()
    if df.empty:
        return {"error": "No data"}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    season_start = df["date"].min()
    season_end = df["date"].max()
    logger.info("Season: %s to %s (%d player-games)",
                season_start.date(), season_end.date(), len(df))

    # Load props lookup
    prop_lines = _load_prop_lookup()
    logger.info("Prop lines loaded: %d unique (name, date, line) combos",
                len(prop_lines))

    # Player name map: (player_id, date_str) -> normalized name
    conn = data_collector.get_db()
    name_rows = conn.execute(
        "SELECT player_id, player_name FROM player_game_stats"
    ).fetchall()
    conn.close()
    pid_to_name = {}
    for r in name_rows:
        pid_to_name[int(r["player_id"])] = _normalize_name(r["player_name"])

    # Walk-forward windows
    first_test_start = season_start + timedelta(days=min_train_days)

    # Only test in periods where we have odds data
    odds_dates = set()
    conn2 = nhl_odds_collector.get_db()
    for r in conn2.execute("SELECT DISTINCT game_date FROM nhl_player_props").fetchall():
        odds_dates.add(r["game_date"])
    conn2.close()

    if not odds_dates:
        logger.warning("No odds data found — cannot run backtest")
        return {"error": "No odds data"}

    min_odds_date = pd.Timestamp(min(odds_dates))
    first_test_start = max(first_test_start, min_odds_date)
    logger.info("Odds available from %s, first test window starts %s",
                min_odds_date.date(), first_test_start.date())

    all_bets = []
    window_results = []
    current_test_start = first_test_start
    window_num = 0

    while current_test_start < season_end:
        window_num += 1
        current_test_end = min(
            current_test_start + timedelta(days=test_window_days),
            season_end,
        )

        train_df = df[df["date"] < current_test_start].copy()
        test_df = df[
            (df["date"] >= current_test_start) &
            (df["date"] <= current_test_end)
        ].copy()

        if len(train_df) < 100 or len(test_df) == 0:
            current_test_start += timedelta(days=step_days)
            continue

        logger.info(
            "Window %d: train to %s (%d), test %s-%s (%d players)",
            window_num,
            train_df["date"].max().date(), len(train_df),
            test_df["date"].min().date(),
            test_df["date"].max().date(), len(test_df),
        )

        # Train models for this window
        model_fwd, model_def, var_ratio_map = _train_window_model(train_df)

        # Predict on test set
        test_fwd = test_df[test_df["position_group"] == "F"].copy()
        test_def = test_df[test_df["position_group"] == "D"].copy()

        for subset, mdl, label in [
            (test_fwd, model_fwd, "F"),
            (test_def, model_def, "D"),
        ]:
            if mdl is None or len(subset) == 0:
                continue

            pred_residual = mdl.predict(subset[WF_FEATURE_COLS].values)
            pred_sog = subset["baseline_sog"].values + pred_residual
            pred_sog = np.maximum(pred_sog, 0.0)

            actual_sog = subset["shots"].values
            model_mae = mean_absolute_error(actual_sog, pred_sog)

            # For each prediction, simulate + match with odds + record bets
            for i in range(len(subset)):
                row = subset.iloc[i]
                ps = float(pred_sog[i])
                actual = int(row["shots"])
                pid = int(row["player_id"])
                pname = row.get("player_name", "")
                if not pname:
                    pname = pid_to_name.get(pid, "")
                pname_norm = _normalize_name(str(pname)) if pname else ""
                gd = row["date"].strftime("%Y-%m-%d")
                is_home = int(row["is_home"])
                position = row.get("position", "")
                baseline = float(row["baseline_sog"])
                vr = var_ratio_map.get(pid, 1.0)

                if not pname_norm:
                    continue

                # Simulate SOG distribution
                sim = nhl_simulation.simulate_sog(
                    pred_sog=ps, var_ratio=vr,
                    model_std=0.3, n_sims=10000,
                    seed=42 + window_num * 10000 + i,
                )

                # Match with sportsbook lines using initial + last name
                pname_key = _name_match_key(str(pname)) if pname else ("", "")
                for line in nhl_simulation.PROP_LINES:
                    lk = (pname_key[0], pname_key[1], gd, line)
                    if lk not in prop_lines:
                        continue
                    pl = prop_lines[lk]

                    model_p_over = sim.get(f"P_over_{line}", 0)
                    model_p_under = 1.0 - model_p_over

                    for side, price_key, model_p, won_fn in [
                        ("OVER", "over_price", model_p_over,
                         lambda s, l: s > l),
                        ("UNDER", "under_price", model_p_under,
                         lambda s, l: s <= l),
                    ]:
                        price = pl.get(price_key)
                        if price is None or model_p <= 0.01:
                            continue

                        implied = nhl_simulation.american_to_implied_prob(price)
                        decimal_odds = nhl_simulation.american_to_decimal(price)
                        edge = model_p - implied
                        ev = model_p * (decimal_odds - 1) - (1 - model_p)

                        # Global min edge filter
                        if edge < min_edge:
                            continue

                        all_bets.append({
                            "window": window_num,
                            "date": gd,
                            "player": str(pname),
                            "player_id": pid,
                            "position": position,
                            "position_group": label,
                            "is_home": is_home,
                            "line": line,
                            "side": side,
                            "odds": price,
                            "model_prob": round(model_p, 4),
                            "implied_prob": round(implied, 4),
                            "edge": round(edge, 4),
                            "ev": round(ev, 4),
                            "pred_sog": round(ps, 2),
                            "baseline_sog": round(baseline, 2),
                            "actual_sog": actual,
                            "won": won_fn(actual, line),
                            "var_ratio": round(vr, 3),
                            "decimal_odds": round(decimal_odds, 4),
                            "train_size": len(train_df),
                        })

        # Window summary
        all_test = pd.concat([test_fwd, test_def]) if len(test_fwd) > 0 or len(test_def) > 0 else test_df
        window_results.append({
            "window": window_num,
            "train_end": train_df["date"].max().strftime("%Y-%m-%d"),
            "test_start": test_df["date"].min().strftime("%Y-%m-%d"),
            "test_end": test_df["date"].max().strftime("%Y-%m-%d"),
            "train_size": len(train_df),
            "test_size": len(test_df),
        })

        current_test_start += timedelta(days=step_days)

    # Compile all bets
    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

    if bets_df.empty:
        logger.warning("No bets generated — check odds data coverage")
        return {"error": "No bets", "windows": window_results}

    logger.info("Total candidate bets: %d", len(bets_df))
    logger.info("Matched %d unique players", bets_df["player_id"].nunique())

    # =====================================================================
    # STRATEGY EVALUATION
    # =====================================================================
    strategies = _evaluate_strategies(bets_df, starting_bankroll,
                                      kelly_fraction, max_kelly_pct, min_wager)

    return {
        "season": f"{season_start.date()} to {season_end.date()}",
        "total_player_games": len(df),
        "windows": window_results,
        "total_candidate_bets": len(bets_df),
        "strategies": strategies,
        "bets_df": bets_df,
    }


def _simulate_strategy(bets_df, starting_bankroll, kelly_fraction,
                        max_kelly_pct, min_wager, min_edge=0.0):
    """Simulate P&L for a filtered set of bets using Kelly sizing."""
    # Sort by date for chronological simulation
    bets_df = bets_df.sort_values("date").reset_index(drop=True)

    bankroll = starting_bankroll
    results = []

    for _, bet in bets_df.iterrows():
        if bet["edge"] <= min_edge or bet["ev"] <= 0:
            continue

        # Kelly sizing
        kf = bet["ev"] / (bet["decimal_odds"] - 1)
        kf = max(kf, 0) * kelly_fraction
        kf = min(kf, max_kelly_pct)

        wager = round(bankroll * kf, 2)
        if wager < min_wager:
            continue

        won = bet["won"]
        profit = round(wager * (bet["decimal_odds"] - 1), 2) if won else -wager
        bankroll = round(bankroll + profit, 2)

        results.append({
            "wager": wager, "won": won, "profit": profit,
            "bankroll": bankroll, "edge": bet["edge"],
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

    def _run(name, mask, min_edge=0.0):
        subset = bets_df[mask].copy()
        if len(subset) == 0:
            return
        result = _simulate_strategy(
            subset, starting_bankroll, kelly_fraction,
            max_kelly_pct, min_wager, min_edge,
        )
        if result:
            strategies[name] = result

    ev_plus = bets_df["ev"] > 0

    # --- All +EV bets (baseline) ---
    _run("all_ev_plus", ev_plus)

    # --- By side ---
    _run("overs_only", ev_plus & (bets_df["side"] == "OVER"))
    _run("unders_only", ev_plus & (bets_df["side"] == "UNDER"))

    # --- By line ---
    for line in sorted(bets_df["line"].unique()):
        _run(f"line_{line}", ev_plus & (bets_df["line"] == line))

    # --- By side + line ---
    for line in sorted(bets_df["line"].unique()):
        _run(f"over_{line}", ev_plus & (bets_df["side"] == "OVER") & (bets_df["line"] == line))
        _run(f"under_{line}", ev_plus & (bets_df["side"] == "UNDER") & (bets_df["line"] == line))

    # --- Under combos (the MLB sweet spot was unders at low lines) ---
    for lines in [(2.5,), (3.5,), (2.5, 3.5), (1.5, 2.5), (1.5, 2.5, 3.5)]:
        name = "under_" + "+".join(str(l) for l in lines)
        _run(name, ev_plus & (bets_df["side"] == "UNDER") & bets_df["line"].isin(lines))

    # --- Over combos ---
    for lines in [(2.5,), (3.5,), (2.5, 3.5), (3.5, 4.5)]:
        name = "over_" + "+".join(str(l) for l in lines)
        _run(name, ev_plus & (bets_df["side"] == "OVER") & bets_df["line"].isin(lines))

    # --- By edge threshold ---
    for edge_pct in [3, 5, 8]:
        edge_thresh = edge_pct / 100.0
        _run(f"all_edge_{edge_pct}pct", ev_plus, min_edge=edge_thresh)
        _run(f"unders_edge_{edge_pct}pct",
             ev_plus & (bets_df["side"] == "UNDER"), min_edge=edge_thresh)
        _run(f"overs_edge_{edge_pct}pct",
             ev_plus & (bets_df["side"] == "OVER"), min_edge=edge_thresh)

    # --- By position group ---
    _run("forwards_only", ev_plus & (bets_df["position_group"] == "F"))
    _run("defense_only", ev_plus & (bets_df["position_group"] == "D"))

    # --- Forwards unders ---
    _run("fwd_unders", ev_plus & (bets_df["position_group"] == "F") & (bets_df["side"] == "UNDER"))
    _run("def_unders", ev_plus & (bets_df["position_group"] == "D") & (bets_df["side"] == "UNDER"))

    # --- Home/away ---
    _run("home_players", ev_plus & (bets_df["is_home"] == 1))
    _run("away_players", ev_plus & (bets_df["is_home"] == 0))

    # --- High-volume shooters (baseline >= 2.5) ---
    _run("high_vol_unders",
         ev_plus & (bets_df["side"] == "UNDER") & (bets_df["baseline_sog"] >= 2.5))

    # --- Targeted combos ---
    _run("under_2.5+3.5_edge5",
         ev_plus & (bets_df["side"] == "UNDER") & bets_df["line"].isin([2.5, 3.5]),
         min_edge=0.05)
    _run("under_3.5_edge5",
         ev_plus & (bets_df["side"] == "UNDER") & (bets_df["line"] == 3.5),
         min_edge=0.05)
    _run("under_2.5_fwd",
         ev_plus & (bets_df["side"] == "UNDER") & (bets_df["line"] == 2.5) & (bets_df["position_group"] == "F"))

    return strategies


def print_walkforward(result):
    """Print formatted walk-forward results."""
    if "error" in result:
        print(f"Walk-forward error: {result['error']}")
        if "windows" in result:
            print(f"  Windows completed: {len(result['windows'])}")
        return

    print("=" * 90)
    print("  NHL SOG WALK-FORWARD BACKTEST: Quarter Kelly, Strategy Discovery")
    print("=" * 90)
    print(f"  Season:              {result['season']}")
    print(f"  Player-games:        {result['total_player_games']}")
    print(f"  Walk-forward windows: {len(result['windows'])}")
    print(f"  Candidate bets:      {result['total_candidate_bets']}")

    # Windows
    print()
    print("  WALK-FORWARD WINDOWS:")
    print(f"  {'Win':>4s}  {'Train End':>12s} {'Trn':>6s}  {'Test Period':>24s} {'Tst':>5s}")
    print("  " + "-" * 60)
    for w in result["windows"]:
        print(f"  {w['window']:4d}  {w['train_end']:>12s} {w['train_size']:6d}  "
              f"{w['test_start']} - {w['test_end']}  {w['test_size']:5d}")

    # Strategy leaderboard
    strategies = result.get("strategies", {})
    if strategies:
        print()
        print("  STRATEGY LEADERBOARD (sorted by yield):")
        print(f"  {'Strategy':35s} {'Bets':>5s} {'Wins':>5s} {'W%':>6s} "
              f"{'Profit':>9s} {'Yield':>7s} {'MaxDD':>7s} {'AvgEdge':>8s}")
        print("  " + "-" * 92)

        ranked = sorted(strategies.items(), key=lambda x: x[1]["yield_pct"], reverse=True)
        for name, s in ranked:
            if s["bets"] < 5:
                continue
            print(f"  {name:35s} {s['bets']:5d} {s['wins']:5d} "
                  f"{s['win_rate']:5.1f}% ${s['profit']:+8.2f} "
                  f"{s['yield_pct']:+6.1f}% {s['max_drawdown_pct']:6.1f}% "
                  f"{s['avg_edge']:7.1f}%")

        # Highlight the best strategies
        print()
        print("  TOP STRATEGIES (min 10 bets, positive yield):")
        profitable = [
            (n, s) for n, s in ranked
            if s["bets"] >= 10 and s["yield_pct"] > 0
        ]
        if profitable:
            for name, s in profitable[:5]:
                print(f"    >>> {name}: {s['bets']} bets, {s['win_rate']:.1f}% wins, "
                      f"yield {s['yield_pct']:+.1f}%, "
                      f"drawdown {s['max_drawdown_pct']:.1f}%")
        else:
            print("    No strategies with 10+ bets and positive yield found.")

        # Overs vs Unders comparison
        print()
        print("  OVERS vs UNDERS:")
        for key in ["overs_only", "unders_only"]:
            if key in strategies:
                s = strategies[key]
                print(f"    {key:20s}: {s['bets']} bets, {s['win_rate']:.1f}% W, "
                      f"yield {s['yield_pct']:+.1f}%, profit ${s['profit']:+.2f}")

        # By line breakdown
        print()
        print("  BY LINE (all +EV bets):")
        for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            key = f"line_{line}"
            if key in strategies:
                s = strategies[key]
                print(f"    {line}: {s['bets']} bets, {s['win_rate']:.1f}% W, "
                      f"yield {s['yield_pct']:+.1f}%, profit ${s['profit']:+.2f}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result = run_walkforward(
        starting_bankroll=100.0,
        kelly_fraction=0.25,
        min_edge=0.03,
        min_train_days=60,
        test_window_days=14,
        step_days=14,
    )
    print_walkforward(result)
