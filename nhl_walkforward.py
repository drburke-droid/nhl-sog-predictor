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


def _load_per_book_props():
    """Load per-bookmaker props for sharp-vs-soft analysis.

    Returns dict of (initial, last_name, game_date, line, bookmaker) ->
        {over_price, under_price}
    """
    conn = nhl_odds_collector.get_db()
    nhl_odds_collector.create_odds_tables(conn)
    rows = conn.execute(
        "SELECT player_name, game_date, bookmaker, line, over_under, price "
        "FROM nhl_player_props"
    ).fetchall()
    conn.close()

    result = {}
    for r in rows:
        ini, last = _name_match_key(r["player_name"])
        key = (ini, last, r["game_date"], r["line"], r["bookmaker"])
        if key not in result:
            result[key] = {}
        if r["over_under"] == "Over":
            result[key]["over_price"] = int(r["price"])
        else:
            result[key]["under_price"] = int(r["price"])

    return result


def _compute_sharp_consensus(per_book_props, ini, last, gd, line):
    """Compute vig-free implied probability from sharp books.

    Returns dict with sharp_prob_over, sharp_prob_under, n_sharp_books,
    or None if no sharp books have this line.
    """
    sharp_probs = []
    for book in nhl_odds_collector.SHARP_BOOKS:
        bk = (ini, last, gd, line, book)
        bp = per_book_props.get(bk)
        if bp is None:
            continue
        ov = bp.get("over_price")
        un = bp.get("under_price")
        if ov is None or un is None:
            continue
        imp_ov = nhl_simulation.american_to_implied_prob(ov)
        imp_un = nhl_simulation.american_to_implied_prob(un)
        total = imp_ov + imp_un
        if total > 0:
            sharp_probs.append(imp_ov / total)

    if not sharp_probs:
        return None

    avg_over = sum(sharp_probs) / len(sharp_probs)
    return {
        "sharp_prob_over": avg_over,
        "sharp_prob_under": 1.0 - avg_over,
        "n_sharp_books": len(sharp_probs),
    }


def _get_soft_book_prices(per_book_props, ini, last, gd, line):
    """Get BetMGM (soft book / PlayAlberta proxy) prices for a line.

    Returns dict with soft_over_price, soft_under_price, or None.
    """
    for book in nhl_odds_collector.SOFT_BOOKS:
        bk = (ini, last, gd, line, book)
        bp = per_book_props.get(bk)
        if bp and bp.get("over_price") is not None:
            pa_over, pa_under = nhl_odds_collector.betmgm_to_playalberta(
                bp.get("over_price"), bp.get("under_price")
            )
            return {
                "soft_over_price": bp.get("over_price"),
                "soft_under_price": bp.get("under_price"),
                "pa_over_est": pa_over,
                "pa_under_est": pa_under,
            }
    return None


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

    # Per-bookmaker props for sharp-vs-soft analysis
    per_book_props = _load_per_book_props()
    logger.info("Per-book props loaded: %d entries", len(per_book_props))

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
                ini_k, last_k = pname_key

                for line in nhl_simulation.PROP_LINES:
                    lk = (ini_k, last_k, gd, line)
                    if lk not in prop_lines:
                        continue
                    pl = prop_lines[lk]

                    model_p_over = sim.get(f"P_over_{line}", 0)
                    model_p_under = 1.0 - model_p_over

                    # Sharp-vs-soft signals for this player-game-line
                    sharp = _compute_sharp_consensus(
                        per_book_props, ini_k, last_k, gd, line
                    )
                    soft = _get_soft_book_prices(
                        per_book_props, ini_k, last_k, gd, line
                    )

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

                        # Sharp book signal
                        sharp_prob = None
                        sharp_edge = None
                        n_sharp = 0
                        if sharp:
                            sp = (sharp["sharp_prob_over"] if side == "OVER"
                                  else sharp["sharp_prob_under"])
                            sharp_prob = round(sp, 4)
                            sharp_edge = round(sp - implied, 4)
                            n_sharp = sharp["n_sharp_books"]

                        # Soft book (BetMGM / PlayAlberta) signal
                        soft_price = None
                        soft_implied = None
                        soft_edge = None
                        pa_price = None
                        pa_implied = None
                        pa_edge = None
                        sharp_vs_soft_edge = None
                        has_soft = False
                        if soft:
                            has_soft = True
                            sp_key = ("soft_over_price" if side == "OVER"
                                      else "soft_under_price")
                            pa_key = ("pa_over_est" if side == "OVER"
                                      else "pa_under_est")
                            soft_price = soft.get(sp_key)
                            pa_price = soft.get(pa_key)
                            if soft_price is not None:
                                soft_implied = nhl_simulation.american_to_implied_prob(soft_price)
                                soft_edge = round(model_p - soft_implied, 4)
                            if pa_price is not None:
                                pa_implied = nhl_simulation.american_to_implied_prob(pa_price)
                                pa_edge = round(model_p - pa_implied, 4)
                            if sharp_prob is not None and soft_implied is not None:
                                sharp_vs_soft_edge = round(
                                    sharp_prob - soft_implied, 4
                                )

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
                            # Sharp book consensus
                            "sharp_prob": sharp_prob,
                            "sharp_edge": sharp_edge,
                            "n_sharp_books": n_sharp,
                            # Sharp agrees with model on direction?
                            "sharp_agrees": (
                                sharp_prob is not None
                                and (
                                    (side == "OVER" and sharp_prob > 0.5)
                                    or (side == "UNDER" and sharp_prob < 0.5)
                                )
                            ),
                            # Blended prob: 50/50 model + sharp
                            "blended_prob": (
                                round(0.5 * model_p + 0.5 * sharp_prob, 4)
                                if sharp_prob is not None else None
                            ),
                            # Soft book (BetMGM) prices
                            "has_soft": has_soft,
                            "soft_price": soft_price,
                            "soft_implied": (round(soft_implied, 4)
                                             if soft_implied else None),
                            "soft_edge": soft_edge,
                            # PlayAlberta estimated prices
                            "pa_price": pa_price,
                            "pa_implied": (round(pa_implied, 4)
                                           if pa_implied else None),
                            "pa_edge": pa_edge,
                            # Sharp-vs-soft divergence
                            "sharp_vs_soft_edge": sharp_vs_soft_edge,
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
                        max_kelly_pct, min_wager, min_edge=0.0,
                        use_soft_odds=False, use_sharp_prob=False,
                        use_blended_prob=False):
    """Simulate P&L for a filtered set of bets using Kelly sizing.

    If use_soft_odds=True, uses BetMGM/PlayAlberta odds for payouts
    (simulating actually betting on the soft book).
    If use_sharp_prob=True, uses sharp book consensus probability
    instead of model probability for edge/EV calculation.
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
            dec_odds = nhl_simulation.american_to_decimal(bet["soft_price"])
            imp = nhl_simulation.american_to_implied_prob(bet["soft_price"])
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

    # =================================================================
    # SHARP-VS-SOFT META STRATEGIES
    # =================================================================
    # BetMGM vig is ~7.6% (3.8% per side), so pure sharp-vs-soft arb
    # never works. Instead, we use sharp agreement as a CONFIDENCE
    # FILTER for model-driven bets targeting the soft book.

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

    # -- SHARP AGREES: model +EV AND sharp books confirm direction --
    # This is the key filter: when both model AND sharp books agree,
    # the bet has dual confirmation. Use model prob for sizing.
    sharp_confirm = soft_ev & has_both & sharp_agrees
    _run("BMG+sharp_all", sharp_confirm, use_soft_odds=True)
    _run("BMG+sharp_overs",
         sharp_confirm & (bets_df["side"] == "OVER"), use_soft_odds=True)
    _run("BMG+sharp_unders",
         sharp_confirm & (bets_df["side"] == "UNDER"), use_soft_odds=True)

    for line in [1.5, 2.5, 3.5, 4.5]:
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

    # Position combos with sharp confirmation
    _run("BMG+sharp_fwd_unders",
         sharp_confirm & (bets_df["position_group"] == "F")
         & (bets_df["side"] == "UNDER"), use_soft_odds=True)
    _run("BMG+sharp_def_unders",
         sharp_confirm & (bets_df["position_group"] == "D")
         & (bets_df["side"] == "UNDER"), use_soft_odds=True)
    _run("BMG+sharp_fwd_overs",
         sharp_confirm & (bets_df["position_group"] == "F")
         & (bets_df["side"] == "OVER"), use_soft_odds=True)

    # N sharp books filter with confirmation
    for n_min in [2, 3]:
        n_mask = sharp_confirm & (bets_df["n_sharp_books"] >= n_min)
        _run(f"BMG+sharp_{n_min}books",
             n_mask, use_soft_odds=True)
        _run(f"BMG+sharp_{n_min}books_unders",
             n_mask & (bets_df["side"] == "UNDER"), use_soft_odds=True)
        _run(f"BMG+sharp_{n_min}books_overs",
             n_mask & (bets_df["side"] == "OVER"), use_soft_odds=True)

    # -- SHARP DISAGREES: model says +EV but sharp books lean other way --
    # When sharp books disagree, the bet is riskier. Track separately.
    sharp_disagree = soft_ev & has_both & (~sharp_agrees)
    _run("BMG_sharp_disagrees", sharp_disagree, use_soft_odds=True)
    _run("BMG_sharp_disagrees_unders",
         sharp_disagree & (bets_df["side"] == "UNDER"), use_soft_odds=True)
    _run("BMG_sharp_disagrees_overs",
         sharp_disagree & (bets_df["side"] == "OVER"), use_soft_odds=True)

    # -- BLENDED PROB: 50/50 model + sharp for sizing --
    # Uses the averaged probability for edge calculation and Kelly.
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

    for line in [1.5, 2.5, 3.5]:
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

    # -- Best strategies on BMG specifically --
    for lines in [(3.5,), (3.5, 4.5), (2.5, 3.5)]:
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

    # -- High-volume shooters at soft book --
    _run("BMG_highvol_unders",
         soft_ev & (bets_df["side"] == "UNDER") & (bets_df["baseline_sog"] >= 2.5),
         use_soft_odds=True)
    _run("BMG+sharp_highvol_unders",
         sharp_confirm & (bets_df["side"] == "UNDER") & (bets_df["baseline_sog"] >= 2.5),
         use_soft_odds=True)

    return strategies


def print_walkforward(result):
    """Print formatted walk-forward results."""
    if "error" in result:
        print(f"Walk-forward error: {result['error']}")
        if "windows" in result:
            print(f"  Windows completed: {len(result['windows'])}")
        return

    print("=" * 100)
    print("  NHL SOG WALK-FORWARD BACKTEST: Sharp-vs-Soft Meta Analysis")
    print("=" * 100)
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

    # ---- BASELINE STRATEGIES (any book odds) ----
    _print_section("BASELINE STRATEGIES (consensus odds)",
                   keys=["all_ev_plus", "overs_only", "unders_only",
                          "forwards_only", "defense_only",
                          "fwd_unders", "def_unders",
                          "home_players", "away_players"])

    # By line
    _print_section("BY LINE",
                   keys=[f"line_{l}" for l in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]])

    _print_section("BY SIDE + LINE",
                   keys=[f"{s}_{l}" for l in [1.5, 2.5, 3.5, 4.5]
                         for s in ["over", "under"]])

    # ---- BETMGM/PLAYALBERTA STRATEGIES (model edge vs soft book) ----
    _print_section("BETMGM: Model Edge vs Soft Book",
                   keys=["BMG_all_ev_plus", "BMG_overs", "BMG_unders"]
                   + [f"BMG_{s}_{l}" for l in [1.5, 2.5, 3.5, 4.5]
                      for s in ["over", "under"]]
                   + [f"BMG_all_edge_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG_unders_edge_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG_overs_edge_{e}pct" for e in [3, 5, 8]])

    # ---- SHARP CONFIRMS MODEL (dual confirmation) ----
    _print_section("BMG + SHARP CONFIRMS (model +EV AND sharp agrees on direction)",
                   keys=["BMG+sharp_all", "BMG+sharp_overs", "BMG+sharp_unders"]
                   + [f"BMG+sharp_{s}_{l}" for l in [1.5, 2.5, 3.5, 4.5]
                      for s in ["over", "under"]]
                   + [f"BMG+sharp_all_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG+sharp_unders_{e}pct" for e in [3, 5, 8]]
                   + [f"BMG+sharp_overs_{e}pct" for e in [3, 5, 8]]
                   + ["BMG+sharp_fwd_unders", "BMG+sharp_def_unders",
                      "BMG+sharp_fwd_overs"]
                   + [f"BMG+sharp_{n}books{s}"
                      for n in [2, 3] for s in ["", "_unders", "_overs"]])

    # ---- SHARP DISAGREES (model says bet but sharp says other way) ----
    _print_section("SHARP DISAGREES (model +EV but sharp leans other direction)",
                   keys=["BMG_sharp_disagrees", "BMG_sharp_disagrees_overs",
                         "BMG_sharp_disagrees_unders"])

    # ---- BLENDED PROB (50/50 model + sharp) ----
    _print_section("BLENDED PROB (50/50 model+sharp, bet on BMG)",
                   prefix_filter="BMG_blend_")

    # ---- BMG LINE COMBOS ----
    _print_section("BMG LINE COMBOS",
                   keys=[f"BMG_{p}_{c}" for c in ["3.5", "3.5+4.5", "2.5+3.5"]
                         for p in ["under", "over"]]
                   + [f"BMG+sharp_{p}_{c}" for c in ["3.5", "3.5+4.5", "2.5+3.5"]
                      for p in ["under", "over"]]
                   + ["BMG_highvol_unders", "BMG+sharp_highvol_unders"])

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

    print()
    print("=" * 100)


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
