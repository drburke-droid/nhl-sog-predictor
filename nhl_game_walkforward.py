"""
Game-Level Walk-Forward Backtest.

Tests moneyline, totals, and puckline bets using expanding-window training.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

import nhl_game_model as gm
import nhl_simulation

logger = logging.getLogger(__name__)


def run_walkforward(starting_bankroll=100.0, min_train_games=400,
                    test_window_days=14, step_days=14):
    """Run game-level walk-forward backtest."""

    logger.info("Building game training data...")
    df = gm.build_game_training_df()
    if df.empty:
        return {"error": "No data"}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    avail = [f for f in gm.GAME_FEATURES if f in df.columns]
    season_start = df["date"].min()
    season_end = df["date"].max()

    # Only test where we have odds
    has_odds = df["home_ml"].notna()
    if has_odds.sum() == 0:
        return {"error": "No odds data"}

    first_odds_date = df.loc[has_odds, "date"].min()
    first_test = max(season_start + timedelta(days=90), first_odds_date)

    logger.info("Season: %s to %s (%d games, %d with odds)",
                season_start.date(), season_end.date(), len(df), has_odds.sum())

    all_bets = []
    windows = []
    current = first_test
    wnum = 0

    while current < season_end:
        wnum += 1
        test_end = min(current + timedelta(days=test_window_days), season_end)

        train = df[df["date"] < current]
        test = df[(df["date"] >= current) & (df["date"] <= test_end)]

        if len(train) < min_train_games or len(test) == 0:
            current += timedelta(days=step_days)
            continue

        logger.info("Window %d: train %d games, test %s-%s (%d games)",
                    wnum, len(train), test["date"].min().date(),
                    test["date"].max().date(), len(test))

        # Train totals model
        X_train = train[avail].values
        totals_mdl = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=10, random_state=42, verbosity=0,
        )
        totals_mdl.fit(X_train, train["total_goals"].values)

        # Train win model
        win_mdl = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=10, random_state=42, verbosity=0,
            eval_metric="logloss",
        )
        win_mdl.fit(X_train, train["home_win"].values)

        # Predict test games
        X_test = test[avail].values
        pred_total = totals_mdl.predict(X_test)
        pred_home_wp = win_mdl.predict_proba(X_test)[:, 1]

        for i in range(len(test)):
            row = test.iloc[i]
            pt = float(pred_total[i])
            hwp = float(pred_home_wp[i])

            # Split total by win prob
            phg = pt * hwp if hwp > 0 else pt / 2
            pag = pt - phg

            sim = gm.simulate_game(phg, pag)

            actual_total = int(row["total_goals"])
            actual_hw = int(row["home_win"])
            actual_diff = int(row["goal_diff"])
            gdate = row["date"].strftime("%Y-%m-%d")
            home = row["home_team"]
            away = row["away_team"]

            # --- Moneyline bets ---
            for team, ml_col, model_p, won in [
                (home, "home_ml", sim["home_win_prob"], actual_hw == 1),
                (away, "away_ml", sim["away_win_prob"], actual_hw == 0),
            ]:
                ml = row.get(ml_col)
                if ml is None or pd.isna(ml):
                    continue
                ml = int(ml)
                imp = gm._american_to_prob(ml)
                edge = model_p - imp
                dec = gm._american_to_decimal(ml)
                ev = model_p * (dec - 1) - (1 - model_p)

                all_bets.append({
                    "window": wnum, "date": gdate,
                    "home": home, "away": away,
                    "market": "ML", "pick": team,
                    "odds": ml, "decimal_odds": round(dec, 4),
                    "model_prob": round(model_p, 4),
                    "implied_prob": round(imp, 4),
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                    "won": won,
                    "pred_total": round(pt, 2),
                    "actual_total": actual_total,
                })

            # --- Totals bets ---
            tl = row.get("game_total_line")
            if tl is not None and not pd.isna(tl):
                over_key = f"over_{tl}"
                under_key = f"under_{tl}"
                if over_key in sim:
                    # We don't have per-game total_over/under_price in training df
                    # Use standard -110 as proxy for backtest
                    for side, prob, actual_won in [
                        ("OVER", sim[over_key], actual_total > tl),
                        ("UNDER", sim[under_key], actual_total <= tl),
                    ]:
                        imp = 0.5238  # -110 implied
                        edge = prob - imp
                        dec = 1.909  # -110 decimal
                        ev = prob * (dec - 1) - (1 - prob)

                        all_bets.append({
                            "window": wnum, "date": gdate,
                            "home": home, "away": away,
                            "market": "TOTAL", "pick": f"{side} {tl}",
                            "odds": -110, "decimal_odds": 1.909,
                            "model_prob": round(prob, 4),
                            "implied_prob": round(imp, 4),
                            "edge": round(edge, 4),
                            "ev": round(ev, 4),
                            "won": actual_won,
                            "pred_total": round(pt, 2),
                            "actual_total": actual_total,
                        })

            # --- Spread bets ---
            sp = row.get("home_spread_point")
            sp_price = row.get("home_spread_price")
            if sp is not None and not pd.isna(sp) and sp_price is not None and not pd.isna(sp_price):
                sp_price = int(sp_price)
                if sp == -1.5:
                    model_p = sim["home_cover_minus_1_5"]
                    won = actual_diff >= 2
                else:
                    model_p = sim["away_cover_plus_1_5"]
                    won = actual_diff <= 1

                imp = gm._american_to_prob(sp_price)
                dec = gm._american_to_decimal(sp_price)
                edge = model_p - imp
                ev = model_p * (dec - 1) - (1 - model_p)

                all_bets.append({
                    "window": wnum, "date": gdate,
                    "home": home, "away": away,
                    "market": "SPREAD", "pick": f"{home} {sp}",
                    "odds": sp_price, "decimal_odds": round(dec, 4),
                    "model_prob": round(model_p, 4),
                    "implied_prob": round(imp, 4),
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                    "won": won,
                    "pred_total": round(pt, 2),
                    "actual_total": actual_total,
                })

        windows.append({
            "window": wnum,
            "train_size": len(train),
            "test_start": test["date"].min().strftime("%Y-%m-%d"),
            "test_end": test["date"].max().strftime("%Y-%m-%d"),
            "test_games": len(test),
        })
        current += timedelta(days=step_days)

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
    if bets_df.empty:
        return {"error": "No bets", "windows": windows}

    logger.info("Total game bets: %d", len(bets_df))

    # Evaluate strategies
    strategies = _evaluate(bets_df, starting_bankroll)

    return {
        "season": f"{season_start.date()} to {season_end.date()}",
        "total_games": len(df),
        "windows": windows,
        "total_bets": len(bets_df),
        "strategies": strategies,
        "bets_df": bets_df,
    }


def _sim_strategy(bets, bankroll, kelly_frac=0.25, max_pct=0.10, min_wager=1.0):
    """Simulate Kelly-sized flat strategy."""
    bets = bets.sort_values("date")
    br = bankroll
    results = []
    for _, b in bets.iterrows():
        if b["ev"] <= 0:
            continue
        dec = b["decimal_odds"]
        kf = min(max(b["ev"] / (dec - 1), 0) * kelly_frac, max_pct)
        wager = br * kf
        if wager < min_wager:
            continue
        profit = wager * (dec - 1) if b["won"] else -wager
        br += profit
        results.append({"wager": wager, "won": b["won"], "profit": profit, "br": br})

    if not results:
        return None
    rdf = pd.DataFrame(results)
    wagered = rdf["wager"].sum()
    profit = br - bankroll
    return {
        "bets": len(rdf),
        "wins": int(rdf["won"].sum()),
        "win_rate": round(rdf["won"].mean() * 100, 1),
        "profit": round(profit, 2),
        "wagered": round(wagered, 2),
        "yield_pct": round(profit / max(wagered, 1) * 100, 1),
        "ending_bankroll": round(br, 2),
    }


def _evaluate(bets_df, bankroll):
    """Evaluate game betting strategies."""
    strats = {}

    def _run(name, mask, **kwargs):
        sub = bets_df[mask]
        if len(sub) < 5:
            return
        r = _sim_strategy(sub, bankroll, **kwargs)
        if r:
            strats[name] = r

    ev_plus = bets_df["ev"] > 0

    # By market
    _run("all_ev_plus", ev_plus)
    _run("moneyline", ev_plus & (bets_df["market"] == "ML"))
    _run("totals", ev_plus & (bets_df["market"] == "TOTAL"))
    _run("spread", ev_plus & (bets_df["market"] == "SPREAD"))

    # Totals sides
    _run("overs", ev_plus & bets_df["pick"].str.contains("OVER", na=False))
    _run("unders", ev_plus & bets_df["pick"].str.contains("UNDER", na=False))

    # ML by side
    _run("ML_home", ev_plus & (bets_df["market"] == "ML") & (bets_df["pick"] == bets_df["home"]))
    _run("ML_away", ev_plus & (bets_df["market"] == "ML") & (bets_df["pick"] == bets_df["away"]))

    # ML underdogs (positive odds)
    _run("ML_underdogs", ev_plus & (bets_df["market"] == "ML") & (bets_df["odds"] > 0))
    _run("ML_favorites", ev_plus & (bets_df["market"] == "ML") & (bets_df["odds"] < 0))

    # Edge thresholds
    for pct in [3, 5, 8, 10]:
        t = pct / 100
        _run(f"all_edge_{pct}pct", ev_plus & (bets_df["edge"] >= t))
        _run(f"ML_edge_{pct}pct", ev_plus & (bets_df["market"] == "ML") & (bets_df["edge"] >= t))
        _run(f"totals_edge_{pct}pct", ev_plus & (bets_df["market"] == "TOTAL") & (bets_df["edge"] >= t))

    # Totals when model disagrees strongly with line
    pred_diff = bets_df["pred_total"] - 5.5  # rough center
    _run("over_when_model_high",
         ev_plus & bets_df["pick"].str.contains("OVER", na=False) & (bets_df["pred_total"] >= 6.5))
    _run("under_when_model_low",
         ev_plus & bets_df["pick"].str.contains("UNDER", na=False) & (bets_df["pred_total"] <= 5.0))

    return strats


def print_results(result):
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("=" * 90)
    print("  NHL GAME-LEVEL WALK-FORWARD BACKTEST")
    print("=" * 90)
    print(f"  Season: {result['season']}")
    print(f"  Games: {result['total_games']}, Windows: {len(result['windows'])}")
    print(f"  Total bets: {result['total_bets']}")

    strats = result["strategies"]
    ranked = sorted(strats.items(), key=lambda x: x[1]["yield_pct"], reverse=True)

    print(f"\n  {'Strategy':35s} {'Bets':>6s} {'Wins':>5s} {'W%':>6s} "
          f"{'Profit':>9s} {'Yield':>8s}")
    print("  " + "-" * 75)
    for name, s in ranked:
        if s["bets"] >= 5:
            print(f"  {name:35s} {s['bets']:6d} {s['wins']:5d} "
                  f"{s['win_rate']:5.1f}% ${s['profit']:+8.2f} "
                  f"{s['yield_pct']:+7.1f}%")

    # Top profitable
    profitable = [(n, s) for n, s in ranked if s["bets"] >= 10 and s["yield_pct"] > 0]
    print(f"\n  TOP PROFITABLE (min 10 bets):")
    print("  " + "-" * 75)
    if profitable:
        for n, s in profitable[:8]:
            print(f"    >>> {n}: {s['bets']} bets, {s['win_rate']:.1f}% wins, "
                  f"yield {s['yield_pct']:+.1f}%, profit ${s['profit']:+.2f}")
    else:
        print("    No strategies with 10+ bets and positive yield.")

    print("=" * 90)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_walkforward()
    print_results(result)
