"""
V2 Walk-Forward Backtest — compares V2 model (MoneyPuck + clusters) against
V1, with full sharp-vs-soft meta analysis for profitability assessment.

Same framework as nhl_walkforward.py but uses model_v2 features/training.
Runs both V1 and V2 side-by-side on the same windows for fair comparison.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import model as nhl_model
import model_v2
import data_collector
import nhl_odds_collector
import nhl_simulation
import nhl_walkforward as wf_v1  # Reuse V1 helpers

logger = logging.getLogger(__name__)

# V2 features excluding sog_prop_line (not in V2 anyway) and sharp_consensus_prob
WF_V2_FEATURE_COLS = [c for c in model_v2.FEATURE_COLS]


def _train_v2_window(train_df):
    """Train V2 forward + defense models on a training window."""
    available = [f for f in WF_V2_FEATURE_COLS if f in train_df.columns]
    weights = nhl_model._compute_sample_weights(train_df)

    fwd = train_df[train_df["position_group"] == "F"]
    dmen = train_df[train_df["position_group"] == "D"]

    xgb_params = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.3, reg_lambda=1.5, random_state=42, verbosity=0,
    )

    model_fwd = None
    if len(fwd) >= 50:
        fwd_w = weights[train_df["position_group"] == "F"]
        model_fwd = XGBRegressor(**xgb_params)
        model_fwd.fit(fwd[available].values, fwd["sog_residual"].values,
                      sample_weight=fwd_w)

    model_def = None
    if len(dmen) >= 50:
        def_w = weights[train_df["position_group"] == "D"]
        model_def = XGBRegressor(**xgb_params)
        model_def.fit(dmen[available].values, dmen["sog_residual"].values,
                      sample_weight=def_w)

    var_ratio_map = {}
    for pid, grp in train_df.groupby("player_id"):
        shots = grp["shots"].values
        if len(shots) >= 10:
            mean_s = shots.mean()
            if mean_s > 0:
                var_ratio_map[int(pid)] = float(shots.var() / mean_s)

    return model_fwd, model_def, var_ratio_map, available


def run_walkforward(starting_bankroll=100.0, kelly_fraction=0.25,
                    max_kelly_pct=0.10, min_edge=0.0, min_wager=1.0,
                    min_train_days=60, test_window_days=14, step_days=14):
    """Run V2 walk-forward and compare to V1 side-by-side."""

    # Build V2 feature matrix
    logger.info("V2 WF: Building feature matrix...")
    df_v2 = model_v2._build_feature_dataframe()
    if df_v2.empty:
        return {"error": "No V2 data"}

    df_v2["date"] = pd.to_datetime(df_v2["date"])
    df_v2 = df_v2.sort_values("date").reset_index(drop=True)

    # Also build V1 for comparison
    logger.info("V1 WF: Building feature matrix...")
    df_v1 = nhl_model._build_feature_dataframe()
    df_v1["date"] = pd.to_datetime(df_v1["date"])
    df_v1 = df_v1.sort_values("date").reset_index(drop=True)

    season_start = df_v2["date"].min()
    season_end = df_v2["date"].max()
    logger.info("Season: %s to %s (V2: %d rows, V1: %d rows)",
                season_start.date(), season_end.date(), len(df_v2), len(df_v1))

    # Load props
    prop_lines = wf_v1._load_prop_lookup()
    per_book_props = wf_v1._load_per_book_props()
    logger.info("Props loaded: %d lines, %d per-book", len(prop_lines), len(per_book_props))

    # Player name map
    conn = data_collector.get_db()
    name_rows = conn.execute(
        "SELECT player_id, player_name FROM player_game_stats"
    ).fetchall()
    conn.close()
    pid_to_name = {int(r["player_id"]): wf_v1._normalize_name(r["player_name"])
                   for r in name_rows}

    # Odds date range
    conn2 = nhl_odds_collector.get_db()
    odds_dates = set(
        r["game_date"] for r in
        conn2.execute("SELECT DISTINCT game_date FROM nhl_player_props").fetchall()
    )
    conn2.close()

    if not odds_dates:
        return {"error": "No odds data"}

    min_odds_date = pd.Timestamp(min(odds_dates))
    first_test_start = max(
        season_start + timedelta(days=min_train_days),
        min_odds_date,
    )

    # V1 feature cols (no sog_prop_line)
    wf_v1_cols = [c for c in nhl_model.FEATURE_COLS if c != "sog_prop_line"]

    # Walk-forward loop
    all_bets_v2 = []
    all_bets_v1 = []
    window_results = []
    current = first_test_start
    window_num = 0

    while current < season_end:
        window_num += 1
        test_end = min(current + timedelta(days=test_window_days), season_end)

        # V2 train/test split
        train_v2 = df_v2[df_v2["date"] < current].copy()
        test_v2 = df_v2[(df_v2["date"] >= current) & (df_v2["date"] <= test_end)].copy()

        # V1 train/test split
        train_v1 = df_v1[df_v1["date"] < current].copy()
        test_v1 = df_v1[(df_v1["date"] >= current) & (df_v1["date"] <= test_end)].copy()

        if len(train_v2) < 100 or len(test_v2) == 0:
            current += timedelta(days=step_days)
            continue

        logger.info("Window %d: train to %s, test %s-%s (V2: %d/%d, V1: %d/%d)",
                    window_num,
                    train_v2["date"].max().date(),
                    test_v2["date"].min().date(),
                    test_v2["date"].max().date(),
                    len(train_v2), len(test_v2),
                    len(train_v1), len(test_v1))

        # Train V2
        m_fwd_v2, m_def_v2, vr_v2, avail_v2 = _train_v2_window(train_v2)

        # Train V1
        m_fwd_v1, m_def_v1, vr_v1 = wf_v1._train_window_model(train_v1)

        # Predict + generate bets for both models
        for model_label, test_df, m_fwd, m_def, vr_map, feat_cols, bets_list in [
            ("V2", test_v2, m_fwd_v2, m_def_v2, vr_v2, avail_v2, all_bets_v2),
            ("V1", test_v1, m_fwd_v1, m_def_v1, vr_v1, wf_v1_cols, all_bets_v1),
        ]:
            test_fwd = test_df[test_df["position_group"] == "F"].copy()
            test_def = test_df[test_df["position_group"] == "D"].copy()

            for subset, mdl, label in [
                (test_fwd, m_fwd, "F"),
                (test_def, m_def, "D"),
            ]:
                if mdl is None or len(subset) == 0:
                    continue

                avail = [c for c in feat_cols if c in subset.columns]
                pred_residual = mdl.predict(subset[avail].values)
                pred_sog = np.maximum(subset["baseline_sog"].values + pred_residual, 0.0)
                actual_sog = subset["shots"].values

                for i in range(len(subset)):
                    row = subset.iloc[i]
                    ps = float(pred_sog[i])
                    actual = int(row["shots"])
                    pid = int(row["player_id"])
                    pname = row.get("player_name", "") or pid_to_name.get(pid, "")
                    pname_norm = wf_v1._normalize_name(str(pname)) if pname else ""
                    gd = row["date"].strftime("%Y-%m-%d")
                    baseline = float(row["baseline_sog"])
                    vr = vr_map.get(pid, 1.0)

                    if not pname_norm:
                        continue

                    sim = nhl_simulation.simulate_sog(
                        pred_sog=ps, var_ratio=vr,
                        model_std=0.3, n_sims=10000,
                        seed=42 + window_num * 10000 + i,
                    )

                    ini_k, last_k = wf_v1._name_match_key(str(pname))

                    for line in nhl_simulation.PROP_LINES:
                        lk = (ini_k, last_k, gd, line)
                        if lk not in prop_lines:
                            continue
                        pl = prop_lines[lk]

                        model_p_over = sim.get(f"P_over_{line}", 0)
                        model_p_under = 1.0 - model_p_over

                        sharp = wf_v1._compute_sharp_consensus(
                            per_book_props, ini_k, last_k, gd, line)
                        soft = wf_v1._get_soft_book_prices(
                            per_book_props, ini_k, last_k, gd, line)

                        for side, price_key, model_p, won_fn in [
                            ("OVER", "over_price", model_p_over, lambda s, l: s > l),
                            ("UNDER", "under_price", model_p_under, lambda s, l: s <= l),
                        ]:
                            price = pl.get(price_key)
                            if price is None or model_p <= 0.01:
                                continue

                            implied = nhl_simulation.american_to_implied_prob(price)
                            decimal_odds = nhl_simulation.american_to_decimal(price)
                            edge = model_p - implied
                            ev = model_p * (decimal_odds - 1) - (1 - model_p)

                            if edge < min_edge:
                                continue

                            sharp_prob = None
                            sharp_edge = None
                            n_sharp = 0
                            if sharp:
                                sp = (sharp["sharp_prob_over"] if side == "OVER"
                                      else sharp["sharp_prob_under"])
                                sharp_prob = round(sp, 4)
                                sharp_edge = round(sp - implied, 4)
                                n_sharp = sharp["n_sharp_books"]

                            soft_price = None
                            soft_implied = None
                            pa_price = None
                            pa_implied = None
                            has_soft = False
                            if soft:
                                has_soft = True
                                sp_key = "soft_over_price" if side == "OVER" else "soft_under_price"
                                pa_key = "pa_over_est" if side == "OVER" else "pa_under_est"
                                soft_price = soft.get(sp_key)
                                pa_price = soft.get(pa_key)
                                if soft_price is not None:
                                    soft_implied = nhl_simulation.american_to_implied_prob(soft_price)
                                if pa_price is not None:
                                    pa_implied = nhl_simulation.american_to_implied_prob(pa_price)

                            bets_list.append({
                                "window": window_num,
                                "date": gd,
                                "player": str(pname),
                                "player_id": pid,
                                "position_group": label,
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
                                "train_size": len(train_v2) if model_label == "V2" else len(train_v1),
                                "sharp_prob": sharp_prob,
                                "sharp_edge": sharp_edge,
                                "n_sharp_books": n_sharp,
                                "sharp_agrees": (
                                    sharp_prob is not None and (
                                        (side == "OVER" and sharp_prob > 0.5)
                                        or (side == "UNDER" and sharp_prob < 0.5)
                                    )
                                ),
                                "blended_prob": (
                                    round(0.5 * model_p + 0.5 * sharp_prob, 4)
                                    if sharp_prob is not None else None
                                ),
                                "has_soft": has_soft,
                                "soft_price": soft_price,
                                "soft_implied": round(soft_implied, 4) if soft_implied else None,
                                "soft_edge": round(model_p - soft_implied, 4) if soft_implied else None,
                                "pa_price": pa_price,
                                "pa_implied": round(pa_implied, 4) if pa_implied else None,
                                "pa_edge": round(model_p - pa_implied, 4) if pa_implied else None,
                                "sharp_vs_soft_edge": (
                                    round(sharp_prob - soft_implied, 4)
                                    if sharp_prob is not None and soft_implied is not None
                                    else None
                                ),
                                "is_home": int(row.get("is_home", 0)),
                            })

        window_results.append({
            "window": window_num,
            "train_end": train_v2["date"].max().strftime("%Y-%m-%d"),
            "test_start": test_v2["date"].min().strftime("%Y-%m-%d"),
            "test_end": test_v2["date"].max().strftime("%Y-%m-%d"),
            "v2_train": len(train_v2), "v2_test": len(test_v2),
            "v1_train": len(train_v1), "v1_test": len(test_v1),
        })
        current += timedelta(days=step_days)

    # Build dataframes
    bets_v2 = pd.DataFrame(all_bets_v2) if all_bets_v2 else pd.DataFrame()
    bets_v1 = pd.DataFrame(all_bets_v1) if all_bets_v1 else pd.DataFrame()

    if bets_v2.empty:
        return {"error": "No V2 bets generated"}

    logger.info("V2 candidate bets: %d, V1 candidate bets: %d",
                len(bets_v2), len(bets_v1))

    # Evaluate strategies for both
    strats_v2 = wf_v1._evaluate_strategies(
        bets_v2, starting_bankroll, kelly_fraction, max_kelly_pct, min_wager)
    strats_v1 = wf_v1._evaluate_strategies(
        bets_v1, starting_bankroll, kelly_fraction, max_kelly_pct, min_wager)

    return {
        "season": f"{season_start.date()} to {season_end.date()}",
        "windows": window_results,
        "v2": {"bets": len(bets_v2), "strategies": strats_v2, "bets_df": bets_v2},
        "v1": {"bets": len(bets_v1), "strategies": strats_v1, "bets_df": bets_v1},
    }


def print_comparison(result):
    """Print side-by-side V1 vs V2 comparison."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("=" * 110)
    print("  NHL SOG V1 vs V2 WALK-FORWARD COMPARISON")
    print("=" * 110)
    print(f"  Season: {result['season']}")
    print(f"  Windows: {len(result['windows'])}")
    print(f"  V1 candidate bets: {result['v1']['bets']}")
    print(f"  V2 candidate bets: {result['v2']['bets']}")

    # Windows
    print()
    print("  WINDOWS:")
    print(f"  {'Win':>4s}  {'Train End':>12s}  {'Test Period':>24s}  {'V2 Trn/Tst':>12s}  {'V1 Trn/Tst':>12s}")
    print("  " + "-" * 80)
    for w in result["windows"]:
        print(f"  {w['window']:4d}  {w['train_end']:>12s}  "
              f"{w['test_start']} - {w['test_end']}  "
              f"{w['v2_train']:>5d}/{w['v2_test']:<5d}  "
              f"{w['v1_train']:>5d}/{w['v1_test']:<5d}")

    # Side-by-side strategy comparison
    key_strategies = [
        "all_ev_plus", "overs_only", "unders_only",
        "BMG_all_ev_plus", "BMG_overs", "BMG_unders",
        "BMG+sharp_all", "BMG+sharp_overs", "BMG+sharp_unders",
        "BMG_blend_all", "BMG_blend_overs", "BMG_blend_unders",
        "BMG_blend_unders_3pct", "BMG_blend_unders_5pct",
        "BMG+sharp_unders_3pct", "BMG+sharp_unders_5pct",
        "BMG+sharp_fwd_unders", "BMG+sharp_def_unders",
    ]

    strats_v1 = result["v1"]["strategies"]
    strats_v2 = result["v2"]["strategies"]

    print()
    print("  V1 vs V2 HEAD-TO-HEAD (key strategies):")
    print(f"  {'Strategy':40s}  {'--- V1 ---':>30s}  {'--- V2 ---':>30s}  {'Delta':>8s}")
    print("  " + "-" * 115)

    for name in key_strategies:
        s1 = strats_v1.get(name)
        s2 = strats_v2.get(name)

        v1_str = (f"{s1['bets']:4d} bets {s1['yield_pct']:+6.1f}% yield"
                  if s1 and s1["bets"] >= 5 else "—")
        v2_str = (f"{s2['bets']:4d} bets {s2['yield_pct']:+6.1f}% yield"
                  if s2 and s2["bets"] >= 5 else "—")

        delta = ""
        if s1 and s2 and s1["bets"] >= 5 and s2["bets"] >= 5:
            d = s2["yield_pct"] - s1["yield_pct"]
            delta = f"{d:+.1f}%"

        print(f"  {name:40s}  {v1_str:>30s}  {v2_str:>30s}  {delta:>8s}")

    # Top 10 for each
    for label, strats in [("V1", strats_v1), ("V2", strats_v2)]:
        ranked = sorted(strats.items(), key=lambda x: x[1]["yield_pct"], reverse=True)
        profitable = [(n, s) for n, s in ranked if s["bets"] >= 10 and s["yield_pct"] > 0]

        print()
        print(f"  {label} TOP 10 STRATEGIES (min 10 bets, positive yield):")
        print("  " + "-" * 97)
        if profitable:
            for name, s in profitable[:10]:
                print(f"    >>> {name}: {s['bets']} bets, "
                      f"{s['win_rate']:.1f}% wins, yield {s['yield_pct']:+.1f}%, "
                      f"profit ${s['profit']:+.2f}, DD {s['max_drawdown_pct']:.1f}%")
        else:
            print("    No profitable strategies found.")

    # Sharp confirmation value comparison
    print()
    print("  SHARP CONFIRMATION VALUE (V1 vs V2):")
    print("  " + "-" * 97)
    for lbl, base, confirm in [
        ("Unders", "BMG_unders", "BMG+sharp_unders"),
        ("Overs", "BMG_overs", "BMG+sharp_overs"),
    ]:
        for model_lbl, strats in [("V1", strats_v1), ("V2", strats_v2)]:
            b = strats.get(base)
            c = strats.get(confirm)
            b_str = f"{b['bets']} bets, {b['yield_pct']:+.1f}%" if b and b["bets"] >= 5 else "—"
            c_str = f"{c['bets']} bets, {c['yield_pct']:+.1f}%" if c and c["bets"] >= 5 else "—"
            print(f"  {model_lbl} {lbl:8s}  Base: {b_str:25s}  Sharp confirms: {c_str}")

    print()
    print("=" * 110)


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
    print_comparison(result)
