"""
Track A/B/C Evaluation — separates model signal from market signal.

Track A: Pure model (no market features) — measures independent predictive power
Track B: Market-only baseline — measures how much comes from the market
Track C: Hybrid (model + market) — our production strategy

For each track, runs walk-forward and produces unified evaluation via evaluation.py.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import model as nhl_model
import data_collector
import nhl_odds_collector
import nhl_simulation
import nhl_walkforward as wf_v1
import evaluation as ev

logger = logging.getLogger(__name__)

# Feature groups
PURE_FEATURES = [
    "baseline_sog", "is_home", "opp_shots_allowed", "opp_shots_allowed_pos",
    "avg_toi", "toi_last_5", "avg_shift_length", "rolling_pp_rate",
    "player_cv", "pct_games_3plus", "rest_days", "is_back_to_back",
    "linemate_quality", "arena_bias",
]

MARKET_FEATURES = [
    "game_total", "implied_team_total", "sog_prop_line", "sharp_consensus_prob",
]

ALL_FEATURES = [c for c in nhl_model.FEATURE_COLS if c != "sog_prop_line"]
PURE_ONLY = [c for c in ALL_FEATURES if c not in MARKET_FEATURES]
MARKET_ONLY = [c for c in MARKET_FEATURES if c in ALL_FEATURES]


def run_track_evaluation(min_train_days=60, test_days=14, step_days=14):
    """Run walk-forward for Track A, B, C and produce unified evaluation."""

    logger.info("Building feature matrix...")
    df = nhl_model._build_feature_dataframe()
    if df.empty:
        return {"error": "No data"}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Load props and odds for bet matching
    prop_lines = wf_v1._load_prop_lookup()
    per_book_props = wf_v1._load_per_book_props()

    conn = data_collector.get_db()
    name_rows = conn.execute("SELECT player_id, player_name FROM player_game_stats").fetchall()
    conn.close()
    pid_to_name = {int(r["player_id"]): wf_v1._normalize_name(r["player_name"]) for r in name_rows}

    # Odds date range
    conn2 = nhl_odds_collector.get_db()
    odds_dates = set(r["game_date"] for r in
        conn2.execute("SELECT DISTINCT game_date FROM nhl_player_props").fetchall())
    conn2.close()

    if not odds_dates:
        return {"error": "No odds data"}

    season_start = df["date"].min()
    season_end = df["date"].max()
    min_odds_date = pd.Timestamp(min(odds_dates))
    first_test = max(season_start + timedelta(days=min_train_days), min_odds_date)

    # XGB params
    xgb_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
    )

    # Collect bets for each track
    track_bets = {"A": [], "B": [], "C": []}
    current = first_test
    wnum = 0

    while current < season_end:
        wnum += 1
        test_end = min(current + timedelta(days=test_days), season_end)
        train_df = df[df["date"] < current].copy()
        test_df = df[(df["date"] >= current) & (df["date"] <= test_end)].copy()

        if len(train_df) < 100 or len(test_df) == 0:
            current += timedelta(days=step_days)
            continue

        logger.info("Window %d: train %d, test %d", wnum, len(train_df), len(test_df))

        weights = nhl_model._compute_sample_weights(train_df)

        # Train 3 models
        models = {}
        for track, feat_cols in [("A", PURE_ONLY), ("B", MARKET_ONLY + ["baseline_sog"]), ("C", ALL_FEATURES)]:
            avail = [f for f in feat_cols if f in train_df.columns]
            if len(avail) < 2:
                continue

            fwd = train_df[train_df["position_group"] == "F"]
            dmen = train_df[train_df["position_group"] == "D"]

            m_fwd = m_def = None
            if len(fwd) >= 50:
                fwd_w = weights[train_df["position_group"] == "F"]
                m_fwd = XGBRegressor(**xgb_params)
                m_fwd.fit(fwd[avail].values, fwd["sog_residual"].values, sample_weight=fwd_w)
            if len(dmen) >= 50:
                def_w = weights[train_df["position_group"] == "D"]
                m_def = XGBRegressor(**xgb_params)
                m_def.fit(dmen[avail].values, dmen["sog_residual"].values, sample_weight=def_w)

            models[track] = (m_fwd, m_def, avail)

        # Variance ratio map from training data
        var_ratio_map = {}
        for pid, grp in train_df.groupby("player_id"):
            shots = grp["shots"].values
            if len(shots) >= 10 and shots.mean() > 0:
                var_ratio_map[int(pid)] = float(shots.var() / shots.mean())

        # Score test set and generate bets for each track
        for track, (m_fwd, m_def, avail) in models.items():
            test_fwd = test_df[test_df["position_group"] == "F"].copy()
            test_def = test_df[test_df["position_group"] == "D"].copy()

            for subset, mdl, label in [(test_fwd, m_fwd, "F"), (test_def, m_def, "D")]:
                if mdl is None or len(subset) == 0:
                    continue

                cols = [c for c in avail if c in subset.columns]
                pred_residual = mdl.predict(subset[cols].values)
                pred_sog = np.maximum(subset["baseline_sog"].values + pred_residual, 0.0)
                actual_sog = subset["shots"].values

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
                        seed=42 + wnum * 10000 + i)

                    pname_key = wf_v1._name_match_key(str(pname))
                    ini_k, last_k = pname_key

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
                            edge = model_p - implied

                            # Blended prob for Track C
                            blended = model_p
                            if sharp and track == "C":
                                sp = sharp["sharp_prob_over"] if side == "OVER" else sharp["sharp_prob_under"]
                                blended = 0.5 * model_p + 0.5 * sp

                            # Soft book odds
                            soft_price = None
                            soft_implied = None
                            if soft:
                                sp_key = "soft_over_price" if side == "OVER" else "soft_under_price"
                                soft_price = soft.get(sp_key)
                                if soft_price:
                                    soft_implied = nhl_simulation.american_to_implied_prob(soft_price)

                            # Use blended for edge calc in Track C, model-only for A
                            if track == "C" and soft_implied:
                                final_edge = blended - soft_implied
                            elif track == "B" and sharp:
                                sp = sharp["sharp_prob_over"] if side == "OVER" else sharp["sharp_prob_under"]
                                final_edge = sp - implied
                                blended = sp
                            else:
                                final_edge = edge

                            track_bets[track].append({
                                "window": wnum,
                                "date": gd,
                                "player_id": pid,
                                "line": line,
                                "side": side,
                                "odds": price,
                                "decimal_odds": round(decimal_odds, 4),
                                "model_prob": round(blended, 4),
                                "implied_prob": round(implied, 4),
                                "edge": round(final_edge, 4),
                                "ev": round(blended * (decimal_odds - 1) - (1 - blended), 4),
                                "won": won_fn(actual, line),
                                "pred_sog": round(ps, 2),
                                "actual_sog": actual,
                                "has_soft": soft is not None,
                                "soft_price": soft_price,
                                "soft_implied": round(soft_implied, 4) if soft_implied else None,
                            })

        current += timedelta(days=step_days)

    # Build DataFrames and evaluate
    results = {}
    for track in ["A", "B", "C"]:
        bdf = pd.DataFrame(track_bets[track]) if track_bets[track] else pd.DataFrame()
        if bdf.empty:
            continue

        # Production filter: unders with 5%+ edge vs soft book
        is_under = bdf["side"] == "UNDER"
        has_soft = bdf["has_soft"] == True
        edge_5 = bdf["edge"] >= 0.05

        prod = bdf[is_under & has_soft & edge_5].copy()

        # Full evaluation
        report_all = ev.generate_full_report(bdf[bdf["ev"] > 0], f"Track {track}: All +EV")
        report_prod = ev.generate_full_report(prod, f"Track {track}: Production (unders 5%+ vs BMG)")

        results[track] = {
            "all_ev": report_all,
            "production": report_prod,
            "bets_df": bdf,
        }

    return results


def print_track_comparison(results):
    """Print side-by-side Track A/B/C comparison."""
    print("\n" + "=" * 100)
    print("  TRACK A / B / C COMPARISON")
    print("=" * 100)

    print(f"\n  {'Metric':35s} {'Track A (Pure)':>18s} {'Track B (Market)':>18s} {'Track C (Hybrid)':>18s}")
    print("  " + "-" * 92)

    for strat in ["production"]:
        for metric_name, key, fmt in [
            ("Bets", "n_bets", "d"),
            ("Win Rate", "win_rate", ".1f"),
            ("Yield %", "yield", ".2f"),
            ("Brier Score", "brier_score", ".4f"),
            ("ECE", "ece", ".4f"),
            ("Cal Slope", "calibration_slope", ".3f"),
            ("Bootstrap P(>0)", None, None),
        ]:
            vals = []
            for track in ["A", "B", "C"]:
                r = results.get(track, {}).get(strat, {})
                if key is None:
                    # Bootstrap P(>0)
                    boot = r.get("bootstrap", {})
                    vals.append(f"{boot.get('p_positive', '—')}")
                elif key in r:
                    vals.append(f"{r[key]:{fmt}}" if fmt else str(r[key]))
                else:
                    vals.append("—")

            print(f"  {metric_name:35s} {vals[0]:>18s} {vals[1]:>18s} {vals[2]:>18s}")

    # Edge monotonicity comparison
    print(f"\n  Edge Monotonicity:")
    for track in ["A", "B", "C"]:
        r = results.get(track, {}).get("production", {})
        em = r.get("edge_monotonicity", {})
        mono = em.get("is_monotonic", "?")
        print(f"    Track {track}: monotonic={mono}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    results = run_track_evaluation()

    # Print detailed reports
    for track in ["A", "B", "C"]:
        if track in results:
            ev.print_report(results[track]["production"])

    print_track_comparison(results)
