"""
Feature Ablation Framework — Sprint 3.

For each feature group, runs walk-forward with that group removed
and compares to the full model. Proves whether complexity is
justified by improved calibration, stability, or profitability.

A feature group should only remain in production if removing it
degrades at least one of: Brier score, ROI stability, P(>0), drawdown.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import model as nhl_model
import nhl_walkforward as wf_v1
import nhl_simulation
import evaluation as ev
import distribution_model as dist_model

logger = logging.getLogger(__name__)

# Walk-forward features (excludes sog_prop_line to avoid circularity)
WF_FEATURES = [c for c in nhl_model.FEATURE_COLS if c != "sog_prop_line"]

# Feature groups for ablation
FEATURE_GROUPS = {
    "baseline": ["baseline_sog"],
    "usage": ["avg_toi", "toi_last_5", "avg_shift_length", "rolling_pp_rate"],
    "form_volatility": ["player_cv", "pct_games_3plus"],
    "opponent": ["opp_shots_allowed", "opp_shots_allowed_pos"],
    "schedule": ["rest_days", "is_back_to_back", "is_home"],
    "linemate_venue": ["linemate_quality", "arena_bias"],
    "market": ["game_total", "implied_team_total", "sharp_consensus_prob"],
}


def _run_ablation_walkforward(df, prop_lines, per_book_props, pid_to_name,
                               feature_cols, min_edge=0.03,
                               min_train_days=60, test_window_days=14,
                               step_days=14):
    """Lightweight walk-forward using a custom feature set.

    Returns bets_df for the production filter.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    season_start = df["date"].min()
    season_end = df["date"].max()

    # Only test where odds exist
    odds_dates = set()
    for key in prop_lines:
        odds_dates.add(key[2])  # game_date is 3rd element

    min_odds_date = pd.Timestamp(min(odds_dates)) if odds_dates else season_end
    first_test = max(season_start + timedelta(days=min_train_days), min_odds_date)

    all_bets = []
    current = first_test

    avail = [f for f in feature_cols if f in df.columns]

    while current < season_end:
        test_end = min(current + timedelta(days=test_window_days), season_end)
        train_df = df[df["date"] < current]
        test_df = df[(df["date"] >= current) & (df["date"] <= test_end)]

        if len(train_df) < 100 or len(test_df) == 0:
            current += timedelta(days=step_days)
            continue

        # Train models
        weights = nhl_model._compute_sample_weights(train_df)
        xgb_params = dict(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
        )

        models = {}
        for pos, label in [("F", "fwd"), ("D", "def")]:
            sub = train_df[train_df["position_group"] == pos]
            if len(sub) < 50:
                continue
            w = weights[train_df["position_group"] == pos]
            m = XGBRegressor(**xgb_params)
            m.fit(sub[avail].values, sub["sog_residual"].values, sample_weight=w)
            models[pos] = m

        # Var ratios from training
        var_map = {}
        for pid, grp in train_df.groupby("player_id"):
            s = grp["shots"].values
            if len(s) >= 10 and s.mean() > 0:
                var_map[int(pid)] = float(s.var() / s.mean())

        # Score test set
        for pos in ["F", "D"]:
            mdl = models.get(pos)
            if mdl is None:
                continue
            subset = test_df[test_df["position_group"] == pos]
            if len(subset) == 0:
                continue

            cols = [c for c in avail if c in subset.columns]
            pred_sog = np.maximum(
                subset["baseline_sog"].values + mdl.predict(subset[cols].values), 0)

            for i in range(len(subset)):
                row = subset.iloc[i]
                ps = float(pred_sog[i])
                actual = int(row["shots"])
                pid = int(row["player_id"])
                pname = row.get("player_name", "") or pid_to_name.get(pid, "")
                gd = row["date"].strftime("%Y-%m-%d")
                vr = var_map.get(pid, 1.0)

                if not pname:
                    continue

                sim = nhl_simulation.simulate_sog(
                    pred_sog=ps, var_ratio=vr, model_std=0.3,
                    n_sims=5000, seed=42 + i)

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

                        blended = model_p
                        if sharp:
                            sp = (sharp["sharp_prob_over"] if side == "OVER"
                                  else sharp["sharp_prob_under"])
                            blended = 0.5 * model_p + 0.5 * sp

                        soft_implied = None
                        if soft:
                            sp_key = ("soft_over_price" if side == "OVER"
                                      else "soft_under_price")
                            sp_val = soft.get(sp_key)
                            if sp_val:
                                soft_implied = nhl_simulation.american_to_implied_prob(sp_val)

                        edge = blended - (soft_implied or implied)

                        all_bets.append({
                            "date": gd, "side": side, "odds": price,
                            "decimal_odds": round(decimal_odds, 4),
                            "model_prob": round(blended, 4),
                            "implied_prob": round(implied, 4),
                            "edge": round(edge, 4),
                            "ev": round(blended * (decimal_odds - 1) - (1 - blended), 4),
                            "won": won_fn(actual, line),
                            "has_soft": soft is not None,
                            "soft_implied": round(soft_implied, 4) if soft_implied else None,
                            "line": line,
                        })

        current += timedelta(days=step_days)

    return pd.DataFrame(all_bets) if all_bets else pd.DataFrame()


def _evaluate_production(bets_df, label=""):
    """Apply production filter and generate report."""
    if bets_df.empty:
        return {"label": label, "n_bets": 0}

    is_under = bets_df["side"] == "UNDER"
    has_soft = bets_df["has_soft"] == True
    edge_ok = bets_df["edge"] >= 0.05

    prod = bets_df[is_under & has_soft & edge_ok].copy()
    if prod.empty:
        return {"label": label, "n_bets": 0}

    return ev.generate_full_report(prod, label)


def run_ablation():
    """Run ablation suite: full model + remove-one-group for each group."""
    logger.info("Building feature matrix...")
    df = nhl_model._build_feature_dataframe()
    if df.empty:
        return {"error": "No data"}

    logger.info("Loading odds data...")
    prop_lines = wf_v1._load_prop_lookup()
    per_book_props = wf_v1._load_per_book_props()

    import data_collector
    conn = data_collector.get_db()
    pid_to_name = {
        int(r["player_id"]): wf_v1._normalize_name(r["player_name"])
        for r in conn.execute(
            "SELECT player_id, player_name FROM player_game_stats"
        ).fetchall()
    }
    conn.close()

    results = {}

    # Full model (baseline)
    logger.info("Running FULL model walk-forward...")
    full_bets = _run_ablation_walkforward(
        df, prop_lines, per_book_props, pid_to_name, WF_FEATURES)
    results["FULL"] = _evaluate_production(full_bets, "FULL model")

    # Remove each group
    for group_name, group_features in FEATURE_GROUPS.items():
        ablated_cols = [f for f in WF_FEATURES if f not in group_features]
        logger.info("Running ablation: remove %s (%s)...",
                     group_name, ", ".join(group_features))

        abl_bets = _run_ablation_walkforward(
            df, prop_lines, per_book_props, pid_to_name, ablated_cols)
        results[f"remove_{group_name}"] = _evaluate_production(
            abl_bets, f"Remove {group_name}")

    return results


def print_ablation(results):
    """Print formatted ablation comparison."""
    print(f"\n{'=' * 110}")
    print("  FEATURE ABLATION RESULTS")
    print(f"{'=' * 110}")

    full = results.get("FULL", {})
    full_yield = full.get("yield", 0)
    full_brier = full.get("brier_score", 0)
    full_p = full.get("bootstrap", {}).get("p_positive", 0)

    print(f"\n  {'Config':25s} {'Bets':>5s} {'Yield':>7s} {'dYield':>7s} "
          f"{'Brier':>7s} {'dBrier':>7s} {'P>0':>6s} {'dP>0':>6s} {'Verdict':>12s}")
    print("  " + "-" * 100)

    for name, r in sorted(results.items()):
        n = r.get("n_bets", 0)
        y = r.get("yield", 0)
        b = r.get("brier_score", 0) or 0
        p = r.get("bootstrap", {}).get("p_positive", 0)

        dy = y - full_yield
        db = b - full_brier if full_brier else 0
        dp = p - full_p

        # Verdict: removing this group HURTS (group is useful) or HELPS (overfitting)
        if name == "FULL":
            verdict = "BASELINE"
        elif dy < -2 or db > 0.005 or dp < -0.05:
            verdict = "CRITICAL"
        elif dy < -0.5 or db > 0.002:
            verdict = "USEFUL"
        elif dy > 1:
            verdict = "HARMFUL"
        else:
            verdict = "NEUTRAL"

        print(f"  {name:25s} {n:5d} {y:+6.1f}% {dy:+6.1f}% "
              f"{b:7.4f} {db:+6.4f} {p:5.3f} {dp:+5.3f} {verdict:>12s}")

    print(f"{'=' * 110}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    results = run_ablation()
    print_ablation(results)
