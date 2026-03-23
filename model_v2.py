"""
V2 NHL Shots on Goal Model — expanded features + player clustering.

Coexists with model.py (V1). Same interface: train_model(), predict_player(),
predict_upcoming_games(), get_model_metrics(), save_model(), load_model().

Key differences from V1:
- MoneyPuck shot-style features (slot %, xG/shot, rebound rate, etc.)
- Dynamic player archetype clustering
- Expanded opponent features (high-danger shots, pace, xG allowed)
- Feature registry enforcement (no silent NaN degradation)
- More features: ~30 vs V1's 16
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import data_collector
import moneypuck_collector
import nhl_api
import nhl_odds_collector
from clustering import PlayerClusterer, CLUSTER_FEATURES
import feature_registry

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).resolve().parent / "saved_model_v2"

# V2 feature columns — registered in feature_registry.yaml
FEATURE_COLS = [
    # Core form
    "baseline_sog", "player_cv", "pct_games_3plus", "rolling_sog_ewm",
    # Opportunity / role
    "avg_toi", "toi_last_5", "toi_trend", "avg_shift_length",
    "rolling_pp_rate", "pp_toi_pct", "ice_time_rank",
    # Shot style (MoneyPuck)
    "shots_per_60", "slot_pct", "rebound_rate", "rush_rate",
    "avg_shot_distance", "xg_per_shot", "high_danger_rate",
    # Opponent
    "opp_shots_allowed", "opp_shots_allowed_pos",
    "opp_hd_shots_allowed", "opp_xg_allowed", "opp_pace",
    # Game context
    "is_home", "rest_days", "is_back_to_back",
    "game_total", "implied_team_total",
    # Player cluster
    "cluster_id", "cluster_distance", "cluster_mean_sog", "cluster_x_opp_hd",
    # Team cluster matchup (how player's team archetype performs vs opponent archetype)
    "opp_team_cluster_ga",    # how many goals opponent's cluster typically allows
    "opp_team_cluster_save",  # opponent cluster's save %
    "player_cluster_x_opp_team_cluster",  # player archetype SOG vs opponent team archetype
]

# Global model state
_model_fwd = None
_model_def = None
_model_metrics = {}
_clusterer = PlayerClusterer()
_player_cv = {}

HAS_MONEYPUCK = True  # Set False if MoneyPuck tables are empty


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if _model_fwd:
        _model_fwd.save_model(str(SAVE_DIR / "model_fwd.json"))
    if _model_def:
        _model_def.save_model(str(SAVE_DIR / "model_def.json"))
    _clusterer.save(SAVE_DIR)
    with open(SAVE_DIR / "meta.json", "w") as f:
        json.dump({
            "model_version": 2,
            "metrics": _model_metrics,
            "player_cv": {str(k): v for k, v in _player_cv.items()},
        }, f)
    logger.info("V2 model saved to %s", SAVE_DIR)


def load_model() -> bool:
    global _model_fwd, _model_def, _model_metrics, _player_cv

    meta_path = SAVE_DIR / "meta.json"
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    _model_metrics = meta.get("metrics", {})
    _player_cv = {int(k): v for k, v in meta.get("player_cv", {}).items()}

    fwd_path = SAVE_DIR / "model_fwd.json"
    def_path = SAVE_DIR / "model_def.json"

    if fwd_path.exists():
        _model_fwd = XGBRegressor()
        _model_fwd.load_model(str(fwd_path))

    if def_path.exists():
        _model_def = XGBRegressor()
        _model_def.load_model(str(def_path))

    _clusterer.load(SAVE_DIR)

    mae = _model_metrics.get("mae", "?")
    logger.info("V2 model loaded (MAE: %s)", mae)
    return True


def get_model_metrics() -> dict:
    return _model_metrics


# ---------------------------------------------------------------------------
# Baseline computation (same as V1)
# ---------------------------------------------------------------------------

def _compute_baseline(season_avg, rolling_10, rolling_5):
    return 0.55 * season_avg + 0.30 * rolling_10 + 0.15 * rolling_5


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_dataframe() -> pd.DataFrame:
    """Build training dataframe with V2 features.

    One row per player-game. All features use only pre-game data.
    """
    global _player_cv
    conn = data_collector.get_db()

    # Base data from NHL API (same as V1)
    df = pd.read_sql_query(
        """SELECT pgs.*, g.date, g.home_team, g.away_team
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           ORDER BY pgs.player_id, g.date""",
        conn,
    )

    if df.empty:
        conn.close()
        return pd.DataFrame()

    # MoneyPuck game-by-game data (situation='all')
    mp_df = pd.read_sql_query(
        """SELECT game_id, player_id, game_date, icetime, shifts,
                  ice_time_rank, shots_on_goal, shot_attempts,
                  high_danger_shots, medium_danger_shots, low_danger_shots,
                  xgoals, rebounds_created, xgoals_with_rebounds,
                  on_ice_sog_for, on_ice_sa_against,
                  on_ice_xg_for, on_ice_xg_against,
                  on_ice_hd_shots_for, on_ice_hd_shots_against,
                  game_score
           FROM mp_player_game
           WHERE situation = 'all'
           ORDER BY player_id, game_date""",
        conn,
    )

    # MoneyPuck PP data for pp_toi_pct
    mp_pp = pd.read_sql_query(
        """SELECT game_id, player_id, icetime as pp_icetime
           FROM mp_player_game WHERE situation = '5on4'""",
        conn,
    )

    # Shot-level style data (pre-aggregated per player-game)
    mp_shot_style = pd.read_sql_query(
        """SELECT player_id, game_id,
                  COUNT(*) as n_attempts,
                  SUM(shot_on_goal) as n_sog,
                  SUM(is_rebound) as n_rebounds,
                  SUM(is_rush) as n_rush,
                  AVG(shot_distance_adjusted) as mean_distance,
                  AVG(xgoal) as mean_xg,
                  SUM(CASE WHEN ABS(x_coord) >= 60 AND ABS(y_coord) <= 22 THEN 1 ELSE 0 END) as n_slot
           FROM mp_shots
           GROUP BY player_id, game_id""",
        conn,
    )

    # Opponent defense data (from NHL API, same as V1)
    opp_shots_df = pd.read_sql_query(
        """SELECT pgs.game_id, g.date, pgs.team, pgs.position, pgs.shots,
                  g.home_team, g.away_team
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.position IN ('L', 'C', 'R', 'D')
           ORDER BY g.date""",
        conn,
    )

    # MoneyPuck opponent stats (for expanded opponent features)
    mp_opp_df = pd.read_sql_query(
        """SELECT game_id, game_date, opponent,
                  SUM(shot_attempts) as opp_total_attempts,
                  SUM(high_danger_shots) as opp_hd_shots_total,
                  SUM(xgoals) as opp_xg_total
           FROM mp_player_game
           WHERE situation = 'all'
           GROUP BY game_id, opponent
           ORDER BY game_date""",
        conn,
    )

    # Load odds data
    try:
        odds_bulk = nhl_odds_collector.load_game_odds_bulk()
        props_bulk = nhl_odds_collector.load_sharp_consensus_bulk()
    except Exception:
        odds_bulk = {}
        props_bulk = {}

    conn.close()

    # --- Position grouping ---
    df["position_group"] = df["position"].map(
        {"L": "F", "C": "F", "R": "F", "D": "D"}
    )
    df = df[df["position_group"].notna()].copy()
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # --- Merge MoneyPuck PP data ---
    if not mp_pp.empty:
        mp_pp = mp_pp.drop_duplicates(["game_id", "player_id"])

    # --- Team clustering for opponent matchup features ---
    import team_clustering
    _tc_cache = {}  # month -> (profiles, matrix)

    def _get_team_cluster_data(opp_team, game_date):
        bucket = game_date[:7]
        if bucket not in _tc_cache:
            try:
                tp = team_clustering.build_team_profiles(before_date=game_date, window=60)
                if not tp.empty and len(tp) >= 10:
                    _, _, _, tp = team_clustering.fit_team_clusters(tp)
                    mx = team_clustering.build_cluster_matchup_matrix(tp, game_date)
                    _tc_cache[bucket] = (tp, mx)
                else:
                    _tc_cache[bucket] = None
            except Exception:
                _tc_cache[bucket] = None

        cached = _tc_cache.get(bucket)
        if cached:
            tp, mx = cached
            tc_map = dict(zip(tp["team"], tp["cluster"]))
            opp_cluster = tc_map.get(opp_team)
            if opp_cluster is not None:
                opp_members = tp[tp["cluster"] == opp_cluster]
                return {
                    "opp_cluster": int(opp_cluster),
                    "opp_cluster_ga": float(opp_members["ga_per_game"].mean()),
                    "opp_cluster_save": float(opp_members["save_pct"].mean()),
                }
        return {"opp_cluster": -1, "opp_cluster_ga": 3.0, "opp_cluster_save": 0.89}

    # --- Build rolling features per player ---
    logger.info("V2: Building player features...")

    game_dates = (
        df.drop_duplicates("game_id")[["game_id", "date"]]
        .set_index("game_id")["date"].to_dict()
    )

    # Build rolling defense profiles (same as V1)
    opp_shots_df["opponent"] = np.where(
        opp_shots_df["team"] == opp_shots_df["home_team"],
        opp_shots_df["away_team"],
        opp_shots_df["home_team"],
    )
    opp_shots_df["defending_team"] = opp_shots_df["opponent"]
    opp_shots_df = opp_shots_df.sort_values("date")

    rolling_defense = _build_rolling_defense(opp_shots_df)

    # Build rolling MoneyPuck opponent features
    mp_opp_rolling = _build_mp_opp_rolling(mp_opp_df)

    # Index MoneyPuck data for fast lookup (convert to dicts for safe access)
    mp_idx = {}
    if not mp_df.empty:
        for _, row in mp_df.iterrows():
            mp_idx[(row["game_id"], row["player_id"])] = row.to_dict()

    mp_pp_idx = {}
    if not mp_pp.empty:
        for _, row in mp_pp.iterrows():
            mp_pp_idx[(row["game_id"], row["player_id"])] = row.to_dict()

    mp_style_idx = {}
    if not mp_shot_style.empty:
        for _, row in mp_shot_style.iterrows():
            mp_style_idx.setdefault(row["player_id"], []).append(row.to_dict())

    # --- Per-player rolling computation ---
    records = []
    player_groups = df.groupby("player_id")

    for player_id, group in player_groups:
        group = group.sort_values("date")
        shots_list = group["shots"].tolist()
        toi_list = group["toi"].tolist()
        shifts_list = group["shifts"].tolist()
        pp_goals_list = [(g or 0) for g in group["pp_goals"].tolist()]
        dates = group["date"].tolist()
        game_ids = group["game_id"].tolist()
        positions = group["position"].tolist()
        teams = group["team"].tolist()

        # Compute player CV
        if len(shots_list) >= 5:
            cv = float(np.std(shots_list) / max(np.mean(shots_list), 0.01))
        else:
            cv = 1.0
        _player_cv[player_id] = cv

        for i in range(len(group)):
            if i < 3:
                continue  # Need at least 3 prior games

            prior_shots = shots_list[:i]
            prior_toi = toi_list[:i]
            prior_shifts = shifts_list[:i]
            prior_pp = pp_goals_list[:i]

            # Core form features
            season_avg = np.mean(prior_shots)
            rolling_10 = np.mean(prior_shots[-10:])
            rolling_5 = np.mean(prior_shots[-5:])
            baseline = _compute_baseline(season_avg, rolling_10, rolling_5)

            pct_3plus = sum(1 for s in prior_shots if s >= 3) / len(prior_shots)

            # EWM SOG
            ewm_series = pd.Series(prior_shots).ewm(span=10, min_periods=3).mean()
            rolling_sog_ewm = float(ewm_series.iloc[-1]) if len(ewm_series) > 0 else baseline

            # Opportunity features
            avg_toi_val = np.mean(prior_toi[-20:]) if prior_toi else 15.0
            toi_l5 = np.mean(prior_toi[-5:]) if len(prior_toi) >= 5 else avg_toi_val
            toi_l20 = np.mean(prior_toi[-20:]) if len(prior_toi) >= 5 else avg_toi_val
            toi_trend = toi_l5 / max(toi_l20, 1.0)

            valid_shifts = [s for s in prior_shifts[-20:] if s and s > 0]
            valid_toi_for_shift = prior_toi[-20:]
            if valid_shifts and valid_toi_for_shift:
                avg_shift = np.mean([
                    t * 60 / s for t, s in zip(valid_toi_for_shift, valid_shifts) if s > 0
                ])
            else:
                avg_shift = 45.0

            pp_rate = np.mean(prior_pp[-10:]) if prior_pp else 0.0

            # PP TOI % from MoneyPuck
            pp_toi_val = 0.0
            recent_games = game_ids[max(0, i - 10):i]
            total_ice = 0.0
            total_pp_ice = 0.0
            for gid in recent_games:
                mp_row = mp_idx.get((gid, player_id))
                pp_row = mp_pp_idx.get((gid, player_id))
                if mp_row is not None and mp_row["icetime"]:
                    total_ice += mp_row["icetime"]
                if pp_row is not None and pp_row["pp_icetime"]:
                    total_pp_ice += pp_row["pp_icetime"]
            if total_ice > 0:
                pp_toi_val = total_pp_ice / total_ice

            # Ice time rank (most recent game from MoneyPuck)
            prev_gid = game_ids[i - 1]
            mp_prev = mp_idx.get((prev_gid, player_id))
            itr = mp_prev["ice_time_rank"] if mp_prev and mp_prev["ice_time_rank"] else None

            # Shot style features (rolling 10 from shot-level data)
            style_rows = mp_style_idx.get(player_id, [])
            # Filter to pre-game only
            prior_game_set = set(game_ids[:i])
            prior_style = [r for r in style_rows if r["game_id"] in prior_game_set]
            prior_style = prior_style[-10:]  # last 10 games

            if prior_style:
                total_att = sum(r["n_attempts"] for r in prior_style)
                total_sog_mp = sum(r["n_sog"] for r in prior_style)
                total_reb = sum(r["n_rebounds"] for r in prior_style)
                total_rush = sum(r["n_rush"] for r in prior_style)
                total_slot = sum(r["n_slot"] for r in prior_style)

                slot_pct_val = total_slot / max(total_att, 1)
                rebound_rate_val = total_reb / max(total_att, 1)
                rush_rate_val = total_rush / max(total_att, 1)
                avg_dist_val = np.mean([r["mean_distance"] for r in prior_style
                                        if r["mean_distance"] is not None])
                xg_per_shot_val = np.mean([r["mean_xg"] for r in prior_style
                                           if r["mean_xg"] is not None])
                # Shots per 60 (use MoneyPuck icetime)
                total_mp_ice = sum(
                    mp_idx.get((r["game_id"], player_id), {}).get("icetime", 0) or 0
                    for r in prior_style
                    if isinstance(mp_idx.get((r["game_id"], player_id)), dict)
                )
                if total_mp_ice == 0:
                    # Fallback to mp_idx rows
                    total_mp_ice = sum(
                        mp_idx[(r["game_id"], player_id)]["icetime"]
                        for r in prior_style
                        if (r["game_id"], player_id) in mp_idx
                        and mp_idx[(r["game_id"], player_id)]["icetime"]
                    )
                shots_per_60_val = (total_sog_mp / max(total_mp_ice, 1)) * 3600 if total_mp_ice > 60 else None

                # High danger rate
                hd_total = sum(
                    mp_idx.get((r["game_id"], player_id), {}).get("high_danger_shots", 0) or 0
                    for r in prior_style
                    if isinstance(mp_idx.get((r["game_id"], player_id)), dict)
                )
                if hd_total == 0:
                    hd_total = sum(
                        mp_idx[(r["game_id"], player_id)]["high_danger_shots"] or 0
                        for r in prior_style
                        if (r["game_id"], player_id) in mp_idx
                    )
                hd_rate = hd_total / max(total_sog_mp, 1) if total_sog_mp > 0 else None
            else:
                slot_pct_val = None
                rebound_rate_val = None
                rush_rate_val = None
                avg_dist_val = None
                xg_per_shot_val = None
                shots_per_60_val = None
                hd_rate = None

            # --- Opponent features ---
            row_data = group.iloc[i]
            opp_team = (row_data["away_team"]
                        if row_data["team"] == row_data["home_team"]
                        else row_data["home_team"])
            game_date = dates[i]
            pos = positions[i]

            # V1-style opponent defense
            def_key = (game_date, opp_team)
            opp_def = rolling_defense.get(def_key, {})
            opp_sa = opp_def.get("overall", 30.0)
            opp_sa_pos = opp_def.get(pos, opp_sa / 18)

            # MoneyPuck expanded opponent features
            mp_opp_key = (game_date, opp_team)
            mp_opp = mp_opp_rolling.get(mp_opp_key, {})
            opp_hd = mp_opp.get("hd_per_game")
            opp_xg = mp_opp.get("xg_per_game")
            opp_pace_val = mp_opp.get("attempts_per_game")

            # --- Game context ---
            rest = row_data.get("rest_days", 2)
            if rest is None or rest < 0:
                rest = 2
            rest = min(rest, 4)
            is_b2b = 1 if rest == 0 else 0
            is_home_val = 1 if row_data["is_home"] else 0

            # Odds features
            game_total_val = np.nan
            implied_tt_val = np.nan

            home_team = row_data["home_team"]
            away_team = row_data["away_team"]
            team = teams[i]

            if odds_bulk:
                odds_key = (game_date, home_team, away_team)
                odds_ctx = odds_bulk.get(odds_key)
                if odds_ctx:
                    gt = odds_ctx.get("game_total")
                    if gt is not None:
                        game_total_val = gt
                    if is_home_val:
                        itt = odds_ctx.get("implied_home_total")
                    else:
                        itt = odds_ctx.get("implied_away_total")
                    if itt is not None:
                        implied_tt_val = itt

            # Target
            actual_sog = shots_list[i]
            residual = actual_sog - baseline

            records.append({
                "game_id": game_ids[i],
                "player_id": player_id,
                "date": game_date,
                "team": team,
                "opponent": opp_team,
                "position": pos,
                "position_group": "D" if pos == "D" else "F",
                "shots": actual_sog,
                "baseline_sog": baseline,
                "sog_residual": residual,
                # Core form
                "player_cv": cv,
                "pct_games_3plus": pct_3plus,
                "rolling_sog_ewm": rolling_sog_ewm,
                # Opportunity
                "avg_toi": avg_toi_val,
                "toi_last_5": toi_l5,
                "toi_trend": toi_trend,
                "avg_shift_length": avg_shift,
                "rolling_pp_rate": pp_rate,
                "pp_toi_pct": pp_toi_val,
                "ice_time_rank": itr,
                # Shot style
                "shots_per_60": shots_per_60_val,
                "slot_pct": slot_pct_val,
                "rebound_rate": rebound_rate_val,
                "rush_rate": rush_rate_val,
                "avg_shot_distance": avg_dist_val,
                "xg_per_shot": xg_per_shot_val,
                "high_danger_rate": hd_rate,
                # Opponent
                "opp_shots_allowed": opp_sa,
                "opp_shots_allowed_pos": opp_sa_pos,
                "opp_hd_shots_allowed": opp_hd,
                "opp_xg_allowed": opp_xg,
                "opp_pace": opp_pace_val,
                # Game context
                "is_home": is_home_val,
                "rest_days": rest,
                "is_back_to_back": is_b2b,
                "game_total": game_total_val,
                "implied_team_total": implied_tt_val,
                # Team cluster (opponent archetype)
                **{f"opp_team_cluster_{k}": v
                   for k, v in [("ga", _get_team_cluster_data(opp_team, game_date).get("opp_cluster_ga", 3.0)),
                                ("save", _get_team_cluster_data(opp_team, game_date).get("opp_cluster_save", 0.89))]},
            })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # --- Player clustering ---
    result = _add_cluster_features(result)

    # --- Player cluster × opponent team cluster interaction ---
    # How does this player archetype's avg SOG relate to the opponent's defensive archetype?
    if "cluster_mean_sog" in result.columns and "opp_team_cluster_ga" in result.columns:
        opp_ga_mean = result["opp_team_cluster_ga"].mean()
        if opp_ga_mean and opp_ga_mean > 0:
            result["player_cluster_x_opp_team_cluster"] = (
                result["cluster_mean_sog"] * result["opp_team_cluster_ga"] / opp_ga_mean
            )
        else:
            result["player_cluster_x_opp_team_cluster"] = 0.0
    else:
        result["player_cluster_x_opp_team_cluster"] = 0.0

    # --- Feature registry validation ---
    available_features = [f for f in FEATURE_COLS if f in result.columns]
    result = feature_registry.apply_null_policies(result, available_features)

    logger.info("V2 feature matrix: %d rows, %d features (%d with MoneyPuck)",
                len(result), len(available_features),
                result["slot_pct"].notna().sum())

    return result


# ---------------------------------------------------------------------------
# Rolling defense (same logic as V1)
# ---------------------------------------------------------------------------

def _build_rolling_defense(opp_shots_df):
    """Build date-keyed rolling opponent defense profiles."""
    defense_by_date = {}
    team_history = {}

    for _, row in opp_shots_df.iterrows():
        def_team = row["defending_team"]
        if def_team not in team_history:
            team_history[def_team] = {"overall": [], "pos": {}}

        hist = team_history[def_team]
        hist["overall"].append(row["shots"])
        pos = row["position"]
        if pos not in hist["pos"]:
            hist["pos"][pos] = []
        hist["pos"][pos].append(row["shots"])

        # Keep last 20 games worth
        if len(hist["overall"]) > 20 * 18:
            hist["overall"] = hist["overall"][-20 * 18:]

        date_key = (row["date"], def_team)
        if date_key not in defense_by_date:
            defense_by_date[date_key] = {
                "overall": np.mean(hist["overall"][-20 * 18:]) * 18
                           if hist["overall"] else 30.0,
            }
            for p, vals in hist["pos"].items():
                recent = vals[-20 * 3:] if len(vals) > 20 * 3 else vals
                defense_by_date[date_key][p] = np.mean(recent) if recent else 1.5

    return defense_by_date


# ---------------------------------------------------------------------------
# MoneyPuck opponent rolling features
# ---------------------------------------------------------------------------

def _build_mp_opp_rolling(mp_opp_df):
    """Build rolling opponent features from MoneyPuck data."""
    if mp_opp_df.empty:
        return {}

    mp_opp_df = mp_opp_df.sort_values("game_date")
    result = {}
    team_history = {}

    for _, row in mp_opp_df.iterrows():
        opp = row["opponent"]
        if opp not in team_history:
            team_history[opp] = []

        # Before recording this game, store current rolling for this date
        date_key = (row["game_date"], opp)
        if date_key not in result and team_history[opp]:
            recent = team_history[opp][-15:]
            n = len(recent)
            result[date_key] = {
                "hd_per_game": sum(r["hd"] for r in recent) / n,
                "xg_per_game": sum(r["xg"] for r in recent) / n,
                "attempts_per_game": sum(r["att"] for r in recent) / n,
            }

        team_history[opp].append({
            "hd": row["opp_hd_shots_total"] or 0,
            "xg": row["opp_xg_total"] or 0,
            "att": row["opp_total_attempts"] or 0,
        })
        # Keep last 20 games
        if len(team_history[opp]) > 20:
            team_history[opp] = team_history[opp][-20:]

    return result


# ---------------------------------------------------------------------------
# Clustering features
# ---------------------------------------------------------------------------

def _add_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fit clusters on training data and add cluster features."""
    global _clusterer

    # Build cluster input: per-player averages of style/deployment features
    cluster_input_cols = {
        "avg_toi": "avg_toi",
        "pp_toi_pct": "pp_toi_pct",
        "slot_pct": "slot_pct",
        "avg_shot_distance": "avg_shot_distance",
        "rebound_rate": "rebound_rate",
        "rush_rate": "rush_rate",
        "shots_per_60": "shots_per_60",
        "avg_shift_length": "avg_shift_length",
    }

    # Compute per-player averages (using only available data)
    player_avgs = df.groupby("player_id")[list(cluster_input_cols.values())].mean()
    player_avgs = player_avgs.dropna(thresh=5)  # Need at least 5 non-null features

    if len(player_avgs) < 20:
        logger.warning("V2: Not enough players for clustering (%d)", len(player_avgs))
        df["cluster_id"] = -1
        df["cluster_distance"] = 1.0
        df["cluster_mean_sog"] = df["baseline_sog"]
        df["cluster_x_opp_hd"] = 0.0
        return df

    # Fill remaining NaN with column means for clustering
    player_avgs_filled = player_avgs.fillna(player_avgs.mean())
    X_cluster = player_avgs_filled.values

    # Player SOG averages for cluster profiles
    player_sog_avg = df.groupby("player_id")["shots"].mean()
    player_sog = player_sog_avg.reindex(player_avgs_filled.index).fillna(2.0).values

    # Fit clusters
    logger.info("V2: Fitting player clusters...")
    _clusterer.fit(X_cluster, player_sog)
    cluster_ids, distances = _clusterer.predict(X_cluster)

    # Map player_id -> cluster info
    player_cluster_map = {}
    for pid, cid, dist in zip(player_avgs_filled.index, cluster_ids, distances):
        player_cluster_map[pid] = {
            "cluster_id": int(cid),
            "cluster_distance": float(dist),
            "cluster_mean_sog": _clusterer.get_cluster_mean_sog(int(cid)),
        }

    # Apply to dataframe
    df["cluster_id"] = df["player_id"].map(
        lambda pid: player_cluster_map.get(pid, {}).get("cluster_id", -1)
    )
    df["cluster_distance"] = df["player_id"].map(
        lambda pid: player_cluster_map.get(pid, {}).get("cluster_distance", 1.0)
    )
    df["cluster_mean_sog"] = df["player_id"].map(
        lambda pid: player_cluster_map.get(pid, {}).get("cluster_mean_sog", 2.0)
    )

    # Interaction: cluster_mean_sog * opponent high-danger rate (normalized)
    opp_hd_mean = df["opp_hd_shots_allowed"].mean()
    if opp_hd_mean and opp_hd_mean > 0:
        df["cluster_x_opp_hd"] = (
            df["cluster_mean_sog"] * df["opp_hd_shots_allowed"].fillna(opp_hd_mean) / opp_hd_mean
        )
    else:
        df["cluster_x_opp_hd"] = 0.0

    logger.info("V2: Clusters assigned — %s", _clusterer.describe_clusters()[:200])
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Same weighting as V1: upweight high-volume, recent games."""
    shots = df["shots"].values
    baseline = df["baseline_sog"].values

    volume_w = np.where(baseline >= 2.5, 1.5, np.where(baseline >= 1.5, 1.0, 0.7))

    dates = pd.to_datetime(df["date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values
    recency_w = np.exp(-days_ago / 120)

    return volume_w * recency_w


def _train_single_model(train_df, test_df, label):
    """Train one XGBoost model. Returns (model, metrics)."""
    available = [f for f in FEATURE_COLS if f in train_df.columns]

    X_train = train_df[available].values
    y_train = train_df["sog_residual"].values
    X_test = test_df[available].values
    y_test_residual = test_df["sog_residual"].values
    y_test_actual = test_df["shots"].values
    baseline_test = test_df["baseline_sog"].values

    weights = _compute_sample_weights(train_df)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.5,
        min_child_weight=10,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=weights)

    pred_residual = model.predict(X_test)
    pred_sog = np.maximum(baseline_test + pred_residual, 0.0)
    mae = float(np.mean(np.abs(pred_sog - y_test_actual)))
    rmse = float(np.sqrt(np.mean((pred_sog - y_test_actual) ** 2)))

    # Feature importance
    importance = model.feature_importances_
    feat_imp = {
        col: round(float(imp), 4)
        for col, imp in sorted(zip(available, importance),
                                key=lambda x: x[1], reverse=True)[:12]
    }

    # Calibration buckets
    cal = {}
    for lo, hi, lbl in [(0, 1.5, "<1.5"), (1.5, 2.5, "1.5-2.5"),
                         (2.5, 3.5, "2.5-3.5"), (3.5, 99, "3.5+")]:
        mask = (pred_sog >= lo) & (pred_sog < hi)
        if mask.sum() > 0:
            cal[lbl] = {
                "n": int(mask.sum()),
                "avg_pred": round(float(pred_sog[mask].mean()), 2),
                "avg_actual": round(float(y_test_actual[mask].mean()), 2),
            }

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "calibration": cal,
        "top_features": feat_imp,
    }

    logger.info("V2 %s model: MAE=%.3f RMSE=%.3f (%d train / %d test)",
                label, mae, rmse, len(train_df), len(test_df))
    return model, metrics


def train_model() -> dict:
    """Train V2 model with expanded features + clustering."""
    global _model_fwd, _model_def, _model_metrics

    logger.info("V2: Building feature matrix...")
    df = _build_feature_dataframe()

    if df.empty or len(df) < 50:
        logger.warning("V2: Not enough data (%d rows)", len(df))
        _model_metrics = {"error": "Not enough data", "n_rows": len(df)}
        return _model_metrics

    # Coverage report
    report = feature_registry.validate_coverage(df, FEATURE_COLS, strict=False)

    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - timedelta(days=14)

    fwd = df[df["position_group"] == "F"]
    dmen = df[df["position_group"] == "D"]

    fwd_train, fwd_test = fwd[fwd["date"] <= cutoff], fwd[fwd["date"] > cutoff]
    def_train, def_test = dmen[dmen["date"] <= cutoff], dmen[dmen["date"] > cutoff]

    fwd_metrics = {"error": "insufficient data"}
    def_metrics = {"error": "insufficient data"}

    if len(fwd_train) >= 50 and len(fwd_test) >= 10:
        _model_fwd, fwd_metrics = _train_single_model(fwd_train, fwd_test, "Forward")
    if len(def_train) >= 50 and len(def_test) >= 10:
        _model_def, def_metrics = _train_single_model(def_train, def_test, "Defense")

    combined_mae = None
    if "mae" in fwd_metrics and "mae" in def_metrics:
        total_test = fwd_metrics["test_samples"] + def_metrics["test_samples"]
        combined_mae = round(
            (fwd_metrics["mae"] * fwd_metrics["test_samples"]
             + def_metrics["mae"] * def_metrics["test_samples"]) / total_test, 3)

    _model_metrics = {
        "mae": combined_mae or fwd_metrics.get("mae") or def_metrics.get("mae"),
        "rmse": None,
        "train_samples": fwd_metrics.get("train_samples", 0) + def_metrics.get("train_samples", 0),
        "test_samples": fwd_metrics.get("test_samples", 0) + def_metrics.get("test_samples", 0),
        "holdout_period": f"{cutoff.strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "model_type": "V2 XGBoost (F/D split + MoneyPuck + clusters)",
        "forward_model": fwd_metrics,
        "defense_model": def_metrics,
        "cluster_k": _clusterer.k,
        "feature_coverage": {k: v["coverage"] for k, v in report.items()},
    }

    save_model()
    return _model_metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_player(player_id: int, opponent_team: str, is_home: bool) -> dict | None:
    """Predict SOG for a player. Same interface as V1."""
    if _model_fwd is None and _model_def is None:
        return None

    conn = data_collector.get_db()
    rows = conn.execute(
        """SELECT pgs.*, g.date FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.player_id = ? ORDER BY g.date DESC""",
        (player_id,),
    ).fetchall()

    if not rows:
        conn.close()
        return None

    latest = rows[0]
    player_name = latest["player_name"]
    team = latest["team"]
    position = latest["position"]
    is_defense = position == "D"
    model = _model_def if is_defense else _model_fwd
    if model is None:
        conn.close()
        return None

    shots_list = [r["shots"] for r in rows]
    toi_list = [r["toi"] for r in rows]
    shifts_list = [r["shifts"] for r in rows]
    pp_list = [(r["pp_goals"] or 0) for r in rows]

    if len(shots_list) < 3:
        conn.close()
        return None

    # Core form
    season_avg = np.mean(shots_list)
    rolling_10 = np.mean(shots_list[:10])
    rolling_5 = np.mean(shots_list[:5])
    baseline = _compute_baseline(season_avg, rolling_10, rolling_5)
    cv = _player_cv.get(player_id, 1.0)
    pct_3plus = sum(1 for s in shots_list if s >= 3) / len(shots_list)
    ewm = float(pd.Series(shots_list[::-1]).ewm(span=10, min_periods=3).mean().iloc[-1])

    # Opportunity
    avg_toi_val = np.mean(toi_list[:20])
    toi_l5 = np.mean(toi_list[:5]) if len(toi_list) >= 5 else avg_toi_val
    toi_l20 = np.mean(toi_list[:20])
    toi_trend = toi_l5 / max(toi_l20, 1.0)

    valid = [(t, s) for t, s in zip(toi_list[:20], shifts_list[:20]) if s and s > 0]
    avg_shift = np.mean([t * 60 / s for t, s in valid]) if valid else 45.0
    pp_rate = np.mean(pp_list[:10])

    # MoneyPuck features
    today = date.today().isoformat()
    style = moneypuck_collector.get_player_shot_style(player_id, before_date=today)
    opp_mp = moneypuck_collector.get_opponent_shot_profile(opponent_team, before_date=today)

    # PP TOI from MoneyPuck
    mp_conn = moneypuck_collector.get_db()
    pp_data = mp_conn.execute(
        """SELECT SUM(CASE WHEN situation='5on4' THEN icetime ELSE 0 END) as pp_ice,
                  SUM(CASE WHEN situation='all' THEN icetime ELSE 0 END) as total_ice
           FROM mp_player_game WHERE player_id = ? AND game_date < ?
           AND game_id IN (SELECT DISTINCT game_id FROM mp_player_game
                           WHERE player_id = ? AND situation='all'
                           ORDER BY game_date DESC LIMIT 10)""",
        (player_id, today, player_id),
    ).fetchone()
    pp_toi_val = (pp_data["pp_ice"] or 0) / max(pp_data["total_ice"] or 1, 1) if pp_data else 0

    # Ice time rank
    itr_row = mp_conn.execute(
        "SELECT ice_time_rank FROM mp_player_game WHERE player_id = ? AND situation='all' ORDER BY game_date DESC LIMIT 1",
        (player_id,),
    ).fetchone()
    itr = itr_row["ice_time_rank"] if itr_row and itr_row["ice_time_rank"] else None
    mp_conn.close()

    # Opponent features
    opp_def = data_collector.build_opponent_defense_profile(opponent_team)
    opp_sa = opp_def.get("shots_allowed_per_game", 30.0) if opp_def else 30.0
    pos_key = f"shots_allowed_to_{position}"
    opp_sa_pos = opp_def.get(pos_key, opp_sa / 18) if opp_def else opp_sa / 18

    opp_hd = opp_mp.get("opp_hd_shots_allowed") if opp_mp else None
    opp_xg = opp_mp.get("opp_xg_allowed") if opp_mp else None
    opp_pace_val = opp_mp.get("opp_pace") if opp_mp else None

    # Game context
    rest = latest["rest_days"] if latest["rest_days"] is not None and latest["rest_days"] >= 0 else 2
    rest = min(rest, 4)
    is_b2b = 1 if rest == 0 else 0
    is_home_val = 1 if is_home else 0

    # Odds
    game_total_val = np.nan
    implied_tt_val = np.nan
    market_line = None
    consensus = None
    sharp_over = None

    try:
        nhl_odds_collector.ensure_todays_odds()
        home_t = team if is_home else opponent_team
        away_t = opponent_team if is_home else team
        odds_ctx = nhl_odds_collector.get_game_context(home_t, away_t)
        if odds_ctx:
            gt = odds_ctx.get("game_total")
            if gt is not None:
                game_total_val = gt
            itt = odds_ctx.get("implied_home_total" if is_home else "implied_away_total")
            if itt is not None:
                implied_tt_val = itt

        consensus = nhl_odds_collector.get_consensus_sog_line(player_name, team=team)
        if consensus:
            market_line = consensus["line"]
            sp = consensus.get("sharp_prob_over")
            if sp is not None:
                sharp_over = sp
    except Exception:
        pass

    # Cluster features
    cluster_id = -1
    cluster_dist = 1.0
    cluster_mean = baseline

    if _clusterer.model is not None and style:
        cluster_input = np.array([
            avg_toi_val, pp_toi_val,
            style.get("slot_pct", 0.3),
            style.get("avg_shot_distance", 30),
            style.get("rebound_rate", 0.05),
            style.get("rush_rate", 0.02),
            style.get("sog", 0) / max(style.get("n_games", 1) * avg_toi_val / 60, 0.01) * 60
            if avg_toi_val > 0 else 5.0,
            avg_shift,
        ])
        try:
            cids, dists = _clusterer.predict(cluster_input)
            cluster_id = int(cids[0])
            cluster_dist = float(dists[0])
            cluster_mean = _clusterer.get_cluster_mean_sog(cluster_id)
        except Exception:
            pass

    # Interaction feature
    opp_hd_for_interaction = opp_hd if opp_hd else 3.0
    cluster_x_opp_hd = cluster_mean * opp_hd_for_interaction / 3.0

    # Build feature vector
    feature_map = {
        "baseline_sog": baseline,
        "player_cv": cv,
        "pct_games_3plus": pct_3plus,
        "rolling_sog_ewm": ewm,
        "avg_toi": avg_toi_val,
        "toi_last_5": toi_l5,
        "toi_trend": toi_trend,
        "avg_shift_length": avg_shift,
        "rolling_pp_rate": pp_rate,
        "pp_toi_pct": pp_toi_val,
        "ice_time_rank": itr if itr else 12,
        "shots_per_60": style.get("sog", 0) / max(style.get("n_games", 1) * avg_toi_val / 60, 0.01) * 60
                        if style and avg_toi_val > 0 else None,
        "slot_pct": style.get("slot_pct") if style else None,
        "rebound_rate": style.get("rebound_rate") if style else None,
        "rush_rate": style.get("rush_rate") if style else None,
        "avg_shot_distance": style.get("avg_shot_distance") if style else None,
        "xg_per_shot": style.get("xg_per_shot") if style else None,
        "high_danger_rate": None,  # Will be filled by registry
        "opp_shots_allowed": opp_sa,
        "opp_shots_allowed_pos": opp_sa_pos,
        "opp_hd_shots_allowed": opp_hd,
        "opp_xg_allowed": opp_xg,
        "opp_pace": opp_pace_val,
        "is_home": is_home_val,
        "rest_days": rest,
        "is_back_to_back": is_b2b,
        "game_total": game_total_val,
        "implied_team_total": implied_tt_val,
        "cluster_id": cluster_id,
        "cluster_distance": cluster_dist,
        "cluster_mean_sog": cluster_mean,
        "cluster_x_opp_hd": cluster_x_opp_hd,
        # Team cluster features
        "opp_team_cluster_ga": None,
        "opp_team_cluster_save": None,
        "player_cluster_x_opp_team_cluster": 0.0,
    }

    # Add team cluster data for opponent
    try:
        import team_clustering
        tp = team_clustering.build_team_profiles(before_date=today, window=60)
        if not tp.empty and len(tp) >= 10:
            _, _, _, tp = team_clustering.fit_team_clusters(tp)
            tc_map = dict(zip(tp["team"], tp["cluster"]))
            opp_cluster = tc_map.get(opponent_team)
            if opp_cluster is not None:
                opp_members = tp[tp["cluster"] == opp_cluster]
                feature_map["opp_team_cluster_ga"] = float(opp_members["ga_per_game"].mean())
                feature_map["opp_team_cluster_save"] = float(opp_members["save_pct"].mean())
                opp_ga_mean = tp["ga_per_game"].mean()
                if opp_ga_mean > 0:
                    feature_map["player_cluster_x_opp_team_cluster"] = (
                        cluster_mean * feature_map["opp_team_cluster_ga"] / opp_ga_mean
                    )
    except Exception:
        pass

    # Use the same feature order as training
    available = [f for f in FEATURE_COLS if f in feature_map]
    features = np.array([[feature_map.get(f, np.nan) for f in available]])

    pred_residual = model.predict(features)[0]
    predicted = max(baseline + pred_residual, 0.2)

    # Blend with market (same as V1: 30% market weight)
    if market_line is not None:
        predicted = 0.70 * predicted + 0.30 * market_line

    # Variance ratio for probability calculation
    var_ratio = cv ** 2 if cv > 0 else 1.0
    var_ratio = max(var_ratio, 0.5)

    conn.close()

    return {
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "opponent": opponent_team,
        "position": position,
        "predicted_sog": round(predicted, 2),
        "baseline_sog": round(baseline, 2),
        "is_home": is_home_val,
        # Form
        "rolling_3": round(np.mean(shots_list[:3]), 2),
        "rolling_5": round(rolling_5, 2),
        "rolling_10": round(rolling_10, 2),
        "rolling_20": round(np.mean(shots_list[:20]), 2),
        "season_avg": round(season_avg, 2),
        "player_cv": round(cv, 3),
        # Opportunity
        "avg_toi": round(avg_toi_val, 1),
        "rest_days": rest,
        # Shot style (V2 exclusive)
        "slot_pct": round(style.get("slot_pct", 0), 3) if style else None,
        "xg_per_shot": round(style.get("xg_per_shot", 0), 4) if style else None,
        "avg_shot_distance": round(style.get("avg_shot_distance", 0), 1) if style else None,
        # Cluster (V2 exclusive)
        "cluster_id": cluster_id,
        "cluster_mean_sog": round(cluster_mean, 2),
        # Opponent
        "opp_shots_allowed": round(opp_sa, 1),
        "opp_shots_allowed_pos": round(opp_sa_pos, 2),
        "opp_hd_shots_allowed": round(opp_hd, 1) if opp_hd else None,
        # Odds (shared with V1)
        "market_sog_line": market_line,
        "game_total": game_total_val if not np.isnan(game_total_val) else None,
        "implied_team_total": implied_tt_val if not np.isnan(implied_tt_val) else None,
        "sharp_prob_over": consensus.get("sharp_prob_over") if consensus else None,
        "sharp_prob_under": consensus.get("sharp_prob_under") if consensus else None,
        "n_sharp_books": consensus.get("n_sharp_books", 0) if consensus else 0,
        "pa_over_est": consensus.get("pa_over_est") if consensus else None,
        "pa_under_est": consensus.get("pa_under_est") if consensus else None,
        "var_ratio": round(var_ratio, 3),
    }


def predict_upcoming_games() -> list[dict]:
    """Predict SOG for all skaters in today's games. Same interface as V1."""
    if _model_fwd is None and _model_def is None:
        return []

    today = date.today().isoformat()
    sched = nhl_api.get_schedule(today)
    if not sched:
        return []

    games = []
    for week in sched.get("gameWeek", []):
        if week.get("date") == today:
            games.extend(week.get("games", []))

    if not games:
        return []

    predictions = []
    conn = data_collector.get_db()

    for game in games:
        home = game.get("homeTeam", {}).get("abbrev", "")
        away = game.get("awayTeam", {}).get("abbrev", "")
        if not home or not away:
            continue

        for team, opp, is_home in [(home, away, True), (away, home, False)]:
            player_rows = conn.execute(
                """SELECT DISTINCT p.player_id, p.player_name, p.position
                   FROM player_game_stats p
                   JOIN (
                       SELECT player_id, team, MAX(game_id) as max_gid
                       FROM player_game_stats
                       GROUP BY player_id
                   ) latest ON p.player_id = latest.player_id
                              AND p.game_id = latest.max_gid
                   WHERE latest.team = ? AND p.position IN ('L', 'C', 'R', 'D')
                     AND (SELECT COUNT(*) FROM player_game_stats
                          WHERE player_id = p.player_id) >= 3
                   ORDER BY (SELECT AVG(shots) FROM player_game_stats
                             WHERE player_id = p.player_id) DESC""",
                (team,),
            ).fetchall()

            for pr in player_rows:
                pred = predict_player(int(pr["player_id"]), opp, is_home)
                if pred:
                    predictions.append(pred)

    conn.close()
    predictions.sort(key=lambda x: x["predicted_sog"], reverse=True)
    return predictions
