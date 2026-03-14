"""
Prediction model for NHL player shots on goal.

Uses XGBoost with features derived from rolling averages, player position,
home/away status, opponent defensive profile, time on ice, power play
involvement, rest days, takeaways, shift patterns, linemate quality,
and player predictability scores.
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import nhl_api
import data_collector

logger = logging.getLogger(__name__)

# Global model state
_model: XGBRegressor | None = None
_feature_cols: list[str] = []
_model_metrics: dict = {}
# Player predictability cache: player_id -> CV
_player_cv: dict[int, float] = {}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_dataframe() -> pd.DataFrame:
    """
    Build a flat DataFrame with one row per player-game.
    Features:
      - rolling avg SOG (3/5/10/20)
      - season avg SOG
      - position dummies (C, L, R, D)
      - is_home
      - opponent shots allowed (overall + position-specific)
      - average TOI
      - avg shift length (TOI / shifts)
      - PP involvement (rolling pp_goals rate as proxy for PP time)
      - rolling takeaways
      - rest days (0=back-to-back, 1, 2, 3+)
      - back-to-back flag
      - player CV (predictability score)
      - TOI * position interaction
      - linemate quality
    Target: shots
    """
    conn = data_collector.get_db()

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

    # Load linemate data
    linemate_df = pd.read_sql_query(
        "SELECT player_id, linemate_id, game_id, shared_toi_seconds FROM linemates",
        conn,
    )

    # Load opponent shot data per game for computing rolling defense profiles
    opp_shots_df = pd.read_sql_query(
        """SELECT pgs.game_id, g.date, pgs.team, pgs.position, pgs.shots,
                  g.home_team, g.away_team
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.position IN ('L', 'C', 'R', 'D')
           ORDER BY g.date""",
        conn,
    )
    conn.close()

    # Build game-date lookup for ordering
    game_dates = df.drop_duplicates("game_id")[["game_id", "date"]].set_index("game_id")["date"].to_dict()

    # Pre-compute per-game opponent defense using only prior games (no leakage)
    # For each game, compute each team's shots-allowed stats from games BEFORE that date
    opp_shots_df["opponent"] = np.where(
        opp_shots_df["team"] == opp_shots_df["home_team"],
        opp_shots_df["away_team"],
        opp_shots_df["home_team"],
    )
    # Group: for each (opponent_team, game_date), what shots did they allow?
    # We need: shots allowed BY a team = shots scored AGAINST that team
    opp_shots_df["defending_team"] = opp_shots_df["opponent"]

    # Sort by date for expanding calculations
    opp_shots_df = opp_shots_df.sort_values("date")

    # Build expanding defense profiles per defending team
    # For each game, compute cumulative shots allowed before that game
    def _build_rolling_defense(opp_df):
        """Build a map: (defending_team, game_date) -> defense profile using only prior games."""
        defense_by_date = {}
        team_history = {}  # team -> list of (date, position, shots)

        for _, row in opp_df.iterrows():
            def_team = row["defending_team"]
            game_date = row["date"]
            pos = row["position"]
            shots = row["shots"]

            # Record the current game's data
            if def_team not in team_history:
                team_history[def_team] = []
            team_history[def_team].append((game_date, pos, shots))

        # Now compute cumulative stats at each date for each team
        for team, history in team_history.items():
            history.sort(key=lambda x: x[0])
            cumul_shots = 0
            cumul_games = set()
            cumul_by_pos = {"C": 0, "L": 0, "R": 0, "D": 0}
            cumul_by_pos_count = {"C": 0, "L": 0, "R": 0, "D": 0}

            prev_date = None
            buffer = []

            for game_date, pos, shots in history:
                if prev_date is not None and game_date != prev_date:
                    # Flush buffer: stats available as of prev_date (for any game AFTER prev_date)
                    n_games = len(cumul_games)
                    if n_games > 0:
                        profile = {
                            "shots_allowed_per_game": cumul_shots / n_games,
                        }
                        for p in ["C", "L", "R", "D"]:
                            cnt = cumul_by_pos_count[p]
                            profile[f"shots_allowed_to_{p}"] = (
                                cumul_by_pos[p] / cnt if cnt > 0
                                else profile["shots_allowed_per_game"] / 4
                            )
                        defense_by_date[(team, game_date)] = profile

                    # Process buffered entries into cumulative
                    for bd, bp, bs in buffer:
                        cumul_shots += bs
                        cumul_games.add(bd)
                        if bp in cumul_by_pos:
                            cumul_by_pos[bp] += bs
                            cumul_by_pos_count[bp] += 1
                    buffer = []

                buffer.append((game_date, pos, shots))
                prev_date = game_date

            # Final flush
            if buffer:
                n_games = len(cumul_games)
                if n_games > 0:
                    last_date = buffer[-1][0]
                    profile = {
                        "shots_allowed_per_game": cumul_shots / n_games,
                    }
                    for p in ["C", "L", "R", "D"]:
                        cnt = cumul_by_pos_count[p]
                        profile[f"shots_allowed_to_{p}"] = (
                            cumul_by_pos[p] / cnt if cnt > 0
                            else profile["shots_allowed_per_game"] / 4
                        )
                    # Use a sentinel for "latest available"
                    defense_by_date[(team, "latest")] = profile

        return defense_by_date

    rolling_defense = _build_rolling_defense(opp_shots_df)

    def _get_defense_at_date(team, game_date):
        """Get the most recent defense profile for a team before game_date."""
        if (team, game_date) in rolling_defense:
            return rolling_defense[(team, game_date)]
        # Fall back to latest available
        return rolling_defense.get((team, "latest"), None)

    # Pre-compute expanding linemate quality (only using prior games' data)
    # player -> sorted list of (date, shots) for expanding avg
    player_shots_by_date = {}
    for _, row in df.sort_values("date").iterrows():
        pid = int(row["player_id"])
        if pid not in player_shots_by_date:
            player_shots_by_date[pid] = []
        player_shots_by_date[pid].append((row["date"], row["shots"]))

    def _get_player_avg_before(player_id, game_date):
        """Get a player's average SOG from games strictly before game_date."""
        history = player_shots_by_date.get(player_id, [])
        prior = [s for d, s in history if d < game_date]
        return np.mean(prior) if prior else 0.0

    linemate_quality_map: dict[tuple[int, int], float] = {}
    if not linemate_df.empty:
        for (pid, gid), grp in linemate_df.groupby(["player_id", "game_id"]):
            game_date = game_dates.get(int(gid), "")
            top_mates = grp.nlargest(3, "shared_toi_seconds")
            mate_avgs = [
                _get_player_avg_before(int(mid), game_date)
                for mid in top_mates["linemate_id"]
            ]
            linemate_quality_map[(int(pid), int(gid))] = (
                np.mean(mate_avgs) if mate_avgs else 0.0
            )

    # Pre-compute expanding player CV (only using prior games, no leakage)
    # Also compute final CV for prediction cache
    global _player_cv
    player_cv_data = df.groupby("player_id")["shots"].agg(["mean", "std", "count"])
    player_cv_data["cv"] = np.where(
        (player_cv_data["mean"] > 0) & (player_cv_data["count"] >= 10),
        player_cv_data["std"] / player_cv_data["mean"],
        1.0,
    )
    _player_cv = player_cv_data["cv"].to_dict()

    def _get_expanding_cv(player_id, game_date):
        """Compute CV using only games before game_date."""
        history = player_shots_by_date.get(player_id, [])
        prior = [s for d, s in history if d < game_date]
        if len(prior) < 10:
            return 1.0  # default high CV for insufficient data
        mean = np.mean(prior)
        if mean <= 0:
            return 1.0
        return float(np.std(prior) / mean)

    # Ensure numeric types for new columns
    for col in ["pp_goals", "shifts", "takeaways", "rest_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Build features per player
    records = []
    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Shifted rolling stats (use only data available BEFORE the game)
        for w in [3, 5, 10, 20]:
            grp[f"rolling_{w}"] = (
                grp["shots"].shift(1).rolling(window=w, min_periods=1).mean()
            )
        grp["season_avg_sog"] = grp["shots"].expanding().mean().shift(1)
        grp["avg_toi"] = grp["toi"].expanding().mean().shift(1)

        # New rolling features
        grp["rolling_pp_rate"] = (
            grp["pp_goals"].shift(1).rolling(window=10, min_periods=1).mean()
        )
        grp["rolling_takeaways"] = (
            grp["takeaways"].shift(1).rolling(window=10, min_periods=1).mean()
        )
        grp["avg_shifts"] = grp["shifts"].expanding().mean().shift(1)

        # Fill NaNs
        fill_cols = (
            [f"rolling_{w}" for w in [3, 5, 10, 20]]
            + ["season_avg_sog", "avg_toi", "rolling_pp_rate",
               "rolling_takeaways", "avg_shifts"]
        )
        for col in fill_cols:
            grp[col] = grp[col].fillna(0)

        for idx, row in grp.iterrows():
            if row["is_home"]:
                opponent = row["away_team"]
            else:
                opponent = row["home_team"]

            pos = row["position"]
            game_date = row["date"]

            # Use rolling defense profile (only prior games)
            opp_def = _get_defense_at_date(opponent, game_date)
            if opp_def is not None:
                opp_sa = opp_def.get("shots_allowed_per_game", 30.0)
                pos_key = f"shots_allowed_to_{pos}"
                opp_sa_pos = opp_def.get(pos_key, opp_sa / 4)
            else:
                opp_sa = 30.0
                opp_sa_pos = 7.5

            lm_quality = linemate_quality_map.get(
                (int(row["player_id"]), int(row["game_id"])), 0.0
            )

            # Expanding CV using only prior games
            cv = _get_expanding_cv(pid, game_date)

            # Avg shift length (minutes per shift)
            avg_shift_len = (
                row["avg_toi"] / row["avg_shifts"]
                if row["avg_shifts"] > 0 else 0.0
            )

            rest = int(row["rest_days"]) if row["rest_days"] >= 0 else 2
            is_b2b = 1 if rest == 0 else 0

            records.append({
                "player_id": row["player_id"],
                "game_id": row["game_id"],
                "date": row["date"],
                "shots": row["shots"],
                # Rolling averages
                "rolling_3": row["rolling_3"],
                "rolling_5": row["rolling_5"],
                "rolling_10": row["rolling_10"],
                "rolling_20": row["rolling_20"],
                "season_avg_sog": row["season_avg_sog"],
                # Position
                "is_C": 1 if pos == "C" else 0,
                "is_L": 1 if pos == "L" else 0,
                "is_R": 1 if pos == "R" else 0,
                "is_D": 1 if pos == "D" else 0,
                # Context
                "is_home": row["is_home"],
                "opp_shots_allowed": opp_sa,
                "opp_shots_allowed_pos": opp_sa_pos,
                # Ice time
                "avg_toi": row["avg_toi"],
                "avg_shift_length": avg_shift_len,
                # New features
                "rolling_pp_rate": row["rolling_pp_rate"],
                "rolling_takeaways": row["rolling_takeaways"],
                "rest_days": min(rest, 4),  # cap at 4+
                "is_back_to_back": is_b2b,
                "player_cv": cv,
                # Interaction
                "toi_x_is_forward": row["avg_toi"] * (1 if pos != "D" else 0),
                # Linemate
                "linemate_quality": lm_quality,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "rolling_3", "rolling_5", "rolling_10", "rolling_20",
    "season_avg_sog", "is_C", "is_L", "is_R", "is_D", "is_home",
    "opp_shots_allowed", "opp_shots_allowed_pos",
    "avg_toi", "avg_shift_length",
    "rolling_pp_rate", "rolling_takeaways",
    "rest_days", "is_back_to_back",
    "player_cv", "toi_x_is_forward",
    "linemate_quality",
]


def train_model() -> dict:
    """
    Train an XGBoost model on all available game data.
    Returns evaluation metrics dict.
    """
    global _model, _feature_cols, _model_metrics

    logger.info("Building feature matrix...")
    df = _build_feature_dataframe()

    if df.empty or len(df) < 50:
        logger.warning("Not enough data to train (%d rows)", len(df))
        _model_metrics = {"error": "Not enough data", "n_rows": len(df)}
        return _model_metrics

    _feature_cols = FEATURE_COLS

    # Hold out last 2 weeks for testing
    from datetime import timedelta
    df["date"] = pd.to_datetime(df["date"])
    cutoff_date = df["date"].max() - timedelta(days=14)
    train_df = df[df["date"] <= cutoff_date]
    test_df = df[df["date"] > cutoff_date]

    if train_df.empty or test_df.empty:
        logger.warning("Not enough data for 2-week holdout split")
        train_df = df
        test_df = df.tail(int(len(df) * 0.2))

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["shots"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["shots"].values

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Feature importance from XGBoost (gain-based)
    importance = model.feature_importances_
    feat_imp = {
        col: round(float(imp), 4)
        for col, imp in sorted(
            zip(FEATURE_COLS, importance),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    _model = model
    _model_metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "holdout_period": f"{cutoff_date.strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "model_type": "XGBoost",
        "feature_importance": feat_imp,
    }
    logger.info(
        "XGBoost trained. MAE=%.3f  RMSE=%.3f  Holdout: last 2 weeks (%d samples)",
        mae, rmse, len(X_test),
    )
    return _model_metrics


def get_model_metrics() -> dict:
    return _model_metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_player(player_id: int, opponent_team: str, is_home: bool) -> dict | None:
    """
    Predict SOG for a player against a given opponent.
    """
    if _model is None:
        return None

    conn = data_collector.get_db()

    rows = conn.execute(
        """SELECT pgs.*, g.date
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.player_id = ?
           ORDER BY g.date DESC
           LIMIT 20""",
        (player_id,),
    ).fetchall()

    if not rows:
        conn.close()
        return None

    latest = rows[0]
    player_name = latest["player_name"]
    team = latest["team"]
    position = latest["position"]
    shots_list = [r["shots"] for r in rows]
    toi_list = [r["toi"] for r in rows]
    pp_goals_list = [r["pp_goals"] or 0 for r in rows]
    takeaways_list = [r["takeaways"] or 0 for r in rows]
    shifts_list = [r["shifts"] or 0 for r in rows]

    def _rolling(data, n):
        subset = data[:n]
        return np.mean(subset) if subset else 0.0

    rolling_3 = _rolling(shots_list, 3)
    rolling_5 = _rolling(shots_list, 5)
    rolling_10 = _rolling(shots_list, 10)
    rolling_20 = _rolling(shots_list, 20)
    season_avg = np.mean(shots_list)
    avg_toi = np.mean(toi_list)
    avg_shifts = np.mean(shifts_list) if shifts_list else 0
    avg_shift_length = avg_toi / avg_shifts if avg_shifts > 0 else 0
    rolling_pp_rate = _rolling(pp_goals_list, 10)
    rolling_takeaways = _rolling(takeaways_list, 10)

    # Rest days: days since last game
    if len(rows) >= 2:
        try:
            from datetime import datetime
            d1 = datetime.strptime(rows[0]["date"][:10], "%Y-%m-%d")
            d2 = datetime.strptime(rows[1]["date"][:10], "%Y-%m-%d")
            rest = max((d1 - d2).days - 1, 0)
        except (ValueError, TypeError):
            rest = 2
    else:
        rest = 2
    rest = min(rest, 4)
    is_b2b = 1 if rest == 0 else 0

    # Player CV
    cv = _player_cv.get(player_id, 1.0)

    # Opponent defense
    opp_row = conn.execute(
        "SELECT * FROM team_defense WHERE team = ?", (opponent_team,)
    ).fetchone()

    opp_sa = opp_row["shots_allowed_per_game"] if opp_row else 30.0
    if opp_row:
        pos_key = f"shots_allowed_to_{position}"
        opp_sa_pos = opp_row[pos_key] if pos_key in opp_row.keys() else opp_sa / 4
    else:
        opp_sa_pos = 7.5

    # Linemate quality
    lm_rows = conn.execute(
        """SELECT lm.linemate_id, lm.shared_toi_seconds
           FROM linemates lm
           WHERE lm.player_id = ?
           ORDER BY lm.game_id DESC
           LIMIT 5""",
        (player_id,),
    ).fetchall()
    conn.close()

    if lm_rows:
        mate_ids = [r["linemate_id"] for r in lm_rows]
        conn2 = data_collector.get_db()
        mate_avgs = []
        for mid in mate_ids[:3]:
            mr = conn2.execute(
                "SELECT AVG(shots) as avg_s FROM player_game_stats WHERE player_id = ?",
                (mid,),
            ).fetchone()
            if mr and mr["avg_s"]:
                mate_avgs.append(mr["avg_s"])
        conn2.close()
        lm_quality = np.mean(mate_avgs) if mate_avgs else 0.0
    else:
        lm_quality = 0.0

    is_forward = 1 if position != "D" else 0

    features = np.array([[
        rolling_3, rolling_5, rolling_10, rolling_20,
        season_avg,
        1 if position == "C" else 0,
        1 if position == "L" else 0,
        1 if position == "R" else 0,
        1 if position == "D" else 0,
        1 if is_home else 0,
        opp_sa, opp_sa_pos,
        avg_toi, avg_shift_length,
        rolling_pp_rate, rolling_takeaways,
        rest, is_b2b,
        cv, avg_toi * is_forward,
        lm_quality,
    ]])

    pred = float(_model.predict(features)[0])
    pred = max(0.0, round(pred, 2))

    return {
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "position": position,
        "opponent": opponent_team,
        "is_home": is_home,
        "predicted_sog": pred,
        "rolling_3": round(rolling_3, 2),
        "rolling_5": round(rolling_5, 2),
        "rolling_10": round(rolling_10, 2),
        "rolling_20": round(rolling_20, 2),
        "season_avg": round(season_avg, 2),
        "avg_toi": round(avg_toi, 1),
        "opp_shots_allowed": opp_sa,
        "opp_shots_allowed_pos": opp_sa_pos,
        "player_cv": round(cv, 3),
        "rest_days": rest,
    }


def predict_upcoming_games() -> list[dict]:
    """
    Get today's schedule and predict SOG for every skater in today's games.
    """
    if _model is None:
        return []

    from datetime import date
    today = date.today().isoformat()
    sched = nhl_api.get_schedule(today)
    if not sched:
        return []

    games = []
    game_weeks = sched.get("gameWeek", [])
    for week in game_weeks:
        week_date = week.get("date", "")
        if week_date != today:
            continue
        for g in week.get("games", []):
            games.append(g)

    if not games:
        return []

    predictions = []
    conn = data_collector.get_db()

    for game in games:
        home_abbrev = game.get("homeTeam", {}).get("abbrev", "")
        away_abbrev = game.get("awayTeam", {}).get("abbrev", "")

        if not home_abbrev or not away_abbrev:
            continue

        for team, opp, is_home in [
            (home_abbrev, away_abbrev, True),
            (away_abbrev, home_abbrev, False),
        ]:
            player_rows = conn.execute(
                """SELECT DISTINCT player_id, player_name, position
                   FROM player_game_stats
                   WHERE team = ? AND position IN ('L', 'C', 'R', 'D')
                   GROUP BY player_id
                   HAVING COUNT(*) >= 3
                   ORDER BY AVG(shots) DESC""",
                (team,),
            ).fetchall()

            for pr in player_rows:
                pred = predict_player(int(pr["player_id"]), opp, is_home)
                if pred:
                    predictions.append(pred)

    conn.close()
    predictions.sort(key=lambda x: x["predicted_sog"], reverse=True)
    return predictions
