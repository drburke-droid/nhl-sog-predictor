"""
Prediction model for NHL player shots on goal.

Uses separate XGBoost models for forwards and defensemen, trained on
residual targets (actual SOG minus a stable player baseline). This avoids
the downward bias that occurs when one model covers all NHL players.

Key design choices (per nhl_sog_model_guide.md):
  - Baseline = 0.55*season + 0.30*last10 + 0.15*last5
  - Target = actual_sog - baseline (residual)
  - Training filtered to market-relevant players
  - Separate forward / defense models
  - Time-based validation (last 2 weeks holdout)
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import nhl_api
import data_collector

try:
    import nhl_odds_collector
    HAS_ODDS = True
except ImportError:
    HAS_ODDS = False

MODEL_DIR = Path(__file__).parent / "saved_model"
MODEL_VERSION = 3  # v3: added sharp_consensus_prob feature

logger = logging.getLogger(__name__)

# Global model state — separate models for forwards and defensemen
_model_fwd: XGBRegressor | None = None
_model_def: XGBRegressor | None = None
_model_metrics: dict = {}
# Player predictability cache: player_id -> CV and variance/mean ratio
_player_cv: dict[int, float] = {}
_player_var_ratio: dict[int, float] = {}


def save_model():
    """Save trained models and metadata to disk for fast cold starts."""
    MODEL_DIR.mkdir(exist_ok=True)
    if _model_fwd is not None:
        _model_fwd.save_model(str(MODEL_DIR / "model_fwd.json"))
    if _model_def is not None:
        _model_def.save_model(str(MODEL_DIR / "model_def.json"))
    meta = {
        "model_version": MODEL_VERSION,
        "metrics": _model_metrics,
        "player_cv": {str(k): v for k, v in _player_cv.items()},
        "player_var_ratio": {str(k): v for k, v in _player_var_ratio.items()},
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f)
    logger.info("Model saved to %s", MODEL_DIR)


def load_model() -> bool:
    """Load saved models from disk. Returns True if successful."""
    global _model_fwd, _model_def, _model_metrics, _player_cv, _player_var_ratio
    fwd_path = MODEL_DIR / "model_fwd.json"
    def_path = MODEL_DIR / "model_def.json"
    meta_path = MODEL_DIR / "meta.json"

    if not fwd_path.exists() or not meta_path.exists():
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        # Check model version compatibility
        saved_version = meta.get("model_version", 1)
        if saved_version != MODEL_VERSION:
            logger.info(
                "Model version mismatch (saved=%d, current=%d) — retrain needed",
                saved_version, MODEL_VERSION,
            )
            return False

        _model_fwd = XGBRegressor()
        _model_fwd.load_model(str(fwd_path))
        if def_path.exists():
            _model_def = XGBRegressor()
            _model_def.load_model(str(def_path))

        _model_metrics = meta.get("metrics", {})
        _player_cv = {int(k): v for k, v in meta.get("player_cv", {}).items()}
        _player_var_ratio = {int(k): v for k, v in meta.get("player_var_ratio", {}).items()}
        logger.info("Loaded saved model (MAE: %s)", _model_metrics.get("mae"))
        return True
    except Exception as exc:
        logger.warning("Failed to load saved model: %s", exc)
        _model_fwd = None
        _model_def = None
        return False


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def _compute_baseline(season_avg, rolling_10, rolling_5):
    """Weighted baseline: 55% season + 30% last10 + 15% last5."""
    return 0.55 * season_avg + 0.30 * rolling_10 + 0.15 * rolling_5


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Baseline (already incorporates season + rolling 10 + rolling 5)
    "baseline_sog",
    # Context
    "is_home",
    "opp_shots_allowed", "opp_shots_allowed_pos",
    # Usage
    "avg_toi", "toi_last_5",
    "avg_shift_length",
    "rolling_pp_rate",
    # Form / volatility
    "player_cv",
    "pct_games_3plus",
    # Game context
    "rest_days", "is_back_to_back",
    # Linemate
    "linemate_quality",
    # Venue bias (scorekeeper generosity)
    "arena_bias",
    # Odds-derived (NaN when not available — XGBoost handles natively)
    "game_total",           # over/under on total goals (pace proxy)
    "implied_team_total",   # team's expected goals from ML + total
    "sog_prop_line",        # market consensus player SOG line
    "sharp_consensus_prob", # vig-free over prob from sharp books (betonline/dk/fd)
]


def _build_feature_dataframe() -> pd.DataFrame:
    """
    Build a flat DataFrame with one row per player-game.
    All features use only pre-game information (no leakage).
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

    # Load opponent shot data for rolling defense profiles
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

    # Game-date lookup
    game_dates = (
        df.drop_duplicates("game_id")[["game_id", "date"]]
        .set_index("game_id")["date"]
        .to_dict()
    )

    # --- Rolling defense profiles (no leakage) ---
    opp_shots_df["opponent"] = np.where(
        opp_shots_df["team"] == opp_shots_df["home_team"],
        opp_shots_df["away_team"],
        opp_shots_df["home_team"],
    )
    opp_shots_df["defending_team"] = opp_shots_df["opponent"]
    opp_shots_df = opp_shots_df.sort_values("date")

    def _build_rolling_defense(opp_df):
        defense_by_date = {}
        team_history = {}

        for _, row in opp_df.iterrows():
            def_team = row["defending_team"]
            if def_team not in team_history:
                team_history[def_team] = []
            team_history[def_team].append(
                (row["date"], row["position"], row["shots"])
            )

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

                    for bd, bp, bs in buffer:
                        cumul_shots += bs
                        cumul_games.add(bd)
                        if bp in cumul_by_pos:
                            cumul_by_pos[bp] += bs
                            cumul_by_pos_count[bp] += 1
                    buffer = []

                buffer.append((game_date, pos, shots))
                prev_date = game_date

            if buffer:
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
                    defense_by_date[(team, "latest")] = profile

        return defense_by_date

    rolling_defense = _build_rolling_defense(opp_shots_df)

    # --- Arena bias: per-player SOG deviation from league average by arena ---
    # Group by (home_team, game_id) to get total per-player SOG per game in each arena
    arena_game_df = pd.read_sql_query(
        """SELECT g.home_team as arena, g.game_id, g.date,
                  AVG(pgs.shots) as avg_player_sog
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.position IN ('L', 'C', 'R', 'D')
           GROUP BY g.game_id""",
        data_collector.get_db(),
    ).sort_values("date")

    # Build expanding arena bias (only prior games, no leakage)
    def _build_arena_bias(arena_df):
        """Return dict of (arena, game_date) -> bias (arena avg - league avg)."""
        bias_by_date = {}
        arena_history = {}   # arena -> list of per-player avg SOG
        league_history = []  # all per-player avg SOG values

        prev_date = None
        date_buffer = []

        for _, row in arena_df.iterrows():
            game_date = row["date"]
            arena = row["arena"]
            avg_sog = row["avg_player_sog"]

            if prev_date is not None and game_date != prev_date:
                # Compute bias for each arena using data before this date
                if len(league_history) >= 50:
                    league_avg = np.mean(league_history)
                    for a, hist in arena_history.items():
                        if len(hist) >= 5:
                            bias_by_date[(a, game_date)] = np.mean(hist) - league_avg
                # Flush buffer
                for buf_arena, buf_sog in date_buffer:
                    if buf_arena not in arena_history:
                        arena_history[buf_arena] = []
                    arena_history[buf_arena].append(buf_sog)
                    league_history.append(buf_sog)
                date_buffer = []

            date_buffer.append((arena, avg_sog))
            prev_date = game_date

        # Final flush for "latest" lookup
        if date_buffer:
            for buf_arena, buf_sog in date_buffer:
                if buf_arena not in arena_history:
                    arena_history[buf_arena] = []
                arena_history[buf_arena].append(buf_sog)
                league_history.append(buf_sog)
        if len(league_history) >= 50:
            league_avg = np.mean(league_history)
            for a, hist in arena_history.items():
                if len(hist) >= 5:
                    bias_by_date[(a, "latest")] = np.mean(hist) - league_avg

        return bias_by_date

    arena_bias_map = _build_arena_bias(arena_game_df)

    # --- Odds data (optional — NaN when not available) ---
    odds_game_map = {}   # (game_date, home_team_abbrev) -> {game_total, implied_*}
    odds_prop_map = {}   # (game_date, player_name) -> consensus SOG line
    sharp_map = {}       # (game_date, player_name) -> {sharp_prob_over, ...}

    if HAS_ODDS:
        try:
            odds_game_map = nhl_odds_collector.load_game_odds_bulk()
            odds_prop_map = nhl_odds_collector.load_player_props_bulk()
            sharp_map = nhl_odds_collector.load_sharp_consensus_bulk()
            logger.info(
                "Loaded odds data: %d games, %d player props, %d sharp consensus",
                len(odds_game_map), len(odds_prop_map), len(sharp_map),
            )
        except Exception as exc:
            logger.warning("Could not load odds data: %s", exc)

    def _get_arena_bias(arena, game_date):
        if (arena, game_date) in arena_bias_map:
            return arena_bias_map[(arena, game_date)]
        return arena_bias_map.get((arena, "latest"), 0.0)

    def _get_defense_at_date(team, game_date):
        if (team, game_date) in rolling_defense:
            return rolling_defense[(team, game_date)]
        return rolling_defense.get((team, "latest"), None)

    # --- Expanding linemate quality (no leakage) ---
    player_shots_by_date = {}
    for _, row in df.sort_values("date").iterrows():
        pid = int(row["player_id"])
        if pid not in player_shots_by_date:
            player_shots_by_date[pid] = []
        player_shots_by_date[pid].append((row["date"], row["shots"]))

    def _get_player_avg_before(player_id, game_date):
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

    # --- Expanding player CV and variance ratio (no leakage) ---
    global _player_cv, _player_var_ratio
    player_cv_data = df.groupby("player_id")["shots"].agg(["mean", "std", "count", "var"])
    player_cv_data["cv"] = np.where(
        (player_cv_data["mean"] > 0) & (player_cv_data["count"] >= 10),
        player_cv_data["std"] / player_cv_data["mean"],
        1.0,
    )
    player_cv_data["var_ratio"] = np.where(
        (player_cv_data["mean"] > 0) & (player_cv_data["count"] >= 10),
        player_cv_data["var"] / player_cv_data["mean"],
        1.0,  # 1.0 = Poisson assumption
    )
    _player_cv = player_cv_data["cv"].to_dict()
    _player_var_ratio = player_cv_data["var_ratio"].to_dict()

    def _get_expanding_cv(player_id, game_date):
        history = player_shots_by_date.get(player_id, [])
        prior = [s for d, s in history if d < game_date]
        if len(prior) < 10:
            return 1.0
        mean = np.mean(prior)
        if mean <= 0:
            return 1.0
        return float(np.std(prior) / mean)

    # Ensure numeric types
    for col in ["pp_goals", "shifts", "takeaways", "rest_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # --- Build features per player ---
    records = []
    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Shifted rolling stats (pre-game only)
        for w in [3, 5, 10, 20]:
            grp[f"rolling_{w}"] = (
                grp["shots"].shift(1).rolling(window=w, min_periods=1).mean()
            )
        grp["season_avg_sog"] = grp["shots"].expanding().mean().shift(1)
        grp["avg_toi"] = grp["toi"].expanding().mean().shift(1)

        # TOI rolling windows
        grp["toi_last_5"] = (
            grp["toi"].shift(1).rolling(window=5, min_periods=1).mean()
        )

        # PP rate and takeaways
        grp["rolling_pp_rate"] = (
            grp["pp_goals"].shift(1).rolling(window=10, min_periods=1).mean()
        )
        grp["rolling_takeaways"] = (
            grp["takeaways"].shift(1).rolling(window=10, min_periods=1).mean()
        )
        grp["avg_shifts"] = grp["shifts"].expanding().mean().shift(1)

        # Pct of prior games with 3+ SOG
        grp["pct_games_3plus"] = (
            (grp["shots"] >= 3).astype(float).shift(1)
            .expanding(min_periods=1).mean()
        )

        # Fill NaNs
        fill_cols = (
            [f"rolling_{w}" for w in [3, 5, 10, 20]]
            + ["season_avg_sog", "avg_toi", "toi_last_5",
               "rolling_pp_rate", "rolling_takeaways", "avg_shifts",
               "pct_games_3plus"]
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

            # Baseline
            baseline = _compute_baseline(
                row["season_avg_sog"], row["rolling_10"], row["rolling_5"]
            )

            # Relative-to-self deltas
            sog_l5_minus_season = row["rolling_5"] - row["season_avg_sog"]
            sog_l10_minus_season = row["rolling_10"] - row["season_avg_sog"]
            toi_l5_minus_season = row["toi_last_5"] - row["avg_toi"]

            # Rolling defense (prior games only)
            opp_def = _get_defense_at_date(opponent, game_date)
            if opp_def is not None:
                opp_sa = opp_def.get("shots_allowed_per_game", 30.0)
                pos_key = f"shots_allowed_to_{pos}"
                opp_sa_pos = opp_def.get(pos_key, opp_sa / 4)
            else:
                opp_sa = 30.0
                opp_sa_pos = 1.5

            lm_quality = linemate_quality_map.get(
                (int(row["player_id"]), int(row["game_id"])), 0.0
            )

            cv = _get_expanding_cv(pid, game_date)

            # Arena bias (home_team = arena)
            arena = row["home_team"]
            arena_bias = _get_arena_bias(arena, game_date)

            avg_shift_len = (
                row["avg_toi"] / row["avg_shifts"]
                if row["avg_shifts"] > 0 else 0.0
            )

            rest = int(row["rest_days"]) if row["rest_days"] >= 0 else 2
            rest = min(rest, 4)
            is_b2b = 1 if rest == 0 else 0

            is_home_val = int(row["is_home"])

            # Odds features (NaN when not available)
            game_total = float("nan")
            implied_team_total = float("nan")
            sog_prop_line = float("nan")

            odds_key = (str(game_date)[:10], arena)  # arena = home_team
            odds_ctx = odds_game_map.get(odds_key)
            if odds_ctx:
                gt = odds_ctx.get("game_total")
                if gt is not None:
                    game_total = gt
                if is_home_val:
                    itt = odds_ctx.get("implied_home_total")
                else:
                    itt = odds_ctx.get("implied_away_total")
                if itt is not None:
                    implied_team_total = itt

            prop_key = (str(game_date)[:10], row["player_name"])
            prop_val = odds_prop_map.get(prop_key)
            if prop_val is not None:
                sog_prop_line = prop_val

            sharp_consensus_prob = float("nan")
            sharp_entry = sharp_map.get(prop_key)
            if sharp_entry is not None:
                sharp_consensus_prob = sharp_entry["sharp_prob_over"]

            records.append({
                "player_id": row["player_id"],
                "game_id": row["game_id"],
                "date": row["date"],
                "position": pos,
                "position_group": "D" if pos == "D" else "F",
                "shots": row["shots"],
                "toi": row["toi"],
                # Baseline and target
                "baseline_sog": baseline,
                "sog_residual": row["shots"] - baseline,
                # Rolling averages
                "rolling_5": row["rolling_5"],
                "rolling_10": row["rolling_10"],
                "rolling_20": row["rolling_20"],
                "season_avg_sog": row["season_avg_sog"],
                # Relative-to-self
                "sog_last5_minus_season": sog_l5_minus_season,
                "sog_last10_minus_season": sog_l10_minus_season,
                "toi_last5_minus_season": toi_l5_minus_season,
                # Context
                "is_home": is_home_val,
                "opp_shots_allowed": opp_sa,
                "opp_shots_allowed_pos": opp_sa_pos,
                # Usage
                "avg_toi": row["avg_toi"],
                "toi_last_5": row["toi_last_5"],
                "avg_shift_length": avg_shift_len,
                "rolling_pp_rate": row["rolling_pp_rate"],
                # Form / volatility
                "player_cv": cv,
                "pct_games_3plus": row["pct_games_3plus"],
                # Game context
                "rest_days": rest,
                "is_back_to_back": is_b2b,
                # Linemate
                "linemate_quality": lm_quality,
                # Venue
                "arena_bias": arena_bias,
                # Odds
                "game_total": game_total,
                "implied_team_total": implied_team_total,
                "sog_prop_line": sog_prop_line,
                "sharp_consensus_prob": sharp_consensus_prob,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Market-relevant filters
MIN_GAMES = 20
FWD_MIN_TOI = 14.0
FWD_MIN_SOG = 1.6
DEF_MIN_TOI = 18.0
DEF_MIN_SOG = 1.0


def _compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute sample weights combining recency and market relevance.
    Keeps all data but upweights recent games and market-relevant players.
    """
    # Recency weighting: linear decay from 1.0 (most recent) to 0.5 (oldest)
    dates = pd.to_datetime(df["date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days
    max_days = max(days_ago.max(), 1)
    recency_weight = 1.0 - 0.5 * (days_ago / max_days)

    # Market relevance: upweight players with higher baseline SOG
    # Mild boost: players above 2.0 baseline get 1.0-1.5x, below get 0.7-1.0x
    baseline = df["baseline_sog"].values
    relevance_weight = np.clip(0.5 + baseline * 0.25, 0.7, 1.5)

    return (recency_weight.values * relevance_weight).astype(float)


def _train_single_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str,
) -> tuple[XGBRegressor, dict]:
    """Train one XGBoost model on residual target. Returns (model, metrics)."""
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["sog_residual"].values
    X_test = test_df[FEATURE_COLS].values
    y_test_residual = test_df["sog_residual"].values
    y_test_actual = test_df["shots"].values
    baseline_test = test_df["baseline_sog"].values

    sample_weights = _compute_sample_weights(train_df)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test_residual)],
        verbose=False,
    )

    # Reconstruct predictions: baseline + predicted_residual
    pred_residual = model.predict(X_test)
    pred_sog = baseline_test + pred_residual
    pred_sog = np.maximum(pred_sog, 0.0)

    mae = mean_absolute_error(y_test_actual, pred_sog)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_sog))

    # Calibration by bucket
    test_eval = pd.DataFrame({
        "actual": y_test_actual,
        "predicted": pred_sog,
        "baseline": baseline_test,
    })
    calibration = {}
    for lo, hi, lbl in [(0, 1.5, "<1.5"), (1.5, 2.5, "1.5-2.5"),
                         (2.5, 3.5, "2.5-3.5"), (3.5, 99, "3.5+")]:
        mask = (test_eval["baseline"] >= lo) & (test_eval["baseline"] < hi)
        bucket = test_eval[mask]
        if len(bucket) > 0:
            calibration[lbl] = {
                "n": len(bucket),
                "avg_pred": round(bucket["predicted"].mean(), 2),
                "avg_actual": round(bucket["actual"].mean(), 2),
            }

    # Threshold accuracy (over 2.5 and 3.5)
    threshold_acc = {}
    for thresh in [2.5, 3.5]:
        pred_over = pred_sog >= thresh
        actual_over = y_test_actual >= thresh
        if actual_over.sum() > 0:
            correct = (pred_over == actual_over).sum()
            threshold_acc[f"over_{thresh}"] = round(correct / len(pred_sog), 3)

    # Feature importance
    importance = model.feature_importances_
    feat_imp = {
        col: round(float(imp), 4)
        for col, imp in sorted(
            zip(FEATURE_COLS, importance),
            key=lambda x: x[1],
            reverse=True,
        )[:10]  # Top 10
    }

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "calibration": calibration,
        "threshold_accuracy": threshold_acc,
        "top_features": feat_imp,
    }

    logger.info(
        "%s model: MAE=%.3f RMSE=%.3f (%d train / %d test)",
        label, mae, rmse, len(X_train), len(X_test),
    )

    return model, metrics


def train_model() -> dict:
    """
    Train separate forward and defense XGBoost models on residual target.
    Returns evaluation metrics dict.
    """
    global _model_fwd, _model_def, _model_metrics

    logger.info("Building feature matrix...")
    df = _build_feature_dataframe()

    if df.empty or len(df) < 50:
        logger.warning("Not enough data to train (%d rows)", len(df))
        _model_metrics = {"error": "Not enough data", "n_rows": len(df)}
        return _model_metrics

    # Use all data with sample weighting (not hard filtering)
    # Market-relevant players get higher weight, low-usage players stay
    # to provide more training signal without distorting the model.
    df["date"] = pd.to_datetime(df["date"])
    cutoff_date = df["date"].max() - timedelta(days=14)

    fwd = df[df["position_group"] == "F"]
    dmen = df[df["position_group"] == "D"]

    fwd_train = fwd[fwd["date"] <= cutoff_date]
    fwd_test = fwd[fwd["date"] > cutoff_date]
    def_train = dmen[dmen["date"] <= cutoff_date]
    def_test = dmen[dmen["date"] > cutoff_date]

    # Train forward model
    if len(fwd_train) >= 50 and len(fwd_test) >= 10:
        _model_fwd, fwd_metrics = _train_single_model(
            fwd_train, fwd_test, "Forward"
        )
    else:
        logger.warning("Not enough forward data for split model")
        _model_fwd = None
        fwd_metrics = {"error": "insufficient data"}

    # Train defense model
    if len(def_train) >= 50 and len(def_test) >= 10:
        _model_def, def_metrics = _train_single_model(
            def_train, def_test, "Defense"
        )
    else:
        logger.warning("Not enough defense data for split model")
        _model_def = None
        def_metrics = {"error": "insufficient data"}

    # Combined metrics
    combined_mae = None
    if "mae" in fwd_metrics and "mae" in def_metrics:
        total_test = fwd_metrics["test_samples"] + def_metrics["test_samples"]
        combined_mae = round(
            (fwd_metrics["mae"] * fwd_metrics["test_samples"]
             + def_metrics["mae"] * def_metrics["test_samples"]) / total_test,
            3,
        )

    _model_metrics = {
        "mae": combined_mae or fwd_metrics.get("mae") or def_metrics.get("mae"),
        "rmse": None,
        "train_samples": (
            fwd_metrics.get("train_samples", 0)
            + def_metrics.get("train_samples", 0)
        ),
        "test_samples": (
            fwd_metrics.get("test_samples", 0)
            + def_metrics.get("test_samples", 0)
        ),
        "holdout_period": f"{cutoff_date.strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "model_type": "XGBoost (F/D split, residual target)",
        "forward_model": fwd_metrics,
        "defense_model": def_metrics,
    }

    save_model()
    return _model_metrics


def get_model_metrics() -> dict:
    return _model_metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_player(player_id: int, opponent_team: str, is_home: bool) -> dict | None:
    """Predict SOG for a player against a given opponent."""
    if _model_fwd is None and _model_def is None:
        return None

    conn = data_collector.get_db()

    rows = conn.execute(
        """SELECT pgs.*, g.date
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.player_id = ?
           ORDER BY g.date DESC""",
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

    # Pick the right model
    model = _model_def if is_defense else _model_fwd
    if model is None:
        conn.close()
        return None

    shots_list = [r["shots"] for r in rows]
    toi_list = [r["toi"] for r in rows]
    pp_goals_list = [r["pp_goals"] or 0 for r in rows]
    shifts_list = [r["shifts"] or 0 for r in rows]

    def _rolling(data, n):
        subset = data[:n]
        return np.mean(subset) if subset else 0.0

    rolling_5 = _rolling(shots_list, 5)
    rolling_10 = _rolling(shots_list, 10)
    rolling_20 = _rolling(shots_list, 20)
    season_avg = np.mean(shots_list)  # All games, matches training
    avg_toi = np.mean(toi_list)
    toi_last_5 = _rolling(toi_list, 5)
    avg_shifts = np.mean(shifts_list) if shifts_list else 0
    avg_shift_length = avg_toi / avg_shifts if avg_shifts > 0 else 0
    rolling_pp_rate = _rolling(pp_goals_list, 10)

    # Baseline
    baseline = _compute_baseline(season_avg, rolling_10, rolling_5)

    # Relative-to-self deltas
    sog_l5_minus_season = rolling_5 - season_avg
    sog_l10_minus_season = rolling_10 - season_avg
    toi_l5_minus_season = toi_last_5 - avg_toi

    # Pct games with 3+ SOG
    pct_3plus = sum(1 for s in shots_list if s >= 3) / len(shots_list)

    # Rest days
    if len(rows) >= 2:
        try:
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

    # Opponent defense — per-player averages to match training scale
    opp_stats_pos = conn.execute(
        """SELECT AVG(pgs.shots) as avg_shots_per_player
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE ((g.home_team = ? AND pgs.is_home = 0)
              OR  (g.away_team = ? AND pgs.is_home = 1))
             AND pgs.position = ?""",
        (opponent_team, opponent_team, position),
    ).fetchone()

    opp_row = conn.execute(
        "SELECT shots_allowed_per_game FROM team_defense WHERE team = ?",
        (opponent_team,),
    ).fetchone()
    opp_sa = opp_row["shots_allowed_per_game"] if opp_row else 30.0
    opp_sa_pos = (
        opp_stats_pos["avg_shots_per_player"]
        if opp_stats_pos and opp_stats_pos["avg_shots_per_player"]
        else 1.5
    )

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

    is_home_val = 1 if is_home else 0

    # Arena bias — arena is always the home team
    arena_team = team if is_home else opponent_team
    conn3 = data_collector.get_db()
    arena_row = conn3.execute(
        """SELECT g.home_team, AVG(pgs.shots) as avg_player_sog
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.position IN ('L', 'C', 'R', 'D')
           GROUP BY g.home_team""",
        [],
    ).fetchall()
    league_avg_sog = conn3.execute(
        """SELECT AVG(pgs.shots) as avg_sog
           FROM player_game_stats pgs
           WHERE pgs.position IN ('L', 'C', 'R', 'D')""",
    ).fetchone()
    conn3.close()
    league_avg = league_avg_sog["avg_sog"] if league_avg_sog else 2.0
    arena_avg_map = {r["home_team"]: r["avg_player_sog"] for r in arena_row}
    arena_bias = arena_avg_map.get(arena_team, league_avg) - league_avg

    # --- Odds features ---
    game_total = float("nan")
    implied_team_total = float("nan")
    sog_prop_line = float("nan")
    sharp_consensus_prob = float("nan")
    market_line = None  # For post-prediction blending
    consensus = None  # Full consensus dict for return

    if HAS_ODDS:
        try:
            nhl_odds_collector.ensure_todays_odds()

            home_team = team if is_home else opponent_team
            away_team = opponent_team if is_home else team
            odds_ctx = nhl_odds_collector.get_game_context(
                home_team, away_team
            )
            if odds_ctx:
                gt = odds_ctx.get("game_total")
                if gt is not None:
                    game_total = gt
                if is_home:
                    itt = odds_ctx.get("implied_home_total")
                else:
                    itt = odds_ctx.get("implied_away_total")
                if itt is not None:
                    implied_team_total = itt

            consensus = nhl_odds_collector.get_consensus_sog_line(player_name)
            if consensus:
                sog_prop_line = consensus["line"]
                market_line = consensus["line"]
                sp = consensus.get("sharp_prob_over")
                if sp is not None:
                    sharp_consensus_prob = sp
        except Exception as exc:
            logger.debug("Could not fetch odds features: %s", exc)

    features = np.array([[
        baseline,
        is_home_val,
        opp_sa, opp_sa_pos,
        avg_toi, toi_last_5,
        avg_shift_length,
        rolling_pp_rate,
        cv,
        pct_3plus,
        rest, is_b2b,
        lm_quality,
        arena_bias,
        game_total,
        implied_team_total,
        sog_prop_line,
        sharp_consensus_prob,
    ]])

    # Predict residual, reconstruct final SOG
    # Cap residual to ±25% of baseline to prevent extreme predictions
    pred_residual = float(model.predict(features)[0])
    max_adj = max(baseline * 0.25, 0.5)  # at least ±0.5
    pred_residual = np.clip(pred_residual, -max_adj, max_adj)
    pred_sog = baseline + pred_residual
    pred_sog = max(0.0, round(pred_sog, 2))

    # Blend with market line when available (30% market weight)
    MARKET_BLEND_WEIGHT = 0.3
    if market_line is not None:
        pred_sog = round(
            (1 - MARKET_BLEND_WEIGHT) * pred_sog
            + MARKET_BLEND_WEIGHT * market_line,
            2,
        )
        pred_sog = max(0.0, pred_sog)

    return {
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "position": position,
        "opponent": opponent_team,
        "is_home": is_home,
        "predicted_sog": pred_sog,
        "baseline_sog": round(baseline, 2),
        "rolling_3": round(_rolling(shots_list, 3), 2),
        "rolling_5": round(rolling_5, 2),
        "rolling_10": round(rolling_10, 2),
        "rolling_20": round(rolling_20, 2),
        "season_avg": round(season_avg, 2),
        "avg_toi": round(avg_toi, 1),
        "opp_shots_allowed": opp_sa,
        "opp_shots_allowed_pos": round(opp_sa_pos, 2),
        "player_cv": round(cv, 3),
        "var_ratio": round(_player_var_ratio.get(player_id, 1.0), 3),
        "rest_days": rest,
        # Odds data
        "market_sog_line": market_line,
        "game_total": game_total if not np.isnan(game_total) else None,
        "implied_team_total": (
            round(implied_team_total, 2)
            if not np.isnan(implied_team_total) else None
        ),
        # Sharp consensus (vig-free from BetOnline/DK/FanDuel)
        "sharp_prob_over": (
            consensus.get("sharp_prob_over") if consensus else None
        ),
        "sharp_prob_under": (
            consensus.get("sharp_prob_under") if consensus else None
        ),
        "n_sharp_books": (
            consensus.get("n_sharp_books", 0) if consensus else 0
        ),
        # PlayAlberta estimated odds (derived from BetMGM)
        "pa_over_est": consensus.get("pa_over_est") if consensus else None,
        "pa_under_est": consensus.get("pa_under_est") if consensus else None,
    }


# ---------------------------------------------------------------------------
# Prediction history — save, score, and compute confidence
# ---------------------------------------------------------------------------

def _poisson_cdf_py(lam: float, k: int) -> float:
    """P(X <= k) for Poisson(lam)."""
    if lam <= 0:
        return 1.0
    s = 0.0
    log_p = -lam
    s += np.exp(log_p)
    for i in range(1, k + 1):
        log_p += np.log(lam) - np.log(i)
        s += np.exp(log_p)
    return min(s, 1.0)


def _negbin_cdf_py(r: float, p: float, k: int) -> float:
    """P(X <= k) for NegBin(r, p)."""
    from math import lgamma
    s = 0.0
    for i in range(k + 1):
        log_pmf = (lgamma(r + i) - lgamma(r) - lgamma(i + 1)
                   + r * np.log(p) + i * np.log(1 - p))
        s += np.exp(log_pmf)
    return min(s, 1.0)


def calc_prob_over(pred_sog: float, line: float, var_ratio: float = 1.0) -> float:
    """P(X > line) using NegBin (or Poisson when var_ratio ≈ 1)."""
    if pred_sog <= 0:
        return 0.0
    k = int(line)
    if var_ratio <= 1.05:
        return 1 - _poisson_cdf_py(pred_sog, k)
    r = pred_sog / (var_ratio - 1)
    p = 1 / var_ratio
    if r <= 0 or p <= 0 or p >= 1:
        return 1 - _poisson_cdf_py(pred_sog, k)
    return 1 - _negbin_cdf_py(r, p, k)


def _american_to_decimal_profit(odds: int) -> float:
    if not odds:
        return 0.0
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def _american_to_implied_prob(odds: int) -> float:
    if not odds:
        return 0.0
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _kelly_fraction(prob_win: float, american_odds: int, fraction: float = 0.25) -> float:
    b = _american_to_decimal_profit(american_odds)
    if b <= 0:
        return 0.0
    q = 1 - prob_win
    f = (b * prob_win - q) / b
    if f <= 0:
        return 0.0
    return f * fraction


def save_predictions_to_history(predictions: list[dict], odds_map: dict,
                                bankroll: float) -> int:
    """Save today's predictions + odds to prediction_history. Returns count saved."""
    from datetime import date
    today = date.today().isoformat()
    conn = data_collector.get_db()

    saved = 0
    for p in predictions:
        pid = p["player_id"]
        odds = odds_map.get(pid, {})
        sog_line = odds.get("line")
        over_odds = odds.get("over_odds")
        under_odds = odds.get("under_odds")

        # Compute bet side + Kelly server-side
        bet_side = None
        bet_kelly = 0.0
        var_ratio = p.get("var_ratio", 1.0)
        cv = p.get("player_cv", 1.0)
        variance = p.get("predicted_sog", 0) - p.get("season_avg", 0)
        signal = abs(variance) / cv if cv > 0 else 0
        confidence = min(signal / 1.5, 1.0)

        if sog_line is not None:
            prob_over = calc_prob_over(p["predicted_sog"], sog_line, var_ratio)
            prob_under = 1 - prob_over

            kelly_over = _kelly_fraction(prob_over, over_odds) if over_odds else 0
            kelly_under = _kelly_fraction(prob_under, under_odds) if under_odds else 0

            best_kelly = max(kelly_over, kelly_under)
            adj_kelly = best_kelly * confidence

            if adj_kelly > 0:
                bet_side = "OVER" if kelly_over >= kelly_under else "UNDER"
                bet_kelly = adj_kelly

        bet_amount = round(bankroll * bet_kelly) if bankroll > 0 else 0

        try:
            conn.execute("""
                INSERT OR REPLACE INTO prediction_history
                (prediction_date, player_id, player_name, team, opponent, position,
                 is_home, predicted_sog, baseline_sog, signal, sog_line, over_odds,
                 under_odds, bet_side, bet_kelly_pct, bet_amount, bankroll_at_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (today, pid, p.get("player_name"), p.get("team"),
                  p.get("opponent"), p.get("position"),
                  1 if p.get("is_home") else 0,
                  p["predicted_sog"], p.get("baseline_sog"),
                  round(signal, 3), sog_line, over_odds, under_odds,
                  bet_side, round(bet_kelly, 5), bet_amount, bankroll))
            saved += 1
        except Exception as exc:
            logger.debug("Failed to save prediction for %s: %s", pid, exc)

    conn.commit()
    conn.close()
    logger.info("Saved %d predictions to history for %s", saved, today)
    return saved


def score_past_predictions():
    """Score unscored predictions by comparing to actual game results."""
    from datetime import date
    today = date.today().isoformat()
    conn = data_collector.get_db()

    unscored = conn.execute("""
        SELECT ph.id, ph.prediction_date, ph.player_id, ph.sog_line, ph.bet_side
        FROM prediction_history ph
        WHERE ph.actual_sog IS NULL
          AND ph.prediction_date < ?
    """, (today,)).fetchall()

    scored = 0
    for row in unscored:
        # Find actual shots from that game day
        actual = conn.execute("""
            SELECT pgs.shots
            FROM player_game_stats pgs
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.player_id = ? AND g.date = ?
        """, (row["player_id"], row["prediction_date"])).fetchone()

        if not actual:
            continue

        actual_sog = actual["shots"]
        bet_won = None
        if row["bet_side"] and row["sog_line"] is not None:
            if row["bet_side"] == "OVER":
                bet_won = 1 if actual_sog > row["sog_line"] else 0
            elif row["bet_side"] == "UNDER":
                bet_won = 1 if actual_sog < row["sog_line"] else 0
            # Exact push = loss (conservative)

        conn.execute("""
            UPDATE prediction_history
            SET actual_sog = ?, bet_won = ?, scored_at = datetime('now')
            WHERE id = ?
        """, (actual_sog, bet_won, row["id"]))
        scored += 1

    conn.commit()
    conn.close()
    if scored > 0:
        logger.info("Scored %d past predictions", scored)
    return scored


def get_player_historical_confidence() -> dict:
    """Per-player confidence based on prediction history accuracy.

    Returns dict of player_id -> {
        total_preds, total_bets, bets_won, win_rate,
        avg_error, confidence_score (0-1)
    }
    """
    conn = data_collector.get_db()
    rows = conn.execute("""
        SELECT player_id,
               COUNT(*) as total_preds,
               AVG(ABS(predicted_sog - actual_sog)) as avg_error,
               SUM(CASE WHEN bet_won IS NOT NULL THEN 1 ELSE 0 END) as total_bets,
               SUM(CASE WHEN bet_won = 1 THEN 1 ELSE 0 END) as bets_won
        FROM prediction_history
        WHERE actual_sog IS NOT NULL
        GROUP BY player_id
        HAVING total_preds >= 3
    """).fetchall()
    conn.close()

    result = {}
    for r in rows:
        total_bets = r["total_bets"] or 0
        bets_won = r["bets_won"] or 0
        win_rate = bets_won / total_bets if total_bets > 0 else 0
        avg_error = r["avg_error"] or 2.0

        # Composite confidence score
        accuracy = min(max((win_rate - 0.40) / 0.30, 0), 1.0) if total_bets > 0 else 0.5
        error_score = 1.0 - min(avg_error / 2.0, 1.0)
        sample_score = min(r["total_preds"] / 20, 1.0)

        confidence_score = 0.4 * accuracy + 0.3 * error_score + 0.3 * sample_score

        result[r["player_id"]] = {
            "total_preds": r["total_preds"],
            "total_bets": total_bets,
            "bets_won": bets_won,
            "win_rate": round(win_rate, 3),
            "avg_error": round(avg_error, 2),
            "confidence_score": round(confidence_score, 3),
        }

    return result


def predict_team_sog() -> list[dict]:
    """
    Predict team-level SOG for and against for today's games.

    Blends bottom-up (sum of player predictions) with top-down
    (team historical averages and opponent defensive profiles).
    Weights: 60% bottom-up, 40% top-down.
    """
    if _model_fwd is None and _model_def is None:
        return []

    from datetime import date as date_cls
    today = date_cls.today().isoformat()
    sched = nhl_api.get_schedule(today)
    if not sched:
        return []

    games = []
    for week in sched.get("gameWeek", []):
        if week.get("date", "") != today:
            continue
        for g in week.get("games", []):
            games.append(g)

    if not games:
        return []

    conn = data_collector.get_db()

    # Team-level historical SOG for/against
    team_sog_for = {}
    team_sog_against = {}
    rows = conn.execute(
        """SELECT team, AVG(shots) as avg_sog_per_player, COUNT(DISTINCT game_id) as games
           FROM player_game_stats
           WHERE position IN ('L', 'C', 'R', 'D')
           GROUP BY team"""
    ).fetchall()
    for r in rows:
        team_sog_for[r["team"]] = r["avg_sog_per_player"] * _approx_skaters_per_game(conn, r["team"])

    # Shots against from team_defense table
    def_rows = conn.execute("SELECT team, shots_allowed_per_game FROM team_defense").fetchall()
    for r in def_rows:
        team_sog_against[r["team"]] = r["shots_allowed_per_game"]

    results = []
    for game in games:
        home = game.get("homeTeam", {}).get("abbrev", "")
        away = game.get("awayTeam", {}).get("abbrev", "")
        if not home or not away:
            continue

        # Get player-level predictions for each team
        home_player_preds = []
        away_player_preds = []
        for team_abbrev, opp, is_home, bucket in [
            (home, away, True, home_player_preds),
            (away, home, False, away_player_preds),
        ]:
            player_rows = conn.execute(
                """SELECT DISTINCT player_id, player_name, position
                   FROM player_game_stats
                   WHERE team = ? AND position IN ('L', 'C', 'R', 'D')
                   GROUP BY player_id
                   HAVING COUNT(*) >= 3
                   ORDER BY AVG(toi) DESC
                   LIMIT 18""",
                (team_abbrev,),
            ).fetchall()
            for pr in player_rows:
                pred = predict_player(int(pr["player_id"]), opp, is_home)
                if pred:
                    bucket.append(pred)

        # Bottom-up: sum of player predictions
        home_bu = sum(p["predicted_sog"] for p in home_player_preds)
        away_bu = sum(p["predicted_sog"] for p in away_player_preds)

        # Top-down: blend team SOG avg with opponent SA
        home_for_hist = team_sog_for.get(home, 28.0)
        away_for_hist = team_sog_for.get(away, 28.0)
        home_sa = team_sog_against.get(home, 28.0)
        away_sa = team_sog_against.get(away, 28.0)

        # Team's offensive avg blended with opponent's defensive avg
        home_td = (home_for_hist + away_sa) / 2
        away_td = (away_for_hist + home_sa) / 2

        # Blend: 60% bottom-up, 40% top-down
        home_pred = round(0.60 * home_bu + 0.40 * home_td, 1)
        away_pred = round(0.60 * away_bu + 0.40 * away_td, 1)

        results.append({
            "home_team": home,
            "away_team": away,
            "home_sog_pred": home_pred,
            "away_sog_pred": away_pred,
            "total_pred": round(home_pred + away_pred, 1),
            "home_sog_hist": round(home_for_hist, 1),
            "away_sog_hist": round(away_for_hist, 1),
            "home_sa_hist": round(home_sa, 1),
            "away_sa_hist": round(away_sa, 1),
            "home_players": len(home_player_preds),
            "away_players": len(away_player_preds),
        })

    conn.close()
    return results


def _approx_skaters_per_game(conn, team: str) -> float:
    """Average number of skaters per game for a team."""
    row = conn.execute(
        """SELECT AVG(cnt) as avg_skaters FROM (
               SELECT game_id, COUNT(*) as cnt
               FROM player_game_stats
               WHERE team = ? AND position IN ('L', 'C', 'R', 'D')
               GROUP BY game_id
           )""",
        (team,),
    ).fetchone()
    return row["avg_skaters"] if row and row["avg_skaters"] else 18.0


def predict_upcoming_games() -> list[dict]:
    """Get today's schedule and predict SOG for every skater in today's games."""
    if _model_fwd is None and _model_def is None:
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
