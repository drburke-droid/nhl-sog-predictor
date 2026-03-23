"""
NHL Shots on Goal Predictor — Flask application.

Provides a web dashboard and JSON API for viewing player shot statistics,
team defensive profiles, and machine-learning predictions for upcoming games.
"""

import sys
import logging
import threading
from datetime import date, datetime, timezone, timedelta

from flask import Flask, jsonify, render_template, request

MST = timezone(timedelta(hours=-7))
from apscheduler.schedulers.background import BackgroundScheduler

import nhl_api
import data_collector
import model
import model_v2
import moneypuck_collector
import nhl_odds_collector
import nhl_game_model
import mlb_api
import mlb_ou_v1
import mlb_data_collector
import mlb_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Track background refresh state
_refresh_status = {"running": False, "message": "idle"}
_timestamps = {"db_updated": None, "model_trained": None}

# MLB state
_mlb_refresh_status = {"running": False, "message": "idle"}
_mlb_timestamps = {"db_updated": None, "model_trained": None}


def _sync_db_to_github():
    """Push updated database to GitHub so Render gets fresh data."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "add", "nhl_data.db", "mlb_data.db", "saved_model_mlb/"],
            cwd=str(data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "commit", "-m", "Auto-update databases"],
            cwd=str(data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "push"],
            cwd=str(data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info("Database synced to GitHub successfully.")
        else:
            logger.warning("Git push failed: %s", result.stderr)
    except Exception as exc:
        logger.warning("Database sync to GitHub failed: %s", exc)


def _do_refresh():
    """Run data collection and model training in background."""
    global _refresh_status
    _refresh_status = {"running": True, "message": "Starting data collection..."}
    try:
        def progress(msg):
            _refresh_status["message"] = msg
            logger.info(msg)

        data_collector.collect_season_data(progress_callback=progress)
        _timestamps["db_updated"] = datetime.now(MST).strftime("%Y-%m-%d %I:%M %p MST")

        # Score yesterday's predictions against actual results
        _refresh_status["message"] = "Scoring past predictions..."
        try:
            model.score_past_predictions()
        except Exception as exc:
            logger.warning("Prediction scoring failed (non-fatal): %s", exc)

        # Collect today's odds (non-fatal if it fails)
        _refresh_status["message"] = "Collecting NHL odds..."
        try:
            nhl_odds_collector.collect_todays_odds(progress_callback=progress)
        except Exception as exc:
            logger.warning("NHL odds collection failed (non-fatal): %s", exc)

        _refresh_status["message"] = "Training model..."
        model.train_model()
        _timestamps["model_trained"] = datetime.now(MST).strftime("%Y-%m-%d %I:%M %p MST")
        _refresh_status["message"] = "Syncing database to GitHub..."
        _sync_db_to_github()
        _refresh_status = {"running": False, "message": "Refresh complete."}
    except Exception as exc:
        logger.exception("Refresh failed")
        _refresh_status = {"running": False, "message": f"Error: {exc}"}


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard page."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/players")
def api_players():
    """List all players with season SOG totals."""
    conn = data_collector.get_db()
    rows = conn.execute(
        """SELECT player_id, player_name, team, position,
                  COUNT(*) as games,
                  SUM(shots) as total_shots,
                  ROUND(AVG(shots), 2) as avg_shots,
                  ROUND(AVG(toi), 1) as avg_toi
           FROM player_game_stats
           WHERE position IN ('L', 'C', 'R', 'D')
           GROUP BY player_id
           HAVING games >= 1
           ORDER BY avg_shots DESC"""
    ).fetchall()
    conn.close()

    players = [dict(r) for r in rows]
    return jsonify(players)


@app.route("/api/player/<int:player_id>")
def api_player_detail(player_id: int):
    """Player detail with game log, rolling averages, and prediction."""
    df = data_collector.build_player_game_log(player_id)
    if df.empty:
        return jsonify({"error": "Player not found"}), 404

    df = data_collector.calculate_rolling_averages(df)

    # Get linemates from most recent game
    conn = data_collector.get_db()
    linemates = conn.execute(
        """SELECT lm.linemate_id, pgs.player_name, lm.shared_toi_seconds
           FROM linemates lm
           JOIN player_game_stats pgs ON lm.linemate_id = pgs.player_id
           WHERE lm.player_id = ?
           ORDER BY lm.game_id DESC, lm.shared_toi_seconds DESC
           LIMIT 10""",
        (player_id,),
    ).fetchall()
    conn.close()

    game_log = df.to_dict(orient="records")
    linemate_list = [dict(lm) for lm in linemates]

    # Latest info for context
    latest = game_log[-1] if game_log else {}
    player_name = latest.get("player_name", "Unknown")
    team = latest.get("team", "")
    position = latest.get("position", "")

    return jsonify({
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "position": position,
        "game_log": game_log,
        "linemates": linemate_list,
    })


@app.route("/api/teams")
def api_teams():
    """List all teams."""
    teams = nhl_api.get_all_teams()
    conn = data_collector.get_db()
    defense_rows = conn.execute("SELECT * FROM team_defense").fetchall()
    conn.close()

    defense_map = {r["team"]: dict(r) for r in defense_rows}

    result = []
    for t in teams:
        info = defense_map.get(t, {
            "team": t,
            "games_played": 0,
            "shots_allowed_per_game": 0,
            "shots_allowed_to_forwards": 0,
            "shots_allowed_to_defense": 0,
        })
        result.append(info)

    return jsonify(result)


@app.route("/api/team/<team_abbrev>/defense")
def api_team_defense(team_abbrev: str):
    """Team's shots allowed profile by position."""
    team_abbrev = team_abbrev.upper()
    profile = data_collector.build_opponent_defense_profile(team_abbrev)
    return jsonify(profile)


@app.route("/api/predictions")
def api_predictions():
    """Predictions for upcoming games, enriched with historical confidence."""
    predictions = model.predict_upcoming_games()
    metrics = model.get_model_metrics()

    # Merge historical confidence into predictions
    hist_conf = model.get_player_historical_confidence()
    for p in predictions:
        pid = p["player_id"]
        if pid in hist_conf:
            p["hist_confidence"] = hist_conf[pid]
        else:
            p["hist_confidence"] = None

    return jsonify({
        "predictions": predictions,
        "model_metrics": metrics,
    })


@app.route("/api/save-predictions", methods=["POST"])
def api_save_predictions():
    """Save today's predictions + odds to history for tracking."""
    data = request.get_json(silent=True) or {}
    bankroll_val = data.get("bankroll", 0)
    pred_list = data.get("predictions", [])

    if not pred_list:
        return jsonify({"saved": 0, "message": "No predictions to save."})

    # Build odds map from the submitted data
    odds_map = {}
    for p in pred_list:
        pid = p.get("player_id")
        if pid:
            odds_map[pid] = {
                "line": p.get("sog_line"),
                "over_odds": p.get("over_odds"),
                "under_odds": p.get("under_odds"),
            }

    # Get current model predictions
    predictions = model.predict_upcoming_games()

    saved = model.save_predictions_to_history(predictions, odds_map, bankroll_val)
    return jsonify({"saved": saved, "message": f"Saved {saved} predictions."})


@app.route("/api/prediction-history")
def api_prediction_history():
    """Historical prediction confidence per player."""
    return jsonify(model.get_player_historical_confidence())


@app.route("/api/predictability")
def api_predictability():
    """Player predictability scores (coefficient of variation)."""
    df = data_collector.get_player_predictability(min_games=20, min_avg_sog=0.5)
    if df.empty:
        return jsonify([])
    cols = ["player_id", "player_name", "team", "position", "games",
            "avg_sog", "std_sog", "cv"]
    result = df[[c for c in cols if c in df.columns]].to_dict(orient="records")
    return jsonify(result)


@app.route("/api/refresh")
def api_refresh():
    """Trigger a data refresh (runs in background thread)."""
    if _refresh_status["running"]:
        return jsonify({"status": "already_running", "message": _refresh_status["message"]})

    thread = threading.Thread(target=_do_refresh, daemon=True)
    thread.start()
    return jsonify({"status": "started", "message": "Data refresh started in background."})


@app.route("/api/refresh/status")
def api_refresh_status():
    """Check refresh status."""
    return jsonify(_refresh_status)


@app.route("/api/team-predictions")
def api_team_predictions():
    """Team-level SOG predictions for today's games."""
    preds = model.predict_team_sog()
    return jsonify(preds)


@app.route("/api/game-predictions")
def api_game_predictions():
    """Game-level predictions with +EV bet recommendations."""
    preds = nhl_game_model.predict_todays_games()
    metrics = nhl_game_model.get_model_metrics()
    return jsonify({"games": preds, "metrics": metrics})


@app.route("/api/nhl/odds")
def api_nhl_odds():
    """Today's NHL game odds and player SOG props."""
    try:
        nhl_odds_collector.ensure_todays_odds()
    except Exception:
        pass
    game_odds = nhl_odds_collector.get_game_odds_for_date()
    player_props = nhl_odds_collector.get_all_player_props_for_date()
    return jsonify({
        "game_odds": game_odds,
        "player_props": player_props,
    })


@app.route("/api/nhl/betmgm-odds")
def api_nhl_betmgm_odds():
    """BetMGM player SOG props with PlayAlberta vig estimates.

    Auto-fetches today's odds if needed, then returns BetMGM-specific
    lines with estimated PlayAlberta adjustments.
    """
    try:
        nhl_odds_collector.ensure_todays_odds()
    except Exception:
        pass
    props = nhl_odds_collector.get_betmgm_player_props()
    return jsonify(props)


@app.route("/api/status")
def api_status():
    """Return DB and model timestamps."""
    return jsonify(_timestamps)


# ---------------------------------------------------------------------------
# Routes — V2 Model API
# ---------------------------------------------------------------------------

@app.route("/api/v2/predictions")
def api_v2_predictions():
    """V2 predictions with MoneyPuck features + clustering."""
    predictions = model_v2.predict_upcoming_games()
    metrics = model_v2.get_model_metrics()

    # Merge historical confidence (reuse V1's history)
    hist_conf = model.get_player_historical_confidence()
    for p in predictions:
        pid = p["player_id"]
        p["hist_confidence"] = hist_conf.get(pid)

    return jsonify({
        "predictions": predictions,
        "model_metrics": metrics,
    })


@app.route("/api/v2/metrics")
def api_v2_metrics():
    """V2 model metrics including cluster info."""
    return jsonify(model_v2.get_model_metrics())


# ---------------------------------------------------------------------------
# Routes — MLB API
# ---------------------------------------------------------------------------

def _do_mlb_refresh():
    """Run MLB data collection and model training in background."""
    global _mlb_refresh_status
    _mlb_refresh_status = {"running": True, "message": "Starting MLB data collection..."}
    try:
        def progress(msg):
            _mlb_refresh_status["message"] = msg
            logger.info("MLB: %s", msg)

        mlb_data_collector.collect_season_data(progress_callback=progress)
        _mlb_timestamps["db_updated"] = datetime.now(MST).strftime("%Y-%m-%d %I:%M %p MST")
        _mlb_refresh_status["message"] = "Training MLB model..."
        mlb_model.train_model()
        _mlb_timestamps["model_trained"] = datetime.now(MST).strftime("%Y-%m-%d %I:%M %p MST")
        _mlb_refresh_status["message"] = "Syncing databases to GitHub..."
        _sync_mlb_db_to_github()
        _mlb_refresh_status = {"running": False, "message": "MLB refresh complete."}
    except Exception as exc:
        logger.exception("MLB refresh failed")
        _mlb_refresh_status = {"running": False, "message": f"Error: {exc}"}


def _sync_mlb_db_to_github():
    """Push MLB database to GitHub."""
    import subprocess
    try:
        subprocess.run(
            ["git", "add", "mlb_data.db"],
            cwd=str(mlb_data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["git", "add", "saved_model_mlb/"],
            cwd=str(mlb_data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", "Auto-update MLB database"],
            cwd=str(mlb_data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "push"],
            cwd=str(mlb_data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info("MLB database synced to GitHub.")
        else:
            logger.warning("MLB git push failed: %s", result.stderr)
    except Exception as exc:
        logger.warning("MLB database sync failed: %s", exc)


@app.route("/api/mlb/predictions")
def api_mlb_predictions():
    """Pitcher strikeout predictions for today's games."""
    predictions = mlb_model.predict_todays_games()
    metrics = mlb_model.get_model_metrics()
    return jsonify({
        "predictions": predictions,
        "model_metrics": metrics,
    })


@app.route("/api/mlb/game-predictions")
def api_mlb_game_predictions():
    """MLB O/U v1: game-level predictions with totals + ML bets."""
    try:
        games = mlb_ou_v1.predict_todays_games()
        return jsonify({"games": games})
    except Exception as exc:
        logger.warning("MLB game predictions failed: %s", exc)
        return jsonify({"games": [], "error": str(exc)})


@app.route("/api/mlb/pitcher/<int:pitcher_id>")
def api_mlb_pitcher_detail(pitcher_id: int):
    """Pitcher detail with game log and rolling stats."""
    df = mlb_data_collector.build_pitcher_game_log(pitcher_id)
    if df.empty:
        return jsonify({"error": "Pitcher not found"}), 404

    game_log = df.to_dict(orient="records")
    stats = mlb_data_collector.get_pitcher_rolling_stats(pitcher_id)
    statcast = mlb_data_collector.get_statcast_for_pitcher(pitcher_id)
    arsenal = mlb_data_collector.get_pitcher_arsenal(pitcher_id)
    tto = mlb_data_collector.get_pitcher_tto_profile(pitcher_id)

    latest = game_log[-1] if game_log else {}
    return jsonify({
        "pitcher_id": pitcher_id,
        "pitcher_name": latest.get("pitcher_name", "Unknown"),
        "team": latest.get("team_abbrev", ""),
        "game_log": game_log,
        "rolling_stats": stats,
        "statcast": statcast,
        "arsenal": arsenal,
        "tto_profile": tto,
    })


@app.route("/api/mlb/teams")
def api_mlb_teams():
    """MLB team batting stats."""
    conn = mlb_data_collector.get_db()
    rows = conn.execute("SELECT * FROM mlb_team_batting ORDER BY k_rate DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/mlb/refresh")
def api_mlb_refresh():
    """Trigger MLB data refresh."""
    if _mlb_refresh_status["running"]:
        return jsonify({"status": "already_running", "message": _mlb_refresh_status["message"]})
    thread = threading.Thread(target=_do_mlb_refresh, daemon=True)
    thread.start()
    return jsonify({"status": "started", "message": "MLB data refresh started."})


@app.route("/api/mlb/refresh/status")
def api_mlb_refresh_status():
    return jsonify(_mlb_refresh_status)


@app.route("/api/mlb/status")
def api_mlb_status():
    return jsonify(_mlb_timestamps)


# ---------------------------------------------------------------------------
# Startup logic (works for both gunicorn and direct run)
# ---------------------------------------------------------------------------

def _needs_refresh() -> bool:
    """Check if the database is stale (no games from today or yesterday)."""
    import os
    from datetime import date, timedelta
    if not os.path.exists(str(data_collector.DB_PATH)):
        return True
    try:
        conn = data_collector.get_db()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        row = conn.execute(
            "SELECT MAX(date) as last_date FROM games"
        ).fetchone()
        conn.close()
        if not row or not row["last_date"]:
            return True
        return row["last_date"] < yesterday
    except Exception:
        return True


def _is_render() -> bool:
    """Detect if we're running on Render."""
    import os
    return bool(os.environ.get("RENDER"))


def _sync_from_github():
    """Pull latest DB/models from GitHub so this PC has the freshest data."""
    import subprocess
    try:
        # Stash any local uncommitted changes to avoid merge conflicts
        subprocess.run(
            ["git", "stash", "--include-untracked"],
            cwd=str(data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "pull", "--rebase"],
            cwd=str(data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info("Git pull successful: %s", result.stdout.strip())
        else:
            logger.warning("Git pull failed: %s", result.stderr)
        # Pop stash if anything was stashed
        subprocess.run(
            ["git", "stash", "pop"],
            cwd=str(data_collector.DB_PATH.parent),
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        logger.warning("Git sync failed (non-fatal): %s", exc)


def _startup():
    """Run on app startup: load saved model or train, start scheduler."""
    import os
    on_render = _is_render()

    # Sync from GitHub first so we have the latest DB from the other PC
    if not on_render:
        logger.info("Syncing from GitHub...")
        _sync_from_github()

    # Try loading saved model first (fast cold start)
    loaded = model.load_model()
    if loaded:
        _timestamps["model_trained"] = "loaded from cache"
        try:
            conn = data_collector.get_db()
            row = conn.execute("SELECT MAX(date) as last_date FROM games").fetchone()
            conn.close()
            if row and row["last_date"]:
                _timestamps["db_updated"] = row["last_date"]
        except Exception:
            pass
        logger.info("Model loaded from saved cache — fast startup")
        # Retrain in background to pick up any new data (both local and Render)
        def _bg_retrain():
            try:
                model.train_model()
                _timestamps["model_trained"] = datetime.now(MST).strftime("%Y-%m-%d %I:%M %p MST")
                logger.info("Background retrain complete")
            except Exception as exc:
                logger.warning("Background retrain failed: %s", exc)
            # Also train V2 model
            try:
                model_v2.train_model()
                logger.info("V2 background retrain complete")
            except Exception as exc:
                logger.warning("V2 background retrain failed: %s", exc)
            # Also train game model
            try:
                nhl_game_model.train_model()
                logger.info("Game model background retrain complete")
            except Exception as exc:
                logger.warning("Game model background retrain failed: %s", exc)
        threading.Thread(target=_bg_retrain, daemon=True).start()
        # Load V2 and game model cached for fast startup
        model_v2.load_model()
        nhl_game_model.load_model()
    else:
        # No saved model — must train (slower startup)
        try:
            if os.path.exists(str(data_collector.DB_PATH)):
                logger.info("No cached model — training from DB...")
                metrics = model.train_model()
                _timestamps["model_trained"] = datetime.now(MST).strftime("%Y-%m-%d %I:%M %p MST")
                try:
                    conn = data_collector.get_db()
                    row = conn.execute("SELECT MAX(date) as last_date FROM games").fetchone()
                    conn.close()
                    if row and row["last_date"]:
                        _timestamps["db_updated"] = row["last_date"]
                except Exception:
                    pass
                logger.info("Model trained. Metrics: %s", metrics)
            else:
                logger.info("No database found. Use /api/refresh to collect data.")
        except Exception as exc:
            logger.warning("Could not train model on startup: %s", exc)

    if on_render:
        # On Render: NHL API blocks shared IPs, so skip data collection.
        # The local PC pushes fresh DB via git; Render just trains the model.
        logger.info("Running on Render — skipping data refresh (using committed DB).")
        return

    # Schedule daily refresh at 9:00 AM with generous misfire grace
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _do_refresh,
        trigger="cron",
        hour=9,
        minute=0,
        id="daily_refresh",
        misfire_grace_time=14400,  # 4 hours — catches wake-from-sleep up to 1 PM
    )
    scheduler.start()
    logger.info("Scheduled daily data refresh at 9:00 AM")

    # If data is stale (missed yesterday's refresh), trigger one now
    if _needs_refresh():
        logger.info("Database is stale — triggering background refresh on startup.")
        thread = threading.Thread(target=_do_refresh, daemon=True)
        thread.start()


def _mlb_startup():
    """Load MLB model on startup."""
    import os
    loaded = mlb_model.load_model()
    if loaded:
        _mlb_timestamps["model_trained"] = "loaded from cache"
        try:
            conn = mlb_data_collector.get_db()
            row = conn.execute("SELECT MAX(date) as last_date FROM mlb_games").fetchone()
            conn.close()
            if row and row["last_date"]:
                _mlb_timestamps["db_updated"] = row["last_date"]
        except Exception:
            pass
        logger.info("MLB model loaded from cache")
    elif os.path.exists(str(mlb_data_collector.DB_PATH)):
        try:
            mlb_model.train_model()
            _mlb_timestamps["model_trained"] = datetime.now(MST).strftime("%Y-%m-%d %I:%M %p MST")
            logger.info("MLB model trained on startup")
        except Exception:
            logger.warning("Could not train MLB model on startup")


# Run startup for both gunicorn and direct execution
_startup()
_mlb_startup()


# ---------------------------------------------------------------------------
# Main (direct run only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--refresh" in sys.argv:
        logger.info("Running initial data collection...")
        _do_refresh()
        # Retrain after fresh data
        try:
            model.train_model()
        except Exception:
            pass

    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
