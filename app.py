"""
NHL Shots on Goal Predictor — Flask application.

Provides a web dashboard and JSON API for viewing player shot statistics,
team defensive profiles, and machine-learning predictions for upcoming games.
"""

import sys
import logging
import threading
from datetime import date

from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler

import nhl_api
import data_collector
import model

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


def _do_refresh():
    """Run data collection and model training in background."""
    global _refresh_status
    _refresh_status = {"running": True, "message": "Starting data collection..."}
    try:
        def progress(msg):
            _refresh_status["message"] = msg
            logger.info(msg)

        data_collector.collect_season_data(progress_callback=progress)
        _refresh_status["message"] = "Training model..."
        model.train_model()
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
    """Predictions for upcoming games."""
    predictions = model.predict_upcoming_games()
    metrics = model.get_model_metrics()
    return jsonify({
        "predictions": predictions,
        "model_metrics": metrics,
    })


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


# ---------------------------------------------------------------------------
# Startup logic (works for both gunicorn and direct run)
# ---------------------------------------------------------------------------

def _startup():
    """Run on app startup: train model if data exists, start scheduler."""
    import os
    # Train model on startup if data exists
    try:
        if os.path.exists(str(data_collector.DB_PATH)):
            metrics = model.train_model()
            logger.info("Model metrics: %s", metrics)
        else:
            logger.info("No database found. Use /api/refresh to collect data.")
    except Exception:
        logger.warning("Could not train model on startup (probably no data yet)")

    # Schedule automatic daily refresh at 9:00 AM
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _do_refresh,
        trigger="cron",
        hour=9,
        minute=0,
        id="daily_refresh",
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info("Scheduled daily data refresh at 9:00 AM")


# Run startup for both gunicorn and direct execution
_startup()


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
