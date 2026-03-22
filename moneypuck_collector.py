"""
MoneyPuck Data Collector for V2 model.

Ingests two data sources from MoneyPuck into the SQLite database:

1. **Game-by-game player stats** (2025.csv / skaters game-by-game):
   Pre-aggregated per player-game with danger zone breakdowns, xGoals,
   rebounds, and situation splits (all, 5v5, 5v4, 4v5).

2. **Shot-level data** (shots_2025.csv):
   Individual shot events with coordinates, type, rebound/rush flags.
   Used to derive slot %, perimeter %, and rush/rebound rates.

Data is downloaded from peter-tanner.com/moneypuck (public mirror).
MoneyPuck uses NHL player IDs — no mapping needed.
"""

import csv
import io
import logging
import os
import sqlite3
import zipfile
from datetime import date, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "nhl_data.db"

# MoneyPuck mirror (Cloudflare-free)
_MIRROR = "https://peter-tanner.com/moneypuck/downloads"

# Current season identifier (MoneyPuck uses start year, e.g. 2025 for 2025-26)
CURRENT_SEASON = 2025


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        -- Per player-game aggregated stats from MoneyPuck game-by-game data
        CREATE TABLE IF NOT EXISTS mp_player_game (
            game_id         INTEGER,
            player_id       INTEGER,
            game_date       TEXT,
            player_name     TEXT,
            team            TEXT,
            opponent        TEXT,
            is_home         INTEGER,
            position        TEXT,
            situation       TEXT,       -- 'all', '5on5', '5on4', '4on5'
            icetime         REAL,       -- seconds
            shifts          INTEGER,
            ice_time_rank   INTEGER,    -- rank within team
            -- Individual for stats (I_F_ prefix)
            shots_on_goal   INTEGER,
            shot_attempts   INTEGER,
            missed_shots    INTEGER,
            blocked_attempts INTEGER,
            goals           INTEGER,
            xgoals          REAL,
            high_danger_shots INTEGER,
            medium_danger_shots INTEGER,
            low_danger_shots INTEGER,
            high_danger_xg  REAL,
            medium_danger_xg REAL,
            low_danger_xg   REAL,
            rebounds_created INTEGER,
            x_rebounds      REAL,
            xgoals_with_rebounds REAL,
            -- On-ice team for stats (OnIce_F_ prefix)
            on_ice_sog_for  INTEGER,
            on_ice_sa_against INTEGER,
            on_ice_xg_for   REAL,
            on_ice_xg_against REAL,
            on_ice_hd_shots_for INTEGER,
            on_ice_hd_shots_against INTEGER,
            -- Game score (MoneyPuck's composite metric)
            game_score      REAL,
            PRIMARY KEY (game_id, player_id, situation)
        );

        CREATE INDEX IF NOT EXISTS idx_mp_pg_player
            ON mp_player_game(player_id, game_date);
        CREATE INDEX IF NOT EXISTS idx_mp_pg_team
            ON mp_player_game(team, game_date);
        CREATE INDEX IF NOT EXISTS idx_mp_pg_opp
            ON mp_player_game(opponent, game_date);

        -- Shot-level data for style features (slot %, rush %, rebound %)
        CREATE TABLE IF NOT EXISTS mp_shots (
            shot_id         INTEGER,
            game_id         INTEGER,
            season          INTEGER,
            player_id       INTEGER,
            player_name     TEXT,
            team            TEXT,
            opponent        TEXT,
            is_home         INTEGER,
            period          INTEGER,
            game_seconds    INTEGER,
            event           TEXT,       -- SHOT, GOAL, MISS
            shot_on_goal    INTEGER,
            goal            INTEGER,
            shot_type       TEXT,       -- WRIST, SLAP, SNAP, BACKHAND, TIP, WRAP
            shot_distance   REAL,
            shot_distance_adjusted REAL,
            x_coord         REAL,
            y_coord         REAL,
            shot_angle      REAL,
            xgoal           REAL,
            is_rebound      INTEGER,
            is_rush         INTEGER,
            is_empty_net    INTEGER,
            home_skaters    INTEGER,
            away_skaters    INTEGER,
            PRIMARY KEY (game_id, shot_id)
        );

        CREATE INDEX IF NOT EXISTS idx_mp_shots_player
            ON mp_shots(player_id, game_id);
        CREATE INDEX IF NOT EXISTS idx_mp_shots_team
            ON mp_shots(team, game_id);

        -- Metadata for tracking last download
        CREATE TABLE IF NOT EXISTS mp_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# NHL game_id mapping
# ---------------------------------------------------------------------------

def _mp_game_id_to_nhl(mp_game_id: int, season: int) -> int:
    """Convert MoneyPuck game_id to NHL API game_id.

    Player-game CSVs use full NHL format (e.g. 2025020001) — pass through.
    Shot CSVs use short form (e.g. 20001) — prepend season prefix.
    Auto-detects based on magnitude.
    """
    if mp_game_id > 1_000_000:
        # Already full NHL format
        return mp_game_id
    # Short form: prepend season + 02 (regular season)
    return season * 1_000_000 + mp_game_id


def _mp_date_to_iso(mp_date: str) -> str:
    """Convert MoneyPuck date format '20251007' to ISO '2025-10-07'."""
    if len(mp_date) == 8:
        return f"{mp_date[:4]}-{mp_date[4:6]}-{mp_date[6:8]}"
    return mp_date


# ---------------------------------------------------------------------------
# Team abbreviation mapping
# ---------------------------------------------------------------------------

# MoneyPuck uses some different abbreviations than the NHL API
_MP_TEAM_MAP = {
    "ARI": "UTA",   # Arizona moved to Utah
    "PHX": "UTA",
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
}


def _normalize_team(mp_team: str) -> str:
    """Normalize MoneyPuck team abbreviation to NHL API standard."""
    return _MP_TEAM_MAP.get(mp_team, mp_team)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_zip(url: str, timeout: int = 120) -> bytes:
    """Download a zip file and return its contents as bytes."""
    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _extract_csv_from_zip(zip_bytes: bytes, csv_name: str = None) -> str:
    """Extract first CSV from a zip file, return as string."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if csv_name:
            target = csv_name
        else:
            csvs = [n for n in names if n.endswith(".csv")]
            target = csvs[0] if csvs else names[0]
        with zf.open(target) as f:
            return f.read().decode("utf-8-sig")


# ---------------------------------------------------------------------------
# Ingest game-by-game player data
# ---------------------------------------------------------------------------

def ingest_player_game_data(csv_path: str = None, season: int = CURRENT_SEASON,
                            progress_callback=None):
    """Ingest MoneyPuck game-by-game player data into mp_player_game.

    Args:
        csv_path: Path to local CSV file. If None, downloads from mirror.
        season: Season identifier (start year).
        progress_callback: Optional callable for status updates.
    """
    conn = get_db()

    if csv_path and os.path.exists(csv_path):
        logger.info("Loading MoneyPuck player-game data from %s", csv_path)
        with open(csv_path, encoding="utf-8-sig") as f:
            csv_text = f.read()
    else:
        url = f"{_MIRROR}/shots_{season}.zip"
        # The game-by-game data URL pattern
        # Try the skaters career game-by-game first
        url = f"{_MIRROR}/seasonPlayersSummary/skaters/{season}.zip"
        try:
            zip_bytes = _download_zip(url)
            csv_text = _extract_csv_from_zip(zip_bytes)
        except Exception as exc:
            logger.warning("Could not download MoneyPuck player-game data: %s", exc)
            conn.close()
            return 0

    reader = csv.DictReader(io.StringIO(csv_text))

    rows_inserted = 0
    batch = []

    for row in reader:
        # Only ingest skater situations we care about
        situation = row.get("situation", "")
        if situation not in ("all", "5on5", "5on4"):
            continue

        # Skip playoff games
        if row.get("isPlayoffGame") == "1":
            continue

        try:
            mp_gid = int(row["gameId"])
            game_id = _mp_game_id_to_nhl(mp_gid, season)
            player_id = int(row["playerId"])
            game_date = _mp_date_to_iso(row["gameDate"])
            team = _normalize_team(row["playerTeam"])
            opponent = _normalize_team(row["opposingTeam"])
            is_home = 1 if row.get("home_or_away") == "HOME" else 0

            batch.append((
                game_id, player_id, game_date, row["name"], team, opponent,
                is_home, row.get("position", ""),
                situation,
                _float(row.get("icetime")),
                _int(row.get("shifts")),
                _int(row.get("iceTimeRank")),
                # Individual stats
                _int(row.get("I_F_shotsOnGoal")),
                _int(row.get("I_F_shotAttempts")),
                _int(row.get("I_F_missedShots")),
                _int(row.get("I_F_blockedShotAttempts")),
                _int(row.get("I_F_goals")),
                _float(row.get("I_F_xGoals")),
                _int(row.get("I_F_highDangerShots")),
                _int(row.get("I_F_mediumDangerShots")),
                _int(row.get("I_F_lowDangerShots")),
                _float(row.get("I_F_highDangerxGoals")),
                _float(row.get("I_F_mediumDangerxGoals")),
                _float(row.get("I_F_lowDangerxGoals")),
                _int(row.get("I_F_rebounds")),
                _float(row.get("I_F_xRebounds")),
                _float(row.get("I_F_xGoals_with_earned_rebounds")),
                # On-ice stats
                _int(row.get("OnIce_F_shotsOnGoal")),
                _int(row.get("OnIce_A_shotsOnGoal")),
                _float(row.get("OnIce_F_xGoals")),
                _float(row.get("OnIce_A_xGoals")),
                _int(row.get("OnIce_F_highDangerShots")),
                _int(row.get("OnIce_A_highDangerShots")),
                _float(row.get("gameScore")),
            ))

        except (ValueError, KeyError) as exc:
            continue

        if len(batch) >= 5000:
            rows_inserted += _insert_player_game_batch(conn, batch)
            batch = []
            if progress_callback:
                progress_callback(f"MoneyPuck: {rows_inserted} player-game rows...")

    if batch:
        rows_inserted += _insert_player_game_batch(conn, batch)

    # Update metadata
    conn.execute(
        "INSERT OR REPLACE INTO mp_meta (key, value) VALUES (?, ?)",
        ("player_game_last_ingest", datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    logger.info("MoneyPuck player-game: %d rows ingested for season %d", rows_inserted, season)
    return rows_inserted


def _insert_player_game_batch(conn, batch):
    conn.executemany("""
        INSERT OR IGNORE INTO mp_player_game (
            game_id, player_id, game_date, player_name, team, opponent,
            is_home, position, situation,
            icetime, shifts, ice_time_rank,
            shots_on_goal, shot_attempts, missed_shots, blocked_attempts,
            goals, xgoals,
            high_danger_shots, medium_danger_shots, low_danger_shots,
            high_danger_xg, medium_danger_xg, low_danger_xg,
            rebounds_created, x_rebounds, xgoals_with_rebounds,
            on_ice_sog_for, on_ice_sa_against,
            on_ice_xg_for, on_ice_xg_against,
            on_ice_hd_shots_for, on_ice_hd_shots_against,
            game_score
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, batch)
    conn.commit()
    return len(batch)


# ---------------------------------------------------------------------------
# Ingest shot-level data
# ---------------------------------------------------------------------------

def ingest_shot_data(csv_path: str = None, season: int = CURRENT_SEASON,
                     progress_callback=None):
    """Ingest MoneyPuck shot-level data into mp_shots.

    Args:
        csv_path: Path to local CSV. If None, downloads from mirror.
        season: Season identifier.
    """
    conn = get_db()

    if csv_path and os.path.exists(csv_path):
        logger.info("Loading MoneyPuck shot data from %s", csv_path)
        with open(csv_path, encoding="utf-8-sig") as f:
            csv_text = f.read()
    else:
        url = f"{_MIRROR}/shots_{season}.zip"
        try:
            zip_bytes = _download_zip(url)
            csv_text = _extract_csv_from_zip(zip_bytes)
        except Exception as exc:
            logger.warning("Could not download MoneyPuck shot data: %s", exc)
            conn.close()
            return 0

    reader = csv.DictReader(io.StringIO(csv_text))

    rows_inserted = 0
    batch = []

    for row in reader:
        # Skip playoff games
        if row.get("isPlayoffGame") == "1":
            continue

        try:
            mp_gid = int(row["game_id"])
            game_id = _mp_game_id_to_nhl(mp_gid, season)
            player_id = int(row["shooterPlayerId"])
            team = _normalize_team(row["teamCode"])

            # Determine opponent
            home = _normalize_team(row["homeTeamCode"])
            away = _normalize_team(row["awayTeamCode"])
            opponent = away if team == home else home
            is_home = 1 if team == home else 0

            # Slot zone: high danger = within ~30ft and good angle
            # We'll store the raw data and compute slot % at feature time

            batch.append((
                int(row["shotID"]), game_id, season, player_id,
                row["shooterName"], team, opponent, is_home,
                _int(row.get("period")),
                _int(row.get("time")),
                row.get("event", ""),
                _int(row.get("shotWasOnGoal")),
                _int(row.get("goal")),
                row.get("shotType", ""),
                _float(row.get("shotDistance")),
                _float(row.get("arenaAdjustedShotDistance")),
                _float(row.get("xCordAdjusted")),
                _float(row.get("yCordAdjusted")),
                _float(row.get("shotAngleAdjusted")),
                _float(row.get("xGoal")),
                _int(row.get("shotRebound")),
                _int(row.get("shotRush")),
                _int(row.get("shotOnEmptyNet")),
                _int(row.get("homeSkatersOnIce")),
                _int(row.get("awaySkatersOnIce")),
            ))
        except (ValueError, KeyError):
            continue

        if len(batch) >= 10000:
            rows_inserted += _insert_shot_batch(conn, batch)
            batch = []
            if progress_callback:
                progress_callback(f"MoneyPuck shots: {rows_inserted} rows...")

    if batch:
        rows_inserted += _insert_shot_batch(conn, batch)

    conn.execute(
        "INSERT OR REPLACE INTO mp_meta (key, value) VALUES (?, ?)",
        ("shot_data_last_ingest", datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    logger.info("MoneyPuck shots: %d rows ingested for season %d", rows_inserted, season)
    return rows_inserted


def _insert_shot_batch(conn, batch):
    conn.executemany("""
        INSERT OR IGNORE INTO mp_shots (
            shot_id, game_id, season, player_id,
            player_name, team, opponent, is_home,
            period, game_seconds, event, shot_on_goal, goal,
            shot_type, shot_distance, shot_distance_adjusted,
            x_coord, y_coord, shot_angle,
            xgoal, is_rebound, is_rush, is_empty_net,
            home_skaters, away_skaters
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, batch)
    conn.commit()
    return len(batch)


# ---------------------------------------------------------------------------
# High-level collection entry point
# ---------------------------------------------------------------------------

def collect_season_data(season: int = CURRENT_SEASON, progress_callback=None,
                        player_game_csv: str = None, shots_csv: str = None):
    """Collect all MoneyPuck data for a season.

    Checks if data is already ingested (by row count) to avoid redundant work.
    """
    conn = get_db()

    # Check existing data
    pg_count = conn.execute(
        "SELECT COUNT(*) FROM mp_player_game WHERE game_id >= ? AND game_id < ?",
        (season * 1_000_000, (season + 1) * 1_000_000),
    ).fetchone()[0]

    shot_count = conn.execute(
        "SELECT COUNT(*) FROM mp_shots WHERE season = ?",
        (season,),
    ).fetchone()[0]
    conn.close()

    total = 0

    # Always refresh (incremental via INSERT OR IGNORE)
    if progress_callback:
        progress_callback("Ingesting MoneyPuck player-game data...")
    n = ingest_player_game_data(csv_path=player_game_csv, season=season,
                                progress_callback=progress_callback)
    total += n
    logger.info("MoneyPuck player-game: %d new rows (had %d)", n, pg_count)

    if progress_callback:
        progress_callback("Ingesting MoneyPuck shot data...")
    n = ingest_shot_data(csv_path=shots_csv, season=season,
                         progress_callback=progress_callback)
    total += n
    logger.info("MoneyPuck shots: %d new rows (had %d)", n, shot_count)

    return total


# ---------------------------------------------------------------------------
# Query helpers for feature engineering
# ---------------------------------------------------------------------------

def get_player_game_stats(player_id: int, situation: str = "all",
                          before_date: str = None, limit: int = 30) -> list[dict]:
    """Get MoneyPuck player-game stats for feature computation.

    Returns rows ordered by game_date DESC (most recent first).
    Only returns pre-game data (before_date exclusive).
    """
    conn = get_db()
    query = """
        SELECT * FROM mp_player_game
        WHERE player_id = ? AND situation = ?
    """
    params = [player_id, situation]
    if before_date:
        query += " AND game_date < ?"
        params.append(before_date)
    query += " ORDER BY game_date DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_player_shot_style(player_id: int, before_date: str = None,
                          last_n_games: int = 10) -> dict:
    """Compute shot style features from shot-level data.

    Returns dict with slot_pct, perimeter_pct, rebound_rate, rush_rate,
    avg_shot_distance, xg_per_shot, shots_per_attempt.
    """
    conn = get_db()

    # Get game_ids for last N games
    params = [player_id]
    if before_date:
        # Need game_date from mp_player_game for date filtering
        gid_query = """
            SELECT DISTINCT s.game_id FROM mp_shots s
            JOIN mp_player_game pg ON s.game_id = pg.game_id
                AND s.player_id = pg.player_id AND pg.situation = 'all'
            WHERE s.player_id = ? AND pg.game_date < ?
            ORDER BY s.game_id DESC LIMIT ?
        """
        params.extend([before_date, last_n_games])
    else:
        gid_query = """
            SELECT DISTINCT game_id FROM mp_shots
            WHERE player_id = ?
            ORDER BY game_id DESC LIMIT ?
        """
        params.append(last_n_games)

    game_ids = [r[0] for r in conn.execute(gid_query, params).fetchall()]

    if not game_ids:
        conn.close()
        return {}

    placeholders = ",".join("?" * len(game_ids))
    shots = conn.execute(f"""
        SELECT shot_distance_adjusted, x_coord, y_coord,
               is_rebound, is_rush, xgoal, shot_on_goal, event
        FROM mp_shots
        WHERE player_id = ? AND game_id IN ({placeholders})
    """, [player_id] + game_ids).fetchall()
    conn.close()

    if not shots:
        return {}

    total = len(shots)
    sog = sum(1 for s in shots if s["shot_on_goal"])
    goals = sum(1 for s in shots if s["event"] == "GOAL")

    # Slot zone: adjusted x >= 60 and |y| <= 22 (roughly the "home plate" area)
    slot = sum(1 for s in shots
               if s["x_coord"] is not None and s["y_coord"] is not None
               and abs(s["x_coord"]) >= 60 and abs(s["y_coord"]) <= 22)

    perimeter = total - slot
    rebounds = sum(1 for s in shots if s["is_rebound"])
    rushes = sum(1 for s in shots if s["is_rush"])
    xg_sum = sum(s["xgoal"] for s in shots if s["xgoal"] is not None)
    dist_sum = sum(s["shot_distance_adjusted"] for s in shots
                   if s["shot_distance_adjusted"] is not None)
    dist_count = sum(1 for s in shots if s["shot_distance_adjusted"] is not None)

    return {
        "total_shots": total,
        "sog": sog,
        "slot_pct": slot / total if total else 0,
        "perimeter_pct": perimeter / total if total else 0,
        "rebound_rate": rebounds / total if total else 0,
        "rush_rate": rushes / total if total else 0,
        "avg_shot_distance": dist_sum / dist_count if dist_count else 0,
        "xg_per_shot": xg_sum / total if total else 0,
        "sog_per_attempt": sog / total if total else 0,
        "n_games": len(game_ids),
    }


def get_opponent_shot_profile(opponent: str, before_date: str = None,
                              last_n_games: int = 15) -> dict:
    """Get opponent defensive profile from MoneyPuck data.

    Returns dict with expanded opponent features: slot_shots_allowed,
    pace, rush_shots_allowed, etc.
    """
    conn = get_db()

    query = """
        SELECT game_id, game_date, opponent,
               SUM(shot_attempts) as total_attempts,
               SUM(shots_on_goal) as total_sog,
               SUM(high_danger_shots) as hd_shots,
               SUM(medium_danger_shots) as md_shots,
               SUM(low_danger_shots) as ld_shots,
               SUM(xgoals) as total_xg
        FROM mp_player_game
        WHERE opponent = ? AND situation = 'all'
    """
    params = [opponent]
    if before_date:
        query += " AND game_date < ?"
        params.append(before_date)
    query += " GROUP BY game_id ORDER BY game_date DESC LIMIT ?"
    params.append(last_n_games)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return {}

    n = len(rows)
    avg_attempts = sum(r["total_attempts"] or 0 for r in rows) / n
    avg_sog = sum(r["total_sog"] or 0 for r in rows) / n
    avg_hd = sum(r["hd_shots"] or 0 for r in rows) / n
    avg_xg = sum(r["total_xg"] or 0 for r in rows) / n

    return {
        "opp_attempts_allowed": round(avg_attempts, 2),
        "opp_sog_allowed": round(avg_sog, 2),
        "opp_hd_shots_allowed": round(avg_hd, 2),
        "opp_xg_allowed": round(avg_xg, 2),
        "opp_slot_rate_allowed": round(avg_hd / avg_attempts, 4) if avg_attempts else 0,
        "opp_pace": round(avg_attempts, 2),  # proxy for pace
        "n_games": n,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(val) -> float:
    """Convert value to float, returning None for empty/invalid."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _int(val) -> int:
    """Convert value to int, returning None for empty/invalid."""
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        # Use local files from user's Downloads
        pg_csv = None
        shots_csv = None
        for path in sys.argv[2:]:
            if "shots" in path.lower():
                shots_csv = path
            else:
                pg_csv = path
        collect_season_data(player_game_csv=pg_csv, shots_csv=shots_csv)
    else:
        collect_season_data()
