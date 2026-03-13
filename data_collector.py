"""
Data collection and processing module.

Pulls NHL boxscores and play-by-play data, extracts per-player shot
statistics, builds opponent defensive profiles, and detects linemate
combinations.  All data is persisted to a local SQLite database.
"""

import sqlite3
import logging
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import nhl_api

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "nhl_data.db"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return a connection to the SQLite database, creating tables if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            game_id     INTEGER PRIMARY KEY,
            date        TEXT,
            home_team   TEXT,
            away_team   TEXT,
            home_score  INTEGER,
            away_score  INTEGER,
            status      TEXT
        );

        CREATE TABLE IF NOT EXISTS player_game_stats (
            game_id     INTEGER,
            player_id   INTEGER,
            player_name TEXT,
            team        TEXT,
            position    TEXT,
            shots       INTEGER,
            goals       INTEGER,
            assists     INTEGER,
            toi         REAL,
            is_home     INTEGER,
            pp_goals    INTEGER DEFAULT 0,
            shifts      INTEGER DEFAULT 0,
            takeaways   INTEGER DEFAULT 0,
            rest_days   INTEGER DEFAULT -1,
            PRIMARY KEY (game_id, player_id)
        );

        CREATE TABLE IF NOT EXISTS team_defense (
            team                     TEXT PRIMARY KEY,
            games_played             INTEGER,
            shots_allowed_per_game   REAL,
            shots_allowed_to_forwards REAL,
            shots_allowed_to_defense  REAL,
            shots_allowed_to_C       REAL DEFAULT 0,
            shots_allowed_to_L       REAL DEFAULT 0,
            shots_allowed_to_R       REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS linemates (
            game_id     INTEGER,
            player_id   INTEGER,
            linemate_id INTEGER,
            shared_toi_seconds REAL,
            PRIMARY KEY (game_id, player_id, linemate_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pgs_player ON player_game_stats(player_id);
        CREATE INDEX IF NOT EXISTS idx_pgs_team   ON player_game_stats(team);
        CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Time-on-ice parsing helper
# ---------------------------------------------------------------------------

def _parse_toi(toi_str: str) -> float:
    """Convert 'MM:SS' string to fractional minutes."""
    if not toi_str:
        return 0.0
    try:
        parts = toi_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# Season data collection
# ---------------------------------------------------------------------------

def collect_season_data(progress_callback=None):
    """
    Iterate through every team's schedule, find completed games, pull
    boxscores, and store per-player shot data.  Works incrementally --
    skips games already in the database.
    """
    conn = get_db()
    existing_game_ids = {
        row[0]
        for row in conn.execute("SELECT game_id FROM games").fetchall()
    }

    # Gather all game ids from every team schedule
    all_games: dict[int, dict] = {}
    teams = nhl_api.get_all_teams()
    for i, team in enumerate(teams):
        if progress_callback:
            progress_callback(f"Fetching schedule for {team} ({i+1}/{len(teams)})")
        sched = nhl_api.get_team_schedule(team)
        if not sched:
            continue
        for game in sched.get("games", []):
            gid = game.get("id")
            if gid and gid not in existing_game_ids and gid not in all_games:
                state = game.get("gameState", "")
                if state in ("FINAL", "OFF"):
                    all_games[gid] = game

    total = len(all_games)
    logger.info("Found %d new completed games to process", total)

    for idx, (gid, game_info) in enumerate(all_games.items()):
        if progress_callback and idx % 10 == 0:
            progress_callback(
                f"Processing boxscore {idx+1}/{total} (game {gid})"
            )
        _process_boxscore(conn, gid, game_info)

    conn.commit()

    # Compute rest days for all player-games
    if progress_callback:
        progress_callback("Computing rest days...")
    _compute_rest_days(conn)
    conn.commit()

    # After collecting game stats, rebuild defense profiles
    if progress_callback:
        progress_callback("Building team defense profiles...")
    build_all_defense_profiles(conn)
    conn.commit()
    conn.close()
    logger.info("Season data collection complete.")


def _process_boxscore(conn: sqlite3.Connection, game_id: int, game_info: dict):
    """Fetch a boxscore and insert game + player stat rows."""
    box = nhl_api.get_boxscore(game_id)
    if not box:
        return

    # Extract game-level info
    game_date = game_info.get("gameDate", "")
    home_abbrev = (
        box.get("homeTeam", {}).get("abbrev", "")
        or game_info.get("homeTeam", {}).get("abbrev", "")
    )
    away_abbrev = (
        box.get("awayTeam", {}).get("abbrev", "")
        or game_info.get("awayTeam", {}).get("abbrev", "")
    )
    home_score = box.get("homeTeam", {}).get("score", 0)
    away_score = box.get("awayTeam", {}).get("score", 0)

    try:
        conn.execute(
            """INSERT OR IGNORE INTO games
               (game_id, date, home_team, away_team, home_score, away_score, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (game_id, game_date, home_abbrev, away_abbrev,
             home_score, away_score, "FINAL"),
        )
    except sqlite3.Error as exc:
        logger.warning("Failed to insert game %s: %s", game_id, exc)
        return

    # Extract player stats from boxscore
    pbgs = box.get("playerByGameStats", {})
    for side, is_home in [("homeTeam", 1), ("awayTeam", 0)]:
        team_data = pbgs.get(side, {})
        team_abbrev = home_abbrev if is_home else away_abbrev

        for pos_group in ("forwards", "defense"):
            players = team_data.get(pos_group, [])
            for p in players:
                pid = p.get("playerId")
                if not pid:
                    continue
                if isinstance(p.get("name"), dict):
                    player_name = p["name"].get("default", str(pid))
                else:
                    player_name = p.get("name", str(pid))
                # Use granular position (L, C, R, D) from boxscore
                position = p.get("position", "F" if pos_group == "forwards" else "D")
                shots = p.get("sog", p.get("shots", 0)) or 0
                goals = p.get("goals", 0) or 0
                assists = p.get("assists", 0) or 0
                toi = _parse_toi(p.get("toi", "0:00"))
                pp_goals = p.get("powerPlayGoals", 0) or 0
                shifts = p.get("shifts", 0) or 0
                takeaways = p.get("takeaways", 0) or 0

                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO player_game_stats
                           (game_id, player_id, player_name, team, position,
                            shots, goals, assists, toi, is_home,
                            pp_goals, shifts, takeaways)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (game_id, pid, player_name, team_abbrev, position,
                         shots, goals, assists, toi, is_home,
                         pp_goals, shifts, takeaways),
                    )
                except sqlite3.Error:
                    pass


# ---------------------------------------------------------------------------
# Player game log
# ---------------------------------------------------------------------------

def build_player_game_log(player_id: int) -> pd.DataFrame:
    """
    Build a per-game SOG dataframe for a player from the local DB.
    Falls back to the NHL API game-log endpoint if DB data is sparse.
    """
    conn = get_db()
    df = pd.read_sql_query(
        """SELECT pgs.*, g.date, g.home_team, g.away_team
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.player_id = ?
           ORDER BY g.date""",
        conn,
        params=(player_id,),
    )
    conn.close()

    if df.empty:
        # Fallback: fetch from NHL API
        data = nhl_api.get_player_game_log(player_id)
        if data and "gameLog" in data:
            rows = []
            for entry in data["gameLog"]:
                rows.append({
                    "game_id": entry.get("gameId", 0),
                    "date": entry.get("gameDate", ""),
                    "shots": entry.get("shots", 0),
                    "goals": entry.get("goals", 0),
                    "assists": entry.get("assists", 0),
                    "toi": _parse_toi(entry.get("toi", "0:00")),
                    "team": entry.get("teamAbbrev", ""),
                    "home_team": entry.get("homeTeamAbbrev", ""),
                    "away_team": entry.get("awayTeamAbbrev", ""),
                    "position": "",
                    "is_home": 1 if entry.get("homeRoadFlag") == "H" else 0,
                })
            df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Opponent defense profiles
# ---------------------------------------------------------------------------

def build_opponent_defense_profile(team_abbrev: str, conn: Optional[sqlite3.Connection] = None) -> dict:
    """
    For a given team, calculate average shots allowed per game by position
    (C, L, R, D) using opponent skater stats from stored boxscores.
    """
    close_conn = False
    if conn is None:
        conn = get_db()
        close_conn = True

    rows = conn.execute(
        """SELECT pgs.game_id, pgs.position, pgs.shots
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE (g.home_team = ? AND pgs.is_home = 0)
              OR (g.away_team = ? AND pgs.is_home = 1)""",
        (team_abbrev, team_abbrev),
    ).fetchall()

    if close_conn:
        conn.close()

    if not rows:
        return {
            "team": team_abbrev,
            "games_played": 0,
            "shots_allowed_per_game": 0.0,
            "shots_allowed_to_forwards": 0.0,
            "shots_allowed_to_defense": 0.0,
            "shots_allowed_to_C": 0.0,
            "shots_allowed_to_L": 0.0,
            "shots_allowed_to_R": 0.0,
        }

    game_shots: dict[str, dict] = defaultdict(
        lambda: {"C": 0, "L": 0, "R": 0, "D": 0, "total": 0}
    )

    for r in rows:
        gid = str(r["game_id"])
        pos = r["position"]
        s = r["shots"] or 0
        game_shots[gid]["total"] += s
        if pos in ("C", "L", "R", "D"):
            game_shots[gid][pos] += s

    gp = len(game_shots)
    total_shots = sum(g["total"] for g in game_shots.values())
    c_shots = sum(g["C"] for g in game_shots.values())
    l_shots = sum(g["L"] for g in game_shots.values())
    r_shots = sum(g["R"] for g in game_shots.values())
    d_shots = sum(g["D"] for g in game_shots.values())
    fwd_shots = c_shots + l_shots + r_shots

    return {
        "team": team_abbrev,
        "games_played": gp,
        "shots_allowed_per_game": round(total_shots / gp, 2) if gp else 0.0,
        "shots_allowed_to_forwards": round(fwd_shots / gp, 2) if gp else 0.0,
        "shots_allowed_to_defense": round(d_shots / gp, 2) if gp else 0.0,
        "shots_allowed_to_C": round(c_shots / gp, 2) if gp else 0.0,
        "shots_allowed_to_L": round(l_shots / gp, 2) if gp else 0.0,
        "shots_allowed_to_R": round(r_shots / gp, 2) if gp else 0.0,
    }


def build_all_defense_profiles(conn: Optional[sqlite3.Connection] = None):
    """Rebuild team_defense table for every team."""
    close_conn = False
    if conn is None:
        conn = get_db()
        close_conn = True

    for team in nhl_api.get_all_teams():
        profile = build_opponent_defense_profile(team, conn)
        conn.execute(
            """INSERT OR REPLACE INTO team_defense
               (team, games_played, shots_allowed_per_game,
                shots_allowed_to_forwards, shots_allowed_to_defense,
                shots_allowed_to_C, shots_allowed_to_L, shots_allowed_to_R)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile["team"],
                profile["games_played"],
                profile["shots_allowed_per_game"],
                profile["shots_allowed_to_forwards"],
                profile["shots_allowed_to_defense"],
                profile["shots_allowed_to_C"],
                profile["shots_allowed_to_L"],
                profile["shots_allowed_to_R"],
            ),
        )
    conn.commit()
    if close_conn:
        conn.close()


# ---------------------------------------------------------------------------
# Linemate detection
# ---------------------------------------------------------------------------

def detect_linemates(game_id: int) -> list[dict]:
    """
    Parse play-by-play to find which forwards appear on ice together.
    Returns list of dicts with player_id, linemate_id, shared_toi_seconds.
    """
    pbp = nhl_api.get_play_by_play(game_id)
    if not pbp:
        return []

    # Track co-occurrences of forwards on ice
    forward_ids: set[int] = set()
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    # We need to know who is a forward. We'll track from the boxscore.
    box = nhl_api.get_boxscore(game_id)
    if not box:
        return []

    pbgs = box.get("playerByGameStats", {})
    for side in ("homeTeam", "awayTeam"):
        for p in pbgs.get(side, {}).get("forwards", []):
            pid = p.get("playerId")
            if pid:
                forward_ids.add(pid)

    # Walk through play-by-play events looking for playersOnIce
    plays = pbp.get("plays", [])
    for play in plays:
        # Check situationCode or playersOnIce in details
        # The NHL API sometimes puts onIce data at the event level
        for side_key in ("homeTeamDefendingSide",):
            pass  # not needed

        # Look for onIce arrays at play level
        home_on_ice = []
        away_on_ice = []

        # Some events have "players" with onIce info
        # The v1 API puts onIce in the play itself sometimes
        if "situationCode" in play:
            pass

        # Try to get from play details
        details = play.get("details", {})

        # NHL v1 pbp: each play may have homePlayersOnIce / awayPlayersOnIce
        for key, target in [
            ("homePlayersOnIce", home_on_ice),
            ("awayPlayersOnIce", away_on_ice),
        ]:
            poi = play.get(key, [])
            for player_info in poi:
                pid = player_info.get("playerId")
                if pid and pid in forward_ids:
                    target.append(pid)

        # Count co-occurrences among forwards on the same side
        for group in (home_on_ice, away_on_ice):
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = min(group[i], group[j]), max(group[i], group[j])
                    pair_counts[(a, b)] += 1

    # Convert counts to approximate shared seconds (each event ~ 1 observation)
    # This is a rough proxy; scale by average event interval
    total_events = len(plays) if plays else 1
    game_seconds = 3600  # 60-min game
    seconds_per_event = game_seconds / total_events

    results = []
    for (a, b), count in pair_counts.items():
        shared_sec = round(count * seconds_per_event, 1)
        results.append({
            "game_id": game_id,
            "player_id": a,
            "linemate_id": b,
            "shared_toi_seconds": shared_sec,
        })
        results.append({
            "game_id": game_id,
            "player_id": b,
            "linemate_id": a,
            "shared_toi_seconds": shared_sec,
        })

    # Store in DB
    conn = get_db()
    for r in results:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO linemates
                   (game_id, player_id, linemate_id, shared_toi_seconds)
                   VALUES (?, ?, ?, ?)""",
                (r["game_id"], r["player_id"], r["linemate_id"],
                 r["shared_toi_seconds"]),
            )
        except sqlite3.Error:
            pass
    conn.commit()
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Rest days computation
# ---------------------------------------------------------------------------

def _compute_rest_days(conn: sqlite3.Connection):
    """
    For each player-game, compute days since that player's previous game.
    0 = back-to-back, 1 = one day off, etc. -1 = first game of season.
    """
    rows = conn.execute(
        """SELECT pgs.rowid, pgs.player_id, g.date
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           ORDER BY pgs.player_id, g.date"""
    ).fetchall()

    updates = []
    prev_player = None
    prev_date = None
    for r in rows:
        pid = r["player_id"]
        game_date = r["date"]
        if pid != prev_player:
            rest = -1  # first game of season
        else:
            try:
                d1 = datetime.strptime(prev_date[:10], "%Y-%m-%d")
                d2 = datetime.strptime(game_date[:10], "%Y-%m-%d")
                rest = (d2 - d1).days - 1  # subtract 1: consecutive days = 0 rest
                rest = max(rest, 0)
            except (ValueError, TypeError):
                rest = -1
        updates.append((rest, r["rowid"]))
        prev_player = pid
        prev_date = game_date

    conn.executemany(
        "UPDATE player_game_stats SET rest_days = ? WHERE rowid = ?",
        updates,
    )


# ---------------------------------------------------------------------------
# Player predictability (coefficient of variation)
# ---------------------------------------------------------------------------

def get_player_predictability(min_games: int = 20, min_avg_sog: float = 0.0) -> pd.DataFrame:
    """
    Calculate coefficient of variation (CV = std/mean) for each player's SOG.
    Lower CV = more predictable. Returns DataFrame sorted by CV ascending.
    """
    conn = get_db()
    df = pd.read_sql_query(
        """SELECT player_id, player_name, team, position,
                  COUNT(*) as games,
                  AVG(shots) as avg_sog,
                  STDEV(shots) as std_sog
           FROM player_game_stats
           WHERE position IN ('L', 'C', 'R', 'D')
           GROUP BY player_id
           HAVING games >= ?""",
        conn,
        params=(min_games,),
    )
    conn.close()

    if df.empty:
        return df

    # SQLite doesn't have STDEV, so compute it manually
    conn = get_db()
    all_stats = pd.read_sql_query(
        """SELECT player_id, shots FROM player_game_stats
           WHERE position IN ('L', 'C', 'R', 'D')""",
        conn,
    )
    conn.close()

    player_stats = all_stats.groupby("player_id")["shots"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    player_stats.columns = ["player_id", "avg_sog", "std_sog", "games"]
    player_stats = player_stats[player_stats["games"] >= min_games].copy()
    player_stats["cv"] = (player_stats["std_sog"] / player_stats["avg_sog"]).round(3)
    player_stats = player_stats[player_stats["avg_sog"] >= min_avg_sog]

    # Merge names back
    conn = get_db()
    names = pd.read_sql_query(
        """SELECT DISTINCT player_id, player_name, team, position
           FROM player_game_stats""",
        conn,
    )
    conn.close()
    # Take the most recent name/team per player
    names = names.drop_duplicates(subset="player_id", keep="last")
    result = player_stats.merge(names, on="player_id", how="left")
    result = result.sort_values("cv", ascending=True)

    return result


# ---------------------------------------------------------------------------
# Rolling averages
# ---------------------------------------------------------------------------

def calculate_rolling_averages(
    player_game_log: pd.DataFrame,
    windows: list[int] = None,
) -> pd.DataFrame:
    """
    Given a player game log DataFrame (must have 'shots' column sorted by date),
    compute rolling average SOG for each window size.
    """
    if windows is None:
        windows = [3, 5, 10, 20]

    df = player_game_log.copy()
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)
    for w in windows:
        col = f"rolling_{w}"
        df[col] = df["shots"].rolling(window=w, min_periods=1).mean().round(2)

    return df
