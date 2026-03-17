"""
MLB data collection and processing module.

Pulls pitcher game logs from the MLB Stats API, fetches Statcast
pitch-level aggregates via pybaseball, and persists everything to
a local SQLite database.
"""

import sqlite3
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import mlb_api

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "mlb_data.db"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mlb_games (
            game_pk          INTEGER PRIMARY KEY,
            date             TEXT,
            home_team_id     INTEGER,
            home_team_abbrev TEXT,
            away_team_id     INTEGER,
            away_team_abbrev TEXT,
            home_score       INTEGER,
            away_score       INTEGER,
            status           TEXT
        );

        CREATE TABLE IF NOT EXISTS mlb_pitcher_game_stats (
            game_pk          INTEGER,
            pitcher_id       INTEGER,
            pitcher_name     TEXT,
            team_abbrev      TEXT,
            opponent_abbrev  TEXT,
            is_home          INTEGER,
            innings_pitched  REAL,
            strikeouts       INTEGER,
            batters_faced    INTEGER,
            pitches_thrown   INTEGER,
            walks            INTEGER,
            hits_allowed     INTEGER,
            earned_runs      INTEGER,
            home_runs_allowed INTEGER DEFAULT 0,
            pitch_hand       TEXT,
            days_rest        INTEGER DEFAULT -1,
            date             TEXT,
            PRIMARY KEY (game_pk, pitcher_id)
        );

        CREATE TABLE IF NOT EXISTS mlb_team_batting (
            team_abbrev      TEXT PRIMARY KEY,
            team_id          INTEGER,
            games_played     INTEGER,
            plate_appearances INTEGER,
            k_rate           REAL,
            bb_rate          REAL,
            avg              TEXT,
            obp              TEXT,
            slg              TEXT
        );

        CREATE TABLE IF NOT EXISTS mlb_park_factors (
            team_abbrev      TEXT PRIMARY KEY,
            park_name        TEXT,
            k_factor         REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS mlb_statcast_pitcher (
            pitcher_id       INTEGER,
            game_date        TEXT,
            csw_rate         REAL,
            whiff_rate       REAL,
            zone_rate        REAL,
            chase_rate       REAL,
            avg_velocity     REAL,
            pitches          INTEGER,
            PRIMARY KEY (pitcher_id, game_date)
        );

        -- Phase 2: Pitcher arsenal profiles (per pitch type)
        CREATE TABLE IF NOT EXISTS mlb_pitcher_arsenal (
            pitcher_id       INTEGER,
            pitch_type       TEXT,
            usage_pct        REAL,
            whiff_rate       REAL,
            csw_rate         REAL,
            chase_rate       REAL,
            zone_rate        REAL,
            avg_velocity     REAL,
            avg_spin_rate    REAL,
            avg_pfx_x        REAL,
            avg_pfx_z        REAL,
            pitches          INTEGER,
            updated_date     TEXT,
            PRIMARY KEY (pitcher_id, pitch_type)
        );

        -- Phase 2: Batter stats (season-level)
        CREATE TABLE IF NOT EXISTS mlb_batter_stats (
            batter_id        INTEGER PRIMARY KEY,
            batter_name      TEXT,
            team_abbrev      TEXT,
            bat_side         TEXT,
            games            INTEGER,
            plate_appearances INTEGER,
            k_rate           REAL,
            bb_rate          REAL,
            whiff_rate       REAL,
            chase_rate       REAL,
            contact_rate     REAL,
            xwoba            REAL,
            updated_date     TEXT
        );

        -- Phase 2: Batter performance vs each pitch type
        CREATE TABLE IF NOT EXISTS mlb_batter_vs_pitch (
            batter_id        INTEGER,
            pitch_type       TEXT,
            pitches_seen     INTEGER,
            whiff_rate       REAL,
            chase_rate       REAL,
            contact_rate     REAL,
            k_rate_vs_type   REAL,
            updated_date     TEXT,
            PRIMARY KEY (batter_id, pitch_type)
        );

        -- Phase 2: Pitcher TTO (times through order) decay profile
        CREATE TABLE IF NOT EXISTS mlb_pitcher_tto (
            pitcher_id       INTEGER,
            tto_number       INTEGER,
            k_rate           REAL,
            whiff_rate       REAL,
            avg_velocity     REAL,
            sample_pa        INTEGER,
            updated_date     TEXT,
            PRIMARY KEY (pitcher_id, tto_number)
        );

        -- Per-game granular tables (for temporal feature computation)
        CREATE TABLE IF NOT EXISTS mlb_pitcher_game_arsenal (
            pitcher_id   INTEGER,
            game_date    TEXT,
            pitch_type   TEXT,
            pitches      INTEGER,
            whiff_count  INTEGER,
            csw_count    INTEGER,
            chase_count  INTEGER,
            zone_count   INTEGER,
            swing_count  INTEGER,
            PRIMARY KEY (pitcher_id, game_date, pitch_type)
        );

        CREATE TABLE IF NOT EXISTS mlb_batter_game_pitch_type (
            batter_id    INTEGER,
            game_date    TEXT,
            team_abbrev  TEXT,
            pitch_type   TEXT,
            pitches_seen INTEGER,
            whiff_count  INTEGER,
            swing_count  INTEGER,
            contact_count INTEGER,
            chase_count  INTEGER,
            zone_count   INTEGER,
            pa_count     INTEGER DEFAULT 0,
            k_count      INTEGER DEFAULT 0,
            PRIMARY KEY (batter_id, game_date, pitch_type)
        );

        CREATE TABLE IF NOT EXISTS mlb_pitcher_game_tto (
            pitcher_id   INTEGER,
            game_date    TEXT,
            tto_number   INTEGER,
            pitches      INTEGER,
            whiff_count  INTEGER,
            swing_count  INTEGER,
            pa_count     INTEGER DEFAULT 0,
            k_count      INTEGER DEFAULT 0,
            PRIMARY KEY (pitcher_id, game_date, tto_number)
        );

        -- Per-game count-level features
        CREATE TABLE IF NOT EXISTS mlb_pitcher_game_counts (
            pitcher_id   INTEGER,
            game_date    TEXT,
            total_pa     INTEGER,
            first_pitch_strikes INTEGER,
            first_pitches INTEGER,
            two_strike_pa INTEGER,
            two_strike_k  INTEGER,
            zone_swings  INTEGER,
            zone_contacts INTEGER,
            PRIMARY KEY (pitcher_id, game_date)
        );

        CREATE INDEX IF NOT EXISTS idx_mlb_pgs_pitcher ON mlb_pitcher_game_stats(pitcher_id);
        CREATE INDEX IF NOT EXISTS idx_mlb_pgs_team ON mlb_pitcher_game_stats(team_abbrev);
        CREATE INDEX IF NOT EXISTS idx_mlb_games_date ON mlb_games(date);
        CREATE INDEX IF NOT EXISTS idx_mlb_arsenal_pitcher ON mlb_pitcher_arsenal(pitcher_id);
        CREATE INDEX IF NOT EXISTS idx_mlb_batter_team ON mlb_batter_stats(team_abbrev);
        CREATE INDEX IF NOT EXISTS idx_mlb_bvp_batter ON mlb_batter_vs_pitch(batter_id);
        CREATE INDEX IF NOT EXISTS idx_mlb_pga_pitcher ON mlb_pitcher_game_arsenal(pitcher_id);
        CREATE INDEX IF NOT EXISTS idx_mlb_bgpt_team ON mlb_batter_game_pitch_type(team_abbrev);
        CREATE INDEX IF NOT EXISTS idx_mlb_pgtto_pitcher ON mlb_pitcher_game_tto(pitcher_id);
        CREATE INDEX IF NOT EXISTS idx_mlb_pgc_pitcher ON mlb_pitcher_game_counts(pitcher_id);
    """)
    conn.commit()

    # Add columns to existing tables (safe if already exists)
    for stmt in [
        "ALTER TABLE mlb_pitcher_game_arsenal ADD COLUMN avg_velocity REAL",
    ]:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists


# ---------------------------------------------------------------------------
# Park factors (hardcoded, updated once per season)
# K-factor > 1.0 = pitcher-friendly for Ks, < 1.0 = hitter-friendly
# ---------------------------------------------------------------------------

PARK_K_FACTORS = {
    "ARI": 1.02, "ATL": 0.98, "BAL": 0.97, "BOS": 0.96, "CHC": 1.00,
    "CWS": 1.01, "CIN": 1.03, "CLE": 0.99, "COL": 0.94, "DET": 1.02,
    "HOU": 1.01, "KC": 0.98, "LAA": 1.00, "LAD": 1.01, "MIA": 1.04,
    "MIL": 1.02, "MIN": 1.01, "NYM": 1.03, "NYY": 0.99, "OAK": 1.00,
    "PHI": 1.01, "PIT": 1.00, "SD": 1.05, "SF": 1.03, "SEA": 1.02,
    "STL": 0.99, "TB": 1.03, "TEX": 0.98, "TOR": 1.01, "WSH": 1.00,
}


def _init_park_factors(conn: sqlite3.Connection):
    """Seed park factors into the database."""
    for team, kf in PARK_K_FACTORS.items():
        conn.execute(
            "INSERT OR REPLACE INTO mlb_park_factors (team_abbrev, k_factor) VALUES (?, ?)",
            (team, kf),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_season_data(progress_callback=None, season: int = None):
    """
    Pull pitcher game logs for all teams via the MLB Stats API.
    Works incrementally — skips games already in the database.
    """
    if season is None:
        # Use current year; if before April, also pull prior year
        season = date.today().year

    conn = get_db()
    _init_park_factors(conn)

    existing_games = {
        row[0] for row in conn.execute("SELECT game_pk FROM mlb_games").fetchall()
    }

    teams = mlb_api.get_all_team_ids()
    all_pitcher_ids = set()
    processed_games = set()
    total_new = 0

    # Step 1: Collect schedule data for recent dates
    if progress_callback:
        progress_callback("Fetching MLB schedule...")

    # Scan last 7 days + today for game results
    today = date.today()
    for days_back in range(0, 8):
        check_date = (today - timedelta(days=days_back)).isoformat()
        games = mlb_api.get_todays_schedule(check_date)
        for g in games:
            gpk = g["game_pk"]
            if gpk in existing_games or gpk in processed_games:
                continue
            if g["status"] != "Final":
                continue

            # Fetch boxscore to get actual scores
            box = mlb_api.get_game_boxscore(gpk)
            home_score = 0
            away_score = 0
            if box:
                home_score = box.get("teams", {}).get("home", {}).get("teamStats", {}).get("batting", {}).get("runs", 0)
                away_score = box.get("teams", {}).get("away", {}).get("teamStats", {}).get("batting", {}).get("runs", 0)

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO mlb_games
                       (game_pk, date, home_team_id, home_team_abbrev,
                        away_team_id, away_team_abbrev, home_score, away_score, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (gpk, check_date, g["home_team_id"], g["home_team_abbrev"],
                     g["away_team_id"], g["away_team_abbrev"], home_score, away_score, "Final"),
                )
                processed_games.add(gpk)
                total_new += 1
            except sqlite3.Error as exc:
                logger.warning("Failed to insert game %s: %s", gpk, exc)

    conn.commit()

    # Step 2: Collect pitcher game logs for all teams
    if progress_callback:
        progress_callback("Fetching pitcher game logs...")

    existing_pitcher_games = set()
    for row in conn.execute("SELECT game_pk, pitcher_id FROM mlb_pitcher_game_stats").fetchall():
        existing_pitcher_games.add((row[0], row[1]))

    for i, (abbrev, team_id) in enumerate(teams.items()):
        if progress_callback:
            progress_callback(f"Fetching pitchers for {abbrev} ({i+1}/{len(teams)})")

        # Get team roster — pitchers
        roster_data = mlb_api._get(
            f"{mlb_api.BASE_URL}/teams/{team_id}/roster",
            params={"rosterType": "active", "season": season},
            cache_ttl=3600,
        )
        if not roster_data:
            continue

        for player in roster_data.get("roster", []):
            pos = player.get("position", {}).get("abbreviation", "")
            if pos not in ("P", "SP", "RP"):
                continue

            pid = player.get("person", {}).get("id")
            pname = player.get("person", {}).get("fullName", "")
            phand = player.get("person", {}).get("pitchHand", {}).get("code", "R")
            if not pid:
                continue

            all_pitcher_ids.add(pid)

            # Get game log
            game_log = mlb_api.get_pitcher_game_log(pid, season)
            for gl in game_log:
                gpk = gl["game_pk"]
                if (gpk, pid) in existing_pitcher_games:
                    continue

                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO mlb_pitcher_game_stats
                           (game_pk, pitcher_id, pitcher_name, team_abbrev,
                            opponent_abbrev, is_home, innings_pitched, strikeouts,
                            batters_faced, pitches_thrown, walks, hits_allowed,
                            earned_runs, home_runs_allowed, pitch_hand, date, is_starter)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (gpk, pid, pname, abbrev,
                         gl["opponent_abbrev"], 1 if gl["is_home"] else 0,
                         gl["innings_pitched"], gl["strikeouts"],
                         gl["batters_faced"], gl["pitches_thrown"],
                         gl["walks"], gl["hits_allowed"],
                         gl["earned_runs"], gl.get("home_runs_allowed", 0),
                         phand, gl["date"], gl.get("is_starter", 0)),
                    )
                    existing_pitcher_games.add((gpk, pid))
                except sqlite3.Error as exc:
                    logger.warning("Failed to insert pitcher stats %s/%s: %s", gpk, pid, exc)

        conn.commit()

    # Step 3: Compute rest days
    if progress_callback:
        progress_callback("Computing rest days...")
    _compute_rest_days(conn)
    conn.commit()

    # Step 4: Collect team batting stats
    if progress_callback:
        progress_callback("Fetching team batting stats...")
    _collect_team_batting(conn, season)
    conn.commit()

    # Step 5: If early season (before May) and not much data, also pull prior year
    row_count = conn.execute("SELECT COUNT(*) FROM mlb_pitcher_game_stats").fetchone()[0]
    if row_count < 500 and date.today().month <= 5:
        prior_season = season - 1
        if progress_callback:
            progress_callback(f"Pulling {prior_season} data for model training...")
        for i, (abbrev, team_id) in enumerate(teams.items()):
            if progress_callback and i % 5 == 0:
                progress_callback(f"Fetching {prior_season} pitchers for {abbrev} ({i+1}/{len(teams)})")
            roster_data = mlb_api._get(
                f"{mlb_api.BASE_URL}/teams/{team_id}/roster",
                params={"rosterType": "active", "season": prior_season},
                cache_ttl=86400,
            )
            if not roster_data:
                continue
            for player in roster_data.get("roster", []):
                pos = player.get("position", {}).get("abbreviation", "")
                if pos not in ("P", "SP", "RP"):
                    continue
                pid = player.get("person", {}).get("id")
                pname = player.get("person", {}).get("fullName", "")
                if not pid:
                    continue
                game_log = mlb_api.get_pitcher_game_log(pid, prior_season)
                for gl in game_log:
                    gpk = gl["game_pk"]
                    if (gpk, pid) in existing_pitcher_games:
                        continue
                    try:
                        # Insert game record
                        conn.execute(
                            """INSERT OR IGNORE INTO mlb_games
                               (game_pk, date, home_team_id, home_team_abbrev,
                                away_team_id, away_team_abbrev, status)
                               VALUES (?, ?, 0, ?, 0, ?, 'Final')""",
                            (gpk, gl["date"],
                             abbrev if gl["is_home"] else gl["opponent_abbrev"],
                             gl["opponent_abbrev"] if gl["is_home"] else abbrev),
                        )
                        conn.execute(
                            """INSERT OR IGNORE INTO mlb_pitcher_game_stats
                               (game_pk, pitcher_id, pitcher_name, team_abbrev,
                                opponent_abbrev, is_home, innings_pitched, strikeouts,
                                batters_faced, pitches_thrown, walks, hits_allowed,
                                earned_runs, home_runs_allowed, pitch_hand, date, is_starter)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (gpk, pid, pname, abbrev,
                             gl["opponent_abbrev"], 1 if gl["is_home"] else 0,
                             gl["innings_pitched"], gl["strikeouts"],
                             gl["batters_faced"], gl["pitches_thrown"],
                             gl["walks"], gl["hits_allowed"],
                             gl["earned_runs"], gl.get("home_runs_allowed", 0),
                             "R", gl["date"], gl.get("is_starter", 0)),
                        )
                        existing_pitcher_games.add((gpk, pid))
                    except sqlite3.Error:
                        pass
            conn.commit()

        # Recompute rest days with prior year data
        _compute_rest_days(conn)
        conn.commit()

    # Step 6: Try Statcast data (optional — may fail on Render)
    # If we pulled prior season data, also pull full Statcast for that season
    statcast_season = 0
    statcast_count = conn.execute("SELECT COUNT(*) FROM mlb_statcast_pitcher").fetchone()[0]
    if statcast_count < 100 and date.today().month <= 5:
        statcast_season = season - 1  # Full 2024 season Statcast
    try:
        if progress_callback:
            if statcast_season:
                progress_callback(f"Fetching full {statcast_season} Statcast data (this will take a while)...")
            else:
                progress_callback("Fetching Statcast data...")
        _collect_statcast_pitching(conn, full_season=statcast_season)
        conn.commit()
    except Exception as exc:
        logger.warning("Statcast collection failed (non-critical): %s", exc)

    conn.close()
    logger.info("MLB data collection complete. %d new games.", total_new)


def _compute_rest_days(conn: sqlite3.Connection):
    """Compute days of rest between starts for each pitcher."""
    pitchers = conn.execute(
        "SELECT DISTINCT pitcher_id FROM mlb_pitcher_game_stats"
    ).fetchall()

    for row in pitchers:
        pid = row[0]
        games = conn.execute(
            """SELECT game_pk, date FROM mlb_pitcher_game_stats
               WHERE pitcher_id = ? AND innings_pitched >= 3.0
               ORDER BY date""",
            (pid,),
        ).fetchall()

        prev_date = None
        for g in games:
            if prev_date and g["date"]:
                try:
                    d1 = datetime.strptime(g["date"][:10], "%Y-%m-%d")
                    d0 = datetime.strptime(prev_date[:10], "%Y-%m-%d")
                    rest = max((d1 - d0).days - 1, 0)
                    conn.execute(
                        "UPDATE mlb_pitcher_game_stats SET days_rest = ? WHERE game_pk = ? AND pitcher_id = ?",
                        (rest, g["game_pk"], pid),
                    )
                except (ValueError, TypeError):
                    pass
            prev_date = g["date"]


def _collect_team_batting(conn: sqlite3.Connection, season: int):
    """Fetch team-level batting stats."""
    teams = mlb_api.get_all_team_ids()
    for abbrev, team_id in teams.items():
        stats = mlb_api.get_team_batting_stats(team_id, season)
        if not stats:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO mlb_team_batting
               (team_abbrev, team_id, games_played, plate_appearances,
                k_rate, bb_rate, avg, obp, slg)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (abbrev, team_id, stats["games_played"], stats["plate_appearances"],
             stats["k_rate"], stats["bb_rate"], stats["avg"], stats["obp"], stats["slg"]),
        )


def _collect_statcast_pitching(conn: sqlite3.Connection, full_season: int = 0):
    """Pull Statcast pitch-level data and build all pitch-level tables.

    Args:
        full_season: If > 0, pull entire season (e.g. 2024). Otherwise incremental (14 days).
    """
    try:
        from pybaseball import statcast
    except ImportError:
        logger.info("pybaseball not installed — skipping Statcast data")
        return

    if full_season > 0:
        start_date = date(full_season, 3, 20)  # Spring training starts
        end_date = date(full_season, 11, 5)     # After World Series
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=14)

    logger.info("Pulling Statcast data from %s to %s", start_date, end_date)

    # Pull in chunks to avoid timeouts (pybaseball can choke on large ranges)
    chunk_days = 7 if full_season else 14
    all_dfs = []
    current = start_date
    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_days), end_date)
        try:
            logger.info("  Statcast chunk: %s to %s", current, chunk_end)
            chunk_df = statcast(
                start_dt=current.strftime("%Y-%m-%d"),
                end_dt=chunk_end.strftime("%Y-%m-%d"),
            )
            if chunk_df is not None and not chunk_df.empty:
                all_dfs.append(chunk_df)
        except Exception as exc:
            logger.warning("Statcast chunk %s-%s failed: %s", current, chunk_end, exc)
        current = chunk_end + timedelta(days=1)

    if not all_dfs:
        logger.warning("No Statcast data retrieved")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total Statcast pitches: %d", len(df))

    if df.empty:
        return

    # --- Common derived columns ---
    # Convert nullable Int64 types to regular float for compatibility
    if "zone" in df.columns:
        df["zone"] = pd.to_numeric(df["zone"], errors="coerce")
    if "at_bat_number" in df.columns:
        df["at_bat_number"] = pd.to_numeric(df["at_bat_number"], errors="coerce").fillna(1)
    if "pitcher" in df.columns:
        df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
    if "batter" in df.columns:
        df["batter"] = pd.to_numeric(df["batter"], errors="coerce")
    # Convert game_date to string for groupby
    df["game_date"] = df["game_date"].astype(str).str[:10]
    # Normalize events column (replace empty strings with NaN)
    df["events"] = df["events"].replace("", np.nan)

    df["is_whiff"] = df["description"].isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])
    df["is_called_strike"] = df["description"] == "called_strike"
    df["is_csw"] = df["is_whiff"] | df["is_called_strike"]
    df["in_zone"] = df["zone"].between(1, 9).fillna(False)
    df["is_swing"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
        "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    ])
    df["is_contact"] = df["is_swing"] & (~df["is_whiff"])
    df["is_chase"] = (~df["in_zone"]) & df["is_swing"]
    df["is_strikeout"] = df["events"] == "strikeout"
    df["is_pa_end"] = df["events"].notna()

    today_str = date.today().isoformat()

    # --- 1) Per-pitcher-per-game aggregates ---
    existing_pg = set()
    for row in conn.execute("SELECT pitcher_id, game_date FROM mlb_statcast_pitcher").fetchall():
        existing_pg.add((row[0], row[1]))

    grouped = df.groupby(["pitcher", "game_date"]).agg(
        pitches=("pitch_type", "count"),
        csw_count=("is_csw", "sum"),
        whiff_count=("is_whiff", "sum"),
        zone_count=("in_zone", "sum"),
        chase_count=("is_chase", "sum"),
        avg_velocity=("release_speed", "mean"),
    ).reset_index()

    grouped["csw_rate"] = grouped["csw_count"] / grouped["pitches"]
    grouped["whiff_rate"] = grouped["whiff_count"] / grouped["pitches"]
    grouped["zone_rate"] = grouped["zone_count"] / grouped["pitches"]
    out_of_zone = grouped["pitches"] - grouped["zone_count"]
    grouped["chase_rate"] = np.where(out_of_zone > 0, grouped["chase_count"] / out_of_zone, 0)

    for _, row in grouped.iterrows():
        pid = int(row["pitcher"])
        gdate = str(row["game_date"])[:10]
        if (pid, gdate) in existing_pg:
            continue
        try:
            conn.execute(
                """INSERT OR IGNORE INTO mlb_statcast_pitcher
                   (pitcher_id, game_date, csw_rate, whiff_rate, zone_rate,
                    chase_rate, avg_velocity, pitches)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, gdate,
                 _safe_round(row["csw_rate"]), _safe_round(row["whiff_rate"]),
                 _safe_round(row["zone_rate"]), _safe_round(row["chase_rate"]),
                 _safe_round(row["avg_velocity"], 1),
                 int(row["pitches"])),
            )
        except sqlite3.Error:
            pass

    logger.info("Statcast: %d pitcher-game rows processed", len(grouped))

    # --- 2) Pitcher arsenal profiles (per pitch type) ---
    logger.info("Building pitcher arsenal profiles...")
    arsenal = df.groupby(["pitcher", "pitch_type"]).agg(
        pitches=("pitch_type", "count"),
        whiff_count=("is_whiff", "sum"),
        csw_count=("is_csw", "sum"),
        chase_count=("is_chase", "sum"),
        zone_count=("in_zone", "sum"),
        swing_count=("is_swing", "sum"),
        avg_velocity=("release_speed", "mean"),
        avg_spin=("release_spin_rate", "mean"),
        avg_pfx_x=("pfx_x", "mean"),
        avg_pfx_z=("pfx_z", "mean"),
    ).reset_index()

    # Per-pitcher total pitches for usage %
    pitcher_totals = arsenal.groupby("pitcher")["pitches"].sum().to_dict()

    for _, row in arsenal.iterrows():
        pid = int(row["pitcher"])
        pt = row["pitch_type"]
        if pd.isna(pt) or not pt or row["pitches"] < 10:
            continue
        total = pitcher_totals.get(pid, 1)
        swings = max(row["swing_count"], 1)
        ooz = max(row["pitches"] - row["zone_count"], 1)

        try:
            conn.execute(
                """INSERT OR REPLACE INTO mlb_pitcher_arsenal
                   (pitcher_id, pitch_type, usage_pct, whiff_rate, csw_rate,
                    chase_rate, zone_rate, avg_velocity, avg_spin_rate,
                    avg_pfx_x, avg_pfx_z, pitches, updated_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, pt,
                 _safe_round(row["pitches"] / total),
                 _safe_round(row["whiff_count"] / swings),
                 _safe_round(row["csw_count"] / row["pitches"]),
                 _safe_round(row["chase_count"] / ooz),
                 _safe_round(row["zone_count"] / row["pitches"]),
                 _safe_round(row["avg_velocity"], 1),
                 _safe_round(row["avg_spin"], 0),
                 _safe_round(row["avg_pfx_x"], 2),
                 _safe_round(row["avg_pfx_z"], 2),
                 int(row["pitches"]),
                 today_str),
            )
        except sqlite3.Error:
            pass

    logger.info("Arsenal: %d pitcher-pitch_type rows", len(arsenal))

    # --- 2b) Per-game arsenal (raw counts for temporal features) ---
    logger.info("Building per-game arsenal profiles...")
    game_arsenal = df.groupby(["pitcher", "game_date", "pitch_type"]).agg(
        pitches=("pitch_type", "count"),
        whiff_count=("is_whiff", "sum"),
        csw_count=("is_csw", "sum"),
        chase_count=("is_chase", "sum"),
        zone_count=("in_zone", "sum"),
        swing_count=("is_swing", "sum"),
        avg_velocity=("release_speed", "mean"),
    ).reset_index()

    for _, row in game_arsenal.iterrows():
        pid = int(row["pitcher"])
        gdate = str(row["game_date"])[:10]
        pt = row["pitch_type"]
        if pd.isna(pt) or not pt:
            continue
        try:
            conn.execute(
                """INSERT OR IGNORE INTO mlb_pitcher_game_arsenal
                   (pitcher_id, game_date, pitch_type, pitches, whiff_count,
                    csw_count, chase_count, zone_count, swing_count, avg_velocity)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, gdate, pt, int(row["pitches"]), int(row["whiff_count"]),
                 int(row["csw_count"]), int(row["chase_count"]),
                 int(row["zone_count"]), int(row["swing_count"]),
                 _safe_round(row["avg_velocity"], 1)),
            )
        except sqlite3.Error:
            pass

    conn.commit()
    logger.info("Per-game arsenal: %d rows", len(game_arsenal))

    # --- 3) Batter stats (season aggregate from Statcast) ---
    logger.info("Building batter profiles...")
    batter_pa = df[df["is_pa_end"]].copy()
    if not batter_pa.empty:
        batter_agg = batter_pa.groupby("batter").agg(
            plate_appearances=("is_pa_end", "sum"),
            strikeouts=("is_strikeout", "sum"),
        ).reset_index()

        # Full pitch data for whiff/chase/contact
        batter_pitch_agg = df.groupby("batter").agg(
            pitches=("pitch_type", "count"),
            whiff_count=("is_whiff", "sum"),
            swing_count=("is_swing", "sum"),
            contact_count=("is_contact", "sum"),
            chase_count=("is_chase", "sum"),
            zone_count=("in_zone", "sum"),
        ).reset_index()

        batter_merged = batter_agg.merge(batter_pitch_agg, on="batter", how="left")

        for _, row in batter_merged.iterrows():
            bid = int(row["batter"])
            pa = max(int(row["plate_appearances"]), 1)
            swings = max(int(row["swing_count"]), 1)
            ooz = max(int(row["pitches"] - row["zone_count"]), 1)

            # Get batter name and team from Statcast
            batter_info = df[df["batter"] == bid].iloc[0] if len(df[df["batter"] == bid]) > 0 else None
            bname = ""
            bteam = ""
            bside = ""
            if batter_info is not None:
                # player_name in statcast is "Last, First"
                bname = str(batter_info.get("player_name", ""))
                bteam = str(batter_info.get("home_team", "")) if batter_info.get("inning_topbot") == "Bot" else str(batter_info.get("away_team", ""))
                bside = str(batter_info.get("stand", "R"))

            try:
                conn.execute(
                    """INSERT OR REPLACE INTO mlb_batter_stats
                       (batter_id, batter_name, team_abbrev, bat_side, games,
                        plate_appearances, k_rate, bb_rate, whiff_rate,
                        chase_rate, contact_rate, xwoba, updated_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (bid, bname, bteam, bside, 0, pa,
                     _safe_round(row["strikeouts"] / pa),
                     0.0,  # BB rate not easily available from pitches
                     _safe_round(row["whiff_count"] / swings),
                     _safe_round(row["chase_count"] / ooz),
                     _safe_round(row["contact_count"] / swings),
                     0.0,  # xwOBA from expected_woba column if available
                     today_str),
                )
            except sqlite3.Error:
                pass

        logger.info("Batter stats: %d batters", len(batter_merged))

    # --- 4) Batter vs pitch type ---
    logger.info("Building batter vs pitch type profiles...")
    bvp = df.groupby(["batter", "pitch_type"]).agg(
        pitches_seen=("pitch_type", "count"),
        whiff_count=("is_whiff", "sum"),
        swing_count=("is_swing", "sum"),
        contact_count=("is_contact", "sum"),
        chase_count=("is_chase", "sum"),
        zone_count=("in_zone", "sum"),
    ).reset_index()

    # Also need K rate per pitch type — count plate appearances ending in K per pitch type
    # Approximation: use at-bat-ending pitches
    bvp_k = df[df["is_pa_end"]].groupby(["batter", "pitch_type"]).agg(
        pa_count=("is_pa_end", "sum"),
        k_count=("is_strikeout", "sum"),
    ).reset_index()

    bvp = bvp.merge(bvp_k, on=["batter", "pitch_type"], how="left")
    bvp["pa_count"] = bvp["pa_count"].fillna(0)
    bvp["k_count"] = bvp["k_count"].fillna(0)

    for _, row in bvp.iterrows():
        bid = int(row["batter"])
        pt = row["pitch_type"]
        if pd.isna(pt) or not pt or row["pitches_seen"] < 5:
            continue
        swings = max(int(row["swing_count"]), 1)
        ooz = max(int(row["pitches_seen"] - row["zone_count"]), 1)
        pa = max(int(row["pa_count"]), 1)

        try:
            conn.execute(
                """INSERT OR REPLACE INTO mlb_batter_vs_pitch
                   (batter_id, pitch_type, pitches_seen, whiff_rate,
                    chase_rate, contact_rate, k_rate_vs_type, updated_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (bid, pt, int(row["pitches_seen"]),
                 _safe_round(row["whiff_count"] / swings),
                 _safe_round(row["chase_count"] / ooz),
                 _safe_round(row["contact_count"] / swings),
                 _safe_round(row["k_count"] / pa),
                 today_str),
            )
        except sqlite3.Error:
            pass

    logger.info("Batter vs pitch type: %d rows", len(bvp))

    # --- 4b) Per-game batter vs pitch type (raw counts for temporal features) ---
    logger.info("Building per-game batter pitch type profiles...")
    df["batter_team"] = np.where(
        df["inning_topbot"] == "Bot",
        df["home_team"],
        df["away_team"],
    )

    batter_game_pt = df.groupby(["batter", "game_date", "batter_team", "pitch_type"]).agg(
        pitches_seen=("pitch_type", "count"),
        whiff_count=("is_whiff", "sum"),
        swing_count=("is_swing", "sum"),
        contact_count=("is_contact", "sum"),
        chase_count=("is_chase", "sum"),
        zone_count=("in_zone", "sum"),
    ).reset_index()

    batter_game_pa = df[df["is_pa_end"]].groupby(["batter", "game_date", "pitch_type"]).agg(
        pa_count=("is_pa_end", "sum"),
        k_count=("is_strikeout", "sum"),
    ).reset_index()

    batter_game_pt = batter_game_pt.merge(
        batter_game_pa, on=["batter", "game_date", "pitch_type"], how="left"
    )
    batter_game_pt["pa_count"] = batter_game_pt["pa_count"].fillna(0).astype(int)
    batter_game_pt["k_count"] = batter_game_pt["k_count"].fillna(0).astype(int)

    for _, row in batter_game_pt.iterrows():
        bid = int(row["batter"])
        gdate = str(row["game_date"])[:10]
        team = str(row["batter_team"])
        pt = row["pitch_type"]
        if pd.isna(pt) or not pt or pd.isna(team) or not team:
            continue
        try:
            conn.execute(
                """INSERT OR IGNORE INTO mlb_batter_game_pitch_type
                   (batter_id, game_date, team_abbrev, pitch_type, pitches_seen,
                    whiff_count, swing_count, contact_count, chase_count,
                    zone_count, pa_count, k_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (bid, gdate, team, pt, int(row["pitches_seen"]),
                 int(row["whiff_count"]), int(row["swing_count"]),
                 int(row["contact_count"]), int(row["chase_count"]),
                 int(row["zone_count"]), int(row["pa_count"]),
                 int(row["k_count"])),
            )
        except sqlite3.Error:
            pass

    conn.commit()
    logger.info("Per-game batter pitch type: %d rows", len(batter_game_pt))

    # --- 5) Pitcher TTO (times through order) profiles ---
    logger.info("Building pitcher TTO profiles...")
    # Estimate TTO from at_bat_number: 1-9 = 1st, 10-18 = 2nd, 19+ = 3rd
    df["tto"] = np.where(df["at_bat_number"] <= 9, 1,
                np.where(df["at_bat_number"] <= 18, 2, 3))

    tto_agg = df.groupby(["pitcher", "tto"]).agg(
        pitches=("pitch_type", "count"),
        whiff_count=("is_whiff", "sum"),
        swing_count=("is_swing", "sum"),
        avg_velocity=("release_speed", "mean"),
    ).reset_index()

    tto_pa = df[df["is_pa_end"]].groupby(["pitcher", "tto"]).agg(
        pa_count=("is_pa_end", "sum"),
        k_count=("is_strikeout", "sum"),
    ).reset_index()

    tto_merged = tto_agg.merge(tto_pa, on=["pitcher", "tto"], how="left")
    tto_merged["pa_count"] = tto_merged["pa_count"].fillna(0)
    tto_merged["k_count"] = tto_merged["k_count"].fillna(0)

    for _, row in tto_merged.iterrows():
        pid = int(row["pitcher"])
        tto = int(row["tto"])
        pa = max(int(row["pa_count"]), 1)
        swings = max(int(row["swing_count"]), 1)

        try:
            conn.execute(
                """INSERT OR REPLACE INTO mlb_pitcher_tto
                   (pitcher_id, tto_number, k_rate, whiff_rate,
                    avg_velocity, sample_pa, updated_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (pid, tto,
                 _safe_round(row["k_count"] / pa),
                 _safe_round(row["whiff_count"] / swings),
                 _safe_round(row["avg_velocity"], 1),
                 int(row["pa_count"]),
                 today_str),
            )
        except sqlite3.Error:
            pass

    logger.info("TTO: %d pitcher-tto rows", len(tto_merged))

    # --- 5b) Per-game TTO (raw counts for temporal features) ---
    logger.info("Building per-game TTO profiles...")
    game_tto = df.groupby(["pitcher", "game_date", "tto"]).agg(
        pitches=("pitch_type", "count"),
        whiff_count=("is_whiff", "sum"),
        swing_count=("is_swing", "sum"),
    ).reset_index()

    game_tto_pa = df[df["is_pa_end"]].groupby(["pitcher", "game_date", "tto"]).agg(
        pa_count=("is_pa_end", "sum"),
        k_count=("is_strikeout", "sum"),
    ).reset_index()

    game_tto = game_tto.merge(
        game_tto_pa, on=["pitcher", "game_date", "tto"], how="left"
    )
    game_tto["pa_count"] = game_tto["pa_count"].fillna(0).astype(int)
    game_tto["k_count"] = game_tto["k_count"].fillna(0).astype(int)

    for _, row in game_tto.iterrows():
        pid = int(row["pitcher"])
        gdate = str(row["game_date"])[:10]
        tto_num = int(row["tto"])
        try:
            conn.execute(
                """INSERT OR IGNORE INTO mlb_pitcher_game_tto
                   (pitcher_id, game_date, tto_number, pitches, whiff_count,
                    swing_count, pa_count, k_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, gdate, tto_num, int(row["pitches"]),
                 int(row["whiff_count"]), int(row["swing_count"]),
                 int(row["pa_count"]), int(row["k_count"])),
            )
        except sqlite3.Error:
            pass

    conn.commit()
    logger.info("Per-game TTO: %d rows", len(game_tto))

    # --- 6) Per-game count-level features ---
    logger.info("Building per-game count-level features...")

    # Convert pitch_number to numeric
    if "pitch_number" in df.columns:
        df["pitch_number"] = pd.to_numeric(df["pitch_number"], errors="coerce").fillna(1)
    if "balls" in df.columns:
        df["balls"] = pd.to_numeric(df["balls"], errors="coerce").fillna(0)
    if "strikes" in df.columns:
        df["strikes"] = pd.to_numeric(df["strikes"], errors="coerce").fillna(0)

    # First pitch of each PA
    first_pitch = df[df["pitch_number"] == 1].copy()
    first_pitch["is_first_strike"] = first_pitch["type"].isin(["S"])

    first_pitch_agg = first_pitch.groupby(["pitcher", "game_date"]).agg(
        first_pitches=("pitch_number", "count"),
        first_pitch_strikes=("is_first_strike", "sum"),
    ).reset_index()

    # Two-strike data: at-bats that reached 2 strikes
    two_strike_abs = df[df["strikes"] >= 2].groupby(
        ["pitcher", "game_date"]
    )["at_bat_number"].nunique().reset_index(name="two_strike_pa")

    # K's from two-strike counts
    two_strike_k_df = df[
        (df["strikes"] >= 2) & (df["is_strikeout"])
    ].groupby(["pitcher", "game_date"])["at_bat_number"].nunique().reset_index(
        name="two_strike_k"
    )

    # Zone contact
    zone_pitches = df[df["in_zone"]].groupby(["pitcher", "game_date"]).agg(
        zone_swings=("is_swing", "sum"),
        zone_contacts=("is_contact", "sum"),
    ).reset_index()

    # Total PA per pitcher per game
    total_pa_df = df[df["is_pa_end"]].groupby(
        ["pitcher", "game_date"]
    )["at_bat_number"].nunique().reset_index(name="total_pa")

    # Merge all count features
    count_features = first_pitch_agg.merge(
        two_strike_abs, on=["pitcher", "game_date"], how="left"
    ).merge(
        two_strike_k_df, on=["pitcher", "game_date"], how="left"
    ).merge(
        zone_pitches, on=["pitcher", "game_date"], how="left"
    ).merge(
        total_pa_df, on=["pitcher", "game_date"], how="left"
    )

    for col in ["two_strike_pa", "two_strike_k", "zone_swings", "zone_contacts", "total_pa"]:
        count_features[col] = count_features[col].fillna(0).astype(int)

    for _, row in count_features.iterrows():
        pid = int(row["pitcher"])
        gdate = str(row["game_date"])[:10]
        try:
            conn.execute(
                """INSERT OR IGNORE INTO mlb_pitcher_game_counts
                   (pitcher_id, game_date, total_pa, first_pitch_strikes,
                    first_pitches, two_strike_pa, two_strike_k,
                    zone_swings, zone_contacts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, gdate,
                 int(row["total_pa"]),
                 int(row["first_pitch_strikes"]),
                 int(row["first_pitches"]),
                 int(row["two_strike_pa"]),
                 int(row["two_strike_k"]),
                 int(row["zone_swings"]),
                 int(row["zone_contacts"])),
            )
        except sqlite3.Error:
            pass

    conn.commit()
    logger.info("Per-game count features: %d rows", len(count_features))


def _safe_round(val, decimals=4):
    """Safely round a value, handling NaN/None/NAType/Inf."""
    if val is None:
        return None
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(fval) or np.isinf(fval):
        return None
    return round(fval, decimals)


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_pitcher_game_log(pitcher_id: int) -> pd.DataFrame:
    """Build a DataFrame of all starts for a given pitcher."""
    conn = get_db()
    df = pd.read_sql_query(
        """SELECT pgs.*, g.home_team_abbrev, g.away_team_abbrev
           FROM mlb_pitcher_game_stats pgs
           JOIN mlb_games g ON pgs.game_pk = g.game_pk
           WHERE pgs.pitcher_id = ?
           ORDER BY pgs.date""",
        conn,
        params=(pitcher_id,),
    )
    conn.close()
    return df


def get_pitcher_rolling_stats(pitcher_id: int) -> dict:
    """Get rolling averages for a pitcher (for prediction context)."""
    conn = get_db()
    rows = conn.execute(
        """SELECT strikeouts, innings_pitched, batters_faced, pitches_thrown,
                  walks, hits_allowed, earned_runs, days_rest, date
           FROM mlb_pitcher_game_stats
           WHERE pitcher_id = ? AND innings_pitched >= 3.0
           ORDER BY date DESC""",
        (pitcher_id,),
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    k_list = [r["strikeouts"] for r in rows]
    ip_list = [r["innings_pitched"] for r in rows]
    bf_list = [r["batters_faced"] for r in rows]
    pc_list = [r["pitches_thrown"] for r in rows]

    def _avg(data, n):
        subset = data[:n]
        return np.mean(subset) if subset else 0.0

    season_k = np.mean(k_list)
    return {
        "season_avg_k": round(season_k, 2),
        "rolling_3_k": round(_avg(k_list, 3), 2),
        "rolling_5_k": round(_avg(k_list, 5), 2),
        "rolling_10_k": round(_avg(k_list, 10), 2),
        "season_avg_ip": round(np.mean(ip_list), 2),
        "rolling_5_ip": round(_avg(ip_list, 5), 2),
        "season_avg_bf": round(np.mean(bf_list), 2),
        "season_avg_pc": round(np.mean(pc_list), 1),
        "k_per_9": round(season_k / max(np.mean(ip_list), 0.1) * 9, 2),
        "k_rate": round(season_k / max(np.mean(bf_list), 1), 4),
        "bb_rate": round(np.mean([r["walks"] for r in rows]) / max(np.mean(bf_list), 1), 4),
        "games": len(k_list),
        "last_rest": rows[0]["days_rest"] if rows[0]["days_rest"] >= 0 else -1,
        "pitches_last": pc_list[0] if pc_list else 0,
        "innings_last": ip_list[0] if ip_list else 0,
        "bf_last": bf_list[0] if bf_list else 0,
        "rolling_3_bf": round(_avg(bf_list, 3), 2),
        "rolling_5_bf": round(_avg(bf_list, 5), 2),
        "season_avg_bf": round(np.mean(bf_list), 2) if bf_list else 0,
        "std_k": round(float(np.std(k_list)), 2) if len(k_list) >= 5 else 0,
        "var_ratio_k": round(float(np.var(k_list) / max(season_k, 0.1)), 3) if len(k_list) >= 5 else 1.0,
    }


def get_statcast_for_pitcher(pitcher_id: int) -> dict:
    """Get latest Statcast aggregates for a pitcher."""
    conn = get_db()
    rows = conn.execute(
        """SELECT csw_rate, whiff_rate, zone_rate, chase_rate, avg_velocity
           FROM mlb_statcast_pitcher
           WHERE pitcher_id = ?
           ORDER BY game_date DESC
           LIMIT 5""",
        (pitcher_id,),
    ).fetchall()

    # Count-level features
    count_rows = conn.execute(
        """SELECT first_pitch_strikes, first_pitches,
                  two_strike_pa, two_strike_k,
                  zone_swings, zone_contacts
           FROM mlb_pitcher_game_counts
           WHERE pitcher_id = ?
           ORDER BY game_date DESC
           LIMIT 10""",
        (pitcher_id,),
    ).fetchall()

    conn.close()

    result = {}
    if rows:
        def _safe_mean(vals):
            clean = [v for v in vals if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
            return round(np.mean(clean), 4) if clean else 0.0

        result["csw_rate"] = _safe_mean([r["csw_rate"] for r in rows])
        result["whiff_rate"] = _safe_mean([r["whiff_rate"] for r in rows])
        result["zone_rate"] = _safe_mean([r["zone_rate"] for r in rows])
        result["chase_rate"] = _safe_mean([r["chase_rate"] for r in rows])
        result["avg_velocity"] = round(_safe_mean([r["avg_velocity"] for r in rows]), 1)

    if count_rows:
        total_fps = sum(r["first_pitches"] or 0 for r in count_rows)
        total_fps_k = sum(r["first_pitch_strikes"] or 0 for r in count_rows)
        total_2s = sum(r["two_strike_pa"] or 0 for r in count_rows)
        total_2s_k = sum(r["two_strike_k"] or 0 for r in count_rows)
        total_zsw = sum(r["zone_swings"] or 0 for r in count_rows)
        total_zct = sum(r["zone_contacts"] or 0 for r in count_rows)

        result["first_pitch_strike_rate"] = round(total_fps_k / max(total_fps, 1), 4)
        result["two_strike_putaway_rate"] = round(total_2s_k / max(total_2s, 1), 4)
        result["zone_contact_rate"] = round(total_zct / max(total_zsw, 1), 4)

    return result


def get_pitcher_arsenal(pitcher_id: int) -> list[dict]:
    """Get pitcher's arsenal profile (per pitch type stats)."""
    conn = get_db()
    rows = conn.execute(
        """SELECT pitch_type, usage_pct, whiff_rate, csw_rate, chase_rate,
                  zone_rate, avg_velocity, avg_spin_rate, avg_pfx_x, avg_pfx_z, pitches
           FROM mlb_pitcher_arsenal
           WHERE pitcher_id = ? AND pitches >= 10
           ORDER BY usage_pct DESC""",
        (pitcher_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_team_batters(team_abbrev: str) -> list[dict]:
    """Get batter stats for a team's hitters."""
    conn = get_db()
    rows = conn.execute(
        """SELECT batter_id, batter_name, bat_side, plate_appearances,
                  k_rate, whiff_rate, chase_rate, contact_rate
           FROM mlb_batter_stats
           WHERE team_abbrev = ? AND plate_appearances >= 10
           ORDER BY plate_appearances DESC""",
        (team_abbrev,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_batter_vs_pitch_types(batter_id: int) -> dict[str, dict]:
    """Get batter's performance vs each pitch type. Returns {pitch_type: stats}."""
    conn = get_db()
    rows = conn.execute(
        """SELECT pitch_type, pitches_seen, whiff_rate, chase_rate,
                  contact_rate, k_rate_vs_type
           FROM mlb_batter_vs_pitch
           WHERE batter_id = ? AND pitches_seen >= 5
           ORDER BY pitches_seen DESC""",
        (batter_id,),
    ).fetchall()
    conn.close()
    return {r["pitch_type"]: dict(r) for r in rows}


def get_pitcher_tto_profile(pitcher_id: int) -> dict[int, dict]:
    """Get pitcher's TTO decay profile. Returns {tto_number: stats}."""
    conn = get_db()
    rows = conn.execute(
        """SELECT tto_number, k_rate, whiff_rate, avg_velocity, sample_pa
           FROM mlb_pitcher_tto
           WHERE pitcher_id = ?
           ORDER BY tto_number""",
        (pitcher_id,),
    ).fetchall()
    conn.close()
    return {r["tto_number"]: dict(r) for r in rows}


def compute_arsenal_matchup(pitcher_id: int, opponent_team: str) -> dict:
    """
    Compute arsenal-weighted matchup features.

    For each pitch type in the pitcher's arsenal, look up how each opposing
    batter performs against that pitch type, then weight by the pitcher's
    usage % to get a composite matchup score.

    Returns dict of matchup features:
    - matchup_whiff_rate: weighted avg whiff rate of batters vs pitcher's arsenal
    - matchup_chase_rate: weighted avg chase rate
    - matchup_contact_rate: weighted avg contact rate
    - matchup_k_rate: weighted avg K rate vs pitcher's pitch types
    - arsenal_diversity: number of distinct pitch types with >10% usage
    - best_pitch_whiff: pitcher's highest whiff rate on a primary pitch
    - tto_k_decay: K rate drop from 1st to 3rd time through order
    - avg_tto_velocity_drop: velocity drop from 1st to 3rd TTO
    """
    arsenal = get_pitcher_arsenal(pitcher_id)
    if not arsenal:
        return {}

    batters = get_team_batters(opponent_team)
    if not batters:
        return {}

    # Load all batter vs pitch type data for opposing batters
    batter_pitch_profiles = {}
    for b in batters:
        bvp = get_batter_vs_pitch_types(b["batter_id"])
        if bvp:
            batter_pitch_profiles[b["batter_id"]] = bvp

    # Weight each batter equally (approximation — ideally weight by expected PA)
    # For each pitch type in arsenal, compute avg batter performance
    total_usage = sum(p.get("usage_pct", 0) or 0 for p in arsenal)
    if total_usage <= 0:
        return {}

    weighted_whiff = 0.0
    weighted_chase = 0.0
    weighted_contact = 0.0
    weighted_k = 0.0
    best_pitch_whiff = 0.0
    diversity_count = 0

    for pitch in arsenal:
        pt = pitch["pitch_type"]
        usage = (pitch.get("usage_pct") or 0)
        if usage < 0.02:
            continue
        if usage >= 0.10:
            diversity_count += 1

        pitcher_whiff = pitch.get("whiff_rate") or 0

        # Average batter performance vs this pitch type
        batter_whiffs = []
        batter_chases = []
        batter_contacts = []
        batter_ks = []

        for bid, bvp in batter_pitch_profiles.items():
            if pt in bvp:
                stats = bvp[pt]
                batter_whiffs.append(stats.get("whiff_rate") or 0)
                batter_chases.append(stats.get("chase_rate") or 0)
                batter_contacts.append(stats.get("contact_rate") or 0)
                batter_ks.append(stats.get("k_rate_vs_type") or 0)

        if batter_whiffs:
            avg_batter_whiff = np.mean(batter_whiffs)
            avg_batter_chase = np.mean(batter_chases)
            avg_batter_contact = np.mean(batter_contacts)
            avg_batter_k = np.mean(batter_ks)
        else:
            # Fallback: use pitcher's own rate as proxy
            avg_batter_whiff = pitcher_whiff
            avg_batter_chase = pitch.get("chase_rate") or 0.30
            avg_batter_contact = 1.0 - pitcher_whiff
            avg_batter_k = 0.22

        # Combine pitcher's pitch effectiveness with batter vulnerability
        # A high whiff pitch vs a high whiff batter = amplified effect
        combined_whiff = (pitcher_whiff + avg_batter_whiff) / 2

        weighted_whiff += usage * combined_whiff
        weighted_chase += usage * avg_batter_chase
        weighted_contact += usage * avg_batter_contact
        weighted_k += usage * avg_batter_k

        if pitcher_whiff > best_pitch_whiff and usage >= 0.10:
            best_pitch_whiff = pitcher_whiff

    # Normalize by total usage
    if total_usage > 0:
        weighted_whiff /= total_usage
        weighted_chase /= total_usage
        weighted_contact /= total_usage
        weighted_k /= total_usage

    # TTO decay
    tto = get_pitcher_tto_profile(pitcher_id)
    tto_k_decay = 0.0
    tto_velo_drop = 0.0
    if 1 in tto and 3 in tto:
        k1 = tto[1].get("k_rate") or 0.25
        k3 = tto[3].get("k_rate") or 0.20
        tto_k_decay = k1 - k3  # positive = pitcher gets worse
        v1 = tto[1].get("avg_velocity") or 93
        v3 = tto[3].get("avg_velocity") or 92
        tto_velo_drop = v1 - v3  # positive = loses velo

    return {
        "matchup_whiff_rate": round(weighted_whiff, 4),
        "matchup_chase_rate": round(weighted_chase, 4),
        "matchup_contact_rate": round(weighted_contact, 4),
        "matchup_k_rate": round(weighted_k, 4),
        "arsenal_diversity": diversity_count,
        "best_pitch_whiff": round(best_pitch_whiff, 4),
        "tto_k_decay": round(tto_k_decay, 4),
        "tto_velo_drop": round(tto_velo_drop, 2),
    }
