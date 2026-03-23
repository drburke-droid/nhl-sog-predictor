"""
MLB Pitch-Matchup Run Model V2.

Predicts game runs by simulating pitcher-batter matchups at the pitch level.

Key improvements over V1:
1. Calibrated run values (fitted to actual 2024 outcomes, not theoretical)
2. Matchup-driven starter depth prediction (bad matchup = early hook)
3. Team-specific bullpen modeling with closer identification
4. Park run factors
5. Platoon splits (LHP vs RHB, etc.)
6. Blended with Vegas line (market is the anchor, model adjusts)
"""

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "mlb_data.db"

# League averages (2024)
LG = {
    "runs_per_game": 4.55,
    "k_rate": 0.224,
    "bb_rate": 0.082,
    "babip": 0.296,
    "hr_per_fb": 0.12,
    "starter_ip": 5.4,
    "bullpen_era": 3.95,
}

# Platoon adjustments from 2024 data — both K rate AND run value (ERA-based)
# ERA-based multipliers capture contact quality, not just strikeouts.
#
# 2024 measured platoon ERA:
#   LHP vs LHB-heavy: 3.61 ERA  (same-side = pitcher dominates)
#   LHP vs RHB-heavy: 4.82 ERA  (opposite-side = batters crush)
#   RHP vs LHB-heavy: 5.09 ERA  (LHB lineups hit RHP hard)
#   RHP vs RHB-heavy: 4.92 ERA  (baseline)
#
# Normalized to league avg ERA (~4.55):
PLATOON_RUN_MULT = {
    # (pitcher_hand, batter_side) -> multiplier on runs produced
    # <1.0 = pitcher has advantage (fewer runs), >1.0 = batter advantage
    ("L", "L"): 0.79,   # LHP vs LHB: 3.61/4.55 = dominant
    ("L", "R"): 1.06,   # LHP vs RHB: 4.82/4.55 = batters hit well
    ("L", "S"): 0.88,   # LHP vs Switch: switch bats righty, slight pitcher edge
    ("R", "L"): 1.12,   # RHP vs LHB: 5.09/4.55 = LHB crush RHP
    ("R", "R"): 1.08,   # RHP vs RHB: 4.92/4.55
    ("R", "S"): 1.00,   # RHP vs Switch: neutral
}

# K rate platoon adjustments (separate from run value)
PLATOON_K_MULT = {
    ("L", "L"): 1.05, ("L", "R"): 1.01, ("L", "S"): 0.92,
    ("R", "L"): 1.00, ("R", "R"): 1.01, ("R", "S"): 0.94,
}

# HR rate platoon adjustments (from 2024 data)
PLATOON_HR_MULT = {
    ("L", "L"): 0.80,   # LHP vs LHB: 0.025 HR/BF (below avg)
    ("L", "R"): 1.00,   # LHP vs RHB: 0.032 HR/BF (avg)
    ("L", "S"): 0.85,   # Estimated
    ("R", "L"): 1.03,   # RHP vs LHB: 0.033 HR/BF (slightly above)
    ("R", "R"): 1.03,   # RHP vs RHB: 0.033 HR/BF
    ("R", "S"): 1.00,
}

# Bullpen platoon management probability
BULLPEN_PLATOON_RATE = 0.67

# Seasonal scoring adjustments (2024 data — normalized to season avg)
# March/April runs much higher (cold weather paradox: more walks, errors)
# September drops (tired arms, expanded rosters, meaningless games)
MONTH_RUN_FACTOR = {
    3: 1.15, 4: 0.98, 5: 0.97, 6: 1.02, 7: 1.04, 8: 1.03, 9: 0.96, 10: 1.00,
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Pitcher Arsenal
# ---------------------------------------------------------------------------

def _get_pitcher_platoon_mult(pitcher_id, pitcher_hand, before_date, conn):
    """Compute per-pitcher platoon ERA multiplier from their game history.

    Returns dict: {vs_L_mult, vs_R_mult} where mult is relative to pitcher's
    overall ERA. Shrunk 50% toward league-average platoon split to handle
    small samples.
    """
    # Build lineup composition per game (cached would be better but functional)
    rows = conn.execute("""
        SELECT pgs.date, pgs.opponent_abbrev, pgs.innings_pitched, pgs.earned_runs
        FROM mlb_pitcher_game_stats pgs
        WHERE pgs.pitcher_id = ? AND pgs.is_starter = 1 AND pgs.date < ?
        AND pgs.innings_pitched >= 3
        ORDER BY pgs.date DESC LIMIT 20
    """, (pitcher_id, before_date)).fetchall()

    if len(rows) < 6:
        # Not enough data — return league average platoon
        lg_mult = PLATOON_RUN_MULT.get((pitcher_hand, "L"), 1.0)
        rg_mult = PLATOON_RUN_MULT.get((pitcher_hand, "R"), 1.0)
        return {"vs_L": lg_mult, "vs_R": rg_mult}

    # Classify each start by opposing lineup LHB fraction
    vs_lhb_eras = []
    vs_rhb_eras = []
    for r in rows:
        comp = conn.execute("""
            SELECT bs.bat_side, COUNT(DISTINCT bgpt.batter_id) as n
            FROM mlb_batter_game_pitch_type bgpt
            JOIN mlb_batter_stats bs ON bgpt.batter_id = bs.batter_id
            WHERE bgpt.game_date = ? AND bgpt.team_abbrev = ?
            GROUP BY bs.bat_side
        """, (r["date"], r["opponent_abbrev"])).fetchall()
        if not comp:
            continue
        comp_dict = {c["bat_side"]: c["n"] for c in comp}
        total = sum(comp_dict.values())
        if total == 0:
            continue
        lhb_pct = comp_dict.get("L", 0) / total
        era = r["earned_runs"] * 9.0 / r["innings_pitched"]

        if lhb_pct >= 0.35:
            vs_lhb_eras.append(era)
        else:
            vs_rhb_eras.append(era)

    # Compute individual platoon multiplier
    overall_era = np.mean([r["earned_runs"] * 9.0 / r["innings_pitched"] for r in rows])
    if overall_era <= 0:
        overall_era = LG["runs_per_game"]

    if len(vs_lhb_eras) >= 3:
        raw_l = np.mean(vs_lhb_eras) / overall_era
    else:
        raw_l = PLATOON_RUN_MULT.get((pitcher_hand, "L"), 1.0)

    if len(vs_rhb_eras) >= 3:
        raw_r = np.mean(vs_rhb_eras) / overall_era
    else:
        raw_r = PLATOON_RUN_MULT.get((pitcher_hand, "R"), 1.0)

    # Shrink 50% toward league-average platoon split (regularization)
    lg_l = PLATOON_RUN_MULT.get((pitcher_hand, "L"), 1.0)
    lg_r = PLATOON_RUN_MULT.get((pitcher_hand, "R"), 1.0)
    shrink = 0.50
    vs_l = shrink * lg_l + (1 - shrink) * raw_l
    vs_r = shrink * lg_r + (1 - shrink) * raw_r

    return {"vs_L": round(vs_l, 3), "vs_R": round(vs_r, 3)}


def _get_tto_decay(pitcher_id, before_date, conn):
    """Get times-through-order K rate decay for a pitcher.

    Returns multiplier for 3rd TTO vs 1st TTO. >1.0 means pitcher gets
    BETTER (rare), <1.0 means pitcher degrades (typical).
    Used to adjust late-game run estimates.
    """
    rows = conn.execute("""
        SELECT tto_number, AVG(k_count * 1.0 / NULLIF(pa_count, 0)) as k_rate,
               SUM(pa_count) as total_pa
        FROM mlb_pitcher_game_tto
        WHERE pitcher_id = ? AND game_date < ?
        GROUP BY tto_number
    """, (pitcher_id, before_date)).fetchall()

    if not rows or len(rows) < 2:
        return 0.85  # league average: ~15% decline 3rd time through

    tto_k = {r["tto_number"]: r["k_rate"] for r in rows if r["k_rate"]}
    first = tto_k.get(1, LG["k_rate"])
    third = tto_k.get(3, first * 0.85)

    if first > 0:
        return max(min(third / first, 1.2), 0.50)
    return 0.85


def _get_batter_individual_platoon(batter_id, before_date, conn):
    """Get this specific batter's K rate split vs LHP and RHP.

    Returns {vs_L_k, vs_R_k} — individual K rates, shrunk 40% toward
    league average to handle small samples.
    """
    # Get starter hand per game this batter appeared in
    rows = conn.execute("""
        SELECT bgpt.game_date, bgpt.team_abbrev,
               SUM(bgpt.k_count) as k, SUM(bgpt.pa_count) as pa
        FROM mlb_batter_game_pitch_type bgpt
        WHERE bgpt.batter_id = ? AND bgpt.game_date < ?
        GROUP BY bgpt.game_date
        HAVING pa > 0
    """, (batter_id, before_date)).fetchall()

    splits = {"L": {"k": 0, "pa": 0}, "R": {"k": 0, "pa": 0}}
    for r in rows:
        game = conn.execute("""
            SELECT game_pk, home_team_abbrev FROM mlb_games
            WHERE date = ? AND (home_team_abbrev = ? OR away_team_abbrev = ?) LIMIT 1
        """, (r["game_date"], r["team_abbrev"], r["team_abbrev"])).fetchone()
        if not game:
            continue
        is_home = 1 if game["home_team_abbrev"] == r["team_abbrev"] else 0
        # Find opposing starter's hand
        starter = conn.execute("""
            SELECT pitch_hand FROM mlb_pitcher_game_stats
            WHERE game_pk = ? AND is_starter = 1 AND is_home = ?
        """, (game["game_pk"], 1 - is_home)).fetchone()
        if not starter:
            continue
        hand = starter["pitch_hand"]
        splits[hand]["k"] += r["k"] or 0
        splits[hand]["pa"] += r["pa"] or 0

    result = {}
    for hand in ["L", "R"]:
        if splits[hand]["pa"] >= 15:
            raw_k = splits[hand]["k"] / splits[hand]["pa"]
        else:
            raw_k = LG["k_rate"]
        # Shrink 40% toward league average
        result[f"vs_{hand}_k"] = 0.40 * LG["k_rate"] + 0.60 * raw_k

    return result


def _get_pitcher_recent_form(pitcher_id, before_date, conn):
    """Get pitcher's recent form as ERA multiplier.

    Recent 3 starts ERA relative to season ERA.
    >1.0 = struggling recently, <1.0 = hot streak.
    Shrunk 70% toward 1.0 (mean regression — form is noisy).
    """
    rows = conn.execute("""
        SELECT earned_runs, innings_pitched
        FROM mlb_pitcher_game_stats
        WHERE pitcher_id = ? AND is_starter = 1 AND date < ?
        ORDER BY date DESC LIMIT 10
    """, (pitcher_id, before_date)).fetchall()

    if len(rows) < 5:
        return 1.0

    # Season ERA (all rows)
    season_era = sum(r["earned_runs"] for r in rows) * 9 / max(sum(r["innings_pitched"] for r in rows), 1)

    # Recent 3 ERA
    recent = rows[:3]
    recent_era = sum(r["earned_runs"] for r in recent) * 9 / max(sum(r["innings_pitched"] for r in recent), 1)

    if season_era <= 0:
        return 1.0

    raw_mult = recent_era / season_era
    # Heavy shrinkage — form signal is weak (r=0.036)
    return 0.70 * 1.0 + 0.30 * max(min(raw_mult, 2.0), 0.3)


def _get_arsenal(pitcher_id, before_date, conn):
    """Get pitcher's recent pitch mix from game-level data."""
    rows = conn.execute("""
        SELECT pitch_type,
               SUM(pitches) as total_p,
               SUM(whiff_count) as whiffs,
               SUM(csw_count) as csws,
               SUM(swing_count) as swings,
               SUM(zone_count) as zones,
               AVG(avg_velocity) as vel
        FROM mlb_pitcher_game_arsenal
        WHERE pitcher_id = ? AND game_date < ?
        AND game_date IN (
            SELECT DISTINCT game_date FROM mlb_pitcher_game_arsenal
            WHERE pitcher_id = ? AND game_date < ?
            ORDER BY game_date DESC LIMIT 7
        )
        GROUP BY pitch_type HAVING total_p >= 5
    """, (pitcher_id, before_date, pitcher_id, before_date)).fetchall()

    if not rows:
        # Fallback to season-level
        rows = conn.execute(
            "SELECT pitch_type, usage_pct, whiff_rate, csw_rate, zone_rate, avg_velocity "
            "FROM mlb_pitcher_arsenal WHERE pitcher_id = ?",
            (pitcher_id,),
        ).fetchall()
        if not rows:
            return []
        return [{
            "pt": r["pitch_type"], "usage": r["usage_pct"],
            "whiff": r["whiff_rate"] or 0, "csw": r["csw_rate"] or 0,
            "zone": r["zone_rate"] or 0.45, "vel": r["avg_velocity"] or 90,
        } for r in rows]

    total = sum(r["total_p"] for r in rows)
    return [{
        "pt": r["pitch_type"],
        "usage": r["total_p"] / total,
        "whiff": r["whiffs"] / max(r["swings"], 1),
        "csw": r["csws"] / r["total_p"],
        "zone": r["zones"] / r["total_p"],
        "vel": r["vel"] or 90,
    } for r in rows]


# ---------------------------------------------------------------------------
# Batter vs Pitch Type
# ---------------------------------------------------------------------------

def _get_batter_vs_pitch(batter_id, before_date, conn):
    """Get batter's K rate vs each pitch type using ONLY pre-game data.

    Uses rolling game-level pitch type data (mlb_batter_game_pitch_type)
    instead of season-level aggregates to prevent data leakage.
    """
    if before_date:
        rows = conn.execute("""
            SELECT pitch_type,
                   SUM(k_count) as total_k,
                   SUM(pa_count) as total_pa,
                   SUM(whiff_count) as total_whiff,
                   SUM(swing_count) as total_swing,
                   SUM(pitches_seen) as total_pitches
            FROM mlb_batter_game_pitch_type
            WHERE batter_id = ? AND game_date < ?
            GROUP BY pitch_type
            HAVING total_pitches >= 5
        """, (batter_id, before_date)).fetchall()
    else:
        rows = []

    if rows:
        return {r["pitch_type"]: {
            "k_rate": r["total_k"] / max(r["total_pa"], 1) if r["total_pa"] else LG["k_rate"],
            "whiff": r["total_whiff"] / max(r["total_swing"], 1) if r["total_swing"] else 0.25,
            "contact": 1 - (r["total_whiff"] / max(r["total_swing"], 1)) if r["total_swing"] else 0.75,
            "n": r["total_pitches"],
        } for r in rows}

    # Fallback to season-level (only if no game-level data at all)
    rows = conn.execute(
        "SELECT pitch_type, k_rate_vs_type, whiff_rate, contact_rate, pitches_seen "
        "FROM mlb_batter_vs_pitch WHERE batter_id = ?",
        (batter_id,),
    ).fetchall()
    return {r["pitch_type"]: {
        "k_rate": r["k_rate_vs_type"] or LG["k_rate"],
        "whiff": r["whiff_rate"] or 0.25,
        "contact": r["contact_rate"] or 0.75,
        "n": r["pitches_seen"] or 0,
    } for r in rows}


# ---------------------------------------------------------------------------
# Matchup Expected Runs (calibrated)
# ---------------------------------------------------------------------------

def _matchup_k_rate(arsenal, batter_vs_pitch, pitcher_hand="R", batter_side="R"):
    """Compute expected K rate for one pitcher vs one batter.

    Applies platoon multiplier based on pitcher hand vs batter side.
    Switch hitters bat from the favorable side.
    """
    if not arsenal:
        return LG["k_rate"]

    # Platoon adjustment
    platoon_mult = PLATOON_K_MULT.get((pitcher_hand, batter_side), 1.0)

    weighted_k = 0.0
    for p in arsenal:
        bvp = batter_vs_pitch.get(p["pt"])
        if bvp and bvp["n"] >= 10:
            pitcher_k_power = p["whiff"] * 2.5
            batter_k_vuln = bvp["k_rate"]
            matchup_k = np.sqrt(pitcher_k_power * batter_k_vuln)
            matchup_k = min(matchup_k, 0.55)
        else:
            matchup_k = p["whiff"] * 1.8
            matchup_k = min(matchup_k, 0.40)
        weighted_k += p["usage"] * matchup_k

    # Apply platoon adjustment
    weighted_k *= platoon_mult

    return max(min(weighted_k, 0.50), 0.05)


def _matchup_runs_per_pa(matchup_k_rate, pitcher_bb_rate,
                         pitcher_hand="R", batter_side="R"):
    """Convert K rate and BB rate into expected runs per PA.

    Applies full platoon adjustments:
    - K rate (already adjusted in _matchup_k_rate)
    - HR rate multiplier (same-side pitchers allow fewer HR)
    - Overall run value multiplier (ERA-based platoon splits)
    """
    k = matchup_k_rate
    bb = pitcher_bb_rate
    contact = max(1.0 - k - bb - 0.01, 0.10)

    # Platoon-adjusted HR rate
    hr_mult = PLATOON_HR_MULT.get((pitcher_hand, batter_side), 1.0)
    hr_rate = LG["hr_per_fb"] * 0.35 * contact * hr_mult

    babip = LG["babip"]
    hit_rate = babip * contact * (1 - 0.35 * LG["hr_per_fb"])

    runs = (
        k * (-0.28)
        + bb * 0.33
        + 0.01 * 0.35
        + hr_rate * 1.40
        + hit_rate * 0.52
        + (contact - hr_rate - hit_rate) * (-0.27)
    )

    # Apply overall ERA-based platoon multiplier
    # This captures contact quality, sequencing, and all the things K rate misses
    run_mult = PLATOON_RUN_MULT.get((pitcher_hand, batter_side), 1.0)
    runs *= run_mult

    return runs


# ---------------------------------------------------------------------------
# Starter Depth Prediction
# ---------------------------------------------------------------------------

def _predict_starter_depth(pitcher_id, matchup_difficulty, before_date, conn):
    """Predict how deep a starter goes based on rolling average + matchup.

    matchup_difficulty: avg K rate the starter achieves vs this lineup.
    Higher K rate = easier matchup = more innings.
    """
    # Rolling average IP from last 5 starts
    rows = conn.execute("""
        SELECT innings_pitched, batters_faced, earned_runs
        FROM mlb_pitcher_game_stats
        WHERE pitcher_id = ? AND is_starter = 1 AND date < ?
        ORDER BY date DESC LIMIT 5
    """, (pitcher_id, before_date)).fetchall()

    if not rows:
        return LG["starter_ip"]

    avg_ip = np.mean([r["innings_pitched"] for r in rows])
    avg_er = np.mean([r["earned_runs"] for r in rows])

    # Adjust for matchup: easier matchup (higher K rate) = +0.5 IP potential
    # Harder matchup (lower K rate) = -0.5 IP
    k_vs_avg = matchup_difficulty - LG["k_rate"]
    ip_adjustment = k_vs_avg * 3.0  # +/- 0.3 to 0.7 IP
    ip_adjustment = max(min(ip_adjustment, 0.7), -0.7)

    predicted_ip = avg_ip + ip_adjustment
    return max(min(predicted_ip, 8.0), 3.0)


# ---------------------------------------------------------------------------
# Team Bullpen Profile
# ---------------------------------------------------------------------------

def _get_bullpen_profile(team, before_date, conn):
    """Get team bullpen stats split by pitcher hand + closer info.

    Returns {lhp_era, rhp_era, lhp_k_rate, rhp_k_rate, closer_*,
             lhp_count, rhp_count}
    """
    cutoff = (pd.Timestamp(before_date) - timedelta(days=30)).strftime("%Y-%m-%d")

    # Split by pitcher hand
    bp_splits = {}
    for hand in ["L", "R"]:
        row = conn.execute("""
            SELECT SUM(earned_runs) * 9.0 / NULLIF(SUM(innings_pitched), 0) as era,
                   AVG(strikeouts * 1.0 / NULLIF(batters_faced, 0)) as k_rate,
                   COUNT(DISTINCT pitcher_id) as n_pitchers,
                   COUNT(*) as apps
            FROM mlb_pitcher_game_stats
            WHERE team_abbrev = ? AND is_starter = 0 AND pitch_hand = ?
            AND date >= ? AND date < ?
        """, (team, hand, cutoff, before_date)).fetchone()
        bp_splits[hand] = {
            "era": row["era"] if row and row["era"] else LG["bullpen_era"],
            "k_rate": row["k_rate"] if row and row["k_rate"] else 0.25,
            "n_pitchers": row["n_pitchers"] if row else 0,
            "apps": row["apps"] if row else 0,
        }

    # Overall
    overall_era = (
        (bp_splits["L"]["era"] * bp_splits["L"]["apps"] +
         bp_splits["R"]["era"] * bp_splits["R"]["apps"])
        / max(bp_splits["L"]["apps"] + bp_splits["R"]["apps"], 1)
    )

    # Closer
    closer = conn.execute("""
        SELECT pitcher_id, pitcher_name, pitch_hand,
               SUM(earned_runs) * 9.0 / NULLIF(SUM(innings_pitched), 0) as era,
               AVG(strikeouts * 1.0 / NULLIF(batters_faced, 0)) as k_rate,
               COUNT(*) as apps
        FROM mlb_pitcher_game_stats
        WHERE team_abbrev = ? AND is_starter = 0 AND date >= ? AND date < ?
        AND innings_pitched BETWEEN 0.67 AND 1.33
        GROUP BY pitcher_id ORDER BY apps DESC LIMIT 1
    """, (team, cutoff, before_date)).fetchone()

    return {
        "era": round(overall_era, 2),
        "lhp_era": round(bp_splits["L"]["era"], 2),
        "rhp_era": round(bp_splits["R"]["era"], 2),
        "lhp_k_rate": round(bp_splits["L"]["k_rate"], 3),
        "rhp_k_rate": round(bp_splits["R"]["k_rate"], 3),
        "lhp_count": bp_splits["L"]["n_pitchers"],
        "rhp_count": bp_splits["R"]["n_pitchers"],
        "closer_id": closer["pitcher_id"] if closer else None,
        "closer_hand": closer["pitch_hand"] if closer else "R",
        "closer_era": closer["era"] if closer and closer["era"] else overall_era,
        "closer_k_rate": closer["k_rate"] if closer and closer["k_rate"] else 0.25,
    }


# ---------------------------------------------------------------------------
# Park Run Factor
# ---------------------------------------------------------------------------

def _get_park_factor(home_team, before_date, conn):
    """Get park run factor from actual 2024 game scores. >1.0 = hitter's park.

    Uses total runs scored (both teams) in home games vs away games
    for all teams that play at this park.
    """
    home = conn.execute("""
        SELECT AVG(home_score + away_score) as avg_total, COUNT(*) as n
        FROM mlb_games WHERE home_team_abbrev = ? AND date < ? AND status = 'Final'
    """, (home_team, before_date)).fetchone()

    away = conn.execute("""
        SELECT AVG(home_score + away_score) as avg_total, COUNT(*) as n
        FROM mlb_games WHERE away_team_abbrev = ? AND date < ? AND status = 'Final'
    """, (home_team, before_date)).fetchone()

    if (home and away and home["n"] >= 10 and away["n"] >= 10
            and home["avg_total"] and away["avg_total"] and away["avg_total"] > 0):
        factor = home["avg_total"] / away["avg_total"]
        return max(min(factor, 1.35), 0.70)
    return 1.0


def _get_team_defense(team, before_date, conn):
    """Get team defensive quality as BABIP multiplier.

    Lower BABIP = better defense (fewer hits on balls in play).
    Returns multiplier: <1.0 = good defense (suppresses runs),
    >1.0 = bad defense (more runs).
    """
    row = conn.execute("""
        SELECT SUM(hits_allowed) * 1.0 /
               NULLIF(SUM(batters_faced - walks - strikeouts - home_runs_allowed), 0) as babip,
               COUNT(DISTINCT date) as games
        FROM mlb_pitcher_game_stats
        WHERE team_abbrev = ? AND date < ?
        AND date >= (SELECT date FROM mlb_pitcher_game_stats
                     WHERE team_abbrev = ? AND date < ?
                     ORDER BY date DESC LIMIT 1 OFFSET 29)
    """, (team, before_date, team, before_date)).fetchone()

    if row and row["babip"] and row["games"] >= 10:
        # Normalize to league avg BABIP (~0.296)
        return row["babip"] / LG["babip"]
    return 1.0


# ---------------------------------------------------------------------------
# Full Game Prediction
# ---------------------------------------------------------------------------

def predict_game_runs(home_pitcher_id, away_pitcher_id, home_team, away_team,
                      game_date, conn=None):
    """Predict runs for both sides using pitch-matchup simulation.

    Flow:
    1. Get pitcher arsenals
    2. Get batting lineups
    3. Compute matchup K rates for each batter vs starter
    4. Predict starter depth from matchup difficulty
    5. Model bullpen innings with team-specific bullpen
    6. Apply park factor
    7. Return predicted runs for each side
    """
    close = conn is None
    if conn is None:
        conn = get_db()

    park_factor = _get_park_factor(home_team, game_date, conn)

    # Defensive quality for each fielding team
    home_defense = _get_team_defense(home_team, game_date, conn)
    away_defense = _get_team_defense(away_team, game_date, conn)

    results = {}
    for side, pitcher_id, batting_team, fielding_team, def_mult in [
        ("away", home_pitcher_id, away_team, home_team, home_defense),
        ("home", away_pitcher_id, home_team, away_team, away_defense),
    ]:
        arsenal = _get_arsenal(pitcher_id, game_date, conn)

        # Get batting lineup
        batters = conn.execute("""
            SELECT batter_id, batter_name, k_rate, bb_rate, plate_appearances
            FROM mlb_batter_stats
            WHERE team_abbrev = ? AND plate_appearances >= 50
            ORDER BY plate_appearances DESC LIMIT 9
        """, (batting_team,)).fetchall()

        if not arsenal or not batters:
            results[side] = {
                "runs": LG["runs_per_game"],
                "starter_runs": LG["runs_per_game"] * 0.6,
                "bullpen_runs": LG["runs_per_game"] * 0.4,
                "starter_ip": LG["starter_ip"],
                "detail": "no_data",
            }
            continue

        # Pitcher BB rate from recent starts
        pitcher_recent = conn.execute("""
            SELECT AVG(walks * 1.0 / NULLIF(batters_faced, 0)) as bb_rate
            FROM mlb_pitcher_game_stats
            WHERE pitcher_id = ? AND is_starter = 1 AND date < ?
            ORDER BY date DESC LIMIT 7
        """, (pitcher_id, game_date)).fetchone()
        pitcher_bb = pitcher_recent["bb_rate"] if pitcher_recent and pitcher_recent["bb_rate"] else LG["bb_rate"]

        # Get pitcher handedness
        pitcher_hand_row = conn.execute(
            "SELECT pitch_hand FROM mlb_pitcher_game_stats WHERE pitcher_id = ? LIMIT 1",
            (pitcher_id,),
        ).fetchone()
        pitcher_hand = pitcher_hand_row["pitch_hand"] if pitcher_hand_row else "R"

        # Per-pitcher individual platoon multiplier (shrunk toward league avg)
        pitcher_platoon = _get_pitcher_platoon_mult(pitcher_id, pitcher_hand, game_date, conn)

        # TTO decay — how much worse does this pitcher get 3rd time through?
        tto_decay = _get_tto_decay(pitcher_id, game_date, conn)

        # Recent form — is pitcher running hot or cold?
        form_mult = _get_pitcher_recent_form(pitcher_id, game_date, conn)

        # Compute per-batter matchup with full platoon stack:
        #  1. Pitch arsenal × batter vulnerability per pitch type
        #  2. League platoon adjustment (K rate + HR rate + ERA)
        #  3. Per-pitcher individual platoon multiplier
        #  4. Per-batter individual platoon K rate
        #  5. Pitcher recent form
        batter_matchups = []
        for b in batters:
            bvp = _get_batter_vs_pitch(b["batter_id"], game_date, conn)
            batter_side = conn.execute(
                "SELECT bat_side FROM mlb_batter_stats WHERE batter_id = ?",
                (b["batter_id"],),
            ).fetchone()
            bside = batter_side["bat_side"] if batter_side else "R"

            # Individual batter platoon K rate
            batter_plat = _get_batter_individual_platoon(b["batter_id"], game_date, conn)
            batter_k_vs_hand = batter_plat.get(f"vs_{pitcher_hand}_k", LG["k_rate"])

            # Base matchup K rate from pitch arsenal × batter pitch-type data
            mk = _matchup_k_rate(arsenal, bvp, pitcher_hand, bside)

            # Blend arsenal-based K with individual batter's platoon K
            # 60% arsenal matchup (specific pitch types) + 40% individual platoon K
            mk = 0.60 * mk + 0.40 * batter_k_vs_hand

            runs_pa = _matchup_runs_per_pa(mk, pitcher_bb, pitcher_hand, bside)

            # Apply per-pitcher individual platoon adjustment
            pitcher_plat_mult = pitcher_platoon.get(f"vs_{bside}", 1.0)
            league_plat_mult = PLATOON_RUN_MULT.get((pitcher_hand, bside), 1.0)
            if league_plat_mult > 0:
                individual_adj = pitcher_plat_mult / league_plat_mult
            else:
                individual_adj = 1.0
            runs_pa *= individual_adj

            # Apply recent form
            runs_pa *= form_mult

            batter_matchups.append({
                "name": b["batter_name"],
                "k_rate": mk,
                "runs_pa": runs_pa,
                "bat_side": bside,
            })

        # Avg matchup K rate (for starter depth prediction)
        avg_matchup_k = np.mean([m["k_rate"] for m in batter_matchups])

        # Predict starter depth
        starter_ip = _predict_starter_depth(pitcher_id, avg_matchup_k, game_date, conn)
        starter_fraction = starter_ip / 9.0

        # Expected PAs per batter per 9 innings (by order position)
        pa_per_9 = [4.8, 4.7, 4.6, 4.5, 4.3, 4.1, 3.9, 3.8, 3.7]

        # Starter runs: split into 1st/2nd TTO (innings 1-6) and 3rd TTO (innings 6+)
        # Apply TTO decay: pitchers get worse the 3rd time through the order
        if starter_ip <= 6.0:
            # Starter only faces 1st and 2nd TTO — no 3rd TTO penalty
            starter_runs_above_avg = sum(
                m["runs_pa"] * pa * starter_fraction
                for m, pa in zip(batter_matchups, pa_per_9[:len(batter_matchups)])
            )
        else:
            # Split: first 6 IP at normal rate, remaining at degraded rate
            first_6_frac = 6.0 / 9.0
            extra_frac = (starter_ip - 6.0) / 9.0
            # 3rd TTO: K rate drops by tto_decay factor → more contact → more runs
            # Inverse of K decay = run increase (if pitcher K's less, runs go up)
            tto_run_increase = 1.0 / max(tto_decay, 0.50)  # e.g., 0.85 decay -> 1.18x runs
            tto_run_increase = min(tto_run_increase, 1.50)  # cap

            starter_runs_above_avg = sum(
                m["runs_pa"] * pa * first_6_frac
                for m, pa in zip(batter_matchups, pa_per_9[:len(batter_matchups)])
            ) + sum(
                m["runs_pa"] * pa * extra_frac * tto_run_increase
                for m, pa in zip(batter_matchups, pa_per_9[:len(batter_matchups)])
            )

        starter_base = LG["runs_per_game"] * starter_fraction
        starter_runs = starter_base + starter_runs_above_avg

        # Bullpen runs: team-specific bullpen × remaining innings with platoon management
        bullpen = _get_bullpen_profile(fielding_team, game_date, conn)
        bullpen_ip = 9.0 - starter_ip
        bullpen_fraction = bullpen_ip / 9.0

        # Compute lineup's platoon-weighted bullpen ERA
        # Managers bring in the favorable-matchup reliever ~67% of the time
        # For each batter: what ERA does the bullpen have against their hand?
        n_lhb = sum(1 for m in batter_matchups if m["bat_side"] == "L")
        n_rhb = sum(1 for m in batter_matchups if m["bat_side"] in ("R", "S"))
        total_batters = max(len(batter_matchups), 1)
        lhb_frac = n_lhb / total_batters
        rhb_frac = n_rhb / total_batters

        # Bullpen platoon management:
        # Managers bring same-side relievers ~67% of the time to exploit platoon advantage.
        # Use both the bullpen's hand-specific ERA AND the platoon run multiplier.
        #
        # For LHB in lineup:
        #   67% chance: LHP faces them (PLATOON_RUN_MULT L,L = 0.79 = pitcher dominates)
        #   33% chance: RHP faces them (PLATOON_RUN_MULT R,L = 1.12 = batter advantage)
        # For RHB in lineup:
        #   67% chance: RHP faces them (PLATOON_RUN_MULT R,R = 1.08)
        #   33% chance: LHP faces them (PLATOON_RUN_MULT L,R = 1.06)
        #
        # Weight by both the bullpen's hand-specific ERA and the platoon run multiplier
        def _bp_era_for_batter(bside):
            if bside == "L":
                # LHB: 67% LHP (same-side), 33% RHP
                same_era = bullpen["lhp_era"] * PLATOON_RUN_MULT.get(("L", "L"), 1.0)
                opp_era = bullpen["rhp_era"] * PLATOON_RUN_MULT.get(("R", "L"), 1.0)
            elif bside == "S":
                # Switch: always gets favorable side from bullpen, but manager still optimizes
                same_era = bullpen["rhp_era"] * PLATOON_RUN_MULT.get(("R", "S"), 1.0)
                opp_era = bullpen["lhp_era"] * PLATOON_RUN_MULT.get(("L", "S"), 1.0)
            else:  # R
                # RHB: 67% RHP (same-side), 33% LHP
                same_era = bullpen["rhp_era"] * PLATOON_RUN_MULT.get(("R", "R"), 1.0)
                opp_era = bullpen["lhp_era"] * PLATOON_RUN_MULT.get(("L", "R"), 1.0)
            return BULLPEN_PLATOON_RATE * same_era + (1 - BULLPEN_PLATOON_RATE) * opp_era

        bp_era_vs_lineup = sum(
            _bp_era_for_batter(m["bat_side"]) for m in batter_matchups
        ) / max(len(batter_matchups), 1)

        if bullpen_ip >= 2.0:
            middle_ip = bullpen_ip - 1.0
            middle_runs = LG["runs_per_game"] * (middle_ip / 9.0) * (bp_era_vs_lineup / LG["runs_per_game"])
            closer_runs = LG["runs_per_game"] * (1.0 / 9.0) * (bullpen["closer_era"] / LG["runs_per_game"])
            bullpen_runs = middle_runs + closer_runs
        else:
            bullpen_runs = LG["runs_per_game"] * bullpen_fraction * (bp_era_vs_lineup / LG["runs_per_game"])

        total_runs = max(starter_runs + bullpen_runs, 1.0)

        # Defense adjustment: affects contact-based runs
        # Good defense (mult < 1.0) turns more balls in play into outs
        # Only affects the non-K, non-BB portion of runs
        # Apply as a moderate multiplier (dampened — defense is partially
        # captured by pitcher stats already)
        defense_adj = 0.6 * def_mult + 0.4 * 1.0  # 60% weight to actual defense
        total_runs *= defense_adj

        # Park factor adjustment
        total_runs *= park_factor

        # Season month adjustment (scoring varies ~15% across months)
        try:
            month = int(game_date.split("-")[1])
            month_factor = MONTH_RUN_FACTOR.get(month, 1.0)
            total_runs *= month_factor
        except (ValueError, IndexError):
            pass

        results[side] = {
            "runs": round(total_runs, 3),
            "starter_runs": round(starter_runs, 3),
            "bullpen_runs": round(bullpen_runs, 3),
            "starter_ip": round(starter_ip, 1),
            "avg_matchup_k": round(avg_matchup_k, 3),
            "pitcher_hand": pitcher_hand,
            "tto_decay": round(tto_decay, 3),
            "form_mult": round(form_mult, 3),
            "pitcher_plat_vs_L": pitcher_platoon["vs_L"],
            "pitcher_plat_vs_R": pitcher_platoon["vs_R"],
            "bullpen_era": bullpen["era"],
            "bp_era_platoon": round(bp_era_vs_lineup, 2),
            "bp_lhp_era": bullpen["lhp_era"],
            "bp_rhp_era": bullpen["rhp_era"],
            "closer_era": bullpen["closer_era"],
            "closer_hand": bullpen["closer_hand"],
            "lineup_lhb_pct": round(lhb_frac, 2),
            "defense_mult": round(def_mult, 3),
            "defense_adj": round(defense_adj, 3),
            "park_factor": round(park_factor, 3),
            "n_batters": len(batter_matchups),
        }

    if close:
        conn.close()

    return {
        "home": results.get("home", {}),
        "away": results.get("away", {}),
        "pred_home_runs": results.get("home", {}).get("runs", LG["runs_per_game"]),
        "pred_away_runs": results.get("away", {}).get("runs", LG["runs_per_game"]),
        "pred_total": (results.get("home", {}).get("runs", LG["runs_per_game"])
                       + results.get("away", {}).get("runs", LG["runs_per_game"])),
    }


# ---------------------------------------------------------------------------
# Live Prediction — Today's Games
# ---------------------------------------------------------------------------

def predict_todays_games():
    """Predict all of today's MLB games and find profitable bets.

    Returns list of game dicts with predictions, matchup details, and bets.
    """
    import mlb_api

    games = mlb_api.get_todays_schedule()
    if not games:
        return []

    conn = get_db()

    # Get today's game odds
    MLB_MAP = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL", "Detroit Tigers": "DET",
        "Houston Astros": "HOU", "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN", "New York Mets": "NYM",
        "New York Yankees": "NYY", "Oakland Athletics": "OAK",
        "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD", "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    }
    ABBREV_TO_NAME = {v: k for k, v in MLB_MAP.items()}

    from datetime import date as dt_date
    today = dt_date.today().isoformat()

    # Load odds
    odds_map = {}
    ml_map = {}
    for r in conn.execute("""
        SELECT game_date, home_team, outcome_name, outcome_price, outcome_point, market
        FROM mlb_game_odds WHERE game_date = ? AND bookmaker = 'draftkings'
    """, (today,)).fetchall():
        ha = MLB_MAP.get(r["home_team"])
        if not ha:
            continue
        key = (today, ha)
        if r["market"] == "totals" and r["outcome_name"] == "Over":
            odds_map[key] = r["outcome_point"]
        elif r["market"] == "h2h":
            if key not in ml_map:
                ml_map[key] = {}
            oa = MLB_MAP.get(r["outcome_name"])
            if oa == ha:
                ml_map[key]["home_ml"] = r["outcome_price"]
            elif oa:
                ml_map[key]["away_ml"] = r["outcome_price"]

    from nhl_game_model import simulate_game, _american_to_prob, _american_to_decimal

    results = []
    for game in games:
        home_abbrev = game.get("home_team_abbrev", "")
        away_abbrev = game.get("away_team_abbrev", "")
        home_pp = game.get("home_probable_pitcher", {}) or {}
        away_pp = game.get("away_probable_pitcher", {}) or {}
        home_pitcher_id = home_pp.get("id")
        away_pitcher_id = away_pp.get("id")

        if not home_abbrev or not away_abbrev:
            continue
        if not home_pitcher_id or not away_pitcher_id:
            continue

        try:
            pred = predict_game_runs(
                home_pitcher_id, away_pitcher_id,
                home_abbrev, away_abbrev, today, conn,
            )
        except Exception as exc:
            logger.warning("Failed to predict %s@%s: %s", away_abbrev, home_abbrev, exc)
            continue

        # Odds
        vl = odds_map.get((today, home_abbrev))
        ml = ml_map.get((today, home_abbrev), {})
        matchup_diff = pred["pred_total"] - vl if vl else 0

        # Simulate
        sim = simulate_game(pred["pred_home_runs"], pred["pred_away_runs"], correlation=0.10)

        # Build bet recommendations
        bets = []

        # --- TOTALS ---
        if vl is not None:
            for line_try in [vl, vl + 0.5, vl - 0.5]:
                ok = f"over_{line_try}"
                uk = f"under_{line_try}"
                if ok in sim:
                    # UNDER when matchup < Vegas (our validated edge)
                    if matchup_diff < -0.3:
                        under_prob = sim[uk]
                        imp = 0.5238
                        edge = under_prob - imp
                        ev = under_prob * 0.909 - (1 - under_prob)
                        if ev > 0:
                            stars = 5 if matchup_diff < -1.0 else 3 if matchup_diff < -0.5 else 2
                            hist_yield = 17.7 if matchup_diff < -1.0 else 16.5
                            bets.append({
                                "market": "Total",
                                "pick": f"UNDER {vl}",
                                "model_prob": round(under_prob, 3),
                                "edge": round(edge, 3),
                                "odds": -110,
                                "stars": stars,
                                "hist_yield": hist_yield,
                            })
                    break

        # --- MONEYLINE ---
        home_ml = ml.get("home_ml")
        away_ml = ml.get("away_ml")
        if home_ml is not None and away_ml is not None:
            for team, ml_price, model_p in [
                (home_abbrev, home_ml, sim["home_win_prob"]),
                (away_abbrev, away_ml, sim["away_win_prob"]),
            ]:
                imp = _american_to_prob(ml_price)
                dec = _american_to_decimal(ml_price)
                edge = model_p - imp
                ev = model_p * (dec - 1) - (1 - model_p)

                # Validated strategies: Dogs edge>=8% (+8.6%), Favs edge>=3% (+5.3%)
                is_dog = ml_price > 0
                if is_dog and edge >= 0.08 and ev > 0:
                    bets.append({
                        "market": "ML",
                        "pick": team,
                        "model_prob": round(model_p, 3),
                        "edge": round(edge, 3),
                        "odds": ml_price,
                        "stars": 3 if edge >= 0.12 else 2,
                        "hist_yield": 8.6,
                    })
                elif not is_dog and edge >= 0.03 and ev > 0:
                    bets.append({
                        "market": "ML",
                        "pick": team,
                        "model_prob": round(model_p, 3),
                        "edge": round(edge, 3),
                        "odds": ml_price,
                        "stars": 2 if edge >= 0.05 else 1,
                        "hist_yield": 5.3,
                    })

        results.append({
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_pitcher": home_pp.get("name", "TBD"),
            "away_pitcher": away_pp.get("name", "TBD"),
            "pred_home_runs": pred["pred_home_runs"],
            "pred_away_runs": pred["pred_away_runs"],
            "pred_total": pred["pred_total"],
            "vegas_total": vl,
            "matchup_diff": round(matchup_diff, 2),
            "home_ml": home_ml,
            "away_ml": away_ml,
            "sim": {
                "home_win_prob": sim["home_win_prob"],
                "away_win_prob": sim["away_win_prob"],
            },
            "home_detail": {
                "starter_ip": pred["home"].get("starter_ip"),
                "matchup_k": pred["home"].get("avg_matchup_k"),
                "bullpen_era": pred["home"].get("bullpen_era"),
            },
            "away_detail": {
                "starter_ip": pred["away"].get("starter_ip"),
                "matchup_k": pred["away"].get("avg_matchup_k"),
                "bullpen_era": pred["away"].get("bullpen_era"),
            },
            "bets": sorted(bets, key=lambda b: b.get("stars", 0), reverse=True),
        })

    conn.close()
    return results


# ---------------------------------------------------------------------------
# Build Training Data
# ---------------------------------------------------------------------------

def build_training_data():
    """Build game-level training data with matchup predictions."""
    conn = get_db()

    games = conn.execute("""
        SELECT g.game_pk, g.date, g.home_team_abbrev, g.away_team_abbrev,
               g.home_score, g.away_score
        FROM mlb_games g WHERE g.status = 'Final' AND g.date LIKE '2024%'
        ORDER BY g.date
    """).fetchall()

    starters = {}
    for s in conn.execute("""
        SELECT game_pk, pitcher_id, pitcher_name, team_abbrev, is_home,
               innings_pitched, earned_runs, pitch_hand
        FROM mlb_pitcher_game_stats WHERE is_starter = 1 AND date LIKE '2024%'
    """).fetchall():
        starters[(s["game_pk"], s["is_home"])] = dict(s)

    # Vegas lines
    MLB_MAP = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL", "Detroit Tigers": "DET",
        "Houston Astros": "HOU", "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN", "New York Mets": "NYM",
        "New York Yankees": "NYY", "Oakland Athletics": "OAK",
        "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD", "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    }
    odds_map = {}  # total lines
    ml_map = {}    # moneyline odds
    for r in conn.execute("""
        SELECT game_date, home_team, outcome_point
        FROM mlb_game_odds WHERE market = 'totals' AND outcome_name = 'Over'
        AND bookmaker = 'draftkings' GROUP BY game_date, home_team
    """).fetchall():
        a = MLB_MAP.get(r["home_team"])
        if a:
            odds_map[(r["game_date"], a)] = r["outcome_point"]

    # Moneyline odds
    for r in conn.execute("""
        SELECT game_date, home_team, outcome_name, outcome_price
        FROM mlb_game_odds WHERE market = 'h2h' AND bookmaker = 'draftkings'
    """).fetchall():
        ha = MLB_MAP.get(r["home_team"])
        oa = MLB_MAP.get(r["outcome_name"])
        if ha:
            key = (r["game_date"], ha)
            if key not in ml_map:
                ml_map[key] = {}
            if oa == ha:
                ml_map[key]["home_ml"] = r["outcome_price"]
            elif oa:
                ml_map[key]["away_ml"] = r["outcome_price"]

    logger.info("Building training data: %d games, %d starters, %d odds",
                len(games), len(starters), len(odds_map))

    records = []
    for g in games:
        gk, gdate = g["game_pk"], g["date"]
        home, away = g["home_team_abbrev"], g["away_team_abbrev"]
        hs = g["home_starter"] if "home_starter" in g.keys() else starters.get((gk, 1))
        aws = starters.get((gk, 0))

        if not hs or not aws:
            continue

        try:
            pred = predict_game_runs(
                hs["pitcher_id"], aws["pitcher_id"],
                home, away, gdate, conn,
            )
        except Exception:
            continue

        vegas = odds_map.get((gdate, home))
        ml_odds = ml_map.get((gdate, home), {})

        records.append({
            "game_pk": gk, "date": gdate,
            "home_team": home, "away_team": away,
            "actual_home": g["home_score"], "actual_away": g["away_score"],
            "actual_total": g["home_score"] + g["away_score"],
            "pred_home": pred["pred_home_runs"],
            "pred_away": pred["pred_away_runs"],
            "pred_total": pred["pred_total"],
            "home_starter_ip": pred["home"].get("starter_ip", 5.4),
            "away_starter_ip": pred["away"].get("starter_ip", 5.4),
            "home_matchup_k": pred["home"].get("avg_matchup_k", LG["k_rate"]),
            "away_matchup_k": pred["away"].get("avg_matchup_k", LG["k_rate"]),
            "home_tto_decay": pred["home"].get("tto_decay", 0.85),
            "away_tto_decay": pred["away"].get("tto_decay", 0.85),
            "home_form": pred["home"].get("form_mult", 1.0),
            "away_form": pred["away"].get("form_mult", 1.0),
            "month": int(gdate.split("-")[1]) if "-" in gdate else 6,
            "home_bullpen_era": pred["home"].get("bullpen_era", LG["bullpen_era"]),
            "away_bullpen_era": pred["away"].get("bullpen_era", LG["bullpen_era"]),
            "home_bp_plat_era": pred["home"].get("bp_era_platoon", LG["bullpen_era"]),
            "away_bp_plat_era": pred["away"].get("bp_era_platoon", LG["bullpen_era"]),
            "home_lineup_lhb": pred["home"].get("lineup_lhb_pct", 0.4),
            "away_lineup_lhb": pred["away"].get("lineup_lhb_pct", 0.4),
            "home_defense": pred["home"].get("defense_mult", 1.0),
            "away_defense": pred["away"].get("defense_mult", 1.0),
            "park_factor": pred["home"].get("park_factor", 1.0),
            "vegas_total": vegas,
            "home_ml": ml_odds.get("home_ml"),
            "away_ml": ml_odds.get("away_ml"),
            "actual_home_win": 1 if g["home_score"] > g["away_score"] else 0,
            "actual_margin": g["home_score"] - g["away_score"],
        })

    conn.close()
    df = pd.DataFrame(records)
    logger.info("Training data: %d games", len(df))
    return df


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------

def run_walkforward(min_train=300, test_days=14, step_days=14):
    df = build_training_data()
    if df.empty:
        return {"error": "No data"}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Correction features: full matchup signal set
    feat_cols = [
        "pred_home", "pred_away", "pred_total",
        "home_starter_ip", "away_starter_ip",
        "home_matchup_k", "away_matchup_k",
        "home_tto_decay", "away_tto_decay",
        "home_form", "away_form",
        "home_bullpen_era", "away_bullpen_era",
        "home_bp_plat_era", "away_bp_plat_era",
        "home_lineup_lhb", "away_lineup_lhb",
        "home_defense", "away_defense",
        "park_factor", "month", "vegas_total",
    ]

    first_test = df["date"].min() + timedelta(days=90)
    season_end = df["date"].max()

    all_bets = []
    windows = []
    current = first_test
    wnum = 0

    while current < season_end:
        wnum += 1
        test_end = min(current + timedelta(days=test_days), season_end)
        train = df[df["date"] < current]
        test = df[(df["date"] >= current) & (df["date"] <= test_end)]

        if len(train) < min_train or len(test) == 0:
            current += timedelta(days=step_days)
            continue

        logger.info("Win %d: train %d, test %s-%s (%d)", wnum, len(train),
                    test["date"].min().date(), test["date"].max().date(), len(test))

        # Train: predict residual off Vegas line
        train_v = train[train["vegas_total"].notna()].copy()
        avail = [f for f in feat_cols if f in train.columns]

        if len(train_v) >= 100:
            train_v["residual"] = train_v["actual_total"] - train_v["vegas_total"]
            mdl = XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=5.0,
                min_child_weight=20, random_state=42, verbosity=0,
            )
            mdl.fit(train_v[avail].values, train_v["residual"].values)
            pred_res = mdl.predict(test[avail].values)
            corrected = np.where(
                test["vegas_total"].notna(),
                test["vegas_total"].fillna(0).values + pred_res,
                test["pred_total"].values,
            )
        else:
            corrected = test["pred_total"].values

        raw_mae = np.mean(np.abs(test["pred_total"].values - test["actual_total"].values))
        corr_mae = np.mean(np.abs(corrected - test["actual_total"].values))

        windows.append({
            "window": wnum, "train": len(train), "test": len(test),
            "raw_mae": round(raw_mae, 3), "corr_mae": round(corr_mae, 3),
        })

        # Bets
        from nhl_game_model import simulate_game
        for i in range(len(test)):
            row = test.iloc[i]
            vl = row.get("vegas_total")
            if vl is None or pd.isna(vl):
                continue

            pred = float(corrected[i])
            matchup_raw = float(row["pred_total"])
            actual = int(row["actual_total"])
            matchup_diff = matchup_raw - vl

            # Split predicted total into home/away using matchup ratio
            raw_home = float(row["pred_home"])
            raw_away = float(row["pred_away"])
            ratio = raw_home / max(raw_home + raw_away, 0.1)
            pred_home = pred * ratio
            pred_away = pred * (1 - ratio)

            sim = simulate_game(pred_home, pred_away, correlation=0.10)

            actual_home_win = int(row.get("actual_home_win", 0))
            actual_margin = int(row.get("actual_margin", 0))

            # --- TOTALS BETS ---
            for line_try in [vl, vl + 0.5, vl - 0.5]:
                ok = f"over_{line_try}"
                uk = f"under_{line_try}"
                if ok in sim:
                    imp = 0.5238
                    for side, prob, won in [
                        ("OVER", sim[ok], actual > vl),
                        ("UNDER", sim[uk], actual <= vl),
                    ]:
                        edge = prob - imp
                        ev = prob * 0.909 - (1 - prob)
                        all_bets.append({
                            "window": wnum,
                            "date": row["date"].strftime("%Y-%m-%d"),
                            "market": "TOTAL",
                            "pick": f"{side} {vl}",
                            "pred_total": round(pred, 2),
                            "matchup_diff": round(matchup_diff, 2),
                            "edge": round(edge, 4),
                            "ev": round(ev, 4),
                            "won": won,
                            "actual_total": actual,
                            "odds": -110,
                            "decimal_odds": 1.909,
                        })
                    break

            # --- MONEYLINE BETS ---
            home_ml = row.get("home_ml")
            away_ml = row.get("away_ml")
            if home_ml is not None and not pd.isna(home_ml) and away_ml is not None and not pd.isna(away_ml):
                home_ml = int(home_ml)
                away_ml = int(away_ml)
                from nhl_game_model import _american_to_prob, _american_to_decimal

                for team_pick, ml, model_p, won in [
                    (row["home_team"], home_ml, sim["home_win_prob"], actual_home_win == 1),
                    (row["away_team"], away_ml, sim["away_win_prob"], actual_home_win == 0),
                ]:
                    imp = _american_to_prob(ml)
                    dec = _american_to_decimal(ml)
                    edge = model_p - imp
                    ev = model_p * (dec - 1) - (1 - model_p)
                    all_bets.append({
                        "window": wnum,
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "market": "ML",
                        "pick": team_pick,
                        "pred_total": round(pred, 2),
                        "matchup_diff": round(matchup_diff, 2),
                        "edge": round(edge, 4),
                        "ev": round(ev, 4),
                        "won": won,
                        "actual_total": actual,
                        "odds": ml,
                        "decimal_odds": round(dec, 4),
                        "model_prob": round(model_p, 4),
                        "implied_prob": round(imp, 4),
                    })

        current += timedelta(days=step_days)

    # Results
    bets = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

    print("\n" + "=" * 95)
    print("  MLB O/U v1: WALK-FORWARD RESULTS (Totals + Moneyline)")
    print("=" * 95)
    print(f"  Windows: {len(windows)}")
    for w in windows:
        print(f"  Win {w['window']}: {w['train']} train, {w['test']} test | "
              f"Raw MAE: {w['raw_mae']:.3f}, Corrected: {w['corr_mae']:.3f}")

    if bets.empty:
        print("  No bets generated")
        return {"windows": windows}

    b = bets
    is_total = b["market"] == "TOTAL"
    is_ml = b["market"] == "ML"
    is_under = b["pick"].str.contains("UNDER")
    is_over = b["pick"].str.contains("OVER")
    ev_pos = b["ev"] > 0
    mu = b["matchup_diff"] < -0.3
    muu = b["matchup_diff"] < -1.0
    mo = b["matchup_diff"] > 0.3

    # ML-specific filters
    is_fav = is_ml & (b["odds"] < 0)
    is_dog = is_ml & (b["odds"] > 0)
    ml_matchup_favors_pick = is_ml & (
        ((b["matchup_diff"] < 0) & (b["model_prob"].fillna(0) > 0.5)) |   # model sees under = away pitcher good = home advantage... complex
        (b["ev"] > 0)
    )

    print(f"\n  Total bets: {len(b)}, Totals: {is_total.sum()}, ML: {is_ml.sum()}")

    print(f"\n  {'--- TOTALS ---':40s}")
    print(f"  {'Strategy':40s} {'N':>5s} {'WR':>6s} {'Flat Yld':>9s}")
    print("  " + "-" * 65)

    for name, mask in [
        ("TOTAL: UNDER matchup<Vegas", is_total & is_under & mu),
        ("TOTAL: UNDER matchup<<Vegas", is_total & is_under & muu),
        ("TOTAL: OVER matchup>Vegas", is_total & is_over & mo),
        ("TOTAL: All +EV", is_total & ev_pos),
        ("TOTAL: OVER +EV", is_total & ev_pos & is_over),
        ("TOTAL: UNDER +EV", is_total & ev_pos & is_under),
    ]:
        sub = b[mask]
        if len(sub) < 5:
            continue
        wr = sub["won"].mean()
        yld = wr * 1.909 - 1  # -110 flat bet yield
        print(f"  {name:40s} {len(sub):5d} {wr:5.1%} {yld*100:+8.1f}%")

    # --- MONEYLINE ---
    print(f"\n  {'--- MONEYLINE ---':40s}")
    print(f"  {'Strategy':40s} {'N':>5s} {'WR':>6s} {'Flat Yld':>9s}")
    print("  " + "-" * 65)

    def _ml_yield(subset):
        """Compute flat-bet yield for ML bets (variable odds)."""
        if len(subset) == 0:
            return 0, 0
        profits = []
        for _, bet in subset.iterrows():
            dec = bet["decimal_odds"]
            if bet["won"]:
                profits.append(dec - 1)
            else:
                profits.append(-1)
        avg_profit = np.mean(profits)
        return subset["won"].mean(), avg_profit

    for name, mask in [
        ("ML: All +EV", is_ml & ev_pos),
        ("ML: Favorites +EV", is_fav & ev_pos),
        ("ML: Underdogs +EV", is_dog & ev_pos),
        ("ML: Edge >= 3%", is_ml & (b["edge"] >= 0.03)),
        ("ML: Edge >= 5%", is_ml & (b["edge"] >= 0.05)),
        ("ML: Edge >= 8%", is_ml & (b["edge"] >= 0.08)),
        ("ML: Edge >= 10%", is_ml & (b["edge"] >= 0.10)),
        # Matchup-driven ML: bet team whose pitcher has better matchup
        ("ML: Matchup favors (edge>3%)", is_ml & (b["edge"] >= 0.03) & ev_pos),
        ("ML: Strong matchup (edge>5%)", is_ml & (b["edge"] >= 0.05) & ev_pos),
        # Underdogs with edge
        ("ML: Dogs edge>=5%", is_dog & (b["edge"] >= 0.05)),
        ("ML: Dogs edge>=8%", is_dog & (b["edge"] >= 0.08)),
        # Favorites with edge
        ("ML: Favs edge>=3%", is_fav & (b["edge"] >= 0.03)),
        ("ML: Favs edge>=5%", is_fav & (b["edge"] >= 0.05)),
    ]:
        sub = b[mask]
        if len(sub) < 5:
            continue
        wr, avg_pnl = _ml_yield(sub)
        print(f"  {name:40s} {len(sub):5d} {wr:5.1%} {avg_pnl*100:+8.1f}%")

    # --- COMBINED SUMMARY ---
    print(f"\n  {'--- BEST STRATEGIES ---':40s}")
    print("  " + "-" * 65)

    # Collect all strategies and rank
    all_strats = []
    for name, mask, is_ml_bet in [
        ("TOTAL: UNDER matchup<V", is_total & is_under & mu, False),
        ("TOTAL: UNDER matchup<<V", is_total & is_under & muu, False),
        ("ML: All +EV", is_ml & ev_pos, True),
        ("ML: Dogs edge>=5%", is_dog & (b["edge"] >= 0.05), True),
        ("ML: Dogs edge>=8%", is_dog & (b["edge"] >= 0.08), True),
        ("ML: Favs edge>=3%", is_fav & (b["edge"] >= 0.03), True),
        ("ML: Edge >= 5%", is_ml & (b["edge"] >= 0.05), True),
    ]:
        sub = b[mask]
        if len(sub) < 5:
            continue
        if is_ml_bet:
            wr, avg_pnl = _ml_yield(sub)
        else:
            wr = sub["won"].mean()
            avg_pnl = wr * 1.909 - 1
        if avg_pnl > 0:
            all_strats.append((name, len(sub), wr, avg_pnl))

    all_strats.sort(key=lambda x: x[3], reverse=True)
    for name, n, wr, pnl in all_strats:
        print(f"  >>> {name:38s} {n:5d} bets {wr:5.1%} WR {pnl*100:+7.1f}% yield")

    if not all_strats:
        print("  No profitable strategies found.")

    print("=" * 95)
    return {"windows": windows, "bets": bets}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run_walkforward()
