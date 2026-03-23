"""
MLB Pitch-Level Run Value Model.

Predicts game runs by simulating every pitcher-batter matchup:

1. For each starting pitcher, get their arsenal (pitch types + usage rates)
2. For each batter in the opposing lineup, get their performance vs each pitch type
3. Compute expected run value per plate appearance:
   EV(PA) = Σ(pitch_usage × batter_outcome_vs_pitch × run_value)
4. Multiply by expected PAs (based on batting order position + game context)
5. Add bullpen factor for innings the starter doesn't cover
6. Blend with sharp market lines for calibration
7. Compare model total vs Vegas line to find +EV game bets

Run values (linear weights):
  Single=0.47, Double=0.78, Triple=1.07, HR=1.40
  Walk=0.33, HBP=0.35, Out=-0.27, K=-0.28
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
SAVE_DIR = Path(__file__).resolve().parent / "saved_model_mlb_runs"

# Standard linear weights (runs above average per event)
# Source: FanGraphs linear weights, 2024 calibration
RUN_VALUES = {
    "K": -0.28,       # strikeout
    "BB": 0.33,       # walk
    "HBP": 0.35,      # hit by pitch
    "1B": 0.47,       # single
    "2B": 0.78,       # double
    "3B": 1.07,       # triple
    "HR": 1.40,       # home run
    "OUT": -0.27,     # non-K out (flyout, groundout, etc.)
}

# League average rates (2024 MLB)
LEAGUE_AVG = {
    "k_rate": 0.224,
    "bb_rate": 0.082,
    "hr_rate": 0.032,
    "single_rate": 0.150,
    "double_rate": 0.044,
    "triple_rate": 0.004,
    "hbp_rate": 0.012,
    "babip": 0.296,
    "runs_per_game": 4.55,
    "starter_ip": 5.4,
    "bullpen_era": 3.95,
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Pitcher Arsenal
# ---------------------------------------------------------------------------

def get_pitcher_arsenal(pitcher_id, game_date=None, conn=None):
    """Get pitcher's pitch mix with per-type effectiveness.

    Returns list of {pitch_type, usage, whiff_rate, csw_rate, chase_rate, zone_rate}
    Uses rolling game-by-game data if available, else season-level.
    """
    close = conn is None
    if conn is None:
        conn = get_db()

    # Try rolling from recent games first (last 5 starts)
    if game_date:
        rows = conn.execute("""
            SELECT pitch_type,
                   SUM(pitches) as total_pitches,
                   SUM(whiff_count) as total_whiff,
                   SUM(csw_count) as total_csw,
                   SUM(chase_count) as total_chase,
                   SUM(zone_count) as total_zone,
                   SUM(swing_count) as total_swing,
                   AVG(avg_velocity) as avg_vel
            FROM mlb_pitcher_game_arsenal
            WHERE pitcher_id = ? AND game_date < ?
            AND game_date IN (
                SELECT DISTINCT game_date FROM mlb_pitcher_game_arsenal
                WHERE pitcher_id = ? AND game_date < ?
                ORDER BY game_date DESC LIMIT 5
            )
            GROUP BY pitch_type
            HAVING total_pitches >= 5
        """, (pitcher_id, game_date, pitcher_id, game_date)).fetchall()
    else:
        rows = []

    if rows:
        total_p = sum(r["total_pitches"] for r in rows)
        arsenal = []
        for r in rows:
            p = r["total_pitches"]
            arsenal.append({
                "pitch_type": r["pitch_type"],
                "usage": p / total_p,
                "whiff_rate": r["total_whiff"] / max(r["total_swing"], 1) if r["total_swing"] else 0,
                "csw_rate": r["total_csw"] / p,
                "chase_rate": r["total_chase"] / max(p - r["total_zone"], 1),
                "zone_rate": r["total_zone"] / p,
                "velocity": r["avg_vel"] or 0,
                "pitches": p,
            })
        if close:
            conn.close()
        return arsenal

    # Fallback to season-level arsenal
    rows = conn.execute(
        "SELECT * FROM mlb_pitcher_arsenal WHERE pitcher_id = ?",
        (pitcher_id,),
    ).fetchall()

    if close:
        conn.close()

    if not rows:
        return []

    return [{
        "pitch_type": r["pitch_type"],
        "usage": r["usage_pct"],
        "whiff_rate": r["whiff_rate"] or 0,
        "csw_rate": r["csw_rate"] or 0,
        "chase_rate": r["chase_rate"] or 0,
        "zone_rate": r["zone_rate"] or 0,
        "velocity": r["avg_velocity"] or 0,
        "pitches": r["pitches"] or 0,
    } for r in rows]


# ---------------------------------------------------------------------------
# Batter Pitch-Type Performance
# ---------------------------------------------------------------------------

def get_batter_vs_arsenal(batter_id, conn=None):
    """Get batter's performance against each pitch type.

    Returns dict: pitch_type -> {whiff_rate, contact_rate, k_rate, chase_rate, pitches}
    """
    close = conn is None
    if conn is None:
        conn = get_db()

    rows = conn.execute(
        "SELECT * FROM mlb_batter_vs_pitch WHERE batter_id = ?",
        (batter_id,),
    ).fetchall()

    if close:
        conn.close()

    if not rows:
        return {}

    return {
        r["pitch_type"]: {
            "whiff_rate": r["whiff_rate"] or LEAGUE_AVG["k_rate"],
            "contact_rate": r["contact_rate"] or (1 - LEAGUE_AVG["k_rate"]),
            "k_rate": r["k_rate_vs_type"] or LEAGUE_AVG["k_rate"],
            "chase_rate": r["chase_rate"] or 0.30,
            "pitches": r["pitches_seen"] or 0,
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Expected Run Value Per Plate Appearance
# ---------------------------------------------------------------------------

def compute_matchup_ev(pitcher_arsenal, batter_vs_pitch, pitcher_stats=None):
    """Compute expected run value for one pitcher vs one batter.

    For each pitch type the pitcher throws:
      - What's the probability of a K? (pitcher's whiff ability × batter's K vulnerability)
      - What's the probability of contact? (1 - K probability, adjusted)
      - On contact, what happens? (use batter's overall profile for hit rates)

    Returns expected runs above average per plate appearance.
    """
    if not pitcher_arsenal:
        return 0.0

    total_k_prob = 0.0
    total_contact_prob = 0.0
    total_bb_prob = 0.0

    for pitch in pitcher_arsenal:
        pt = pitch["pitch_type"]
        usage = pitch["usage"]

        # Batter's performance vs this pitch type
        batter_data = batter_vs_pitch.get(pt)
        if batter_data and batter_data["pitches"] >= 10:
            # Blend pitcher effectiveness with batter vulnerability
            # Pitcher's whiff rate on this pitch × batter's whiff rate vs this pitch
            # Geometric mean gives a balanced matchup estimate
            pitcher_whiff = pitch["whiff_rate"]
            batter_whiff = batter_data["whiff_rate"]
            matchup_whiff = np.sqrt(pitcher_whiff * batter_whiff) if pitcher_whiff > 0 and batter_whiff > 0 else (pitcher_whiff + batter_whiff) / 2

            batter_k = batter_data["k_rate"]
            batter_contact = batter_data["contact_rate"]
        else:
            # No data for this pitch type — use league average
            matchup_whiff = pitch["whiff_rate"]
            batter_k = LEAGUE_AVG["k_rate"]
            batter_contact = 1 - LEAGUE_AVG["k_rate"]

        # K probability from this pitch type (weighted by usage)
        # Higher whiff = more Ks. Scale K rate by relative whiff.
        k_prob = batter_k * (matchup_whiff / max(LEAGUE_AVG["k_rate"], 0.01))
        k_prob = min(k_prob, 0.60)  # cap at 60%

        total_k_prob += usage * k_prob

    # BB probability (mostly pitcher-driven)
    if pitcher_stats:
        bb_rate = pitcher_stats.get("bb_rate", LEAGUE_AVG["bb_rate"])
    else:
        # Estimate from arsenal: low zone_rate = more walks
        avg_zone = np.mean([p["zone_rate"] for p in pitcher_arsenal]) if pitcher_arsenal else 0.45
        bb_rate = LEAGUE_AVG["bb_rate"] * (0.45 / max(avg_zone, 0.30))
    bb_rate = min(bb_rate, 0.20)

    # Contact probability = 1 - K - BB - HBP
    hbp_rate = LEAGUE_AVG["hbp_rate"]
    contact_rate = max(1.0 - total_k_prob - bb_rate - hbp_rate, 0.10)

    # On contact, distribute into outcomes using BABIP-derived rates
    # Higher contact quality (lower whiff) = slightly higher BABIP
    babip = LEAGUE_AVG["babip"]
    hr_share = LEAGUE_AVG["hr_rate"] / max(contact_rate, 0.01)
    hr_share = min(hr_share, 0.15)

    hit_rate = babip * (1 - hr_share) + hr_share
    out_on_contact = 1.0 - hit_rate

    # Split hits into types (league average ratios)
    hit_total = hit_rate * contact_rate
    single_rate = hit_total * 0.64   # ~64% of hits are singles
    double_rate = hit_total * 0.22
    triple_rate = hit_total * 0.02
    hr_rate = hr_share * contact_rate
    out_rate = out_on_contact * contact_rate

    # Expected run value per PA
    ev = (
        total_k_prob * RUN_VALUES["K"]
        + bb_rate * RUN_VALUES["BB"]
        + hbp_rate * RUN_VALUES["HBP"]
        + single_rate * RUN_VALUES["1B"]
        + double_rate * RUN_VALUES["2B"]
        + triple_rate * RUN_VALUES["3B"]
        + hr_rate * RUN_VALUES["HR"]
        + out_rate * RUN_VALUES["OUT"]
    )

    return ev


# ---------------------------------------------------------------------------
# Team Lineup Expected Runs
# ---------------------------------------------------------------------------

def get_team_batters(team_abbrev, conn=None):
    """Get team's batting lineup (top batters by PA)."""
    close = conn is None
    if conn is None:
        conn = get_db()

    rows = conn.execute("""
        SELECT batter_id, batter_name, bat_side, plate_appearances, k_rate, bb_rate
        FROM mlb_batter_stats
        WHERE team_abbrev = ? AND plate_appearances >= 50
        ORDER BY plate_appearances DESC
        LIMIT 13
    """, (team_abbrev,)).fetchall()

    if close:
        conn.close()
    return [dict(r) for r in rows]


def compute_team_expected_runs(pitcher_id, batting_team, game_date=None,
                                pitcher_stats=None, conn=None):
    """Compute expected runs a batting team scores against a pitcher.

    Simulates each batter in the lineup vs the pitcher's arsenal,
    then estimates total runs = Σ(batter_EV × expected_PAs) + league_avg_base.

    Returns dict with expected_runs, per_batter EVs, starter/bullpen split.
    """
    close = conn is None
    if conn is None:
        conn = get_db()

    arsenal = get_pitcher_arsenal(pitcher_id, game_date, conn)
    batters = get_team_batters(batting_team, conn)

    if not arsenal or not batters:
        if close:
            conn.close()
        return {"expected_runs": LEAGUE_AVG["runs_per_game"], "detail": "no_data"}

    # Get pitcher's recent stats for BB rate estimation
    if pitcher_stats is None:
        recent = conn.execute("""
            SELECT AVG(walks * 1.0 / NULLIF(batters_faced, 0)) as bb_rate,
                   AVG(innings_pitched) as avg_ip,
                   AVG(earned_runs) as avg_er
            FROM mlb_pitcher_game_stats
            WHERE pitcher_id = ? AND is_starter = 1
            AND date < COALESCE(?, '9999-12-31')
            ORDER BY date DESC LIMIT 10
        """, (pitcher_id, game_date)).fetchone()
        pitcher_stats = {
            "bb_rate": recent["bb_rate"] or LEAGUE_AVG["bb_rate"],
            "avg_ip": recent["avg_ip"] or LEAGUE_AVG["starter_ip"],
            "avg_er": recent["avg_er"] or (LEAGUE_AVG["runs_per_game"] * LEAGUE_AVG["starter_ip"] / 9),
        } if recent else {}

    # Compute EV for each batter
    batter_evs = []
    for batter in batters[:9]:  # starting 9
        bvp = get_batter_vs_arsenal(batter["batter_id"], conn)
        ev = compute_matchup_ev(arsenal, bvp, pitcher_stats)
        batter_evs.append({
            "batter_name": batter["batter_name"],
            "batter_id": batter["batter_id"],
            "ev_per_pa": round(ev, 4),
            "k_rate": batter["k_rate"],
        })

    # Expected PAs per batter per game (~4.3 PA per 9 innings for top of order)
    # Batting order position affects PA count
    pa_weights = [4.8, 4.7, 4.6, 4.5, 4.3, 4.1, 3.9, 3.8, 3.7]
    if len(batter_evs) < 9:
        pa_weights = pa_weights[:len(batter_evs)]

    # Starter innings expected
    starter_ip = pitcher_stats.get("avg_ip", LEAGUE_AVG["starter_ip"])
    starter_ip = min(max(starter_ip, 3.0), 8.0)
    starter_fraction = starter_ip / 9.0

    # Starter expected runs (matchup-based)
    starter_ev_sum = sum(
        ev["ev_per_pa"] * pa * starter_fraction
        for ev, pa in zip(batter_evs, pa_weights)
    )

    # Convert EV (runs above average) to actual runs
    # Baseline: league average runs in starter's innings
    starter_base_runs = LEAGUE_AVG["runs_per_game"] * starter_fraction
    starter_expected = starter_base_runs + starter_ev_sum

    # Bullpen innings
    bullpen_ip = 9.0 - starter_ip
    bullpen_fraction = bullpen_ip / 9.0
    bullpen_runs = LEAGUE_AVG["runs_per_game"] * bullpen_fraction * (
        LEAGUE_AVG["bullpen_era"] / (LEAGUE_AVG["runs_per_game"] * 9 / 9)
    )

    total_expected = max(starter_expected + bullpen_runs, 0.5)

    if close:
        conn.close()

    return {
        "expected_runs": round(total_expected, 3),
        "starter_runs": round(starter_expected, 3),
        "bullpen_runs": round(bullpen_runs, 3),
        "starter_ip": round(starter_ip, 1),
        "batter_evs": batter_evs,
        "n_batters": len(batter_evs),
        "n_pitches_in_arsenal": len(arsenal),
    }


# ---------------------------------------------------------------------------
# Game Prediction
# ---------------------------------------------------------------------------

def predict_game(home_pitcher_id, away_pitcher_id, home_team, away_team,
                 game_date=None, odds_total=None, odds_home_ml=None):
    """Predict a game by simulating both sides' matchups.

    Returns dict with predicted runs, total, and betting edges.
    """
    conn = get_db()

    # Away team batting vs home pitcher
    away_batting = compute_team_expected_runs(
        home_pitcher_id, away_team, game_date, conn=conn)

    # Home team batting vs away pitcher
    home_batting = compute_team_expected_runs(
        away_pitcher_id, home_team, game_date, conn=conn)

    conn.close()

    pred_home_runs = home_batting["expected_runs"]
    pred_away_runs = away_batting["expected_runs"]
    pred_total = pred_home_runs + pred_away_runs

    # Blend with market if available (30% market, 70% model)
    blended_total = pred_total
    if odds_total is not None:
        blended_total = 0.70 * pred_total + 0.30 * odds_total

    # Simulate score distribution (Poisson)
    from nhl_game_model import simulate_game, _american_to_prob, _american_to_decimal
    sim = simulate_game(pred_home_runs, pred_away_runs, correlation=0.10)

    # Find edges
    bets = []
    if odds_total is not None:
        over_key = f"over_{odds_total}"
        under_key = f"under_{odds_total}"
        if over_key in sim:
            # Standard -110 odds
            imp = 0.5238
            for side, prob, pick in [
                ("OVER", sim[over_key], f"OVER {odds_total}"),
                ("UNDER", sim[under_key], f"UNDER {odds_total}"),
            ]:
                edge = prob - imp
                ev = prob * 0.909 - (1 - prob)  # -110 payout
                if edge > 0.02 and ev > 0:
                    bets.append({
                        "market": "Total", "pick": pick,
                        "model_prob": round(prob, 4),
                        "implied_prob": round(imp, 4),
                        "edge": round(edge, 4),
                        "ev": round(ev, 4),
                    })

    if odds_home_ml is not None:
        imp = _american_to_prob(odds_home_ml)
        dec = _american_to_decimal(odds_home_ml)
        edge = sim["home_win_prob"] - imp
        ev = sim["home_win_prob"] * (dec - 1) - (1 - sim["home_win_prob"])
        if edge > 0.03 and ev > 0:
            bets.append({
                "market": "ML", "pick": home_team,
                "model_prob": round(sim["home_win_prob"], 4),
                "implied_prob": round(imp, 4),
                "edge": round(edge, 4),
                "ev": round(ev, 4),
            })

    return {
        "home_team": home_team,
        "away_team": away_team,
        "pred_home_runs": round(pred_home_runs, 2),
        "pred_away_runs": round(pred_away_runs, 2),
        "pred_total": round(pred_total, 2),
        "blended_total": round(blended_total, 2),
        "sim": sim,
        "home_detail": home_batting,
        "away_detail": away_batting,
        "bets": sorted(bets, key=lambda b: b["edge"], reverse=True),
    }


# ---------------------------------------------------------------------------
# Training: Learn a correction factor from historical matchups
# ---------------------------------------------------------------------------

def build_training_data():
    """Build training set: predicted runs (matchup-based) vs actual runs.

    For each 2024 game with a starting pitcher, compute the matchup-based
    expected runs and compare with actual game score.
    """
    conn = get_db()

    # Get 2024 games with scores
    games = conn.execute("""
        SELECT g.game_pk, g.date, g.home_team_abbrev, g.away_team_abbrev,
               g.home_score, g.away_score
        FROM mlb_games g
        WHERE g.status = 'Final' AND g.date LIKE '2024%'
        ORDER BY g.date
    """).fetchall()

    if not games:
        conn.close()
        return pd.DataFrame()

    # Get starters for each game
    starters = conn.execute("""
        SELECT game_pk, pitcher_id, pitcher_name, team_abbrev, opponent_abbrev,
               is_home, innings_pitched, earned_runs, strikeouts, walks, batters_faced
        FROM mlb_pitcher_game_stats
        WHERE is_starter = 1 AND date LIKE '2024%'
    """).fetchall()

    starter_map = {}
    for s in starters:
        key = (s["game_pk"], s["is_home"])
        starter_map[key] = dict(s)

    logger.info("Building matchup training data: %d games, %d starters",
                len(games), len(starters))

    records = []
    for g in games:
        gk = g["game_pk"]
        gdate = g["date"]
        home = g["home_team_abbrev"]
        away = g["away_team_abbrev"]

        home_starter = starter_map.get((gk, 1))
        away_starter = starter_map.get((gk, 0))

        if not home_starter or not away_starter:
            continue

        # Compute matchup-based expected runs for each side
        try:
            # Away team batting vs home pitcher
            away_batting = compute_team_expected_runs(
                home_starter["pitcher_id"], away, gdate, conn=conn)

            # Home team batting vs away pitcher
            home_batting = compute_team_expected_runs(
                away_starter["pitcher_id"], home, gdate, conn=conn)
        except Exception:
            continue

        records.append({
            "game_pk": gk,
            "date": gdate,
            "home_team": home,
            "away_team": away,
            # Actuals
            "actual_home_runs": g["home_score"],
            "actual_away_runs": g["away_score"],
            "actual_total": g["home_score"] + g["away_score"],
            # Matchup predictions
            "matchup_home_runs": home_batting["expected_runs"],
            "matchup_away_runs": away_batting["expected_runs"],
            "matchup_total": home_batting["expected_runs"] + away_batting["expected_runs"],
            # Starter quality signals
            "home_starter_ip": home_batting.get("starter_ip", 5.0),
            "away_starter_ip": away_batting.get("starter_ip", 5.0),
            "home_n_batters": home_batting.get("n_batters", 0),
            "away_n_batters": away_batting.get("n_batters", 0),
            # Pitcher IDs for reference
            "home_pitcher_id": home_starter["pitcher_id"],
            "away_pitcher_id": away_starter["pitcher_id"],
        })

    conn.close()

    df = pd.DataFrame(records)
    logger.info("Training data: %d games with complete matchup data", len(df))
    return df


def evaluate_model():
    """Evaluate matchup model against actual game scores."""
    df = build_training_data()
    if df.empty:
        print("No training data")
        return

    df["date"] = pd.to_datetime(df["date"])

    # Overall accuracy
    total_mae = np.mean(np.abs(df["matchup_total"] - df["actual_total"]))
    home_mae = np.mean(np.abs(df["matchup_home_runs"] - df["actual_home_runs"]))
    away_mae = np.mean(np.abs(df["matchup_away_runs"] - df["actual_away_runs"]))

    print(f"Games: {len(df)}")
    print(f"Total MAE: {total_mae:.3f} (avg actual: {df['actual_total'].mean():.2f}, avg pred: {df['matchup_total'].mean():.2f})")
    print(f"Home MAE:  {home_mae:.3f} (avg actual: {df['actual_home_runs'].mean():.2f}, avg pred: {df['matchup_home_runs'].mean():.2f})")
    print(f"Away MAE:  {away_mae:.3f} (avg actual: {df['actual_away_runs'].mean():.2f}, avg pred: {df['matchup_away_runs'].mean():.2f})")

    # Bias
    total_bias = (df["matchup_total"] - df["actual_total"]).mean()
    print(f"Total bias: {total_bias:+.3f}")

    # Correlation
    corr = df["matchup_total"].corr(df["actual_total"])
    print(f"Correlation (pred vs actual total): {corr:.3f}")

    # By month
    df["month"] = df["date"].dt.month
    print("\nBy month:")
    for m, grp in df.groupby("month"):
        mae = np.mean(np.abs(grp["matchup_total"] - grp["actual_total"]))
        print(f"  Month {m}: MAE={mae:.3f}, n={len(grp)}")

    # Over/under accuracy at common lines
    print("\nOver/Under accuracy:")
    for line in [7.5, 8.0, 8.5, 9.0, 9.5]:
        pred_over = (df["matchup_total"] > line).mean()
        actual_over = (df["actual_total"] > line).mean()
        accuracy = ((df["matchup_total"] > line) == (df["actual_total"] > line)).mean()
        print(f"  Line {line}: pred {pred_over:.1%} over, actual {actual_over:.1%} over, accuracy {accuracy:.1%}")

    return df


# ---------------------------------------------------------------------------
# Walk-Forward Backtest
# ---------------------------------------------------------------------------

def run_walkforward(min_train_games=300, test_days=14, step_days=14):
    """Walk-forward backtest: matchup model + XGBoost correction vs game totals."""

    df = build_training_data()
    if df.empty:
        return {"error": "No data"}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Load game odds for total lines
    conn = get_db()
    odds_rows = conn.execute("""
        SELECT g.game_pk, go.outcome_point as total_line
        FROM mlb_game_odds go
        JOIN mlb_games g ON go.event_id = (
            SELECT event_id FROM mlb_odds_events
            WHERE home_team_abbrev = g.home_team_abbrev
            AND game_date = g.date LIMIT 1
        )
        WHERE go.market = 'totals' AND go.outcome_name = 'Over'
        AND go.bookmaker = 'draftkings'
    """).fetchall()
    conn.close()

    # Simpler: just get odds from mlb_game_odds directly
    MLB_NAME_TO_ABBREV = {
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

    conn = get_db()
    odds_map = {}
    for r in conn.execute("""
        SELECT go.game_date, go.home_team, go.outcome_point
        FROM mlb_game_odds go
        WHERE go.market = 'totals' AND go.outcome_name = 'Over'
        AND go.bookmaker = 'draftkings'
        GROUP BY go.game_date, go.home_team
    """).fetchall():
        abbrev = MLB_NAME_TO_ABBREV.get(r["home_team"])
        if abbrev:
            odds_map[(r["game_date"], abbrev)] = r["outcome_point"]
    conn.close()
    logger.info("Odds map: %d games with total lines", len(odds_map))

    # Add odds to df
    df["vegas_total"] = df.apply(
        lambda r: odds_map.get((r["date"].strftime("%Y-%m-%d"), r["home_team"]), None),
        axis=1,
    )

    logger.info("Games with odds: %d / %d", df["vegas_total"].notna().sum(), len(df))

    # Features for correction model
    feat_cols = ["matchup_home_runs", "matchup_away_runs", "matchup_total",
                 "home_starter_ip", "away_starter_ip",
                 "home_n_batters", "away_n_batters"]
    # Add vegas total as feature when available
    feat_cols_with_odds = feat_cols + ["vegas_total"]

    season_start = df["date"].min()
    season_end = df["date"].max()
    first_test = season_start + timedelta(days=90)

    all_bets = []
    windows = []
    current = first_test
    wnum = 0

    while current < season_end:
        wnum += 1
        test_end = min(current + timedelta(days=test_days), season_end)
        train = df[df["date"] < current]
        test = df[(df["date"] >= current) & (df["date"] <= test_end)]

        if len(train) < min_train_games or len(test) == 0:
            current += timedelta(days=step_days)
            continue

        logger.info("Window %d: train %d, test %s-%s (%d games)",
                    wnum, len(train), test["date"].min().date(),
                    test["date"].max().date(), len(test))

        # Train correction model: learn residual off Vegas line using matchup features
        # The matchup signals tell us WHERE the line might be wrong
        train_odds = train[train["vegas_total"].notna()].copy()

        avail = [f for f in feat_cols_with_odds if f in train.columns]

        if len(train_odds) >= 100:
            # Target: actual_total - vegas_total (what Vegas got wrong)
            train_odds["residual"] = train_odds["actual_total"] - train_odds["vegas_total"]
            mdl = XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=5.0,
                min_child_weight=20, random_state=42, verbosity=0,
            )
            mdl.fit(train_odds[avail].values, train_odds["residual"].values)

            # For test games WITH odds: corrected = vegas + predicted residual
            test_copy = test.copy()
            pred_residual = mdl.predict(test_copy[avail].values)
            corrected_total = np.where(
                test_copy["vegas_total"].notna(),
                test_copy["vegas_total"].fillna(0).values + pred_residual,
                test_copy["matchup_total"].values,  # fallback
            )
        else:
            # Not enough odds data — use raw matchup
            corrected_total = test["matchup_total"].values

        # Also get raw matchup prediction
        raw_total = test["matchup_total"].values
        actual_total = test["actual_total"].values

        # Evaluate
        raw_mae = np.mean(np.abs(raw_total - actual_total))
        corr_mae = np.mean(np.abs(corrected_total - actual_total))

        windows.append({
            "window": wnum, "train": len(train), "test": len(test),
            "test_start": test["date"].min().strftime("%Y-%m-%d"),
            "test_end": test["date"].max().strftime("%Y-%m-%d"),
            "raw_mae": round(raw_mae, 3),
            "corrected_mae": round(corr_mae, 3),
        })

        # Generate bets on games with Vegas lines
        from nhl_game_model import simulate_game
        for i in range(len(test)):
            row = test.iloc[i]
            vl = row.get("vegas_total")
            if vl is None or pd.isna(vl):
                continue

            pred = float(corrected_total[i])
            actual = int(row["actual_total"])
            matchup_raw = float(row["matchup_total"])
            matchup_diff = matchup_raw - vl  # negative = matchup says fewer runs than Vegas

            # Simulate score using corrected total
            pred_home = pred * 0.50
            pred_away = pred * 0.50
            sim = simulate_game(pred_home, pred_away, correlation=0.10)

            over_key = f"over_{vl}"
            under_key = f"under_{vl}"
            if over_key not in sim:
                for try_line in [vl, vl + 0.5, vl - 0.5]:
                    ok = f"over_{try_line}"
                    if ok in sim:
                        over_key = ok
                        under_key = f"under_{try_line}"
                        break

            if over_key not in sim:
                continue

            imp = 0.5238  # -110
            for side, prob, won in [
                ("OVER", sim[over_key], actual > vl),
                ("UNDER", sim[under_key], actual <= vl),
            ]:
                edge = prob - imp
                ev = prob * 0.909 - (1 - prob)
                all_bets.append({
                    "window": wnum, "date": row["date"].strftime("%Y-%m-%d"),
                    "home": row["home_team"], "away": row["away_team"],
                    "pick": f"{side} {vl}",
                    "pred_total": round(pred, 2),
                    "vegas_line": vl,
                    "matchup_diff": round(matchup_diff, 2),  # neg=under signal
                    "model_prob": round(prob, 4),
                    "implied_prob": round(imp, 4),
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                    "won": won,
                    "actual_total": actual,
                })

        current += timedelta(days=step_days)

    bets_df = pd.DataFrame(all_bets)

    # Evaluate strategies
    print("\n" + "=" * 90)
    print("  MLB PITCH-MATCHUP MODEL: WALK-FORWARD RESULTS")
    print("=" * 90)
    print(f"  Season: {season_start.date()} to {season_end.date()}")
    print(f"  Windows: {len(windows)}")

    print(f"\n  {'Win':>4s} {'Train':>6s} {'Test Period':>24s} {'Raw MAE':>9s} {'Corrected':>10s}")
    print("  " + "-" * 60)
    for w in windows:
        print(f"  {w['window']:4d} {w['train']:6d} {w['test_start']} - {w['test_end']} "
              f"{w['raw_mae']:9.3f} {w['corrected_mae']:10.3f}")

    if bets_df.empty:
        print("\n  No bets (no odds data matched)")
        return {"windows": windows}

    ev_plus = bets_df[bets_df["ev"] > 0]
    print(f"\n  Total bets: {len(bets_df)}, +EV bets: {len(ev_plus)}")

    # Simulate Kelly betting
    # Matchup-direction strategies (the key insight)
    b = bets_df  # shorthand
    is_under = b["pick"].str.contains("UNDER")
    is_over = b["pick"].str.contains("OVER")
    matchup_under = b["matchup_diff"] < -0.3
    matchup_strong_under = b["matchup_diff"] < -1.0
    ev_pos = b["ev"] > 0

    for name, subset in [
        ("All +EV", b[ev_pos]),
        ("OVER +EV", b[ev_pos & is_over]),
        ("UNDER +EV", b[ev_pos & is_under]),
        ("Edge >= 5%", b[ev_pos & (b["edge"] >= 0.05)]),
        ("Edge >= 8%", b[ev_pos & (b["edge"] >= 0.08)]),
        ("Edge >= 10%", b[ev_pos & (b["edge"] >= 0.10)]),
        # Matchup-direction strategies (core finding)
        ("UNDER matchup<Vegas", b[is_under & matchup_under]),
        ("UNDER matchup<<Vegas", b[is_under & matchup_strong_under]),
        ("OVER matchup>Vegas", b[is_over & (b["matchup_diff"] > 0.3)]),
        ("UNDER m<V +EV", b[is_under & matchup_under & ev_pos]),
        ("UNDER m<<V +EV", b[is_under & matchup_strong_under & ev_pos]),
        ("UNDER m<V edge5%", b[is_under & matchup_under & (b["edge"] >= 0.05)]),
        ("UNDER m<<V edge5%", b[is_under & matchup_strong_under & (b["edge"] >= 0.05)]),
    ]:
        if len(subset) < 5:
            continue
        bankroll = 100.0
        for _, b in subset.sort_values("date").iterrows():
            dec = 1.909  # -110
            kf = min(max(b["ev"] / (dec - 1), 0) * 0.25, 0.08)
            wager = bankroll * kf
            if wager < 1:
                continue
            if b["won"]:
                bankroll += wager * (dec - 1)
            else:
                bankroll -= wager
        profit = bankroll - 100
        wagered = sum(
            100 * min(max(b["ev"] / 0.909, 0) * 0.25, 0.08)
            for _, b in subset.iterrows() if b["ev"] > 0
        )
        wr = subset["won"].mean() * 100
        yld = profit / max(wagered, 1) * 100
        print(f"  {name:20s}: {len(subset):5d} bets, {wr:.1f}% WR, "
              f"profit ${profit:+.2f}, yield {yld:+.1f}%")

    return {"windows": windows, "bets_df": bets_df}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run_walkforward()
