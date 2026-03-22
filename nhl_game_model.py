"""
NHL Game-Level Prediction Model.

Predicts game outcomes (total goals, win probability, spread) by blending:
  - Bottom-up: aggregated V1/V2 player SOG predictions
  - Top-down: team rolling stats (goals, SOG, xG from MoneyPuck)
  - Market signals: moneyline implied probabilities, total line

Compares predictions against sportsbook odds to find +EV game bets
on moneyline, totals (over/under), and puckline (spread).
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

import data_collector
import moneypuck_collector
import nhl_api
import nhl_odds_collector

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).resolve().parent / "saved_model_game"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

GAME_FEATURES = [
    # Team rolling stats (last 10 games)
    "home_gf_10", "home_ga_10", "away_gf_10", "away_ga_10",
    "home_sog_for_10", "home_sog_against_10",
    "away_sog_for_10", "away_sog_against_10",
    # MoneyPuck advanced (last 10)
    "home_xgf_10", "home_xga_10", "away_xgf_10", "away_xga_10",
    "home_hdcf_10", "home_hdca_10", "away_hdcf_10", "away_hdca_10",
    # Rest / schedule
    "home_rest_days", "away_rest_days",
    # Odds-derived
    "game_total_line", "implied_home_total", "implied_away_total",
    "home_win_prob_implied",
]

# Models
_totals_model = None
_win_model = None
_model_metrics = {}


def _build_team_rolling(conn):
    """Build rolling team stats from games table.

    Returns dict: (game_date, team) -> {gf_10, ga_10, sog_for_10, sog_against_10, rest_days}
    """
    games = pd.read_sql_query(
        "SELECT game_id, date, home_team, away_team, home_score, away_score "
        "FROM games WHERE status = 'FINAL' ORDER BY date",
        conn,
    )
    if games.empty:
        return {}

    # SOG per team per game from player_game_stats
    sog_df = pd.read_sql_query(
        """SELECT g.game_id, g.date, pgs.team, SUM(pgs.shots) as team_sog
           FROM player_game_stats pgs
           JOIN games g ON pgs.game_id = g.game_id
           WHERE pgs.position IN ('L','C','R','D')
           GROUP BY g.game_id, pgs.team
           ORDER BY g.date""",
        conn,
    )
    sog_map = {}  # (game_id, team) -> sog
    for _, r in sog_df.iterrows():
        sog_map[(r["game_id"], r["team"])] = r["team_sog"]

    # Build per-team game history
    team_history = {}  # team -> list of {date, gf, ga, sog_for, sog_against}
    rolling = {}  # (date, team) -> {gf_10, ga_10, ...}

    for _, g in games.iterrows():
        gid = g["game_id"]
        gdate = g["date"]
        home = g["home_team"]
        away = g["away_team"]
        hs = g["home_score"] or 0
        as_ = g["away_score"] or 0

        home_sog = sog_map.get((gid, home), 0)
        away_sog = sog_map.get((gid, away), 0)

        for team, gf, ga, sf, sa in [
            (home, hs, as_, home_sog, away_sog),
            (away, as_, hs, away_sog, home_sog),
        ]:
            if team not in team_history:
                team_history[team] = []

            hist = team_history[team]

            # Store rolling BEFORE adding this game (pre-game features)
            if len(hist) >= 5:
                recent = hist[-10:]
                n = len(recent)
                last_date = recent[-1]["date"]
                rest = (pd.Timestamp(gdate) - pd.Timestamp(last_date)).days

                rolling[(gdate, team)] = {
                    "gf_10": sum(h["gf"] for h in recent) / n,
                    "ga_10": sum(h["ga"] for h in recent) / n,
                    "sog_for_10": sum(h["sf"] for h in recent) / n,
                    "sog_against_10": sum(h["sa"] for h in recent) / n,
                    "rest_days": min(rest, 5),
                }

            hist.append({"date": gdate, "gf": gf, "ga": ga, "sf": sf, "sa": sa})

    return rolling


def _build_mp_team_rolling(conn):
    """Build MoneyPuck xG and high-danger rolling stats per team.

    Returns dict: (game_date, team) -> {xgf_10, xga_10, hdcf_10, hdca_10}
    """
    mp_df = pd.read_sql_query(
        """SELECT game_id, game_date, team,
                  SUM(on_ice_xg_for) / COUNT(DISTINCT player_id) * 18 as game_xgf,
                  SUM(on_ice_xg_against) / COUNT(DISTINCT player_id) * 18 as game_xga,
                  SUM(on_ice_hd_shots_for) / COUNT(DISTINCT player_id) * 18 as game_hdcf,
                  SUM(on_ice_hd_shots_against) / COUNT(DISTINCT player_id) * 18 as game_hdca
           FROM mp_player_game
           WHERE situation = 'all'
           GROUP BY game_id, team
           ORDER BY game_date""",
        conn,
    )
    if mp_df.empty:
        return {}

    team_history = {}
    rolling = {}

    for _, r in mp_df.iterrows():
        team = r["team"]
        gdate = r["game_date"]

        if team not in team_history:
            team_history[team] = []

        hist = team_history[team]
        if len(hist) >= 5:
            recent = hist[-10:]
            n = len(recent)
            rolling[(gdate, team)] = {
                "xgf_10": sum(h["xgf"] for h in recent) / n,
                "xga_10": sum(h["xga"] for h in recent) / n,
                "hdcf_10": sum(h["hdcf"] for h in recent) / n,
                "hdca_10": sum(h["hdca"] for h in recent) / n,
            }

        hist.append({
            "xgf": r["game_xgf"] or 0,
            "xga": r["game_xga"] or 0,
            "hdcf": r["game_hdcf"] or 0,
            "hdca": r["game_hdca"] or 0,
        })

    return rolling


def build_game_training_df():
    """Build training DataFrame with one row per completed game.

    All features use pre-game data only (no leakage).
    """
    conn = data_collector.get_db()

    games = pd.read_sql_query(
        "SELECT game_id, date, home_team, away_team, home_score, away_score "
        "FROM games WHERE status = 'FINAL' ORDER BY date",
        conn,
    )
    if games.empty:
        conn.close()
        return pd.DataFrame()

    # Build rolling stats
    logger.info("Building team rolling stats...")
    team_rolling = _build_team_rolling(conn)

    mp_conn = moneypuck_collector.get_db()
    mp_rolling = _build_mp_team_rolling(mp_conn)
    mp_conn.close()

    # Load odds
    odds_bulk = nhl_odds_collector.load_game_odds_bulk()

    conn.close()

    records = []
    for _, g in games.iterrows():
        gdate = g["date"]
        home = g["home_team"]
        away = g["away_team"]

        hr = team_rolling.get((gdate, home))
        ar = team_rolling.get((gdate, away))
        if not hr or not ar:
            continue

        mhr = mp_rolling.get((gdate, home), {})
        mar = mp_rolling.get((gdate, away), {})

        # Odds features
        odds_key = (gdate, home)
        odds = odds_bulk.get(odds_key, {})

        total_line = odds.get("game_total")
        imp_home = odds.get("implied_home_total")
        imp_away = odds.get("implied_away_total")
        home_wp = odds.get("home_win_prob")

        record = {
            "game_id": g["game_id"],
            "date": gdate,
            "home_team": home,
            "away_team": away,
            # Targets
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "total_goals": (g["home_score"] or 0) + (g["away_score"] or 0),
            "home_win": 1 if (g["home_score"] or 0) > (g["away_score"] or 0) else 0,
            "goal_diff": (g["home_score"] or 0) - (g["away_score"] or 0),
            # Team rolling
            "home_gf_10": hr["gf_10"],
            "home_ga_10": hr["ga_10"],
            "away_gf_10": ar["gf_10"],
            "away_ga_10": ar["ga_10"],
            "home_sog_for_10": hr["sog_for_10"],
            "home_sog_against_10": hr["sog_against_10"],
            "away_sog_for_10": ar["sog_for_10"],
            "away_sog_against_10": ar["sog_against_10"],
            "home_rest_days": hr["rest_days"],
            "away_rest_days": ar["rest_days"],
            # MoneyPuck
            "home_xgf_10": mhr.get("xgf_10"),
            "home_xga_10": mhr.get("xga_10"),
            "away_xgf_10": mar.get("xgf_10"),
            "away_xga_10": mar.get("xga_10"),
            "home_hdcf_10": mhr.get("hdcf_10"),
            "home_hdca_10": mhr.get("hdca_10"),
            "away_hdcf_10": mar.get("hdcf_10"),
            "away_hdca_10": mar.get("hdca_10"),
            # Odds
            "game_total_line": total_line,
            "implied_home_total": imp_home,
            "implied_away_total": imp_away,
            "home_win_prob_implied": home_wp,
            # Raw odds for walk-forward
            "home_ml": odds.get("home_ml"),
            "away_ml": odds.get("away_ml"),
            "total_over_price": None,  # TODO from raw data
            "total_under_price": None,
            "home_spread_point": odds.get("home_spread_point"),
            "home_spread_price": odds.get("home_spread_price"),
            "away_spread_price": odds.get("away_spread_price"),
        }
        records.append(record)

    df = pd.DataFrame(records)
    logger.info("Game training data: %d games", len(df))
    return df


# ---------------------------------------------------------------------------
# Score simulation
# ---------------------------------------------------------------------------

def simulate_game(pred_home_goals, pred_away_goals, n_sims=50000, correlation=0.12):
    """Simulate game scores using correlated Poisson model.

    Returns dict with P(home_win), P(over X.5) for various X, P(home covers -1.5).
    """
    rng = np.random.default_rng(42)

    # Correlated Poisson via normal copula
    mean = [np.log(max(pred_home_goals, 0.5)), np.log(max(pred_away_goals, 0.5))]
    cov = [[1, correlation], [correlation, 1]]
    z = rng.multivariate_normal(mean, cov, size=n_sims)
    lam_h = np.exp(z[:, 0])
    lam_a = np.exp(z[:, 1])
    home_goals = rng.poisson(lam_h)
    away_goals = rng.poisson(lam_a)

    total = home_goals + away_goals
    diff = home_goals.astype(int) - away_goals.astype(int)

    result = {
        "pred_home_goals": round(pred_home_goals, 2),
        "pred_away_goals": round(pred_away_goals, 2),
        "pred_total": round(pred_home_goals + pred_away_goals, 2),
        "home_win_prob": round(float((diff > 0).mean()), 4),
        "away_win_prob": round(float((diff < 0).mean()), 4),
        "draw_prob": round(float((diff == 0).mean()), 4),
        "home_cover_minus_1_5": round(float((diff >= 2).mean()), 4),
        "away_cover_plus_1_5": round(float((diff <= 1).mean()), 4),
    }

    for line in [4.5, 5.0, 5.5, 6.0, 6.5]:
        result[f"over_{line}"] = round(float((total > line).mean()), 4)
        result[f"under_{line}"] = round(float((total <= line).mean()), 4)

    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model():
    """Train game-level models (totals + win probability)."""
    global _totals_model, _win_model, _model_metrics

    df = build_game_training_df()
    if df.empty or len(df) < 100:
        logger.warning("Not enough game data (%d)", len(df))
        return {}

    df["date"] = pd.to_datetime(df["date"])
    avail = [f for f in GAME_FEATURES if f in df.columns]

    # Fill NaN for XGBoost (it handles natively, but need consistent columns)
    cutoff = df["date"].max() - timedelta(days=14)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    if len(train) < 50 or len(test) < 5:
        logger.warning("Not enough train/test data")
        return {}

    X_train = train[avail].values
    X_test = test[avail].values

    # --- Totals model ---
    y_total_train = train["total_goals"].values
    y_total_test = test["total_goals"].values

    _totals_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=10, random_state=42, verbosity=0,
    )
    _totals_model.fit(X_train, y_total_train)

    pred_total = _totals_model.predict(X_test)
    total_mae = float(np.mean(np.abs(pred_total - y_total_test)))
    total_rmse = float(np.sqrt(np.mean((pred_total - y_total_test) ** 2)))
    logger.info("Totals model: MAE=%.3f RMSE=%.3f (%d train / %d test)",
                total_mae, total_rmse, len(train), len(test))

    # --- Win probability model ---
    y_win_train = train["home_win"].values
    y_win_test = test["home_win"].values

    _win_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=10, random_state=42, verbosity=0,
        eval_metric="logloss",
    )
    _win_model.fit(X_train, y_win_train)

    pred_win = _win_model.predict_proba(X_test)[:, 1]
    win_acc = float(((pred_win > 0.5) == y_win_test).mean())
    # Brier score
    brier = float(np.mean((pred_win - y_win_test) ** 2))
    logger.info("Win model: accuracy=%.3f brier=%.4f", win_acc, brier)

    # Feature importance
    feat_imp_total = dict(sorted(
        zip(avail, _totals_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:8])
    feat_imp_win = dict(sorted(
        zip(avail, _win_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:8])

    _model_metrics = {
        "totals": {
            "mae": round(total_mae, 3),
            "rmse": round(total_rmse, 3),
            "train": len(train), "test": len(test),
            "top_features": {k: round(float(v), 4) for k, v in feat_imp_total.items()},
        },
        "win_prob": {
            "accuracy": round(win_acc, 3),
            "brier": round(brier, 4),
            "top_features": {k: round(float(v), 4) for k, v in feat_imp_win.items()},
        },
    }

    save_model()
    return _model_metrics


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if _totals_model:
        _totals_model.save_model(str(SAVE_DIR / "totals.json"))
    if _win_model:
        _win_model.save_model(str(SAVE_DIR / "win_prob.json"))
    with open(SAVE_DIR / "meta.json", "w") as f:
        json.dump({"metrics": _model_metrics}, f)
    logger.info("Game model saved")


def load_model() -> bool:
    global _totals_model, _win_model, _model_metrics
    meta_path = SAVE_DIR / "meta.json"
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        _model_metrics = json.load(f).get("metrics", {})

    tp = SAVE_DIR / "totals.json"
    wp = SAVE_DIR / "win_prob.json"
    if tp.exists():
        _totals_model = XGBRegressor()
        _totals_model.load_model(str(tp))
    if wp.exists():
        _win_model = XGBClassifier()
        _win_model.load_model(str(wp))

    logger.info("Game model loaded (totals MAE: %s)",
                _model_metrics.get("totals", {}).get("mae", "?"))
    return True


def get_model_metrics():
    return _model_metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _american_to_prob(odds):
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5


def _american_to_decimal(odds):
    if odds > 0:
        return odds / 100.0 + 1
    elif odds < 0:
        return 100.0 / (-odds) + 1
    return 2.0


def predict_game(home_abbrev, away_abbrev, game_date=None):
    """Predict a single game and find +EV bets.

    Returns dict with predictions, simulation results, and bet recommendations.
    """
    if _totals_model is None or _win_model is None:
        return None

    if game_date is None:
        game_date = date.today().isoformat()

    conn = data_collector.get_db()
    team_rolling = _build_team_rolling(conn)
    mp_conn = moneypuck_collector.get_db()
    mp_rolling = _build_mp_team_rolling(mp_conn)
    mp_conn.close()
    conn.close()

    # Find latest rolling stats for each team (may not be from today)
    def _latest_rolling(rolling_dict, team, before_date):
        # Try exact date first, then search backwards
        if (before_date, team) in rolling_dict:
            return rolling_dict[(before_date, team)]
        # Find most recent entry for this team
        candidates = [(d, t) for (d, t) in rolling_dict if t == team and d <= before_date]
        if candidates:
            best = max(candidates, key=lambda x: x[0])
            return rolling_dict[best]
        return None

    hr = _latest_rolling(team_rolling, home_abbrev, game_date)
    ar = _latest_rolling(team_rolling, away_abbrev, game_date)
    if not hr or not ar:
        return None

    mhr = _latest_rolling(mp_rolling, home_abbrev, game_date) or {}
    mar = _latest_rolling(mp_rolling, away_abbrev, game_date) or {}

    # Odds
    odds_ctx = nhl_odds_collector.get_game_context(home_abbrev, away_abbrev, game_date)
    game_odds = nhl_odds_collector.get_game_odds_for_date(game_date)
    this_game = None
    for g in game_odds:
        if g.get("home_abbrev") == home_abbrev and g.get("away_abbrev") == away_abbrev:
            this_game = g
            break

    total_line = odds_ctx.get("game_total") if odds_ctx else None
    imp_home = odds_ctx.get("implied_home_total") if odds_ctx else None
    imp_away = odds_ctx.get("implied_away_total") if odds_ctx else None
    home_wp_implied = odds_ctx.get("home_win_prob") if odds_ctx else None

    features = {
        "home_gf_10": hr["gf_10"], "home_ga_10": hr["ga_10"],
        "away_gf_10": ar["gf_10"], "away_ga_10": ar["ga_10"],
        "home_sog_for_10": hr["sog_for_10"], "home_sog_against_10": hr["sog_against_10"],
        "away_sog_for_10": ar["sog_for_10"], "away_sog_against_10": ar["sog_against_10"],
        "home_xgf_10": mhr.get("xgf_10"), "home_xga_10": mhr.get("xga_10"),
        "away_xgf_10": mar.get("xgf_10"), "away_xga_10": mar.get("xga_10"),
        "home_hdcf_10": mhr.get("hdcf_10"), "home_hdca_10": mhr.get("hdca_10"),
        "away_hdcf_10": mar.get("hdcf_10"), "away_hdca_10": mar.get("hdca_10"),
        "home_rest_days": hr["rest_days"], "away_rest_days": ar["rest_days"],
        "game_total_line": total_line,
        "implied_home_total": imp_home, "implied_away_total": imp_away,
        "home_win_prob_implied": home_wp_implied,
    }

    avail = [f for f in GAME_FEATURES if f in features]
    X = np.array([[features.get(f, np.nan) for f in avail]])

    pred_total = float(_totals_model.predict(X)[0])
    pred_home_wp = float(_win_model.predict_proba(X)[:, 1][0])

    # Split total into home/away expected goals using win prob ratio
    if pred_home_wp > 0 and pred_home_wp < 1:
        pred_home_goals = pred_total * pred_home_wp / (pred_home_wp + (1 - pred_home_wp))
        pred_away_goals = pred_total - pred_home_goals
    else:
        pred_home_goals = pred_total / 2
        pred_away_goals = pred_total / 2

    # Simulate
    sim = simulate_game(pred_home_goals, pred_away_goals)

    # --- Find +EV bets ---
    bets = []

    # Moneyline
    if this_game:
        home_ml = this_game.get("home_ml")
        away_ml = this_game.get("away_ml")

        if home_ml is not None:
            imp = _american_to_prob(home_ml)
            edge = sim["home_win_prob"] - imp
            dec = _american_to_decimal(home_ml)
            ev = sim["home_win_prob"] * (dec - 1) - (1 - sim["home_win_prob"])
            if edge > 0.02 and ev > 0:
                bets.append({
                    "market": "Moneyline",
                    "pick": home_abbrev,
                    "model_prob": sim["home_win_prob"],
                    "implied_prob": round(imp, 4),
                    "odds": home_ml,
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                })

        if away_ml is not None:
            imp = _american_to_prob(away_ml)
            edge = sim["away_win_prob"] - imp
            dec = _american_to_decimal(away_ml)
            ev = sim["away_win_prob"] * (dec - 1) - (1 - sim["away_win_prob"])
            if edge > 0.02 and ev > 0:
                bets.append({
                    "market": "Moneyline",
                    "pick": away_abbrev,
                    "model_prob": sim["away_win_prob"],
                    "implied_prob": round(imp, 4),
                    "odds": away_ml,
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                })

        # Totals
        tl = this_game.get("total_line")
        if tl is not None:
            over_key = f"over_{tl}"
            under_key = f"under_{tl}"
            if over_key in sim:
                over_price = this_game.get("total_over")
                under_price = this_game.get("total_under")

                if over_price is not None:
                    imp = _american_to_prob(over_price)
                    edge = sim[over_key] - imp
                    dec = _american_to_decimal(over_price)
                    ev = sim[over_key] * (dec - 1) - (1 - sim[over_key])
                    if edge > 0.02 and ev > 0:
                        bets.append({
                            "market": f"Over {tl}",
                            "pick": "OVER",
                            "model_prob": sim[over_key],
                            "implied_prob": round(imp, 4),
                            "odds": over_price,
                            "edge": round(edge, 4),
                            "ev": round(ev, 4),
                        })

                if under_price is not None:
                    imp = _american_to_prob(under_price)
                    edge = sim[under_key] - imp
                    dec = _american_to_decimal(under_price)
                    ev = sim[under_key] * (dec - 1) - (1 - sim[under_key])
                    if edge > 0.02 and ev > 0:
                        bets.append({
                            "market": f"Under {tl}",
                            "pick": "UNDER",
                            "model_prob": sim[under_key],
                            "implied_prob": round(imp, 4),
                            "odds": under_price,
                            "edge": round(edge, 4),
                            "ev": round(ev, 4),
                        })

        # Puckline / spread
        sp_price = this_game.get("home_spread_price")
        sp_point = this_game.get("home_spread_point")
        if sp_price is not None and sp_point is not None:
            if sp_point == -1.5:
                model_p = sim["home_cover_minus_1_5"]
            else:
                model_p = sim["away_cover_plus_1_5"]
            imp = _american_to_prob(sp_price)
            edge = model_p - imp
            dec = _american_to_decimal(sp_price)
            ev = model_p * (dec - 1) - (1 - model_p)
            if edge > 0.02 and ev > 0:
                pick = f"{home_abbrev} {sp_point}" if sp_point < 0 else f"{home_abbrev} +{sp_point}"
                bets.append({
                    "market": "Puckline",
                    "pick": pick,
                    "model_prob": model_p,
                    "implied_prob": round(imp, 4),
                    "odds": sp_price,
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                })

        asp_price = this_game.get("away_spread_price")
        asp_point = this_game.get("away_spread_point")
        if asp_price is not None and asp_point is not None:
            if asp_point == 1.5:
                model_p = sim["away_cover_plus_1_5"]
            else:
                model_p = sim["home_cover_minus_1_5"]
            imp = _american_to_prob(asp_price)
            edge = model_p - imp
            dec = _american_to_decimal(asp_price)
            ev = model_p * (dec - 1) - (1 - model_p)
            if edge > 0.02 and ev > 0:
                pick = f"{away_abbrev} +{asp_point}" if asp_point > 0 else f"{away_abbrev} {asp_point}"
                bets.append({
                    "market": "Puckline",
                    "pick": pick,
                    "model_prob": model_p,
                    "implied_prob": round(imp, 4),
                    "odds": asp_price,
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                })

    return {
        "home_team": home_abbrev,
        "away_team": away_abbrev,
        "pred_total": round(pred_total, 2),
        "pred_home_goals": round(pred_home_goals, 2),
        "pred_away_goals": round(pred_away_goals, 2),
        "pred_home_win_prob": round(pred_home_wp, 4),
        "sim": sim,
        "total_line": total_line,
        "home_ml": this_game.get("home_ml") if this_game else None,
        "away_ml": this_game.get("away_ml") if this_game else None,
        "home_wp_implied": round(home_wp_implied, 4) if home_wp_implied else None,
        # Rolling context
        "home_gf_10": round(hr["gf_10"], 2),
        "home_ga_10": round(hr["ga_10"], 2),
        "away_gf_10": round(ar["gf_10"], 2),
        "away_ga_10": round(ar["ga_10"], 2),
        "home_xgf_10": round(mhr.get("xgf_10", 0), 2),
        "away_xgf_10": round(mar.get("xgf_10", 0), 2),
        # Bets
        "bets": sorted(bets, key=lambda b: b["edge"], reverse=True),
    }


def predict_todays_games():
    """Predict all of today's games."""
    if _totals_model is None or _win_model is None:
        return []

    today = date.today().isoformat()
    sched = nhl_api.get_schedule(today)
    if not sched:
        return []

    games = []
    for week in sched.get("gameWeek", []):
        if week.get("date") == today:
            games.extend(week.get("games", []))

    results = []
    for g in games:
        home = g.get("homeTeam", {}).get("abbrev", "")
        away = g.get("awayTeam", {}).get("abbrev", "")
        if not home or not away:
            continue
        pred = predict_game(home, away, today)
        if pred:
            results.append(pred)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    metrics = train_model()
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")

    print("\nToday's predictions:")
    for g in predict_todays_games():
        print(f"\n  {g['away_team']} @ {g['home_team']}")
        print(f"  Pred total: {g['pred_total']}, Line: {g['total_line']}")
        print(f"  Home win: {g['pred_home_win_prob']:.1%} (implied: {g['home_wp_implied']:.1%})" if g['home_wp_implied'] else f"  Home win: {g['pred_home_win_prob']:.1%}")
        for b in g["bets"]:
            print(f"    +EV: {b['market']} {b['pick']} @ {b['odds']:+d} "
                  f"(edge: {b['edge']:.1%}, model: {b['model_prob']:.1%})")
