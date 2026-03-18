"""
MLB Pitcher Strikeout Prediction — 3-Layer Architecture.

Layer 1 (BF):  Opportunity model — predicts batters faced
Layer 2 (KBF): Skill/matchup model — predicts K rate per BF
Layer 3:       Distribution & betting — Monte Carlo simulation, prop probs, fair odds

Final:  predicted_K = pred_BF × pred_K_per_BF
        + full distribution via simulation

Both XGBoost models use residual targets and fully temporal features
(no future data leakage).
"""

import json
import logging
import math
import unicodedata
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _normalize_name(name: str) -> str:
    """Strip accents and lowercase for fuzzy name matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

import mlb_api
import mlb_data_collector
import mlb_odds_collector

# ---------------------------------------------------------------------------
# Team name mapping (odds API full names ↔ MLB abbreviations)
# ---------------------------------------------------------------------------
TEAM_ABBREV_TO_FULL = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}
TEAM_FULL_TO_ABBREV = {v: k for k, v in TEAM_ABBREV_TO_FULL.items()}

MODEL_DIR = Path(__file__).parent / "saved_model_mlb"
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model_bf: XGBRegressor | None = None
_model_kbf: XGBRegressor | None = None
_model_metrics: dict = {}
_pitcher_var_ratio: dict[int, float] = {}
_pitcher_bf_std: dict[int, float] = {}
_pitcher_kbf_std: dict[int, float] = {}

# Pitch type groupings
TRACKED_PT = ["FF", "SL", "CH", "CU"]
BREAKING_PT = {"SL", "CU", "ST", "KC", "FS", "SV"}
FB_PT = {"FF", "SI", "FC"}

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------
BF_FEATURES = [
    "baseline_bf",
    "is_home",
    "avg_pitch_count",
    "pitches_last",
    "innings_last",
    "days_rest",
    "bf_trend",
    "season_bb_rate",
    "park_k_factor",
    "rolling_velocity",
    # Efficiency features
    "pitches_per_bf",
    "pitches_per_ip",
    "rolling_walk_rate",
    "rolling_whip",
    "opp_contact_rate",
    "rolling_3_pc",
    # Odds-derived features
    "implied_team_win_prob",
    "game_total_line",
    "team_moneyline",
    # Market K prop line
    "market_k_line",
    # Sharp book consensus
    "sharp_consensus_prob",
]

KBF_FEATURES = [
    "baseline_k_rate",
    "is_home",
    "opp_k_rate",
    "park_k_factor",
    # Market K prop line
    "market_k_line",
    # Statcast rolling
    "csw_rate",
    "whiff_rate",
    "zone_rate",
    "chase_rate",
    # Count-level
    "first_pitch_strike_rate",
    "two_strike_putaway_rate",
    "zone_contact_rate",
    # Pitcher profile
    "k_minus_bb_rate",
    "pitcher_cv",
    "days_rest",
    # Per-pitch-type usage
    "ff_usage", "sl_usage", "ch_usage", "cu_usage",
    # Per-pitch-type whiff
    "ff_whiff", "sl_whiff", "ch_whiff", "cu_whiff",
    # Deception
    "velo_gap",
    "pitch_entropy",
    "breaking_usage",
    # Matchup (arsenal-weighted)
    "matchup_whiff_rate",
    "matchup_k_rate",
    "matchup_contact_rate",
    # Pitch × lineup interactions
    "ff_x_lineup_whiff",
    "sl_x_lineup_whiff",
    "ch_x_lineup_whiff",
    "cu_x_lineup_whiff",
    # Opponent aggregate
    "opp_chase_rate",
    "opp_contact_rate",
    # TTO
    "tto_k_decay",
    # Velocity
    "rolling_velocity",
    # Sharp book consensus
    "sharp_consensus_prob",
]


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_model():
    MODEL_DIR.mkdir(exist_ok=True)
    if _model_bf is not None:
        _model_bf.save_model(str(MODEL_DIR / "model_bf.json"))
    if _model_kbf is not None:
        _model_kbf.save_model(str(MODEL_DIR / "model_kbf.json"))
    meta = {
        "metrics": _model_metrics,
        "pitcher_var_ratio": {str(k): v for k, v in _pitcher_var_ratio.items()},
        "pitcher_bf_std": {str(k): v for k, v in _pitcher_bf_std.items()},
        "pitcher_kbf_std": {str(k): v for k, v in _pitcher_kbf_std.items()},
        # Self-describing artifact metadata
        "artifact_info": {
            "bf_feature_names": BF_FEATURES,
            "kbf_feature_names": KBF_FEATURES,
            "bf_feature_count": len(BF_FEATURES),  # 20 with odds + sharp
            "kbf_feature_count": len(KBF_FEATURES),
            "bf_target": "batters_faced residual (actual - baseline)",
            "kbf_target": "K/BF residual (actual - season_k_rate)",
            "final_prediction": "pred_BF × pred_K_per_BF",
            "hyperparameters": {
                "n_estimators": 400, "max_depth": 4, "learning_rate": 0.04,
                "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 8,
                "reg_alpha": 1.0, "reg_lambda": 3.0,
            },
            "random_seed": RANDOM_SEED,
            "saved_at": datetime.now().isoformat(),
            "model_type": "Dual XGBoost (BF × K/BF) with Monte Carlo distribution layer",
            "filter": "is_starter = 1",
        },
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("MLB model saved to %s", MODEL_DIR)


def load_model() -> bool:
    global _model_bf, _model_kbf, _model_metrics, _pitcher_var_ratio
    global _pitcher_bf_std, _pitcher_kbf_std
    bf_path = MODEL_DIR / "model_bf.json"
    kbf_path = MODEL_DIR / "model_kbf.json"
    meta_path = MODEL_DIR / "meta.json"

    if not meta_path.exists():
        return False
    if not bf_path.exists() or not kbf_path.exists():
        return False

    try:
        _model_bf = XGBRegressor()
        _model_bf.load_model(str(bf_path))
        _model_kbf = XGBRegressor()
        _model_kbf.load_model(str(kbf_path))
        with open(meta_path) as f:
            meta = json.load(f)
        _model_metrics = meta.get("metrics", {})
        _pitcher_var_ratio = {int(k): v for k, v in meta.get("pitcher_var_ratio", {}).items()}
        _pitcher_bf_std = {int(k): v for k, v in meta.get("pitcher_bf_std", {}).items()}
        _pitcher_kbf_std = {int(k): v for k, v in meta.get("pitcher_kbf_std", {}).items()}
        logger.info("MLB dual model loaded (MAE: %s)", _model_metrics.get("mae"))
        return True
    except Exception as exc:
        logger.warning("Failed to load MLB model: %s", exc)
        _model_bf = _model_kbf = None
        return False


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _compute_baseline(season_avg, rolling_5, rolling_3):
    return 0.50 * season_avg + 0.30 * rolling_5 + 0.20 * rolling_3


def _safe_stat(val):
    """Replace inf/nan with 0."""
    if val is None:
        return 0.0
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _build_feature_dataframe() -> pd.DataFrame:
    """Build feature matrix for training. One row per pitcher-game (starts only)."""
    conn = mlb_data_collector.get_db()

    df = pd.read_sql_query(
        "SELECT * FROM mlb_pitcher_game_stats WHERE is_starter = 1 ORDER BY pitcher_id, date",
        conn,
    )
    if df.empty:
        conn.close()
        return pd.DataFrame()

    # Static lookups
    park_factors = {}
    for row in conn.execute("SELECT team_abbrev, k_factor FROM mlb_park_factors").fetchall():
        park_factors[row["team_abbrev"]] = row["k_factor"]

    # Statcast per-game (with velocity, zone, chase)
    statcast_df = pd.read_sql_query(
        "SELECT pitcher_id, game_date, csw_rate, whiff_rate, zone_rate, chase_rate, avg_velocity "
        "FROM mlb_statcast_pitcher", conn,
    )

    # Per-game granular tables
    game_arsenal_df = pd.read_sql_query(
        "SELECT pitcher_id, game_date, pitch_type, pitches, whiff_count, "
        "csw_count, chase_count, zone_count, swing_count, avg_velocity "
        "FROM mlb_pitcher_game_arsenal", conn,
    )
    game_bvp_df = pd.read_sql_query(
        "SELECT batter_id, game_date, team_abbrev, pitch_type, pitches_seen, "
        "whiff_count, swing_count, contact_count, chase_count, zone_count, "
        "pa_count, k_count FROM mlb_batter_game_pitch_type", conn,
    )
    game_tto_df = pd.read_sql_query(
        "SELECT pitcher_id, game_date, tto_number, pa_count, k_count "
        "FROM mlb_pitcher_game_tto", conn,
    )
    # Count-level per-game
    try:
        game_counts_df = pd.read_sql_query(
            "SELECT pitcher_id, game_date, total_pa, first_pitch_strikes, first_pitches, "
            "two_strike_pa, two_strike_k, zone_swings, zone_contacts "
            "FROM mlb_pitcher_game_counts", conn,
        )
    except Exception:
        game_counts_df = pd.DataFrame()

    # --- Load pitcher K prop lines (consensus across books) ---
    prop_line_map = {}  # normalized_name + game_date -> consensus line
    try:
        prop_df = pd.read_sql_query(
            """SELECT pitcher_name, game_date, AVG(line) as consensus_line,
                      COUNT(DISTINCT bookmaker) as num_books
               FROM mlb_pitcher_props
               WHERE over_under = 'Over'
               GROUP BY pitcher_name, game_date""",
            conn,
        )
        for _, r in prop_df.iterrows():
            key = (_normalize_name(r["pitcher_name"]), r["game_date"])
            prop_line_map[key] = float(r["consensus_line"])
        logger.info("Loaded %d pitcher-date prop lines", len(prop_line_map))
    except Exception as e:
        logger.warning("Could not load pitcher prop lines: %s", e)

    # --- Load odds data for feature integration ---
    odds_map = {}  # (game_date, team_abbrev) -> {ml, impl_prob, total}
    try:
        odds_df = pd.read_sql_query(
            """SELECT g.game_date, g.home_team, g.away_team,
                      o.market, o.outcome_name, o.outcome_price, o.outcome_point
               FROM mlb_game_odds o
               JOIN mlb_odds_events g ON o.event_id = g.event_id
               WHERE o.bookmaker = 'draftkings'""",
            conn,
        )
        for (gd, ht, at), grp in odds_df.groupby(["game_date", "home_team", "away_team"]):
            h_abbr = TEAM_FULL_TO_ABBREV.get(ht)
            a_abbr = TEAM_FULL_TO_ABBREV.get(at)
            if not h_abbr or not a_abbr:
                continue
            h_ml, a_ml, total = None, None, None
            for _, r in grp.iterrows():
                if r["market"] == "h2h":
                    if r["outcome_name"] == ht:
                        h_ml = int(r["outcome_price"])
                    elif r["outcome_name"] == at:
                        a_ml = int(r["outcome_price"])
                elif r["market"] == "totals" and r["outcome_point"] is not None:
                    total = float(r["outcome_point"])

            def _ml_to_prob(ml):
                if ml is None:
                    return 0.50
                if ml < 0:
                    return abs(ml) / (abs(ml) + 100)
                return 100 / (ml + 100)

            if h_ml is not None:
                odds_map[(gd, h_abbr)] = {
                    "ml": h_ml, "impl_prob": _ml_to_prob(h_ml),
                    "total": total or 8.5,
                }
            if a_ml is not None:
                odds_map[(gd, a_abbr)] = {
                    "ml": a_ml, "impl_prob": _ml_to_prob(a_ml),
                    "total": total or 8.5,
                }
        logger.info("Loaded odds for %d team-game entries", len(odds_map))
    except Exception as e:
        logger.warning("Could not load odds data: %s", e)

    # --- Load sharp book consensus probabilities ---
    sharp_map = {}
    try:
        sharp_map = mlb_odds_collector.load_sharp_consensus_bulk()
        logger.info("Loaded sharp consensus for %d pitcher-dates", len(sharp_map))
    except Exception as e:
        logger.warning("Could not load sharp consensus data: %s", e)

    conn.close()

    # --- Build lookup dicts ---
    statcast_map = {}
    if not statcast_df.empty:
        for (pid,), grp in statcast_df.groupby(["pitcher_id"]):
            statcast_map[pid] = grp.sort_values("game_date")

    pitcher_game_arsenal = defaultdict(list)
    for _, r in game_arsenal_df.iterrows():
        v = _safe_stat(r.get("avg_velocity"))
        pitcher_game_arsenal[(int(r["pitcher_id"]), r["game_date"])].append({
            "pitch_type": r["pitch_type"],
            "pitches": int(r["pitches"]),
            "whiff": int(r["whiff_count"]),
            "swing": int(r["swing_count"]),
            "avg_velocity": v,
        })

    pitcher_game_tto = defaultdict(list)
    for _, r in game_tto_df.iterrows():
        pitcher_game_tto[(int(r["pitcher_id"]), r["game_date"])].append({
            "tto_number": int(r["tto_number"]),
            "pa": int(r["pa_count"]),
            "k": int(r["k_count"]),
        })

    pitcher_game_counts = {}
    if not game_counts_df.empty:
        for _, r in game_counts_df.iterrows():
            pitcher_game_counts[(int(r["pitcher_id"]), r["game_date"])] = {
                "fps": int(r["first_pitches"]),
                "fps_k": int(r["first_pitch_strikes"]),
                "two_s": int(r["two_strike_pa"]),
                "two_s_k": int(r["two_strike_k"]),
                "z_sw": int(r["zone_swings"]),
                "z_ct": int(r["zone_contacts"]),
            }

    # --- Temporal opponent K rates ---
    _tok_dates, _tok_rates = {}, {}
    for team, tdf in df.groupby("opponent_abbrev"):
        daily = tdf.groupby("date").agg(k=("strikeouts", "sum"), bf=("batters_faced", "sum")).reset_index().sort_values("date")
        ck, cb = 0, 0
        dl, rl = [], []
        for _, dr in daily.iterrows():
            ck += dr["k"]; cb += dr["bf"]
            dl.append(dr["date"]); rl.append(ck / cb if cb > 0 else 0.22)
        _tok_dates[team] = dl; _tok_rates[team] = rl

    def _opp_k(team, before):
        d = _tok_dates.get(team)
        if not d: return 0.22
        i = bisect_left(d, before) - 1
        return _tok_rates[team][i] if i >= 0 else 0.22

    # --- Temporal team BVP per pitch type ---
    _bvp_d, _bvp_s = {}, {}
    if not game_bvp_df.empty:
        tdb = game_bvp_df.groupby(["team_abbrev", "pitch_type", "game_date"]).agg({
            "swing_count": "sum", "whiff_count": "sum", "contact_count": "sum",
            "pa_count": "sum", "k_count": "sum",
        }).reset_index()
        for (tm, pt), g in tdb.groupby(["team_abbrev", "pitch_type"]):
            g = g.sort_values("game_date")
            cs, cw, cc, cp, ck = 0, 0, 0, 0, 0
            dl, sl = [], []
            for _, r in g.iterrows():
                cs += int(r["swing_count"]); cw += int(r["whiff_count"])
                cc += int(r["contact_count"]); cp += int(r["pa_count"]); ck += int(r["k_count"])
                dl.append(r["game_date"])
                sl.append({"whiff": cw/cs if cs > 10 else 0, "contact": cc/cs if cs > 10 else 0, "k": ck/cp if cp > 5 else 0})
            _bvp_d[(tm, pt)] = dl; _bvp_s[(tm, pt)] = sl

    def _bvp(team, pt, before):
        k = (team, pt)
        d = _bvp_d.get(k)
        if not d: return None
        i = bisect_left(d, before) - 1
        return _bvp_s[k][i] if i >= 0 else None

    # --- Temporal team-level opponent chase & contact ---
    _och_d, _och_r, _oct_d, _oct_r = {}, {}, {}, {}
    if not game_bvp_df.empty:
        td2 = game_bvp_df.groupby(["team_abbrev", "game_date"]).agg({
            "pitches_seen": "sum", "swing_count": "sum", "whiff_count": "sum",
            "contact_count": "sum", "chase_count": "sum", "zone_count": "sum",
        }).reset_index()
        for tm, g in td2.groupby("team_abbrev"):
            g = g.sort_values("game_date")
            csw, cch, cct, cpit, cz = 0, 0, 0, 0, 0
            dc, rc, dt, rt = [], [], [], []
            for _, r in g.iterrows():
                csw += int(r["swing_count"]); cch += int(r["chase_count"])
                cct += int(r["contact_count"]); cpit += int(r["pitches_seen"]); cz += int(r["zone_count"])
                ooz = cpit - cz
                dc.append(r["game_date"]); rc.append(cch / ooz if ooz > 30 else 0.30)
                dt.append(r["game_date"]); rt.append(cct / csw if csw > 30 else 0.75)
            _och_d[tm] = dc; _och_r[tm] = rc; _oct_d[tm] = dt; _oct_r[tm] = rt

    def _opp_chase(tm, before):
        d = _och_d.get(tm)
        if not d: return 0.30
        i = bisect_left(d, before) - 1
        return _och_r[tm][i] if i >= 0 else 0.30

    def _opp_contact(tm, before):
        d = _oct_d.get(tm)
        if not d: return 0.75
        i = bisect_left(d, before) - 1
        return _oct_r[tm][i] if i >= 0 else 0.75

    # --- Pitcher variance ---
    global _pitcher_var_ratio
    pa = df.groupby("pitcher_id")["strikeouts"].agg(["mean", "std", "count", "var"])
    pa["var_ratio"] = np.where((pa["mean"] > 0) & (pa["count"] >= 5), pa["var"] / pa["mean"], 1.0)
    _pitcher_var_ratio = pa["var_ratio"].to_dict()

    default_csw, default_whiff = 0.29, 0.23

    # -----------------------------------------------------------------------
    # Build records
    # -----------------------------------------------------------------------
    records = []
    for pid, grp in df.groupby("pitcher_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Shifted rolling stats
        for w in [3, 5, 10]:
            grp[f"roll_{w}_k"] = grp["strikeouts"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"roll_{w}_bf"] = grp["batters_faced"].shift(1).rolling(w, min_periods=1).mean()
        grp["season_avg_k"] = grp["strikeouts"].expanding().mean().shift(1)
        grp["season_avg_bf"] = grp["batters_faced"].expanding().mean().shift(1)
        grp["avg_ip"] = grp["innings_pitched"].expanding().mean().shift(1)
        grp["roll_5_ip"] = grp["innings_pitched"].shift(1).rolling(5, min_periods=1).mean()
        grp["avg_pc"] = grp["pitches_thrown"].expanding().mean().shift(1)
        grp["roll_3_pc"] = grp["pitches_thrown"].shift(1).rolling(3, min_periods=1).mean()
        grp["pitches_last"] = grp["pitches_thrown"].shift(1)
        grp["innings_last"] = grp["innings_pitched"].shift(1)

        # BF trend (slope of last 5 BF)
        grp["bf_trend"] = grp["batters_faced"].shift(1).rolling(5, min_periods=3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0, raw=True
        )

        # Efficiency features
        cum_pc = grp["pitches_thrown"].expanding().sum().shift(1)
        cum_bf_eff = grp["batters_faced"].expanding().sum().shift(1)
        cum_ip_eff = grp["innings_pitched"].expanding().sum().shift(1)
        grp["pitches_per_bf"] = np.where(cum_bf_eff > 0, cum_pc / cum_bf_eff, 4.0)
        grp["pitches_per_ip"] = np.where(cum_ip_eff > 0, cum_pc / cum_ip_eff, 16.0)

        # Rolling walk rate (last 5)
        cum_bb_r5 = grp["walks"].shift(1).rolling(5, min_periods=1).sum()
        cum_bf_r5 = grp["batters_faced"].shift(1).rolling(5, min_periods=1).sum()
        grp["rolling_walk_rate"] = np.where(cum_bf_r5 > 0, cum_bb_r5 / cum_bf_r5, 0.08)

        # Rolling WHIP (last 5)
        cum_h_r5 = grp["hits_allowed"].shift(1).rolling(5, min_periods=1).sum()
        cum_ip_r5 = grp["innings_pitched"].shift(1).rolling(5, min_periods=1).sum()
        grp["rolling_whip"] = np.where(cum_ip_r5 > 0, (cum_bb_r5 + cum_h_r5) / cum_ip_r5, 1.30)

        # Cumulative rates
        cum_ip = grp["innings_pitched"].expanding().sum().shift(1)
        cum_k = grp["strikeouts"].expanding().sum().shift(1)
        cum_bf = grp["batters_faced"].expanding().sum().shift(1)
        cum_bb = grp["walks"].expanding().sum().shift(1)
        grp["season_k_per_9"] = np.where(cum_ip > 0, cum_k / cum_ip * 9, 0)
        grp["season_k_rate"] = np.where(cum_bf > 0, cum_k / cum_bf, 0)
        grp["season_bb_rate"] = np.where(cum_bf > 0, cum_bb / cum_bf, 0)
        grp["k_minus_bb_rate"] = grp["season_k_rate"] - grp["season_bb_rate"]

        # CV
        e_std = grp["strikeouts"].expanding().std().shift(1)
        e_mean = grp["strikeouts"].expanding().mean().shift(1)
        grp["pitcher_cv"] = np.where((e_mean > 0) & (grp.index >= 5), e_std / e_mean, 1.0)

        # Fill NaN
        for c in ["roll_3_k", "roll_5_k", "roll_10_k", "roll_3_bf", "roll_5_bf",
                   "season_avg_k", "season_avg_bf", "avg_ip", "roll_5_ip", "avg_pc",
                   "roll_3_pc", "pitches_last", "innings_last", "bf_trend",
                   "pitches_per_bf", "pitches_per_ip", "rolling_walk_rate", "rolling_whip",
                   "season_k_per_9", "season_k_rate", "season_bb_rate",
                   "k_minus_bb_rate", "pitcher_cv"]:
            grp[c] = grp[c].fillna(0)

        # Running accumulators
        ar_cum = defaultdict(lambda: {"p": 0, "wh": 0, "sw": 0, "vs": 0.0, "vp": 0})
        tto_cum = defaultdict(lambda: {"pa": 0, "k": 0})
        cnt_cum = {"fps": 0, "fps_k": 0, "two_s": 0, "two_s_k": 0, "z_sw": 0, "z_ct": 0}

        for idx, row in grp.iterrows():
            gd = row["date"]
            opp = row["opponent_abbrev"]
            home = int(row["is_home"])

            if idx > 0:
                # --- BF features ---
                baseline_bf = _compute_baseline(row["season_avg_bf"], row["roll_5_bf"], row["roll_3_bf"])
                bf_residual = row["batters_faced"] - baseline_bf

                # --- K/BF features ---
                actual_kbf = row["strikeouts"] / max(row["batters_faced"], 1)
                baseline_k_rate = row["season_k_rate"]
                kbf_residual = actual_kbf - baseline_k_rate

                opp_k = _opp_k(opp, gd)
                park_team = row["team_abbrev"] if home else opp
                park_k = park_factors.get(park_team, 1.0)
                rest = min(int(row["days_rest"]) if row["days_rest"] >= 0 else 4, 10)

                # Rolling Statcast (temporal)
                csw, whiff, zr, cr, rv = default_csw, default_whiff, 0.45, 0.30, 93.0
                if pid in statcast_map:
                    sc = statcast_map[pid]
                    prior = sc[sc["game_date"] < gd]
                    if not prior.empty:
                        rec = prior.tail(5)
                        _sc_cols = {"csw_rate": default_csw, "whiff_rate": default_whiff,
                                    "zone_rate": 0.45, "chase_rate": 0.30, "avg_velocity": 93.0}
                        _sc_vals = {}
                        for col, dfl in _sc_cols.items():
                            v = rec[col].replace([np.inf, -np.inf], np.nan).mean()
                            _sc_vals[col] = float(v) if pd.notna(v) else dfl
                        csw = _sc_vals["csw_rate"]
                        whiff = _sc_vals["whiff_rate"]
                        zr = _sc_vals["zone_rate"]
                        cr = _sc_vals["chase_rate"]
                        rv = _sc_vals["avg_velocity"]
                csw = _safe_stat(csw) or default_csw
                whiff = _safe_stat(whiff) or default_whiff
                zr = _safe_stat(zr) or 0.45
                cr = _safe_stat(cr) or 0.30
                rv = _safe_stat(rv) or 93.0

                # Count-level (temporal from accumulator)
                fps_rate = cnt_cum["fps_k"] / max(cnt_cum["fps"], 1) if cnt_cum["fps"] >= 20 else 0.60
                putaway = cnt_cum["two_s_k"] / max(cnt_cum["two_s"], 1) if cnt_cum["two_s"] >= 10 else 0.30
                z_contact = cnt_cum["z_ct"] / max(cnt_cum["z_sw"], 1) if cnt_cum["z_sw"] >= 10 else 0.80

                # Arsenal from accumulator
                total_p = sum(v["p"] for v in ar_cum.values())
                arsenal = []
                if total_p > 0:
                    for pt, c in ar_cum.items():
                        if c["p"] < 10:
                            continue
                        sw = max(c["sw"], 1)
                        velo = c["vs"] / c["vp"] if c["vp"] > 0 else 0
                        arsenal.append({
                            "pt": pt, "usage": c["p"] / total_p,
                            "whiff": c["wh"] / sw, "velo": velo,
                        })

                # Per-pitch-type features
                pt_usage, pt_whiff = {}, {}
                for a in arsenal:
                    pt_usage[a["pt"]] = a["usage"]
                    pt_whiff[a["pt"]] = a["whiff"]

                ff_u = pt_usage.get("FF", 0); sl_u = pt_usage.get("SL", 0)
                ch_u = pt_usage.get("CH", 0); cu_u = pt_usage.get("CU", 0)
                ff_w = pt_whiff.get("FF", 0); sl_w = pt_whiff.get("SL", 0)
                ch_w = pt_whiff.get("CH", 0); cu_w = pt_whiff.get("CU", 0)

                # Velocity gap (FB - offspeed)
                fb_v = [a["velo"] for a in arsenal if a["pt"] in FB_PT and a["velo"] > 0]
                os_v = [a["velo"] for a in arsenal if a["pt"] not in FB_PT and a["velo"] > 0]
                vg = (np.mean(fb_v) - np.mean(os_v)) if fb_v and os_v else 10.0

                # Pitch entropy
                usages = [a["usage"] for a in arsenal if a["usage"] > 0.02]
                entropy = -sum(u * math.log(u) for u in usages) if usages else 0

                # Breaking ball usage
                brk = sum(pt_usage.get(p, 0) for p in BREAKING_PT)

                # TTO decay
                tto_decay = 0.0
                if tto_cum[1]["pa"] >= 5 and tto_cum[3]["pa"] >= 3:
                    tto_decay = tto_cum[1]["k"] / tto_cum[1]["pa"] - tto_cum[3]["k"] / tto_cum[3]["pa"]

                # Arsenal-weighted matchup (temporal BVP)
                mw, mk, mc = 0.0, 0.0, 0.0
                if arsenal:
                    tu = sum(a["usage"] for a in arsenal)
                    for a in arsenal:
                        if a["usage"] < 0.02:
                            continue
                        pw = a["whiff"]
                        bs = _bvp(opp, a["pt"], gd)
                        bw = bs["whiff"] if bs and bs["whiff"] > 0 else pw
                        bk = bs["k"] if bs and bs["k"] > 0 else 0.22
                        bc = bs["contact"] if bs and bs["contact"] > 0 else (1.0 - pw)
                        mw += a["usage"] * (pw + bw) / 2
                        mk += a["usage"] * bk
                        mc += a["usage"] * bc
                    if tu > 0:
                        mw /= tu; mk /= tu; mc /= tu

                # Per-pitch × lineup interactions
                def _pt_lineup_whiff(pt_code, usage):
                    bs = _bvp(opp, pt_code, gd)
                    lw = bs["whiff"] if bs and bs["whiff"] > 0 else 0.20
                    return usage * lw

                ff_xl = _pt_lineup_whiff("FF", ff_u)
                sl_xl = _pt_lineup_whiff("SL", sl_u)
                ch_xl = _pt_lineup_whiff("CH", ch_u)
                cu_xl = _pt_lineup_whiff("CU", cu_u)

                # Opponent chase / contact
                o_chase = _opp_chase(opp, gd)
                o_contact = _opp_contact(opp, gd)

                # Prop line lookup (starter filter)
                pitcher_name_norm = _normalize_name(row["pitcher_name"])
                mk_line = prop_line_map.get((pitcher_name_norm, gd))
                if mk_line is None:
                    continue  # No prop line = not a confirmed starter, skip

                # Sharp consensus probability (NaN when not available — XGBoost handles natively)
                sharp_entry = sharp_map.get((gd, pitcher_name_norm))
                sharp_val = sharp_entry["sharp_prob_over"] if sharp_entry else float("nan")

                # Odds features
                team_abbr = row["team_abbrev"]
                odds_info = odds_map.get((gd, team_abbr), {})
                impl_prob = odds_info.get("impl_prob", 0.50)
                game_total = odds_info.get("total", 8.5)
                team_ml = odds_info.get("ml", 0)

                records.append({
                    "pitcher_id": row["pitcher_id"], "game_pk": row["game_pk"],
                    "date": gd, "strikeouts": row["strikeouts"],
                    "batters_faced": row["batters_faced"],
                    "actual_k_rate": actual_kbf,
                    # BF model
                    "baseline_bf": baseline_bf, "bf_residual": bf_residual,
                    "avg_pitch_count": row["avg_pc"], "pitches_last": row["pitches_last"],
                    "innings_last": row["innings_last"], "bf_trend": row["bf_trend"],
                    "pitches_per_bf": row["pitches_per_bf"],
                    "pitches_per_ip": row["pitches_per_ip"],
                    "rolling_walk_rate": row["rolling_walk_rate"],
                    "rolling_whip": row["rolling_whip"],
                    "rolling_3_pc": row["roll_3_pc"],
                    # Odds-derived
                    "implied_team_win_prob": impl_prob,
                    "game_total_line": game_total,
                    "team_moneyline": team_ml,
                    "market_k_line": mk_line,
                    "sharp_consensus_prob": sharp_val,
                    # KBF model
                    "baseline_k_rate": baseline_k_rate, "kbf_residual": kbf_residual,
                    "opp_k_rate": opp_k, "k_minus_bb_rate": row["k_minus_bb_rate"],
                    "csw_rate": csw, "whiff_rate": whiff,
                    "zone_rate": zr, "chase_rate": cr,
                    "first_pitch_strike_rate": fps_rate,
                    "two_strike_putaway_rate": putaway,
                    "zone_contact_rate": z_contact,
                    # Shared
                    "is_home": home, "park_k_factor": park_k, "days_rest": rest,
                    "season_bb_rate": row["season_bb_rate"], "pitcher_cv": row["pitcher_cv"],
                    "rolling_velocity": rv,
                    # Arsenal
                    "ff_usage": ff_u, "sl_usage": sl_u, "ch_usage": ch_u, "cu_usage": cu_u,
                    "ff_whiff": ff_w, "sl_whiff": sl_w, "ch_whiff": ch_w, "cu_whiff": cu_w,
                    "velo_gap": vg, "pitch_entropy": entropy, "breaking_usage": brk,
                    # Matchup
                    "matchup_whiff_rate": mw, "matchup_k_rate": mk, "matchup_contact_rate": mc,
                    "ff_x_lineup_whiff": ff_xl, "sl_x_lineup_whiff": sl_xl,
                    "ch_x_lineup_whiff": ch_xl, "cu_x_lineup_whiff": cu_xl,
                    "opp_chase_rate": o_chase, "opp_contact_rate": o_contact,
                    "tto_k_decay": tto_decay,
                    # Evaluation columns (not used as features, but needed for baselines)
                    "season_avg_k": row["season_avg_k"],
                    "roll_3_k": row["roll_3_k"],
                    "roll_5_k": row["roll_5_k"],
                    "pitch_hand": row.get("pitch_hand", "R"),
                })

            # Update accumulators
            for e in pitcher_game_arsenal.get((pid, gd), []):
                pt = e["pitch_type"]
                ar_cum[pt]["p"] += e["pitches"]; ar_cum[pt]["wh"] += e["whiff"]
                ar_cum[pt]["sw"] += e["swing"]
                if e["avg_velocity"] > 0:
                    ar_cum[pt]["vs"] += e["avg_velocity"] * e["pitches"]
                    ar_cum[pt]["vp"] += e["pitches"]

            for e in pitcher_game_tto.get((pid, gd), []):
                tto_cum[e["tto_number"]]["pa"] += e["pa"]
                tto_cum[e["tto_number"]]["k"] += e["k"]

            gc = pitcher_game_counts.get((pid, gd))
            if gc:
                for k in cnt_cum:
                    cnt_cum[k] += gc[k]

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(df["date"])
    days_ago = (dates.max() - dates).dt.days
    return (1.0 - 0.5 * (days_ago / max(days_ago.max(), 1))).values.astype(float)


def _train_xgb(X_train, y_train, X_test, y_test, weights, **kwargs):
    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=8,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0,
        **kwargs,
    )
    model.fit(X_train, y_train, sample_weight=weights,
              eval_set=[(X_test, y_test)], verbose=False)
    return model


def train_model() -> dict:
    """Train dual BF + K/BF models. Returns metrics dict."""
    global _model_bf, _model_kbf, _model_metrics
    global _pitcher_bf_std, _pitcher_kbf_std

    logger.info("Building MLB feature matrix...")
    df = _build_feature_dataframe()

    if df.empty or len(df) < 30:
        logger.warning("Not enough MLB data to train (%d rows)", len(df))
        _model_metrics = {"error": "Not enough data", "n_rows": len(df)}
        return _model_metrics

    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - timedelta(days=14)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff]

    if len(train_df) < 30 or len(test_df) < 10:
        _model_metrics = {"error": "Insufficient data for split"}
        return _model_metrics

    weights = _compute_sample_weights(train_df)

    # --- Model A: Batters Faced ---
    logger.info("Training BF model (%d features)...", len(BF_FEATURES))
    _model_bf = _train_xgb(
        train_df[BF_FEATURES].values, train_df["bf_residual"].values,
        test_df[BF_FEATURES].values, test_df["bf_residual"].values,
        weights,
    )

    # --- Model B: K per BF ---
    logger.info("Training K/BF model (%d features)...", len(KBF_FEATURES))
    _model_kbf = _train_xgb(
        train_df[KBF_FEATURES].values, train_df["kbf_residual"].values,
        test_df[KBF_FEATURES].values, test_df["kbf_residual"].values,
        weights,
    )

    # --- Pitcher-specific variance estimation ---
    logger.info("Estimating pitcher-specific variance...")
    # Compute residual std per pitcher from training data
    train_bf_resid = train_df["bf_residual"].values - _model_bf.predict(train_df[BF_FEATURES].values)
    train_kbf_resid = train_df["kbf_residual"].values - _model_kbf.predict(train_df[KBF_FEATURES].values)
    _train_var = train_df[["pitcher_id"]].copy()
    _train_var["bf_resid"] = train_bf_resid
    _train_var["kbf_resid"] = train_kbf_resid

    # Per-pitcher residual std (shrink toward global if few games)
    global_bf_std = float(np.std(train_bf_resid))
    global_kbf_std = float(np.std(train_kbf_resid))

    for pid, grp in _train_var.groupby("pitcher_id"):
        n = len(grp)
        if n >= 5:
            shrink = min(n / 15.0, 1.0)  # Full weight at 15+ games
            _pitcher_bf_std[pid] = shrink * float(grp["bf_resid"].std()) + (1 - shrink) * global_bf_std
            _pitcher_kbf_std[pid] = shrink * float(grp["kbf_resid"].std()) + (1 - shrink) * global_kbf_std
        else:
            _pitcher_bf_std[pid] = global_bf_std
            _pitcher_kbf_std[pid] = global_kbf_std

    # --- Evaluate combined prediction ---
    bf_pred = test_df["baseline_bf"].values + _model_bf.predict(test_df[BF_FEATURES].values)
    bf_pred = np.clip(bf_pred, 10, 40)
    kbf_pred = test_df["baseline_k_rate"].values + _model_kbf.predict(test_df[KBF_FEATURES].values)
    kbf_pred = np.clip(kbf_pred, 0.0, 0.60)

    pred_k = bf_pred * kbf_pred
    pred_k = np.maximum(pred_k, 0.0)
    y_actual = test_df["strikeouts"].values

    mae = mean_absolute_error(y_actual, pred_k)
    rmse = np.sqrt(mean_squared_error(y_actual, pred_k))
    mean_bias = float(np.mean(pred_k - y_actual))

    # Baseline comparisons on test set
    baseline_season = test_df["season_avg_k"].values
    baseline_r5 = test_df["roll_5_k"].values
    baseline_r3 = test_df["roll_3_k"].values
    baseline_weighted = test_df["baseline_bf"].values * test_df["baseline_k_rate"].values
    baseline_opp = 25.0 * test_df["opp_k_rate"].values

    baselines = {
        "season_avg": round(mean_absolute_error(y_actual, baseline_season), 3),
        "rolling_5": round(mean_absolute_error(y_actual, baseline_r5), 3),
        "rolling_3": round(mean_absolute_error(y_actual, baseline_r3), 3),
        "weighted_baseline": round(mean_absolute_error(y_actual, baseline_weighted), 3),
        "opp_k_only": round(mean_absolute_error(y_actual, baseline_opp), 3),
        "model": round(mae, 3),
    }

    # Calibration
    calibration = {}
    te = pd.DataFrame({"actual": y_actual, "predicted": pred_k,
                        "baseline": baseline_weighted})
    for lo, hi, lbl in [(0, 4, "<4"), (4, 6, "4-6"), (6, 8, "6-8"), (8, 99, "8+")]:
        m = (te["baseline"] >= lo) & (te["baseline"] < hi)
        b = te[m]
        if len(b) > 0:
            calibration[lbl] = {"n": len(b),
                                "avg_pred": round(b["predicted"].mean(), 2),
                                "avg_actual": round(b["actual"].mean(), 2)}

    # Feature importance (K/BF model)
    imp = _model_kbf.feature_importances_
    feat_imp = {c: round(float(v), 4) for c, v in sorted(
        zip(KBF_FEATURES, imp), key=lambda x: x[1], reverse=True)[:12]}

    # BF feature importance
    bf_imp = _model_bf.feature_importances_
    bf_feat_imp = {c: round(float(v), 4) for c, v in sorted(
        zip(BF_FEATURES, bf_imp), key=lambda x: x[1], reverse=True)[:10]}

    # Threshold accuracy
    thresh_acc = {}
    for t in [3.5, 4.5, 5.5, 6.5, 7.5]:
        correct = ((pred_k >= t) == (y_actual >= t)).sum()
        thresh_acc[f"over_{t:.1f}"] = round(correct / len(pred_k), 3)

    # Segmented MAE by pitcher tier
    seg_mae = {}
    for lo, hi, lbl in [(0, 4, "low_k"), (4, 6, "mid_k"), (6, 8, "high_k"), (8, 99, "elite_k")]:
        m = (baseline_weighted >= lo) & (baseline_weighted < hi)
        if m.sum() > 0:
            seg_mae[lbl] = {"n": int(m.sum()),
                            "mae": round(mean_absolute_error(y_actual[m], pred_k[m]), 3),
                            "bias": round(float(np.mean(pred_k[m] - y_actual[m])), 3)}

    # Tail diagnostics
    tail_diag = {}
    for lo, hi, lbl in [(0, 3, "<3"), (3, 5, "3-5"), (5, 7, "5-7"), (7, 9, "7-9"), (9, 99, "9+")]:
        m = (pred_k >= lo) & (pred_k < hi)
        if m.sum() > 0:
            tail_diag[lbl] = {"n": int(m.sum()),
                              "avg_pred": round(float(pred_k[m].mean()), 2),
                              "avg_actual": round(float(y_actual[m].mean()), 2),
                              "mae": round(mean_absolute_error(y_actual[m], pred_k[m]), 3),
                              "bias": round(float(np.mean(pred_k[m] - y_actual[m])), 3)}

    # --- Market backtest on holdout ---
    market_backtest = {}
    if "market_k_line" in test_df.columns:
        mkt_lines = test_df["market_k_line"].values
        mkt_mae = mean_absolute_error(y_actual, mkt_lines)
        mkt_rmse = np.sqrt(mean_squared_error(y_actual, mkt_lines))
        mkt_bias = float(np.mean(mkt_lines - y_actual))

        # Model vs market: how often does model beat market?
        model_closer = (np.abs(pred_k - y_actual) < np.abs(mkt_lines - y_actual)).sum()
        market_closer = (np.abs(mkt_lines - y_actual) < np.abs(pred_k - y_actual)).sum()
        ties = len(y_actual) - model_closer - market_closer

        # Over/under accuracy vs market line
        model_over_correct = ((pred_k > mkt_lines) & (y_actual > mkt_lines)).sum()
        model_under_correct = ((pred_k <= mkt_lines) & (y_actual <= mkt_lines)).sum()
        model_ou_acc = (model_over_correct + model_under_correct) / len(y_actual)

        # Threshold accuracy using market line as the threshold
        for_lines = {}
        for line_val in [3.5, 4.5, 5.5, 6.5, 7.5]:
            mask = np.abs(mkt_lines - line_val) < 0.01
            if mask.sum() > 5:
                model_correct = ((pred_k >= line_val) == (y_actual >= line_val))[mask].sum()
                market_base = ((mkt_lines >= line_val) == (y_actual >= line_val))[mask].sum()
                for_lines[f"{line_val}"] = {
                    "n": int(mask.sum()),
                    "model_acc": round(model_correct / mask.sum(), 3),
                }

        market_backtest = {
            "market_mae": round(mkt_mae, 3),
            "market_rmse": round(mkt_rmse, 3),
            "market_bias": round(mkt_bias, 3),
            "model_mae": round(mae, 3),
            "model_rmse": round(rmse, 3),
            "mae_edge_vs_market": round(mkt_mae - mae, 3),
            "model_closer_count": int(model_closer),
            "market_closer_count": int(market_closer),
            "ties": int(ties),
            "model_vs_market_win_rate": round(model_closer / max(model_closer + market_closer, 1), 3),
            "model_ou_accuracy_vs_market_line": round(model_ou_acc, 3),
            "per_line_accuracy": for_lines,
        }
        logger.info("Market backtest: Model MAE %.3f vs Market MAE %.3f (edge: %.3f)",
                     mae, mkt_mae, mkt_mae - mae)

    _model_metrics = {
        "mae": round(mae, 3), "rmse": round(rmse, 3),
        "mean_bias": round(mean_bias, 3),
        "train_samples": len(train_df), "test_samples": len(test_df),
        "holdout_period": f"{cutoff:%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
        "train_date_range": f"{df['date'].min():%Y-%m-%d} to {cutoff:%Y-%m-%d}",
        "model_type": "3-Layer: Dual XGBoost (BF × K/BF) + Monte Carlo",
        "bf_features": len(BF_FEATURES), "kbf_features": len(KBF_FEATURES),
        "baselines": baselines,
        "calibration": calibration,
        "threshold_accuracy": thresh_acc,
        "top_kbf_features": feat_imp,
        "top_bf_features": bf_feat_imp,
        "segments": seg_mae,
        "tail_diagnostics": tail_diag,
        "variance_info": {
            "global_bf_residual_std": round(global_bf_std, 3),
            "global_kbf_residual_std": round(global_kbf_std, 4),
            "pitchers_with_variance": len(_pitcher_bf_std),
        },
        "market_backtest": market_backtest,
    }

    save_model()
    logger.info("MLB dual model trained: MAE=%.3f RMSE=%.3f (%d/%d)",
                mae, rmse, len(train_df), len(test_df))
    return _model_metrics


def get_model_metrics() -> dict:
    return _model_metrics


# ---------------------------------------------------------------------------
# Prediction (live)
# ---------------------------------------------------------------------------
def predict_pitcher(pitcher_id: int, opponent_abbrev: str, is_home: bool,
                    market_line: float | None = None,
                    market_over_odds: float | None = None,
                    market_under_odds: float | None = None,
                    team_moneyline: int | None = None,
                    game_total: float | None = None) -> dict | None:
    """Predict strikeouts with full distribution output.

    Returns dict with:
    - Point predictions (mean K, median K, pred_bf, pred_k_rate)
    - Prop probabilities (P_over_3.5 through P_over_7.5)
    - Fair odds (American)
    - Market edge (if odds supplied)
    - Supporting stats for interpretability
    """
    if _model_bf is None or _model_kbf is None:
        return None

    stats = mlb_data_collector.get_pitcher_rolling_stats(pitcher_id)
    if not stats or stats.get("games", 0) < 3:
        return None

    conn = mlb_data_collector.get_db()
    row = conn.execute(
        "SELECT pitcher_name, team_abbrev, pitch_hand FROM mlb_pitcher_game_stats "
        "WHERE pitcher_id = ? ORDER BY date DESC LIMIT 1", (pitcher_id,),
    ).fetchone()
    if not row:
        conn.close()
        return None
    pitcher_name, team, pitch_hand = row["pitcher_name"], row["team_abbrev"], row["pitch_hand"]

    opp_row = conn.execute("SELECT k_rate FROM mlb_team_batting WHERE team_abbrev = ?", (opponent_abbrev,)).fetchone()
    opp_k_rate = opp_row["k_rate"] if opp_row else 0.22

    park_team = team if is_home else opponent_abbrev
    park_row = conn.execute("SELECT k_factor FROM mlb_park_factors WHERE team_abbrev = ?", (park_team,)).fetchone()
    park_k = park_row["k_factor"] if park_row else 1.0

    # Team-level BVP per pitch type for interactions
    bvp_rows = conn.execute(
        "SELECT pitch_type, whiff_rate FROM mlb_batter_vs_pitch bvp "
        "JOIN mlb_batter_stats bs ON bvp.batter_id = bs.batter_id "
        "WHERE bs.team_abbrev = ? AND bvp.pitches_seen >= 10", (opponent_abbrev,),
    ).fetchall()
    team_pt_whiff = defaultdict(list)
    for r in bvp_rows:
        team_pt_whiff[r["pitch_type"]].append(r["whiff_rate"] or 0)
    lineup_pt_whiff = {pt: np.mean(v) for pt, v in team_pt_whiff.items()}

    conn.close()

    sc = mlb_data_collector.get_statcast_for_pitcher(pitcher_id)
    csw = sc.get("csw_rate", 0.29)
    whiff = sc.get("whiff_rate", 0.23)
    zr = sc.get("zone_rate", 0.45)
    cr_val = sc.get("chase_rate", 0.30)
    rv = sc.get("avg_velocity", 93.0)
    fps_rate = sc.get("first_pitch_strike_rate", 0.60)
    putaway = sc.get("two_strike_putaway_rate", 0.30)
    z_contact = sc.get("zone_contact_rate", 0.80)

    matchup = mlb_data_collector.compute_arsenal_matchup(pitcher_id, opponent_abbrev)
    arsenal_list = mlb_data_collector.get_pitcher_arsenal(pitcher_id)

    rest = stats.get("last_rest", 4)
    if rest < 0: rest = 4
    rest = min(rest, 10)

    cv = stats["std_k"] / max(stats["season_avg_k"], 0.1) if stats["std_k"] > 0 and stats["games"] >= 5 else 1.0

    # Per-pitch-type from season arsenal
    pt_map = {}
    for p in arsenal_list:
        pt_map[p["pitch_type"]] = {"u": p.get("usage_pct") or 0, "w": p.get("whiff_rate") or 0, "v": p.get("avg_velocity") or 0}

    ff_u = pt_map.get("FF", {}).get("u", 0); sl_u = pt_map.get("SL", {}).get("u", 0)
    ch_u = pt_map.get("CH", {}).get("u", 0); cu_u = pt_map.get("CU", {}).get("u", 0)
    ff_w = pt_map.get("FF", {}).get("w", 0); sl_w = pt_map.get("SL", {}).get("w", 0)
    ch_w = pt_map.get("CH", {}).get("w", 0); cu_w = pt_map.get("CU", {}).get("w", 0)

    fb_v = [pt_map[p]["v"] for p in ["FF", "SI", "FC"] if p in pt_map and pt_map[p]["v"] > 0]
    os_v = [pt_map[p]["v"] for p in ["SL", "CU", "CH", "FS", "ST"] if p in pt_map and pt_map[p]["v"] > 0]
    vg = (np.mean(fb_v) - np.mean(os_v)) if fb_v and os_v else 10.0

    usages = [pt_map[p]["u"] for p in pt_map if pt_map[p]["u"] > 0.02]
    entropy = -sum(u * math.log(u) for u in usages) if usages else 0

    brk = sum(pt_map.get(p, {}).get("u", 0) for p in BREAKING_PT)
    k_minus_bb = stats["k_rate"] - stats["bb_rate"]

    # BF baselines
    bf3 = stats.get("rolling_3_bf", stats.get("season_avg_bf", 25))
    bf5 = stats.get("rolling_5_bf", stats.get("season_avg_bf", 25))
    baseline_bf = _compute_baseline(stats.get("season_avg_bf", 25), bf5, bf3)

    bf_trend = 0
    baseline_k_rate = stats["k_rate"]

    # Efficiency features for BF model
    ppbf = stats.get("season_avg_pc", 90) / max(stats.get("season_avg_bf", 25), 1)
    ppip = stats.get("season_avg_pc", 90) / max(stats.get("season_avg_ip", 5.5), 0.1)

    # Opponent chase/contact from BVP
    opp_chase = 0.30
    opp_contact_val = 0.75
    all_bvp_whiffs = [v for vals in team_pt_whiff.values() for v in vals]
    if all_bvp_whiffs:
        opp_contact_val = 1.0 - np.mean(all_bvp_whiffs)

    # Compute implied win prob from moneyline if provided
    def _ml_to_prob(ml):
        if ml is None:
            return 0.50
        if ml < 0:
            return abs(ml) / (abs(ml) + 100)
        return 100 / (ml + 100)

    live_impl_prob = _ml_to_prob(team_moneyline) if team_moneyline is not None else 0.50
    live_total = game_total if game_total is not None else 8.5
    live_ml = team_moneyline if team_moneyline is not None else 0

    # Sharp consensus probability (from live odds)
    sharp_prob_val = float("nan")
    consensus_data = None
    if market_line is not None:
        # consensus_data will be set by predict_todays_games if available
        pass

    # BF features (21 — includes odds-derived + market K line + sharp consensus)
    bf_feats = np.array([[
        baseline_bf, 1 if is_home else 0, stats["season_avg_pc"],
        stats.get("pitches_last", stats["season_avg_pc"]),
        stats.get("innings_last", stats["season_avg_ip"]),
        rest, bf_trend, stats["bb_rate"], park_k, rv,
        ppbf, ppip,
        stats["bb_rate"],  # rolling_walk_rate approx
        1.30,  # rolling_whip default
        opp_contact_val,
        stats.get("season_avg_pc", 90),  # rolling_3_pc approx
        live_impl_prob,
        live_total,
        live_ml,
        market_line if market_line is not None else stats.get("season_avg_k", 5.0),
        sharp_prob_val,
    ]])

    # Pitch × lineup interaction
    def _lw(pt_code, usage):
        return usage * lineup_pt_whiff.get(pt_code, 0.20)

    kbf_feats = np.array([[
        baseline_k_rate, 1 if is_home else 0, opp_k_rate, park_k,
        market_line if market_line is not None else stats.get("season_avg_k", 5.0),
        csw, whiff, zr, cr_val,
        fps_rate, putaway, z_contact,
        k_minus_bb, cv, rest,
        ff_u, sl_u, ch_u, cu_u, ff_w, sl_w, ch_w, cu_w,
        vg, entropy, brk,
        matchup.get("matchup_whiff_rate", 0), matchup.get("matchup_k_rate", 0),
        matchup.get("matchup_contact_rate", 0),
        _lw("FF", ff_u), _lw("SL", sl_u), _lw("CH", ch_u), _lw("CU", cu_u),
        opp_chase, opp_contact_val,
        matchup.get("tto_k_decay", 0),
        rv,
        sharp_prob_val,
    ]])

    # Predict point estimates
    pred_bf = max(baseline_bf + float(_model_bf.predict(bf_feats)[0]), 10)
    pred_kbf = max(baseline_k_rate + float(_model_kbf.predict(kbf_feats)[0]), 0)
    pred_kbf = min(pred_kbf, 0.55)
    pred_k = max(0.0, round(pred_bf * pred_kbf, 2))

    baseline_k = _compute_baseline(stats["season_avg_k"], stats["rolling_5_k"], stats["rolling_3_k"])

    # --- Layer 3: Monte Carlo Simulation ---
    bf_std = _pitcher_bf_std.get(pitcher_id, pred_bf * 0.18)
    kbf_std = _pitcher_kbf_std.get(pitcher_id, pred_kbf * 0.20)

    try:
        import mlb_simulation
        sim = mlb_simulation.simulate_with_market(
            pred_bf, pred_kbf,
            bf_std=bf_std, kbf_std=kbf_std,
            market_line=market_line,
            market_over_odds=market_over_odds,
            market_under_odds=market_under_odds,
            n_sims=10000, seed=RANDOM_SEED,
        )
    except ImportError:
        sim = {}

    result = {
        # Identity
        "pitcher_id": pitcher_id, "pitcher_name": pitcher_name,
        "team": team, "opponent": opponent_abbrev,
        "pitch_hand": pitch_hand, "is_home": is_home,
        # Layer 1: Opportunity
        "projected_BF": round(pred_bf, 1),
        # Layer 2: Skill/Matchup
        "projected_K_per_BF": round(pred_kbf, 4),
        # Combined point estimate
        "projected_mean_K": pred_k,
        "baseline_k": round(baseline_k, 2),
        # Layer 3: Distribution (from simulation)
        "projected_median_K": sim.get("median_k", round(pred_k)),
        "projected_std_K": sim.get("std_k", 0),
        # Prop probabilities
        "P_over_3_5": sim.get("P_over_3.5", 0),
        "P_over_4_5": sim.get("P_over_4.5", 0),
        "P_over_5_5": sim.get("P_over_5.5", 0),
        "P_over_6_5": sim.get("P_over_6.5", 0),
        "P_over_7_5": sim.get("P_over_7.5", 0),
        # Fair odds
        "fair_odds_over_3_5": sim.get("fair_over_3.5", 0),
        "fair_odds_over_4_5": sim.get("fair_over_4.5", 0),
        "fair_odds_over_5_5": sim.get("fair_over_5.5", 0),
        "fair_odds_over_6_5": sim.get("fair_over_6.5", 0),
        "fair_odds_over_7_5": sim.get("fair_over_7.5", 0),
        # Supporting stats
        "season_avg_k": stats["season_avg_k"],
        "rolling_3_k": stats["rolling_3_k"], "rolling_5_k": stats["rolling_5_k"],
        "k_per_9": stats["k_per_9"], "avg_ip": stats["season_avg_ip"],
        "opp_k_rate": round(opp_k_rate, 4), "park_k_factor": park_k,
        "csw_rate": round(csw, 4), "whiff_rate": round(whiff, 4),
        "days_rest": rest, "games": stats["games"],
        "var_ratio": round(_pitcher_var_ratio.get(pitcher_id, 1.0), 3),
        "matchup_whiff": round(matchup.get("matchup_whiff_rate", 0), 4),
        "matchup_k": round(matchup.get("matchup_k_rate", 0), 4),
        "arsenal_diversity": matchup.get("arsenal_diversity", 0),
        "tto_k_decay": round(matchup.get("tto_k_decay", 0), 4),
    }

    # Market edge (from simulation)
    if market_line is not None:
        result["market_line"] = market_line
        result["model_prob_over"] = sim.get("model_prob_over", 0)
        result["model_prob_under"] = sim.get("model_prob_under", 0)
    if market_over_odds is not None:
        result["market_over_odds"] = market_over_odds
        result["market_implied_prob_over"] = sim.get("market_implied_prob_over", 0)
        result["edge_over"] = sim.get("edge_over", 0)
        result["ev_over"] = sim.get("ev_over", 0)
    if market_under_odds is not None:
        result["market_under_odds"] = market_under_odds
        result["market_implied_prob_under"] = sim.get("market_implied_prob_under", 0)
        result["edge_under"] = sim.get("edge_under", 0)
        result["ev_under"] = sim.get("ev_under", 0)

    # Backward compat aliases
    result["predicted_k"] = pred_k
    result["pred_bf"] = result["projected_BF"]
    result["pred_k_rate"] = result["projected_K_per_BF"]

    # Sharp consensus + soft book data (set by predict_todays_games)
    result["sharp_prob_over"] = None
    result["sharp_prob_under"] = None
    result["n_sharp_books"] = 0
    result["betmgm_over"] = None
    result["betmgm_under"] = None

    return result


def predict_todays_games() -> list[dict]:
    if _model_bf is None or _model_kbf is None:
        return []

    games = mlb_api.get_todays_schedule()
    if not games:
        return []

    predictions = []
    for game in games:
        if game["status"] == "Final":
            continue
        home = game["home_team_abbrev"]
        away = game["away_team_abbrev"]
        for side, opp, ih in [("home", away, True), ("away", home, False)]:
            pp = game.get(f"{side}_probable_pitcher")
            if not pp or not pp.get("id"):
                continue

            # Get consensus line with sharp data
            pitcher_name_full = pp.get("name", "")
            consensus = None
            market_line_val = None
            market_over = None
            market_under = None
            if pitcher_name_full:
                try:
                    consensus = mlb_odds_collector.get_consensus_line(
                        pitcher_name_full,
                        date.today().isoformat(),
                    )
                    if consensus:
                        market_line_val = consensus["line"]
                        market_over = consensus.get("over_odds")
                        market_under = consensus.get("under_odds")
                except Exception:
                    pass

            pred = predict_pitcher(
                pp["id"], opp, ih,
                market_line=market_line_val,
                market_over_odds=market_over,
                market_under_odds=market_under,
            )
            if pred:
                pred["game_pk"] = game["game_pk"]
                pred["game_display"] = f"{away} @ {home}"

                # Attach sharp consensus + BetMGM data
                if consensus:
                    pred["sharp_prob_over"] = consensus.get("sharp_prob_over")
                    pred["sharp_prob_under"] = consensus.get("sharp_prob_under")
                    pred["n_sharp_books"] = consensus.get("n_sharp_books", 0)
                    pred["betmgm_over"] = consensus.get("betmgm_over")
                    pred["betmgm_under"] = consensus.get("betmgm_under")
                    pred["market_line"] = consensus["line"]

                predictions.append(pred)

    predictions.sort(key=lambda x: x["predicted_k"], reverse=True)
    return predictions
