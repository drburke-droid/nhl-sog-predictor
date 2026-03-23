"""
NHL Team Clustering — team archetypes and matchup interactions.

Clusters teams on play style (offense, defense, pace, special teams,
goaltending) then computes how each archetype performs against other
archetypes. These become features for the game-level model.
"""

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import data_collector

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "nhl_data.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def build_team_profiles(before_date=None, window=30):
    """Build team style profiles from rolling data.

    Features per team:
    - Offense: goals/game, xGF, SOG/game, high-danger chances, PP rate
    - Defense: GA/game, xGA, SOG against, high-danger against, PK
    - Pace: total shot attempts per game (both teams)
    - Goaltending: save % proxy (1 - GA/SA)
    - Style: 5v5 xG%, ratio of HD shots to total, PP dependency

    Returns DataFrame with one row per team.
    """
    conn = get_db()
    mp_conn = get_db()

    if before_date is None:
        from datetime import date
        before_date = date.today().isoformat()

    # Date window
    cutoff = (pd.Timestamp(before_date) - pd.Timedelta(days=window * 2)).strftime("%Y-%m-%d")

    # --- Goals for/against from games table ---
    games = pd.read_sql_query(f"""
        SELECT date, home_team, away_team, home_score, away_score
        FROM games WHERE status = 'FINAL' AND date >= '{cutoff}' AND date < '{before_date}'
    """, conn)

    if games.empty:
        conn.close()
        return pd.DataFrame()

    team_stats = defaultdict(lambda: {
        "gf": [], "ga": [], "sog_for": [], "sog_against": [],
        "pp_goals": [], "games": 0,
    })

    # SOG per team per game
    sog_df = pd.read_sql_query(f"""
        SELECT g.game_id, g.date, pgs.team, SUM(pgs.shots) as team_sog,
               SUM(pgs.pp_goals) as team_pp
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        WHERE g.date >= '{cutoff}' AND g.date < '{before_date}'
        AND pgs.position IN ('L','C','R','D')
        GROUP BY g.game_id, pgs.team
    """, conn)
    sog_map = {}
    pp_map = {}
    for _, r in sog_df.iterrows():
        sog_map[(r["game_id"], r["team"])] = r["team_sog"]
        pp_map[(r["game_id"], r["team"])] = r["team_pp"]

    for _, g in games.iterrows():
        home, away = g["home_team"], g["away_team"]
        hs, as_ = g["home_score"] or 0, g["away_score"] or 0
        gid = None
        # Find game_id
        gid_row = conn.execute(
            "SELECT game_id FROM games WHERE date = ? AND home_team = ? LIMIT 1",
            (g["date"], home),
        ).fetchone()
        gid = gid_row["game_id"] if gid_row else None

        home_sog = sog_map.get((gid, home), 30) if gid else 30
        away_sog = sog_map.get((gid, away), 30) if gid else 30
        home_pp = pp_map.get((gid, home), 0) if gid else 0
        away_pp = pp_map.get((gid, away), 0) if gid else 0

        team_stats[home]["gf"].append(hs)
        team_stats[home]["ga"].append(as_)
        team_stats[home]["sog_for"].append(home_sog)
        team_stats[home]["sog_against"].append(away_sog)
        team_stats[home]["pp_goals"].append(home_pp)
        team_stats[home]["games"] += 1

        team_stats[away]["gf"].append(as_)
        team_stats[away]["ga"].append(hs)
        team_stats[away]["sog_for"].append(away_sog)
        team_stats[away]["sog_against"].append(home_sog)
        team_stats[away]["pp_goals"].append(away_pp)
        team_stats[away]["games"] += 1

    # --- MoneyPuck advanced stats ---
    mp_df = pd.read_sql_query(f"""
        SELECT team, game_id,
               AVG(xgoals) as avg_xg,
               AVG(high_danger_shots) as avg_hd,
               AVG(on_ice_xg_for) / COUNT(DISTINCT player_id) * 18 as team_xgf,
               AVG(on_ice_xg_against) / COUNT(DISTINCT player_id) * 18 as team_xga,
               AVG(on_ice_hd_shots_for) / COUNT(DISTINCT player_id) * 18 as team_hdcf,
               AVG(on_ice_hd_shots_against) / COUNT(DISTINCT player_id) * 18 as team_hdca,
               AVG(shot_attempts) as avg_att
        FROM mp_player_game
        WHERE situation = 'all' AND game_date >= '{cutoff}' AND game_date < '{before_date}'
        GROUP BY team, game_id
    """, mp_conn)

    mp_team = {}
    if not mp_df.empty:
        for team, grp in mp_df.groupby("team"):
            mp_team[team] = {
                "xgf": grp["team_xgf"].mean(),
                "xga": grp["team_xga"].mean(),
                "hdcf": grp["team_hdcf"].mean(),
                "hdca": grp["team_hdca"].mean(),
                "pace": grp["avg_att"].mean(),
            }

    # --- 5v5 xG% ---
    fivon = pd.read_sql_query(f"""
        SELECT team,
               AVG(on_ice_xg_for) / COUNT(DISTINCT player_id) * 18 as xgf_5v5,
               AVG(on_ice_xg_against) / COUNT(DISTINCT player_id) * 18 as xga_5v5
        FROM mp_player_game
        WHERE situation = '5on5' AND game_date >= '{cutoff}' AND game_date < '{before_date}'
        GROUP BY team
    """, mp_conn)
    fivon_map = {}
    for _, r in fivon.iterrows():
        xgf = r["xgf_5v5"] or 1
        xga = r["xga_5v5"] or 1
        fivon_map[r["team"]] = xgf / (xgf + xga)

    conn.close()

    # --- Build feature DataFrame ---
    records = []
    for team, s in team_stats.items():
        if s["games"] < 10:
            continue
        n = s["games"]
        mp = mp_team.get(team, {})

        gf = np.mean(s["gf"])
        ga = np.mean(s["ga"])
        sog_f = np.mean(s["sog_for"])
        sog_a = np.mean(s["sog_against"])
        pp_pg = np.mean(s["pp_goals"])

        # Goaltending: save % proxy = 1 - (GA / SOG_against)
        save_pct = 1 - (ga / max(sog_a, 1)) if sog_a > 0 else 0.90

        records.append({
            "team": team,
            # Offense
            "gf_per_game": round(gf, 2),
            "xgf": round(mp.get("xgf", gf), 2),
            "sog_per_game": round(sog_f, 1),
            "hd_chances_for": round(mp.get("hdcf", 0), 2),
            "pp_goals_per_game": round(pp_pg, 2),
            # Defense
            "ga_per_game": round(ga, 2),
            "xga": round(mp.get("xga", ga), 2),
            "sog_against": round(sog_a, 1),
            "hd_chances_against": round(mp.get("hdca", 0), 2),
            # Goaltending
            "save_pct": round(save_pct, 4),
            # Pace / Style
            "pace": round(mp.get("pace", sog_f + sog_a), 1),
            "fivon5_xg_pct": round(fivon_map.get(team, 0.50), 4),
            "hd_ratio": round(mp.get("hdcf", 1) / max(mp.get("hdcf", 1) + mp.get("hdca", 1), 0.1), 3),
            "pp_dependency": round(pp_pg / max(gf, 0.1), 3),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = [
    "gf_per_game", "xgf", "hd_chances_for", "pp_goals_per_game",
    "ga_per_game", "xga", "hd_chances_against",
    "save_pct", "pace", "fivon5_xg_pct", "hd_ratio",
]


def fit_team_clusters(profiles, k_range=range(3, 7)):
    """Cluster teams into archetypes.

    Returns (labels, model, scaler, profiles_with_cluster).
    """
    X = profiles[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k = 4
    best_score = -1
    for k in k_range:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xs, labels)
        logger.info("Team clustering k=%d: silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(Xs)

    profiles = profiles.copy()
    profiles["cluster"] = labels

    # Describe clusters
    for cid in range(best_k):
        members = profiles[profiles["cluster"] == cid]
        teams = ", ".join(members["team"].tolist())
        avg_gf = members["gf_per_game"].mean()
        avg_ga = members["ga_per_game"].mean()
        avg_sv = members["save_pct"].mean()
        logger.info("Cluster %d (%d teams): GF=%.2f GA=%.2f SV%%=%.3f — %s",
                    cid, len(members), avg_gf, avg_ga, avg_sv, teams)

    return labels, model, scaler, profiles


# ---------------------------------------------------------------------------
# Matchup interaction: how does each cluster perform vs other clusters?
# ---------------------------------------------------------------------------

def build_cluster_matchup_matrix(profiles, before_date=None):
    """Compute how each team cluster performs against other clusters.

    Returns dict: (attacking_cluster, defending_cluster) -> avg_goals_scored
    """
    conn = get_db()

    if before_date is None:
        from datetime import date
        before_date = date.today().isoformat()

    team_cluster = dict(zip(profiles["team"], profiles["cluster"]))

    games = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score
        FROM games WHERE status = 'FINAL' AND date < ?
    """, (before_date,)).fetchall()
    conn.close()

    matchups = defaultdict(list)  # (atk_cluster, def_cluster) -> [goals]
    for g in games:
        hc = team_cluster.get(g["home_team"])
        ac = team_cluster.get(g["away_team"])
        if hc is None or ac is None:
            continue
        matchups[(hc, ac)].append(g["home_score"] or 0)  # home attacking vs away defending
        matchups[(ac, hc)].append(g["away_score"] or 0)  # away attacking vs home defending

    matrix = {}
    for key, goals in matchups.items():
        matrix[key] = round(np.mean(goals), 3) if goals else 3.0

    return matrix


def get_matchup_feature(home_team, away_team, profiles, matrix):
    """Get cluster matchup features for a specific game.

    Returns dict with:
    - home_cluster, away_cluster
    - home_vs_away_cluster_gf: how home's cluster scores vs away's cluster
    - away_vs_home_cluster_gf: how away's cluster scores vs home's cluster
    - cluster_total: expected total from cluster matchup
    - cluster_edge: home advantage from cluster matchup
    """
    team_cluster = dict(zip(profiles["team"], profiles["cluster"]))
    hc = team_cluster.get(home_team)
    ac = team_cluster.get(away_team)

    if hc is None or ac is None:
        return {
            "home_cluster": -1, "away_cluster": -1,
            "cluster_home_gf": 3.0, "cluster_away_gf": 3.0,
            "cluster_total": 6.0, "cluster_edge": 0.0,
        }

    home_gf = matrix.get((hc, ac), 3.0)
    away_gf = matrix.get((ac, hc), 3.0)

    return {
        "home_cluster": int(hc),
        "away_cluster": int(ac),
        "cluster_home_gf": home_gf,
        "cluster_away_gf": away_gf,
        "cluster_total": round(home_gf + away_gf, 2),
        "cluster_edge": round(home_gf - away_gf, 2),
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    profiles = build_team_profiles()
    print(f"\nTeam profiles: {len(profiles)} teams")
    print(profiles[["team", "gf_per_game", "ga_per_game", "save_pct", "pace", "fivon5_xg_pct"]].to_string())

    labels, model, scaler, profiles = fit_team_clusters(profiles)

    print(f"\nClusters (k={model.n_clusters}):")
    for cid in range(model.n_clusters):
        members = profiles[profiles["cluster"] == cid]
        print(f"\n  Cluster {cid}: {', '.join(members['team'].tolist())}")
        for col in CLUSTER_FEATURES:
            print(f"    {col:25s}: {members[col].mean():.3f}")

    matrix = build_cluster_matchup_matrix(profiles)
    print(f"\nCluster matchup matrix (attacker rows, defender cols):")
    k = model.n_clusters
    header = "     " + "".join(f"  vs{c}  " for c in range(k))
    print(header)
    for atk in range(k):
        row = f"  {atk}: "
        for def_ in range(k):
            gf = matrix.get((atk, def_), 0)
            row += f" {gf:5.2f} "
        print(row)

    # Test specific matchup
    print("\nSample matchup: COL @ DAL")
    mf = get_matchup_feature("DAL", "COL", profiles, matrix)
    print(f"  {mf}")
