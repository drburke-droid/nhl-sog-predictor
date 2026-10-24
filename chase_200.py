"""
Can a player reach 200 points this season?  Reads nhl_history.db (built by
nhl_history_collector.py) and writes the chase tables as JSON next to the
hot-start tables in docs/mcdavid/data/.

    python chase_200.py                 # Connor McDavid
    python chase_200.py 8471675         # any player id

Required pace = points still needed / the team's games left, assuming the
player plays every one.  Streaks run over consecutive regular-season games
across his whole career, season breaks included: a K-game stretch counts when
it averages at least the required pace, and stretches are counted without
overlap (scan forward, and after a qualifying stretch skip past it).
"""

import os
import sqlite3
import sys

import pandas as pd

from hot_start import DB_PATH, MCDAVID, OUT_DIR, season_label

TARGET = 200
WINDOWS = [10, 20, 30, 41, 60]   # streak lengths to test, plus the games left
REFERENCE_SEASONS = [            # (player id, season) drawn on the chase chart
    (8447400, 19851986),         # Gretzky's record 215
    (8448782, 19881989),         # Lemieux's 199
]


def streaks(points: pd.Series, k: int, pace: float) -> dict:
    avg = points.rolling(k).sum() / k
    count, i = 0, k - 1
    while i < len(avg):
        if avg.iloc[i] >= pace:
            count += 1
            i += k
        else:
            i += 1
    best = int(avg.idxmax()) if avg.notna().any() else None
    return {"avg": avg, "count": count, "best": best,
            "share": float((avg >= pace).sum() / avg.notna().sum()) if avg.notna().any() else 0.0}


def main(player_id: int):
    conn = sqlite3.connect(DB_PATH)
    current = conn.execute("SELECT MAX(seasonId) FROM skater_games").fetchone()[0]
    season_games = conn.execute("SELECT numberOfGames FROM seasons WHERE id = ?",
                                (current,)).fetchone()[0]
    log = pd.read_sql(
        "SELECT seasonId, gameId, gameDate, teamAbbrev AS team, goals, assists, points, "
        "skaterFullName AS name FROM skater_games "
        "WHERE playerId = ? AND gameTypeId = 2 ORDER BY gameDate, gameId",
        conn, params=(player_id,)).reset_index(drop=True)
    player = log.name.iloc[-1]
    now = log[log.seasonId == current]
    team = now.team.iloc[-1]
    team_gp = conn.execute(
        "SELECT COUNT(DISTINCT gameId) FROM skater_games "
        "WHERE seasonId = ? AND gameTypeId = 2 AND teamAbbrev = ?", (current, team)).fetchone()[0]
    pts, gp = int(now.points.sum()), len(now)
    left = season_games - team_gp
    needed = TARGET - pts
    pace = needed / left

    seasons = log.groupby("seasonId").agg(gp=("points", "size"), pts=("points", "sum"),
                                          goals=("goals", "sum"), assists=("assists", "sum"))
    seasons["ppg"] = seasons.pts / seasons.gp
    done = seasons[seasons.index < current]
    best_season = done.ppg.idxmax()
    top_season = done.pts.idxmax()
    career_ppg = done.pts.sum() / done.gp.sum()
    last = done.index.max()

    def finish(ppg):
        return round(pts + ppg * left)

    # Streaks at the required pace, over the whole career
    tests = {}
    for k in sorted(set(WINDOWS + [left])):
        tests[k] = streaks(log.points, k, pace)
    streak_table = pd.DataFrame([{
        "games": k, "is_rest_of_season": k == left,
        "streaks": t["count"], "share_of_stretches": round(t["share"], 4),
        "best_ppg": round(float(t["avg"].iloc[t["best"]]), 3),
        "best_from": log.gameDate.iloc[t["best"] - k + 1], "best_to": log.gameDate.iloc[t["best"]],
        "best_pts": int(log.points.iloc[t["best"] - k + 1:t["best"] + 1].sum()),
    } for k, t in tests.items()])

    rolling = pd.DataFrame({"game": log.index + 1, "date": log.gameDate,
                            "season": log.seasonId.map(season_label)})
    for k, t in tests.items():
        rolling[f"ppg_{k}"] = t["avg"].round(3)

    season_table = pd.DataFrame({
        "season": [season_label(s) for s in seasons.index], "gp": seasons.gp,
        "goals": seasons.goals, "assists": seasons.assists, "pts": seasons.pts,
        "ppg": seasons.ppg.round(3), "pace_84": (seasons.ppg * season_games).round(0).astype(int),
        "this_season": seasons.index == current})

    # Running totals: this season, his biggest season, and the 200-point benchmarks
    refs = [(player_id, current), (player_id, int(top_season))] + REFERENCE_SEASONS
    curves = []
    for pid, sid in refs:
        g = pd.read_sql("SELECT gameDate, points, skaterFullName AS name FROM skater_games "
                        "WHERE playerId = ? AND seasonId = ? AND gameTypeId = 2 ORDER BY gameDate",
                        conn, params=(pid, sid))
        curves.append(pd.DataFrame({
            "key": f"{pid}-{sid}", "player": g.name.iloc[-1], "season": season_label(sid),
            "this_season": pid == player_id and sid == current,
            "gp": range(1, len(g) + 1), "pts": g.points.cumsum()}))
    chase_curves = pd.concat(curves)

    two_hundred = pd.read_sql(
        "SELECT skaterFullName AS player, seasonId, SUM(points) AS pts FROM skater_games "
        "WHERE gameTypeId = 2 GROUP BY playerId, seasonId HAVING pts >= ? ORDER BY pts DESC",
        conn, params=(TARGET,))
    rest = tests[left]
    chase = pd.DataFrame([{
        "player": player, "season": season_label(current), "team": team,
        "as_of": now.gameDate.max(), "target": TARGET, "pts": pts, "gp": gp,
        "team_gp": team_gp, "season_games": season_games, "games_left": left,
        "needed": needed, "required_ppg": round(pace, 3),
        "this_season_ppg": round(pts / gp, 3),
        "career_ppg": round(career_ppg, 3), "career_gp": int(done.gp.sum()),
        "career_seasons": len(done),
        "all_gp": len(log),
        "best_season": season_label(int(best_season)),
        "best_season_ppg": round(float(done.ppg.max()), 3),
        "best_season_pts": int(done.pts[best_season]),
        "best_season_gp": int(done.gp[best_season]),
        "top_season": season_label(int(top_season)),
        "top_season_pts": int(done.pts[top_season]),
        "top_season_gp": int(done.gp[top_season]),
        "last_season": season_label(int(last)),
        "last_season_ppg": round(float(done.ppg[last]), 3),
        "finish_at_this_season_pace": finish(pts / gp),
        "finish_at_best_season_pace": finish(float(done.ppg.max())),
        "finish_at_career_pace": finish(career_ppg),
        "finish_at_last_season_pace": finish(float(done.ppg[last])),
        "best_rest_ppg": streak_table.loc[streak_table.is_rest_of_season, "best_ppg"].iloc[0],
        "best_rest_from": streak_table.loc[streak_table.is_rest_of_season, "best_from"].iloc[0],
        "best_rest_to": streak_table.loc[streak_table.is_rest_of_season, "best_to"].iloc[0],
        "rest_streaks": rest["count"],
        "seasons_at_pace": int((done.ppg >= pace).sum()),
        "pace_vs_best_season": round(pace / float(done.ppg.max()) - 1, 3),
        "pace_vs_career": round(pace / career_ppg - 1, 3),
        "players_200": int(two_hundred.player.nunique()),
        "seasons_200": len(two_hundred),
        "last_200": season_label(int(two_hundred.seasonId.max())) if len(two_hundred) else None,
        "last_200_player": (two_hundred.sort_values("seasonId").player.iloc[-1]
                            if len(two_hundred) else None),
    }])

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, table in [("chase", chase), ("chase_seasons", season_table),
                        ("chase_streaks", streak_table), ("chase_rolling", rolling),
                        ("chase_curves", chase_curves)]:
        table.to_json(f"{OUT_DIR}/{name}.json", orient="records")
    c = chase.iloc[0]
    print(f"{player}: {pts} pts in {gp} GP, needs {needed} in {left} games = {pace:.3f} PPG "
          f"(career {career_ppg:.3f}, best season {c.best_season_ppg}, "
          f"best {left}-game stretch {c.best_rest_ppg}) -> {OUT_DIR}/")
    print(streak_table.to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else MCDAVID)
