"""
How does a player's start to this season compare with every start in NHL
history?  Reads nhl_history.db (built by nhl_history_collector.py) and writes
the tables behind the chart as JSON to docs/reports/hot_start/.

    python hot_start.py                 # Connor McDavid
    python hot_start.py 8471675         # any player id

"Start" = a player's first N games played in a regular season, where N is the
games the player has played so far this season.  Era-adjusted figures scale
goals and assists separately to the league's per-game rates over the last
three completed seasons, so a 1980s point counts like today's.  Early seasons
credited far fewer assists per goal, which no rate scaling fixes, so the
era-adjusted ranking only covers the unbroken run of seasons, back from now,
whose assists per goal are within 15% of today's (1949-50 on).
"""

import os
import sqlite3
import sys

import pandas as pd

DB_PATH = "nhl_history.db"
OUT_DIR = "docs/reports/hot_start"
MCDAVID = 8478402
CURVE_GAMES = 20   # how far to draw each start's curve
CURVE_LEADERS = 12 # best starts drawn as curves, on each scale
TOP = 25           # starts shown in the ranking, on each scale
ASSIST_TOLERANCE = 0.15


def load(conn) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT playerId, skaterFullName AS name, seasonId, gameId, gameDate, "
        "teamAbbrev AS team, opponentTeamAbbrev AS opponent, positionCode AS pos, "
        "goals, assists, points "
        "FROM skater_games WHERE gameTypeId = 2", conn)
    df = df.sort_values(["playerId", "seasonId", "gameDate", "gameId"])
    df["gp"] = df.groupby(["playerId", "seasonId"]).cumcount() + 1
    return df


def era_factors(df: pd.DataFrame, current: int) -> pd.DataFrame:
    """Per-season multipliers that put goals and assists on a modern scale."""
    s = df.groupby("seasonId").agg(goals=("goals", "sum"), assists=("assists", "sum"),
                                  games=("gameId", "nunique"))
    s["gpg"] = s.goals / s.games
    s["apg"] = s.assists / s.games
    ref = s[s.index < current].tail(3)
    s["g_factor"] = ref.goals.sum() / ref.games.sum() / s.gpg
    s["a_factor"] = ref.assists.sum() / ref.games.sum() / s.apg
    ratio = s.apg / s.gpg
    close = (ratio >= ref.assists.sum() / ref.goals.sum() * (1 - ASSIST_TOLERANCE))
    s["comparable"] = close[::-1].cummin()[::-1]  # unbroken run back from now
    return s


def season_label(sid: int) -> str:
    return f"{str(sid)[:4]}-{str(sid)[6:]}"


def main(player_id: int):
    conn = sqlite3.connect(DB_PATH)
    df = load(conn)
    current = int(df.seasonId.max())
    era = era_factors(df, current)
    df = df.join(era[["g_factor", "a_factor", "comparable"]], on="seasonId")
    df["adj_points"] = (df.goals * df.g_factor + df.assists * df.a_factor).where(df.comparable)

    me = df[(df.playerId == player_id) & (df.seasonId == current)]
    n = int(me.gp.max())
    player_name = me.name.iloc[0]

    # Every player-season that reached n games played
    first_n = df[df.gp <= n]
    starts = first_n.groupby(["playerId", "seasonId"]).agg(
        name=("name", "last"), team=("team", "first"), pos=("pos", "last"),
        gp=("gp", "max"), goals=("goals", "sum"), assists=("assists", "sum"),
        points=("points", "sum"), adj_points=("adj_points", "sum"))
    starts = starts[starts.gp == n].reset_index()
    full = df.groupby(["playerId", "seasonId"]).agg(
        season_gp=("gp", "max"), season_points=("points", "sum"))
    starts = starts.join(full, on=["playerId", "seasonId"])
    starts["season"] = starts.seasonId.map(season_label)
    starts["is_target"] = (starts.playerId == player_id) & (starts.seasonId == current)

    starts.loc[~starts.seasonId.map(era.comparable), "adj_points"] = None

    def rank_of(col):
        mine = starts.loc[starts.is_target, col].iloc[0]
        better = int((starts[col] > mine).sum())
        tied = int((starts[col] == mine).sum()) - 1
        return {"value": round(float(mine), 1), "rank": better + 1, "tied": tied}

    raw_rank = rank_of("points")
    adj_rank = rank_of("adj_points")
    goal_rank = rank_of("goals")
    assist_rank = rank_of("assists")

    # Curves: the target's own seasons, plus the best starts on either scale
    own = starts[starts.playerId == player_id]
    leaders = pd.concat([starts.nlargest(CURVE_LEADERS, "points"),
                         starts.nlargest(CURVE_LEADERS, "adj_points")])
    keys = set(zip(own.playerId, own.seasonId)) | set(zip(leaders.playerId, leaders.seasonId))
    in_curve = pd.Series(list(zip(df.playerId, df.seasonId)), index=df.index).isin(keys)
    c = df[in_curve & (df.gp <= CURVE_GAMES)].copy()
    c["cum_pts"] = c.groupby(["playerId", "seasonId"]).points.cumsum()
    c["cum_adj"] = c.groupby(["playerId", "seasonId"]).adj_points.cumsum().round(2)
    curves = pd.DataFrame({
        "key": c.playerId.astype(str) + "-" + c.seasonId.astype(str),
        "player": c.name, "season": c.seasonId.map(season_label), "gp": c.gp,
        "date": c.gameDate, "pts": c.cum_pts, "adj_pts": c.cum_adj,
        "group": [("this season" if (p == player_id and sid == current) else
                   f"{player_name}, other seasons" if p == player_id else "top start")
                  for p, sid in zip(c.playerId, c.seasonId)]})

    # Ranking: the best starts on either scale, plus the target
    starts["rank"] = starts.points.rank(method="min", ascending=False).astype(int)
    starts["adj_rank"] = starts.adj_points.rank(method="min", ascending=False).astype("Int64")
    shown = starts[(starts["rank"] <= TOP) | (starts.adj_rank <= TOP) | starts.is_target]
    ranking = pd.DataFrame({
        "key": shown.playerId.astype(str) + "-" + shown.seasonId.astype(str),
        "player": shown.name, "season": shown.season, "team": shown.team,
        "goals": shown.goals, "assists": shown.assists, "pts": shown.points,
        "adj_pts": shown.adj_points.round(1), "rank": shown["rank"],
        "adj_rank": shown.adj_rank, "season_gp": shown.season_gp,
        "season_pts": shown.season_points, "this_season": shown.is_target,
    }).sort_values("rank")

    a = starts.sort_values(["assists", "points"], ascending=False).head(10)
    assist_leaders = pd.DataFrame({
        "key": a.playerId.astype(str) + "-" + a.seasonId.astype(str),
        "player": a.name, "season": a.season, "assists": a.assists, "pts": a.points,
        "assist_rank": a.assists.rank(method="min", ascending=False).astype(int),
        "this_season": a.is_target})

    own_starts = own.sort_values("seasonId")
    own_table = pd.DataFrame({
        "season": own_starts.season, "goals": own_starts.goals,
        "assists": own_starts.assists, "pts": own_starts.points,
        "adj_pts": own_starts.adj_points.round(1), "season_gp": own_starts.season_gp,
        "season_pts": own_starts.season_points, "this_season": own_starts.is_target})

    log = me.assign(cum_pts=me.points.cumsum())
    game_log = pd.DataFrame({"gp": log.gp, "date": log.gameDate, "opponent": log.opponent,
                             "goals": log.goals, "assists": log.assists,
                             "pts": log.points, "cum_pts": log.cum_pts})

    summary = pd.DataFrame([{
        "player": player_name, "season": season_label(current), "gp": n,
        "as_of": me.gameDate.max(), "goals": int(me.goals.sum()),
        "assists": int(me.assists.sum()), "pts": int(me.points.sum()),
        "adj_pts": adj_rank["value"],
        "rank": raw_rank["rank"], "tied": raw_rank["tied"],
        "adj_rank": adj_rank["rank"], "adj_tied": adj_rank["tied"],
        "adj_starts": int(starts.adj_points.notna().sum()),
        "adj_first_season": season_label(int(era.index[era.comparable].min())),
        "goal_rank": goal_rank["rank"], "goal_tied": goal_rank["tied"],
        "assist_rank": assist_rank["rank"], "assist_tied": assist_rank["tied"],
        "own_rank": int((own.points > own.loc[own.is_target, "points"].iloc[0]).sum()) + 1,
        "own_seasons": len(own), "starts": len(starts),
        "first_season": season_label(int(df.seasonId.min())),
        "seasons": int(df.seasonId.nunique()), "game_rows": len(df)}])

    era_table = pd.DataFrame({"season": [season_label(int(k)) for k in era.index],
                              "goals_per_game": era.gpg.round(3),
                              "assists_per_game": era.apg.round(3),
                              "goal_factor": era.g_factor.round(3),
                              "assist_factor": era.a_factor.round(3),
                              "in_adjusted_ranking": era.comparable})

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, table in [("summary", summary), ("ranking", ranking), ("curves", curves),
                        ("assist_leaders", assist_leaders), ("own_starts", own_table),
                        ("game_log", game_log), ("era", era_table)]:
        table.to_json(f"{OUT_DIR}/{name}.json", orient="records")
    print(f"{player_name} {season_label(current)}: {summary.pts[0]} pts in {n} GP, "
          f"#{raw_rank['rank']} (+{raw_rank['tied']} tied) raw, "
          f"#{adj_rank['rank']} era-adjusted, of {len(starts):,} starts -> {OUT_DIR}/")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else MCDAVID)
