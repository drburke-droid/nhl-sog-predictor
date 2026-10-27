"""Career fantasy value since 2000: points over a replacement player, 12-team PPR league.

    python vorp.py fetch     # nflverse weekly player stats and season rosters, 2000-2025 -> data/
    python vorp.py           # -> ../media/fantasy/seasons.csv, replacement.csv, weekly.csv

Scoring is standard PPR (nflverse's fantasy_points_ppr, recomputed here and checked against it): 1 point per 25 passing
yards, 4 per passing TD, -2 per interception, 1 per 10 rushing or receiving yards, 6 per rushing, receiving or
special-teams TD, 2 per two-point conversion, -2 per fumble lost, 1 per reception. Kickers: 3 per field goal up to 39
yards, 4 from 40-49, 5 from 50 on, 1 per extra point, -1 per missed (or blocked) field goal or extra point. Regular
season only.

Replacement level, per season and position, is the season total of the Nth-best player at the position. The league
starts a QB, 2 RBs, 3 WRs, a TE, a flex (RB, WR or TE) and a K, twelve teams:
* RB30 and WR42: the 24 RB and 36 WR starters, and about half the 12 flex spots each;
* TE14: the 12 starters, and the odd flex;
* QB16: the 12 starters, and the bye-week and injury fill-ins teams start along the way;
* K13: the 12 starters, and the next one.
A season's VORP is the player's season total minus that line, and never less than zero: a season under replacement is
one a manager would have filled off waivers.

Within a season the VORP accrues week by week in step with the player's points (each week's share of his season's
points), so each season's board is exact at its end and a player hurt in Week 3 stops climbing in Week 3.
"""
import os, sys, glob, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MEDIA = os.path.join(os.path.dirname(HERE), "media", "fantasy")
FIRST, LAST = 2000, 2025
REPL = {"QB": 16, "RB": 30, "WR": 42, "TE": 14, "K": 13}
POS = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "K": "K", "PK": "K"}
REL = "https://github.com/nflverse/nflverse-data/releases/download"


def fetch():
    for kind, url in (("weekly", REL + "/stats_player/stats_player_week_{y}.csv"), ("rosters", REL + "/rosters/roster_{y}.csv")):
        os.makedirs(os.path.join(DATA, kind), exist_ok=True)
        for y in range(FIRST, LAST + 1):
            f = os.path.join(DATA, kind, os.path.basename(url.format(y=y)))
            if not os.path.exists(f):
                subprocess.run(["curl", "-sSLf", "-o", f, url.format(y=y)], check=True)
                print("fetched", f, flush=True)


def ppr(d):
    c = lambda k: d[k].fillna(0)
    return (c("passing_yards") / 25 + 4 * c("passing_tds") - 2 * c("passing_interceptions")
            + (c("rushing_yards") + c("receiving_yards")) / 10
            + 6 * (c("rushing_tds") + c("receiving_tds") + c("special_teams_tds"))
            + 2 * (c("passing_2pt_conversions") + c("rushing_2pt_conversions") + c("receiving_2pt_conversions"))
            - 2 * (c("sack_fumbles_lost") + c("rushing_fumbles_lost") + c("receiving_fumbles_lost"))
            + c("receptions"))


def kicking(d):
    c = lambda k: d[k].fillna(0)
    return (3 * (c("fg_made_0_19") + c("fg_made_20_29") + c("fg_made_30_39")) + 4 * c("fg_made_40_49")
            + 5 * (c("fg_made_50_59") + c("fg_made_60_")) + c("pat_made")
            - (c("fg_missed") + c("fg_blocked") + c("pat_missed") + c("pat_blocked")))


def load():
    wk = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(DATA, "weekly", "*.csv")))])
    wk = wk[(wk.season_type == "REG") & wk.season.between(FIRST, LAST)].copy()
    wk["pts"] = ppr(wk)
    assert np.allclose(wk.pts, wk.fantasy_points_ppr.fillna(0), atol=1e-6), "PPR formula differs from nflverse's"
    wk["pts"] += kicking(wk)
    made = wk.fg_made_0_19 + wk.fg_made_20_29 + wk.fg_made_30_39 + wk.fg_made_40_49 + wk.fg_made_50_59 + wk.fg_made_60_
    assert (made.fillna(0) == wk.fg_made.fillna(0)).all(), "field goals by distance don't add up"
    # the position on that season's roster (a WR who became a RB counts as each in turn); the stats file's position
    # (the player's latest) where the roster has none or lists another position
    ro = pd.concat([pd.read_csv(f, low_memory=False, usecols=["season", "gsis_id", "position"])
                    for f in sorted(glob.glob(os.path.join(DATA, "rosters", "*.csv")))])
    ro = ro.dropna(subset=["gsis_id"])
    ro["rpos"] = ro.position.map(POS)
    ro = ro.dropna(subset=["rpos"]).groupby(["season", "gsis_id"]).rpos.agg(lambda s: s.mode().iloc[0]).reset_index()
    wk = wk.merge(ro, how="left", left_on=["season", "player_id"], right_on=["season", "gsis_id"])
    wk["pos"] = wk.rpos.fillna(wk.position.map(POS))
    return wk[wk.pos.notna()]


def build():
    wk = load()
    # season totals; the team he played the most games for (the later one on a tie), and the last one
    g = wk.sort_values("week").groupby(["player_id", "season"])
    sea = g.agg(player=("player_display_name", "last"), pos=("pos", "last"), games=("week", "size"), points=("pts", "sum"),
                last_team=("team", "last"), pass_yds=("passing_yards", "sum"), pass_td=("passing_tds", "sum"),
                rush_yds=("rushing_yards", "sum"), rush_td=("rushing_tds", "sum"), rec=("receptions", "sum"),
                rec_yds=("receiving_yards", "sum"), rec_td=("receiving_tds", "sum"), fg=("fg_made", "sum"),
                fg_50=("fg_made_50_59", "sum"), xp=("pat_made", "sum")).reset_index()
    main = (wk.groupby(["player_id", "season", "team"]).agg(n=("week", "size"), w=("week", "max")).reset_index()
              .sort_values(["n", "w"]).groupby(["player_id", "season"]).team.last().rename("team"))
    sea = sea.merge(main, on=["player_id", "season"])
    rows = []
    for (y, p), s in sea.groupby(["season", "pos"]):
        s = s.sort_values("points", ascending=False).reset_index(drop=True)
        n = REPL[p]
        r = s.iloc[n - 1]
        rows.append(dict(season=y, pos=p, rank=n, replacement=round(r.points, 2), player=r.player, team=r.team,
                         top=round(s.points.iloc[0], 2), top_player=s.player.iloc[0], starters_avg=round(s.points.iloc[:n - 1].mean(), 2)))
    rep = pd.DataFrame(rows)
    sea = sea.merge(rep[["season", "pos", "replacement"]], on=["season", "pos"])
    sea["vorp"] = (sea.points - sea.replacement).clip(lower=0)
    sea["pos_rank"] = sea.groupby(["season", "pos"]).points.rank(ascending=False, method="first").astype(int)
    sea = sea.sort_values(["player_id", "season"])
    sea["career_vorp"] = sea.groupby("player_id").vorp.cumsum()
    sea["career_points"] = sea.groupby("player_id").points.cumsum()
    sea["seasons_above"] = sea.groupby("player_id").vorp.transform(lambda v: (v > 0).cumsum())
    # weekly: the season's VORP accrues with each week's share of his (non-negative) points
    wk = wk.merge(sea[["player_id", "season", "vorp"]], on=["player_id", "season"])
    wk = wk.sort_values(["player_id", "season", "week"])
    pos_pts = wk.pts.clip(lower=0)
    tot = pos_pts.groupby([wk.player_id, wk.season]).transform("sum")
    share = np.where(tot > 0, pos_pts / tot.where(tot > 0, 1), 0)
    wk["vorp_week"] = wk.vorp * share
    weeks = wk[["player_id", "player_display_name", "season", "week", "team", "pos", "pts", "vorp_week"]].rename(
        columns={"player_display_name": "player", "pts": "points"})
    os.makedirs(MEDIA, exist_ok=True)
    cols = ["player_id", "player", "pos", "season", "team", "last_team", "games", "pass_yds", "pass_td", "rush_yds", "rush_td",
            "rec", "rec_yds", "rec_td", "fg", "fg_50", "xp", "points", "pos_rank", "replacement", "vorp", "career_vorp", "career_points",
            "seasons_above"]
    sea[cols].round(2).sort_values(["season", "vorp", "points"], ascending=[True, False, False]).to_csv(os.path.join(MEDIA, "seasons.csv"), index=False)
    rep.to_csv(os.path.join(MEDIA, "replacement.csv"), index=False)
    # the weekly file keeps players with a season of VORP (everyone else stays at zero)
    weeks = weeks[weeks.player_id.isin(sea.loc[sea.vorp > 0, "player_id"])]
    weeks.round(3).to_csv(os.path.join(MEDIA, "weekly.csv"), index=False)
    car = sea.groupby("player_id").agg(player=("player", "last"), pos=("pos", "last"), first=("season", "min"), last=("season", "max"),
                                       vorp=("vorp", "sum"), points=("points", "sum"), above=("seasons_above", "max")).sort_values("vorp", ascending=False)
    print(rep.pivot(index="season", columns="pos", values="replacement").round(0).astype(int).to_string())
    print(car.head(30).round(0).to_string())
    return sea, rep, car


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        build()
