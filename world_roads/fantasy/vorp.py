"""Career fantasy value since 2000: PPR points and value over a replacement player, 12-team league.

    python vorp.py fetch     # nflverse weekly player stats and season rosters, 2000-2025 -> data/
    python vorp.py           # -> ../media/fantasy/seasons.csv, replacement.csv, weekly.csv

Scoring is standard PPR (nflverse's fantasy_points_ppr, recomputed here and checked against it): 1 point per 25 passing
yards, 4 per passing TD, -2 per interception, 1 per 10 rushing or receiving yards, 6 per rushing, receiving or
special-teams TD, 2 per two-point conversion, -2 per fumble lost, 1 per reception. Regular season only.

Replacement level, per season and position, is the season total of the Nth-best player at the position: the 13th QB
and TE (a 12-team league starts 12 of each, so the 13th is the best one left on waivers), the 36th RB and WR (two
starters, a flex share and a bench spot each, three per team). A season's VORP is the player's season total minus that
line, and never less than zero: a season under replacement is one a manager would have filled off waivers.

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
REPL = {"QB": 13, "RB": 36, "WR": 36, "TE": 13}
POS = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
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


def load():
    wk = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(DATA, "weekly", "*.csv")))])
    wk = wk[(wk.season_type == "REG") & wk.season.between(FIRST, LAST)].copy()
    wk["pts"] = ppr(wk)
    assert np.allclose(wk.pts, wk.fantasy_points_ppr.fillna(0), atol=1e-6), "PPR formula differs from nflverse's"
    # the position on that season's roster (a WR who became a RB counts as each in turn); the stats file's position
    # (the player's latest) where the roster has none or lists a non-skill position
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
    sea = g.agg(player=("player_display_name", "last"), pos=("pos", "last"), games=("week", "size"), ppr=("pts", "sum"),
                last_team=("team", "last"), pass_yds=("passing_yards", "sum"), pass_td=("passing_tds", "sum"),
                rush_yds=("rushing_yards", "sum"), rush_td=("rushing_tds", "sum"), rec=("receptions", "sum"),
                rec_yds=("receiving_yards", "sum"), rec_td=("receiving_tds", "sum")).reset_index()
    main = (wk.groupby(["player_id", "season", "team"]).agg(n=("week", "size"), w=("week", "max")).reset_index()
              .sort_values(["n", "w"]).groupby(["player_id", "season"]).team.last().rename("team"))
    sea = sea.merge(main, on=["player_id", "season"])
    rows = []
    for (y, p), s in sea.groupby(["season", "pos"]):
        s = s.sort_values("ppr", ascending=False).reset_index(drop=True)
        n = REPL[p]
        r = s.iloc[n - 1]
        rows.append(dict(season=y, pos=p, rank=n, replacement=round(r.ppr, 2), player=r.player, team=r.team,
                         top=round(s.ppr.iloc[0], 2), top_player=s.player.iloc[0], starters_avg=round(s.ppr.iloc[:n - 1].mean(), 2)))
    rep = pd.DataFrame(rows)
    sea = sea.merge(rep[["season", "pos", "replacement"]], on=["season", "pos"])
    sea["vorp"] = (sea.ppr - sea.replacement).clip(lower=0)
    sea["pos_rank"] = sea.groupby(["season", "pos"]).ppr.rank(ascending=False, method="first").astype(int)
    sea = sea.sort_values(["player_id", "season"])
    sea["career_vorp"] = sea.groupby("player_id").vorp.cumsum()
    sea["career_ppr"] = sea.groupby("player_id").ppr.cumsum()
    sea["seasons_above"] = sea.groupby("player_id").vorp.transform(lambda v: (v > 0).cumsum())
    # weekly: the season's VORP accrues with each week's share of his (non-negative) points
    wk = wk.merge(sea[["player_id", "season", "vorp"]], on=["player_id", "season"])
    wk = wk.sort_values(["player_id", "season", "week"])
    pos_pts = wk.pts.clip(lower=0)
    tot = pos_pts.groupby([wk.player_id, wk.season]).transform("sum")
    share = np.where(tot > 0, pos_pts / tot.where(tot > 0, 1), 0)
    wk["vorp_week"] = wk.vorp * share
    weeks = wk[["player_id", "player_display_name", "season", "week", "team", "pos", "pts", "vorp_week"]].rename(
        columns={"player_display_name": "player", "pts": "ppr"})
    os.makedirs(MEDIA, exist_ok=True)
    cols = ["player_id", "player", "pos", "season", "team", "last_team", "games", "pass_yds", "pass_td", "rush_yds", "rush_td",
            "rec", "rec_yds", "rec_td", "ppr", "pos_rank", "replacement", "vorp", "career_vorp", "career_ppr", "seasons_above"]
    sea[cols].round(2).sort_values(["season", "vorp", "ppr"], ascending=[True, False, False]).to_csv(os.path.join(MEDIA, "seasons.csv"), index=False)
    rep.to_csv(os.path.join(MEDIA, "replacement.csv"), index=False)
    # the weekly file keeps players with a season of VORP (everyone else stays at zero)
    weeks = weeks[weeks.player_id.isin(sea.loc[sea.vorp > 0, "player_id"])]
    weeks.round(3).to_csv(os.path.join(MEDIA, "weekly.csv"), index=False)
    car = sea.groupby("player_id").agg(player=("player", "last"), pos=("pos", "last"), first=("season", "min"), last=("season", "max"),
                                       vorp=("vorp", "sum"), ppr=("ppr", "sum"), above=("seasons_above", "max")).sort_values("vorp", ascending=False)
    print(rep.pivot(index="season", columns="pos", values="replacement").round(0).astype(int).to_string())
    print(car.head(30).round(0).to_string())
    return sea, rep, car


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        build()
