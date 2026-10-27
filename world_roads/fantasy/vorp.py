"""Fantasy football's most valuable seasons since 1980: points over a replacement player, 12-team PPR league.

    python vorp.py fetch     # season tables 1980-1998, nflverse weekly player stats and rosters 1999-2025 -> data/
    python vorp.py           # -> ../media/fantasy/seasons.csv, replacement.csv, weekly.csv

Sources. 1980-1998: Pro Football Reference's season fantasy tables, as collected by fantasydatapros/data on GitHub
(season totals only; Pro Football Reference itself refuses this machine). 1999-2025: nflverse's weekly player stats,
with each season's roster position.

Scoring is standard PPR: 1 point per 25 passing yards, 4 per passing TD, -2 per interception, 1 per 10 rushing or
receiving yards, 6 per rushing or receiving TD, 1 per reception, -2 per fumble lost; from 1999 also 6 per
special-teams TD and 2 per two-point conversion (nflverse's fantasy_points_ppr, recomputed here and checked against it;
the 1980-1998 tables carry neither). Fumbles lost are recorded from 1994; for 1980-1993 they are estimated as 54.5% of
fumbles, the league's share lost in 1994-1998. Regular season only.

Replacement level, per season and position, is the season total of the Nth-best player at the position. The league
starts a QB, 2 RBs, 3 WRs, a TE and a flex (RB, WR or TE), twelve teams:
* RB30 and WR42: the 24 RB and 36 WR starters, and about half the 12 flex spots each;
* TE14: the 12 starters, and the odd flex;
* QB16: the 12 starters, and the bye-week and injury fill-ins teams start along the way.
Kickers are left out: there is no kicking data before 1999 here, and from 1999 no kicker season comes near the board.

VORP is counted by the game: a season is worth the player's points minus replacement's share of the games he played
(the line over the season's games: 16; 9 in the 1982 strike, 15 in 1987, 17 from 2021). A full season is worth his
total minus the line; a season of 8 games, his total minus half the line. A missed game counts nothing, since his
manager starts a replacement that week.

Within a season the film grows each season's value toward its total: from 1999 evenly over the games he played (by
week, from the weekly stats), before 1999 evenly over the season's weeks (only season totals exist).
"""
import os, sys, glob, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MEDIA = os.path.join(os.path.dirname(HERE), "media", "fantasy")
FIRST, WEEKLY_FROM, LAST = 1980, 1999, 2025
REPL = {"QB": 16, "RB": 30, "WR": 42, "TE": 14}
POS = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
REL = "https://github.com/nflverse/nflverse-data/releases/download"
FDP = "https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/{y}.csv"
LOST_SHARE = 0.545                                    # fumbles lost / fumbles, 1994-1998 (1,775 of 3,257)
CANDIDATES = 13                                       # a season can reach the all-time 12 only from its year's 12


def team_games(season):
    s = np.asarray(season)
    return np.select([s == 1982, s == 1987, s >= 2021], [9, 15, 17], 16)


def weeks(season):
    """weeks in the season, for the week counter: 16 games a week apart to 1989 (9 in 1982, 15 in 1987), byes from
    1990 (17 weeks; 18 in 1993), 18 from 2021"""
    return {1982: 9, 1987: 15, 1993: 18}.get(season, 16 if season < 1990 else 18 if season >= 2021 else 17)


def vorp(points, replacement, games, season):
    """a season's value over replacement, game by game: his points minus the line's share of the games he played"""
    return points - replacement * games / team_games(season)


def nickname(code, season):
    """a team's nickname, from Pro Football Reference's or nflverse's code, in that season"""
    if isinstance(code, str) and code.endswith("TM"):
        return f"{code[0]} teams"
    by_era = {"STL": "Cardinals" if season <= 1987 else "Rams", "BAL": "Colts" if season <= 1983 else "Ravens",
              "HOU": "Oilers" if season <= 1996 else "Texans", "TEN": "Oilers" if season <= 1998 else "Titans"}
    return by_era.get(code) or TEAM.get(code, code)


TEAM = {"ARI": "Cardinals", "PHO": "Cardinals", "ATL": "Falcons", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
        "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
        "GNB": "Packers", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "KAN": "Chiefs", "LA": "Rams", "RAM": "Rams",
        "LAR": "Rams", "LAC": "Chargers", "SD": "Chargers", "SDG": "Chargers", "LV": "Raiders", "OAK": "Raiders",
        "RAI": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NWE": "Patriots", "NO": "Saints",
        "NOR": "Saints", "NYG": "Giants", "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
        "SF": "49ers", "SFO": "49ers", "TB": "Buccaneers", "TAM": "Buccaneers", "WAS": "Washington"}


def fetch():
    os.makedirs(os.path.join(DATA, "pfr"), exist_ok=True)
    jobs = [(os.path.join(DATA, "pfr", f"{y}.csv"), FDP.format(y=y)) for y in range(FIRST, WEEKLY_FROM)]
    for kind, url in (("weekly", REL + "/stats_player/stats_player_week_{y}.csv"), ("rosters", REL + "/rosters/roster_{y}.csv")):
        os.makedirs(os.path.join(DATA, kind), exist_ok=True)
        jobs += [(os.path.join(DATA, kind, os.path.basename(url.format(y=y))), url.format(y=y)) for y in range(WEEKLY_FROM, LAST + 1)]
    for f, url in jobs:
        if not os.path.exists(f):
            subprocess.run(["curl", "-sSLf", "-o", f, url], check=True)
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
    """nflverse's weekly lines from 1999, regular season, with PPR points and the season's roster position"""
    wk = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(DATA, "weekly", "*.csv")))])
    wk = wk[(wk.season_type == "REG") & wk.season.between(WEEKLY_FROM, LAST)].copy()
    wk["pts"] = ppr(wk)
    assert np.allclose(wk.pts, wk.fantasy_points_ppr.fillna(0), atol=1e-6), "PPR formula differs from nflverse's"
    # the position on that season's roster (a WR who became a RB counts as each in turn); the stats file's position
    # (the player's latest) where the roster has none or lists another position
    ro = pd.concat([pd.read_csv(f, low_memory=False, usecols=["season", "gsis_id", "position"])
                    for f in sorted(glob.glob(os.path.join(DATA, "rosters", "*.csv")))])
    ro = ro.dropna(subset=["gsis_id"])
    ro["rpos"] = ro.position.map(POS)
    ro = ro.dropna(subset=["rpos"]).groupby(["season", "gsis_id"]).rpos.agg(lambda s: s.mode().iloc[0]).reset_index()
    wk = wk.merge(ro, how="left", left_on=["season", "player_id"], right_on=["season", "gsis_id"])
    wk["pos"] = wk.rpos.fillna(wk.position.map(POS))
    wk = wk[wk.pos.notna()].copy()
    wk["team_name"] = [nickname(t, y) for t, y in zip(wk.team, wk.season)]
    return wk


def pfr():
    """season totals 1980-1998 from Pro Football Reference's fantasy tables (via fantasydatapros/data)"""
    rows = []
    for y in range(FIRST, WEEKLY_FROM):
        d = pd.read_csv(os.path.join(DATA, "pfr", f"{y}.csv"))
        d = d[d.Pos.astype(str).isin(REPL)].copy()
        c = lambda k: d[k].fillna(0)
        lost = c("FumblesLost") if y >= 1994 else LOST_SHARE * c("Fumbles")
        d["points"] = (c("PassingYds") / 25 + 4 * c("PassingTD") - 2 * c("Int") + c("RushingYds") / 10 + 6 * c("RushingTD")
                       + c("Rec") + c("ReceivingYds") / 10 + 6 * c("ReceivingTD") - 2 * lost)
        name = d.Player.str.replace(r"[*+]", "", regex=True).str.strip()
        rows.append(pd.DataFrame(dict(
            player_id=[f"pfr:{y}:{n}" for n in name], player=name, pos=d.Pos.astype(str), season=y,
            team=[nickname(t, y) for t in d.Tm], games=c("G").astype(int), pass_yds=c("PassingYds"), pass_td=c("PassingTD"),
            rush_yds=c("RushingYds"), rush_td=c("RushingTD"), rec=c("Rec"), rec_yds=c("ReceivingYds"), rec_td=c("ReceivingTD"),
            points=d.points, source="pfr")))
    return pd.concat(rows, ignore_index=True)


def nflverse(wk):
    """season totals from 1999, from the weekly lines; the team he played the most games for (the later on a tie)"""
    g = wk.sort_values("week").groupby(["player_id", "season"])
    sea = g.agg(player=("player_display_name", "last"), pos=("pos", "last"), games=("week", "size"), points=("pts", "sum"),
                pass_yds=("passing_yards", "sum"), pass_td=("passing_tds", "sum"), rush_yds=("rushing_yards", "sum"),
                rush_td=("rushing_tds", "sum"), rec=("receptions", "sum"), rec_yds=("receiving_yards", "sum"),
                rec_td=("receiving_tds", "sum")).reset_index()
    main = (wk.groupby(["player_id", "season", "team_name"]).agg(n=("week", "size"), w=("week", "max")).reset_index()
              .sort_values(["n", "w"]).groupby(["player_id", "season"]).team_name.last().rename("team"))
    sea = sea.merge(main, on=["player_id", "season"])
    sea["source"] = "nflverse"
    return sea


def seasons(wk=None, ranks=REPL):
    """every QB, RB, WR and TE season 1980-2025 with its replacement line and VORP under the given ranks"""
    wk = load() if wk is None else wk
    sea = pd.concat([pfr(), nflverse(wk)], ignore_index=True)
    sea = sea[sea.pos.isin(ranks)].copy()
    sea["pos_rank"] = sea.groupby(["season", "pos"]).points.rank(ascending=False, method="first").astype(int)
    line = sea[sea.pos_rank == sea.pos.map(ranks)].set_index(["season", "pos"])
    sea["replacement"] = [line.points[y, p] for y, p in zip(sea.season, sea.pos)]
    sea["vorp"] = vorp(sea.points, sea.replacement, sea.games, sea.season)
    sea["season_rank"] = sea.groupby("season").vorp.rank(ascending=False, method="first").astype(int)
    return sea, line


def build():
    wk = load()
    sea, line = seasons(wk)
    rep = line.reset_index()[["season", "pos", "pos_rank", "points", "player", "team"]].rename(
        columns={"pos_rank": "rank", "points": "replacement"})
    top = sea.sort_values("points", ascending=False).groupby(["season", "pos"]).first()
    rep["top_player"] = [top.player[y, p] for y, p in zip(rep.season, rep.pos)]
    rep["top"] = [top.points[y, p] for y, p in zip(rep.season, rep.pos)]
    # weekly, from 1999, for the seasons that can reach the board: the season's VORP spread evenly over his games
    cand = sea[(sea.season_rank <= CANDIDATES) & (sea.source == "nflverse")][["player_id", "season", "vorp", "games"]]
    w = wk.merge(cand, on=["player_id", "season"]).sort_values(["season", "player_id", "week"])
    w["vorp_week"] = w.vorp / w.games
    weekly = w[["player_id", "player_display_name", "season", "week", "team_name", "pts", "vorp_week"]].rename(
        columns={"player_display_name": "player", "team_name": "team", "pts": "points"})
    os.makedirs(MEDIA, exist_ok=True)
    cols = ["player_id", "player", "pos", "season", "team", "games", "pass_yds", "pass_td", "rush_yds", "rush_td", "rec",
            "rec_yds", "rec_td", "points", "pos_rank", "replacement", "vorp", "season_rank", "source"]
    sea[cols].round(2).sort_values(["season", "season_rank"]).to_csv(os.path.join(MEDIA, "seasons.csv"), index=False)
    rep.round(2).to_csv(os.path.join(MEDIA, "replacement.csv"), index=False)
    weekly.round(3).to_csv(os.path.join(MEDIA, "weekly.csv"), index=False)
    print(rep.pivot(index="season", columns="pos", values="replacement").round(0).astype(int).to_string())
    print(sea.nlargest(25, "vorp")[["season", "player", "pos", "team", "games", "points", "replacement", "vorp"]].round(1).to_string())
    return sea, rep


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        build()
