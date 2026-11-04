"""Single games in team sports, adjusted for length: each player's best game per season in every team category (NBA,
NHL, MLB, NFL), chosen on the value scaled to a regulation-length game, for "The biggest single-game outliers in team
sports". Extra innings and overtime give more chances, so:

  MLB batting (hits, home runs, RBI, steals)  x 9 / max(9, innings the team batted)     (Burnett's 9 hits in 18 -> 4.5)
  MLB pitcher strikeouts                      x 9 / max(9, innings that pitcher pitched) (Cheney's 21 in 16 -> 11.8)
  NBA (points, rebounds, assists, threes)     x 48 / game minutes (from the players' minutes; 48 + 5 per overtime)
  NHL (goals, assists, points)                x 60 / (60 + overtime minutes played); shootout goals are not goals
  NFL (yards, touchdowns, interceptions, sacks) x 60 / (60 + overtime minutes played, from play-by-play)

Shortened games are never scaled up. Nothing else changes: the same sources as build_data.py, the same seasons and
short seasons. Output keeps the raw value (what happened) next to the adjusted one (what is ranked).

    python team_games.py mlb | nba | nhl | nfl    # -> ../cache/singlegame/<league>_bests_len.csv
    python team_games.py merge                    # -> data/tg_categories.csv, data/tg_season_lists.csv (top 15)
    python team_games.py league                   # league per-team-game rates the data can give (NHL, NFL, MLB)
    python team_games.py check                    # record games, with their length factors
    python team_games.py nflold [offline]         # NFL 1932-98 + AFL from Pro Football Archives (offline: cached pages only)
    python team_games.py pfateams                 # the season film's PFA team pages (-> ../cache/pfa, for its NFL rates)
"""
import csv, glob, io, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_data as B                                      # noqa: E402  (fetch, sources, categories, short seasons)

CACHE = B.CACHE
DATA = B.DATA
CTX = os.path.join(REC, "followup", "data", "context")
FIELDS = ["category", "season", "athlete_id", "name", "value", "raw", "factor", "length", "date", "detail"]


def keep(best, key, adj, raw, factor, length, info):
    """best adjusted game per (category, season, athlete); ties keep the earliest"""
    if raw is None or raw != raw or raw <= 0:
        return
    cur = best.get(key)
    if cur is None or adj > cur[0] or (adj == cur[0] and info["date"] < cur[4]["date"]):
        best[key] = (adj, raw, factor, length, info)


def write(league, best):
    p = os.path.join(CACHE, f"{league}_bests_len.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (c, s, a), (adj, raw, fac, ln, i) in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1][0])):
            w.writerow({"category": c, "season": s, "athlete_id": a, "name": i["name"], "value": round(adj, 4),
                        "raw": raw, "factor": round(fac, 4), "length": ln, "date": i["date"], "detail": i["detail"]})
    print(league, len(best), "player-season bests ->", p, flush=True)


# ------------------------------------------------------------------ MLB
def mlb_raw(y):
    """one season of Retrosheet daybyday lines, cached (build_data.py reads them uncached)"""
    p = B.fetch(f"https://raw.githubusercontent.com/chadwickbureau/retrosplits/master/daybyday/playing-{y}.csv",
                os.path.join(CACHE, "retrosplits", f"playing-{y}.csv"), pause=2.0, timeout=900)
    return p


def mlb():
    names = B.mlb_names()
    bat = {"sg_mlb_h": "B_H", "sg_mlb_hr": "B_HR", "sg_mlb_rbi": "B_RBI", "sg_mlb_sb": "B_SB"}
    rank = {"evt": 0, "box": 1, "ded": 2}
    best, rates = {}, {}
    for y in range(1898, 2027):
        p = mlb_raw(y)
        if p is None:
            continue
        seen = {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("season.phase", "R") != "R":
                continue
            key = (r["game.key"], r["person.key"])
            src = rank.get(r.get("game.source", "evt"), 3)
            if key in seen and seen[key][0] <= src:
                continue
            seen[key] = (src, r)
        outs = defaultdict(float)                              # outs recorded by each team's pitchers
        for (g, _), (_, r) in seen.items():
            if r.get("P_OUT") not in ("", None):
                outs[(g, r["team.key"])] += float(r["P_OUT"])
        tot = defaultdict(float)
        games = {g for g, _ in seen}
        for (g, pid), (_, r) in seen.items():
            info = {"name": names.get(pid, pid), "date": r["game.date"], "detail": g}
            opp = outs.get((g, r["opponent.key"]))             # the team batted while the opponent recorded outs
            inn = opp / 3 if opp else None
            fac = 9 / max(9.0, inn) if inn else 1.0
            for cat, col in bat.items():
                v = r.get(col)
                if v not in ("", None):
                    v = float(v)
                    tot[col] += v
                    keep(best, (cat, y, "retro:" + pid), v * fac, v, fac, f"{inn:.1f} innings batted" if inn else "", info)
            if r.get("P_SO") not in ("", None):
                v = float(r["P_SO"])
                tot["P_SO"] += v
                ip = float(r["P_OUT"]) / 3 if r.get("P_OUT") not in ("", None) else None
                fac = 9 / max(9.0, ip) if ip else 1.0
                keep(best, ("sg_mlb_so", y, "retro:" + pid), v * fac, v, fac, f"{ip:.1f} innings pitched" if ip else "", info)
        rates[y] = {"games": len(games), **{k: tot[k] / (2 * len(games)) for k in tot}}
        print("  MLB", y, len(games), "games", flush=True)
    write("mlb", best)
    json.dump(rates, open(os.path.join(CACHE, "mlb_rates_len.json"), "w"))


# ------------------------------------------------------------------ NBA
def nba():
    src = B.fetch("https://huggingface.co/datasets/CoachCheezy/nba-stats/resolve/main/PlayerStatistics.csv",
                  os.path.join(CACHE, "nba", "PlayerStatistics.csv"))
    cols = {"sg_nba_pts": "points", "sg_nba_reb": "reboundsTotal", "sg_nba_ast": "assists",
            "sg_nba_3pm": "threePointersMade"}
    rows, mins, most = [], defaultdict(float), defaultdict(float)
    for r in csv.DictReader(open(src, encoding="utf-8")):
        if r["gameType"] not in ("Regular Season", "NBA Emirates Cup", "NBA Cup"):
            continue
        if r["gameLabel"] in ("Emirates NBA Cup Championship", "NBA Cup Championship") or "Championship" in (r["gameSubLabel"] or ""):
            continue
        rows.append({k: r[k] for k in ("firstName", "lastName", "personId", "gameId", "gameDate", "playerteamName",
                                       "opponentteamName", "numMinutes", *cols.values())})
        try:
            mins[(r["gameId"], r["playerteamName"])] += float(r["numMinutes"] or 0)
            most[r["gameId"]] = max(most[r["gameId"]], float(r["numMinutes"] or 0))
        except ValueError:
            pass
    best = {}
    for r in rows:
        d = r["gameDate"][:10]
        y, m = int(d[:4]), int(d[5:7])
        season = y + 1 if m >= 8 else y
        gm = mins.get((r["gameId"], r["playerteamName"]), 0) / 5          # five players on the floor
        top = most.get(r["gameId"], 0)                                      # one player's minutes, when the rest are missing
        if gm >= 46:
            ot = round((gm - 48) / 5) if gm > 50 else 0                     # minutes logged to the nearest overtime
        elif top > 48:
            ot, gm = -(-int(top - 48) // 5), top                            # e.g. Wilt's 63 minutes: three overtimes
        else:
            ot = 0
        length = 48 + 5 * ot if (gm >= 46 or top > 48) else None            # no minutes at all: no adjustment
        fac = 48 / length if length else 1.0
        info = {"name": f"{r['firstName']} {r['lastName']}".strip(), "date": d,
                "detail": f"{r['playerteamName']} vs {r['opponentteamName']}"}
        for cat, col in cols.items():
            if r[col] not in ("", None):
                v = float(r[col])
                keep(best, (cat, season, "nba:" + r["personId"]), v * fac, v, fac,
                     (f"{ot} overtime" + ("s" if ot > 1 else "") if ot else "regulation") if length else "", info)
    write("nba", best)


# ------------------------------------------------------------------ NHL
def mmss(t):
    try:
        m, s = str(t).split(":")
        return int(m) + int(s) / 60
    except ValueError:
        return 0.0


def nhl():
    import pandas as pd
    base = "https://huggingface.co/datasets/RentoSaijo/NHL_DB/resolve/main/data/game"
    best, rates, so_dropped = {}, {}, 0
    for sid in B.nhl_seasons():
        pp = B.fetch(f"{base}/pbps/gc/NHL_PBPS_GC_Raw_{sid}.parquet", os.path.join(CACHE, "nhl_db", f"NHL_PBPS_GC_Raw_{sid}.parquet"))
        rp = B.fetch(f"{base}/rosters/NHL_ROSTERS_{sid}.parquet", os.path.join(CACHE, "nhl_db", f"NHL_ROSTERS_{sid}.parquet"))
        if not pp or not rp:
            continue
        pbp = pd.read_parquet(pp)
        ro = pd.read_parquet(rp)
        fn = lambda v: (v.get("default") if isinstance(v, dict) else v) or ""
        names = {int(r.playerId): f"{fn(r.playerFirstName)} {fn(r.playerLastName)}".strip() for r in ro.itertuples()}
        pbp = pbp[pbp["gameId"].astype(str).str[4:6] == "02"]                 # regular season
        season = int(sid[4:])
        ptype = pbp.get("periodDescriptor.periodType", pd.Series("REG", index=pbp.index)).fillna("REG")
        pnum = pbp.get("periodDescriptor.number", pd.Series(1, index=pbp.index)).fillna(1)
        ot = (ptype == "OT") | ((ptype == "REG") & (pnum > 3))
        otmin = pbp[ot].assign(t=pbp.loc[ot, "timeInPeriod"].map(mmss)).groupby("gameId")["t"].max().to_dict()
        so = ptype == "SO"
        g = pbp[(pbp["typeDescKey"] == "goal")]
        so_dropped += int((g.index.isin(pbp[so].index)).sum())
        g = g[~g.index.isin(pbp[so].index)]
        n_games = int(pbp["gameId"].nunique())
        tally = defaultdict(lambda: [0, 0])
        n_assists = 0
        for col, k in (("details.scoringPlayerId", 0), ("details.assist1PlayerId", 1), ("details.assist2PlayerId", 1)):
            if col not in g:
                continue
            for gid, pid in zip(g["gameId"], g[col]):
                if pid == pid and pid is not None:
                    tally[(int(gid), int(pid))][k] += 1
                    n_assists += k
        rates[season] = {"games": n_games, "goals_per_team_g": len(g) / (2 * n_games) if n_games else None,
                         "assists_per_team_g": n_assists / (2 * n_games) if n_games else None}
        for (gid, pid), (gl, a) in tally.items():
            om = min(otmin.get(gid, 0.0), 20.0)
            fac = 60 / (60 + om)
            info = {"name": names.get(pid, str(pid)), "date": f"{season}:{gid}", "detail": str(gid)}
            ln = f"{om:.1f} min overtime" if om else "regulation"
            for cat, v in (("sg_nhl_goals", gl), ("sg_nhl_assists", a), ("sg_nhl_points", gl + a)):
                keep(best, (cat, season, f"nhl:{pid}"), v * fac, float(v), fac, ln, info)
        print("  NHL", sid, n_games, "games", len(g), "goals", flush=True)
    print("shootout 'goals' left out:", so_dropped)
    write("nhl", best)
    json.dump(rates, open(os.path.join(CACHE, "nhl_rates_len.json"), "w"))


# ------------------------------------------------------------------ NFL
def nfl_ot():
    """game_id -> overtime minutes played, and (season, week, team) -> game_id, from nflverse play-by-play"""
    import pandas as pd
    sched = list(csv.DictReader(open(B.fetch("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
                                              os.path.join(CACHE, "nflverse_pbp", "games.csv")), encoding="utf-8")))
    gid = {}
    for r in sched:
        for t in (r["home_team"], r["away_team"]):
            gid[(int(r["season"]), int(r["week"]), t)] = r["game_id"]
    otm = {}
    for y in range(1999, 2027):
        p = B.fetch(f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet",
                    os.path.join(CACHE, "nflverse_pbp", f"play_by_play_{y}.parquet"), pause=2.0)
        if not p:
            continue
        d = pd.read_parquet(p, columns=["game_id", "qtr", "quarter_seconds_remaining", "season_type"])
        d = d[(d["qtr"] >= 5) & (d["season_type"] == "REG")]
        qlen = 10.0 if y >= 2017 else 15.0
        for g_, s in d.groupby("game_id")["quarter_seconds_remaining"].min().items():
            otm[g_] = qlen - float(s) / 60
    return gid, otm


MOVED = {"LV": ["OAK"], "LAC": ["SD"], "LA": ["STL", "LAR"], "LAR": ["STL", "LA"]}   # weekly stats use today's codes


def nfl():
    gid, otm = nfl_ot()
    best = {}
    for name, cols in (("player_stats.csv", {"sg_nfl_pass_yds": "passing_yards", "sg_nfl_pass_td": "passing_tds",
                                             "sg_nfl_rush_yds": "rushing_yards", "sg_nfl_rec_yds": "receiving_yards"}),
                       ("player_stats_def.csv", {"sg_nfl_int": "def_interceptions", "sg_nfl_sacks": "def_sacks"})):
        p = B.fetch(f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/{name}",
                    os.path.join(CACHE, "nflverse_weekly", name))
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("season_type", "REG") != "REG":
                continue
            s, wk = int(r["season"]), int(r["week"])
            team = r.get("recent_team") or r.get("team", "")
            g = next((gid[(s, wk, t)] for t in [team] + MOVED.get(team, []) if (s, wk, t) in gid), None)
            om = otm.get(g, 0.0) if g else 0.0
            fac = 60 / (60 + om)
            info = {"name": r.get("player_display_name") or r.get("player_name"), "date": f"{s}-w{wk:02d}",
                    "detail": f"{team} vs {r.get('opponent_team', '')}" + (f" ({g})" if g else "")}
            for cat, col in cols.items():
                v = r.get(col)
                if v not in ("", None, "NA"):
                    v = float(v)
                    keep(best, (cat, s, "gsis:" + r["player_id"]), v * fac, v, fac,
                         f"{om:.1f} min overtime" if om else ("regulation" if g else ""), info)
    write("nfl", best)


# ------------------------------------------------------------------ NFL before 1999: Pro Football Archives box scores
PFA = "https://www.profootballarchives.com"
PFA_SEASONS = [(y, "nfl") for y in range(1932, 1999)] + [(y, "afl") for y in range(1960, 1970)]
PFA_COLS = {"RUSHING": {"sg_nfl_rush_yds": "YDS"}, "PASSING": {"sg_nfl_pass_yds": "YDS", "sg_nfl_pass_td": "TD"},
            "RECEIVING": {"sg_nfl_rec_yds": "YDS"}, "INTERCEPTIONS": {"sg_nfl_int": "NO"}, "SACKS": {"sg_nfl_sacks": "NO"}}


OFFLINE = [False]          # True: read only what is already cached (python team_games.py nflold offline)


def pfa_get(url, path, pause=3.0):
    """a polite fetch for Pro Football Archives: cached, `pause` seconds apart, and when the host answers with its
    "One moment, please" slow-down page, nothing is cached and we wait (1, 2, 4 ... minutes) before asking again"""
    import time, urllib.request, urllib.error
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if OFFLINE[0]:
        return None
    for attempt in range(8):
        wait = pause - (time.time() - B._last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": B.UA}), timeout=120) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            B._last[0] = time.time()
            if e.code == 404:
                return None
            data = b""
        except (urllib.error.URLError, TimeoutError):
            B._last[0] = time.time()
            data = b""
        B._last[0] = time.time()
        if data and b"One moment, please" not in data[:4000]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path + ".tmp", "wb").write(data)
            os.replace(path + ".tmp", path)
            return path
        print(f"    host asked us to slow down ({url.rsplit('/', 1)[-1]}); waiting {60 * 2 ** min(attempt, 4)} s", flush=True)
        time.sleep(60 * 2 ** min(attempt, 4))
    raise RuntimeError(f"gave up on {url}")


def pfa_games(y, lg):
    """regular-season games of one season: [(game page path, date, visitor, home)] (the list's rows before its first
    playoff heading)"""
    from bs4 import BeautifulSoup
    p = pfa_get(f"{PFA}/{y}{lg}-boxscores.html", os.path.join(CACHE, "pfa_box", f"{y}{lg}-boxscores.html"))
    if not p:
        return []
    out, started = [], False
    for r in BeautifulSoup(open(p, encoding="utf-8", errors="replace").read(), "lxml").find_all("tr"):
        cells = r.find_all(["td", "th"])
        if len(cells) == 1:
            t = cells[0].get_text(" ", strip=True)
            if "Boxscores" in t or "Season" in t:          # the title and the season links
                started = True
                continue
            if started and out:
                break                                          # the first heading after the season: playoffs
            continue
        a = r.find("a", href=True)
        if started and a and "boxscores" in a["href"]:
            c = [x.get_text(" ", strip=True) for x in cells]
            out.append((a["href"].lstrip("/"), c[0], c[1], c[3]))
    return out


def pfa_parse(path):
    """one box score -> (overtime?, tied?, {(section, player id): (name, team, {column: value})})"""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(open(path, encoding="utf-8", errors="replace").read(), "lxml")
    ot, tied, lines = False, False, {}
    for t in s.find_all("table"):
        rows = t.find_all("tr")
        if not rows:
            continue
        head = [c.get_text(" ", strip=True) for c in rows[0].find_all(["td", "th"])]
        if not head:
            continue
        if head[0] == "Score By Quarters":
            ot = any(h.upper().startswith("OT") for h in head[1:-1])
            finals = [r.find_all(["td", "th"])[-1].get_text(strip=True) for r in rows[1:3]]
            tied = len(finals) == 2 and finals[0] == finals[1]
            continue
        if head[0] not in PFA_COLS:
            continue
        team = ""
        for r in rows[1:]:
            cells = r.find_all(["td", "th"])
            name = cells[0].get_text(" ", strip=True) if cells else ""
            a = cells[0].find("a", href=True) if cells else None
            if len(cells) <= 2 and not a:
                team = name                                    # a team-name row
                continue
            if not a or name == "Team Totals":
                continue
            vals = {h: (cells[i].get_text(strip=True) if i < len(cells) else "") for i, h in enumerate(head) if i}
            lines[(head[0], a["href"])] = (name, team, vals)
    return ot, tied, lines


def nflold(pause=3.0, offline=False):
    """every player's best regular-season game per season, 1932-1998 (NFL) and 1960-69 (AFL); overtime (from 1974)
    scaled as 60 / (60 + 7.5), or 75 minutes for a tie, since box scores don't give the winning score's time. Also
    writes each season's totals and how many games have each table (nflold_totals.json), to set where each stat starts
    and the league rates."""
    OFFLINE[0] = offline
    best = {}
    tot = defaultdict(lambda: defaultdict(float))
    games, cover, teams = defaultdict(int), defaultdict(lambda: defaultdict(int)), defaultdict(set)
    for y, lg in PFA_SEASONS:
        gl = pfa_games(y, lg)
        n = 0
        for path, date, vis, home in gl:
            p = pfa_get(f"{PFA}/{path}", os.path.join(CACHE, "pfa_box", path.replace("/", "_")), pause=pause)
            if not p:
                continue
            ot, tied, lines = pfa_parse(p)
            om = (15.0 if tied else 7.5) if ot else 0.0
            fac = 60 / (60 + om)
            m_, d_, y_ = date.split("/")
            iso = f"{y_}-{int(m_):02d}-{int(d_):02d}"
            games[y] += 1
            teams[y] |= {vis, home}
            n += 1
            seen = set()
            for (sec, pid), (name, team, vals) in lines.items():
                for cat, col in PFA_COLS[sec].items():
                    v = vals.get(col, "")
                    try:
                        v = float(v)
                    except ValueError:
                        continue
                    seen.add(cat)
                    tot[y][cat] += v
                    info = {"name": name, "date": iso, "detail": f"{team} ({vis} at {home}; {lg.upper()})"}
                    keep(best, (cat, y, "pfa:" + pid.rsplit("/", 1)[-1].replace(".html", "")), v * fac, v, fac,
                         ("overtime (length not recorded)" if ot else "regulation"), info)
            for cat in seen:
                cover[y][cat] += 1
        print("  PFA", y, lg, len(gl), "games", f"({n} read)" if n < len(gl) else "", flush=True)
    write("nflold", best)
    json.dump({"games": games, "totals": tot, "games_with": cover, "teams": {y: len(t) for y, t in teams.items()}},
              open(os.path.join(CACHE, "nflold_totals.json"), "w"), indent=0)


# ------------------------------------------------------------------ merge, league rates, checks
# Where the old NFL box scores start to count. Before 1950 they are partial reconstructions from newspaper accounts:
# a team's receiving yards match its passing yards in only 29-79% of team-games (1940-49), against 97-100% from 1950.
# Individual interceptions are complete from 1950 too. Sacks were not an official stat before 1982; 1982 itself (nine
# games, a strike) is left out like the other short seasons.
TG_CATS = dict(B.SG_CATS)
for _c in ("sg_nfl_pass_yds", "sg_nfl_pass_td", "sg_nfl_rush_yds", "sg_nfl_rec_yds", "sg_nfl_int"):
    TG_CATS[_c] = TG_CATS[_c][:2] + (1950, "1982 2026")
TG_CATS["sg_nfl_sacks"] = TG_CATS["sg_nfl_sacks"][:2] + (1983, "1982 2026")


def norm_name(n):
    import re, unicodedata
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[.'`]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return " ".join(n.replace("-", " ").split())


def nfl_ids(rows_old, rows_new, lo=1994, hi=2004):
    """Pro Football Archives id -> nflverse (gsis) id for players on both sides of 1998/99, matched by name when the
    name is unique on each side, so a player's own games stay out of his curve across the change of source"""
    def by_name(rows, y0, y1):
        m = defaultdict(set)
        for r in rows:
            if y0 <= int(r["season"]) <= y1:
                m[norm_name(r["name"])].add(r["athlete_id"])
        return m
    old, new = by_name(rows_old, lo, 1998), by_name(rows_new, 1999, hi)
    return {next(iter(old[n])): next(iter(new[n])) for n in old if len(old[n]) == 1 and len(new.get(n, ())) == 1}


def merge(depth=15):
    os.makedirs(DATA, exist_ok=True)
    lists = defaultdict(list)
    src = {}
    for league in ("nba", "nhl", "mlb", "nfl", "nflold"):
        p = os.path.join(CACHE, f"{league}_bests_len.csv")
        if not os.path.exists(p):
            print("missing", p)
            continue
        src[league] = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["category"] in TG_CATS]
    if "nflold" in src and "nfl" in src:
        ids = nfl_ids(src["nflold"], src["nfl"])
        for r in src["nflold"]:
            r["athlete_id"] = ids.get(r["athlete_id"], r["athlete_id"])
        print(len(ids), "players matched across 1998/99")
    for league, rs in src.items():
        for r in rs:
            lists[(r["category"], int(r["season"]))].append(r)
    cats, rows = [], []
    for cat, (sport, title, start, short) in TG_CATS.items():
        seasons = sorted(s for c, s in lists if c == cat and s >= start)
        if not seasons:
            continue
        cats.append({"category": cat, "sport": sport, "title": title, "performance": "per regulation-length game",
                     "higher_raw_is_better": "True", "is_total_scaled_by_schedule": "False", "first_season": start,
                     "last_season": seasons[-1], "performances_count_from": start, "short_seasons_left_out": short,
                     "level_shift_seasons": "", "cutoff_raw": "", "cutoff_performance": ""})
        for s in seasons:
            lst = sorted(lists[(cat, s)], key=lambda r: (-float(r["value"]), r["date"]))[:depth]
            for i, r in enumerate(lst):
                rows.append({"category": cat, "season": s, "rank": i + 1, "athlete_id": r["athlete_id"],
                             "name_display": r["name"], "performance": float(r["value"]), "raw_value": r["raw"],
                             "factor": r["factor"], "length": r["length"], "date": r["date"], "detail": r["detail"]})
    with open(os.path.join(DATA, "tg_categories.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=B.CAT_FIELDS)
        w.writeheader()
        w.writerows(cats)
    with open(os.path.join(DATA, "tg_season_lists.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(cats), "categories,", len(rows), "season-list rows")


def league():
    """league per-team-game rates for the stats the Reference league tables lack: the NFL (Pro Football Archives box
    scores 1950-98, the AFL's games counted in 1960-69; nflverse from 1999). The NHL and MLB tables come from Hockey-
    and Baseball-Reference (followup/build/parse_context.py), as in the season film; only if those are missing are
    they computed from the game data here, as a stand-in."""
    os.makedirs(CTX, exist_ok=True)
    have = lambda name, col: os.path.exists(os.path.join(CTX, name)) and col in open(os.path.join(CTX, name)).readline()
    if not have("nhl_league.csv", "assists_per_goal"):
        nh = json.load(open(os.path.join(CACHE, "nhl_rates_len.json")))
        with open(os.path.join(CTX, "nhl_league.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["season", "games", "goals_per_team_g", "assists_per_team_g", "points_per_team_g"])
            for s, r in sorted(nh.items(), key=lambda kv: int(kv[0])):
                if r["goals_per_team_g"]:
                    w.writerow([s, r["games"], round(r["goals_per_team_g"], 4), round(r["assists_per_team_g"], 4),
                                round(r["goals_per_team_g"] + r["assists_per_team_g"], 4)])
    if not have("mlb_league.csv", "era"):
        ml = json.load(open(os.path.join(CACHE, "mlb_rates_len.json")))
        with open(os.path.join(CTX, "mlb_league.csv"), "w", newline="") as f:
            w = csv.writer(f)
            cols = [("so_pitch", "P_SO"), ("h", "B_H"), ("hr", "B_HR"), ("rbi", "B_RBI"), ("sb", "B_SB")]
            w.writerow(["season", "games"] + [k + "_per_team_g" for k, _ in cols])
            for s, r in sorted(ml.items(), key=lambda kv: int(kv[0])):
                w.writerow([s, r["games"]] + [round(r.get(c, 0.0), 5) for _, c in cols])
    tot = defaultdict(lambda: defaultdict(float))
    for name, cols in (("player_stats.csv", {"pass_yds": "passing_yards", "pass_td": "passing_tds",
                                             "rush_yds": "rushing_yards", "rec_yds": "receiving_yards"}),
                       ("player_stats_def.csv", {"int": "def_interceptions", "sacks": "def_sacks"})):
        for r in csv.DictReader(open(os.path.join(CACHE, "nflverse_weekly", name), encoding="utf-8")):
            if r.get("season_type", "REG") != "REG":
                continue
            for k, c in cols.items():
                if r.get(c) not in (None, "", "NA"):
                    tot[int(r["season"])][k] += float(r[c])
    keys = ["pass_yds", "rush_yds", "rec_yds", "pass_td", "int", "sacks"]
    old = json.load(open(os.path.join(CACHE, "nflold_totals.json")))           # 1932-1998, NFL and AFL together
    with open(os.path.join(CTX, "nfl_league.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "teams", "games_per_team"] + [k + "_per_team_g" for k in keys])
        for y in sorted(old["games"], key=int):
            n, tg = old["games"][y], old["totals"][y]
            if int(y) < 1950:
                continue
            full = {k: tg.get("sg_nfl_" + k, 0) for k in keys}
            if int(y) < 1982:
                full["sacks"] = 0                                             # not an official stat before 1982
            w.writerow([y, old["teams"][y], round(2 * n / old["teams"][y], 2)]
                       + [round(full[k] / (2 * n), 4) if full[k] else "" for k in keys])
        for y in sorted(tot):
            n, g = (31 if y <= 2001 else 32), (17 if y >= 2021 else 16)
            w.writerow([y, n, g] + [round(tot[y][k] / (n * g), 4) for k in keys])
    print("league rates written to", CTX)


def pfateams():
    """the Pro Football Archives season and team pages the season film's NFL league rates were summed from (NFL 1932-98,
    AFL 1960-69), into ../cache/pfa where followup/build/parse_context.py nfl reads them; cached, polite, resumable"""
    import re
    for y, lg in [(y, "nfl") for y in range(1932, 1999)] + [(y, "afl") for y in range(1960, 1970)]:
        p = pfa_get(f"{PFA}/{y}{lg}.html", os.path.join(REC, "cache", "pfa", f"{y}{lg}.html"))
        txt = open(p, encoding="utf-8", errors="replace").read() if p else ""
        teams = sorted({f for f, t in re.findall(rf'href="/?({y}{lg}([a-z]+)\.html)"', txt) if t not in ("draft", "awards")})
        miss = [f for f in teams if not pfa_get(f"{PFA}/{f}", os.path.join(REC, "cache", "pfa", f))]
        print(f"  PFA {y} {lg} {len(teams)} team pages {'missing ' + ' '.join(miss) if miss else ''}", flush=True)


CHECKS = [("sg_nba_pts", 1962, "Chamberlain", 100), ("sg_nba_pts", 2006, "Bryant", 81), ("sg_nba_ast", 1991, "Skiles", 30),
          ("sg_nhl_goals", 1920, "Malone", 7), ("sg_nhl_points", 1976, "Sittler", 10), ("sg_mlb_h", 1932, "Burnett", 9),
          ("sg_mlb_h", 1975, "Stennett", 7), ("sg_mlb_hr", 1932, "Gehrig", 4), ("sg_mlb_so", 1998, "Wood", 20),
          ("sg_mlb_so", 1962, "Cheney", 21), ("sg_nfl_rec_yds", 2013, "Johnson", 329), ("sg_nfl_rush_yds", 2007, "Peterson", 296)]


def check():
    rows = list(csv.DictReader(open(os.path.join(DATA, "tg_season_lists.csv"), encoding="utf-8")))
    for cat, s, nm, v in CHECKS:
        hit = [r for r in rows if r["category"] == cat and int(r["season"]) == s and nm in r["name_display"]
               and abs(float(r["raw_value"]) - v) < 1e-9]
        print(("OK  " if hit else "FAIL"), cat, s, nm, v, (f"{hit[0]['date']} {hit[0]['length']} -> {float(hit[0]['performance']):.2f}" if hit else ""))


if __name__ == "__main__":
    if sys.argv[1:] == ["nflold", "offline"]:
        nflold(offline=True)
    else:
        {"mlb": mlb, "nba": nba, "nhl": nhl, "nfl": nfl, "nflold": nflold, "merge": merge, "league": league,
         "check": check, "pfateams": pfateams}[sys.argv[1]]()
