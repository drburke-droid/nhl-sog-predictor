"""Single-game data for "The 10 Biggest Single-Day Outliers": each player's best single regular-season game in
each season, for the team-sport categories of the greatest-seasons project. Athletics and swimming need nothing
new (a season best is already one performance on one day) and are taken from followup/data/v2.

    python build_data.py nba | nhl | mlb | nfl     # -> ../cache/singlegame/<league>_bests.csv
    python build_data.py merge                     # -> data/sg_categories.csv, data/sg_season_lists.csv
    python build_data.py check                     # the record games every source must reproduce

Sources (all fetched politely: sequential, a pause between requests, descriptive User-Agent, robots.txt allows):
  NBA  Eoin Moore, "Historical NBA Data and Player Box Scores" (CC0), as mirrored on Hugging Face
       (datasets/CoachCheezy/nba-stats, PlayerStatistics.csv). 1946-47 to 4 Mar 2025.
  NHL  Hugging Face datasets/RentoSaijo/NHL_DB: game-centre play-by-play (every goal with scorer and assists)
       and rosters, 1917-18 on. Derived from NHL.com game data.
  MLB  Retrosheet via the Chadwick Bureau's retrosplits daybyday files (GitHub), 1898 on. Names from the
       Chadwick register. The Retrosheet notice applies.
  NFL  nflverse weekly player stats (GitHub releases), 1999 on. Earlier seasons have no openly downloadable
       game-level source (Pro-Football-Reference restricts automated access), so football starts in 1999.
"""
import csv, io, json, os, sys, time, glob, urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
CACHE = os.path.join(REC, "cache", "singlegame")
DATA = os.path.join(HERE, "data")
UA = "OutliersFilm/1.0 (https://github.com/drburke-droid/nhl-sog-predictor; single-game research)"
_last = [0.0]


def fetch(url, path=None, pause=1.5, timeout=600):
    """GET with a pause between requests; cached to path when given"""
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for attempt in range(4):
        wait = pause - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            _last[0] = time.time()
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path + ".tmp", "wb").write(data)
                os.replace(path + ".tmp", path)
                return path
            return data
        except urllib.error.HTTPError as e:
            _last[0] = time.time()
            if e.code == 404:
                return None
            if attempt == 3:
                raise
            ra = e.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.isdigit() else 30 * (attempt + 1))


def write_bests(league, rows):
    """rows: dicts with category, season, athlete_id, name, value, date, detail"""
    p = os.path.join(CACHE, f"{league}_bests.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "season", "athlete_id", "name", "value", "date", "detail"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["category"], r["season"], -r["value"])))
    print(league, len(rows), "player-season bests ->", p, flush=True)


def keep_best(best, key, value, info):
    """best single game per (category, season, athlete); ties keep the earliest game"""
    if value is None or value != value:
        return
    cur = best.get(key)
    if cur is None or value > cur[0] or (value == cur[0] and info["date"] < cur[1]["date"]):
        best[key] = (value, info)


# ------------------------------------------------------------------ NBA
def nba():
    src = fetch("https://huggingface.co/datasets/CoachCheezy/nba-stats/resolve/main/PlayerStatistics.csv",
                os.path.join(CACHE, "nba", "PlayerStatistics.csv"))
    cols = {"sg_nba_pts": "points", "sg_nba_reb": "reboundsTotal", "sg_nba_ast": "assists",
            "sg_nba_3pm": "threePointersMade"}
    best, games = {}, defaultdict(set)
    for r in csv.DictReader(open(src, encoding="utf-8")):
        if r["gameType"] not in ("Regular Season", "NBA Emirates Cup", "NBA Cup"):   # Cup games count as regular season
            continue
        if r["gameLabel"] in ("Emirates NBA Cup Championship", "NBA Cup Championship") or "Championship" in (r["gameSubLabel"] or ""):
            continue                                                               # the Cup final does not
        d = r["gameDate"][:10]
        y, m = int(d[:4]), int(d[5:7])
        season = y + 1 if m >= 8 else y
        games[season].add(r["gameId"])
        name = f"{r['firstName']} {r['lastName']}".strip()
        for cat, col in cols.items():
            v = r[col]
            if v in ("", None):
                continue
            keep_best(best, (cat, season, "nba:" + r["personId"]), float(v),
                      {"name": name, "date": d, "detail": f"{r['playerteamName']} vs {r['opponentteamName']}"})
    rows = [{"category": c, "season": s, "athlete_id": a, "name": i["name"], "value": v, "date": i["date"],
             "detail": i["detail"]} for (c, s, a), (v, i) in best.items()]
    write_bests("nba", rows)
    json.dump({s: len(g) for s, g in sorted(games.items())}, open(os.path.join(CACHE, "nba_games_per_season.json"), "w"))


# ------------------------------------------------------------------ NHL
def nhl_seasons():
    return [f"{y}{y + 1}" for y in range(1917, 2027)]


def nhl():
    import pandas as pd
    base = "https://huggingface.co/datasets/RentoSaijo/NHL_DB/resolve/main/data/game"
    best, games = {}, {}
    for sid in nhl_seasons():
        pp = fetch(f"{base}/pbps/gc/NHL_PBPS_GC_Raw_{sid}.parquet", os.path.join(CACHE, "nhl_db", f"NHL_PBPS_GC_Raw_{sid}.parquet"))
        rp = fetch(f"{base}/rosters/NHL_ROSTERS_{sid}.parquet", os.path.join(CACHE, "nhl_db", f"NHL_ROSTERS_{sid}.parquet"))
        if not pp or not rp:
            print("  no NHL file for", sid, flush=True)
            continue
        pbp = pd.read_parquet(pp)
        ro = pd.read_parquet(rp)
        fn = lambda v: (v.get("default") if isinstance(v, dict) else v) or ""
        names = {int(r.playerId): f"{fn(r.playerFirstName)} {fn(r.playerLastName)}".strip() for r in ro.itertuples()}
        pbp = pbp[pbp["gameId"].astype(str).str[4:6] == "02"]                 # regular season
        season = int(sid[4:])
        games[season] = int(pbp["gameId"].nunique())
        g = pbp[pbp["typeDescKey"] == "goal"]
        tally = defaultdict(lambda: [0, 0])
        for col, k in (("details.scoringPlayerId", 0), ("details.assist1PlayerId", 1), ("details.assist2PlayerId", 1)):
            if col not in g:
                continue
            for gid, pid in zip(g["gameId"], g[col]):
                if pid == pid and pid is not None:
                    tally[(int(gid), int(pid))][k] += 1
        # game dates are not in the play-by-play: the game id orders games within a season
        for (gid, pid), (gl, a) in tally.items():
            info = {"name": names.get(pid, str(pid)), "date": f"{season}:{gid}", "detail": str(gid)}
            for cat, v in (("sg_nhl_goals", gl), ("sg_nhl_assists", a), ("sg_nhl_points", gl + a)):
                keep_best(best, (cat, season, f"nhl:{pid}"), float(v), info)
        print("  NHL", sid, games[season], "games", len(g), "goals", flush=True)
    rows = [{"category": c, "season": s, "athlete_id": a, "name": i["name"], "value": v, "date": i["date"],
             "detail": i["detail"]} for (c, s, a), (v, i) in best.items()]
    write_bests("nhl", rows)
    json.dump(games, open(os.path.join(CACHE, "nhl_games_per_season.json"), "w"))


# ------------------------------------------------------------------ MLB
def mlb_names():
    names = {}
    for p in sorted(glob.glob(os.path.join(REC, "cache", "register", "people-*.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("key_retro"):
                names[r["key_retro"]] = f"{r.get('name_first', '')} {r.get('name_last', '')}".strip()
    return names


def mlb():
    names = mlb_names()
    cols = {"sg_mlb_h": "B_H", "sg_mlb_hr": "B_HR", "sg_mlb_rbi": "B_RBI", "sg_mlb_sb": "B_SB", "sg_mlb_so": "P_SO"}
    rank = {"evt": 0, "box": 1, "ded": 2}
    best, games = {}, {}
    for y in range(1898, 2027):
        url = f"https://raw.githubusercontent.com/chadwickbureau/retrosplits/master/daybyday/playing-{y}.csv"
        raw = fetch(url, pause=2.0, timeout=900)
        if raw is None:
            continue
        seen = {}
        for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
            if r.get("season.phase", "R") != "R":
                continue
            key = (r["game.key"], r["person.key"])
            src = rank.get(r.get("game.source", "evt"), 3)
            if key in seen and seen[key][0] <= src:
                continue
            seen[key] = (src, r)
        games[y] = len({k[0] for k in seen})
        for (gkey, pid), (_, r) in seen.items():
            info = {"name": names.get(pid, pid), "date": r["game.date"], "detail": gkey}
            for cat, col in cols.items():
                v = r.get(col)
                if v not in ("", None):
                    try:
                        keep_best(best, (cat, y, "retro:" + pid), float(v), info)
                    except ValueError:
                        pass
        print("  MLB", y, games[y], "games", flush=True)
    rows = [{"category": c, "season": s, "athlete_id": a, "name": i["name"], "value": v, "date": i["date"],
             "detail": i["detail"]} for (c, s, a), (v, i) in best.items() if v > 0]
    write_bests("mlb", rows)
    json.dump(games, open(os.path.join(CACHE, "mlb_games_per_season.json"), "w"))


# ------------------------------------------------------------------ NFL
def nfl():
    base = "https://github.com/nflverse/nflverse-data/releases/download/player_stats"
    best = {}
    for name, cols in (("player_stats.csv", {"sg_nfl_pass_yds": "passing_yards", "sg_nfl_pass_td": "passing_tds",
                                             "sg_nfl_rush_yds": "rushing_yards", "sg_nfl_rec_yds": "receiving_yards"}),
                       ("player_stats_def.csv", {"sg_nfl_int": "def_interceptions", "sg_nfl_sacks": "def_sacks"})):
        p = fetch(f"{base}/{name}", os.path.join(CACHE, "nflverse_weekly", name))
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("season_type", "REG") != "REG":
                continue
            s = int(r["season"])
            info = {"name": r.get("player_display_name") or r.get("player_name"), "date": f"{s}-w{int(r['week']):02d}",
                    "detail": f"{r.get('recent_team') or r.get('team', '')} vs {r.get('opponent_team', '')}"}
            for cat, col in cols.items():
                v = r.get(col)
                if v not in ("", None, "NA"):
                    keep_best(best, (cat, s, "gsis:" + r["player_id"]), float(v), info)
    rows = [{"category": c, "season": s, "athlete_id": a, "name": i["name"], "value": v, "date": i["date"],
             "detail": i["detail"]} for (c, s, a), (v, i) in best.items() if v > 0]
    write_bests("nfl", rows)


if __name__ == "__main__" and sys.argv[1] in ("nba", "nhl", "mlb", "nfl"):
    {"nba": nba, "nhl": nhl, "mlb": mlb, "nfl": nfl}[sys.argv[1]]()


# ------------------------------------------------------------------ merge into the model's input format
SG_CATS = {  # category: (sport, title, counted from, short or incomplete seasons left out)
    "sg_nba_pts": ("NBA", "Points in a game", 1956, "1999 2012 2020 2025 2026"),
    "sg_nba_reb": ("NBA", "Rebounds in a game", 1956, "1999 2012 2020 2025 2026"),
    "sg_nba_ast": ("NBA", "Assists in a game", 1956, "1999 2012 2020 2025 2026"),
    "sg_nba_3pm": ("NBA", "Three-pointers in a game", 1980, "1999 2012 2020 2025 2026"),
    "sg_nhl_goals": ("NHL", "Goals in a game", 1918, "1995 2013 2020 2021 2027"),
    "sg_nhl_assists": ("NHL", "Assists in a game", 1918, "1995 2013 2020 2021 2027"),
    "sg_nhl_points": ("NHL", "Points in a game", 1918, "1995 2013 2020 2021 2027"),
    "sg_mlb_h": ("MLB", "Hits in a game", 1901, "1918 1981 1994 2020"),
    "sg_mlb_hr": ("MLB", "Home runs in a game", 1901, "1918 1981 1994 2020"),
    "sg_mlb_rbi": ("MLB", "Runs batted in, one game", 1901, "1918 1981 1994 2020"),
    "sg_mlb_sb": ("MLB", "Stolen bases in a game", 1901, "1918 1981 1994 2020"),
    "sg_mlb_so": ("MLB", "Strikeouts by a pitcher, one game", 1901, "1918 1981 1994 2020"),
    "sg_nfl_pass_yds": ("NFL", "Passing yards in a game", 1999, "2026"),
    "sg_nfl_pass_td": ("NFL", "Touchdown passes in a game", 1999, "2026"),
    "sg_nfl_rush_yds": ("NFL", "Rushing yards in a game", 1999, "2026"),
    "sg_nfl_rec_yds": ("NFL", "Receiving yards in a game", 1999, "2026"),
    "sg_nfl_int": ("NFL", "Interceptions in a game", 1999, "2026"),
    "sg_nfl_sacks": ("NFL", "Sacks in a game", 1999, "2026"),
}
CAT_FIELDS = ["category", "sport", "title", "performance", "higher_raw_is_better", "is_total_scaled_by_schedule",
              "first_season", "last_season", "performances_count_from", "short_seasons_left_out", "level_shift_seasons",
              "cutoff_raw", "cutoff_performance"]


def merge():
    os.makedirs(DATA, exist_ok=True)
    lists = defaultdict(list)
    for league in ("nba", "nhl", "mlb", "nfl"):
        p = os.path.join(CACHE, f"{league}_bests.csv")
        if not os.path.exists(p):
            print("missing", p)
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["category"] in SG_CATS and float(r["value"]) > 0:
                lists[(r["category"], int(r["season"]))].append(r)
    cats, rows = [], []
    for cat, (sport, title, start, short) in SG_CATS.items():
        seasons = sorted(s for c, s in lists if c == cat and s >= start)   # incomplete early box scores never enter
        if not seasons:
            continue
        cats.append({"category": cat, "sport": sport, "title": title, "performance": "the stat itself (one game)",
                     "higher_raw_is_better": "True", "is_total_scaled_by_schedule": "False", "first_season": start,
                     "last_season": seasons[-1], "performances_count_from": start, "short_seasons_left_out": short,
                     "level_shift_seasons": "", "cutoff_raw": "", "cutoff_performance": ""})
        for s in seasons:
            lst = sorted(lists[(cat, s)], key=lambda r: (-float(r["value"]), r["date"]))[:10]
            for i, r in enumerate(lst):
                rows.append({"category": cat, "season": s, "rank": i + 1, "athlete_id": r["athlete_id"],
                             "name_display": r["name"], "performance": float(r["value"]), "raw_value": r["value"],
                             "date": r["date"], "detail": r["detail"]})
    with open(os.path.join(DATA, "sg_categories.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAT_FIELDS)
        w.writeheader()
        w.writerows(cats)
    with open(os.path.join(DATA, "sg_season_lists.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(cats), "categories,", len(rows), "season-list rows")


CHECKS = [  # (category, season, name fragment, value): record games every build must reproduce
    ("sg_nba_pts", 1962, "Chamberlain", 100), ("sg_nba_pts", 2006, "Bryant", 81), ("sg_nba_reb", 1961, "Chamberlain", 55),
    ("sg_nba_ast", 1991, "Skiles", 30), ("sg_nhl_goals", 1920, "Malone", 7), ("sg_nhl_points", 1976, "Sittler", 10),
    ("sg_mlb_rbi", 1993, "Whiten", 12), ("sg_mlb_rbi", 1924, "Bottomley", 12), ("sg_mlb_hr", 1932, "Gehrig", 4),
    ("sg_mlb_h", 1975, "Stennett", 7), ("sg_mlb_so", 1998, "Wood", 20), ("sg_mlb_so", 1986, "Clemens", 20),
    ("sg_nfl_rec_yds", 2013, "Johnson", 329), ("sg_nfl_rush_yds", 2007, "Peterson", 296),
]


def check():
    rows = list(csv.DictReader(open(os.path.join(DATA, "sg_season_lists.csv"), encoding="utf-8")))
    ok = 0
    for cat, s, nm, v in CHECKS:
        hit = [r for r in rows if r["category"] == cat and int(r["season"]) == s and nm in r["name_display"]
               and abs(float(r["performance"]) - v) < 1e-9]
        print(("OK  " if hit else "FAIL"), cat, s, nm, v, (hit[0]["date"] if hit else ""))
        ok += bool(hit)
    print(f"{ok}/{len(CHECKS)} record games reproduced")


if __name__ == "__main__" and sys.argv[1] in ("merge", "check"):
    {"merge": merge, "check": check}[sys.argv[1]]()
