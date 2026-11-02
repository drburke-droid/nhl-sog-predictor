"""Parse the cached league-context pages into plain CSVs under followup/data/context/ (needs ../cache/).

    python followup/build/parse_context.py        # from world_roads/records

Writes (season = the calendar year a season ends; 1961-62 -> 1962):
  nba_league.csv      league per-team-game averages (Basketball-Reference leagues/NBA_stats_per_game.html), the
                      number of teams, and the mean/min/max of Basketball-Reference's *estimated* team paces
                      (season pages, "Advanced" team table). The league table publishes pace only from 1973-74.
  nba_team_pace.csv   every team-season's estimated pace and points per game
  nba_players.csv     every player-team line (per game: G, MP, PTS, TRB, AST, 3P) with the Reference player ID
  nhl_league.csv      teams, games, goals per team-game (Hockey-Reference leagues/stats.html), and assists per
                      goal and per team-game summed from the skater pages
  nhl_skaters.csv     every skater line (GP, G, A, PTS, average TOI where recorded) with the Reference ID
  mlb_league.csv      AL+NL pooled per team-game rates (Baseball-Reference leagues/{AL,NL}/{bat,pitch}.shtml)
  nfl_league.csv      teams, games per team and league per-team-game totals of the six NFL categories (Pro Football
                      Archives team pages to 1998, nflverse from 1999)
  nfl_players.csv     every player-season line in the six categories with a stable ID
"""
import os, sys, re, csv, glob, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from bs4 import BeautifulSoup                                   # noqa: E402
import bigfour as B                                            # noqa: E402

C = os.path.join(ROOT, "cache", "context")
OUT = os.path.join(ROOT, "followup", "data", "context")


def soup(p):
    return BeautifulSoup(open(p, encoding="utf-8").read().replace("<!--", "").replace("-->", ""), "lxml")


def rows(t):
    for tr in t.select("tbody tr"):
        if "thead" in (tr.get("class") or []):
            continue
        d = {c.get("data-stat"): c for c in tr.find_all(["th", "td"])}
        yield d


def txt(d, k):
    c = d.get(k)
    return c.get_text(strip=True) if c is not None else ""


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def end_year(season):                          # "1961-62" -> 1962; "1999-00" -> 2000; "1941" -> 1941
    m = re.match(r"(\d{4})-(\d{2})", season)
    if m:
        y = int(m.group(1)) + 1
        return y
    return int(season[:4]) if season[:4].isdigit() else None


def pid(c):
    a = c.find("a") if c is not None else None
    m = re.search(r"/players/\w/([\w.]+?)\.s?html", a["href"]) if a and a.get("href") else None
    return m.group(1) if m else ""


def write(name, cols, data):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(data)
    print(name, len(data), flush=True)


def nba():
    lg = {}
    for d in rows(soup(os.path.join(C, "nba_league_per_game.html")).find("table", id="stats-Regular-Season")):
        y = end_year(txt(d, "season"))
        if y:
            lg[y] = {"season": y, "lg": txt(d, "lg_id"), "pts_per_team_g": num(txt(d, "pts_per_g")),
                     "fga_per_team_g": num(txt(d, "fga_per_g")), "fta_per_team_g": num(txt(d, "fta_per_g")),
                     "orb_per_team_g": num(txt(d, "orb_per_g")), "tov_per_team_g": num(txt(d, "tov_per_g")),
                     "trb_per_team_g": num(txt(d, "trb_per_g")), "ast_per_team_g": num(txt(d, "ast_per_g")),
                     "fg3_per_team_g": num(txt(d, "fg3_per_g")), "fg3a_per_team_g": num(txt(d, "fg3a_per_g")),
                     "pace_league_table": num(txt(d, "pace"))}
    teams, players = [], []
    for y in range(1947, 2027):
        p = os.path.join(C, f"nba_{y}.html")
        if not os.path.exists(p):
            continue
        s = soup(p)
        adv, pg = s.find("table", id="advanced-team"), s.find("table", id="per_game-team")
        ppg = {txt(d, "team").rstrip("*"): num(txt(d, "pts")) for d in rows(pg)} if pg else {}
        abbr = {}
        for d in (rows(pg) if pg else []):
            a = d.get("team").find("a") if d.get("team") is not None else None
            m = re.search(r"/teams/(\w+)/", a["href"]) if a and a.get("href") else None
            if m:
                abbr[txt(d, "team").rstrip("*")] = m.group(1)
        paces = []
        for d in (rows(adv) if adv else []):
            t = txt(d, "team").rstrip("*")
            if not t or t.startswith("League"):
                continue
            pc = num(txt(d, "pace"))
            teams.append({"season": y, "team": t, "abbr": abbr.get(t, ""), "pace_estimated": pc, "pts_per_g": ppg.get(t)})
            if pc:
                paces.append(pc)
        n = len(ppg) or len([x for x in teams if x["season"] == y])
        if y in lg:
            lg[y].update({"teams": n, "pace_team_mean": round(sum(paces) / len(paces), 2) if paces else None,
                          "pace_team_min": min(paces) if paces else None, "pace_team_max": max(paces) if paces else None})
        q = os.path.join(C, f"nba_{y}_per_game.html")
        if os.path.exists(q):
            t = soup(q).find("table", id="per_game_stats")
            for d in rows(t):
                nm = txt(d, "name_display")
                if not nm or nm == "League Average":
                    continue
                players.append({"season": y, "player_id": pid(d.get("name_display")), "name": nm.rstrip("*"),
                                "team": txt(d, "team_name_abbr"), "g": num(txt(d, "games")),
                                "mp": num(txt(d, "mp_per_g")), "pts": num(txt(d, "pts_per_g")),
                                "trb": num(txt(d, "trb_per_g")), "ast": num(txt(d, "ast_per_g")),
                                "fg3": num(txt(d, "fg3_per_g")), "fga": num(txt(d, "fga_per_g")),
                                "fta": num(txt(d, "fta_per_g"))})
    cols = ["season", "lg", "teams", "pts_per_team_g", "fga_per_team_g", "fta_per_team_g", "orb_per_team_g",
            "tov_per_team_g", "trb_per_team_g", "ast_per_team_g", "fg3_per_team_g", "fg3a_per_team_g",
            "pace_league_table", "pace_team_mean", "pace_team_min", "pace_team_max"]
    write("nba_league.csv", cols, [{k: lg[y].get(k) for k in cols} for y in sorted(lg)])
    have = {(x["season"], x["team"]) for x in teams}         # seasons without an advanced table (1947-50)
    for y in range(1947, 2027):
        p = os.path.join(C, f"nba_{y}.html")
        if os.path.exists(p):
            pg = soup(p).find("table", id="per_game-team")
            for d in (rows(pg) if pg else []):
                t = txt(d, "team").rstrip("*")
                a = d.get("team").find("a") if d.get("team") is not None else None
                m = re.search(r"/teams/(\w+)/", a["href"]) if a and a.get("href") else None
                if t and not t.startswith("League") and (y, t) not in have:
                    teams.append({"season": y, "team": t, "abbr": m.group(1) if m else "", "pace_estimated": None,
                                  "pts_per_g": num(txt(d, "pts"))})
    write("nba_team_pace.csv", ["season", "team", "abbr", "pace_estimated", "pts_per_g"], teams)
    write("nba_players.csv", ["season", "player_id", "name", "team", "g", "mp", "pts", "trb", "ast", "fg3", "fga", "fta"], players)


def nhl():
    lg = {}
    for d in rows(soup(os.path.join(C, "nhl_league_stats.html")).find("table", id="stats")):
        y = end_year(txt(d, "season"))
        if y and y <= 2026:
            lg[y] = {"season": y, "games": num(txt(d, "games")), "goals_per_team_g": num(txt(d, "goals"))}
    sk = []
    for y in range(1918, 2027):
        p = os.path.join(C, f"nhl_{y}_skaters.html")
        if not os.path.exists(p):
            continue
        t = soup(p).find("table", id="player_stats")
        G = A = 0.0
        tms = set()
        for d in rows(t):
            nm = txt(d, "name_display")
            if not nm:
                continue
            tm = txt(d, "team_name_abbr")
            g, a = num(txt(d, "goals")) or 0.0, num(txt(d, "assists")) or 0.0
            if tm and not re.fullmatch(r"\dTM|TOT", tm):
                G, A = G + g, A + a
                tms.add(tm)
            sk.append({"season": y, "player_id": pid(d.get("name_display")), "name": nm.rstrip("*"), "team": tm,
                       "pos": txt(d, "pos"), "gp": num(txt(d, "games")), "g": g, "a": a, "pts": num(txt(d, "points")),
                       "toi_avg": txt(d, "time_on_ice_avg")})
        if y in lg:
            lg[y].update({"teams": len(tms), "skater_goals": G, "skater_assists": A,
                          "assists_per_goal": round(A / G, 4) if G else None})
            if lg[y]["goals_per_team_g"] and lg[y].get("assists_per_goal"):
                lg[y]["assists_per_team_g"] = round(lg[y]["goals_per_team_g"] * lg[y]["assists_per_goal"], 4)
                lg[y]["points_per_team_g"] = round(lg[y]["goals_per_team_g"] * (1 + lg[y]["assists_per_goal"]), 4)
    cols = ["season", "teams", "games", "goals_per_team_g", "assists_per_goal", "assists_per_team_g",
            "points_per_team_g", "skater_goals", "skater_assists"]
    write("nhl_league.csv", cols, [{k: lg[y].get(k) for k in cols} for y in sorted(lg)])
    write("nhl_skaters.csv", ["season", "player_id", "name", "team", "pos", "gp", "g", "a", "pts", "toi_avg"], sk)


def mlb():
    tot = defaultdict(lambda: defaultdict(float))
    for lgid in ("AL", "NL"):
        for kind, tid in (("bat", "teams_standard_batting_totals"), ("pitch", "teams_standard_pitching_totals")):
            p = os.path.join(C, f"mlb_{lgid}_{kind}.html")
            t = soup(p).find("table", id=tid)
            for d in rows(t):
                y = num(txt(d, "year_ID"))
                if not y or y > 2026 or (lgid == "AL" and y < 1901):
                    continue
                y = int(y)
                rec = tot[y]
                if kind == "bat":
                    rec["teams"] += num(txt(d, "teams")) or 0
                    for k in ("G", "PA", "AB", "R", "H", "HR", "RBI", "SB", "SO", "BB"):
                        rec["bat_" + k] += num(txt(d, k)) or 0
                else:
                    for k in ("G", "CG", "IP", "ER", "SO", "R"):
                        v = txt(d, k)
                        if k == "IP" and v:                 # 1234.1 means 1234 1/3 innings
                            w, _, f = v.partition(".")
                            rec["pit_IP"] += float(w) + (int(f or 0) / 3)
                        else:
                            rec["pit_" + k] += num(v) or 0
    out = []
    for y in sorted(tot):
        r = tot[y]
        tg = r["bat_G"]                                   # team-games (each team's games summed)
        if not tg:
            continue
        row = {"season": y, "teams": int(r["teams"]), "team_games": tg, "games_per_team": round(tg / r["teams"], 2)}
        for k in ("PA", "AB", "R", "H", "HR", "RBI", "SB", "SO", "BB"):
            row[k.lower() + "_per_team_g"] = round(r["bat_" + k] / tg, 5)
        row["ba"] = round(r["bat_H"] / r["bat_AB"], 5) if r["bat_AB"] else None
        row["era"] = round(9 * r["pit_ER"] / r["pit_IP"], 4) if r["pit_IP"] else None
        row["cg_per_team_g"] = round(r["pit_CG"] / tg, 5)
        row["so_pitch_per_team_g"] = round(r["pit_SO"] / tg, 5)
        out.append(row)
    write("mlb_league.csv", list(out[0].keys()), out)


def nfl():
    """league totals from every player line: PFA team pages (summed per player over teams) and nflverse"""
    pl = []
    P = B.pfa_load()                       # top-40 lists only: re-sum every player for league totals
    tot = defaultdict(lambda: defaultdict(float))
    players = defaultdict(lambda: defaultdict(float))
    names = {}
    for tp in glob.glob(os.path.join(ROOT, "cache", "pfa", "*.html")):
        base = os.path.basename(tp)[:-5]
        if not re.fullmatch(r"\d{4}[an]fl[a-z]+", base):
            continue
        y = int(base[:4])
        for t in BeautifulSoup(open(tp, encoding="utf-8").read(), "lxml").find_all("table"):
            trs = t.find_all("tr")
            if not trs:
                continue
            h = [c.get_text(" ", strip=True) for c in trs[0].find_all(["td", "th"])]
            for key, (tab, col) in B.PFA_STAT.items():
                if not h or h[0] != tab or col not in h:
                    continue
                i = h.index(col)
                for r in trs[1:]:
                    c = r.find_all(["td", "th"])
                    a = c[0].find("a") if c else None
                    if not a or len(c) <= i:
                        continue
                    v = B.pfa_num(c[i].get_text(strip=True))
                    if v is not None:
                        k = "pfa:" + a["href"].split("/")[-1].replace(".html", "")
                        players[(y, k)][key] += v
                        names[k] = a.get_text(" ", strip=True)
                        tot[y][key] += v
    for (y, k), d in players.items():
        pl.append({"season": y, "player_id": k, "name": names[k], **{x: d.get(x, "") for x in B.PFA_STAT}})
    nv_cols = {"pass_yds": "passing_yards", "pass_td": "passing_tds", "rush_yds": "rushing_yards",
               "rec_yds": "receiving_yards", "int": "def_interceptions", "sacks": "def_sacks"}
    nv = defaultdict(lambda: defaultdict(float))
    nvn = {}
    for f in ("offense.csv", "defense.csv", "reg_2025.csv"):
        for r in csv.DictReader(open(os.path.join(ROOT, "cache", "nflverse", f))):
            if r.get("season_type") != "REG" or not r.get("season", "").isdigit():
                continue
            y = int(r["season"])
            if y < 1999:
                continue
            k = "nflv:" + r["player_id"]
            nvn[k] = r.get("player_display_name") or r.get("player_name")
            for key, col in nv_cols.items():
                if r.get(col) not in (None, "", "NA"):
                    try:
                        val = float(r[col])
                    except ValueError:
                        continue
                    if f == "reg_2025.csv" or y != 2025:
                        nv[(y, k)][key] = max(nv[(y, k)][key], val) if f != "reg_2025.csv" else val
    for (y, k), d in nv.items():
        pl.append({"season": y, "player_id": k, "name": nvn[k], **{x: d.get(x, "") for x in B.PFA_STAT}})
        for x, v in d.items():
            tot[y][x] += v
    out = []
    for y in sorted(tot):
        g = B.nfl_games(y)
        n = None
        if y <= 1998:
            n = len({re.match(r"\d{4}([an]fl[a-z]+)", os.path.basename(p)[:-5]).group(1)
                     for p in glob.glob(os.path.join(ROOT, "cache", "pfa", f"{y}[an]fl[a-z]*.html"))
                     if re.fullmatch(r"\d{4}[an]fl[a-z]+", os.path.basename(p)[:-5])})
        else:
            n = 31 if y <= 2001 else 32
        row = {"season": y, "teams": n, "games_per_team": round(g, 3)}
        for x in B.PFA_STAT:
            row[x + "_per_team_g"] = round(tot[y][x] / (n * g), 4) if n else None
        out.append(row)
    write("nfl_league.csv", list(out[0].keys()), out)
    write("nfl_players.csv", ["season", "player_id", "name"] + list(B.PFA_STAT), pl)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["nba", "nhl", "mlb", "nfl"]
    for w in which:
        globals()[w]()
