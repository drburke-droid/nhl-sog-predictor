"""Records across athletics and the big four leagues, on one footing.

    python bigfour.py        # writes tables/allsports.json and tables/era_adjusted.csv; prints the findings

Every category (34 athletics events, 20 league records) is reduced to the same shape: for each season, the
best performers that season (athletics: each athlete's season best; leagues: the official leaders list),
plus the record line. Then, for every category:

* median line   the median of each season's top ten, smoothed with a centred five-season average;
* today         how often a mark as good as the record would come up at today's level: a Gumbel
                distribution fitted by moments to the season-leading marks of the last 15 completed
                seasons, and the record's return period in seasons. Counting stats are scaled to today's
                schedule (a season's value x today's games / that season's games); seasons cut short
                (strikes, lockouts, the pandemic) are left out;
* era-adjusted  for every season-leading performance in history: the same return period, but against
                everyone else's season bests in the 15 seasons around it (the season itself and seven
                either side, the performer's own seasons left out, counting stats scaled to the
                performance's own schedule). It answers: in its own era, how often would anyone else
                have done this?

"Performance" is speed for running events, distance or height for jumps and throws, the stat itself for
the leagues (minus the ERA, so bigger is always better).
"""
import os, re, csv, json, math, glob, statistics, unicodedata, datetime as dt
from collections import defaultdict
import numpy as np
from bs4 import BeautifulSoup
import analysis as A
import deep

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TABLES = os.path.join(HERE, "tables")
WINDOW = 7                                   # seasons either side for the era-adjusted measure


def norm(name):
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def titlecase_wa(name):
    return " ".join(w.title() if w.isupper() and len(w) > 1 else w for w in name.split())


def gumbel_T(values, x):
    v = np.array(values, float)
    if len(v) < 6 or v.std(ddof=1) <= 0:
        return None
    beta = v.std(ddof=1) * math.sqrt(6) / math.pi
    mu = v.mean() - 0.5772 * beta
    z = (x - mu) / beta
    if z > 700:
        return float("inf")
    F = math.exp(-math.exp(-z))
    return 1 / max(1 - F, 1e-300)


# ------------------------------------------------------------------ schedules (games per team)
def mlb_games(y):
    if y <= 1903: return 140
    if y == 1918: return 126
    if y == 1919: return 140
    if y <= 1960: return 154
    return {1972: 155, 1981: 107, 1994: 115, 1995: 144, 2020: 60}.get(y, 162)


def nba_games(y):
    return {1999: 50, 2012: 66, 2020: 72, 2021: 72}.get(y, 82 if y >= 1968 else 80)


def nhl_games(y):
    for a, b, g in [(1918, 1918, 22), (1919, 1919, 18), (1920, 1924, 24), (1925, 1925, 30), (1926, 1926, 36),
                    (1927, 1931, 44), (1932, 1942, 48), (1943, 1946, 50), (1947, 1949, 60), (1950, 1967, 70),
                    (1968, 1968, 74), (1969, 1970, 76), (1971, 1974, 78), (1975, 1992, 80), (1993, 1994, 84),
                    (1995, 1995, 48), (1996, 2012, 82), (2013, 2013, 48), (2014, 2019, 82), (2020, 2020, 70),
                    (2021, 2021, 56)]:
        if a <= y <= b:
            return g
    return 82


def nfl_games(y):
    if y <= 1934: return 12
    if y <= 1936: return 12
    if y <= 1942: return 11
    if y <= 1945: return 10
    if y == 1946: return 11
    if y <= 1960: return 12
    if y <= 1977: return 14
    return {1982: 9, 1987: 15}.get(y, 17 if y >= 2021 else 16)


SHORT = {"MLB": {1918, 1981, 1994, 2020}, "NBA": {1999, 2012, 2020}, "NHL": {1995, 2013, 2020, 2021},
         "NFL": {1982}, "Athletics": set()}
# athletics keeps 2020 in the fits (Cheptegei's 5000 m and 10,000 m records came that season) but not in
# the smoothed median line, where its thin fields would show as a dip
MEDIAN_SKIP = {"Athletics": {2020}}
DISPLAY = {"Florence Griffith-Joyner": "Florence Griffith Joyner", "Jan Zelezný": "Jan Železný",
           "Yuriy Syedikh": "Yuriy Sedykh"}


# ------------------------------------------------------------------ league parsers
def sref_entries(div, with_team=False):
    out = []
    for e in div.select("div > div"):
        w, v = e.find(class_="who"), e.find(class_="value")
        if not w or not v:
            continue
        a = w.find("a")
        name = (a.get_text(strip=True) if a else w.get_text(" ", strip=True)).replace("*", "").strip()
        if a and a.get("title") and len(re.sub(r"\s*\([A-Z0-9]{2,4}\)$", "", a["title"])) > len(name):
            name = re.sub(r"\s*\([A-Z0-9]{2,4}\)$", "", a["title"]).strip()
        m = re.search(r"\(([A-Z0-9]{2,4})\)\s*$", a.get("title", "") if a else "")
        val = v.get_text(strip=True).replace(",", "")
        team = (m.group(1) if m else "") + "|" + (a.get("data-entry-id", "") if a else "")
        try:
            out.append((name, float(val), team))
        except ValueError:
            pass
    return out if with_team else [(n, v) for n, v, _ in out]


# the sixteen American and National League franchises of 1914-1948, as Baseball-Reference codes them
ALNL_1914_48 = {"BOS", "CHW", "CLE", "DET", "NYY", "PHA", "SLB", "WSH",
                "BSN", "BRO", "CHC", "CIN", "NYG", "PHI", "PIT", "STL"}


def load_mlb(stat):
    """AL and NL only. Baseball-Reference's leaderboards now include the Negro Leagues (1920-48), which MLB
    recognises as major leagues; their short schedules make season leaderboards hard to compare, so they
    are left out here (team codes kept are those that also appear in AL/NL seasons outside 1914-15 and
    1920-48, plus multi-team codes such as 2TM)."""
    s = BeautifulSoup(open(os.path.join(CACHE, "sref", f"mlb_{stat}.html"), encoding="utf-8").read(), "lxml")
    raw = {}
    for div in s.select(f'div[id^="leaders_{stat}_y"]'):
        y = int(div["id"].rsplit("_y", 1)[1])
        if y >= 1901:
            raw[y] = sref_entries(div, with_team=True)
    team = lambda t: t.split("|")[0]
    pid = lambda t: t.split("|")[1]
    alnl = {team(t) for y, ent in raw.items() if not (1914 <= y <= 1948) for _, _, t in ent}
    alnl_players = {pid(t) for y, ent in raw.items() for _, _, t in ent
                    if (team(t) in ALNL_1914_48 if 1914 <= y <= 1948 else team(t) in alnl)}
    out = {}
    for y, ent in raw.items():
        if 1914 <= y <= 1948:                        # a multi-team line counts only for an AL/NL player
            kept = [(n, v) for n, v, t in ent if team(t) in ALNL_1914_48 or
                    (re.match(r"^\dTM$", team(t)) and pid(t) in alnl_players)]
        else:
            kept = [(n, v) for n, v, t in ent if team(t) in alnl or re.match(r"^\dTM$", team(t))]
        if kept:
            out[y] = kept
    # rate stats in 1920-48: the AL and NL leader pages, merged (the combined top tens are mostly Negro
    # Leagues players there)
    lg_id = {"batting_avg": ("batting", "leaderboard_batting_batting_avg"),
             "earned_run_avg": ("pitching", "leaderboard_pitching_earned_run_avg")}.get(stat)
    if lg_id:
        for y in range(1920, 1949):
            merged = []
            for lg in ("AL", "NL"):
                p = os.path.join(CACHE, "sref", f"mlb_{lg}_{y}_{lg_id[0]}.html")
                if not os.path.exists(p):
                    continue
                div = BeautifulSoup(open(p, encoding="utf-8").read(), "lxml").find(id=lg_id[1])
                if div:
                    merged += sref_entries(div)
            if len(merged) >= 10:
                out[y] = merged
    return out


def load_sref_seasons(league, stat, first, last):
    out = {}
    for y in range(first, last + 1):
        p = os.path.join(CACHE, "sref", f"{league}_{y}.html")
        if not os.path.exists(p):
            continue
        s = BeautifulSoup(open(p, encoding="utf-8").read(), "lxml")
        div = s.find(id=f"leaders_{stat}")
        if div:
            ent = sref_entries(div)
            if ent:
                out[y] = ent
    return out


def load_nfl(key, col, first):
    from wikitables import read_table
    j = json.load(open(os.path.join(CACHE, "nflwiki", f"{key}.json")))
    s = BeautifulSoup(j["parse"]["text"], "lxml")
    lead = defaultdict(list)
    for t in s.find_all("table", class_="wikitable"):
        h, rows = read_table(t)
        ncol = next((c for c in ("Leader", "Winner", "Player") if c in h), None)
        if "Season" not in h or not ncol:
            continue
        seasons_in = [int(r["Season"][:4]) for r in rows if r.get("Season", "")[:4].isdigit()]
        if not seasons_in or max(seasons_in) <= 1949:           # the AAFC (1946-49) is not in NFL records
            continue
        vcol = [c for c in h if c not in ("Season", ncol, "GP", "Games", "Team", "Refs", "Ref.")][0]
        for r in rows:
            try:
                y = int(r["Season"][:4])
                v = float(re.sub(r"[^\d.]", "", r[vcol]))
            except (ValueError, KeyError):
                continue
            nm = re.sub(r"\s*[†^*‡§#]|\(\d+\)", "", r[ncol]).strip()
            lead[y].append((nm, v))
    out = {}
    for y, ent in lead.items():
        if y >= first:
            best = max(ent, key=lambda e: e[1])
            out[y] = [best]                                      # NFL and AFL (1960-69): the better of the two
    # nflverse top tens from 1999
    files = [os.path.join(CACHE, "nflverse", "defense.csv" if col == "def_sacks" else "offense.csv"),
             os.path.join(CACHE, "nflverse", "reg_2025.csv")]
    for f in files:
        by = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r.get("season_type") != "REG" or not r.get(col):
                continue
            try:
                by[int(r["season"])].append((r.get("player_display_name") or r["player_name"], float(r[col])))
            except ValueError:
                pass
        for y, ent in by.items():
            ent.sort(key=lambda e: -e[1])
            top = ent[:12]
            if y >= first and y in out and abs(top[0][1] - out[y][0][1]) <= 1.0:
                out[y] = [(out[y][0][0], out[y][0][1])] + [e for e in top[1:]]
            elif y >= first and y not in out:
                out[y] = top
    return out


def load_streaks():
    names = {}
    for f in glob.glob(os.path.join(CACHE, "register", "people-*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["key_retro"]:
                names[r["key_retro"]] = f"{r['name_first']} {r['name_last']}".strip()
    out = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(CACHE, "streaks.csv"))):
        out[int(r["year"])].append((names.get(r["player"], r["player"]), float(r["streak"])))
    for y in out:
        out[y].sort(key=lambda e: -e[1])
    return dict(out)


# ------------------------------------------------------------------ athletics
def load_athletics():
    perfs, _ = deep.load_lists()
    wa = defaultdict(dict)
    for p in glob.glob(os.path.join(CACHE, "wa", "*.html")):
        sex, ev, y = os.path.basename(p)[:-5].rsplit("_", 2)
        s = BeautifulSoup(open(p, encoding="utf-8").read(), "lxml")
        t = s.find("table")
        rows = []
        for tr in t.find_all("tr")[1:] if t else []:
            td = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(td) < 4:
                continue
            try:
                m = deep_mark(td[1])
            except ValueError:
                continue
            rows.append((titlecase_wa(td[3]), m))
        wa[(sex, ev)][int(y)] = rows
    out = {}
    for (sex, ev), rows in perfs.items():
        if ev not in A.ORDER:
            continue
        field = ev in A.FIELD
        by = defaultdict(dict)
        for r in rows:
            if (sex, ev) in deep.IMPL and r["year"] < deep.IMPL[(sex, ev)]:
                continue
            k = r["athlete"]
            cur = by[r["year"]].get(k)
            if cur is None or (r["mark"] > cur if field else r["mark"] < cur):
                by[r["year"]][k] = r["mark"]
        cut = min(r["mark"] for r in rows) if field else max(r["mark"] for r in rows)
        seasons = {y: sorted(d.items(), key=lambda e: -deep.perf(ev, e[1])) for y, d in by.items()}
        # World Athletics' season lists fill only the ranks below the all-time lists' cut-off
        for y, wr in wa.get((sex, ev), {}).items():
            have = {norm(n) for n, _ in seasons.get(y, [])}
            below = [(n, m) for n, m in wr if (m < cut if field else m > cut) and norm(n) not in have]
            seasons[y] = sorted(seasons.get(y, []) + below, key=lambda e: -deep.perf(ev, e[1]))
        out[(sex, ev)] = (seasons, cut)
    return out


def deep_mark(s):
    s = s.strip().split()[0].rstrip("hA*")
    if ":" in s:
        v = 0.0
        for p in s.split(":"):
            v = v * 60 + float(p)
        return v
    return float(s)


# ------------------------------------------------------------------ categories
def fmt_val(cat, v):
    k = cat["kind"]
    if k == "athletics":
        return A.fmt(cat["event"], v)
    if k == "ba":
        return f"{v:.3f}".lstrip("0")
    if k == "era":
        return f"{v:.2f}"
    if k == "rate":
        return f"{v:.1f}"
    if k == "half":
        return f"{v:g}"
    return f"{v:,.0f}"


def season_label(cat, y):
    return f"{y - 1}–{str(y)[2:]}" if cat["sport"] in ("NBA", "NHL") else str(y)


def categories():
    cats = []
    ath = load_athletics()
    rec = A.load_records()
    for sex in ("M", "W"):
        for ev in A.ORDER:
            if (sex == "M" and ev == "100mH") or (sex == "W" and ev == "110mH"):
                continue
            seasons, cut = ath[(sex, ev)]
            line = [r for r in rec[(sex, ev)] if not r["equalled"]]
            cats.append({
                "id": f"ath_{sex}_{ev}", "sport": "Athletics", "group": group_of(ev), "event": ev, "sex": sex,
                "title": f"{'Men' if sex == 'M' else 'Women'}'s {A.NAMES[ev] if ev in A.RUN_DIST else A.NAMES[ev].lower()}",
                "kind": "athletics", "higher": ev in A.FIELD, "counting": False,
                "perf": (lambda ev: lambda v: deep.perf(ev, v))(ev), "games": None, "short": SHORT["Athletics"],
                "seasons": seasons, "cutoff": cut, "today": list(range(2012, 2027)),
                "record_line": [(deep_yearf(r["date"]), r["mark"], clean(r["athlete"])) for r in line],
                "record": (line[-1]["mark"], clean(line[-1]["athlete"]), line[-1]["date"].year),
            })
    ident = lambda v: v
    league = [
        ("MLB", "mlb_streak", "Hitting streak", "games in a row with a hit", load_streaks(), "count", False, None,
         list(range(2011, 2026))),
        ("MLB", "mlb_hr", "Home runs in a season", "", load_mlb("HR"), "count", True, mlb_games, None),
        ("MLB", "mlb_ba", "Batting average", "best since 1901", load_mlb("batting_avg"), "ba", False, None, None),
        ("MLB", "mlb_h", "Hits in a season", "", load_mlb("H"), "count", True, mlb_games, None),
        ("MLB", "mlb_sb", "Stolen bases in a season", "", load_mlb("SB"), "count", True, mlb_games, None),
        ("MLB", "mlb_so", "Strikeouts by a pitcher", "in a season", load_mlb("SO_p"), "count", True, mlb_games, None),
        ("MLB", "mlb_w", "Wins by a pitcher", "in a season", load_mlb("W"), "count", True, mlb_games, None),
        ("MLB", "mlb_era", "Earned run average", "lowest, qualified", load_mlb("earned_run_avg"), "era", False, None, None),
        ("NBA", "nba_ppg", "Points per game", "season average", load_sref_seasons("nba", "pts_per_g", 1947, 2026), "rate", False, None, None),
        ("NBA", "nba_rpg", "Rebounds per game", "season average", load_sref_seasons("nba", "trb_per_g", 1951, 2026), "rate", False, None, None),
        ("NBA", "nba_apg", "Assists per game", "season average", load_sref_seasons("nba", "ast_per_g", 1947, 2026), "rate", False, None, None),
        ("NBA", "nba_3pm", "Three-pointers made", "in a season", load_sref_seasons("nba", "fg3", 1980, 2026), "count", True, nba_games, None),
        ("NHL", "nhl_goals", "Goals in a season", "", load_sref_seasons("nhl", "goals", 1918, 2026), "count", True, nhl_games, None),
        ("NHL", "nhl_assists", "Assists in a season", "", load_sref_seasons("nhl", "assists", 1918, 2026), "count", True, nhl_games, None),
        ("NHL", "nhl_points", "Points in a season", "", load_sref_seasons("nhl", "points", 1918, 2026), "count", True, nhl_games, None),
        ("NFL", "nfl_pass_yds", "Passing yards in a season", "", load_nfl("pass_yds", "passing_yards", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_pass_td", "Touchdown passes in a season", "", load_nfl("pass_td", "passing_tds", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_rush_yds", "Rushing yards in a season", "", load_nfl("rush_yds", "rushing_yards", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_rec_yds", "Receiving yards in a season", "", load_nfl("rec_yds", "receiving_yards", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_sacks", "Sacks in a season", "official since 1982", load_nfl("sacks", "def_sacks", 1982), "half", True, nfl_games, list(range(2011, 2026))),
    ]
    for sport, cid, title, sub, seasons, kind, counting, games, today in league:
        lower = kind == "era"
        seasons = {y: sorted(e, key=lambda x: x[1] if lower else -x[1]) for y, e in seasons.items() if e}
        cat = {"id": cid, "sport": sport, "group": sport, "title": title, "sub": sub, "kind": kind,
               "higher": not lower, "counting": counting, "games": games, "short": SHORT[sport] if counting else set(),
               "perf": (lambda v: -v) if lower else ident, "seasons": seasons,
               "today": today or list(range(2012, 2027)), "cutoff": None}
        # the record line: the running best since the category starts
        best, line = None, []
        for y in sorted(seasons):
            if y in cat["short"]:
                continue
            nm, v = seasons[y][0]
            if best is None or cat["perf"](v) > cat["perf"](best[1]):
                best = (y, v, nm)
                line.append((y + 0.5, v, nm))
        cat["record_line"] = line
        cat["record"] = (best[1], best[2], best[0])
        cats.append(cat)
    return cats


def clean(s):
    import film
    return film.clean_name(s)


def deep_yearf(d):
    return d.year + (d.timetuple().tm_yday - 0.5) / 365.25


def group_of(ev):
    return {"100m": "Sprints", "200m": "Sprints", "400m": "Sprints", "110mH": "Hurdles", "100mH": "Hurdles",
            "400mH": "Hurdles", "800m": "Middle and long distance", "1500m": "Middle and long distance",
            "5000m": "Middle and long distance", "10000m": "Middle and long distance",
            "HJ": "Jumps", "PV": "Jumps", "LJ": "Jumps", "TJ": "Jumps",
            "SP": "Throws", "DT": "Throws", "JT": "Throws", "HT": "Throws"}[ev]


# ------------------------------------------------------------------ the measures
def scale(cat, v, y, to_games):
    if not cat["counting"] or not cat["games"]:
        return v
    return v * to_games / cat["games"](y)


def median_series(cat):
    raw = {}
    for y, ent in cat["seasons"].items():
        if y in cat["short"] or y in MEDIAN_SKIP.get(cat["sport"], ()) or len(ent) < 10:
            continue
        vals = sorted([v for _, v in ent], key=cat["perf"], reverse=True)[:10]
        raw[y] = statistics.median(vals)
    sm = []
    for y in sorted(raw):
        win = [raw[k] for k in range(y - 2, y + 3) if k in raw]
        if len(win) >= 3:
            sm.append((y, sum(win) / len(win)))
    return raw, sm


def today_T(cat):
    tg = cat["games"](max(cat["today"])) if cat["counting"] and cat["games"] else None
    lead = []
    for y in cat["today"]:
        if y in cat["short"] or y not in cat["seasons"]:
            continue
        lead.append(cat["perf"](scale(cat, cat["seasons"][y][0][1], y, tg)))
    rv, rn, ry = cat["record"]
    rv_today = scale(cat, rv, ry, tg) if tg else rv
    return gumbel_T(lead, cat["perf"](rv_today)), rv_today, len(lead)


def era_adjusted(cat):
    """the return period of every season-leading performance against everyone else's season bests in
    the 15 seasons around it"""
    out = []
    first = min(cat["seasons"])
    for y in sorted(cat["seasons"]):
        if y in cat["short"]:
            continue
        if cat["sport"] == "Athletics" and y < 1960:
            continue
        name, v = cat["seasons"][y][0]
        me = norm(name)
        tg = cat["games"](y) if cat["counting"] and cat["games"] else None
        others, real = [], 0
        for s in range(y - WINDOW, y + WINDOW + 1):
            if s in cat["short"] or s not in cat["seasons"]:
                continue
            o = [(n, val) for n, val in cat["seasons"][s] if norm(n) != me]
            if o:
                others.append(cat["perf"](scale(cat, o[0][1], s, tg) if tg else o[0][1]))
                real += 1
            elif cat["cutoff"] is not None and s >= first:
                others.append(cat["perf"](cat["cutoff"]))      # no one else above the list's cut-off
        # at least eight seasons with a real value: a window filled with cut-offs would look too narrow
        T = gumbel_T(others, cat["perf"](v)) if real >= 8 else None
        if T:
            out.append({"cat": cat["id"], "year": y, "name": name, "value": v, "T": T, "n": len(others),
                        "z": (cat["perf"](v) - np.mean(others)) / np.std(others, ddof=1)})
    return out


FMT = {"athletics": None, "ba": "ba", "era": "era", "rate": "rate", "half": "half", "count": "count"}


def fmt_key(cat):
    if cat["kind"] == "athletics":
        return "metres" if cat["event"] in A.FIELD else ("time" if cat["event"] in ("100m", "200m", "400m", "110mH",
                                                                                  "100mH", "400mH") else "mmss")
    return cat["kind"]


def z_for(cat, year, value, name):
    """the era-adjusted measure for a given mark (used for the side-checks quoted on the page)"""
    me, others = norm(name), []
    for s in range(year - WINDOW, year + WINDOW + 1):
        if s in cat["short"] or s not in cat["seasons"]:
            continue
        o = [(n, v) for n, v in cat["seasons"][s] if norm(n) != me]
        if o:
            others.append(cat["perf"](o[0][1]))
        elif cat["cutoff"] is not None:
            others.append(cat["perf"](cat["cutoff"]))
    z = (cat["perf"](value) - np.mean(others)) / np.std(others, ddof=1)
    return {"z": float(z), "T": gumbel_T(others, cat["perf"](value))}


CHECKS = [("beamon_minus_31cm", "ath_M_LJ", 1968, 8.59, "Bob Beamon"),
          ("fgj_official_1049", "ath_W_100m", 1988, 10.49, "Florence Griffith-Joyner"),
          ("bolt_958", "ath_M_100m", 2009, 9.58, "Usain Bolt"),
          ("ruth_1920", "mlb_hr", 1920, 54, "Babe Ruth")]


def main():
    cats = categories()
    data, era_all = [], []
    by_cat = {c["id"]: c for c in cats}
    for cat in cats:
        raw, sm = median_series(cat)
        T, rv_today, n = today_T(cat)
        rv, rn, ry = cat["record"]
        era = era_adjusted(cat)
        era_all += era
        best = max(era, key=lambda e: e["z"]) if era else None
        x0 = (1900 if cat["sex"] == "M" else 1920) if cat["sport"] == "Athletics" else min(cat["seasons"])
        if cat["id"] in ("ath_M_JT", "ath_W_JT"):
            x0 = deep.IMPL[(cat["sex"], "JT")] - 1
        data.append({
            "id": cat["id"], "sport": cat["sport"], "group": cat["group"], "title": cat["title"],
            "sub": cat.get("sub", ""), "higher": cat["higher"], "fmt": fmt_key(cat), "x0": x0,
            "record": fmt_val(cat, rv), "record_by": rn, "record_season": season_label(cat, ry), "record_year": ry,
            "today_T": T, "today_n": n,
            "record_line": [[round(x, 3), v, nm] for x, v, nm in cat["record_line"] if x >= x0 - 1],
            "median": [[y, round(v, 4)] for y, v in sm],
            "era_best": ({"season": season_label(cat, best["year"]), "name": DISPLAY.get(best["name"], best["name"]),
                          "value": fmt_val(cat, best["value"]), "z": best["z"], "T": best["T"]} if best else None),
        })
        print(f"{cat['sport']:9} {cat['title'][:34]:34} record {fmt_val(cat, rv):>9} {rn[:22]:22} "
              f"{season_label(cat, ry):7}  today: once in {T:,.0f} seasons (n={n})")
    rows = []
    for e in era_all:
        c = by_cat[e["cat"]]
        rows.append({"category": c["title"], "sport": c["sport"], "season": season_label(c, e["year"]),
                     "name": DISPLAY.get(e["name"], e["name"]), "value": fmt_val(c, e["value"]), "T": e["T"], "z": e["z"], "n": e["n"],
                     "cat": c["id"]})
    rows.sort(key=lambda r: -r["z"])
    with open(os.path.join(TABLES, "era_adjusted.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    seen, per_cat = set(), []
    for r in rows:                                   # the best performance in each category, ranked
        if r["cat"] not in seen:
            seen.add(r["cat"])
            per_cat.append(r)
    print("\nera-adjusted, the best in each category:")
    for r in per_cat[:20]:
        print(f"  {r['z']:5.1f}  T~{r['T']:.3g}  {r['sport']:9} {r['category'][:30]:30} {r['value']:>9} "
              f"{r['name'][:24]:24} {r['season']}")
    checks = {k: z_for(by_cat[c], y, v, n) for k, c, y, v, n in CHECKS}
    print("checks:", {k: round(v["z"], 2) for k, v in checks.items()})
    json.dump({"categories": data, "era_top": rows[:40], "era_per_category": per_cat, "checks": checks},
              open(os.path.join(TABLES, "allsports.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
