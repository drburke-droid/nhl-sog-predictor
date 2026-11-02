"""Records and seasons across athletics, swimming and the big four leagues, on one footing.

    python bigfour.py        # writes tables/allsports.json, tables/era_adjusted.csv and
                             # tables/greatest_seasons.csv; prints the findings
    ERA_WINDOW=7 python bigfour.py      # the same with another era window (default 15 seasons either side)

Every category (30 athletics events, 22 swimming events, 22 league records) is reduced to the same shape:
for each season, the best performers that season (athletics and swimming: each athlete's season best;
leagues: the season's leaders), plus the record line. Then, for every category:

* median line   the median of each season's top ten, smoothed with a centred five-season average;
* today         how often a mark as good as the record would come up at today's level: a Gumbel
                distribution fitted by moments to the season-leading marks of the last 15 completed
                seasons, and the record's return period in seasons. Counting stats are scaled to today's
                schedule (a season's value x today's games / that season's games); seasons cut short
                (strikes, lockouts, the pandemic) are left out;
* era-adjusted  for every season-leading performance in history: everyone else's season bests in each of
                the WINDOW seasons before and after it (the performer's own seasons left out, counting
                stats scaled to the performance's own schedule), a straight trend line through them, and
                the performance's distance above that trend at its own season, in standard deviations of
                the scatter around the line (z), with a Gumbel return period fitted to the same scatter
                (T). It answers: in its own era, how often would anyone else have done this?

The greatest seasons are ranked by z, one entry per athlete-season (the category where the season stands
out most). "Performance" is speed for running and swimming, distance or height for jumps and throws, the
stat itself for the leagues (minus the ERA, so bigger is always better).
"""
import os, re, csv, json, math, glob, functools, statistics, unicodedata, datetime as dt
from collections import defaultdict
import numpy as np
from bs4 import BeautifulSoup
import analysis as A
import deep

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TABLES = os.path.join(HERE, "tables")
WINDOW = int(os.environ.get("ERA_WINDOW", 15))  # seasons either side for the era-adjusted measure
ALT_WINDOWS = (7, 10, 20)                       # robustness: the same ranking at other windows


@functools.lru_cache(maxsize=None)
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
_NL_G = {}


def nl_games(y):
    """games per team in the National League of 1876-1900, from Baseball-Reference's standings (W + L)"""
    if not _NL_G:
        from bs4 import Comment
        for yy in range(1876, 1901):
            p = os.path.join(CACHE, "sref", f"mlb_NL_{yy}_standings.html")
            if not os.path.exists(p):
                continue
            s = BeautifulSoup(open(p, encoding="utf-8").read(), "lxml")
            for c in s.find_all(string=lambda x: isinstance(x, Comment)):
                if 'id="expanded_standings_overall"' in c:
                    t = BeautifulSoup(c, "lxml").find("table")
                    g = [int(r.find(attrs={"data-stat": "W"}).get_text()) + int(r.find(attrs={"data-stat": "L"}).get_text())
                         for r in t.find("tbody").find_all("tr") if r.find(attrs={"data-stat": "W"})]
                    _NL_G[yy] = sum(g) / len(g)
    return _NL_G.get(y, 140)


def mlb_games(y):
    if y <= 1900: return nl_games(y)
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
    if y in pfa_games():
        return pfa_games()[y]
    if y <= 1934: return 12
    if y <= 1936: return 12
    if y <= 1942: return 11
    if y <= 1945: return 10
    if y == 1946: return 11
    if y <= 1960: return 12
    if y <= 1977: return 14
    return {1982: 9, 1987: 15}.get(y, 17 if y >= 2021 else 16)


SHORT = {"MLB": {1918, 1981, 1994, 2020}, "NBA": {1999, 2012, 2020}, "NHL": {1995, 2013, 2020, 2021},
         "NFL": {1982}, "Athletics": set(), "Swimming": set()}
# athletics keeps 2020 in the fits (Cheptegei's 5000 m and 10,000 m records came that season) but not in
# the smoothed median line, where its thin fields would show as a dip
MEDIAN_SKIP = {"Athletics": {2020}, "Swimming": {2020}}
# too niche for a greatest-seasons list: the 5000 m and 10,000 m (and, in swimming, the 200 m backstroke,
# breaststroke and butterfly; sports_fetch.SWIM)
NICHE = {"5000m", "10000m"}
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
    seen, uniq = set(), []                       # league leader pages repeat each board's leader
    for e in out:
        if (e[0], e[2]) not in seen:
            seen.add((e[0], e[2]))
            uniq.append(e)
    return uniq if with_team else [(n, v) for n, v, _ in uniq]


# the sixteen American and National League franchises of 1914-1948, as Baseball-Reference codes them
ALNL_1914_48 = {"BOS", "CHW", "CLE", "DET", "NYY", "PHA", "SLB", "WSH",
                "BSN", "BRO", "CHC", "CIN", "NYG", "PHI", "PIT", "STL"}


def load_mlb(stat):
    """AL and NL only (the NL from 1876, the AL from 1901). Baseball-Reference's leaderboards now include the Negro Leagues (1920-48), which MLB
    recognises as major leagues; their short schedules make season leaderboards hard to compare, so they
    are left out here (team codes kept are those that also appear in AL/NL seasons outside 1914-15 and
    1920-48, plus multi-team codes such as 2TM)."""
    s = BeautifulSoup(open(os.path.join(CACHE, "sref", f"mlb_{stat}.html"), encoding="utf-8").read(), "lxml")
    raw = {}
    early = {}
    kind = "pitching" if stat in ("SO_p", "W", "earned_run_avg") else "batting"
    for y in range(1876, 1901):                  # the National League alone, from its own leader pages
        p = os.path.join(CACHE, "sref", f"mlb_NL_{y}_{kind}.html")
        if os.path.exists(p):
            div = BeautifulSoup(open(p, encoding="utf-8").read(), "lxml").find(id=f"leaderboard_{kind}_{stat}")
            if div and sref_entries(div):
                early[y] = sref_entries(div)
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
    out.update(early)
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


# ------------------------------------------------------------------ NFL before 1999: Pro Football Archives
PFA_STAT = {"pass_yds": ("PASSING", "YDS"), "pass_td": ("PASSING", "TD"), "rush_yds": ("RUSHING", "YDS"),
            "rec_yds": ("RECEIVING", "YDS"), "int": ("INTERCEPTIONS", "INT"), "sacks": ("SACKS", "NO")}
_PFA, _PFA_G = {}, {}


def pfa_num(t):
    t = t.replace(",", "").rstrip("t").strip()
    return float(t) if re.fullmatch(r"-?\d+(\.\d+)?", t) else None


def pfa_load():
    """every player's season line, summed over the teams he played for: the NFL from 1932 (its first season
    with official statistics) and the AFL of 1960-69, which share a record book. Also each season's games
    per team (mean of W + L + T in the standings)."""
    if _PFA:
        return _PFA
    files = sorted(glob.glob(os.path.join(CACHE, "pfa", "*.html")))
    sig = [len(files), sum(os.path.getsize(f) for f in files)]
    cp = os.path.join(CACHE, "pfa_parsed.json")
    if os.path.exists(cp):
        j = json.load(open(cp))
        if j["sig"] == sig:
            _PFA.update({k: {int(y): [tuple(e) for e in v] for y, v in d.items()} for k, d in j["stats"].items()})
            _PFA_G.update({int(y): g for y, g in j["games"].items()})
            return _PFA
    tot = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))      # key -> year -> player -> value
    names = {}
    for sp in sorted(glob.glob(os.path.join(CACHE, "pfa", "[12][0-9][0-9][0-9][an]fl.html"))):
        base = os.path.basename(sp)[:-5]
        y, lg = int(base[:4]), base[4:]
        g = []
        for t in BeautifulSoup(open(sp, encoding="utf-8").read(), "lxml").find_all("table"):
            rows = t.find_all("tr")
            h = [c.get_text(" ", strip=True) for c in rows[0].find_all(["td", "th"])] if rows else []
            if h[1:4] == ["W", "L", "T"]:
                for r in rows[1:]:
                    c = [x.get_text(strip=True) for x in r.find_all(["td", "th"])]
                    if len(c) >= 4 and all(v.isdigit() for v in c[1:4]):
                        g.append(sum(int(v) for v in c[1:4]))
        if g:
            _PFA_G.setdefault(y, []).extend(g)
        for tp in glob.glob(os.path.join(CACHE, "pfa", f"{y}{lg}[a-z]*.html")):
            if not re.fullmatch(rf"{y}{lg}[a-z]+\.html", os.path.basename(tp)):
                continue
            for t in BeautifulSoup(open(tp, encoding="utf-8").read(), "lxml").find_all("table"):
                rows = t.find_all("tr")
                if not rows:
                    continue
                h = [c.get_text(" ", strip=True) for c in rows[0].find_all(["td", "th"])]
                for key, (tab, col) in PFA_STAT.items():
                    if h[0] != tab or col not in h:
                        continue
                    i = h.index(col)
                    for r in rows[1:]:
                        c = r.find_all(["td", "th"])
                        a = c[0].find("a") if c else None
                        if not a or len(c) <= i:
                            continue
                        v = pfa_num(c[i].get_text(strip=True))
                        if v is not None:
                            tot[key][y][a["href"]] += v
                            names[a["href"]] = a.get_text(" ", strip=True)
    for key, by in tot.items():
        _PFA[key] = {y: sorted([(names[pid], v) for pid, v in d.items()], key=lambda e: -e[1])[:40]
                     for y, d in by.items()}
    for y, g in list(_PFA_G.items()):
        _PFA_G[y] = sum(g) / len(g)
    json.dump({"sig": sig, "stats": _PFA, "games": _PFA_G}, open(cp, "w"))
    return _PFA


def pfa_games():
    pfa_load()
    return _PFA_G


def load_nfl_full(key, col, first):
    """season top lists: Pro Football Archives up to 1998, nflverse from 1999"""
    out = {y: e for y, e in pfa_load().get(key, {}).items() if first <= y <= 1998 and e}
    files = [os.path.join(CACHE, "nflverse", "defense.csv" if col.startswith("def_") else "offense.csv"),
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
            if y >= max(first, 1999) and y not in out:
                out[y] = sorted(ent, key=lambda e: -e[1])[:40]
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


# ------------------------------------------------------------------ swimming
STROKE = {"FREESTYLE": "freestyle", "BACKSTROKE": "backstroke", "BREASTSTROKE": "breaststroke",
          "BUTTERFLY": "butterfly", "MEDLEY": "individual medley"}
SWIM_FULL = 20          # a season counts once World Aquatics lists at least this many swimmers (thin before ~1997)


def swim_secs(t):
    t = t.strip().rstrip("*")
    v = 0.0
    for p in t.split(":"):
        v = v * 60 + float(p)
    return v


FAMILY_FIRST = {"CHN", "KOR", "PRK"}


def swim_name(r):
    if r.get("participantCountryCode") in FAMILY_FIRST:
        return f"{(r.get('lastName') or '').title()} {(r.get('firstName') or '').strip()}".strip()
    last = " ".join(w.capitalize() if w.isupper() else w for w in re.split(r"(?<=[ -])", (r.get("lastName") or "").strip()))
    last = re.sub(r"(?<=Mc)([a-z])", lambda m: m.group(1).upper(), last.replace(" -", "-"))
    return f"{(r.get('firstName') or '').strip()} {last}".strip()


def load_swim(g, dist, stroke):
    seasons = {}
    for p in glob.glob(os.path.join(CACHE, "swim", f"{g}_{dist}_{stroke}_*.json")):
        y = int(os.path.basename(p)[:-5].rsplit("_", 1)[1])
        best = {}
        for r in json.load(open(p, encoding="utf-8")).get("swimmingWorldRankings", []):
            try:
                t = swim_secs(r["time"])
            except (ValueError, KeyError, AttributeError):
                continue
            k = r.get("personId") or swim_name(r)
            if k not in best or t < best[k][1]:
                best[k] = (swim_name(r), t)
        if len(best) >= SWIM_FULL:
            seasons[y] = sorted(best.values(), key=lambda e: e[1])
    return seasons


def swim_records(dist, stroke, sex):
    """the long-course world-record line from Wikipedia's progression page"""
    from wikitables import read_table
    p = os.path.join(CACHE, "swimwiki", f"{dist}_{stroke}.json")
    if not os.path.exists(p):
        return []
    s = BeautifulSoup(json.load(open(p))["parse"]["text"], "lxml")
    h2, h3, out = "", "", []
    want = "men" if sex == "M" else "women"
    for el in s.find_all(["h2", "h3", "table"]):
        if el.name == "h2":
            h2, h3 = el.get_text(" ", strip=True).lower(), ""
        elif el.name == "h3":
            h3 = el.get_text(" ", strip=True).lower()
        elif "wikitable" in (el.get("class") or []):
            ctx = f"{h2} | {h3}"
            words = set(re.findall(r"[a-z]+", ctx))
            if want not in words or (want == "men" and "women" in words) or "long" not in words:
                continue
            h, rows = read_table(el)
            if "Name" not in h or "Time" not in h:
                continue
            for r in rows:
                try:
                    t = swim_secs(r["Time"].split()[0])
                    d = parse_date(r["Date"])
                except (ValueError, KeyError, IndexError):
                    continue
                if not out or t < out[-1][1]:
                    out.append((d, t, re.sub(r"\s*\(.*?\)", "", r["Name"]).strip()))
    return out


def parse_date(t):
    t = re.sub(r"\[.*?\]", "", t).strip()
    for f in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%d %B, %Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(t, f).date()
        except ValueError:
            pass
    m = re.search(r"(18|19|20)\d\d", t)
    if m:
        return dt.date(int(m.group(0)), 7, 1)
    raise ValueError(t)


# ------------------------------------------------------------------ categories
def fmt_val(cat, v):
    k = cat["kind"]
    if k == "athletics":
        return A.fmt(cat["event"], v)
    if k == "swim":
        m, sec = divmod(v, 60)
        return f"{int(m)}:{sec:05.2f}" if m else f"{sec:.2f}"
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
            if (sex == "M" and ev == "100mH") or (sex == "W" and ev == "110mH") or ev in NICHE:
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
    from sports_fetch import SWIM
    for sex in ("M", "W"):
        for dist, stroke in SWIM:
            seasons = load_swim("M" if sex == "M" else "F", dist, stroke)
            line = swim_records(dist, stroke, sex)
            if not seasons or not line:
                print("swimming: no data for", sex, dist, stroke)
                continue
            d = float(dist)
            cats.append({
                "id": f"swim_{sex}_{dist}_{stroke[:2].lower()}", "sport": "Swimming",
                "group": "Freestyle" if stroke == "FREESTYLE" else ("Medley" if stroke == "MEDLEY" else "Strokes"),
                "event": f"{dist} {stroke}", "sex": sex,
                "title": f"{'Men' if sex == 'M' else 'Women'}'s {dist} m {STROKE[stroke]}",
                "kind": "swim", "higher": False, "counting": False,
                "perf": (lambda d: lambda v: d / v)(d), "games": None, "short": SHORT["Swimming"],
                "seasons": seasons, "cutoff": None, "today": list(range(2012, 2027)),
                "record_line": [(deep_yearf(r[0]), r[1], r[2]) for r in line],
                "record": (line[-1][1], line[-1][2], line[-1][0].year),
            })
    ident = lambda v: v
    league = [
        ("MLB", "mlb_streak", "Hitting streak", "games in a row with a hit", load_streaks(), "count", False, None,
         list(range(2011, 2026))),
        ("MLB", "mlb_rbi", "Runs batted in", "in a season", load_mlb("RBI"), "count", True, mlb_games, None),
        ("MLB", "mlb_hr", "Home runs in a season", "", load_mlb("HR"), "count", True, mlb_games, None),
        ("MLB", "mlb_ba", "Batting average", "", load_mlb("batting_avg"), "ba", False, None, None),
        ("MLB", "mlb_h", "Hits in a season", "", load_mlb("H"), "count", True, mlb_games, None),
        ("MLB", "mlb_sb", "Stolen bases in a season", "since 1898 (today's rule)",
         {y: e for y, e in load_mlb("SB").items() if y >= 1898}, "count", True, mlb_games, None),
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
        ("NFL", "nfl_pass_yds", "Passing yards in a season", "", load_nfl_full("pass_yds", "passing_yards", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_pass_td", "Touchdown passes in a season", "", load_nfl_full("pass_td", "passing_tds", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_rush_yds", "Rushing yards in a season", "", load_nfl_full("rush_yds", "rushing_yards", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_rec_yds", "Receiving yards in a season", "", load_nfl_full("rec_yds", "receiving_yards", 1932), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_int", "Interceptions in a season", "official since 1940", load_nfl_full("int", "def_interceptions", 1940), "count", True, nfl_games, list(range(2011, 2026))),
        ("NFL", "nfl_sacks", "Sacks in a season", "official since 1982", load_nfl_full("sacks", "def_sacks", 1982), "half", True, nfl_games, list(range(2011, 2026))),
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
            "400mH": "Hurdles", "800m": "Middle distance", "1500m": "Middle distance",
            "5000m": "Long distance", "10000m": "Long distance",
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


def era_measure(pts, y, x):
    """pts: (season, performance) for everyone else's season bests in the window; x the performance in season
    y. The window can span a changing sport, so each is measured from the straight-line trend through
    everyone else's bests: z is the distance above the trend in standard deviations of the scatter around it,
    T the return period from a Gumbel fitted to that scatter."""
    yrs, v = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    a, b = np.polyfit(yrs - y, v, 1)
    resid = v - (a * (yrs - y) + b)
    sd = resid.std(ddof=2)
    if sd <= 0:
        return None, None
    return (x - b) / sd, gumbel_T(resid, x - b)


def era_adjusted(cat, window=None):
    """every season-leading performance against everyone else's season bests in the seasons around it
    (WINDOW either side), measured from their trend"""
    window = window or WINDOW
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
        pts, real = [], 0
        for s in range(y - window, y + window + 1):
            if s in cat["short"] or s not in cat["seasons"]:
                continue
            o = next((val for n, val in cat["seasons"][s] if norm(n) != me), None)   # best of everyone else
            if o is not None:
                pts.append((s, cat["perf"](scale(cat, o, s, tg) if tg else o)))
                real += 1
            elif cat["cutoff"] is not None and s >= first:
                pts.append((s, cat["perf"](cat["cutoff"])))     # no one else above the list's cut-off
        # at least eight seasons with a real value: a window filled with cut-offs would look too narrow
        if real < 8:
            continue
        z, T = era_measure(pts, y, cat["perf"](v))
        if z is not None:
            out.append({"cat": cat["id"], "year": y, "name": name, "value": v, "T": T, "n": len(pts), "z": z})
    return out


FMT = {"athletics": None, "ba": "ba", "era": "era", "rate": "rate", "half": "half", "count": "count"}


def fmt_key(cat):
    if cat["kind"] == "athletics":
        return "metres" if cat["event"] in A.FIELD else ("time" if cat["event"] in ("100m", "200m", "400m", "110mH",
                                                                                  "100mH", "400mH") else "mmss")
    return cat["kind"]


def z_for(cat, year, value, name, window=None):
    """the era-adjusted measure for a given mark (used for the side-checks quoted on the page)"""
    window = window or WINDOW
    me, pts = norm(name), []
    tg = cat["games"](year) if cat["counting"] and cat["games"] else None
    for s in range(year - window, year + window + 1):
        if s in cat["short"] or s not in cat["seasons"]:
            continue
        o = next((v for n, v in cat["seasons"][s] if norm(n) != me), None)
        if o is not None:
            pts.append((s, cat["perf"](scale(cat, o, s, tg) if tg else o)))
        elif cat["cutoff"] is not None:
            pts.append((s, cat["perf"](cat["cutoff"])))
    z, T = era_measure(pts, year, cat["perf"](value))
    return {"z": float(z), "T": T}


CHECKS = [("beamon_minus_31cm", "ath_M_LJ", 1968, 8.59, "Bob Beamon"),
          ("fgj_official_1049", "ath_W_100m", 1988, 10.49, "Florence Griffith-Joyner"),
          ("bolt_958", "ath_M_100m", 2009, 9.58, "Usain Bolt"),
          ("ruth_1920", "mlb_hr", 1920, 54, "Babe Ruth")]


PHRASE = {"mlb_streak": "{v}-game hitting streak", "mlb_hr": "{v} home runs", "mlb_ba": "{v} batting average",
          "mlb_h": "{v} hits", "mlb_rbi": "{v} runs batted in", "mlb_sb": "{v} stolen bases",
          "mlb_so": "{v} strikeouts", "mlb_w": "{v} wins", "mlb_era": "{v} earned run average",
          "nba_ppg": "{v} points a game", "nba_rpg": "{v} rebounds a game", "nba_apg": "{v} assists a game",
          "nba_3pm": "{v} three-pointers", "nhl_goals": "{v} goals", "nhl_assists": "{v} assists",
          "nhl_points": "{v} points", "nfl_pass_yds": "{v} passing yards", "nfl_pass_td": "{v} touchdown passes",
          "nfl_rush_yds": "{v} rushing yards", "nfl_rec_yds": "{v} receiving yards", "nfl_int": "{v} interceptions",
          "nfl_sacks": "{v} sacks"}


def phrase(c, v):
    """'50.4 points a game', '8.90 m in the long jump'"""
    if c["id"] in PHRASE:
        return PHRASE[c["id"]].format(v=v)
    t = c["title"].split(" ", 1)[1]
    t = re.sub(r"(\d+) metres", r"\1 m", t)
    return f"{v} in the {t}"


def athlete_seasons(era_all, by_cat):
    """one entry per athlete per season: the category where the season stands out most, with the others it
    also led listed alongside"""
    best = {}
    for e in era_all:
        c = by_cat[e["cat"]]
        k = (c["sport"], norm(DISPLAY.get(e["name"], e["name"])), e["year"])
        best.setdefault(k, []).append(e)
    out = []
    for k, es in best.items():
        es.sort(key=lambda e: -e["z"])
        top = es[0]
        c = by_cat[top["cat"]]
        out.append({"sport": c["sport"], "name": DISPLAY.get(top["name"], top["name"]), "season": season_label(c, top["year"]),
                    "year": top["year"], "category": c["title"], "cat": c["id"], "value": fmt_val(c, top["value"]),
                    "phrase": phrase(c, fmt_val(c, top["value"])), "sex": c.get("sex", ""),
                    "z": top["z"], "T": top["T"], "n": top["n"],
                    "also": [{"category": by_cat[e["cat"]]["title"], "value": fmt_val(by_cat[e["cat"]], e["value"]),
                              "z": e["z"]} for e in es[1:]]})
    out.sort(key=lambda r: -r["z"])
    return out


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
        if cat["sport"] == "Swimming":                       # the record line from 1960, the field when complete
            x0 = max(1960, math.floor(cat["record_line"][0][0]))
        before = [p for p in cat["record_line"] if p[0] < x0 - 1]
        data.append({
            "id": cat["id"], "sport": cat["sport"], "group": cat["group"], "title": cat["title"],
            "sub": cat.get("sub", ""), "higher": cat["higher"], "fmt": fmt_key(cat), "x0": x0,
            "record": fmt_val(cat, rv), "record_by": rn, "record_season": season_label(cat, ry), "record_year": ry,
            "today_T": T, "today_n": n,
            "record_line": [[round(x, 3), v, nm] for x, v, nm in before[-1:] + [p for p in cat["record_line"] if p[0] >= x0 - 1]],
            "median": [[y, round(v, 4)] for y, v in sm],
            "era_best": ({"season": season_label(cat, best["year"]), "name": DISPLAY.get(best["name"], best["name"]),
                          "value": fmt_val(cat, best["value"]), "z": best["z"], "T": best["T"]} if best else None),
        })
        print(f"{cat['sport']:9} {cat['title'][:34]:34} record {fmt_val(cat, rv):>9} {rn[:22]:22} "
              f"{season_label(cat, ry):7}  today: once in {'–' if T is None else f'{T:,.0f}'} seasons (n={n})")
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
    # the greatest seasons: one per athlete-season, at this window and at others (robustness)
    seasons = athlete_seasons(era_all, by_cat)
    alt = {}
    for w in ALT_WINDOWS:
        if w == WINDOW:
            continue
        ranked = athlete_seasons([e for c in cats for e in era_adjusted(c, w)], by_cat)
        alt[w] = {(r["sport"], norm(r["name"]), r["year"]): (i + 1, r["z"]) for i, r in enumerate(ranked)}
    for k, c in checks.items():                      # where each side-check would rank among all seasons
        c["rank"] = 1 + sum(1 for r in seasons if r["z"] > c["z"])
    for i, r in enumerate(seasons):
        r["rank"] = i + 1
        k = (r["sport"], norm(r["name"]), r["year"])
        r["alt"] = {str(w): {"rank": a.get(k, (None, None))[0], "z": a.get(k, (None, None))[1]} for w, a in alt.items()}
    print(f"\nthe greatest seasons (window ±{WINDOW}; ranks at ±{', ±'.join(map(str, alt))}):")
    for r in seasons[:25]:
        print(f"  {r['rank']:2} {r['z']:5.2f}  {r['sport']:9} {r['name'][:24]:24} {r['season']:8} {r['category'][:30]:30} "
              f"{r['value']:>9}   " + "  ".join(f"±{w}: #{r['alt'][str(w)]['rank']}" for w in alt))
    with open(os.path.join(TABLES, "greatest_seasons.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "athlete", "sport", "season", "category", "value", "z", "once_in_seasons", "seasons_compared"]
                   + [f"rank_window_{w_}" for w_ in alt] + ["also_led"])
        for r in seasons[:200]:
            w.writerow([r["rank"], r["name"], r["sport"], r["season"], r["category"], r["value"], round(r["z"], 3),
                        round(r["T"]) if r["T"] and math.isfinite(r["T"]) else "", r["n"]]
                       + [r["alt"][str(w_)]["rank"] for w_ in alt]
                       + ["; ".join(f"{a['category']} {a['value']}" for a in r["also"])])
    json.dump({"categories": data, "era_top": rows[:40], "era_per_category": per_cat, "checks": checks,
               "window": WINDOW, "top_seasons": seasons[:80], "n_seasons": len(seasons), "alt_windows": [w for w in alt]},
              open(os.path.join(TABLES, "allsports.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
