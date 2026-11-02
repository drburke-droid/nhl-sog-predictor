"""Download (and cache) the data for the big-four records and for the athletics season top tens.

    python sports_fetch.py wa        # World Athletics season lists, where the all-time lists hold < 10 athletes
    python sports_fetch.py sref      # baseball-, basketball- and hockey-reference leader pages
    python sports_fetch.py nflwiki   # Wikipedia's NFL annual leaders lists
    python sports_fetch.py nflverse  # nflverse season stats (1999 on)
    python sports_fetch.py streaks   # Retrosheet game logs (Chadwick retrosplits) -> every player's longest
                                     # hitting streak in every season -> cache/streaks.csv
    python sports_fetch.py nl_early  # National League leader and standings pages, 1876-1900
    python sports_fetch.py pfa       # Pro Football Archives team pages, NFL 1932-98 and AFL 1960-69
                                     # (every player's passing, rushing, receiving, interception and sack line)
    python sports_fetch.py swim      # World Aquatics season rankings, 11 Olympic pool events a sex, 1994-2026
    python sports_fetch.py swimwiki  # Wikipedia's swimming world-record progression pages

Everything is cached under cache/ and never fetched twice. sports-reference sites allow about 20 requests a
minute, so they are fetched one at a time with a 3.5 s pause.
"""
import os, sys, csv, io, json, time, urllib.error, urllib.parse, urllib.request
from collections import defaultdict
from fetch import get, UA, CACHE

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ athletics: World Athletics season lists
WA_SLUG = {
    "100m": ("sprints", "100-metres"), "200m": ("sprints", "200-metres"), "400m": ("sprints", "400-metres"),
    "800m": ("middlelong", "800-metres"), "1500m": ("middlelong", "1500-metres"),
    "5000m": ("middlelong", "5000-metres"), "10000m": ("middlelong", "10000-metres"),
    "110mH": ("hurdles", "110-metres-hurdles"), "100mH": ("hurdles", "100-metres-hurdles"),
    "400mH": ("hurdles", "400-metres-hurdles"),
    "HJ": ("jumps", "high-jump"), "PV": ("jumps", "pole-vault"), "LJ": ("jumps", "long-jump"),
    "TJ": ("jumps", "triple-jump"),
    "SP": ("throws", "shot-put"), "DT": ("throws", "discus-throw"), "JT": ("throws", "javelin-throw"),
    "HT": ("throws", "hammer-throw"),
}


def wa_needed():
    import deep, analysis as A
    perfs, _ = deep.load_lists()
    need = []
    for (sex, ev), rows in perfs.items():
        if ev not in WA_SLUG:
            continue
        by = defaultdict(set)
        for r in rows:
            by[r["year"]].add(r["athlete"])
        need += [(sex, ev, y) for y in range(2001, 2027) if len(by.get(y, ())) < 10]
    return need


def wa():
    for sex, ev, y in wa_needed():
        grp, slug = WA_SLUG[ev]
        g = "men" if sex == "M" else "women"
        url = (f"https://worldathletics.org/records/toplists/{grp}/{slug}/outdoor/{g}/senior/{y}"
               f"?regionType=world&page=1&bestResultsOnly=true")
        txt = get(url, os.path.join(CACHE, "wa", f"{sex}_{ev}_{y}.html"), pause=2.0)
        print(sex, ev, y, len(txt), flush=True)


# ------------------------------------------------------------------ sports-reference leader pages
BREF_STATS = ["HR", "batting_avg", "H", "RBI", "SB", "SO_p", "W", "earned_run_avg"]


def sref():
    for st in BREF_STATS:
        txt = get(f"https://www.baseball-reference.com/leaders/{st}_top_ten.shtml",
                  os.path.join(CACHE, "sref", f"mlb_{st}.html"), pause=3.5)
        print("mlb", st, len(txt), flush=True)
    for y in range(1947, 2027):
        lg = "BAA" if y <= 1949 else "NBA"
        txt = get(f"https://www.basketball-reference.com/leagues/{lg}_{y}_leaders.html",
                  os.path.join(CACHE, "sref", f"nba_{y}.html"), pause=3.5)
        print("nba", y, len(txt), flush=True)
    for y in range(1918, 2027):
        if y == 2005:                                        # no 2004-05 season (lockout)
            continue
        txt = get(f"https://www.hockey-reference.com/leagues/NHL_{y}_leaders.html",
                  os.path.join(CACHE, "sref", f"nhl_{y}.html"), pause=3.5)
        print("nhl", y, len(txt), flush=True)


def sref_leagues():
    """AL and NL leader pages for 1920-48, where the combined MLB top tens are mostly Negro Leagues
    players (rate stats: batting average and ERA)"""
    for y in range(1920, 1949):
        for lg in ("AL", "NL"):
            for kind in ("batting", "pitching"):
                txt = get(f"https://www.baseball-reference.com/leagues/{lg}/{y}-{kind}-leaders.shtml",
                          os.path.join(CACHE, "sref", f"mlb_{lg}_{y}_{kind}.html"), pause=3.5)
                print(lg, y, kind, len(txt), flush=True)


def nl_early():
    """the National League before the American League began (1876-1900): leader pages and the standings
    (for games per team)"""
    for y in range(1876, 1901):
        for kind in ("batting-leaders", "pitching-leaders", "standings"):
            url = (f"https://www.baseball-reference.com/leagues/NL/{y}-{kind}.shtml" if kind != "standings"
                   else f"https://www.baseball-reference.com/leagues/NL/{y}-standings.shtml")
            txt = get(url, os.path.join(CACHE, "sref", f"mlb_NL_{y}_{kind.split('-')[0]}.html"), pause=3.5)
            print("NL", y, kind, len(txt), flush=True)


# ------------------------------------------------------------------ NFL
NFL_WIKI = {
    "pass_yds": "List of NFL annual passing yards leaders",
    "pass_td": "List of NFL annual passing touchdowns leaders",
    "rush_yds": "List of NFL annual rushing yards leaders",
    "rec_yds": "List of NFL annual receiving yards leaders",
    "sacks": "List of NFL annual sacks leaders",
}


def nflwiki():
    for k, title in NFL_WIKI.items():
        q = urllib.parse.urlencode({"action": "parse", "page": title, "format": "json", "prop": "text",
                                    "redirects": 1, "formatversion": 2})
        txt = get(f"https://en.wikipedia.org/w/api.php?{q}", os.path.join(CACHE, "nflwiki", f"{k}.json"), pause=4)
        j = json.loads(txt)
        print(k, j.get("parse", {}).get("title"), "error" in j, flush=True)


def nflverse():
    base = "https://github.com/nflverse/nflverse-data/releases/download"
    for name, url in [("offense", f"{base}/player_stats/player_stats_season.csv"),
                      ("defense", f"{base}/player_stats/player_stats_def_season.csv")]:
        path = os.path.join(CACHE, "nflverse", f"{name}.csv")
        try:
            txt = get(url, path, pause=1)
            print(name, len(txt), flush=True)
        except Exception as e:                             # noqa: BLE001
            print(name, "not available:", e, flush=True)


BUSY = "request is being verified"              # the site's busy page: slow down, never cache it


def pfa_get(url, path):
    for wait in (0, 20, 120, 600):
        time.sleep(wait)
        txt = get(url, path, pause=6.0)
        if BUSY not in txt:
            return txt
        os.remove(path)
        print("  busy page; waiting before", url, flush=True)
    raise RuntimeError("still busy: " + url)


def pfa_teams(y, lg):
    import re
    txt = pfa_get(f"https://www.profootballarchives.com/{y}{lg}.html", os.path.join(CACHE, "pfa", f"{y}{lg}.html"))
    found = re.findall(rf'href="/?({y}{lg}([a-z]+)\.html)"', txt)
    return sorted({f for f, t in found if t not in ("draft", "awards")})


def pfa():
    """Pro Football Archives (no robots rules; one page at a time with a 6 s pause, backing off on its busy page)"""
    seasons = [(y, "nfl") for y in range(1932, 1999)] + [(y, "afl") for y in range(1960, 1970)]
    for y, lg in seasons:
        teams = pfa_teams(y, lg)
        for f in teams:
            try:
                pfa_get(f"https://www.profootballarchives.com/{f}", os.path.join(CACHE, "pfa", f))
            except RuntimeError as e:
                print("  missing:", e, flush=True)
        print(y, lg, len(teams), "teams", flush=True)


# ------------------------------------------------------------------ swimming
SWIM = [("50", "FREESTYLE"), ("100", "FREESTYLE"), ("200", "FREESTYLE"), ("400", "FREESTYLE"),
        ("800", "FREESTYLE"), ("1500", "FREESTYLE"), ("100", "BACKSTROKE"), ("100", "BREASTSTROKE"),
        ("100", "BUTTERFLY"), ("200", "MEDLEY"), ("400", "MEDLEY")]
# left out as too niche for a greatest-seasons list: the 200 m backstroke, breaststroke and butterfly, and the
# non-Olympic 50 m backstroke, breaststroke and butterfly


def swim():
    """World Aquatics' world rankings (long course, each swimmer's season best). The API is slow (about 10 s a
    list), so one list at a time, newest season first; its lists are thin before the late 1990s."""
    for y in range(2026, 1993, -1):
        for g in ("M", "F"):
            for dist, stroke in SWIM:
                url = ("https://api.worldaquatics.com/fina/rankings/swimming?" + urllib.parse.urlencode(
                    {"gender": g, "distance": dist, "stroke": stroke, "poolConfiguration": "LCM", "year": y,
                     "timesMode": "BEST_TIMES", "pageSize": 100}))
                path = os.path.join(CACHE, "swim", f"{g}_{dist}_{stroke}_{y}.json")
                if os.path.exists(path):
                    continue
                try:
                    get(url, path, pause=1.0)
                except RuntimeError:                       # the API answers 404 for an empty list
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "w").write('{"swimmingWorldRankings": [], "missing": true}')
        print(y, flush=True)


SWIM_WIKI = {f"{d}_{s}": f"World record progression {d} metres {s.lower().replace('medley', 'individual medley')}"
             for d, s in SWIM}


def swimwiki():
    for k, title in SWIM_WIKI.items():
        q = urllib.parse.urlencode({"action": "parse", "page": title, "format": "json", "prop": "text",
                                    "redirects": 1, "formatversion": 2})
        txt = get(f"https://en.wikipedia.org/w/api.php?{q}", os.path.join(CACHE, "swimwiki", f"{k}.json"), pause=4)
        j = json.loads(txt)
        print(k, j.get("parse", {}).get("title"), "error" in j, flush=True)


# ------------------------------------------------------------------ hitting streaks from game logs
def season_streaks(year):
    """every player's longest hitting streak within one regular season, from retrosplits' daybyday file.
    Rule 9.23: a game with a hit extends the streak; a game with no hit ends it if the player had an
    at-bat or a sacrifice fly; a game whose plate appearances were all walks, hit-by-pitches, sacrifice
    bunts or interference (or with no plate appearance) leaves it unchanged."""
    url = f"https://raw.githubusercontent.com/chadwickbureau/retrosplits/master/daybyday/playing-{year}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    # older seasons list many games twice, from the play-by-play ("evt") and the box score ("box");
    # keep one row per player per game, preferring play-by-play, then box score, then deduced games
    rank = {"evt": 0, "box": 1, "ded": 2}
    seen = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        rd = csv.DictReader(io.TextIOWrapper(r, encoding="utf-8"))
        for row in rd:
            if row.get("season.phase", "R") != "R":
                continue
            try:
                ab, h = int(row["B_AB"] or 0), int(row["B_H"] or 0)
                sf = int(row.get("B_SF") or 0)
            except ValueError:
                continue
            pa = int(row.get("B_PA") or 0)
            key = (row["game.key"], row["person.key"])
            src = rank.get(row.get("game.source", "evt"), 3)
            if key in seen and seen[key][0] <= src:
                continue
            seen[key] = (src, row["person.key"], (row["game.date"], int(row["game.number"] or 0), ab, h, sf, pa))
    games = defaultdict(list)
    for src, pid, g in seen.values():
        games[pid].append(g)
    out = []
    for pid, gl in games.items():
        gl.sort()
        cur = best = 0
        start = best_start = best_end = None
        for date, num, ab, h, sf, pa in gl:
            if h > 0:
                if cur == 0:
                    start = date
                cur += 1
                if cur > best:
                    best, best_start, best_end = cur, start, date
            elif ab > 0 or sf > 0:
                cur = 0
        if best > 0:
            out.append((year, pid, best, best_start, best_end))
    return out


def streaks():
    path = os.path.join(CACHE, "streaks.csv")
    done = set()
    if os.path.exists(path):
        done = {int(r["year"]) for r in csv.DictReader(open(path))}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["year", "player", "streak", "start", "end"])
        for y in range(1897, 2027):                        # retrosplits' game logs start in 1897
            if y in done:
                continue
            try:
                rows = season_streaks(y)
            except Exception as e:                         # noqa: BLE001
                print(y, "missing:", e, flush=True)
                continue
            rows.sort(key=lambda r: -r[2])
            for r in rows[:40]:                            # each season's 40 longest are plenty
                w.writerow(r)
            f.flush()
            print(y, "players", len(rows), "longest", rows[0][2] if rows else None, rows[0][1] if rows else "",
                  flush=True)


if __name__ == "__main__":
    {"wa": wa, "sref": sref, "sref_leagues": sref_leagues, "nflwiki": nflwiki, "nflverse": nflverse, "streaks": streaks,
     "nl_early": nl_early, "pfa": pfa, "swim": swim, "swimwiki": swimwiki}[sys.argv[1]]()
