"""Raw pages and files for the rebuild study, fetched once and cached:  python fetch.py
-> data/sr/<league>/<page>.html (Sports Reference), data/nflverse/*.csv, data/<source>/... (payroll, signings, trades)

* Sports Reference, seasons 2000-01 to 2024-25: hockey-reference.com league pages (standings, points %, roster age),
  basketball-reference.com league pages (W-L, roster age), baseball-reference.com team pitching and batting pages (W-L,
  pitcher and batter age), and the NHL and NBA entry drafts 2000-2025. One request every 4 s (their limit is 20 a
  minute); a page already cached is never asked for again.
* nflverse (github.com/nflverse/nfldata): every NFL game since 1999, every draft pick, every trade since 2002.

Payroll, free agents and trades (the second pass):
* Sports Reference player values each season: NBA win shares (advanced stats), NHL point shares (skaters' misc. stats, goalies),
  MLB wins above replacement (player value, batting and pitching); NBA transactions (trades, signings) and NBA team
  pages (each roster's salaries).
* NHL Trade Tracker (nhltradetracker.com): every NHL trade, by season. CapWages (capwages.com): every NHL contract
  signed since 2002-03, with its type, length and cap hit.
* Retrosheet transactions (retrosheet.org): every MLB trade and free-agent signing to 2021. The Lahman database (the R
  package on CRAN): MLB salaries to 2016, and the Retrosheet-to-Baseball-Reference id map.
* nflverse contracts (OverTheCap): every NFL contract, its type, and what it paid each season; nflverse players (ids).
"""
import os, re, sys, time, threading, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
UA = "Claude-User (research on team-building for a video; one request per page, cached)"
YEARS = range(2001, 2026)                   # season end years: 2000-01 .. 2024-25
PAGES = {
    "nhl": [(f"NHL_{y}", f"https://www.hockey-reference.com/leagues/NHL_{y}.html") for y in YEARS if y != 2005]
           + [(f"draft_{y}", f"https://www.hockey-reference.com/draft/NHL_{y}_entry.html") for y in range(2000, 2026)],
    "nba": [(f"NBA_{y}", f"https://www.basketball-reference.com/leagues/NBA_{y}.html") for y in YEARS]
           + [(f"draft_{y}", f"https://www.basketball-reference.com/draft/NBA_{y}.html") for y in range(2000, 2026)],
    "mlb": [(f"pitching_{y}", f"https://www.baseball-reference.com/leagues/majors/{y}-standard-pitching.shtml") for y in YEARS]
           + [(f"batting_{y}", f"https://www.baseball-reference.com/leagues/majors/{y}-standard-batting.shtml") for y in YEARS],
}
PAGES["nba"] += ([(f"advanced_{y}", f"https://www.basketball-reference.com/leagues/NBA_{y}_advanced.html") for y in YEARS]
                 + [(f"transactions_{y}", f"https://www.basketball-reference.com/leagues/NBA_{y}_transactions.html") for y in YEARS])
PAGES["nhl"] += ([(f"skaters_{y}", f"https://www.hockey-reference.com/leagues/NHL_{y}_skaters.html") for y in YEARS if y != 2005]
                 + [(f"goalies_{y}", f"https://www.hockey-reference.com/leagues/NHL_{y}_goalies.html") for y in YEARS if y != 2005]
                 + [(f"misc_{y}", f"https://www.hockey-reference.com/leagues/NHL_{y}_skaters-misc.html") for y in YEARS if y != 2005])
PAGES["mlb"] += ([(f"value_batting_{y}", f"https://www.baseball-reference.com/leagues/majors/{y}-value-batting.shtml") for y in YEARS]
                 + [(f"value_pitching_{y}", f"https://www.baseball-reference.com/leagues/majors/{y}-value-pitching.shtml") for y in YEARS])
NFLVERSE = {f: f"https://raw.githubusercontent.com/nflverse/nfldata/master/data/{f}.csv" for f in ("games", "draft_picks", "trades")}
FILES = {   # one-off downloads: path under data/ -> url
    "nflverse/historical_contracts.parquet": "https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.parquet",
    "nflverse/players.csv": "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv",
    "retrosheet/tranDB.zip": "https://www.retrosheet.org/transactions/tranDB.zip",
    "lahman/Lahman_14.0-0.tar.gz": "https://cran.r-project.org/src/contrib/Lahman_14.0-0.tar.gz",
}
SEASONS = [f"{y - 1}-{y % 100:02d}" for y in YEARS]                        # "2000-01" .. "2024-25"
CAPWAGES = [(f"signings_{s}", f"https://capwages.com/signings/{s}") for s in SEASONS if s >= "2002-03"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=60).read()


def nba_team_pages():
    """every team's page each season (its salaries table), the team codes as each season's league page links them"""
    out = []
    for y in YEARS:
        h = open(os.path.join(DATA, "sr", "nba", f"NBA_{y}.html"), encoding="utf8").read()
        for code in sorted(set(re.findall(rf'/teams/([A-Z]{{3}})/{y}\.html', h))):
            out.append((f"team_{code}_{y}", f"https://www.basketball-reference.com/teams/{code}/{y}.html"))
    return out


def cache(path, url, wait):
    """fetch url to path unless cached; three tries a minute apart"""
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(3):
        try:
            data = get(url)
            open(path, "wb").write(data)
            time.sleep(wait)
            return True
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"failed {url}: {e}")
            time.sleep(60)


def nhl_trades():
    """NHL Trade Tracker, page by page through each season (its pager says how many)"""
    for s in SEASONS:
        page = 1
        while True:
            p = os.path.join(DATA, "nhltt", f"{s}_{page}.html")
            cache(p, f"https://www.nhltradetracker.com/user/trade_list_by_season/{s}/{page}", 4)
            h = open(p, encoding="utf8", errors="replace").read()
            if f"/trade_list_by_season/{s}/{page + 1}" not in h:
                break
            page += 1
    print("nhltradetracker done", flush=True)


def capwages():
    for name, url in CAPWAGES:
        cache(os.path.join(DATA, "capwages", name + ".html"), url, 4)
    print("capwages done", flush=True)


def sports_reference():
    todo = [(lg, name, url) for lg, pages in PAGES.items() for name, url in pages
            if not os.path.exists(os.path.join(DATA, "sr", lg, name + ".html"))]
    todo += [("nba", name, url) for name, url in nba_team_pages() if not os.path.exists(os.path.join(DATA, "sr", "nba", name + ".html"))]
    print(f"{len(todo)} Sports Reference pages to fetch (about {len(todo) * 4 / 60:.0f} min)", flush=True)
    for i, (lg, name, url) in enumerate(todo):
        cache(os.path.join(DATA, "sr", lg, name + ".html"), url, 4)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} {lg} {name}", flush=True)


def main():
    for f, url in FILES.items():
        if cache(os.path.join(DATA, f), url, 2):
            print("file", f, flush=True)
    for f, url in NFLVERSE.items():
        p = os.path.join(DATA, "nflverse", f + ".csv")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(get(url))
            print("nflverse", f, flush=True)
    errors = []

    def run(f):
        try:
            f()
        except Exception as e:
            errors.append(e)
    ts = [threading.Thread(target=run, args=(f,)) for f in (sports_reference, nhl_trades, capwages)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        sys.exit("; ".join(map(str, errors)))


if __name__ == "__main__":
    main()
