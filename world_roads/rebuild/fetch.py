"""Raw pages and files for the rebuild study, fetched once and cached:  python fetch.py
-> data/sr/<league>/<page>.html (Sports Reference), data/nflverse/*.csv

* Sports Reference, seasons 2000-01 to 2024-25: hockey-reference.com league pages (standings, points %, roster age),
  basketball-reference.com league pages (W-L, roster age), baseball-reference.com team pitching and batting pages (W-L,
  pitcher and batter age), and the NHL and NBA entry drafts 2000-2025. One request every 4 s (their limit is 20 a
  minute); a page already cached is never asked for again.
* nflverse (github.com/nflverse/nfldata): every NFL game since 1999, every draft pick, every trade since 2002.
"""
import os, sys, time, urllib.request

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
NFLVERSE = {f: f"https://raw.githubusercontent.com/nflverse/nfldata/master/data/{f}.csv" for f in ("games", "draft_picks", "trades")}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    for f, url in NFLVERSE.items():
        p = os.path.join(DATA, "nflverse", f + ".csv")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(get(url))
            print("nflverse", f, flush=True)
    todo = [(lg, name, url) for lg, pages in PAGES.items() for name, url in pages
            if not os.path.exists(os.path.join(DATA, "sr", lg, name + ".html"))]
    print(f"{sum(len(v) for v in PAGES.values())} Sports Reference pages, {len(todo)} to fetch (about {len(todo) * 4 / 60:.0f} min)", flush=True)
    for i, (lg, name, url) in enumerate(todo):
        p = os.path.join(DATA, "sr", lg, name + ".html")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for attempt in range(3):
            try:
                open(p, "wb").write(get(url))
                break
            except Exception as e:
                if attempt == 2:
                    sys.exit(f"failed {url}: {e}")
                time.sleep(60)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} {lg} {name}", flush=True)
        time.sleep(4)


if __name__ == "__main__":
    main()
