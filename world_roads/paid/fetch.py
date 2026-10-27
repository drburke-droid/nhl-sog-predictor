"""Raw pages for the highest-paid race, fetched once and cached in data/:  python fetch.py [stage]

  a   the CPI (Minneapolis Fed's table of the annual average CPI-U, 1913 on); Baseball Reference's player value pages
      1913-2000 (WAR, to pick whose salaries to look up; 2001-2025 come from ../rebuild/data); Basketball Reference's
      league pages and every team page 1985-2000 (each roster's salaries; 2001-2025 from ../rebuild/data)
  b   the Baseball Reference page of every candidate (candidates.py): its salary table, Michael Haupert's research of
      the Hall of Fame's contract cards for the early years, the contracts since (and the NBA pages, if a not done)
  c   for the career race: the Minneapolis Fed's CPI estimates for 1800-1912 (careers that began before 1913), and the
      Federal Reserve's H.10 daily US$/C$ rates since 1971 (to convert salaries reported in Canadian dollars)

Sports Reference asks for no more than 20 requests a minute: request starts are kept 3.5 s apart across all threads,
and a page already cached is never asked for again.
"""
import os, re, sys, time, threading, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REBUILD = os.path.join(os.path.dirname(HERE), "rebuild", "data")
UA = "Claude-User (research on athletes' pay for a video; one request per page, cached)"
CPI = "https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1913-"
CPI_1800 = "https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1800-"


GAP = 3.5                                   # seconds between request starts, all threads together (<= 17 a minute)
_lock, _last = threading.Lock(), [0.0]


def get(url):
    with _lock:
        wait = _last[0] + GAP - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=60).read()


def cache(path, url, wait=0):
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(3):
        try:
            data = get(url)
            open(path + ".part", "wb").write(data)
            os.replace(path + ".part", path)           # a page is either all there or not there
            time.sleep(wait)
            return True
        except Exception as e:
            if "404" in str(e):
                open(path, "wb").write(b"")         # no such page: remember that too
                time.sleep(wait)
                return True
            if attempt == 2:
                raise RuntimeError(f"failed {url}: {e}")
            time.sleep(60)


def sr(path, url):
    return os.path.join(DATA, "sr", path), url


def mlb_values():
    jobs = [(os.path.join(DATA, "cpi.html"), CPI)]
    for y in range(1913, 2001):
        for k in ("batting", "pitching"):
            jobs.append(sr(f"mlb/value_{k}_{y}.html", f"https://www.baseball-reference.com/leagues/majors/{y}-value-{k}.shtml"))
    run(jobs, "mlb values")


def nba_teams():
    run([sr(f"nba/NBA_{y}.html", f"https://www.basketball-reference.com/leagues/NBA_{y}.html") for y in range(1985, 2001)], "nba leagues")
    jobs = []
    for y in range(1985, 2001):
        h = open(os.path.join(DATA, "sr", "nba", f"NBA_{y}.html"), encoding="utf8", errors="replace").read()
        for code in sorted(set(re.findall(rf'/teams/([A-Z]{{3}})/{y}\.html', h))):
            jobs.append(sr(f"nba/team_{code}_{y}.html", f"https://www.basketball-reference.com/teams/{code}/{y}.html"))
    run(jobs, "nba teams")


def stage_a():
    together(mlb_values, nba_teams)


def together(*fs):
    errors = []

    def go(f):
        try:
            f()
        except Exception as e:
            errors.append(e)
    ts = [threading.Thread(target=go, args=(f,)) for f in fs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        sys.exit("; ".join(map(str, errors)))


def mlb_players():
    ids = [l.strip() for l in open(os.path.join(DATA, "candidates.txt")) if l.strip()]
    jobs = [sr(f"mlb/players/{pid}.shtml", f"https://www.baseball-reference.com/players/{pid[0]}/{pid}.shtml") for pid in ids]
    # three readers in priority order: a page takes longer to arrive than the rate limit's gap, so one reader alone
    # would use half the allowance
    together(*[(lambda part=jobs[i::3], i=i: run(part, f"mlb players {i + 1}/3")) for i in range(3)])


def stage_b():
    together(mlb_players, nba_teams)


def stage_c():
    run([(os.path.join(DATA, "cpi_1800.html"), CPI_1800)] +
        [(os.path.join(DATA, "fx", f"{f}_ca.htm"), f"https://www.federalreserve.gov/releases/h10/hist/{f}_ca.htm")
         for f in ("dat89", "dat96", "dat00")], "cpi 1800-, exchange rates")


def run(jobs, what=""):
    todo = [(p, u) for p, u in jobs if not os.path.exists(p)]
    print(f"{what}: {len(jobs)} pages, {len(todo)} to fetch", flush=True)
    for i, (p, u) in enumerate(todo):
        cache(p, u)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"  {what} {i + 1}/{len(todo)} {os.path.basename(p)}", flush=True)


if __name__ == "__main__":
    {"a": stage_a, "b": stage_b, "c": stage_c}[sys.argv[1] if len(sys.argv) > 1 else "a"]()
