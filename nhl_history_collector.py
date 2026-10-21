"""
Pull every NHL game-by-game player stat line the league's stats API has,
1917-18 to the current season, into nhl_history.db.

Tables
    skater_games  one row per skater per game (regular season + playoffs)
    goalie_games  one row per goalie per game
    seasons       the league's season list (start/end dates, games per team)

The stats API caps any one query at 10,000 rows, so each season is pulled in
date windows small enough to stay under the cap; a window that hits the cap
is split in half and pulled again.  Window count is sized from the season's
game count so most seasons need only a handful of requests.  Re-running only fetches seasons that are
missing or still in progress.

    python nhl_history_collector.py              # everything
    python nhl_history_collector.py 20262027     # one or more seasons
"""

import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import requests

STATS_URL = "https://api.nhle.com/stats/rest/en"
DB_PATH = "nhl_history.db"
ROW_CAP = 10000
ROWS_PER_WINDOW = 7000
WORKERS = 2

# table -> (endpoint, rough rows per game, used to size the date windows)
ENDPOINTS = {
    "skater_games": ("skater/summary", 40),
    "goalie_games": ("goalie/summary", 3),
}

session = requests.Session()


def _get(url: str) -> dict:
    for attempt in range(8):
        try:
            resp = session.get(url, timeout=(10, 60))
            if resp.status_code == 429:  # rate limited: wait it out
                time.sleep(int(resp.headers.get("Retry-After", 0)) or 5 * 2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"still rate limited: {url}")


def fetch_seasons() -> list:
    return _get(f"{STATS_URL}/season")["data"]


def fetch_window(endpoint: str, season_id: int, start: date, end: date) -> list:
    """All regular season and playoff rows for one season between two dates."""
    exp = (f'seasonId={season_id} and gameTypeId>=2 and gameTypeId<=3 '
           f'and gameDate>="{start}" and gameDate<="{end} 23:59:59"')
    url = (f"{STATS_URL}/{endpoint}?isAggregate=false&isGame=true"
           f"&limit=-1&start=0&cayenneExp={requests.utils.quote(exp)}")
    data = _get(url)
    if data["total"] >= ROW_CAP:
        if start == end:
            raise RuntimeError(f"{endpoint} {start}: one day over the row cap")
        mid = start + (end - start) // 2
        return (fetch_window(endpoint, season_id, start, mid)
                + fetch_window(endpoint, season_id, mid + timedelta(days=1), end))
    if len(data["data"]) != data["total"]:
        raise RuntimeError(f"{endpoint} {start}..{end}: got {len(data['data'])} of {data['total']}")
    return data["data"]


def fetch_season(endpoint: str, rows_per_game: int, season: dict) -> list:
    start = datetime.fromisoformat(season["startDate"]).date() - timedelta(days=1)
    end = min(datetime.fromisoformat(season["endDate"]).date() + timedelta(days=1), date.today())
    games = season["totalRegularSeasonGames"] + (season["totalPlayoffGames"] or 0)
    n_windows = max(1, -(-games * rows_per_game // ROWS_PER_WINDOW))
    step = -(-((end - start).days + 1) // n_windows)
    windows = []
    while start <= end:
        stop = min(start + timedelta(days=step - 1), end)
        windows.append((start, stop))
        start = stop + timedelta(days=1)
    with ThreadPoolExecutor(WORKERS) as pool:
        chunks = pool.map(lambda w: fetch_window(endpoint, season["id"], *w), windows)
    rows = [r for chunk in chunks for r in chunk]
    for r in rows:
        r["seasonId"] = season["id"]
        r["gameTypeId"] = int(str(r["gameId"])[4:6])
    return rows


def store(conn: sqlite3.Connection, table: str, season_id: int, rows: list):
    if not rows:
        return
    cols = sorted({k for r in rows for k in r})
    existing = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    if not existing:
        conn.execute(f"CREATE TABLE {table} ({', '.join(cols)}, "
                     "PRIMARY KEY (gameId, playerId))")
        conn.execute(f"CREATE INDEX {table}_player ON {table} (playerId, gameDate)")
        conn.execute(f"CREATE INDEX {table}_season ON {table} (seasonId, gameTypeId)")
    for c in cols:
        if existing and c not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {c}")
    conn.execute(f"DELETE FROM {table} WHERE seasonId = ?", (season_id,))
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})",
        [tuple(r.get(c) for c in cols) for r in rows])


def main(only=None):
    conn = sqlite3.connect(DB_PATH)
    seasons = fetch_seasons()
    conn.execute("CREATE TABLE IF NOT EXISTS seasons (id INTEGER PRIMARY KEY, "
                 "formattedSeasonId, startDate, regularSeasonEndDate, endDate, "
                 "numberOfGames, totalRegularSeasonGames, fetched_at)")
    done = {sid: (fetched, end) for sid, fetched, end in
            conn.execute("SELECT id, fetched_at, endDate FROM seasons")}
    for season in seasons:
        sid = season["id"]
        if only and str(sid) not in only:
            continue
        if not only and sid in done and done[sid][0] and done[sid][0] > done[sid][1]:
            continue  # fetched after the season ended: nothing new to get
        counts = []
        for table, (endpoint, rows_per_game) in ENDPOINTS.items():
            rows = fetch_season(endpoint, rows_per_game, season)
            store(conn, table, sid, rows)
            counts.append(f"{len(rows):>6} {table}")
        conn.execute("INSERT OR REPLACE INTO seasons VALUES (?,?,?,?,?,?,?,?)",
                     (sid, season["formattedSeasonId"], season["startDate"],
                      season["regularSeasonEndDate"], season["endDate"],
                      season["numberOfGames"], season["totalRegularSeasonGames"],
                      datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        print(season["formattedSeasonId"], *counts, flush=True)
    conn.close()


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
