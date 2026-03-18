"""
MLB Historical Odds Collector — the-odds-api.com

Pulls 2024 season data:
- Game odds (moneyline h2h + totals)
- Pitcher strikeout props (over/under lines from multiple books)

Stores in mlb_data.db for model training, backtesting, and edge detection.
"""

import logging
import sqlite3
import time
from datetime import date, timedelta, datetime

import requests

logger = logging.getLogger(__name__)

API_KEY = "29d902f2352064232e3d4022f78610b3"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"

DB_PATH = "mlb_data.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_odds_tables(conn: sqlite3.Connection):
    """Create tables for odds data."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mlb_game_odds (
            event_id     TEXT,
            game_date    TEXT,
            home_team    TEXT,
            away_team    TEXT,
            commence_time TEXT,
            bookmaker    TEXT,
            market       TEXT,
            outcome_name TEXT,
            outcome_price INTEGER,
            outcome_point REAL,
            snapshot_time TEXT,
            PRIMARY KEY (event_id, bookmaker, market, outcome_name)
        );

        CREATE TABLE IF NOT EXISTS mlb_pitcher_props (
            event_id      TEXT,
            game_date     TEXT,
            home_team     TEXT,
            away_team     TEXT,
            commence_time TEXT,
            bookmaker     TEXT,
            pitcher_name  TEXT,
            market        TEXT,
            over_under    TEXT,
            price         INTEGER,
            line          REAL,
            snapshot_time TEXT,
            PRIMARY KEY (event_id, bookmaker, pitcher_name, over_under)
        );

        CREATE TABLE IF NOT EXISTS mlb_odds_events (
            event_id      TEXT PRIMARY KEY,
            game_date     TEXT,
            home_team     TEXT,
            away_team     TEXT,
            commence_time TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_props_date ON mlb_pitcher_props(game_date);
        CREATE INDEX IF NOT EXISTS idx_props_pitcher ON mlb_pitcher_props(pitcher_name);
        CREATE INDEX IF NOT EXISTS idx_game_odds_date ON mlb_game_odds(game_date);
        CREATE INDEX IF NOT EXISTS idx_odds_events_date ON mlb_odds_events(game_date);
    """)
    conn.commit()


def _api_get(url, params, max_retries=3):
    """Make API request with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                remaining = resp.headers.get("x-requests-remaining", "?")
                return resp.json(), remaining
            elif resp.status_code == 429:
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
            elif resp.status_code == 422:
                # No data for this request
                return None, "N/A"
            else:
                logger.warning("API error %d: %s", resp.status_code, resp.text[:200])
                time.sleep(5)
        except requests.RequestException as e:
            logger.warning("Request failed (attempt %d): %s", attempt + 1, e)
            time.sleep(5)
    return None, "N/A"


def get_historical_events(game_date: str) -> list[dict]:
    """Get all MLB events for a given date.

    Args:
        game_date: ISO date string like '2024-08-10'

    Returns list of event dicts with id, home_team, away_team, commence_time.
    """
    # Query at 14:00 UTC (morning ET) to get pre-game snapshot
    dt = f"{game_date}T14:00:00Z"
    data, remaining = _api_get(
        f"{BASE_URL}/historical/sports/{SPORT}/events",
        {"apiKey": API_KEY, "date": dt},
    )
    if not data:
        return []

    events = data.get("data", [])
    # Filter to events that start on this date (within reason)
    filtered = []
    for ev in events:
        ct = ev.get("commence_time", "")
        # Accept events starting on this date or next day early morning (late west coast)
        if ct.startswith(game_date) or ct.startswith((datetime.strptime(game_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")):
            filtered.append(ev)

    logger.info("  %s: %d events (credits remaining: %s)", game_date, len(filtered), remaining)
    return filtered


def get_event_odds(event_id: str, query_time: str) -> dict | None:
    """Get combined odds (h2h, totals, pitcher_strikeouts) for one event.

    Args:
        event_id: The odds API event ID
        query_time: ISO timestamp to query (should be before game start)

    Returns raw API response data dict, or None.
    """
    data, remaining = _api_get(
        f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds",
        {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "h2h,totals,pitcher_strikeouts",
            "date": query_time,
            "oddsFormat": "american",
        },
    )
    if not data:
        return None
    return data


def _store_event_odds(conn, event, odds_data, game_date):
    """Parse and store odds data for one event."""
    event_id = event["id"]
    home = event["home_team"]
    away = event["away_team"]
    commence = event.get("commence_time", "")
    snapshot = odds_data.get("timestamp", "")

    # Store event
    try:
        conn.execute(
            "INSERT OR IGNORE INTO mlb_odds_events (event_id, game_date, home_team, away_team, commence_time) VALUES (?, ?, ?, ?, ?)",
            (event_id, game_date, home, away, commence),
        )
    except sqlite3.Error:
        pass

    bookmakers = odds_data.get("data", {}).get("bookmakers", [])

    for bk in bookmakers:
        bk_key = bk["key"]
        for mkt in bk.get("markets", []):
            mkt_key = mkt["key"]

            if mkt_key in ("h2h", "totals"):
                # Game-level odds
                for outcome in mkt.get("outcomes", []):
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO mlb_game_odds
                               (event_id, game_date, home_team, away_team, commence_time,
                                bookmaker, market, outcome_name, outcome_price, outcome_point, snapshot_time)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (event_id, game_date, home, away, commence,
                             bk_key, mkt_key, outcome["name"],
                             int(outcome.get("price", 0)),
                             outcome.get("point"),
                             snapshot),
                        )
                    except sqlite3.Error:
                        pass

            elif mkt_key == "pitcher_strikeouts":
                # Player props
                for outcome in mkt.get("outcomes", []):
                    pitcher_name = outcome.get("description", "Unknown")
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO mlb_pitcher_props
                               (event_id, game_date, home_team, away_team, commence_time,
                                bookmaker, pitcher_name, market, over_under, price, line, snapshot_time)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (event_id, game_date, home, away, commence,
                             bk_key, pitcher_name, mkt_key,
                             outcome["name"],  # "Over" or "Under"
                             int(outcome.get("price", 0)),
                             outcome.get("point"),
                             snapshot),
                        )
                    except sqlite3.Error:
                        pass


def collect_season_odds(season: int = 2024, progress_callback=None):
    """Collect full season historical odds.

    Pulls game odds (h2h + totals) and pitcher strikeout props for every game.
    """
    conn = get_db()
    create_odds_tables(conn)

    # Check what we already have
    existing_events = set()
    for row in conn.execute("SELECT event_id FROM mlb_odds_events").fetchall():
        existing_events.add(row[0])
    logger.info("Existing events in DB: %d", len(existing_events))

    if season == 2024:
        start = date(2024, 3, 20)
        end = date(2024, 10, 1)
    else:
        start = date(season, 3, 20)
        end = date(season, 11, 5)

    total_events = 0
    total_props = 0
    total_game_odds = 0
    current = start

    while current <= end:
        game_date = current.strftime("%Y-%m-%d")

        if progress_callback:
            progress_callback(f"Odds: {game_date}")

        events = get_historical_events(game_date)

        new_events = [e for e in events if e["id"] not in existing_events]
        if not new_events:
            current += timedelta(days=1)
            continue

        for ev in new_events:
            eid = ev["id"]
            commence = ev.get("commence_time", "")

            # Query ~3 hours before game time for best pre-game odds
            try:
                ct = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                query_time = (ct - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                query_time = f"{game_date}T14:00:00Z"

            odds_data = get_event_odds(eid, query_time)
            if odds_data:
                _store_event_odds(conn, ev, odds_data, game_date)
                existing_events.add(eid)

                # Count what we got
                bks = odds_data.get("data", {}).get("bookmakers", [])
                has_props = any(
                    m["key"] == "pitcher_strikeouts"
                    for b in bks for m in b.get("markets", [])
                )
                has_game = any(
                    m["key"] in ("h2h", "totals")
                    for b in bks for m in b.get("markets", [])
                )
                if has_props:
                    total_props += 1
                if has_game:
                    total_game_odds += 1
                total_events += 1

            # Small delay to avoid rate limiting
            time.sleep(0.15)

        conn.commit()

        if total_events % 50 == 0 and total_events > 0:
            logger.info("Progress: %d events, %d with props, %d with game odds",
                        total_events, total_props, total_game_odds)

        current += timedelta(days=1)

    conn.commit()
    conn.close()

    logger.info("Collection complete: %d events, %d with props, %d with game odds",
                total_events, total_props, total_game_odds)
    return {
        "events": total_events,
        "with_props": total_props,
        "with_game_odds": total_game_odds,
    }


def get_consensus_line(pitcher_name: str, game_date: str) -> dict | None:
    """Get consensus strikeout line for a pitcher on a date.

    Returns dict with line, over_odds, under_odds, num_books,
    sharp consensus data, and BetMGM/soft book prices.
    """
    conn = get_db()
    rows = conn.execute(
        """SELECT bookmaker, over_under, price, line FROM mlb_pitcher_props
           WHERE pitcher_name = ? AND game_date = ?""",
        (pitcher_name, game_date),
    ).fetchall()
    conn.close()

    if not rows:
        return None

    overs = [(r["price"], r["line"]) for r in rows if r["over_under"] == "Over"]
    unders = [(r["price"], r["line"]) for r in rows if r["over_under"] == "Under"]

    if not overs:
        return None

    from collections import Counter, defaultdict
    line_counts = Counter(r["line"] for r in rows)
    consensus_line = line_counts.most_common(1)[0][0]

    over_at_line = [p for p, l in overs if l == consensus_line]
    under_at_line = [p for p, l in unders if l == consensus_line]

    # Sharp consensus: vig-free implied prob from sharp books
    by_book = defaultdict(dict)
    for r in rows:
        if r["line"] != consensus_line:
            continue
        by_book[r["bookmaker"]][r["over_under"]] = int(r["price"])

    sharp_probs = []
    for book in SHARP_BOOKS:
        bp = by_book.get(book)
        if not bp or "Over" not in bp or "Under" not in bp:
            continue
        imp_ov = _american_to_implied(bp["Over"])
        imp_un = _american_to_implied(bp["Under"])
        total = imp_ov + imp_un
        if total > 0:
            sharp_probs.append(imp_ov / total)

    sharp_over = None
    n_sharp = 0
    if sharp_probs:
        sharp_over = sum(sharp_probs) / len(sharp_probs)
        n_sharp = len(sharp_probs)

    # BetMGM prices (soft book)
    bmg = by_book.get("betmgm")
    bmg_over = bmg.get("Over") if bmg else None
    bmg_under = bmg.get("Under") if bmg else None

    return {
        "line": consensus_line,
        "over_odds": round(sum(over_at_line) / len(over_at_line)) if over_at_line else None,
        "under_odds": round(sum(under_at_line) / len(under_at_line)) if under_at_line else None,
        "num_books": len(over_at_line),
        # Sharp consensus
        "sharp_prob_over": round(sharp_over, 4) if sharp_over else None,
        "sharp_prob_under": round(1.0 - sharp_over, 4) if sharp_over else None,
        "n_sharp_books": n_sharp,
        # BetMGM (soft book target)
        "betmgm_over": bmg_over,
        "betmgm_under": bmg_under,
    }


def get_game_odds_for_date(game_date: str) -> list[dict]:
    """Get moneyline and totals for all games on a date."""
    conn = get_db()
    rows = conn.execute(
        """SELECT event_id, home_team, away_team, bookmaker, market, outcome_name, outcome_price, outcome_point
           FROM mlb_game_odds WHERE game_date = ? AND bookmaker = 'draftkings'""",
        (game_date,),
    ).fetchall()
    conn.close()

    games = {}
    for r in rows:
        eid = r["event_id"]
        if eid not in games:
            games[eid] = {"event_id": eid, "home_team": r["home_team"], "away_team": r["away_team"]}
        g = games[eid]

        if r["market"] == "h2h":
            if r["outcome_name"] == r["home_team"]:
                g["home_ml"] = r["outcome_price"]
            else:
                g["away_ml"] = r["outcome_price"]
        elif r["market"] == "totals":
            g[f"total_{r['outcome_name'].lower()}"] = r["outcome_price"]
            g["total_line"] = r["outcome_point"]

    return list(games.values())


# ---------------------------------------------------------------------------
# Sharp-vs-Soft sportsbook constants
# ---------------------------------------------------------------------------
SHARP_BOOKS = ("fanduel", "draftkings", "betonlineag")
SOFT_BOOKS = ("betmgm", "williamhill_us")


def _american_to_implied(odds: int) -> float:
    """American odds -> implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def load_sharp_consensus_bulk() -> dict:
    """Load sharp-book consensus vig-free over-probability for every pitcher-date.

    Returns dict of (game_date, pitcher_name_normalized) ->
        {line, sharp_prob_over, sharp_prob_under, n_sharp_books,
         soft_over_price, soft_under_price}
    """
    import unicodedata

    def _norm(name):
        nfkd = unicodedata.normalize("NFKD", name)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

    conn = get_db()
    create_odds_tables(conn)
    rows = conn.execute(
        "SELECT game_date, pitcher_name, bookmaker, line, over_under, price "
        "FROM mlb_pitcher_props"
    ).fetchall()
    conn.close()

    # Group by (game_date, pitcher_name_norm, line)
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(dict))
    # grouped[(gd, pname_norm, line)][bookmaker] = {over_price, under_price}
    for r in rows:
        gd = r["game_date"]
        pname = _norm(r["pitcher_name"])
        line = float(r["line"])
        bk = r["bookmaker"]
        key = (gd, pname, line)
        if r["over_under"] == "Over":
            grouped[key][bk]["over_price"] = int(r["price"])
        else:
            grouped[key][bk]["under_price"] = int(r["price"])

    # For each (game_date, pitcher), find the most common line,
    # compute sharp consensus at that line, and capture soft prices
    from collections import Counter
    pitcher_lines = defaultdict(list)  # (gd, pname_norm) -> [line, ...]
    for (gd, pname, line) in grouped:
        pitcher_lines[(gd, pname)].append(line)

    result = {}
    for (gd, pname), lines in pitcher_lines.items():
        # Use the most common line across all books
        line_counts = Counter(lines)
        consensus_line = line_counts.most_common(1)[0][0]

        key = (gd, pname, consensus_line)
        books = grouped[key]

        # Sharp consensus
        sharp_probs = []
        for book in SHARP_BOOKS:
            bp = books.get(book)
            if bp is None:
                continue
            ov = bp.get("over_price")
            un = bp.get("under_price")
            if ov is None or un is None:
                continue
            imp_ov = _american_to_implied(ov)
            imp_un = _american_to_implied(un)
            total = imp_ov + imp_un
            if total > 0:
                sharp_probs.append(imp_ov / total)  # vig-free over prob

        if not sharp_probs:
            continue

        avg_over = sum(sharp_probs) / len(sharp_probs)

        # Soft book prices (try each soft book in order)
        soft_over = None
        soft_under = None
        for book in SOFT_BOOKS:
            bp = books.get(book)
            if bp and bp.get("over_price") is not None:
                soft_over = bp.get("over_price")
                soft_under = bp.get("under_price")
                break

        result[(gd, pname)] = {
            "line": consensus_line,
            "sharp_prob_over": avg_over,
            "sharp_prob_under": 1.0 - avg_over,
            "n_sharp_books": len(sharp_probs),
            "soft_over_price": soft_over,
            "soft_under_price": soft_under,
        }

    logger.info("Sharp consensus loaded for %d pitcher-date entries", len(result))
    return result


def load_per_book_props_bulk() -> dict:
    """Load per-bookmaker pitcher K props for walk-forward analysis.

    Returns dict of (game_date, pitcher_name_normalized, line, bookmaker) ->
        {over_price, under_price}
    """
    import unicodedata

    def _norm(name):
        nfkd = unicodedata.normalize("NFKD", name)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

    conn = get_db()
    create_odds_tables(conn)
    rows = conn.execute(
        "SELECT game_date, pitcher_name, bookmaker, line, over_under, price "
        "FROM mlb_pitcher_props"
    ).fetchall()
    conn.close()

    result = {}
    for r in rows:
        gd = r["game_date"]
        pname = _norm(r["pitcher_name"])
        line = float(r["line"])
        bk = r["bookmaker"]
        key = (gd, pname, line, bk)
        if key not in result:
            result[key] = {}
        if r["over_under"] == "Over":
            result[key]["over_price"] = int(r["price"])
        else:
            result[key]["under_price"] = int(r["price"])

    logger.info("Per-book props loaded: %d entries", len(result))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    collect_season_odds(2024)
