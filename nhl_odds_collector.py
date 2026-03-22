"""
NHL Odds Collector — the-odds-api.com

Fetches game odds (h2h + totals) and player shots-on-goal props.
Stores in nhl_data.db alongside existing NHL game/player data.

Used by model.py for:
  - game_total: over/under on total goals (pace proxy)
  - implied_team_total: team's expected goals from moneyline + total
  - sog_prop_line: market consensus SOG line per player (direct signal)
"""

import logging
import sqlite3
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Rotate API keys — primary exhausted, use backup keys until April 23
_API_KEYS = [
    "ae39819ad9390ec56cdedb47c907195c",
    "789463d60f7b764390177f8ac6e91154",
    "29d902f2352064232e3d4022f78610b3",  # original (exhausted)
]
API_KEY = _API_KEYS[0]
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"

DB_PATH = Path(__file__).resolve().parent / "nhl_data.db"

# Odds API full team name → NHL abbreviation
ODDS_TO_NHL = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St Louis Blues": "STL",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

NHL_TO_ODDS = {v: k for k, v in ODDS_TO_NHL.items()
               if "." not in k and "\u00e9" not in k}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def create_odds_tables(conn=None):
    """Create NHL odds tables in the shared nhl_data.db."""
    close = conn is None
    if conn is None:
        conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nhl_game_odds (
            event_id      TEXT,
            game_date     TEXT,
            home_team     TEXT,
            away_team     TEXT,
            home_abbrev   TEXT,
            away_abbrev   TEXT,
            commence_time TEXT,
            bookmaker     TEXT,
            market        TEXT,
            outcome_name  TEXT,
            outcome_price INTEGER,
            outcome_point REAL,
            snapshot_time TEXT,
            PRIMARY KEY (event_id, bookmaker, market, outcome_name)
        );

        CREATE TABLE IF NOT EXISTS nhl_player_props (
            event_id      TEXT,
            game_date     TEXT,
            home_abbrev   TEXT,
            away_abbrev   TEXT,
            commence_time TEXT,
            bookmaker     TEXT,
            player_name   TEXT,
            market        TEXT,
            over_under    TEXT,
            price         INTEGER,
            line          REAL,
            snapshot_time TEXT,
            PRIMARY KEY (event_id, bookmaker, player_name, over_under)
        );

        CREATE TABLE IF NOT EXISTS nhl_odds_events (
            event_id      TEXT PRIMARY KEY,
            game_date     TEXT,
            home_team     TEXT,
            away_team     TEXT,
            home_abbrev   TEXT,
            away_abbrev   TEXT,
            commence_time TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_nhl_game_odds_date
            ON nhl_game_odds(game_date);
        CREATE INDEX IF NOT EXISTS idx_nhl_props_date
            ON nhl_player_props(game_date);
        CREATE INDEX IF NOT EXISTS idx_nhl_props_player
            ON nhl_player_props(player_name);
        CREATE INDEX IF NOT EXISTS idx_nhl_odds_events_date
            ON nhl_odds_events(game_date);
    """)
    conn.commit()
    if close:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_get(url, params, max_retries=3):
    """Make API request with retry logic and automatic key rotation."""
    global API_KEY
    for attempt in range(max_retries):
        try:
            params["apiKey"] = API_KEY
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                remaining = resp.headers.get("x-requests-remaining", "?")
                logger.info("Odds API credits remaining: %s", remaining)
                return resp.json(), remaining
            elif resp.status_code == 401 and "OUT_OF_USAGE_CREDITS" in resp.text:
                # Rotate to next API key
                idx = _API_KEYS.index(API_KEY) if API_KEY in _API_KEYS else -1
                if idx + 1 < len(_API_KEYS):
                    API_KEY = _API_KEYS[idx + 1]
                    logger.warning("API key exhausted, rotating to next key")
                    continue
                else:
                    logger.warning("All API keys exhausted")
                    return None, "N/A"
            elif resp.status_code == 429:
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
            elif resp.status_code == 422:
                return None, "N/A"
            else:
                logger.warning("API error %d: %s",
                               resp.status_code, resp.text[:200])
                time.sleep(5)
        except requests.RequestException as e:
            logger.warning("Request failed (attempt %d): %s", attempt + 1, e)
            time.sleep(5)
    return None, "N/A"


def _team_abbrev(full_name):
    """Convert Odds API team name to NHL abbreviation."""
    return ODDS_TO_NHL.get(full_name, full_name)


def american_to_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100)
    elif american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 0.5


# ---------------------------------------------------------------------------
# Live / Upcoming odds collection
# ---------------------------------------------------------------------------

def fetch_upcoming_game_odds():
    """Fetch game-level odds (h2h + totals) for all upcoming NHL games."""
    data, _ = _api_get(
        f"{BASE_URL}/sports/{SPORT}/odds",
        {
            "apiKey": API_KEY,
            "regions": "us,eu",
            "markets": "h2h,totals,spreads",
            "oddsFormat": "american",
        },
    )
    return data if data else []


def fetch_player_props(event_id):
    """Fetch player SOG props for a specific event."""
    data, _ = _api_get(
        f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds",
        {
            "apiKey": API_KEY,
            "regions": "us,eu",
            "markets": "player_shots_on_goal",
            "oddsFormat": "american",
        },
    )
    return data


def collect_todays_odds(progress_callback=None):
    """Collect game odds and player SOG props for today's/upcoming games.

    Main entry point for daily odds collection. Should be called before
    games start (e.g., during morning refresh).
    """
    conn = get_db()
    create_odds_tables(conn)

    today = date.today().isoformat()

    if progress_callback:
        progress_callback("Fetching NHL game odds...")

    events = fetch_upcoming_game_odds()
    if not events:
        logger.info("No upcoming NHL events found")
        conn.close()
        return {"events": 0, "with_props": 0}

    # Filter to today's games (including late west-coast starts on next UTC day)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    today_events = []
    for ev in events:
        ct = ev.get("commence_time", "")
        if ct.startswith(today) or ct.startswith(tomorrow):
            today_events.append(ev)

    # Fallback: any event within the next 24 hours
    if not today_events:
        now = datetime.utcnow()
        for ev in events:
            try:
                ct = datetime.fromisoformat(
                    ev["commence_time"].replace("Z", "+00:00")
                )
                delta = (ct.replace(tzinfo=None) - now).total_seconds()
                if 0 <= delta <= 86400:
                    today_events.append(ev)
            except (ValueError, KeyError):
                pass

    logger.info("Found %d NHL games for %s", len(today_events), today)

    total_events = 0
    total_props = 0

    for ev in today_events:
        event_id = ev["id"]
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        home_abbrev = _team_abbrev(home)
        away_abbrev = _team_abbrev(away)
        commence = ev.get("commence_time", "")
        snapshot = datetime.utcnow().isoformat() + "Z"

        # Store event
        try:
            conn.execute(
                """INSERT OR REPLACE INTO nhl_odds_events
                   (event_id, game_date, home_team, away_team,
                    home_abbrev, away_abbrev, commence_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (event_id, today, home, away,
                 home_abbrev, away_abbrev, commence),
            )
        except sqlite3.Error:
            pass

        # Store game-level odds from bulk response
        for bk in ev.get("bookmakers", []):
            bk_key = bk["key"]
            for mkt in bk.get("markets", []):
                mkt_key = mkt["key"]
                for outcome in mkt.get("outcomes", []):
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO nhl_game_odds
                               (event_id, game_date, home_team, away_team,
                                home_abbrev, away_abbrev, commence_time,
                                bookmaker, market, outcome_name,
                                outcome_price, outcome_point, snapshot_time)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (event_id, today, home, away,
                             home_abbrev, away_abbrev, commence,
                             bk_key, mkt_key, outcome["name"],
                             int(outcome.get("price", 0)),
                             outcome.get("point"), snapshot),
                        )
                    except sqlite3.Error:
                        pass

        total_events += 1

        # Fetch player SOG props for this event
        if progress_callback:
            progress_callback(
                f"Fetching SOG props: {away_abbrev} @ {home_abbrev}..."
            )

        props_data = fetch_player_props(event_id)
        if props_data:
            has_props = False
            for bk in props_data.get("bookmakers", []):
                bk_key = bk["key"]
                for mkt in bk.get("markets", []):
                    if mkt["key"] != "player_shots_on_goal":
                        continue
                    has_props = True
                    for outcome in mkt.get("outcomes", []):
                        player_name = outcome.get("description", "Unknown")
                        try:
                            conn.execute(
                                """INSERT OR REPLACE INTO nhl_player_props
                                   (event_id, game_date, home_abbrev,
                                    away_abbrev, commence_time, bookmaker,
                                    player_name, market, over_under,
                                    price, line, snapshot_time)
                                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (event_id, today, home_abbrev, away_abbrev,
                                 commence, bk_key, player_name,
                                 "player_shots_on_goal",
                                 outcome["name"],
                                 int(outcome.get("price", 0)),
                                 outcome.get("point"), snapshot),
                            )
                        except sqlite3.Error:
                            pass
            if has_props:
                total_props += 1

        time.sleep(0.15)

    conn.commit()
    conn.close()

    logger.info("NHL odds collected: %d events, %d with SOG props",
                total_events, total_props)
    return {"events": total_events, "with_props": total_props}


# ---------------------------------------------------------------------------
# Historical odds collection (for backfilling training data)
# ---------------------------------------------------------------------------

def get_historical_events(game_date):
    """Get all NHL events for a given date via historical endpoint."""
    dt = f"{game_date}T18:00:00Z"
    data, remaining = _api_get(
        f"{BASE_URL}/historical/sports/{SPORT}/events",
        {"apiKey": API_KEY, "date": dt},
    )
    if not data:
        return []

    events = data.get("data", [])
    next_day = (datetime.strptime(game_date, "%Y-%m-%d")
                + timedelta(days=1)).strftime("%Y-%m-%d")
    return [
        ev for ev in events
        if ev.get("commence_time", "").startswith(game_date)
        or ev.get("commence_time", "").startswith(next_day)
    ]


def get_historical_event_odds(event_id, query_time):
    """Get combined odds for a historical event."""
    data, _ = _api_get(
        f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds",
        {
            "apiKey": API_KEY,
            "regions": "us,eu",
            "markets": "h2h,totals,spreads,player_shots_on_goal",
            "date": query_time,
            "oddsFormat": "american",
        },
    )
    return data


def _store_historical_odds(conn, event, odds_data, game_date):
    """Parse and store historical odds for one event."""
    event_id = event["id"]
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    home_abbrev = _team_abbrev(home)
    away_abbrev = _team_abbrev(away)
    commence = event.get("commence_time", "")
    snapshot = odds_data.get("timestamp", "")

    try:
        conn.execute(
            """INSERT OR IGNORE INTO nhl_odds_events
               (event_id, game_date, home_team, away_team,
                home_abbrev, away_abbrev, commence_time)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event_id, game_date, home, away,
             home_abbrev, away_abbrev, commence),
        )
    except sqlite3.Error:
        pass

    for bk in odds_data.get("data", {}).get("bookmakers", []):
        bk_key = bk["key"]
        for mkt in bk.get("markets", []):
            mkt_key = mkt["key"]

            if mkt_key in ("h2h", "totals", "spreads"):
                for outcome in mkt.get("outcomes", []):
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO nhl_game_odds
                               (event_id, game_date, home_team, away_team,
                                home_abbrev, away_abbrev, commence_time,
                                bookmaker, market, outcome_name,
                                outcome_price, outcome_point, snapshot_time)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (event_id, game_date, home, away,
                             home_abbrev, away_abbrev, commence,
                             bk_key, mkt_key, outcome["name"],
                             int(outcome.get("price", 0)),
                             outcome.get("point"), snapshot),
                        )
                    except sqlite3.Error:
                        pass

            elif mkt_key == "player_shots_on_goal":
                for outcome in mkt.get("outcomes", []):
                    player_name = outcome.get("description", "Unknown")
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO nhl_player_props
                               (event_id, game_date, home_abbrev,
                                away_abbrev, commence_time, bookmaker,
                                player_name, market, over_under,
                                price, line, snapshot_time)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (event_id, game_date, home_abbrev, away_abbrev,
                             commence, bk_key, player_name,
                             "player_shots_on_goal",
                             outcome["name"],
                             int(outcome.get("price", 0)),
                             outcome.get("point"), snapshot),
                        )
                    except sqlite3.Error:
                        pass


def collect_historical_odds(start_date, end_date, progress_callback=None):
    """Collect historical odds for a date range.

    Uses the historical API endpoints (costs more credits).
    """
    conn = get_db()
    create_odds_tables(conn)

    existing = {
        row[0]
        for row in conn.execute(
            "SELECT event_id FROM nhl_odds_events"
        ).fetchall()
    }

    current = (start_date if isinstance(start_date, date)
               else datetime.strptime(start_date, "%Y-%m-%d").date())
    end = (end_date if isinstance(end_date, date)
           else datetime.strptime(end_date, "%Y-%m-%d").date())

    total = 0
    while current <= end:
        game_date = current.strftime("%Y-%m-%d")
        if progress_callback:
            progress_callback(f"NHL odds: {game_date}")

        events = get_historical_events(game_date)
        new_events = [e for e in events if e["id"] not in existing]

        if not new_events:
            current += timedelta(days=1)
            continue

        for ev in new_events:
            commence = ev.get("commence_time", "")
            try:
                ct = datetime.fromisoformat(
                    commence.replace("Z", "+00:00")
                )
                query_time = (ct - timedelta(hours=3)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            except (ValueError, TypeError):
                query_time = f"{game_date}T18:00:00Z"

            odds_data = get_historical_event_odds(ev["id"], query_time)
            if odds_data:
                _store_historical_odds(conn, ev, odds_data, game_date)
                existing.add(ev["id"])
                total += 1

            time.sleep(0.15)

        conn.commit()
        current += timedelta(days=1)

    conn.close()
    logger.info("Historical NHL odds collected: %d events", total)
    return total


# ---------------------------------------------------------------------------
# Query helpers (used by model.py and app.py)
# ---------------------------------------------------------------------------

def _match_key(name):
    """Convert player name to (first_initial, last_name) for cross-source matching."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.strip().lower().replace(".", "")
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0], " ".join(parts[1:]))
    return ("", name)


def get_consensus_sog_line(player_name, game_date=None, team=None):
    """Get consensus SOG line for a player on a date.

    Matches abbreviated names ('N. MacKinnon') to full names
    ('Nathan MacKinnon') via first initial + last name.
    When *team* (NHL abbreviation) is provided, only match props from
    games that involve that team — prevents cross-team collisions
    (e.g. Anders Lee NYI vs Andre Lee LAK).

    Returns dict with line, over_odds, under_odds, num_books,
    and sharp consensus data — or None.
    """
    conn = get_db()
    if game_date is None:
        game_date = date.today().isoformat()

    # Build optional team filter
    team_clause = ""
    params: list = [player_name, game_date]
    if team:
        team_clause = " AND (home_abbrev = ? OR away_abbrev = ?)"
        params.extend([team, team])

    # Try exact match first, then fall back to initial+last match
    rows = conn.execute(
        f"""SELECT bookmaker, over_under, price, line FROM nhl_player_props
           WHERE player_name = ? AND game_date = ?{team_clause}""",
        params,
    ).fetchall()

    if not rows:
        # Fuzzy match: load all props for date, match on initial+last
        target_key = _match_key(player_name)
        fuzzy_params: list = [game_date]
        fuzzy_team_clause = ""
        if team:
            fuzzy_team_clause = " AND (home_abbrev = ? OR away_abbrev = ?)"
            fuzzy_params.extend([team, team])
        all_rows = conn.execute(
            f"""SELECT player_name, bookmaker, over_under, price, line
               FROM nhl_player_props WHERE game_date = ?{fuzzy_team_clause}""",
            fuzzy_params,
        ).fetchall()
        rows = [r for r in all_rows if _match_key(r["player_name"]) == target_key]

    conn.close()

    if not rows:
        return None

    overs = [(r["price"], r["line"]) for r in rows
             if r["over_under"] == "Over"]
    unders = [(r["price"], r["line"]) for r in rows
              if r["over_under"] == "Under"]

    if not overs:
        return None

    line_counts = Counter(r["line"] for r in rows)
    consensus_line = line_counts.most_common(1)[0][0]

    over_at = [p for p, l in overs if l == consensus_line]
    under_at = [p for p, l in unders if l == consensus_line]

    # Compute sharp consensus: vig-free implied prob from sharp books
    from collections import defaultdict
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
        imp_ov = american_to_prob(bp["Over"])
        imp_un = american_to_prob(bp["Under"])
        total = imp_ov + imp_un
        if total > 0:
            sharp_probs.append(imp_ov / total)

    sharp_over = None
    n_sharp = 0
    if sharp_probs:
        sharp_over = sum(sharp_probs) / len(sharp_probs)
        n_sharp = len(sharp_probs)

    # BetMGM prices for PlayAlberta estimation
    bmg = by_book.get("betmgm")
    bmg_over = bmg.get("Over") if bmg else None
    bmg_under = bmg.get("Under") if bmg else None
    pa_over, pa_under = None, None
    if bmg_over is not None and bmg_under is not None:
        pa_over, pa_under = betmgm_to_playalberta(bmg_over, bmg_under)

    return {
        "line": consensus_line,
        "over_odds": (round(sum(over_at) / len(over_at))
                      if over_at else None),
        "under_odds": (round(sum(under_at) / len(under_at))
                       if under_at else None),
        "num_books": len(over_at),
        # Sharp consensus
        "sharp_prob_over": round(sharp_over, 4) if sharp_over else None,
        "sharp_prob_under": round(1.0 - sharp_over, 4) if sharp_over else None,
        "n_sharp_books": n_sharp,
        # BetMGM / PlayAlberta
        "betmgm_over": bmg_over,
        "betmgm_under": bmg_under,
        "pa_over_est": pa_over,
        "pa_under_est": pa_under,
    }


def get_game_odds_for_date(game_date=None, bookmaker="draftkings"):
    """Get moneyline and totals for all games on a date."""
    conn = get_db()
    create_odds_tables(conn)
    if game_date is None:
        game_date = date.today().isoformat()

    rows = conn.execute(
        """SELECT event_id, home_team, away_team, home_abbrev, away_abbrev,
                  market, outcome_name, outcome_price, outcome_point
           FROM nhl_game_odds
           WHERE game_date = ? AND bookmaker = ?""",
        (game_date, bookmaker),
    ).fetchall()

    # Fall back to any bookmaker if preferred not available
    if not rows:
        rows = conn.execute(
            """SELECT event_id, home_team, away_team, home_abbrev, away_abbrev,
                      market, outcome_name, outcome_price, outcome_point
               FROM nhl_game_odds WHERE game_date = ?""",
            (game_date,),
        ).fetchall()
    conn.close()

    games = {}
    for r in rows:
        eid = r["event_id"]
        if eid not in games:
            games[eid] = {
                "event_id": eid,
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_abbrev": r["home_abbrev"],
                "away_abbrev": r["away_abbrev"],
            }
        g = games[eid]

        if r["market"] == "h2h":
            if r["outcome_name"] == r["home_team"]:
                g["home_ml"] = r["outcome_price"]
            else:
                g["away_ml"] = r["outcome_price"]
        elif r["market"] == "totals":
            g[f"total_{r['outcome_name'].lower()}"] = r["outcome_price"]
            g["total_line"] = r["outcome_point"]
        elif r["market"] == "spreads":
            if r["outcome_name"] == r["home_team"]:
                g["home_spread_point"] = r["outcome_point"]
                g["home_spread_price"] = r["outcome_price"]
            else:
                g["away_spread_point"] = r["outcome_point"]
                g["away_spread_price"] = r["outcome_price"]

    return list(games.values())


def get_game_context(home_abbrev, away_abbrev, game_date=None):
    """Get derived odds features for a specific game.

    Returns dict with game_total, implied_home_total, implied_away_total,
    home_win_prob, away_win_prob — or None if no odds available.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    game_odds = get_game_odds_for_date(game_date)

    for g in game_odds:
        if (g.get("home_abbrev") == home_abbrev
                and g.get("away_abbrev") == away_abbrev):
            result = {}

            total_line = g.get("total_line")
            if total_line is not None:
                result["game_total"] = total_line

            home_ml = g.get("home_ml")
            away_ml = g.get("away_ml")
            if home_ml is not None and away_ml is not None:
                home_imp = american_to_prob(home_ml)
                away_imp = american_to_prob(away_ml)
                total_imp = home_imp + away_imp
                if total_imp > 0:
                    result["home_win_prob"] = round(
                        home_imp / total_imp, 4)
                    result["away_win_prob"] = round(
                        away_imp / total_imp, 4)

            if total_line is not None and "home_win_prob" in result:
                hw = result["home_win_prob"]
                aw = result["away_win_prob"]
                result["implied_home_total"] = round(total_line * hw, 2)
                result["implied_away_total"] = round(total_line * aw, 2)

            return result if result else None

    return None


def get_all_player_props_for_date(game_date=None):
    """Get all player SOG props for a date, keyed by player name.

    Returns dict of player_name -> {line, over_odds, under_odds, num_books}.
    """
    conn = get_db()
    create_odds_tables(conn)
    if game_date is None:
        game_date = date.today().isoformat()

    rows = conn.execute(
        """SELECT player_name, over_under, price, line
           FROM nhl_player_props WHERE game_date = ?""",
        (game_date,),
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    from collections import defaultdict
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_name"]].append(r)

    results = {}
    for name, player_rows in by_player.items():
        overs = [(r["price"], r["line"]) for r in player_rows
                 if r["over_under"] == "Over"]
        unders = [(r["price"], r["line"]) for r in player_rows
                  if r["over_under"] == "Under"]
        if not overs:
            continue

        line_counts = Counter(r["line"] for r in player_rows)
        consensus = line_counts.most_common(1)[0][0]

        over_at = [p for p, l in overs if l == consensus]
        under_at = [p for p, l in unders if l == consensus]

        results[name] = {
            "line": consensus,
            "over_odds": (round(sum(over_at) / len(over_at))
                          if over_at else None),
            "under_odds": (round(sum(under_at) / len(under_at))
                           if under_at else None),
            "num_books": len(over_at),
        }

    return results


# ---------------------------------------------------------------------------
# Bulk odds data loaders (used by model.py for training)
# ---------------------------------------------------------------------------

def load_game_odds_bulk():
    """Load all game odds into a lookup dict.

    Returns dict of (game_date, home_abbrev) -> {
        game_total, home_win_prob, away_win_prob,
        implied_home_total, implied_away_total
    }
    """
    conn = get_db()
    create_odds_tables(conn)

    rows = conn.execute("""
        SELECT game_date, home_abbrev, away_abbrev, home_team, away_team,
               market, outcome_name, outcome_price, outcome_point
        FROM nhl_game_odds
    """).fetchall()
    conn.close()

    # Build raw per-game data, preferring DraftKings
    game_raw = {}
    for r in rows:
        key = (r["game_date"], r["home_abbrev"])
        if key not in game_raw:
            game_raw[key] = {}
        g = game_raw[key]

        if r["market"] == "h2h":
            if r["outcome_name"] == r["home_team"]:
                g["home_ml"] = r["outcome_price"]
            elif r["outcome_name"] == r["away_team"]:
                g["away_ml"] = r["outcome_price"]
        elif r["market"] == "totals":
            if r["outcome_point"] is not None:
                g["total_line"] = r["outcome_point"]
        elif r["market"] == "spreads":
            if r["outcome_name"] == r["home_team"]:
                g["home_spread_point"] = r["outcome_point"]
                g["home_spread_price"] = r["outcome_price"]
            elif r["outcome_name"] == r["away_team"]:
                g["away_spread_point"] = r["outcome_point"]
                g["away_spread_price"] = r["outcome_price"]

    # Compute derived features
    result = {}
    for key, g in game_raw.items():
        ctx = {}
        total = g.get("total_line")
        if total is not None:
            ctx["game_total"] = total

        home_ml = g.get("home_ml")
        away_ml = g.get("away_ml")
        if home_ml is not None and away_ml is not None:
            hp = american_to_prob(home_ml)
            ap = american_to_prob(away_ml)
            total_imp = hp + ap
            if total_imp > 0:
                hw = hp / total_imp
                aw = ap / total_imp
                ctx["home_win_prob"] = hw
                ctx["away_win_prob"] = aw
                if total is not None:
                    ctx["implied_home_total"] = total * hw
                    ctx["implied_away_total"] = total * aw

        # Add spread data
        sp = g.get("home_spread_point")
        if sp is not None:
            ctx["home_spread_point"] = sp
            ctx["home_spread_price"] = g.get("home_spread_price")
            ctx["away_spread_point"] = g.get("away_spread_point")
            ctx["away_spread_price"] = g.get("away_spread_price")

        # Add raw moneyline prices
        if home_ml is not None:
            ctx["home_ml"] = home_ml
            ctx["away_ml"] = away_ml

        if ctx:
            result[key] = ctx

    return result


def load_player_props_bulk():
    """Load all player SOG prop lines into a lookup dict.

    Returns dict of (game_date, player_name) -> consensus_line (float).
    """
    conn = get_db()
    create_odds_tables(conn)

    rows = conn.execute(
        "SELECT game_date, player_name, line FROM nhl_player_props"
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["game_date"], r["player_name"])].append(r["line"])

    result = {}
    for key, lines in grouped.items():
        consensus = Counter(lines).most_common(1)[0][0]
        result[key] = consensus

    return result


# ---------------------------------------------------------------------------
# Sharp book consensus — vig-free implied probabilities
# ---------------------------------------------------------------------------

# Sharpness ranking by calibration RMSE (lower = sharper):
#   1. BetOnlineAg  (0.0077)  — offshore, low vig, best calibrated
#   2. DraftKings   (0.0118)  — lowest raw vig (6.37%)
#   3. FanDuel      (0.0163)  — lowest Brier score (0.24283)
# Soft books (higher vig, worse calibration):
#   - BetMGM / PlayAlberta (7.61% vig, 0.0098 cal RMSE)
#   - BetRivers (6.70% vig, worst Brier)
SHARP_BOOKS = ("pinnacle", "betonlineag", "draftkings", "fanduel")
SOFT_BOOKS = ("betmgm",)  # PlayAlberta mirrors BetMGM lines


def load_sharp_consensus_bulk():
    """Load vig-free implied probability from sharp books per player-game-line.

    For each (game_date, player_name, line), averages the vig-removed
    over-probability from the 3 sharpest books.

    Returns dict of (game_date, player_name) -> {
        line, sharp_prob_over, sharp_prob_under, n_sharp_books,
        soft_over_price, soft_under_price  (BetMGM prices, if available)
    }
    """
    conn = get_db()
    create_odds_tables(conn)

    rows = conn.execute("""
        SELECT game_date, player_name, bookmaker, over_under, price, line
        FROM nhl_player_props
    """).fetchall()
    conn.close()

    if not rows:
        return {}

    from collections import defaultdict

    # Group: (game_date, player_name, bookmaker, line) -> {over_price, under_price}
    paired = defaultdict(dict)
    for r in rows:
        key = (r["game_date"], r["player_name"], r["bookmaker"], r["line"])
        if r["over_under"] == "Over":
            paired[key]["over"] = int(r["price"])
        else:
            paired[key]["under"] = int(r["price"])

    # Per (game_date, player_name, line): collect sharp fair probs + soft prices
    combo = defaultdict(lambda: {
        "sharp_probs": [], "soft_over": None, "soft_under": None,
    })

    for (gd, pname, book, line), prices in paired.items():
        ov = prices.get("over")
        un = prices.get("under")
        if ov is None or un is None:
            continue

        imp_ov = american_to_prob(ov)
        imp_un = american_to_prob(un)
        total_imp = imp_ov + imp_un
        if total_imp <= 0:
            continue
        fair_over = imp_ov / total_imp  # vig-removed

        ck = (gd, pname, line)
        if book in SHARP_BOOKS:
            combo[ck]["sharp_probs"].append(fair_over)
        if book in SOFT_BOOKS:
            combo[ck]["soft_over"] = ov
            combo[ck]["soft_under"] = un

    # Determine consensus line per (game_date, player_name) then output
    # Group all lines for each player-date to pick the consensus line
    by_player_date = defaultdict(list)
    for (gd, pname, line), data in combo.items():
        by_player_date[(gd, pname)].append((line, data))

    result = {}
    for (gd, pname), entries in by_player_date.items():
        # Pick the line with the most sharp books providing data
        best = max(entries, key=lambda e: len(e[1]["sharp_probs"]))
        line, data = best

        if not data["sharp_probs"]:
            # Fall back to any line that has data
            for l2, d2 in entries:
                if d2["sharp_probs"]:
                    line, data = l2, d2
                    break
            else:
                continue

        sharp_over = float(sum(data["sharp_probs"]) / len(data["sharp_probs"]))

        result[(gd, pname)] = {
            "line": line,
            "sharp_prob_over": sharp_over,
            "sharp_prob_under": 1.0 - sharp_over,
            "n_sharp_books": len(data["sharp_probs"]),
            "soft_over_price": data["soft_over"],
            "soft_under_price": data["soft_under"],
        }

    return result


def load_per_book_props_bulk():
    """Load player props per bookmaker for walk-forward analysis.

    Returns dict of (game_date, player_name_key, line, bookmaker) ->
        {over_price, under_price}
    where player_name_key is (first_initial, last_name_lower) for matching.
    """
    conn = get_db()
    create_odds_tables(conn)

    rows = conn.execute("""
        SELECT game_date, player_name, bookmaker, over_under, price, line
        FROM nhl_player_props
    """).fetchall()
    conn.close()

    if not rows:
        return {}

    import unicodedata

    def _match_key(name):
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        name = name.strip().lower().replace(".", "")
        parts = name.split()
        if len(parts) >= 2:
            return (parts[0][0], " ".join(parts[1:]))
        return ("", name)

    result = {}
    for r in rows:
        ini, last = _match_key(r["player_name"])
        key = (r["game_date"], ini, last, r["line"], r["bookmaker"])
        if key not in result:
            result[key] = {}
        if r["over_under"] == "Over":
            result[key]["over_price"] = int(r["price"])
        else:
            result[key]["under_price"] = int(r["price"])

    return result


# ---------------------------------------------------------------------------
# BetMGM odds + PlayAlberta adjustment
# ---------------------------------------------------------------------------

# PlayAlberta benchmarks to BetMGM with ~1% extra vig on overs.
# Measured on 26 lines (Mar 17 2026 NYI@TOR):
#   avg prob diff = 0.42pp, 19/26 within ±2 odds points.
#   Over-side: PA adds ~0.8pp implied probability
#   Under-side: PA subtracts ~0.4pp implied probability
PA_VIG_OVER_PP = 0.008   # extra implied prob added to overs
PA_VIG_UNDER_PP = -0.004  # implied prob shift on unders


def _prob_to_american(prob):
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob > 0.5:
        return round(-100.0 * prob / (1.0 - prob))
    return round(100.0 * (1.0 - prob) / prob)


def _american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5


def betmgm_to_playalberta(over_odds, under_odds):
    """Estimate PlayAlberta odds from BetMGM odds.

    PlayAlberta benchmarks to BetMGM but adds ~0.8% vig on overs
    and shifts unders by ~-0.4%. Returns (pa_over_odds, pa_under_odds).
    """
    pa_over = over_odds
    pa_under = under_odds

    if over_odds is not None:
        prob = _american_to_prob(over_odds)
        pa_over = _prob_to_american(prob + PA_VIG_OVER_PP)

    if under_odds is not None:
        prob = _american_to_prob(under_odds)
        pa_under = _prob_to_american(prob + PA_VIG_UNDER_PP)

    return pa_over, pa_under


def get_betmgm_player_props(game_date=None):
    """Get BetMGM-specific SOG props with PlayAlberta estimates.

    Returns dict of player_name -> {
        line, betmgm_over, betmgm_under,
        pa_over_est, pa_under_est
    }
    """
    conn = get_db()
    create_odds_tables(conn)
    if game_date is None:
        game_date = date.today().isoformat()

    rows = conn.execute(
        """SELECT player_name, over_under, price, line
           FROM nhl_player_props
           WHERE game_date = ? AND bookmaker = 'betmgm'""",
        (game_date,),
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    from collections import defaultdict
    by_player = defaultdict(lambda: {"overs": [], "unders": []})
    for r in rows:
        if r["over_under"] == "Over":
            by_player[r["player_name"]]["overs"].append(
                (int(r["price"]), r["line"])
            )
        else:
            by_player[r["player_name"]]["unders"].append(
                (int(r["price"]), r["line"])
            )

    results = {}
    for name, data in by_player.items():
        if not data["overs"]:
            continue

        # Use the most common line
        line_counts = Counter(
            l for _, l in data["overs"] + data["unders"]
        )
        consensus_line = line_counts.most_common(1)[0][0]

        over_prices = [p for p, l in data["overs"] if l == consensus_line]
        under_prices = [p for p, l in data["unders"] if l == consensus_line]

        mgm_over = over_prices[0] if over_prices else None
        mgm_under = under_prices[0] if under_prices else None

        pa_over, pa_under = betmgm_to_playalberta(mgm_over, mgm_under)

        results[name] = {
            "line": consensus_line,
            "betmgm_over": mgm_over,
            "betmgm_under": mgm_under,
            "pa_over_est": pa_over,
            "pa_under_est": pa_under,
        }

    return results


# ---------------------------------------------------------------------------
# Cache for today's odds (avoid redundant API calls)
# ---------------------------------------------------------------------------

_odds_cache_date = None


def ensure_todays_odds():
    """Ensure today's odds are in the DB. Fetches from API if needed."""
    global _odds_cache_date
    today = date.today().isoformat()

    if _odds_cache_date == today:
        return

    conn = get_db()
    create_odds_tables(conn)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM nhl_odds_events WHERE game_date = ?",
        (today,),
    ).fetchone()
    conn.close()

    if row and row["cnt"] > 0:
        _odds_cache_date = today
        return

    logger.info("Fetching today's NHL odds from API...")
    collect_todays_odds()
    _odds_cache_date = today


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result = collect_todays_odds()
    print(f"Result: {result}")

    # Show what we got
    props = get_all_player_props_for_date()
    print(f"\nPlayer SOG props ({len(props)} players):")
    for name, info in sorted(props.items(),
                              key=lambda x: x[1]["line"],
                              reverse=True)[:20]:
        print(f"  {name}: {info['line']} SOG "
              f"(O {info['over_odds']}/U {info['under_odds']}, "
              f"{info['num_books']} books)")
