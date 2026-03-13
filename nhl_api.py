"""
NHL API client module.

Provides functions to fetch data from the NHL's public web API with
rate limiting, caching, and error handling.
"""

import time
import logging
from typing import Optional
from datetime import date, datetime

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://api-web.nhle.com/v1"
STATS_URL = "https://api.nhle.com/stats/rest/en"
CURRENT_SEASON = "20252026"
RATE_LIMIT_SECONDS = 0.5

ALL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_last_request_time: float = 0.0
_cache: dict = {}  # key -> (timestamp, data)
CACHE_TTL_SECONDS = 300  # 5 minutes


def _rate_limit():
    """Enforce minimum delay between outgoing requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _get(url: str, cache_ttl: Optional[int] = None) -> Optional[dict]:
    """
    Perform a GET request with rate-limiting, optional caching, and
    error handling.  Returns parsed JSON or None on failure.
    """
    ttl = cache_ttl if cache_ttl is not None else CACHE_TTL_SECONDS
    if url in _cache:
        ts, data = _cache[url]
        if time.time() - ts < ttl:
            return data

    _rate_limit()
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        _cache[url] = (time.time(), data)
        return data
    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None


def clear_cache():
    """Clear the in-memory request cache."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def get_schedule(date_str: str) -> Optional[dict]:
    """Fetch the league schedule for a given date (YYYY-MM-DD)."""
    return _get(f"{BASE_URL}/schedule/{date_str}")


def get_current_schedule() -> Optional[dict]:
    """Fetch today's schedule."""
    today = date.today().isoformat()
    return _get(f"{BASE_URL}/schedule/{today}", cache_ttl=60)


def get_boxscore(game_id: int) -> Optional[dict]:
    """Fetch boxscore for a specific game."""
    return _get(f"{BASE_URL}/gamecenter/{game_id}/boxscore", cache_ttl=3600)


def get_play_by_play(game_id: int) -> Optional[dict]:
    """Fetch play-by-play data for a specific game."""
    return _get(f"{BASE_URL}/gamecenter/{game_id}/play-by-play", cache_ttl=3600)


def get_player_game_log(player_id: int) -> Optional[dict]:
    """Fetch a player's game log for the current season (regular season)."""
    return _get(
        f"{BASE_URL}/player/{player_id}/game-log/{CURRENT_SEASON}/2",
        cache_ttl=600,
    )


def get_roster(team_abbrev: str) -> Optional[dict]:
    """Fetch the current roster for a team."""
    return _get(f"{BASE_URL}/roster/{team_abbrev}/now", cache_ttl=3600)


def get_team_schedule(team_abbrev: str) -> Optional[dict]:
    """Fetch the full season schedule for a team."""
    return _get(
        f"{BASE_URL}/club-schedule-season/{team_abbrev}/now",
        cache_ttl=600,
    )


def get_all_teams() -> list[str]:
    """Return the list of all 32 NHL team abbreviations."""
    return list(ALL_TEAMS)
