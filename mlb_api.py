"""
MLB API client module.

Provides functions to fetch data from the MLB Stats API with
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

BASE_URL = "https://statsapi.mlb.com/api/v1"
CURRENT_SEASON = 2025
RATE_LIMIT_SECONDS = 0.3

# MLB team abbreviations and IDs
TEAM_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SF": 137, "SEA": 136,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120,
}

ID_TO_ABBREV = {v: k for k, v in TEAM_MAP.items()}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_last_request_time: float = 0.0
_cache: dict = {}
CACHE_TTL_SECONDS = 300


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _get(url: str, params: Optional[dict] = None, cache_ttl: Optional[int] = None) -> Optional[dict]:
    ttl = cache_ttl if cache_ttl is not None else CACHE_TTL_SECONDS
    cache_key = url + str(params or {})
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < ttl:
            return data

    _rate_limit()
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        _cache[cache_key] = (time.time(), data)
        return data
    except requests.RequestException as exc:
        logger.warning("MLB API request failed: %s — %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def get_todays_schedule(game_date: Optional[str] = None) -> list[dict]:
    """Fetch today's MLB games with probable pitchers."""
    if game_date is None:
        game_date = date.today().isoformat()
    data = _get(
        f"{BASE_URL}/schedule",
        params={
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team",
        },
    )
    if not data:
        return []

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_team = home.get("team", {})
            away_team = away.get("team", {})

            games.append({
                "game_pk": g.get("gamePk"),
                "date": game_date,
                "status": status,
                "home_team_id": home_team.get("id"),
                "home_team_abbrev": _team_abbrev(home_team.get("id")),
                "home_team_name": home_team.get("name", ""),
                "away_team_id": away_team.get("id"),
                "away_team_abbrev": _team_abbrev(away_team.get("id")),
                "away_team_name": away_team.get("name", ""),
                "home_probable_pitcher": _extract_pitcher(home.get("probablePitcher")),
                "away_probable_pitcher": _extract_pitcher(away.get("probablePitcher")),
            })

    return games


def _team_abbrev(team_id: Optional[int]) -> str:
    if team_id is None:
        return ""
    return ID_TO_ABBREV.get(team_id, str(team_id))


def _extract_pitcher(pitcher_data: Optional[dict]) -> Optional[dict]:
    if not pitcher_data:
        return None
    return {
        "id": pitcher_data.get("id"),
        "name": pitcher_data.get("fullName", ""),
        "hand": pitcher_data.get("pitchHand", {}).get("code", "R"),
    }


def get_pitcher_game_log(pitcher_id: int, season: int = CURRENT_SEASON) -> list[dict]:
    """Fetch a pitcher's game-by-game stats for the season."""
    data = _get(
        f"{BASE_URL}/people/{pitcher_id}/stats",
        params={
            "stats": "gameLog",
            "group": "pitching",
            "season": season,
        },
        cache_ttl=3600,
    )
    if not data:
        return []

    games = []
    for split_group in data.get("stats", []):
        for split in split_group.get("splits", []):
            stat = split.get("stat", {})
            game = split.get("game", {})
            opp = split.get("opponent", {})
            is_home = split.get("isHome", False)

            ip_str = stat.get("inningsPitched", "0")
            try:
                ip = float(ip_str)
                # Convert MLB IP format: 5.1 = 5 1/3, 5.2 = 5 2/3
                whole = int(ip)
                frac = ip - whole
                if abs(frac - 0.1) < 0.01:
                    ip = whole + 1/3
                elif abs(frac - 0.2) < 0.01:
                    ip = whole + 2/3
            except (ValueError, TypeError):
                ip = 0.0

            games.append({
                "game_pk": game.get("gamePk"),
                "date": split.get("date", ""),
                "pitcher_id": pitcher_id,
                "opponent_id": opp.get("id"),
                "opponent_abbrev": _team_abbrev(opp.get("id")),
                "is_home": is_home,
                "innings_pitched": round(ip, 2),
                "strikeouts": stat.get("strikeOuts", 0),
                "batters_faced": stat.get("battersFaced", 0),
                "pitches_thrown": stat.get("numberOfPitches", 0),
                "walks": stat.get("baseOnBalls", 0),
                "hits_allowed": stat.get("hits", 0),
                "earned_runs": stat.get("earnedRuns", 0),
                "home_runs_allowed": stat.get("homeRuns", 0),
                "is_starter": 1 if stat.get("gamesStarted", 0) >= 1 else 0,
            })

    return games


def get_team_batting_stats(team_id: int, season: int = CURRENT_SEASON) -> Optional[dict]:
    """Fetch team-level batting stats for the season."""
    data = _get(
        f"{BASE_URL}/teams/{team_id}/stats",
        params={
            "stats": "season",
            "group": "batting",
            "season": season,
        },
        cache_ttl=3600,
    )
    if not data:
        return None

    for split_group in data.get("stats", []):
        for split in split_group.get("splits", []):
            stat = split.get("stat", {})
            pa = stat.get("plateAppearances", 1) or 1
            return {
                "team_id": team_id,
                "games_played": stat.get("gamesPlayed", 0),
                "plate_appearances": pa,
                "strikeouts": stat.get("strikeOuts", 0),
                "k_rate": round(stat.get("strikeOuts", 0) / pa, 4),
                "walks": stat.get("baseOnBalls", 0),
                "bb_rate": round(stat.get("baseOnBalls", 0) / pa, 4),
                "hits": stat.get("hits", 0),
                "avg": stat.get("avg", ".000"),
                "obp": stat.get("obp", ".000"),
                "slg": stat.get("slg", ".000"),
            }
    return None


def get_pitcher_detail(pitcher_id: int, season: int = CURRENT_SEASON) -> Optional[dict]:
    """Fetch pitcher's season summary stats."""
    data = _get(
        f"{BASE_URL}/people/{pitcher_id}",
        params={"hydrate": f"stats(group=[pitching],type=[season],season={season})"},
        cache_ttl=3600,
    )
    if not data:
        return None

    people = data.get("people", [])
    if not people:
        return None

    person = people[0]
    info = {
        "id": person.get("id"),
        "name": person.get("fullName", ""),
        "hand": person.get("pitchHand", {}).get("code", "R"),
        "team_id": person.get("currentTeam", {}).get("id"),
        "team_abbrev": _team_abbrev(person.get("currentTeam", {}).get("id")),
    }

    # Season stats
    for stat_group in person.get("stats", []):
        for split in stat_group.get("splits", []):
            s = split.get("stat", {})
            info.update({
                "games_started": s.get("gamesStarted", 0),
                "innings_pitched": s.get("inningsPitched", "0"),
                "strikeouts": s.get("strikeOuts", 0),
                "walks": s.get("baseOnBalls", 0),
                "era": s.get("era", "0.00"),
                "whip": s.get("whip", "0.00"),
                "batters_faced": s.get("battersFaced", 0),
                "pitches_thrown": s.get("numberOfPitches", 0),
            })
            break

    return info


def get_all_team_ids() -> dict[str, int]:
    """Return abbreviation -> team_id map."""
    return dict(TEAM_MAP)


def get_game_boxscore(game_pk: int) -> Optional[dict]:
    """Fetch boxscore for a completed game."""
    return _get(f"{BASE_URL}/game/{game_pk}/boxscore", cache_ttl=86400)
