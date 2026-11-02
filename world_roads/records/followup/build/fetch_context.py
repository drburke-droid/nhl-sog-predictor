"""Fetch league context for the follow-up audit (cached under ../cache/context, never re-downloaded).

    python followup/build/fetch_context.py          # from world_roads/records; about 20 minutes

Pages (all under /leagues/, allowed by each site's robots.txt; 3.5 s between requests, above the sites'
3 s crawl delay):
  Basketball-Reference  leagues/NBA_stats_per_game.html        league averages per season (pace, FGA, FTA, PTS)
                        leagues/{BAA|NBA}_{y}.html             team tables incl. Miscellaneous (team pace), team count
                        leagues/{BAA|NBA}_{y}_per_game.html    every player's per-game line (G, MP, PTS, TRB, AST, 3P)
  Hockey-Reference      leagues/stats.html                     league averages per season (teams, goals per game)
                        leagues/NHL_{y}_skaters.html           every skater's line (GP, G, A, PTS; TOI from 1998-99)
  Baseball-Reference    leagues/AL/bat.shtml, leagues/NL/bat.shtml, leagues/AL/pitch.shtml, leagues/NL/pitch.shtml
                                                               league batting and pitching by season (teams, R/G, PA)
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from fetch import get                                    # noqa: E402  the project's polite cached fetcher

C = os.path.join(ROOT, "cache", "context")
PAUSE = 3.5


def lg(y):
    return "BAA" if y <= 1949 else "NBA"


def main(which=("league", "nba_seasons", "nba_players", "nhl_skaters")):
    if "league" in which:
        for url, name in [
            ("https://www.basketball-reference.com/leagues/NBA_stats_per_game.html", "nba_league_per_game.html"),
            ("https://www.hockey-reference.com/leagues/stats.html", "nhl_league_stats.html"),
            ("https://www.baseball-reference.com/leagues/AL/bat.shtml", "mlb_AL_bat.html"),
            ("https://www.baseball-reference.com/leagues/NL/bat.shtml", "mlb_NL_bat.html"),
            ("https://www.baseball-reference.com/leagues/AL/pitch.shtml", "mlb_AL_pitch.html"),
            ("https://www.baseball-reference.com/leagues/NL/pitch.shtml", "mlb_NL_pitch.html"),
        ]:
            get(url, os.path.join(C, name), pause=PAUSE)
            print(name, flush=True)
    if "nba_seasons" in which:
        for y in range(1947, 2027):
            get(f"https://www.basketball-reference.com/leagues/{lg(y)}_{y}.html", os.path.join(C, f"nba_{y}.html"), pause=PAUSE)
        print("nba seasons done", flush=True)
    if "nba_players" in which:
        for y in range(1947, 2027):
            get(f"https://www.basketball-reference.com/leagues/{lg(y)}_{y}_per_game.html",
                os.path.join(C, f"nba_{y}_per_game.html"), pause=PAUSE)
        print("nba players done", flush=True)
    if "nhl_skaters" in which:
        for y in range(1918, 2027):
            if y == 2005:                                 # no season (lockout)
                continue
            try:
                get(f"https://www.hockey-reference.com/leagues/NHL_{y}_skaters.html",
                    os.path.join(C, f"nhl_{y}_skaters.html"), pause=PAUSE)
            except RuntimeError as e:
                print("missing", y, e, flush=True)
        print("nhl skaters done", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("league", "nba_seasons", "nba_players", "nhl_skaters"))
