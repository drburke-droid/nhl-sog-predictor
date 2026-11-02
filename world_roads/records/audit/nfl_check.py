"""Football check: every season leader 1932-98 (Pro Football Archives, summed from team pages) against
Wikipedia's leader lists, plus the games-per-team values read from the season standings.

    python audit/nfl_check.py        # from world_roads/records; needs the local cache
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bigfour as B
P = B.pfa_load()
print("seasons parsed:", {k: (min(v), max(v), len(v)) for k, v in P.items()})
for k, col in [('pass_yds','passing_yards'),('pass_td','passing_tds'),('rush_yds','rushing_yards'),('rec_yds','receiving_yards')]:
    W = B.load_nfl(k, col, 1932)
    yrs = [y for y in sorted(P[k]) if y in W and y <= 1998]
    bad = [(y, W[y][0], P[k][y][0]) for y in yrs if abs(W[y][0][1] - P[k][y][0][1]) > 0.5]
    print(k, 'checked', len(yrs), 'mismatch', len(bad))
    for b in bad: print('   ', b)
print({y: round(g, 1) for y, g in sorted(B.pfa_games().items())})
