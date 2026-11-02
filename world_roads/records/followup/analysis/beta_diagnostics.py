"""Diagnostics of the handoff's rank-tail beta (OLS of ranks 2-10 on -ln(rank - 0.5), per season).

    python analysis/beta_diagnostics.py      # writes outputs/beta_diagnostics.csv (by league-size band)

Per season and category: the fit's R^2, whether the slope was clipped at zero, how many of ranks 2-10 are
tied with a neighbour, the number of players in the league that season (NBA, NHL, NFL: players with any
appearance in the player tables) and therefore what share of the league ranks 2-10 span, and how much the
season's beta moves when the most extreme of ranks 2-10 is left out (self-dependence on one value).
"""
import csv, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsmodel as G                                       # noqa: E402
import context as X                                       # noqa: E402

F = os.path.dirname(HERE)
A = np.array([-math.log(i - 0.5) for i in range(2, 11)])


def players_per_season():
    out = {}
    for sp, f, gcol in (("NBA", "nba_players.csv", "g"), ("NHL", "nhl_skaters.csv", "gp")):
        n = defaultdict(set)
        for r in csv.DictReader(open(os.path.join(F, "data", "context", f), encoding="utf-8")):
            if r["player_id"]:
                n[int(r["season"])].add(r["player_id"])
        out[sp] = {y: len(v) for y, v in n.items()}
    n = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(F, "data", "context", "nfl_players.csv"), encoding="utf-8")):
        n[int(r["season"])].add(r["player_id"])
    out["NFL"] = {y: len(v) for y, v in n.items()}
    return out


def main():
    D = G.Data("original")
    E = X.env_series()
    T = X.teams()
    P = players_per_season()
    rows = []
    for cat in X.ENV_MAP:
        sp = D.meta[cat]["sport"]
        for y, lst in D.seasons[cat].items():
            if len(lst) < 10 or y not in E[cat] or y not in T[sp]:
                continue
            v = np.array([e.perf for e in lst[1:10]]) / (E[cat][y] if cat != "mlb_era" else 1.0)
            b, a0 = np.polyfit(A, v, 1)
            fit = A * b + a0
            ss = float(((v - v.mean()) ** 2).sum())
            r2 = 1 - float(((v - fit) ** 2).sum()) / ss if ss > 0 else float("nan")
            ties = int(sum(1 for i in range(len(v) - 1) if abs(v[i] - v[i + 1]) < 1e-9))
            loo = float(np.polyfit(A[1:], v[1:], 1)[0])                  # without rank 2
            rows.append({"category": cat, "sport": sp, "season": y, "teams": T[sp][y], "beta_raw": b,
                         "clipped": b <= 0, "r2": r2, "tied_neighbours": ties,
                         "players": P.get(sp, {}).get(y), "beta_without_rank2": loo})
    bands = [(0, 6, "<= 6 teams"), (7, 12, "7-12"), (13, 24, "13-24"), (25, 99, ">= 25")]
    out = []
    for sp in ("NBA", "NHL", "MLB", "NFL"):
        for lo, hi, lab in bands:
            rs = [r for r in rows if r["sport"] == sp and lo <= r["teams"] <= hi]
            if not rs:
                continue
            pl = [r["players"] for r in rs if r["players"]]
            shares = [10 / r["players"] for r in rs if r["players"]]
            chg = [abs(r["beta_without_rank2"] - r["beta_raw"]) / abs(r["beta_raw"]) for r in rs if abs(r["beta_raw"]) > 1e-9]
            out.append({"sport": sp, "league_size": lab, "category_seasons": len(rs),
                        "median_r2": round(float(np.nanmedian([r["r2"] for r in rs])), 3),
                        "share_r2_below_0.8": round(float(np.mean([r["r2"] < 0.8 for r in rs])), 3),
                        "share_clipped_at_zero": round(float(np.mean([r["clipped"] for r in rs])), 3),
                        "mean_tied_neighbours_of_8": round(float(np.mean([r["tied_neighbours"] for r in rs])), 2),
                        "median_players_in_league": int(np.median(pl)) if pl else "",
                        "ranks_2_10_reach_top_share": f"{100 * float(np.median(shares)):.1f}%" if shares else "",
                        "median_rel_change_without_rank2": round(float(np.median(chg)), 3) if chg else ""})
    path = os.path.join(F, "outputs", "beta_diagnostics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
