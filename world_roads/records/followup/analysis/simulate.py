"""Simulation checks of the field-size adjustment (illustrative counterfactuals, not estimates of real leagues).

    python analysis/simulate.py         # writes outputs/sim_beta_estimator.csv, outputs/sim_field_worlds.csv

1. The rank-tail beta. The handoff estimates the upper-tail scale from ranks 2-10 by OLS on -ln(rank-0.5) and
   adds beta*ln(Nref/N) to every value. We draw n players from a Gumbel (beta = 1 exactly, for any n) and a
   normal (whose local tail scale shrinks like 1/sqrt(2 ln n)) and compare the estimated beta and the implied
   gain from n to Nref with the true gain in the expected maximum.
2. Two expansion worlds, 60 seasons, the league growing from 6 to 30 teams (20 players a team):
     A "more draws":  each season's players are n independent draws from one talent distribution
     B "fixed pool":  the league is always the best of one fixed-size talent pool; expansion only adds
                      weaker players, so the top of the league does not depend on n
   Every player's "season" = ability + noise. We plant one exceptional season (+3.5 sd of season maxima) in
   each era, score all season leaders with the original measure (OLS, +/-15) and with the proposed offset,
   and report (a) the planted seasons' z by era and (b) how often a small-league season lands in the top ten
   among unplanted seasons. A sound adjustment shows no era pattern in either world.
"""
import csv, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsmodel as G                                       # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "outputs")
RANK_A = np.array([-math.log(i - 0.5) for i in range(2, 11)])


def rank_beta(sorted_desc):
    return max(0.0, float(np.polyfit(RANK_A, sorted_desc[1:10], 1)[0]))


def beta_estimator(rng, reps=2000):
    rows = []
    for dist in ("gumbel", "normal"):
        for n in (60, 120, 300, 640):
            b, mx = [], []
            for _ in range(reps):
                x = rng.gumbel(size=n) if dist == "gumbel" else rng.normal(size=n)
                x = np.sort(x)[::-1]
                b.append(rank_beta(x))
                mx.append(x[0])
            rows.append({"distribution": dist, "players": n, "mean_beta_hat": round(float(np.mean(b)), 3),
                         "sd_beta_hat": round(float(np.std(b)), 3), "mean_max": round(float(np.mean(mx)), 3)})
    # implied vs true gain from n to 640 players
    by = {(r["distribution"], r["players"]): r for r in rows}
    for r in rows:
        ref = by[(r["distribution"], 640)]
        r["true_gain_to_640"] = round(ref["mean_max"] - r["mean_max"], 3)
        r["implied_gain_beta_ln"] = round(r["mean_beta_hat"] * math.log(640 / r["players"]), 3)
    return rows


class SimData:
    """minimal stand-in for gsmodel.Data built from simulated season lists"""

    def __init__(self, seasons, N):
        self.seasons = {"sim": seasons}
        self.meta = {"sim": {"sport": "SIM", "title": "simulated", "short": set(), "shift": set(), "scaled": False,
                             "cut": None, "first": min(seasons), "start": min(seasons)}}
        self.N = N


def teams_by_season(n_seasons=60):
    t = []
    for s in range(n_seasons):
        t.append(6 if s < 20 else 12 if s < 30 else 21 if s < 40 else 30)
    return t


def simulate_world(rng, world, plant_seasons, roster=20, n_seasons=60, noise=0.5):
    T = teams_by_season(n_seasons)
    seasons, N = {}, {}
    pool = 30 * roster * 3                               # fixed talent pool for world B
    for s in range(n_seasons):
        n = T[s] * roster
        if world == "A":
            ability = rng.normal(size=n)
        else:
            ability = np.sort(rng.normal(size=pool))[::-1][:n]
        perf = ability + noise * rng.normal(size=n)
        perf = np.sort(perf)[::-1]
        seasons[1900 + s] = [G.Entry(f"p{s}_{i}", f"p{s}_{i}", float(v), "", math.nan) for i, v in enumerate(perf[:10])]
        N[1900 + s] = T[s]
    # season-max scale for planting
    mx = np.array([seasons[y][0].perf for y in seasons])
    sd = float(np.std(np.diff(mx)) / math.sqrt(2)) or 0.3
    for s in plant_seasons:
        y = 1900 + s
        seasons[y][0] = G.Entry("planted", "planted", seasons[y][1].perf + 3.5 * sd, "", math.nan)
    return SimData(seasons, N)


def score_sim(D, field):
    ctx = None
    spec = G.Spec()
    if field:
        raw = {}
        for y, lst in D.seasons["sim"].items():
            raw[y] = rank_beta(np.array([e.perf for e in lst]))
        beta = {y: float(np.median([raw[t] for t in raw if abs(t - y) <= 7])) for y in raw}
        ctx = G.Context(teams={"SIM": D.N}, beta={"sim": beta}, nref={"SIM": 30})
        spec = G.Spec(field=1.0)
    return G.score(D, spec, ctx=ctx)


def field_worlds(rng, reps=200):
    plants = [8, 25, 45]                                  # one exceptional season in each era (6, 12, 30 teams)
    rows = []
    for world in ("A", "B"):
        for field in (0, 1):
            zp = {p: [] for p in plants}
            small_top = []
            for _ in range(reps):
                D = simulate_world(rng, world, plants)
                recs = score_sim(D, field)
                for r in recs:
                    if r["id"] == "planted":
                        zp[r["year"] - 1900].append(r["z"])
                un = sorted([r for r in recs if r["id"] != "planted"], key=lambda r: -r["z"])[:10]
                small_top.append(np.mean([D.N[r["year"]] <= 6 for r in un]))
            rows.append({"world": "A: more independent draws" if world == "A" else "B: fixed talent pool (dilution)",
                         "field_offset": "proposed beta*ln(Nref/N)" if field else "none (original)",
                         "planted_z_6_teams": round(float(np.mean(zp[8])), 2),
                         "planted_z_12_teams": round(float(np.mean(zp[25])), 2),
                         "planted_z_30_teams": round(float(np.mean(zp[45])), 2),
                         "share_of_unplanted_top10_from_6_team_era": round(float(np.mean(small_top)), 3),
                         "share_of_seasons_in_6_team_era": round(20 / 60, 3)})
    return rows


def main():
    rng = np.random.default_rng(2026)
    os.makedirs(OUT, exist_ok=True)
    b = beta_estimator(rng)
    f = field_worlds(rng)
    for name, rows in (("sim_beta_estimator.csv", b), ("sim_field_worlds.csv", f)):
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
