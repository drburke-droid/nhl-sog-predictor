"""Empirical tests of the two proposed adjustments, before using either.

    python analysis/env_field_tests.py        # from followup/; writes outputs/env_elasticity.csv,
                                              # outputs/field_size_test.csv, outputs/nba_denominators.csv,
                                              # outputs/babe_dye_decomposition.csv, outputs/env_field_tests.txt

1. Environment elasticity. For each league category: does the level of the field (median of ranks 2-10)
   move with the league's per-team-game rate of the same statistic? First differences of logs remove slow
   trends: dlog F_s = a + g * dlog env_s. g ~ 1 supports dividing by the league rate; g ~ 0 says the league
   rate carries no information about the field beyond the time trend. The same for the season maximum.
2. Field size. If the season's best is the maximum of N independent draws, then at an expansion
   E[d top_s] = beta * d ln N_s (Gumbel tail). We regress season-to-season changes of the 2nd-best value
   (environment-normalised, scaled to the schedule) on d ln(teams), pooled within sport after dividing each
   category by its own scale, and compare with what the rank-tail beta predicts.
3. NBA points: the candidate denominators side by side (league points, estimated pace, shot opportunities),
   with the 1961-62 figures and what each does to Wilt Chamberlain's score.
4. Babe Dye 1924-25: the handoff's field model decomposed into its parts.
"""
import csv, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsmodel as G                                       # noqa: E402
import context as X                                       # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "outputs")


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    n = len(y)
    s2 = float(r @ r) / max(n - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    ss = float(((y - y.mean()) ** 2).sum()) or 1.0
    return coef[1], math.sqrt(cov[1, 1]), 1 - float(r @ r) / ss, n


def level_series(D, cat, env=None):
    """season -> (median of ranks 2-10, season max), totals divided by games per team"""
    m = D.meta[cat]
    out = {}
    for y, lst in D.seasons[cat].items():
        if y in m["short"] or len(lst) < 6:
            continue
        vals = []
        for e in lst:
            v = e.perf / e.games if m["scaled"] and e.games == e.games else e.perf
            vals.append(v)
        if cat == "mlb_era":
            vals = [-v for v in vals]                  # ERA itself, for log ratios
        out[y] = (float(np.median(vals[1:10])), vals[0] if cat != "mlb_era" else min(vals))
    return out


def elasticity(D):
    E = X.env_series()
    rows = []
    for cat in X.ENV_MAP:
        L = level_series(D, cat)
        e = E[cat]
        ys = sorted(y for y in L if y in e and y - 1 in L and y - 1 in e)
        if len(ys) < 10:
            continue
        de = [math.log(e[y]) - math.log(e[y - 1]) for y in ys]
        dF = [math.log(L[y][0]) - math.log(L[y - 1][0]) for y in ys]
        dM = [math.log(L[y][1]) - math.log(L[y - 1][1]) for y in ys]
        gF, seF, r2F, n = ols(de, dF)
        gM, seM, r2M, _ = ols(de, dM)
        rows.append({"category": cat, "seasons": n, "first": ys[0], "last": ys[-1],
                     "elasticity_field_median": round(gF, 3), "se_field": round(seF, 3), "r2_field": round(r2F, 3),
                     "elasticity_season_max": round(gM, 3), "se_max": round(seM, 3), "r2_max": round(r2M, 3)})
    return rows


def field_size_test(D):
    E, T = X.env_series(), X.teams()
    _, beta = X.rank_tail_beta(D, list(X.ENV_MAP), envs=E)
    out, pooled = [], defaultdict(lambda: {"x": [], "y": [], "pred": []})
    for cat in X.ENV_MAP:
        m = D.meta[cat]
        sp = m["sport"]
        S = D.seasons[cat]
        vals = {}
        for y, lst in S.items():
            if y in m["short"] or len(lst) < 2 or y not in E[cat] or y not in T[sp]:
                continue
            def norm_v(e):
                v = e.perf / e.games if m["scaled"] else e.perf
                if cat == "mlb_era":
                    return E[cat][y] / (-e.perf)
                return v / (E[cat][y] / (1 if m["scaled"] else 1))
            vals[y] = norm_v(lst[1])                  # second best: free of the leader's own dominance
        ys = sorted(y for y in vals if y - 1 in vals)
        d = np.array([vals[y] - vals[y - 1] for y in ys])
        dn = np.array([math.log(T[sp][y]) - math.log(T[sp][y - 1]) for y in ys])
        if len(d) < 10:
            continue
        sc = float(np.std(d, ddof=1)) or 1.0
        pred = np.array([beta[cat].get(y, np.nan) * (1 / (D.seasons[cat][y][1].games if m["scaled"] else 1)) *
                         (1 if not m["scaled"] else 1) for y in ys])
        g, se, r2, n = ols(dn, d / sc)
        exp_ = int((dn > 1e-9).sum())
        out.append({"category": cat, "sport": sp, "seasons": n, "expansion_seasons": exp_,
                    "slope_sd_units_per_dlnN": round(g, 3), "se": round(se, 3)})
        pooled[sp]["x"] += list(dn)
        pooled[sp]["y"] += list(d / sc)
    for sp, v in pooled.items():
        g, se, r2, n = ols(v["x"], v["y"])
        out.append({"category": "POOLED", "sport": sp, "seasons": n,
                    "expansion_seasons": int((np.array(v["x"]) > 1e-9).sum()),
                    "slope_sd_units_per_dlnN": round(g, 3), "se": round(se, 3)})
    return out


def nba_denominators(D):
    L = X.league_tables()["nba"]
    out = []
    for y in (1952, 1962, 1967, 1987, 2006, 2026):
        r = L[y]
        out.append({"season": y, "teams": r["teams"], "pts_per_team_g": r["pts_per_team_g"],
                    "pace_league_table": r["pace_league_table"] or "not published",
                    "pace_mean_of_estimated_team_paces": r["pace_team_mean"],
                    "pace_team_range": f"{r['pace_team_min']}-{r['pace_team_max']}" if r["pace_team_min"] else "",
                    "shot_opportunities_fga_plus_044fta": round(float(r["fga_per_team_g"]) + 0.44 * float(r["fta_per_team_g"]), 1)})
    return out


def wilt_under_denominators(D):
    """Wilt 1961-62 (and the others) with each NBA points denominator, everything else as the original"""
    res = []
    for label, alt in [("none (original)", None), ("league points per team-game", {}),
                       ("mean estimated team pace (1951 on)", {"nba_ppg": "pace_team_mean"}),
                       ("shot opportunities FGA+0.44FTA", {"nba_ppg": "shot_opps"})]:
        if alt is None:
            recs = G.score(D, G.Spec(), cats=["nba_ppg"])
        else:
            ctx = X.make(D, env="ratio", alt=alt, sports=("NBA",))
            recs = G.score(D, G.Spec(env="ratio"), ctx=ctx, cats=["nba_ppg"])
        recs.sort(key=lambda r: -r["z"])
        w = next((r for r in recs if r["year"] == 1962), None)
        res.append({"denominator": label, "wilt_1962_z": round(w["z"], 3) if w else "",
                    "wilt_rank_in_nba_ppg": recs.index(w) + 1 if w else "",
                    "top3_in_category": "; ".join(f"{r['name']} {r['year']} {r['z']:.2f}" for r in recs[:3]),
                    "seasons_scored": len(recs)})
    return res


def babe_dye(D):
    """the handoff model's parts for Babe Dye 1924-25 (NHL goals), recomputed with gsmodel"""
    out = []
    E = X.env_series()
    for label, spec, kw in [
        ("original", G.Spec(), None),
        ("scoring only (goals / league goals per team-game)", G.Spec(env="ratio"), dict(env="ratio")),
        ("scoring + field 50%", G.Spec(env="ratio", field=0.5), dict(env="ratio", field=0.5)),
        ("scoring + field 100%", G.Spec(env="ratio", field=1.0), dict(env="ratio", field=1.0)),
        ("scoring + field 100%, normal-tail growth", G.Spec(env="ratio", field=1.0, field_tail="normal"),
         dict(env="ratio", field=1.0)),
        ("field 100% only (no scoring)", G.Spec(field=1.0), dict(env="none", field=1.0, beta_env=False)),
    ]:
        ctx = X.make(D, sports=("NHL",), **kw) if kw else None
        recs = G.score(D, spec, ctx=ctx, cats=["nhl_goals"], details=True)
        r = next(r for r in recs if r["year"] == 1925)
        pts = dict(r["pts"])
        early = [round(pts[s], 2) for s in sorted(pts) if s < 1925]
        late = [round(pts[s], 2) for s in sorted(pts) if s > 1925]
        b = (ctx.beta.get("nhl_goals", {}).get(1925) if ctx and ctx.beta else None)
        out.append({"model": label, "z": round(r["z"], 3), "x_dye": round(r["x"], 3), "baseline": round(r["base"], 3),
                    "scale": round(r["scale"], 3), "n": r["n"], "beta_1925": round(b, 3) if b else "",
                    "comparators_before": " ".join(map(str, early)), "comparators_after": " ".join(map(str, late))})
    return out


def write(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    D = G.Data(sys.argv[1] if len(sys.argv) > 1 else "original")
    el = elasticity(D)
    fs = field_size_test(D)
    nd = nba_denominators(D)
    wd = wilt_under_denominators(D)
    bd = babe_dye(D)
    write("env_elasticity.csv", el)
    write("field_size_test.csv", fs)
    write("nba_denominators.csv", nd)
    write("nba_points_denominator_effect.csv", wd)
    write("babe_dye_decomposition.csv", bd)
    with open(os.path.join(OUT, "env_field_tests.txt"), "w") as f:
        for title, rows in (("ELASTICITY", el), ("FIELD SIZE", fs), ("NBA DENOMINATORS", nd),
                            ("WILT UNDER EACH DENOMINATOR", wd), ("BABE DYE", bd)):
            print("\n" + title, file=f)
            for r in rows:
                print("  " + " | ".join(f"{k}={v}" for k, v in r.items()), file=f)
    print(open(os.path.join(OUT, "env_field_tests.txt")).read())


if __name__ == "__main__":
    main()
