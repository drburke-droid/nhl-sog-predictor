"""Statistics on the contiguous US network (from nhd.py's per-unit tables).

  junction angles against aridity (Seybold, Rothman & Kirchner 2017), with controls for slope and stream order
  Hack's law, globally and its local exponent by drainage area (Hack 1957; Mueller 1973)
  Horton's ratios by Strahler order (Horton 1945; Strahler 1957)
  every main stem's length over its source-to-mouth distance (the pi claim, at every size)

    python us_stats.py        -> data/us_stats.json
"""
import os, glob, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JCOLS = ["x", "y", "a250", "a500", "a1000", "o1", "o2", "odn", "da1", "da2", "dadn", "ai", "p", "slope", "len1", "len2"]
PCOLS = ["lp", "len", "straight", "da", "lon0", "lat0", "lon1", "lat1", "terminal", "complete", "natural", "slope", "ai", "n", "head"]
AI_EDGES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 6.0]


def load():
    J, H, P, Ho = [], [], [], []
    for p in sorted(glob.glob(os.path.join(HERE, "data", "nhd", "out", "*.npz"))):
        z = np.load(p)
        J.append(z["junctions"]); H.append(z["hack"]); P.append(z["paths"]); Ho.append(z["horton"])
    J = np.vstack([j for j in J if len(j)])
    P = np.vstack([q for q in P if len(q)])
    return ({c: J[:, i] for i, c in enumerate(JCOLS)}, np.vstack(H), {c: P[:, i] for i, c in enumerate(PCOLS)}, Ho)


def cluster_boot(vals, x, y, n=300, seed=1, stat=np.mean):
    rng = np.random.default_rng(seed)
    cell = (np.floor(y) * 1000 + np.floor(x)).astype(np.int64)
    u, inv = np.unique(cell, return_inverse=True)
    groups = [vals[inv == k] for k in range(len(u))] if len(u) < 4000 else None
    sums = np.bincount(inv, vals); cnts = np.bincount(inv)
    out = []
    for _ in range(n):
        k = rng.integers(0, len(u), len(u))
        out.append(sums[k].sum() / cnts[k].sum())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def junction_stats(J, key="a1000"):
    m = np.isfinite(J[key]) & np.isfinite(J["ai"]) & (J["ai"] > 0)
    a, ai = J[key][m], J["ai"][m]
    rows = []
    for lo, hi in zip(AI_EDGES[:-1], AI_EDGES[1:]):
        s = (ai >= lo) & (ai < hi)
        if s.sum() < 200:
            continue
        ci = cluster_boot(a[s], J["x"][m][s], J["y"][m][s])
        rows.append({"ai": [lo, hi], "n": int(s.sum()), "mean": float(a[s].mean()), "median": float(np.median(a[s])), "ci": ci})
    # rank correlation
    from scipy.stats import spearmanr
    rho = spearmanr(ai, a).correlation
    # controls: the same trend within slope terciles and for first-order pairs only
    ctrl = {}
    sl = J["slope"][m]
    good = np.isfinite(sl) & (sl > 0)
    qs = np.percentile(sl[good], [33.3, 66.7])
    for name, s in (("flat", good & (sl < qs[0])), ("middle", good & (sl >= qs[0]) & (sl < qs[1])), ("steep", good & (sl >= qs[1])),
                    ("first_order_pairs", (J["o1"][m] == 1) & (J["o2"][m] == 1))):
        dry = s & (ai < 0.5); wet = s & (ai > 1.0)
        ctrl[name] = {"dry_mean": float(a[dry].mean()) if dry.sum() else None, "wet_mean": float(a[wet].mean()) if wet.sum() else None,
                      "n_dry": int(dry.sum()), "n_wet": int(wet.sum())}
    return {"measure": key, "n": int(m.sum()), "spearman": float(rho), "bins": rows, "controls": ctrl,
            "overall_mean": float(a.mean()), "overall_median": float(np.median(a))}


def hack_stats(H):
    da, lup, comp, nat, order = H.T
    # areas below 1 km2 are left out: at 1:100,000 the mapped channel stops well short of the divide
    # and above ~18,000 km2 basins span several processing units, whose upstream lengths can't be joined here
    m = (comp > 0) & (nat > 0) & (da >= 1.0) & (da <= 10 ** 4.25) & (lup > 0.05)
    x, y = np.log10(da[m]), np.log10(lup[m])
    b, a = np.polyfit(x, y, 1)
    # the binned median curve, and its local slope (the local Hack exponent) over each decade of area
    curve = []
    for lo in np.arange(0, 4.25, 0.25):
        s = (x >= lo) & (x < lo + 0.25)
        if s.sum() >= 30:
            curve.append([float(lo + 0.125), float(np.median(y[s])), float(np.percentile(y[s], 10)), float(np.percentile(y[s], 90)), int(s.sum())])
    cv = np.array(curve)
    local = []
    for lo in np.arange(0, 3.5, 0.5):
        s = (cv[:, 0] >= lo) & (cv[:, 0] < lo + 1.0)
        if s.sum() >= 3:
            bb = np.polyfit(cv[s, 0], cv[s, 1], 1)[0]
            local.append({"log10_area": [float(lo), float(lo + 1)], "n": int(cv[s, 4].sum()), "h": float(bb)})
    return {"n": int(m.sum()), "h": float(b), "c": float(10 ** a), "local": local, "curve": curve}


def horton_stats(Ho):
    N, Lsum = {}, {}
    for h in Ho:
        for w, n, l in h:
            N[int(w)] = N.get(int(w), 0) + n
            Lsum[int(w)] = Lsum.get(int(w), 0) + l
    ws = sorted(N)
    rows = [{"order": w, "streams": int(N[w]), "mean_km": float(Lsum[w] / N[w])} for w in ws]
    rb = [N[w] / N[w + 1] for w in ws[:-1] if w + 1 in N and N[w + 1] > 50]
    rl = [(Lsum[w + 1] / N[w + 1]) / (Lsum[w] / N[w]) for w in ws[:-1] if w + 1 in N and N[w + 1] > 50]
    return {"orders": rows, "R_B": float(np.exp(np.mean(np.log(rb)))), "R_L": float(np.exp(np.mean(np.log(rl))))}


def path_stats(P):
    # rivers that start at a source: level paths that begin at a divergence are side channels of another river
    m = (P["head"] > 0) & (P["complete"] > 0) & (P["natural"] > 0.5) & (P["straight"] > 0.5) & (P["len"] >= P["straight"])
    L, S = P["len"][m], P["straight"][m]
    s = L / S
    rows = []
    for lo, hi in ((1, 3), (3, 10), (10, 30), (30, 100), (100, 300), (300, 1000), (1000, 5000)):
        k = (L >= lo) & (L < hi)
        if k.sum() < 20:
            continue
        rows.append({"len_km": [lo, hi], "n": int(k.sum()), "ratio_of_totals": float(L[k].sum() / S[k].sum()),
                     "median": float(np.median(s[k])), "p90": float(np.percentile(s[k], 90)),
                     "near_pi": float(np.mean(np.abs(s[k] - math.pi) < 0.2))})
    fx, fy = np.log10(S), np.log10(L)
    keep = S >= 1.0
    dfit = np.polyfit(fx[keep], fy[keep], 1)
    curve = []
    for lo in np.arange(0, 3.0, 0.25):
        k = keep & (fx >= lo) & (fx < lo + 0.25)
        if k.sum() >= 20:
            curve.append([float(lo + 0.125), float(np.median(fy[k])), int(k.sum())])
    return {"fractal_D": float(dfit[0]), "fractal_c": float(10 ** dfit[1]), "len_vs_straight": curve,
            "n": int(m.sum()), "km": float(L.sum()), "ratio_of_totals": float(L.sum() / S.sum()), "median": float(np.median(s)),
            "near_pi": float(np.mean(np.abs(s - math.pi) < 0.2)), "above_pi": float(np.mean(s > math.pi)), "by_length": rows}


def main():
    J, H, P, Ho = load()
    out = {"junctions": {k: junction_stats(J, k) for k in ("a250", "a500", "a1000")},
           "hack": hack_stats(H), "horton": horton_stats(Ho), "paths": path_stats(P)}
    json.dump(out, open(os.path.join(HERE, "data", "us_stats.json"), "w"), indent=1)
    j = out["junctions"]["a1000"]
    print("junctions", j["n"], "spearman", round(j["spearman"], 3))
    for r in j["bins"]:
        print(f"  AI {r['ai'][0]:.2f}-{r['ai'][1]:.2f}  n={r['n']:7d}  mean {r['mean']:.1f}  median {r['median']:.1f}  ci {r['ci'][0]:.1f}-{r['ci'][1]:.1f}")
    print("controls", json.dumps(j["controls"], indent=0))
    h = out["hack"]
    print("hack h", round(h["h"], 3), "c", round(h["c"], 2), [(r["log10_area"], round(r["h"], 3)) for r in h["local"]])
    print("horton R_B", round(out["horton"]["R_B"], 2), "R_L", round(out["horton"]["R_L"], 2))
    p = out["paths"]
    print("paths", p["n"], "ratio", round(p["ratio_of_totals"], 3), "median", round(p["median"], 3), "near pi", round(p["near_pi"], 4))
    for r in p["by_length"]:
        print("  ", r)


if __name__ == "__main__":
    main()
