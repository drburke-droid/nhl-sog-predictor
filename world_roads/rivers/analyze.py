"""Do rivers wind by pi? The numbers for the video and ACCURACY.md, from data/rivers.json.

    python analyze.py   -> data/analysis.json, prints the findings
"""
import os, json
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NEAR = 0.2                                      # "near pi": sinuosity within this of 3.14159
FEATURES = [("meander", "meandering (wiggles under 10 km)"), ("wander", "wandering (bends over 10 km)"),
            ("bulge", "farthest from the straight line, / its length"),
            ("gradient_m_per_km", "mean gradient, m per km"), ("lowland_share", "share of the course below 200 m"),
            ("median_elev_m", "median elevation, m"), ("relief_m", "source to mouth drop, m"),
            ("length_km", "length, km"), ("mean_abs_lat", "latitude (distance from the equator), degrees")]
FAMOUS = ["Nile", "Amazon", "Yangtze", "Mississippi", "Yenisei", "Yellow", "Ob", "Paraná", "Congo", "Amur", "Lena",
          "Mekong", "Niger", "Mackenzie", "Murray", "Volga", "Brahmaputra", "Ganges", "Indus", "Danube",
          "St. Lawrence", "Colorado", "Rhine", "Thames", "Seine"]


def display_name(r):
    """an English label for the famous rivers (OSM names are local)"""
    n = r["name"]
    table = {"نهر النيل": "Nile", "Chang Jiang": "Yangtze", "Irtysh": "Ob–Irtysh", "Río Paraná": "Paraná",
             "Saint Lawrence River": "St. Lawrence", "La Loire": "Loire", "La Seine": "Seine", "Дон": "Don",
             "塔里木河": "Tarim", "Mississippi River": "Mississippi–Missouri", "Murray River": "Murray–Darling"}
    if n in table:
        return table[n]
    for suf in (" River", " river"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def main():
    R = json.load(open(os.path.join(DATA, "rivers.json")))
    g = lambda f: np.array([r[f] for r in R], float)
    S, L, D = g("sinuosity"), g("length_km"), g("straight_km")
    near = np.abs(S - np.pi) < NEAR
    out = dict(n=len(R), mean=S.mean(), median=float(np.median(S)), ratio_of_totals=L.sum() / D.sum(),
               geo_mean=float(np.exp(np.log(S).mean())), total_km=L.sum(), total_straight_km=D.sum(),
               n_near=int(near.sum()), n_within_0_1=int((np.abs(S - np.pi) < 0.1).sum()), n_above_pi=int((S > np.pi).sum()),
               percentiles={str(p): float(np.percentile(S, p)) for p in (5, 10, 25, 50, 75, 90, 95)})
    # the running ratio, longest river first (the gauge in the video)
    out["running_ratio"] = (np.cumsum(L) / np.cumsum(D)).tolist()
    out["running_mean"] = (np.cumsum(S) / np.arange(1, len(S) + 1)).tolist()
    # near pi vs the rest
    comp = []
    for f, label in FEATURES:
        x = g(f)
        comp.append(dict(feature=f, label=label, near_median=float(np.median(x[near])), rest_median=float(np.median(x[~near])),
                         p=float(mannwhitneyu(x[near], x[~near]).pvalue), spearman_with_S=float(spearmanr(S, x).statistic)))
    out["near_vs_rest"] = comp
    # flatness: sinuosity by gradient band
    grad, low = g("gradient_m_per_km"), g("lowland_share")
    bands = []
    for lo, hi, label in [(0, 0.1, "under 0.1"), (0.1, 0.3, "0.1–0.3"), (0.3, 1, "0.3–1"), (1, 3, "1–3"), (3, 1e9, "over 3")]:
        m = (grad >= lo) & (grad < hi)
        bands.append(dict(label=label, n=int(m.sum()), mean=float(S[m].mean()), median=float(np.median(S[m])),
                          ratio=float(L[m].sum() / D[m].sum()), share_near=float(near[m].mean())))
    out["by_gradient"] = bands
    # what makes the extra length: meandering M (under 10 km) times wandering C (over 10 km)
    M, C = g("meander"), g("wander")
    route = np.where(np.log(M) > np.log(C), "meanders", "bends")
    out["near_routes"] = {k: int(np.sum(near & (route == k))) for k in ("meanders", "bends")}
    for key in ("continent", "terminus"):
        cats = sorted(set(r[key] for r in R))
        out["by_" + key] = {c: dict(n=int(sum(r[key] == c for r in R)), near=int(sum((r[key] == c) and m for r, m in zip(R, near))),
                                   mean=float(np.mean([r["sinuosity"] for r in R if r[key] == c]))) for c in cats}
    lst = []
    for i in np.argsort(np.abs(S - np.pi)):
        if not near[i]:
            break
        r = R[i]
        lst.append(dict(rank=r["rank"], name=r["name"], display=display_name(r), sinuosity=r["sinuosity"], length_km=r["length_km"],
                        straight_km=r["straight_km"], meander=r["meander"], wander=r["wander"], route=str(route[i]),
                        gradient=r["gradient_m_per_km"], lowland=r["lowland_share"], continent=r["continent"], terminus=r["terminus"],
                        mouth=r["mouth"], source=r["source"]))
    out["near_list"] = lst
    fam = []
    for r in R:
        d = display_name(r)
        if any(d.startswith(f) for f in FAMOUS) and not any(x["display"] == d for x in fam):
            fam.append(dict(rank=r["rank"], name=r["name"], display=d, length_km=r["length_km"], straight_km=r["straight_km"],
                            sinuosity=r["sinuosity"], meander=r["meander"], wander=r["wander"]))
    out["famous"] = fam
    json.dump(out, open(os.path.join(DATA, "analysis.json"), "w"), indent=1)
    # ---- print
    print(f"{out['n']} rivers, {out['total_km']:,.0f} km of river, {out['total_straight_km']:,.0f} km straight")
    print(f"sinuosity: mean {out['mean']:.3f}, median {out['median']:.3f}, ratio of totals {out['ratio_of_totals']:.3f}, "
          f"geometric mean {out['geo_mean']:.3f}")
    print(f"within {NEAR} of pi: {out['n_near']} ({out['n_within_0_1']} within 0.1); above pi: {out['n_above_pi']}")
    print("percentiles", {k: round(v, 2) for k, v in out["percentiles"].items()})
    print(f"\n{'':45s} {'near pi':>9s} {'others':>9s}   p       rho(S)")
    for c in comp:
        print(f"{c['label']:45s} {c['near_median']:9.2f} {c['rest_median']:9.2f}  {c['p']:7.1e}  {c['spearman_with_S']:+.2f}")
    print("\nby gradient (m/km):")
    for b in bands:
        print(f"  {b['label']:10s} n={b['n']:4d} mean S {b['mean']:.2f}  median {b['median']:.2f}  ratio {b['ratio']:.2f}  near pi {b['share_near']:.1%}")
    print("\nnear-pi rivers get there by:", out["near_routes"])
    print("by continent:", {k: (v["n"], v["near"], round(v["mean"], 2)) for k, v in out["by_continent"].items()})
    print("by terminus:", {k: (v["n"], v["near"], round(v["mean"], 2)) for k, v in out["by_terminus"].items()})
    print("\nnearest pi:")
    for x in lst:
        print(f"  #{x['rank']:4d} {x['display'][:26]:26s} {x['length_km']:6.0f}/{x['straight_km']:5.0f} = {x['sinuosity']:.4f}  "
              f"M {x['meander']:.2f} C {x['wander']:.2f} ({x['route']})  grad {x['gradient']:.2f}  low {x['lowland']:.0%}  {x['continent']}")
    print("\nfamous:")
    for x in fam:
        print(f"  #{x['rank']:4d} {x['display'][:24]:24s} {x['length_km']:6.0f} / {x['straight_km']:5.0f} = {x['sinuosity']:.2f}  M {x['meander']:.2f} C {x['wander']:.2f}")


if __name__ == "__main__":
    main()
