"""Choose the window and trend by out-of-sample prediction, not by who wins.

    python analysis/cv_window.py [original|v2]     # writes outputs/cv_window_trend.csv

For every category and season s, predict the season's best mark M_s from the other seasons in the window
(s itself left out), with each trend form, and score the predictive distribution N(baseline, scale) with
the continuous ranked probability score (CRPS, lower is better) and the coverage of the central 90%
interval. Values are standardised by each category's median absolute season-to-season change, so all
categories count equally. The decision rule, fixed before looking at any athlete: the lowest mean CRPS.
"""
import csv, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsmodel as G                                       # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "outputs")
SQPI = 1 / math.sqrt(math.pi)


def crps_normal(y, mu, sd):
    z = (y - mu) / sd
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return sd * (z * (2 * Phi - 1) + 2 * phi - SQPI)


def series(D, cat):
    m = D.meta[cat]
    S = D.seasons[cat]
    out = {}
    for y, lst in S.items():
        if y in m["short"] or not lst:
            continue
        v = lst[0].perf
        if m["scaled"]:
            v = v / lst[0].games                     # per team-game, so schedules are comparable
        out[y] = v
    return out


def main():
    D = G.Data(sys.argv[1] if len(sys.argv) > 1 else "original")
    res = defaultdict(lambda: {"crps": [], "cover": [], "z": []})
    for cat, m in D.meta.items():
        M = series(D, cat)
        ys = sorted(M)
        diffs = [abs(M[b] - M[a]) for a, b in zip(ys, ys[1:]) if b - a == 1]
        unit = float(np.median(diffs)) if diffs else 1.0
        if unit <= 0:
            continue
        shift = m["shift"]
        for w in (5, 7, 10, 15, 20, 25):
            for trend in ("flat", "ols", "huber", "quad", "quad_safe"):
                key = (w, trend)
                for s in ys:
                    if s < m["start"]:
                        continue
                    pts = [(t, M[t]) for t in range(s - w, s + w + 1) if t != s and t in M]
                    if len(pts) < 8:
                        continue
                    fb = G.fit_baseline(pts, s, shift, trend)
                    if fb is None or not fb[2] > 0:
                        continue
                    base, resid, scale, k = fb
                    res[key]["crps"].append(crps_normal(M[s], base, scale) / unit)
                    zz = (M[s] - base) / scale
                    res[key]["cover"].append(abs(zz) <= 1.645)
                    res[key]["z"].append(zz)
    rows = []
    for (w, trend), d in sorted(res.items()):
        z = np.array(d["z"])
        rows.append({"window": w, "trend": trend, "predictions": len(d["crps"]),
                     "mean_crps_std_units": round(float(np.mean(d["crps"])), 4),
                     "coverage_90": round(float(np.mean(d["cover"])), 3),
                     "share_z_above_3": round(float(np.mean(z > 3)), 4),
                     "share_z_above_4": round(float(np.mean(z > 4)), 4)})
    rows.sort(key=lambda r: r["mean_crps_std_units"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cv_window_trend.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
