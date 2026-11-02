"""Rebuild the greatest-seasons ranking from the exported CSVs alone (numpy and the standard library).

    python audit_recompute.py                 # window 15, as published; checks every score against the pipeline
    python audit_recompute.py --window 10     # another window
    python audit_recompute.py --flat          # sensitivity: no trend line (plain mean and SD of the window)
    python audit_recompute.py --no-shift      # sensitivity: no separate level for swimming's 2008-09 suit seasons
    python audit_recompute.py --top 30        # show more of the list

Run it in the folder that holds inputs/ (and, for the comparison with the published list, outputs/).

The measure, for every season-leading performance x in season y of a category:
  1. for each season s in y-W .. y+W that is in the data and not cut short: the best performance by anyone
     other than the performer (names compared after removing accents, case and punctuation). Totals are
     scaled to the performer's schedule: value * games(y) / games(s). In athletics, a season with nobody
     else listed above the all-time lists' cut-off takes the cut-off mark. At least 8 real seasons needed.
  2. least squares: perf_s = a + b*(s - y) [+ c*suit_s in swimming, suit_s = 1 for 2008 and 2009]
  3. baseline = a (+ c if y is a suit season); residuals r_s; sd = sqrt(sum r^2 / (n - k)), k parameters
  4. z = (x - baseline) / sd;  T = 1 / (1 - F(x - baseline)), F a Gumbel fitted by moments to the residuals
  5. one entry per athlete per season (the category where it scores highest), ranked by z; the page's top ten
     then keeps each athlete's best season only.
"""
import argparse, csv, math, os, unicodedata, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=15)
ap.add_argument("--flat", action="store_true")
ap.add_argument("--no-shift", action="store_true")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()
IN = os.path.join(args.dir, "inputs")


def norm(name):
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z ]", "", s.lower()).split())


def gumbel_T(values, x):
    v = np.array(values, float)
    if len(v) < 6 or v.std(ddof=1) <= 0:
        return None
    beta = v.std(ddof=1) * math.sqrt(6) / math.pi
    mu = v.mean() - 0.5772 * beta
    z = (x - mu) / beta
    if z > 700:
        return float("inf")
    return 1 / max(1 - math.exp(-math.exp(-z)), 1e-300)


def measure(pts, y, x, shift, flat):
    yrs = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    if flat:
        base, resid, k = v.mean(), v - v.mean(), 1
    else:
        cols = [yrs - y, np.ones_like(yrs)]
        d = np.isin(yrs, list(shift)).astype(float)
        if shift and d.sum() > 0 and len(yrs) - d.sum() >= 8:
            cols.append(d)
        X = np.column_stack(cols)
        coef = np.linalg.lstsq(X, v, rcond=None)[0]
        resid = v - X @ coef
        base = coef[1] + (coef[2] if len(coef) > 2 and y in shift else 0)
        k = len(coef)
    dof = len(v) - k
    if dof < 2:
        return None, None
    sd = math.sqrt(float(resid @ resid) / dof)
    if sd <= 0:
        return None, None
    return (x - base) / sd, gumbel_T(resid, x - base)


cats = {r["category"]: r for r in csv.DictReader(open(os.path.join(IN, "categories.csv"), encoding="utf-8"))}
seasons = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(os.path.join(IN, "season_lists.csv"), encoding="utf-8")):
    seasons[r["category"]][int(r["season"])].append(r)

rows = []
for cid, c in cats.items():
    S = seasons[cid]
    short = {int(t) for t in c["short_seasons_left_out"].split()}
    shift = set() if args.no_shift else {int(t) for t in c["level_shift_seasons"].split()}
    scaled = c["is_total_scaled_by_schedule"] == "True"
    cut = float(c["cutoff_performance"]) if c["cutoff_performance"] else None
    first, start = int(c["first_season"]), int(c["performances_count_from"])
    for y in sorted(S):
        if y in short or y < start:
            continue
        lead = S[y][0]
        me, x = norm(lead["name_raw"]), float(lead["performance"])
        gy = float(lead["games_per_team"]) if scaled else None
        pts, real = [], 0
        for s in range(y - args.window, y + args.window + 1):
            if s in short or s not in S:
                continue
            o = next((e for e in S[s] if norm(e["name_raw"]) != me), None)
            if o is not None:
                p = float(o["performance"])
                if scaled:
                    p = p * gy / float(o["games_per_team"])
                pts.append((s, p))
                real += 1
            elif cut is not None and s >= first:
                pts.append((s, cut))
        if real < 8:
            continue
        z, T = measure(pts, y, x, shift, args.flat)
        if z is not None:
            rows.append({"cat": cid, "sport": c["sport"], "title": c["title"], "year": y, "name_raw": lead["name_raw"],
                         "name": lead["name_display"], "raw": lead["raw_value"], "z": z, "T": T, "n": len(pts)})

# check against the pipeline's own scores (same window, published settings only)
pe = os.path.join(IN, "pipeline_era.csv")
if os.path.exists(pe) and not args.flat and not args.no_shift:
    P = {(r["category"], int(r["season"])): float(r["z"]) for r in csv.DictReader(open(pe, encoding="utf-8"))
         if int(r["window"]) == args.window}
    if P:
        mine = {(r["cat"], r["year"]): r["z"] for r in rows}
        both = set(P) & set(mine)
        diff = max((abs(P[k] - mine[k]) for k in both), default=float("nan"))
        print(f"check vs pipeline (window {args.window}): {len(mine)} recomputed, {len(P)} in pipeline, "
              f"{len(both)} matched, {len(set(P) ^ set(mine))} unmatched, largest |z difference| {diff:.2e}\n")

# one entry per athlete-season, then the page's view: each athlete's best season
best = {}
for r in rows:
    k = (r["sport"], norm(r["name"]), r["year"])
    if k not in best or r["z"] > best[k]["z"]:
        best[k] = r
ranked = sorted(best.values(), key=lambda r: -r["z"])
seen, top = set(), []
for r in ranked:
    k = (r["sport"], norm(r["name"]))
    if k not in seen:
        seen.add(k)
        top.append(r)
label = f"window ±{args.window}" + (", flat (no trend)" if args.flat else "") + (", no suit shift" if args.no_shift else "")
print(f"greatest seasons, one per athlete ({label}); rank among all {len(ranked)} athlete-seasons in brackets")
for i, r in enumerate(top[:args.top]):
    T = "inf" if r["T"] == float("inf") else (f"{r['T']:,.0f}" if r["T"] else "-")
    print(f"{i + 1:3}. [{ranked.index(r) + 1:3}] z={r['z']:5.2f}  once in {T:>9}  {r['sport']:9} {r['name'][:24]:24} "
          f"{r['year']}  {r['title'][:32]:32} {r['raw']}  (n={r['n']})")

pub = os.path.join(args.dir, "outputs", "greatest_seasons.csv")
if os.path.exists(pub) and args.window == 15 and not args.flat and not args.no_shift:
    G = list(csv.DictReader(open(pub, encoding="utf-8")))
    bad = [(g["rank"], g["athlete"], g["z"], round(ranked[int(g["rank"]) - 1]["z"], 3))
           for g in G if abs(float(g["z"]) - ranked[int(g["rank"]) - 1]["z"]) > 1e-3]
    print(f"\npublished greatest_seasons.csv: {len(G)} rows, {len(bad)} whose z differs from the recomputed list at the same rank")
    for b in bad[:10]:
        print("   ", b)
