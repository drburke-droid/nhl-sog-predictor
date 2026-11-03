"""Cross-validation of the baseline at the level the score actually uses (review round 2, item 3).

    python analysis/cv_comparator.py        # from followup/; writes outputs/cv_comparator_*.csv (about 3 minutes)

Target. For every scored candidate (a season leader i in season y of category c), the quantity the baseline
must predict is the best performance by anyone other than i in season y, C_y^(i) (the runner-up). We withhold
the temporal block {y-1, y, y+1} from i's comparator series, fit each trend form to the remaining comparison
seasons of the window (same exclusion rules as the score), predict at y, and score the predictive N(base, scale):
  * CRPS divided by the candidate's unit (median absolute season-to-season change of its comparator series),
  * coverage of the central 90% interval,
  * the standardised error e = (C_y - base) / scale, which is also the empirical null used in
    calibrate.py: what z looks like for an ordinary (non-target) best performance.

Weighting. Means are reported per prediction, with equal weight per category, and with equal weight per sport.

Selection and test. Categories are split once (seeded, stratified by sport) into a selection half and a test
half. The rule, fixed before results: within the brief's +/-15 window, the trend with the lowest
equal-category-weighted CRPS on the selection half is the main trend; its improvement over the original
straight line is then measured on the test half, with a two-level bootstrap (categories, then 5-season blocks
within a category) for a 90% interval.
"""
import csv, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsmodel as G                                       # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "outputs")
SQPI = 1 / math.sqrt(math.pi)
TRENDS = ("flat", "ols", "huber", "quad", "quad_safe")
WINDOWS = (7, 10, 15, 20)


def crps_normal(y, mu, sd):
    z = (y - mu) / sd
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return sd * (z * (2 * Phi - 1) + 2 * phi - SQPI)


def split_categories(meta, seed=20261102):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for c, m in sorted(meta.items()):
        by[m["sport"]].append(c)
    sel, test = set(), set()
    for sp, cs in sorted(by.items()):
        cs = list(cs)
        rng.shuffle(cs)
        h = (len(cs) + 1) // 2
        sel |= set(cs[:h])
        test |= set(cs[h:])
    return sel, test


def predictions(D):
    rows = []
    for w in WINDOWS:
        recs = G.score(D, G.Spec(ties="all", window=w), details=True)
        for r in recs:
            pts = dict(r["pts"])
            y = r["year"]
            if y not in pts:
                continue
            target = pts[y]
            keep = [(s, v) for s, v in r["pts"] if abs(s - y) > 1]
            if len(keep) < 8:
                continue
            ser = [v for s, v in sorted(r["pts"])]
            diffs = np.abs(np.diff(ser))
            unit = float(np.median(diffs)) if len(diffs) else 0.0
            if unit <= 0:
                continue
            shift = D.meta[r["cat"]]["shift"]
            for tr in TRENDS:
                fb = G.fit_baseline(keep, y, shift, tr)
                if fb is None or not fb[2] > 0:
                    continue
                base, _, scale, _ = fb
                e = (target - base) / scale
                rows.append({"cat": r["cat"], "sport": r["sport"], "year": y, "window": w, "trend": tr,
                             "crps": crps_normal(target, base, scale) / unit, "cover": abs(e) <= 1.645, "e": e})
    return rows


def aggregate(rows, cats=None):
    out = {}
    g = defaultdict(list)
    for r in rows:
        if cats is None or r["cat"] in cats:
            g[(r["window"], r["trend"])].append(r)
    for k, rs in g.items():
        per_cat = defaultdict(list)
        sport_of = {}
        for r in rs:
            per_cat[r["cat"]].append(r["crps"])
            sport_of[r["cat"]] = r["sport"]
        cm = {c: float(np.mean(v)) for c, v in per_cat.items()}
        per_sport = defaultdict(list)
        for c, v in cm.items():
            per_sport[sport_of[c]].append(v)
        out[k] = {"n": len(rs), "per_prediction": float(np.mean([r["crps"] for r in rs])),
                  "equal_category": float(np.mean(list(cm.values()))),
                  "equal_sport": float(np.mean([np.mean(v) for v in per_sport.values()])),
                  "coverage_90": float(np.mean([r["cover"] for r in rs]))}
    return out


def paired_improvement(rows, cats, a, b, B=2000, seed=11, block=5):
    """mean over categories of (CRPS_b - CRPS_a) per candidate; positive = a better. Two-level bootstrap."""
    pa = {(r["cat"], r["year"]): r["crps"] for r in rows if (r["window"], r["trend"]) == a and r["cat"] in cats}
    pb = {(r["cat"], r["year"]): r["crps"] for r in rows if (r["window"], r["trend"]) == b and r["cat"] in cats}
    d = defaultdict(list)
    for k in sorted(set(pa) & set(pb)):
        d[k[0]].append(pb[k] - pa[k])
    cs = sorted(d)
    point = float(np.mean([np.mean(d[c]) for c in cs]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        means = []
        for c in rng.choice(cs, size=len(cs), replace=True):
            v = np.array(d[c])
            nb = max(1, int(math.ceil(len(v) / block)))
            starts = rng.integers(0, max(1, len(v) - block + 1), size=nb)
            samp = np.concatenate([v[s:s + block] for s in starts])[:len(v)]
            means.append(samp.mean())
        boots.append(float(np.mean(means)))
    return point, float(np.percentile(boots, 5)), float(np.percentile(boots, 95)), len(cs)


def main():
    D = G.Data("v2")
    rows = predictions(D)
    sel, test = split_categories(D.meta)
    allagg, selagg, testagg = aggregate(rows), aggregate(rows, sel), aggregate(rows, test)
    out = []
    for k in sorted(allagg, key=lambda k: allagg[k]["equal_category"]):
        out.append({"window": k[0], "trend": k[1], "predictions": allagg[k]["n"],
                    "crps_per_prediction": round(allagg[k]["per_prediction"], 4),
                    "crps_equal_category": round(allagg[k]["equal_category"], 4),
                    "crps_equal_sport": round(allagg[k]["equal_sport"], 4),
                    "coverage_90": round(allagg[k]["coverage_90"], 3),
                    "selection_half_equal_category": round(selagg[k]["equal_category"], 4) if k in selagg else "",
                    "test_half_equal_category": round(testagg[k]["equal_category"], 4) if k in testagg else ""})
    with open(os.path.join(OUT, "cv_comparator_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    # the pre-declared rule
    within = {k: v for k, v in selagg.items() if k[0] == 15}
    chosen = min(within, key=lambda k: within[k]["equal_category"])
    best_all = min(selagg, key=lambda k: selagg[k]["equal_category"])
    imp = paired_improvement(rows, test, chosen, (15, "ols"))
    imp_all = paired_improvement(rows, test, best_all, (15, "ols"))
    dec = {"selection_categories": sorted(sel), "test_categories": sorted(test),
           "chosen_within_15": list(chosen), "best_overall_on_selection": list(best_all),
           "test_improvement_chosen_vs_ols15": {"mean": imp[0], "p05": imp[1], "p95": imp[2], "categories": imp[3]},
           "test_improvement_best_overall_vs_ols15": {"mean": imp_all[0], "p05": imp_all[1], "p95": imp_all[2]}}
    import json
    json.dump(dec, open(os.path.join(OUT, "cv_comparator_decision.json"), "w"), indent=1)
    # empirical null: held-out standardised errors under the chosen specification (all categories)
    with open(os.path.join(OUT, "cv_comparator_null.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "sport", "season", "window", "trend", "standardised_error"])
        for r in rows:
            if (r["window"], r["trend"]) == chosen:
                w.writerow([r["cat"], r["sport"], r["year"], r["window"], r["trend"], round(r["e"], 5)])
    for r in out[:12]:
        print(r)
    print(json.dumps({k: v for k, v in dec.items() if "categories" not in k}, indent=1))


if __name__ == "__main__":
    main()
