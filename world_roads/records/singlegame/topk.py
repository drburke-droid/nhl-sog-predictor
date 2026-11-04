"""The single-game team-sports model (video A): the greatest-seasons method with the changes agreed for this film.

1. Length: every game is scaled to regulation length before anything else (team_games.py), so extra innings and
   overtime don't count as extra chances.
2. The curve runs through the top five, not the top one: for each season in the window (15 either side), the five best
   games by anyone else (the athlete's own games left out, league-adjusted into the athlete's season). The curve is a
   trend through each season's mean of those five (quadratic where there are 5+ seasons on each side); one step is the
   spread of those individual games around it. Small whole-number stats (home runs, goals, hits), where the season's
   single best is usually the same number every year, then get a real spread instead of none, and every team stat is
   measured the same way.
3. Clear of the field: a game makes the ranking only if it beats every other game on its graph by at least one step.
4. Counts are counts: 7 hits stands for anything from 6.5 to 7.5, so the score uses x - half a unit and the step adds
   the rounding variance (unit^2 / 12). Without it, a season whose top five are all exactly 5 hits has almost no
   spread, and a routine 6-hit game scored 3.7; with it, about 1. Big-number stats (points, yards) barely move.
5. The league conversion is capped: another season's game is scaled into the athlete's season by at most 1.5x either
   way (CAP). Uncapped, early hockey's swings (4.8 goals a team-game in 1920, 1.4 in 1928) turned a 4-goal game in 1928
   into 8.8 in 1920 terms, and Joe Malone's record 7 scored about 1.
6. Three-pointers count from 1994-95 (START): in the first seasons of the shot, rates were so low that the conversion
   decided everything (Roche's 8 in 1982 in, Barry's 8 in 1980 out).

League adjustment: value / (league per-team-game rate)^gamma, gamma estimated from these lists and pooled across the 18
categories (run_all.pooled_gamma). Ties at the top of a season all count.

    python topk.py              # -> outputs/tg_ranking.csv, tg_beaten.csv, tg_spec_ensemble.csv, tg_gamma.csv,
                                #    tg_by_category.csv, tg_scale.json
"""
import csv, json, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import score as S                                          # noqa: E402  (league columns, paths)
import gsmodel as G                                        # noqa: E402
import context as X                                        # noqa: E402
import run_all as RA                                       # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
K = 5
CLEAR_STEPS = 1.0
CAP = 1.5                       # no other season's game is converted by more than 1.5x either way
START = {"sg_nba_3pm": 1995}    # threes count from 1994-95, once the shot was established


class TGData(G.Data):
    """the 18 team categories, length-adjusted (team_games.py merge)"""

    def __init__(self):
        self.version = "tg"
        self.cats = {r["category"]: r for r in csv.DictReader(open(os.path.join(DATA, "tg_categories.csv"), encoding="utf-8"))}
        by = defaultdict(list)
        self.detail = {}
        for r in csv.DictReader(open(os.path.join(DATA, "tg_season_lists.csv"), encoding="utf-8")):
            by[(r["category"], int(r["season"]))].append(r)
            self.detail[(r["category"], int(r["season"]), r["athlete_id"])] = r
        self.seasons = defaultdict(dict)
        for (c, y), rs in by.items():
            rs.sort(key=lambda r: int(r["rank"]))
            self.seasons[c][y] = [G.Entry(r["athlete_id"], r["name_display"], float(r["performance"]), r["raw_value"],
                                          math.nan) for r in rs]
        self.meta = {c: {"sport": m["sport"], "title": m["title"],
                         "short": {int(t) for t in m["short_seasons_left_out"].split()}, "shift": set(), "scaled": False,
                         "cut": None, "first": START.get(c, int(m["first_season"])),
                         "start": START.get(c, int(m["performances_count_from"]))}
                     for c, m in self.cats.items()}
        for c, y0 in START.items():                            # earlier seasons are neither scored nor compared
            for y in [y for y in self.seasons[c] if y < y0]:
                del self.seasons[c][y]


def setup():
    D = TGData()
    X.ENV_MAP.clear()
    X.ENV_MAP.update({c: (lg, col, "ratio") for c, (lg, col) in S.LEAGUE_COL.items() if c in D.meta})
    gam, grows = RA.pooled_gamma(D)
    ctx = X.make(D, env="power", gamma=gam)
    return D, gam, grows, ctx


def fit(points, y, trend):
    """points: [(season, [values])] -> (curve at y, step, coefficients, design degree)"""
    ys = np.array([s for s, _ in points], float)
    means = np.array([np.mean(v) for _, v in points])
    quad = trend == "quad" or (trend == "quad_safe" and min((ys < y).sum(), (ys > y).sum()) >= 5)
    deg = 2 if quad else (1 if trend in ("ols", "quad_safe", "quad") else 0)
    X_ = np.column_stack([(ys - y) ** d for d in range(deg + 1)])
    coef = np.linalg.lstsq(X_, means, rcond=None)[0]
    res = np.concatenate([np.asarray(v) - sum(c * (s - y) ** d for d, c in enumerate(coef)) for s, v in points])
    dof = len(res) - (deg + 1)
    step = math.sqrt(float(res @ res) / dof) if dof > 0 else float("nan")
    return float(coef[0]), step, coef, deg


def unit(cat, y, spec, ctx):
    """one counted unit (a hit, a point, a yard) in the comparison scale of season y"""
    one = G.transform(1.0, cat, "", y, spec, ctx)
    return one if one else 1.0


def score(D, spec=G.Spec(ties="all", trend="quad_safe", env="power"), ctx=None, k=K, cats=None, correct=True, cap=CAP):
    """every season-leading game -> record with z (steps above the top-k curve) and 'clear' (steps over the best other
    game on its graph)"""
    ctx = ctx or G.Context()
    out = []
    for cat in (cats or D.meta):
        m, Ss = D.meta[cat], D.seasons[cat]
        for y in sorted(Ss):
            if y in m["short"] or y < m["start"]:
                continue
            for lead in G.leaders(Ss[y], spec.ties):
                x = G.transform(lead.perf, cat, m["sport"], y, spec, ctx)
                if x is None:
                    continue
                ky = (lead.perf / x) if x else 1.0                       # the athlete's season's divisor (1 without env)
                pts, who = [], []
                for s in range(y - spec.window, y + spec.window + 1):
                    if s in m["short"] or s not in Ss:
                        continue
                    o = [e for e in Ss[s] if e.id != lead.id][:k]
                    if len(o) < k:
                        continue
                    v = [G.transform(e.perf, cat, m["sport"], s, spec, ctx) for e in o]
                    if any(q is None for q in v):
                        continue
                    if cap and o[0].perf and v[0]:                         # capped conversion into season y
                        ks = o[0].perf / v[0]
                        r_ = min(max(ky / ks, 1 / cap), cap)
                        v = [e.perf * r_ / ky for e in o]
                    pts.append((s, v))
                    who.append(o)
                if len(pts) < spec.min_real:
                    continue
                base, step, coef, deg = fit(pts, y, spec.trend)
                if not step > 0:
                    continue
                vals = [(s, q, e) for (s, v), o in zip(pts, who) for q, e in zip(v, o)]
                bs, bv, be = max(vals, key=lambda t: t[1])
                u = unit(cat, y, spec, ctx) if correct else 0.0     # counts: 7 stands for 6.5 to 7.5
                xc, stc = x - 0.5 * u, math.sqrt(step ** 2 + u ** 2 / 12)
                before = sum(1 for s, _ in pts if s < y)
                after = sum(1 for s, _ in pts if s > y)
                out.append({"cat": cat, "sport": m["sport"], "title": m["title"], "year": y, "id": lead.id,
                            "name": lead.name, "raw": lead.raw, "perf": lead.perf, "x": x, "base": base, "scale": stc,
                            "z": (xc - base) / stc, "clear": (x - bv) / stc, "beaten_by": (bs, be.name, be.raw, bv),
                            "unit": u, "step_raw_fit": step,
                            "pts": pts, "coef": coef.tolist(), "deg": deg, "n": len(pts), "before": before,
                            "after": after, "edge": min(before, after) < 5})
    return out


def clear_of_field(recs):
    return [r for r in recs if r["clear"] >= CLEAR_STEPS]


def page(recs):
    return G.by_athlete(G.athlete_seasons(recs))


def main():
    os.makedirs(OUT, exist_ok=True)
    D, gam, grows, ctx = setup()
    scored = score(D, ctx=ctx)
    recs = clear_of_field(scored)
    top = page(recs)
    with open(os.path.join(OUT, "tg_gamma.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0]))
        w.writeheader()
        w.writerows(grows)
    ens = defaultdict(int)                                     # 16 versions: windows x trends x league adjustment
    for wdw in (7, 10, 15, 20):
        for tr in ("ols", "quad_safe"):
            for env in ("none", "power"):
                pg = page(clear_of_field(score(D, G.Spec(ties="all", window=wdw, trend=tr, env=env),
                                               ctx=ctx if env == "power" else None)))
                for r in pg[:10]:
                    ens[r["name"]] += 1
    det = lambda r: D.detail[(r["cat"], r["year"], r["id"])]
    cols = ["rank", "athlete", "sport", "category", "title", "season", "raw", "length", "adjusted", "z", "curve", "step",
            "margin_over_best_other", "best_other", "date", "detail", "one_sided_window", "specs_in_top10"]

    def row(i, r):                                         # curve, step and best other in the athlete's own units
        d = det(r)
        bo = r["beaten_by"]
        kc = ctx.env[r["cat"]][r["year"]] ** ctx.gamma.get(r["cat"], 1.0)
        return [i, r["name"], r["sport"], r["cat"], r["title"], r["year"], r["raw"], d["length"], round(r["perf"], 3),
                round(r["z"], 3), round(r["base"] * kc, 3), round(r["scale"] * kc, 3), round(r["clear"], 3),
                f"{bo[1]} {bo[2]} ({bo[0]}; {bo[3] * kc:.2f} in this season's terms)", d["date"], d["detail"], r["edge"],
                ens.get(r["name"], 0)]
    with open(os.path.join(OUT, "tg_ranking.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(top, 1):
            w.writerow(row(i, r))
    with open(os.path.join(OUT, "tg_beaten.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank_without_rule"] + cols[1:])
        for i, r in enumerate(page(scored)[:100], 1):
            if r["clear"] < CLEAR_STEPS:
                w.writerow(row(i, r))
    best_cat = {}
    for r in scored:
        if r["cat"] not in best_cat or r["z"] > best_cat[r["cat"]]["z"]:
            best_cat[r["cat"]] = r
    with open(os.path.join(OUT, "tg_by_category.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols[1:])
        for c, r in sorted(best_cat.items(), key=lambda kv: -kv[1]["z"]):
            w.writerow(row(0, r)[1:])
    with open(os.path.join(OUT, "tg_spec_ensemble.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["athlete", "specs_in_top10", "specs"])
        for n, k_ in sorted(ens.items(), key=lambda kv: -kv[1]):
            w.writerow([n, k_, 16])
    zs = np.array([r["z"] for r in scored])
    json.dump({"n": int(len(zs)), "median": round(float(np.median(zs)), 2), "at_least_4": int((zs >= 4).sum()),
               "at_least_5": int((zs >= 5).sum()), "eligible": len(recs), "n_categories": len(D.meta), "k": K,
               "clear_steps": CLEAR_STEPS}, open(os.path.join(OUT, "tg_scale.json"), "w"), indent=1)
    print(f"{len(scored)} season-leading games in {len(D.meta)} categories; {len(recs)} clear of the field")
    for i, r in enumerate(top[:25], 1):
        d = det(r)
        print(f"{i:3d} {r['z']:5.2f}  {r['name']:24s} {r['year']}  {r['title'][:30]:30s} {r['raw']:>6s} "
              f"({d['length'] or 'length n/a'})  clear {r['clear']:.2f}  [{ens.get(r['name'], 0)}/16]" + ("  one-sided" if r["edge"] else ""))


if __name__ == "__main__":
    main()
