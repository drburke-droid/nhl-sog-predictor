"""Every number film A shows (the ten biggest single-game outliers in team sports), into facts_a.json.

    python facts_a.py

The model is ../singlegame/topk.py (REVIEW_A.md): every game scaled to regulation length; each season's five best games
by anyone else, league-adjusted into the athlete's season (by at most 1.5x either way), make the class; counts are
measured from half a unit below; and the clear-of-the-field rule. Each card's ruler is that model's, in the athlete's
own season's units.
"""
import csv, json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
SGD = os.path.join(REC, "singlegame")
sys.path.insert(0, SGD)
import topk as T                                            # noqa: E402

# athlete: (palette, chart icon, chart word, photo key, achievement, season label)
LOOK = {
    "Wilt Chamberlain": ("warriors", "basketball", "points in a game", "chamberlain", "100 points", "1961–62"),
    "Kobe Bryant": ("lakers", "basketball", "points in a game", "bryant", "81 points", "2005–06"),
    "Norm Van Brocklin": ("rams", "football", "passing yards in a game", "vanbrocklin", "554 passing yards", "1951"),
    "Jim Bottomley": ("cardinals", "baseball", "runs batted in, one game", "bottomley", "12 runs batted in", "1924"),
    "Cloyce Box": ("lions", "football", "receiving yards in a game", "box", "302 receiving yards", "1950"),
    "Bob Cousy": ("celtics", "basketball", "assists in a game", "cousy", "28 assists", "1958–59"),
    "Mark Whiten": ("cardinals", "baseball", "runs batted in, one game", "whiten", "12 runs batted in", "1993"),
    "Darryl Sittler": ("leafs", "puck", "points in a game", "sittler", "10 points", "1975–76"),
    "Bert Olmstead": ("canadiens", "puck", "points in a game", "olmstead", "8 points", "1953–54"),
    "Derrick Thomas": ("chiefs", "football", "sacks in a game", "dthomas", "7 sacks", "1990"),
}
SPAN = {"NBA": True, "NHL": True}
UNIT = {"sg_nba_pts": "points", "sg_nba_ast": "assists", "sg_nfl_pass_yds": "yards", "sg_nfl_rec_yds": "yards",
        "sg_mlb_rbi": "RBI", "sg_nhl_points": "points", "sg_nfl_sacks": "sacks"}


def season(sport, y):
    return f"{y - 1}–{str(y)[2:]}" if SPAN.get(sport) else str(y)


def ruler_strings(raw, base, step, z):
    """'(raw − ½ − base) ÷ step = z' with the fewest decimals that make the arithmetic on screen give the score"""
    for db in range(0, 4):
        for ds in range(0, 4):
            bb, ss = round(base, db), round(step, ds)
            if ss > 0 and abs(round((raw - 0.5 - bb) / ss, 2) - round(z, 2)) < 1e-9:
                return f"{bb:.{db}f}", f"{ss:.{ds}f}"
    raise AssertionError((raw, base, step, z))


def main():
    D, gam, grows, ctx = T.setup()
    sc = T.score(D, ctx=ctx)
    elig = T.clear_of_field(sc)
    top = T.page(elig)[:10]
    ens = {r["athlete"]: int(r["specs_in_top10"]) for r in csv.DictReader(open(os.path.join(T.OUT, "tg_spec_ensemble.csv"),
                                                                                encoding="utf-8"))}
    kf = lambda r: ctx.env[r["cat"]][r["year"]] ** ctx.gamma.get(r["cat"], 1.0)
    cards = []
    for rank, r in enumerate(top, 1):
        pal, icon, word, key, ach, sea = LOOK[r["name"]]
        k = kf(r)
        cat, y = r["cat"], r["year"]
        Ss = D.seasons[cat]
        coef = np.array(r["coef"])
        yrs = [s for s, _ in r["pts"]]
        tr = [[float(s), float(k * sum(cf * (s - y) ** d for d, cf in enumerate(coef)))]
              for s in np.linspace(min(yrs), max(yrs), 60)]
        base, step = r["base"] * k, r["scale"] * k
        assert abs(r["unit"] * k - 1.0) < 1e-6, (cat, r["unit"] * k)
        raw = float(r["perf"])
        z = (raw - 0.5 - base) / step
        assert abs(z - r["z"]) < 1e-9, (cat, z, r["z"])
        bs, bname, braw, bv = r["beaten_by"]
        ru = next((e for e in Ss[y] if e.id != r["id"]), None)
        line, best = [], -1.0                                   # the game data's record line (by length-adjusted value)
        for s_ in sorted(Ss):
            e = Ss[s_][0]
            if e.perf > best:
                line.append((s_, e.raw, e.name))
                best = e.perf
        before = [p for p in line if p[0] < y]
        after = [p for p in line if p[0] > y]
        bb, ss = ruler_strings(raw, base, step, r["z"])
        d = D.detail[(cat, y, r["id"])]
        c = {"rank": rank, "category": cat, "title": r["title"], "sport": r["sport"], "season": sea, "year": y,
             "athlete": r["name"], "achievement": ach, "value": r["raw"], "z_e3": round(float(r["z"]), 2),
             "ruler": {"window5": [[int(s_), [round(float(v) * k, 4) for v in vs]] for s_, vs in r["pts"]],
                       "x_raw": raw, "x": raw - 0.5, "base": float(base), "step": float(step), "trend": tr,
                       "converted": True},
             "palette": pal, "photo": key, "scene": {"kind": "topk", "icon": icon, "word": word},
             "friendly": f"Curve about {base:.3g} {UNIT[cat]} · one step about {step:.2g} {UNIT[cat]}",
             "arith": f"({raw:g} − ½ − {bb}) ÷ {ss} = {r['z']:.2f}",
             "runner_up": {"name": ru.name, "value": ru.raw} if ru else None,
             "previous_record": {"value": before[-1][1], "holder": before[-1][2], "year": int(before[-1][0])} if before else None,
             "next_record": {"value": after[0][1], "holder": after[0][2], "year": int(after[0][0])} if after else None,
             "best_other": {"name": bname, "year": int(bs), "season": season(r["sport"], int(bs)), "value": braw,
                            "converted": round(float(bv) * k, 2)},
             "clear_steps": round(float(r["clear"]), 2), "specs_in_top10": ens.get(r["name"]), "one_sided": bool(r["edge"]),
             "league_rate": round(float(ctx.env[cat][y]), 3), "date": d["date"], "length": d["length"],
             "game": d["detail"], "games_per_team": None}
        cards.append(c)
    allpage = T.page(sc)
    excluded = []
    for r in allpage[:12]:
        if r["clear"] < T.CLEAR_STEPS:
            bs, bname, braw, bv = r["beaten_by"]
            excluded.append({"athlete": r["name"], "category": r["cat"], "year": r["year"],
                             "season": season(r["sport"], r["year"]), "title": r["title"], "value": r["raw"],
                             "z": round(float(r["z"]), 2), "clear_steps": round(float(r["clear"]), 2),
                             "best_other": {"name": bname, "year": int(bs), "season": season(r["sport"], int(bs)),
                                            "value": braw, "converted": round(float(bv) * kf(r), 2)}})
    fl = next(rr for rr in sc if rr["cat"] == "sg_nfl_rec_yds" and rr["year"] == 1989 and "Anderson" in rr["name"])
    fd = D.detail[(fl["cat"], fl["year"], fl["id"])]
    zs = np.array([r["z"] for r in sc])
    top_ids = {(c["category"], c["year"], c["athlete"]) for c in cards}
    F = {"model": "topk.py: length-adjusted games, top-5 curve, league adjustment capped at 1.5x, counts from half a "
                  "unit, clear of the field by one step",
         "cards": cards, "excluded": excluded,
         "anderson": {"raw": fd["raw_value"], "performance": round(float(fd["performance"]), 1), "length": fd["length"],
                      "date": fd["date"]},
         "scale": {"n": int(len(zs)), "n_categories": len(D.meta), "median": round(float(np.median(zs)), 2),
                   "at_least_4": int((zs >= 4).sum()), "at_least_5": int((zs >= 5).sum()),
                   "clear": len(elig), "hist_edges": [round(e, 2) for e in np.arange(-4.0, 4.01, 0.5)],
                   "hist_counts": [int(((zs >= lo) & (zs < lo + 0.5)).sum()) for lo in np.arange(-4.0, 4.0, 0.5)],
                   "below_hist": int((zs < -4).sum()),
                   "tail": sorted([{"name": r["name"], "year": r["year"], "cat": r["cat"], "z": round(float(r["z"]), 2),
                                    "top10": (r["cat"], r["year"], r["name"]) in top_ids} for r in sc if r["z"] >= 4],
                                  key=lambda d: -d["z"]),
                   "first_season": min(r["year"] for r in sc), "last_season": max(r["year"] for r in sc)},
         "cap": T.CAP, "robustness": {"n_specs": 16}}
    json.dump(F, open(os.path.join(HERE, "facts_a.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for c in cards:
        print(c["rank"], c["athlete"], c["z_e3"], c["arith"], "|", c["friendly"], "| ens", c["specs_in_top10"],
              "| ru", c["runner_up"], "| pr", c["previous_record"], "| nr", c["next_record"], "| bo", c["best_other"],
              "| rate", c["league_rate"])
    print("excluded:", excluded)
    print("anderson:", F["anderson"])
    print("scale:", {k: v for k, v in F["scale"].items() if k not in ("tail", "hist_edges", "hist_counts")})


if __name__ == "__main__":
    main()
