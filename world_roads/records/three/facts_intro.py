"""The numbers the three films' openings show (why performances must be compared within their era, and how), into
intro.json.

    python facts_intro.py

improve: world records in athletics and swimming since 1950 (followup/data/original/record_lines.csv): for every event
with a record by 1950, the record in force at the end of each year, as a share above its 1950 level (races as speed).
tiles: "same number, different era" for each film, with the league rate of the time (Baseball-Reference, from the
context tables). nhl_rate: NHL goals per team per game (Hockey-Reference). example: each film's worked example of the
method, in the athlete's own units. throws: film B's women's throws of the 1980s.
"""
import csv, json, os, re, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
CTX = os.path.join(REC, "followup", "data", "context")
sys.path.insert(0, os.path.join(REC, "singlegame"))
sys.path.insert(0, os.path.join(REC, "outliers"))


def dist(cat):
    m = re.match(r"ath_[MW]_(\d+)m", cat) or re.match(r"swim_[MW]_(\d+)_", cat)
    return float(m.group(1)) if m else None


def improve():
    cats = {r["category"]: r for r in csv.DictReader(open(os.path.join(REC, "followup", "data", "original", "categories.csv"),
                                                          encoding="utf-8"))}
    rl = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(REC, "followup", "data", "original", "record_lines.csv"), encoding="utf-8")):
        if r["category"].startswith(("ath_", "swim_")):
            rl[r["category"]].append((float(r["year_fraction"]), float(r["raw_value"])))
    years = list(range(1950, 2027))
    lines = {}
    for cat, pts in rl.items():
        pts.sort()
        if pts[0][0] > 1950.0 or cat in ("ath_W_PV", "ath_W_TJ"):   # not contested internationally until the 1990s
            continue
        higher = cats[cat]["higher_raw_is_better"] == "True"
        L = dist(cat)
        perf = (lambda v: v) if higher else (lambda v: L / v)
        def at(y):
            best = None
            for t_, v in pts:
                if t_ <= y + 1.0:
                    p = perf(v)
                    best = p if best is None else max(best, p)
            return best
        base = at(1949)
        lines[cat] = [round(100.0 * (at(y) / base - 1.0), 3) for y in years]
    arr = np.array(list(lines.values()))
    return {"years": years, "median": np.median(arr, axis=0).round(3).tolist(),
            "q25": np.percentile(arr, 25, axis=0).round(3).tolist(), "q75": np.percentile(arr, 75, axis=0).round(3).tolist(),
            "lines": lines, "n_events": len(lines),
            "examples": {c: {"y1950": next(v for t_, v in sorted(rl[c], reverse=True) if t_ <= 1950.0),
                             "now": (lambda b: b)(max(rl[c], key=lambda p: (p[1] if cats[c]["higher_raw_is_better"] == "True" else -p[1]))[1]),
                             "now_year": int(max(rl[c], key=lambda p: (p[1] if cats[c]["higher_raw_is_better"] == "True" else -p[1]))[0])}
                         for c in ("ath_M_100m", "ath_M_PV", "swim_M_1500_fr", "ath_M_SP") if c in rl}}


def league(name):
    return {int(r["season"]): r for r in csv.DictReader(open(os.path.join(CTX, name), encoding="utf-8"))}


def ruler_season(D, ctx, G, r, y):
    """window (best by anyone else per season), curve and step of a season-model record, in the season's own units"""
    cat = r["cat"]
    m = D.meta[cat]
    k = (ctx.env[cat][y] ** ctx.gamma.get(cat, 1.0)) if (ctx and cat in getattr(ctx, "env_cats", ctx.env)) else 1.0
    eq = [(s_, v_ * k) for s_, v_ in r["pts"]]
    fb = G.fit_baseline(eq, y, m["shift"], "quad_safe")
    ye = np.array([q[0] for q in eq], float)
    ve = np.array([q[1] for q in eq], float)
    qe = min((ye < y).sum(), (ye > y).sum()) >= 5
    ce = np.polyfit(ye - y, ve, 2 if qe else 1)
    return {"pts": [[int(s_), [round(float(v_), 4)]] for (s_, v_), pid in zip(eq, r["pids"]) if pid != "cutoff"],
            "x": float(r["x"] * k), "base": float(fb[0]), "step": float(fb[2]), "z": round(float(r["z"]), 2),
            "trend": [[float(s_), float(np.polyval(ce, s_ - y))] for s_ in np.linspace(ye.min(), ye.max(), 60)]}


def main():
    out = {"improve": improve(), "nhl_rate": [[s, float(r["goals_per_team_g"])] for s, r in sorted(league("nhl_league.csv").items())
                                              if r.get("goals_per_team_g")]}
    mlb = league("mlb_league.csv")
    # ---- film C: 60 home runs, three eras; Bonds's 73 as the worked example
    import review_C as RC
    D, gam, ctx = RC.setup()
    G = RC.G
    tiles = []
    for y, who in ((1927, "Babe Ruth"), (1998, "Mark McGwire"), (2022, "Aaron Judge")):
        e = next(e for e in D.seasons["mlb_hr"][y] if e.name == who)
        tiles.append({"value": RC.val(e.raw), "who": who, "year": y, "rate": round(float(mlb[y]["hr_per_team_g"]), 2)})
    sc = RC.SG.margins(D, G.score(D, G.Spec(ties="all", env="power", trend="quad_safe"), ctx=ctx, details=True))
    b = next(r for r in sc if r["cat"] == "mlb_hr" and r["year"] == 2001 and r["name"] == "Barry Bonds")
    ex = ruler_season(D, ctx, G, b, 2001)
    s_, who = b["beaten_by"]
    ex.update(name="Barry Bonds", year=2001, raw=73, unit="home runs", label="Barry Bonds, 2001: 73",
              best_other={"name": who, "year": s_, "value": RC.val(next(e.raw for e in D.seasons["mlb_hr"][s_] if e.name == who)),
                          "converted": round(RC.to_own_units(dict(b["pts"])[s_], b, ctx), 1)}, clear=round(float(b["clear"]), 2))
    out["c"] = {"tiles": tiles, "example": ex, "nhl": {y: float(league("nhl_league.csv")[y]["goals_per_team_g"]) for y in (1982, 1986, 2004, 2025)}}
    # ---- film A: four home runs in a game, three eras; Adrian Peterson's 296 as the worked example
    import topk as T
    DA, gamA, growsA, ctxA = T.setup()
    tiles = []
    for y, who in ((1932, "Lou Gehrig"), (2002, "Mike Cameron"), (2017, "J.D. Martinez")):
        e = next(e for e in DA.seasons["sg_mlb_hr"][y] if who.split()[-1] in e.name and float(e.raw) == 4.0)
        tiles.append({"value": "4", "who": e.name, "year": y, "rate": round(float(mlb[y]["hr_per_team_g"]), 2)})
    scA = T.score(DA, ctx=ctxA)
    p = next(r for r in scA if r["cat"] == "sg_nfl_rush_yds" and r["year"] == 2007 and "Peterson" in r["name"])
    k = ctxA.env[p["cat"]][2007] ** ctxA.gamma.get(p["cat"], 1.0)
    coef = np.array(p["coef"])
    yrs = [s for s, _ in p["pts"]]
    exA = {"pts": [[int(s), [round(float(v) * k, 3) for v in vs]] for s, vs in p["pts"]], "x": float(p["perf"]) - 0.5,
           "base": float(p["base"] * k), "step": float(p["scale"] * k), "z": round(float(p["z"]), 2),
           "trend": [[float(s), float(k * sum(c * (s - 2007) ** d for d, c in enumerate(coef)))] for s in np.linspace(min(yrs), max(yrs), 60)],
           "name": p["name"], "year": 2007, "raw": float(p["perf"]), "unit": "rushing yards",
           "label": f"Adrian Peterson, 2007: {float(p['perf']):.0f}"}
    # the rest of the field: each season's tenth-best game by anyone else, in 2007 terms (the curve is the best, not
    # everyone: this line shows how far below the best the rest of a season's top ten sits)
    fld = []
    for s, vs in exA["pts"]:
        others = sorted((e for e in DA.seasons[p["cat"]][s] if not (s == 2007 and "Peterson" in e.name)),
                        key=lambda e: -float(e.perf))
        if len(others) >= 10:
            fld.append([s, round(float(others[9].perf) * vs[0] / float(others[0].perf), 1)])
    fc = np.polyfit([a for a, _ in fld], [b for _, b in fld], 2)
    exA["field"] = {"rank": 10, "pts": fld, "trend": [[float(s), float(np.polyval(fc, s))] for s in np.linspace(min(yrs), max(yrs), 60)],
                    "own_year": next(b for a, b in fld if a == 2007), "label": "the tenth-best game each season"}
    exA["field"]["gap_text"] = f"{exA['raw'] - exA['field']['own_year']:.0f} more yards than\n2007's tenth-best game"
    out["a"] = {"tiles": tiles, "example": exA}
    # ---- film B: world records then and now; Ledecky's 800 m as the worked example; the women's throws
    import review_B as RB
    DB = RB.Individual()
    scB, _ = RB.ranked(DB, G.Spec(ties="all", trend="quad_safe"))
    led = next(r for r in scB if r["cat"] == "swim_W_800_fr" and r["year"] == 2016)
    exB = ruler_season(DB, None, G, led, 2016)
    exB.update(name="Katie Ledecky", year=2016, raw=float(led["raw"]), unit="seconds", length=800.0,
               label="Katie Ledecky, 2016: 8:04.79")
    thr = {}
    for cat, y in (("ath_W_SP", 1987), ("ath_W_DT", 1988)):
        r = next(r for r in scB if r["cat"] == cat and r["year"] == y)
        s_, who = r["beaten_by"]
        thr[cat] = {"name": r["name"], "year": y, "value": float(r["raw"]), "z": round(float(r["z"]), 2),
                    "clear": round(float(r["clear"]), 2), "best_other": {"name": who, "year": s_},
                    "leads": [[s, float(DB.seasons[cat][s][0].raw)] for s in sorted(DB.seasons[cat])],
                    "ruler": ruler_season(DB, None, G, r, y)}
    season_film = json.load(open(os.path.join(REC, "outliers", "facts.json"), encoding="utf-8"))["womens_throws"]
    for cat in ("ath_W_SP", "ath_W_DT"):
        sf = season_film[cat]
        thr[cat].update(threshold=sf["threshold"], marks_1980s=sf["marks_at_or_beyond_threshold_by_decade"]["1980"],
                        athletes_1980s=sf["athletes_at_or_beyond_threshold_by_decade"]["1980"],
                        last_year_at_threshold=sf.get("last_year_at_or_beyond_threshold"))
    out["b"] = {"example": exB, "throws": thr}
    json.dump(out, open(os.path.join(HERE, "intro.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("improve:", out["improve"]["n_events"], "events; median 2026:", out["improve"]["median"][-1],
          "q25/q75:", out["improve"]["q25"][-1], out["improve"]["q75"][-1], out["improve"]["examples"])
    print("C tiles:", out["c"]["tiles"], "| Bonds z", ex["z"], "base", round(ex["base"], 1), "step", round(ex["step"], 2),
          "clear", ex["clear"], ex["best_other"], "| nhl", out["c"]["nhl"])
    print("A tiles:", out["a"]["tiles"], "| Peterson z", exA["z"], "base", round(exA["base"], 1), "step", round(exA["step"], 1))
    print("B: Ledecky z", exB["z"], "base speed", round(exB["base"], 4), "| throws:",
          {c: (v["name"], v["value"], v["z"], v["clear"], v["marks_1980s"], v["athletes_1980s"], v["last_year_at_threshold"])
           for c, v in thr.items()})


if __name__ == "__main__":
    main()
