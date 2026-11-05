"""Every number film B shows (the ten biggest single performances in athletics and swimming), into facts_b.json.

    python facts_b.py

The model is ../singlegame/review_B.py's: the single-day film's model on athletics and swimming only, each season's
best mark graded against the best mark by anyone else in the 15 seasons either side (quadratic trend, the athlete's own
seasons left out), and the clear-of-the-field rule. Cards are built exactly as ../singlegame/film/facts_sg.py builds
them (ruler, curve, records before and after), so the seven marks both films share come out the same.
"""
import csv, json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
SGD = os.path.join(REC, "singlegame")
sys.path.insert(0, SGD)
import review_B as RB                                       # noqa: E402  (Individual, ranked)

G, SG = RB.G, RB.SG
FU = os.path.join(REC, "followup")
# athlete: (palette, scene, photo key, achievement)
LOOK = {
    "Bob Beamon": ("usa", dict(kind="pit", event="long jump"), "beamon", "8.90 m long jump"),
    "Jan Železný": ("czech", dict(kind="javelin"), "zelezny", "98.48 m javelin"),
    "Yulimar Rojas": ("venezuela", dict(kind="pit", event="triple jump"), "rojas", "15.67 m triple jump"),
    "Sergey Bubka": ("ukraine", dict(kind="vault"), "bubka", "6.15 m pole vault, indoors"),
    "Léon Marchand": ("france", dict(kind="pool", length=200), "marchand", "1:52.69, 200 m individual medley"),
    "Johannes Vetter": ("germany", dict(kind="javelin"), "vetter", "97.76 m javelin"),
    "Gretchen Walsh": ("usa", dict(kind="pool", length=100), "gretchen_walsh", "54.33, 100 m butterfly"),
    "Armand Duplantis": ("sweden", dict(kind="vault"), "duplantis", "6.30 m pole vault"),
    "Summer McIntosh": ("canada", dict(kind="pool", length=400), "mcintosh", "4:23.65, 400 m individual medley"),
    "Usain Bolt": ("jamaica", dict(kind="track", length=100), "bolt", "9.58, 100 m"),
}
RACE_LEN = {"swim_W_800_fr": 800, "swim_M_200_me": 200, "swim_W_100_bu": 100, "swim_W_400_me": 400, "ath_M_100m": 100}


def mmss(sec):
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:05.2f}" if m else f"{sec:.2f}"


def main():
    D = RB.Individual()
    spec = G.Spec(ties="all", trend="quad_safe")
    scored, page = RB.ranked(D, spec)
    top = page[:10]
    ens = defaultdict(int)
    beamon_out = []                                            # the versions without Beamon, and what came within a step
    for wdw in (7, 10, 15, 20):
        for tr in ("ols", "quad_safe"):
            for env in ("none", "power"):                      # no league adjustment in these sports: counted twice
                sc_v, pg = RB.ranked(D, G.Spec(ties="all", window=wdw, trend=tr, env=env))
                for r in pg[:10]:
                    ens[(r["name"], r["sport"])] += 1
                if env == "none" and not any(r["name"] == "Bob Beamon" for r in pg[:10]):
                    b = next(r for r in sc_v if r["cat"] == "ath_M_LJ" and r["year"] == 1968)
                    s_, who = b["beaten_by"]
                    e = next(e for e in D.seasons["ath_M_LJ"][s_] if e.name == who)
                    beamon_out.append({"window": wdw, "trend": tr, "clear_steps": round(float(b["clear"]), 2),
                                       "best_other": {"name": who, "year": s_, "value": e.raw}})
    recl = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(FU, "data", "original", "record_lines.csv"), encoding="utf-8")):
        recl[r["category"]].append((float(r["year_fraction"]), r["raw_value"], r["holder"]))
    cards = []
    for rank, r in enumerate(top, 1):
        pal, scene, key, ach = LOOK[r["name"]]
        cat, y = r["cat"], r["year"]
        m = D.meta[cat]
        lst = D.seasons[cat][y]
        ru = next((e for e in lst if e.id != r["id"]), None)
        eq = [(s_, v_) for s_, v_ in r["pts"]]
        fb = G.fit_baseline(eq, y, m["shift"], "quad_safe")
        ye = np.array([q[0] for q in eq], float)
        ve = np.array([q[1] for q in eq], float)
        qe = min((ye < y).sum(), (ye > y).sum()) >= 5
        ce = np.polyfit(ye - y, ve, 2 if qe else 1)
        xr = r["x"]
        z = (xr - fb[0]) / fb[2]
        assert abs(z - r["z"]) < 1e-9, cat
        ruler = {"window": [[int(s_), round(v_, 6)] for (s_, v_), pid in zip(eq, r["pids"]) if pid != "cutoff"],
                 "x": float(xr), "base": float(fb[0]), "step": float(fb[2]), "z": float(z),
                 "trend": [[float(s_), float(np.polyval(ce, s_ - y))] for s_ in np.linspace(ye.min(), ye.max(), 60)],
                 "converted": False}
        hb = D.cats[cat].get("higher_raw_is_better", "True") == "True"
        line = sorted(recl.get(cat, []), key=lambda p: (p[0], float(p[1]) if hb else -float(p[1])))
        before = [p for p in line if p[0] < y]
        after = [p for p in line if p[0] >= y + 1]
        s_o, who = r["beaten_by"]
        oth = next((e for e in D.seasons[cat][s_o] if e.name == who), None)
        c = {"rank": rank, "category": cat, "title": D.cats[cat]["title"], "sport": m["sport"], "season": str(y),
             "year": y, "athlete": r["name"], "achievement": ach, "value": r["raw"], "z_e3": round(float(r["z"]), 2),
             "ruler": ruler, "palette": pal, "scene": scene, "photo": key,
             "runner_up": {"name": ru.name, "value": ru.raw} if ru else None,
             "previous_record": {"value": before[-1][1], "holder": before[-1][2], "year": int(before[-1][0])} if before else None,
             "next_record": {"value": after[0][1], "holder": after[0][2], "year": int(after[0][0])} if after else None,
             "games_per_team": None, "one_sided": bool(r["edge"]), "specs_in_top10": ens[(r["name"], r["sport"])],
             "clear_steps": round(float(r["clear"]), 2),
             "best_other": {"name": who, "year": s_o, "value": oth.raw if oth else None},
             "records_before": [{"value": p[1], "holder": p[2], "year": int(p[0])} for p in before[-4:]]}
        if cat in RACE_LEN:                                    # say it in seconds and metres, not metres per second
            L = RACE_LEN[cat]
            t_x, t_b = L / xr, L / fb[0]
            c["swim"] = {"length": L, "time": t_x, "curve_time": t_b, "gap_s": t_b - t_x, "step_s": (t_b - t_x) / z,
                         "behind_m_curve": L * (1 - fb[0] / xr), "step_m": L * fb[2] / xr}
            c["friendly"] = f"Curve about {mmss(t_b)} · one step about {(t_b - t_x) / z:.2f} seconds"
        cards.append(c)
    excluded = []
    for r in G.by_athlete(G.athlete_seasons(scored))[:12]:
        if r["clear"] < SG.CLEAR_STEPS:
            s_, who = r["beaten_by"]
            e = next(e for e in D.seasons[r["cat"]][s_] if e.name == who)
            excluded.append({"athlete": r["name"], "category": r["cat"], "year": r["year"], "title": D.cats[r["cat"]]["title"],
                             "value": r["raw"], "z": round(float(r["z"]), 2), "clear_steps": round(float(r["clear"]), 2),
                             "best_other": {"name": who, "year": s_, "value": e.raw}})
    flo = next(r for r in scored if r["cat"] == "ath_W_100m" and r["year"] == 1988 and "Griffith" in r["name"])
    s_, who = flo["beaten_by"]
    flo_o = next(e for e in D.seasons["ath_W_100m"][s_] if e.name == who)
    zs = np.array([r["z"] for r in scored])
    top_ids = {(c["category"], c["year"], c["athlete"]) for c in cards}
    F = {"model": "review_B: athletics and swimming, the single-day model, clear of the field by one step",
         "cards": cards, "excluded": excluded,
         "flojo": {"z": round(float(flo["z"]), 2), "clear_steps": round(float(flo["clear"]), 2), "value": flo["raw"],
                   "best_other": {"name": who, "year": s_, "value": flo_o.raw}},
         "scale": {"n": int(len(zs)), "n_categories": len(D.meta), "median": round(float(np.median(zs)), 2),
                   "at_least_4": int((zs >= 4).sum()), "at_least_5": int((zs >= 5).sum()),
                   "clear": sum(r["clear"] >= SG.CLEAR_STEPS for r in scored),
                   "hist_edges": [round(e, 2) for e in np.arange(-4.0, 4.01, 0.5)],
                   "hist_counts": [int(((zs >= lo) & (zs < lo + 0.5)).sum()) for lo in np.arange(-4.0, 4.0, 0.5)],
                   "below_hist": int((zs < -4).sum()),
                   "tail": sorted([{"name": r["name"], "year": r["year"], "cat": r["cat"], "z": round(float(r["z"]), 2),
                                    "top10": (r["cat"], r["year"], r["name"]) in top_ids} for r in scored if r["z"] >= 4],
                                  key=lambda d: -d["z"]),
                   "first_season": min(r["year"] for r in scored), "last_season": max(r["year"] for r in scored)},
         "robustness": {"n_specs": 16, "in_top10_specs": {c["athlete"]: c["specs_in_top10"] for c in cards},
                        "beamon_out": beamon_out}}
    json.dump(F, open(os.path.join(HERE, "facts_b.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for c in cards:
        print(c["rank"], c["athlete"], c["category"], c["z_e3"], c["specs_in_top10"], c["clear_steps"], c["best_other"],
              "| ru", c["runner_up"], "| pr", c["previous_record"], "| nr", c["next_record"])
    print("excluded:", [(e["athlete"], e["year"], e["z"], e["clear_steps"], e["best_other"]) for e in excluded])
    print("flojo:", F["flojo"])
    print("beamon out:", beamon_out)
    print("scale:", {k: v for k, v in F["scale"].items() if k not in ("tail", "hist_edges", "hist_counts")})


if __name__ == "__main__":
    main()
