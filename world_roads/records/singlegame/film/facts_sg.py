"""Every number shown in "The 10 Biggest Single-Day Outliers in Sports History", computed from the single-game model
(../score.py, variant "big": team stats whose season-best game is usually 15 or more, plus athletics and swimming)
and written to facts_sg.json. The film reads only that file.

    python facts_sg.py [--stored-gamma]       # from singlegame/film/ (--stored-gamma: see ../score.py)

The top ten applies ../score.py's eligibility rule (clear of the field: at least one step clear of every other mark on
its own graph); the hook's distribution is every scored day, before the rule. If the NHL lists are not built, the
Sittler segment's numbers are carried over from the previous facts_sg.json (they come only from NHL data).
"""
import csv, json, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SG = os.path.dirname(HERE)
REC = os.path.dirname(SG)
sys.path.insert(0, SG)
import score as S                                           # noqa: E402
import gsmodel as G                                         # noqa: E402

FU = os.path.join(REC, "followup")
# rank: (palette, scene, photo key, achievement, how to say the season)
CARDS = {
    10: ("celtics", dict(kind="stack", icon="basketball", per=2, word="assists in a game", key="one ball = 2 assists",
                         bin_label="each stack: the most assists by\nanyone else that season"),
         "cousy", "28 assists in a game", "1958–59"),
    9: ("usa", dict(kind="pool", length=100), "gretchen_walsh", "54.33, 100 m butterfly", None),
    8: ("germany", dict(kind="javelin"), "vetter", "97.76 m javelin", None),
    7: ("france", dict(kind="pool", length=200), "marchand", "1:52.69, 200 m individual medley", None),
    6: ("ukraine", dict(kind="vault"), "bubka", "6.15 m pole vault, indoors", None),
    5: ("venezuela", dict(kind="pit", event="triple jump"), "rojas", "15.67 m triple jump", None),
    4: ("lakers", dict(kind="stack", icon="basketball", per=5, word="points in a game",
                       key="one ball = 5 points"), "bryant", "81 points in a game", "2005–06"),
    3: ("czech", dict(kind="javelin"), "zelezny", "98.48 m javelin", None),
    2: ("warriors", dict(kind="stack", icon="basketball", per=5, word="points in a game",
                         key="one ball = 5 points"), "chamberlain", "100 points in a game", "1961–62"),
    1: ("usa", dict(kind="pit", event="long jump"), "beamon", "8.90 m long jump", None),
}
SWIM_LEN = {"swim_W_800_fr": 800, "swim_M_200_me": 200, "swim_W_100_bu": 100}


def mmss(sec):
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:05.2f}" if m else f"{sec:.2f}"


def main():
    D, gam, grows, ctx = S.setup(stored_gamma="--stored-gamma" in sys.argv)
    spec = G.Spec(ties="all", trend="quad_safe", env="power")
    scored = S.margins(D, S.variant(D, ctx, G.score(D, spec, ctx=ctx, details=True), "big"))
    recs = S.clear_of_field(scored)
    page = G.by_athlete(G.athlete_seasons(recs))
    top = page[:10]
    recl = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(FU, "data", "original", "record_lines.csv"), encoding="utf-8")):
        recl[r["category"]].append((float(r["year_fraction"]), r["raw_value"], r["holder"]))
    ens = {r["athlete"]: int(r["specs_in_top10"]) for r in csv.DictReader(open(os.path.join(SG, "outputs", "sg_spec_ensemble_big.csv")))}
    cards = []
    for i, r in enumerate(top):
        rank = 10 - i if False else i + 1
        pal, scene, key, ach, season = CARDS[rank]
        cat, y = r["cat"], r["year"]
        m = D.meta[cat]
        lst = D.seasons[cat][y]
        ru = next((e for e in lst if e.id != r["id"]), None)
        kconv = ctx.env[cat][y] ** ctx.gamma.get(cat, 1.0) if cat in ctx.env_cats else 1.0
        eq = [(s_, v_ * kconv) for s_, v_ in r["pts"]]
        fb = G.fit_baseline(eq, y, m["shift"], "quad_safe")
        ye = np.array([q[0] for q in eq], float)
        ve = np.array([q[1] for q in eq], float)
        qe = min((ye < y).sum(), (ye > y).sum()) >= 5
        ce = np.polyfit(ye - y, ve, 2 if qe else 1)
        xr = r["x"] * kconv
        z = (xr - fb[0]) / fb[2]
        assert abs(z - r["z"]) < 1e-9, cat
        ruler = {"window": [[int(s_), round(v_, 6)] for (s_, v_), pid in zip(eq, r["pids"]) if pid != "cutoff"],
                 "x": float(xr), "base": float(fb[0]), "step": float(fb[2]), "z": float(z),
                 "trend": [[float(s_), float(np.polyval(ce, s_ - y))] for s_ in np.linspace(ye.min(), ye.max(), 60)],
                 "converted": cat in ctx.env_cats}
        hb = D.cats[cat].get("higher_raw_is_better", "True") == "True"
        line = sorted(recl.get(cat, []), key=lambda p: (p[0], float(p[1]) if hb else -float(p[1])))  # same day: in order
        if cat.startswith("sg_"):                              # team sports: the game data's own record line
            best_ = -1.0
            for s_ in sorted(D.seasons[cat]):
                e = D.seasons[cat][s_][0]
                if e.perf > best_:
                    line.append((float(s_), e.raw, e.name))
                    best_ = e.perf
        before = [p for p in line if p[0] < y]
        after = [p for p in line if p[0] >= y + 1]
        c = {"rank": rank, "category": cat, "title": D.cats[cat]["title"], "sport": m["sport"],
             "season": season or str(y), "year": y, "athlete": r["name"], "achievement": ach, "value": r["raw"],
             "z_e3": round(float(r["z"]), 2), "ruler": ruler, "palette": pal, "scene": scene, "photo": key,
             "runner_up": {"name": ru.name, "value": ru.raw} if ru else None,
             "previous_record": {"value": before[-1][1], "holder": before[-1][2], "year": int(before[-1][0])} if before else None,
             "next_record": {"value": after[0][1], "holder": after[0][2], "year": int(after[0][0])} if after else None,
             "games_per_team": None, "one_sided": bool(r["edge"]), "specs_in_top10": ens.get(r["name"]),
             "clear_steps": round(float(r["clear"]), 2), "best_other": list(r["beaten_by"]),
             "records_before": [{"value": p[1], "holder": p[2], "year": int(p[0])} for p in before[-4:]],
             "detail": D.detail.get((cat, y, r["id"])) if hasattr(D, "detail") else None}
        if cat in SWIM_LEN:                                    # say it in seconds and metres, not metres per second
            L = SWIM_LEN[cat]
            t_x, t_b = L / xr, L / fb[0]
            c["swim"] = {"length": L, "time": t_x, "curve_time": t_b, "gap_s": t_b - t_x, "step_s": (t_b - t_x) / z,
                         "behind_m_curve": L * (1 - fb[0] / xr), "step_m": L * fb[2] / xr}
            c["friendly"] = f"Curve about {mmss(t_b)} · one step about {(t_b - t_x) / z:.1f} seconds"
            c["arith"] = f"({mmss(t_b)} − {mmss(t_x)}) ÷ {(t_b - t_x) / z:.2f} s = {c['z_e3']:.2f}"
        cards.append(c)
        if cat.startswith("sg_"):                              # later games that equalled it (raw), before it was beaten
            nxt = c["next_record"]["year"] if c["next_record"] else 9999
            c["matched_by"] = [{"name": e.name, "year": s_, "value": e.raw,
                                "converted": next((v_ for s2, v_ in ruler["window"] if s2 == s_), None)}
                               for s_ in sorted(D.seasons[cat]) if y < s_ < nxt
                               for e in D.seasons[cat][s_] if e.perf >= float(r["raw"]) and e.id != r["id"]]
    # what the rule took out of the top ten: their score, and the mark on their graph that came within a step
    excluded = []
    for r in G.by_athlete(G.athlete_seasons(scored))[:10]:
        if r["clear"] < S.CLEAR_STEPS:
            s_, who = r["beaten_by"]
            e = next(e for e in D.seasons[r["cat"]][s_] if e.name == who)
            excluded.append({"athlete": r["name"], "year": r["year"], "title": r["title"], "value": r["raw"],
                             "z": round(float(r["z"]), 2), "clear_steps": round(float(r["clear"]), 2),
                             "best_other": {"name": who, "year": s_, "value": e.raw}})
    # the hook: every scored record of this variant, before the eligibility rule
    zs = np.array([r["z"] for r in scored])
    top_ids = {(c["category"], c["year"], c["athlete"]) for c in cards}
    scale = {"n": int(len(zs)), "median": round(float(np.median(zs)), 2), "at_least_4": int((zs >= 4).sum()),
             "at_least_5": int((zs >= 5).sum()), "below_1": int((zs < 1).sum()),
             "hist_edges": [round(e, 2) for e in np.arange(-4.0, 4.01, 0.5)],
             "hist_counts": [int(((zs >= lo) & (zs < lo + 0.5)).sum()) for lo in np.arange(-4.0, 4.0, 0.5)],
             "tail": sorted([{"name": r["name"], "year": r["year"], "cat": r["cat"], "z": round(r["z"], 2),
                              "top10": (r["cat"], r["year"], r["name"]) in top_ids} for r in scored if r["z"] >= 4],
                            key=lambda d: -d["z"]),
             "n_categories": len({r["cat"] for r in scored})}
    # Sittler: every category scored (variant "all"), so the whole-number problem can be shown
    allrec = G.score(D, spec, ctx=ctx, details=True)
    if "sg_nhl_points" not in D.meta:
        prev = json.load(open(os.path.join(HERE, "facts_sg.json"), encoding="utf-8"))
        print("NHL lists not built: the Sittler segment's numbers are carried over from the previous facts_sg.json")
        return finish(D, ctx, cards, scale, prev["sittler"], allrec, ens, excluded)
    st = next(r for r in allrec if r["cat"] == "sg_nhl_points" and r["year"] == 1976)
    k = ctx.env["sg_nhl_points"][1976] ** ctx.gamma["sg_nhl_points"]
    raw_others = {}
    for (s_, v_), pid in zip(st["pts"], st["pids"]):
        e = next(e for e in D.seasons["sg_nhl_points"][s_] if e.id == pid)
        raw_others[int(s_)] = (e.perf, e.name)
    eq = [(s_, v_ * k) for s_, v_ in st["pts"]]
    ye = np.array([q[0] for q in eq], float)
    ve = np.array([q[1] for q in eq], float)
    ce = np.polyfit(ye - 1976, ve, 2)
    allpage = G.by_athlete(G.athlete_seasons(allrec))
    corr = S.variant(D, ctx, allrec, "corrected")
    corrpage = G.by_athlete(G.athlete_seasons(corr))
    sittler = {"value": 10.0, "z": round(st["z"], 2), "base": st["base"] * k, "step": st["scale"] * k,
               "window": [[int(s_), round(v_, 4)] for s_, v_ in eq],
               "raw_others": {str(s_): v for s_, v in raw_others.items()},
               "trend": [[float(s_), float(np.polyval(ce, s_ - 1976))] for s_ in np.linspace(ye.min(), ye.max(), 60)],
               "rank_all": next(i + 1 for i, r in enumerate(allpage) if "Sittler" in r["name"]),
               "z_corrected": round(float(next(r["z"] for r in corr if r["cat"] == "sg_nhl_points" and r["year"] == 1976)), 2),
               "rank_corrected": next(i + 1 for i, r in enumerate(corrpage) if "Sittler" in r["name"]),
               "eights": sorted([(s_, n) for s_, (v, n) in raw_others.items() if v >= 8]),
               "sevens": sum(1 for v, _ in raw_others.values() if v == 7),
               "gamma": round(ctx.gamma["sg_nhl_points"], 2), "points_per_team_game": round(ctx.env["sg_nhl_points"][1976], 2)}
    return finish(D, ctx, cards, scale, sittler, allrec, ens, excluded)


def beamon_out(D, ctx):
    """the versions of the 16 where the rule takes Beamon out, and the jump that comes within a step"""
    out = []
    for wdw in (7, 10, 15, 20):
        for tr in ("ols", "quad_safe"):
            for env in ("none", "power"):
                sc = S.margins(D, S.variant(D, ctx, G.score(D, G.Spec(ties="all", window=wdw, trend=tr, env=env),
                                                            ctx=ctx if env == "power" else None, cats=["ath_M_LJ"],
                                                            details=True), "big"))
                b = next(r for r in sc if r["year"] == 1968)
                if b["clear"] < S.CLEAR_STEPS:
                    s_, who = b["beaten_by"]
                    e = next(e for e in D.seasons["ath_M_LJ"][s_] if e.name == who)
                    out.append({"window": wdw, "trend": tr, "env": env, "clear_steps": round(float(b["clear"]), 2),
                                "best_other": {"name": who, "year": s_, "value": e.raw}})
    return out


def finish(D, ctx, cards, scale, sittler, allrec, ens, excluded):
    burnett = next(r for r in allrec if r["cat"] == "sg_mlb_h" and r["year"] == 1932 and "Burnett" in r["name"])
    schmidt = next(r for r in allrec if r["cat"] == "sg_mlb_hr" and r["year"] == 1976 and "Schmidt" in r["name"])
    whole = {"burnett_z": round(burnett["z"], 2), "schmidt_z": round(schmidt["z"], 2)}
    # the two lists, for the close
    season = json.load(open(os.path.join(REC, "outliers", "facts.json"), encoding="utf-8"))
    lists = {"season": [(c["rank"], c["athlete"], c["achievement"], c["season"], c["z_e3"]) for c in season["cards"]],
             "day": [(c["rank"], c["athlete"], c["achievement"], c["season"], c["z_e3"]) for c in cards]}
    rob = {"in_top10_specs": {c["athlete"]: c["specs_in_top10"] for c in cards}, "beamon_out": beamon_out(D, ctx)}
    extras = {}
    import pandas as pd                                         # Wilt's earlier record and Kobe's second half
    ps = pd.read_csv(os.path.join(REC, "cache", "singlegame", "nba", "PlayerStatistics.csv"),
                     usecols=["firstName", "lastName", "gameDate", "points", "gameType"], low_memory=False)
    ps = ps[ps.gameType == "Regular Season"]
    w = ps[(ps.lastName == "Chamberlain") & (ps.gameDate.str[:10] < "1962-03-02") & (ps.gameDate.str[:10] >= "1961-10-01")]
    extras["wilt_best_before_100"] = [float(w.points.max()), str(w.loc[w.points.idxmax(), "gameDate"])[:10]]
    top_games = ps.sort_values("points", ascending=False).head(3)
    extras["nba_top_games"] = [(f"{a} {b}", float(p), str(d)[:10]) for a, b, p, d in
                               zip(top_games.firstName, top_games.lastName, top_games.points, top_games.gameDate)]
    extras["nba_points_per_team_game"] = {str(y): round(ctx.env["sg_nba_pts"][y], 1) for y in (1962, 2006)}
    extras["nba_assists_per_team_game"] = {str(y): round(ctx.env["sg_nba_ast"][y], 1) for y in (1959, 1963)}
    out = {"model": "single-game, E3_main_env method, variant big", "cards": cards, "scale": scale, "sittler": sittler,
           "whole_numbers": whole, "lists": lists, "robustness": rob, "extras": extras, "excluded": excluded,
           "rule_steps": S.CLEAR_STEPS}
    json.dump(out, open(os.path.join(HERE, "facts_sg.json"), "w"), indent=1, ensure_ascii=False, default=float)
    for c in cards:
        print(c["rank"], c["athlete"], c["season"], c["value"], c["z_e3"], "| ru", c["runner_up"], "| prev", c["previous_record"],
              "| next", c["next_record"], "| specs", c["specs_in_top10"], "| one-sided", c["one_sided"])
    print({k: v for k, v in scale.items() if k not in ("tail", "hist_counts", "hist_edges")})
    print({k: v for k, v in sittler.items() if k not in ("window", "trend", "raw_others")})
    print(whole, extras)


if __name__ == "__main__":
    main()
