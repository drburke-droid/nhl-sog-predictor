"""Every number shown in "The 10 Biggest Statistical Outliers in Sports History", computed from the locked
follow-up analysis (model E3_main_env) and written to facts.json. The film reads only facts.json.

    python facts.py          # from world_roads/records/outliers/

Sources: ../followup/data (original, v2, context) and ../followup/outputs (rankings_sensitivity.csv,
spec_ensemble.csv, gamma_estimates.csv), plus ../cache/alltime_parsed.csv for the women's throws counts.
"""
import csv, json, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
FU = os.path.join(REC, "followup")
sys.path.insert(0, os.path.join(FU, "analysis"))
import gsmodel as G                                           # noqa: E402
import context as X                                           # noqa: E402

COUNTDOWN = [  # rank, category, season (ending year), how to say the season, short achievement
    (10, "mlb_h", 2004, "2004", "262 hits"),
    (9, "mlb_hr", 1920, "1920", "54 home runs"),
    (8, "nba_rpg", 1993, "1992–93", "18.3 rebounds a game"),
    (7, "nba_3pm", 2016, "2015–16", "402 three-pointers"),
    (6, "mlb_so", 1924, "1924", "262 strikeouts"),
    (5, "ath_M_JT", 1996, "1996", "98.48 m javelin"),
    (4, "nhl_assists", 1986, "1985–86", "163 assists"),
    (3, "mlb_streak", 1941, "1941", "56-game hitting streak"),
    (2, "nba_ppg", 1962, "1961–62", "50.4 points a game"),
    (1, "ath_M_LJ", 1968, "1968", "8.90 m long jump"),
]


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def main():
    D = G.Data("v2")
    import run_all as RA                                      # the same unrounded pooled elasticities as the locked run
    gam, _ = RA.pooled_gamma(D)
    ctx = X.make(D, env="power", gamma=gam)
    e3_all = G.score(D, G.Spec(ties="all", trend="quad_safe", env="power"), ctx=ctx, details=True)
    e3 = {(r["cat"], r["year"]): r for r in e3_all}
    raw = {(r["cat"], r["year"]): r for r in G.score(D, G.Spec(ties="all", trend="quad_safe"), details=True)}
    S = read(os.path.join(FU, "outputs", "rankings_sensitivity.csv"))
    E = X.env_series()
    names = {}
    for c, by in D.seasons.items():
        for y, lst in by.items():
            for e in lst:
                names.setdefault(e.id, e.name)
    recl = defaultdict(list)
    for r in read(os.path.join(FU, "data", "original", "record_lines.csv")):
        recl[r["category"]].append((float(r["year_fraction"]), r["raw_value"], r["holder"]))
    cards = []
    for rank, cat, y, season, ach in COUNTDOWN:
        r3, rr = e3[(cat, y)], raw[(cat, y)]
        m = D.meta[cat]
        lst = D.seasons[cat][y]
        row = next(s for s in S if s["category"] == cat and s["season"] == str(y) and s["E3_main_env_z"])
        pts = [{"season": s, "value": round(v, 4), "holder": names.get(pid, "cut-off") if pid != "cutoff" else "list cut-off"}
               for (s, v), pid in zip(rr["pts"], rr["pids"])]
        # trend of everyone else's best in raw units (same method as the score; a visual guide)
        fb = G.fit_baseline(rr["pts"], y, m["shift"], "quad_safe")
        yrs = np.array([p[0] for p in rr["pts"]], float)
        v = np.array([p[1] for p in rr["pts"]], float)
        use_quad = min((yrs < y).sum(), (yrs > y).sum()) >= 5
        deg = 2 if use_quad else 1
        coef = np.polyfit(yrs - y, v, deg)
        curve = [[float(s), float(np.polyval(coef, s - y))] for s in np.linspace(yrs.min(), yrs.max(), 60)]
        line = recl[cat]
        before = [p for p in line if p[0] < y + 0.999 and p[0] < y]
        after = [p for p in line if p[0] >= y + 1]
        card = {"rank": rank, "category": cat, "title": m["title"], "sport": m["sport"], "season": season, "year": y,
                "athlete": r3["name"], "achievement": ach, "value": r3["raw"],
                "x_raw": rr["x"], "z_e3": round(float(row["E3_main_env_z"]), 2),          # the locked ranking's value
                "z_e3_recomputed": round(r3["z"], 4), "z_main_no_env": round(float(row["A2_main_z"]), 2),
                "z_original": round(float(row["A0_original_z"]), 2) if row["A0_original_z"] else None,
                "rank_original": row["A0_original_rank_page"], "env_adjusted": cat in ctx.env_cats,
                "gamma": round(gam.get(cat, 0), 2) if cat in ctx.env_cats else None,
                "runner_up": {"name": lst[1].name, "value": lst[1].raw} if len(lst) > 1 else None,
                "window": pts, "trend_raw_units": curve, "trend_form": "quadratic" if use_quad else "straight line",
                "baseline_raw": round(fb[0], 4), "scale_raw": round(fb[2], 4),
                "previous_record": {"value": before[-1][1], "holder": before[-1][2], "year": int(before[-1][0])} if before else None,
                "next_record": {"value": after[0][1], "holder": after[0][2], "year": int(after[0][0])} if after else None,
                "games_per_team": lst[0].games if m["scaled"] else None,
                "one_sided": r3["edge"]}
        # the ruler in the season's own units: the model's comparison values times (league rate in season y)^gamma,
        # i.e. every other season "converted into this season's money". Same fit, same z (a constant rescaling).
        kconv = ctx.env[cat][y] ** ctx.gamma.get(cat, 1.0) if cat in ctx.env_cats else 1.0
        eq_pts = [(s_, v_ * kconv) for s_, v_ in r3["pts"]]
        fbe = G.fit_baseline(eq_pts, y, m["shift"], "quad_safe")
        ye = np.array([q[0] for q in eq_pts], float)
        ve = np.array([q[1] for q in eq_pts], float)
        qe = min((ye < y).sum(), (ye > y).sum()) >= 5
        ce = np.polyfit(ye - y, ve, 2 if qe else 1)
        assert abs(np.polyval(ce, 0) - fbe[0]) < 1e-6 * max(1, abs(fbe[0]))
        z_eq = (r3["x"] * kconv - fbe[0]) / fbe[2]
        assert abs(z_eq - r3["z"]) < 1e-9 and abs(r3["x"] * kconv - float(r3["raw"])) < 1e-6 * float(r3["raw"]), cat
        card["ruler"] = {"window": [[int(s_), round(v_, 4)] for (s_, v_), pid in zip(eq_pts, r3["pids"]) if pid != "cutoff"],
                         "x": float(r3["raw"]), "base": round(fbe[0], 4), "step": round(fbe[2], 4), "z": round(z_eq, 4),
                         "trend": [[float(s_), float(np.polyval(ce, s_ - y))] for s_ in np.linspace(ye.min(), ye.max(), 60)],
                         "converted": cat in ctx.env_cats, "factor_note": "comparison values x (league rate in season)^gamma"}
        if cat in E:
            card["league_env"] = [[s, E[cat][s]] for s in range(y - 15, y + 16) if s in E[cat]]
            card["league_env_at_season"] = E[cat].get(y)
            # the chart for team sports: each value as a share of an average team's total (value / league rate, and
            # for totals / the performer's schedule) - the score's adjustment with elasticity 1
            G_y = lst[0].games if m["scaled"] else 1.0
            sh = [(s_, v_ / E[cat][s_] / G_y) for s_, v_ in rr["pts"] if s_ in E[cat]]
            xs = rr["x"] / E[cat][y] / G_y
            yrs2 = np.array([p_[0] for p_ in sh], float)
            v2 = np.array([p_[1] for p_ in sh], float)
            use_q2 = min((yrs2 < y).sum(), (yrs2 > y).sum()) >= 5
            c2 = np.polyfit(yrs2 - y, v2, 2 if use_q2 else 1)
            card["share"] = {"window": [[s_, round(v_, 5)] for s_, v_ in sh], "x": xs,
                             "trend": [[float(s_), float(np.polyval(c2, s_ - y))] for s_ in np.linspace(yrs2.min(), yrs2.max(), 60)],
                             "label": "share of an average team's " + ("season total" if m["scaled"] else "per-game total"),
                             "per_game": not m["scaled"]}
        cards.append(card)
    extras = {}
    # Wilt's minutes; league scoring; Gretzky assists vs the next-best point total; Ruth vs the average team
    for r in read(os.path.join(FU, "data", "context", "nba_players.csv")):
        if r["player_id"] == "chambwi01" and r["season"] == "1962":
            extras["wilt_minutes_per_game"] = float(r["mp"])
            extras["wilt_games"] = int(float(r["g"]))
    nba = {int(r["season"]): r for r in read(os.path.join(FU, "data", "context", "nba_league.csv"))}
    extras["nba_points_per_team_game_1962"] = float(nba[1962]["pts_per_team_g"])
    extras["nba_points_per_team_game_2026"] = float(nba[2026]["pts_per_team_g"])
    pts86 = D.seasons["nhl_points"][1986]
    extras["nhl_1986_points_leaders"] = [(e.name, e.raw) for e in pts86[:3]]
    mlb = {int(r["season"]): r for r in read(os.path.join(FU, "data", "context", "mlb_league.csv"))}
    extras["mlb_1920_hr_per_team_game"] = float(mlb[1920]["hr_per_team_g"])
    extras["mlb_1920_games_per_team"] = float(mlb[1920]["games_per_team"])
    extras["mlb_1920_avg_team_hr"] = round(float(mlb[1920]["hr_per_team_g"]) * float(mlb[1920]["games_per_team"]), 1)
    extras["mlb_1924_so_per_team_game"] = float(mlb[1924]["so_pitch_per_team_g"])
    extras["mlb_2004_h_per_team_game"] = float(mlb[2004]["h_per_team_g"])
    extras["rodman_1992_z"] = round(float(next(s_["E3_main_env_z"] for s_ in S if s_["category"] == "nba_rpg" and s_["season"] == "1992"
                                              and s_["athlete"] == "Dennis Rodman")), 2)
    # longest streaks since 1941 (Retrosheet streak table, names via the season lists)
    streak = []
    for yy, lst in D.seasons["mlb_streak"].items():
        if yy > 1941 and lst:
            streak.append((float(lst[0].raw), yy, lst[0].name))
    extras["longest_streak_since_1941"] = max(streak)
    # women's throws: the 'today' return periods, the records, counts of marks by decade, and how they rank
    allsp = json.load(open(os.path.join(REC, "tables", "allsports.json")))
    cats = {c["id"]: c for c in allsp["categories"]}
    W = {}
    for cid, thr in (("ath_W_SP", 22.0), ("ath_W_DT", 72.0)):
        c = cats[cid]
        dec = defaultdict(set)
        leads = {}
        for r in csv.DictReader(open(os.path.join(REC, "cache", "alltime_parsed.csv"), encoding="utf-8")):
            if r["list"] != "main" or f"ath_{r['sex']}_{r['event']}" != cid:
                continue
            yy, mk = int(r["date"][:4]), float(r["mark"])
            if yy > 2026:
                continue
            if mk >= thr:
                dec[yy // 10 * 10].add((r["athlete"], r["date"], mk))
            if yy not in leads or mk > leads[yy][0]:
                leads[yy] = (mk, r["athlete"])
        rline = [[round(x, 3), float(v_), h] for x, v_, h in recl[cid] if x >= 1960]
        rs = [s for s in S if s["category"] == cid and s["E3_main_env_z"]]
        best = max(rs, key=lambda s: float(s["E3_main_env_z"]))
        rec_season = [s for s in rs if s["athlete"] == c["record_by"].replace("Griffith Joyner", "Griffith Joyner")]
        rec_best = max(rec_season, key=lambda s: float(s["E3_main_env_z"])) if rec_season else None
        W[cid] = {"record": c["record"], "record_by": c["record_by"], "record_season": c["record_season"],
                  "today_once_in_seasons": round(c["today_T"]), "threshold": thr,
                  "marks_at_or_beyond_threshold_by_decade": {str(d): len(v) for d, v in sorted(dec.items())},
                  "athletes_at_or_beyond_threshold_by_decade": {str(d): len({a for a, _, _ in v}) for d, v in sorted(dec.items())},
                  "season_leads": [[yy, leads[yy][0]] for yy in sorted(leads) if yy >= 1965],
                  "record_line": rline,
                  "last_year_at_or_beyond_threshold": max((int(d_) for d_, _a, _m in [(dd[1][:4], 0, 0) for v_ in dec.values() for dd in v_]), default=None),
                  "best_since_2000": max(((m_, yy) for yy, (m_, _) in leads.items() if yy >= 2000)),
                  "best_on_list": {"athlete": best["athlete"], "season": best["season"], "z": round(float(best["E3_main_env_z"]), 2),
                                   "rank": best["E3_main_env_rank_page"] or best["E3_main_env_rank_all"]},
                  "record_holder_best_season": ({"season": rec_best["season"], "z": round(float(rec_best["E3_main_env_z"]), 2)}
                                                if rec_best else None)}
    best_w = min((s for s in S if s["category"].startswith("ath_W_") and s["category"].split("_")[2] in ("SP", "DT", "HT", "JT")
                  and s["E3_main_env_rank_page"]), key=lambda s: int(s["E3_main_env_rank_page"]))
    W["best_womens_throw_on_list"] = {"athlete": best_w["athlete"], "season": best_w["season"], "category": best_w["category"],
                                      "rank": int(best_w["E3_main_env_rank_page"]), "z": round(float(best_w["E3_main_env_z"]), 2)}
    # robustness
    ens = read(os.path.join(FU, "outputs", "spec_ensemble.csv"))
    rob = {"in_all_16": [r["athlete"] for r in ens if r["specs_in_top10"] == r["specs"]],
           "n_specs": int(ens[0]["specs"])}
    def page_rank(name, year, col):
        rows = [s for s in S if s["athlete"] == name and s["season"] == str(year) and s[col + "_rank_page"]]
        return int(rows[0][col + "_rank_page"]) if rows else None
    rob["all_seasons_rank"] = {n: page_rank(n, y, "B1_allseasons") for n, y in
                               (("Wayne Gretzky", 1986), ("Wilt Chamberlain", 1962), ("Joe DiMaggio", 1941), ("Bob Beamon", 1968))}
    rob["no_env_rank"] = {n: page_rank(n, y, "A2_main") for n, y in (("Dan Marino", 1984), ("Katie Ledecky", 2016))}
    mab = read(os.path.join(FU, "outputs", "bootstrap_model_averaged.csv"))
    rob["model_averaged_top10_frequency"] = {r["athlete"]: float(r["top10_frequency_over_specs_and_refits"]) for r in mab[:4]}
    rob["n_categories"] = len(D.meta)
    rob["n_scored"] = len(G.score(D, G.Spec(ties="all", trend="quad_safe", env="power"), ctx=ctx))
    # the method example: Ledecky 2016 (outside the top ten; no environment adjustment)
    rl = raw[("swim_W_800_fr", 2016)]
    yrs = np.array([p[0] for p in rl["pts"]], float)
    v = np.array([p[1] for p in rl["pts"]], float)
    coef = np.polyfit(yrs - 2016, v, 2)
    lsec = lambda sp: 800 / sp
    method = {"athlete": "Katie Ledecky", "season": 2016, "event": "Women's 800 m freestyle", "value": "8:04.79",
              "z": round(rl["z"], 2), "rank_e3": page_rank("Katie Ledecky", 2016, "E3_main_env"),
              "window": [{"season": int(s), "time": round(lsec(val), 2), "holder": names.get(pid, "")}
                         for (s, val), pid in zip(rl["pts"], rl["pids"])],
              "trend_time": [[float(s), round(lsec(float(np.polyval(coef, s - 2016))), 3)] for s in np.linspace(yrs.min(), yrs.max(), 60)],
              "baseline_time": round(lsec(rl["base"]), 2), "x_time": round(lsec(rl["x"]), 2),
              "scale_speed": rl["scale"], "base_speed": rl["base"], "x_speed": rl["x"]}
    # the scale: every season leader's score (all 4,940 scored records, ties counted)
    zs = np.array([r["z"] for r in e3_all])
    bands = [(-99, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 99)]
    top_names = {(c["category"], c["year"]) for c in cards}
    scale = {"n": int(len(zs)), "median": round(float(np.median(zs)), 2),
             "bands": [{"lo": lo, "hi": hi, "n": int(((zs >= lo) & (zs < hi)).sum())} for lo, hi in bands],
             "below_1": int((zs < 1).sum()), "at_least_4": int((zs >= 4).sum()), "at_least_5": int((zs >= 5).sum()),
             "hist_edges": [round(e, 2) for e in np.arange(-4.0, 4.01, 0.5)],
             "hist_counts": [int(((zs >= lo) & (zs < lo + 0.5)).sum()) for lo in np.arange(-4.0, 4.0, 0.5)],
             "below_hist": int((zs < -4).sum()),
             "tail": sorted([{"name": r["name"], "year": r["year"], "cat": r["cat"], "z": round(r["z"], 2),
                              "top10": (r["cat"], r["year"]) in top_names and r["name"] == next(c["athlete"] for c in cards if (c["category"], c["year"]) == (r["cat"], r["year"]))}
                             for r in e3_all if r["z"] >= 4], key=lambda d: -d["z"])}
    method["step_seconds"] = round((method["baseline_time"] - method["x_time"]) / rl["z"], 2)
    method["gap_seconds"] = round(method["baseline_time"] - method["x_time"], 2)
    out = {"model": "E3_main_env (locked)", "cards": cards, "scale": scale, "extras": extras, "womens_throws": W, "robustness": rob,
           "method_example": method}
    json.dump(out, open(os.path.join(HERE, "facts.json"), "w"), indent=1, default=str, ensure_ascii=False)
    for c in cards:
        print(c["rank"], c["athlete"], c["season"], c["value"], "E3", c["z_e3"], "noenv", c["z_main_no_env"], "orig", c["z_original"],
              "| runner-up", c["runner_up"], "| prev", c["previous_record"], "| next", c["next_record"], "| one-sided", c["one_sided"])
    print(json.dumps(extras, default=str))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "season_leads"}) for k, v in W.items()}, default=str, ensure_ascii=False))
    print(json.dumps(rob))
    print({k: v for k, v in method.items() if k not in ("window", "trend_time")})


if __name__ == "__main__":
    main()
