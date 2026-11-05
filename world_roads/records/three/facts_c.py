"""Every number film C shows (the five biggest season-long outliers in team sports), into facts_c.json.

    python facts_c.py

The model is the season film's locked one (E3_main_env) on the four team sports, plus the rule all three companion films
use: a season counts only if it is at least one step clear of every other season on its graph. The film's five are the
seasons in the top ten of at least 12 of the 16 versions of the method (../outliers/review_C.py, REVIEW_C.md). Each
card's ruler, window and score are the season film's own (../outliers/facts.json), which the review reproduces.
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, REC)
sys.path.insert(0, os.path.join(REC, "outliers"))
import review_C as RC                                       # noqa: E402  (setup, ranked, the 16 versions)

G, X = RC.G, RC.X
SEASON_FILM = json.load(open(os.path.join(REC, "outliers", "facts.json"), encoding="utf-8"))
OLD = {c["athlete"]: c for c in SEASON_FILM["cards"]}
PHOTO = {"Ichiro Suzuki": "ichiro", "Babe Ruth": "ruth", "Dennis Rodman": "rodman", "Stephen Curry": "curry",
         "Dazzy Vance": "vance", "Wayne Gretzky": "gretzky", "Joe DiMaggio": "dimaggio", "Wilt Chamberlain": "chamberlain",
         "Bob Feller": "feller", "Maury Wills": "wills", "Ted Williams": "williams"}
PALETTE = {"Bob Feller": "indians", "Maury Wills": "dodgers", "Ted Williams": "redsox"}
UNIT = {"mlb_so": "strikeouts", "mlb_hr": "home runs", "mlb_sb": "stolen bases", "mlb_ba": "batting average",
        "mlb_h": "hits", "nba_rpg": "rebounds a game", "nba_ppg": "points a game", "nhl_assists": "assists",
        "mlb_streak": "game hitting streak", "nba_3pm": "three-pointers"}
# why each of the rest of the top ten by score missed the film, in a line (REVIEW_C.md has the per-version detail)
WHY = {"Dazzy Vance": "It rests on the league adjustment. Without it, the bigger totals of 1904–1915 win: Rube "
                      "Waddell's 349, Walter Johnson's 313.",
       "Babe Ruth": "It rests on the league adjustment. Without it, the home-run boom he started catches him: Hack "
                    "Wilson's 56 in 1930, Jimmie Foxx's 58 in 1932.",
       "Bob Feller": "Always clear of its own era, but short comparison windows rank it 12th to 19th, and the widest "
                     "unadjusted one reaches Sandy Koufax's 382 in 1965.",
       "Maury Wills": "Without the league adjustment, the steals boom that followed him wins: Lou Brock's 118 in 1974, "
                      "Rickey Henderson's 130 in 1982.",
       "Ted Williams": "Without the adjustment, Rogers Hornsby's .424 and Bill Terry's .401 come within a step. With it, "
                       "only the longer windows rank it in the ten."}


def achievement(r):
    v = RC.val(r["raw"])
    if r["cat"] == "mlb_streak":
        return f"{v}-game hitting streak"
    if r["cat"] == "mlb_ba":
        return f"{v} batting average"
    return f"{v} {UNIT.get(r['cat'], RC.G.Data.__name__)}"


def main():
    D, gam, ctx = RC.setup()
    spec = G.Spec(ties="all", env="power", trend="quad_safe")
    sc, top = RC.ranked(D, ctx, spec)
    ens, versions = defaultdict(int), {}
    for env in ("none", "power"):
        for wdw in (7, 10, 15, 20):
            for tr in ("ols", "quad_safe"):
                versions[(env, wdw, tr)] = RC.ranked(D, ctx if env == "power" else None,
                                                     G.Spec(ties="all", window=wdw, trend=tr, env=env))
                for r in versions[(env, wdw, tr)][1][:10]:
                    ens[(r["name"], r["sport"])] += 1
    ten = top[:10]
    five = [r for r in ten if ens[(r["name"], r["sport"])] >= RC.AGREE]
    names = [r["name"] for r in five]
    assert names == ["Wilt Chamberlain", "Joe DiMaggio", "Wayne Gretzky", "Dennis Rodman", "Ichiro Suzuki"], names

    def other(r):
        s_, who = r["beaten_by"]
        raw = next((e.raw for e in D.seasons[r["cat"]][s_] if e.name == who), None)
        return {"name": who, "year": s_, "season": RC.season(r["sport"], s_), "value": RC.val(raw),
                "converted": round(RC.to_own_units(dict(r["pts"])[s_], r, ctx), 2)}

    cards = []
    for rank, r in enumerate(five, 1):
        c = dict(OLD[r["name"]])
        assert c["category"] == r["cat"] and c["year"] == r["year"], (c["category"], r["cat"])
        assert abs(c["z_e3"] - r["z"]) < 0.006, (r["name"], c["z_e3"], r["z"])   # the locked score, reproduced
        c.update(old_rank=c["rank"], rank=rank, palette=c["rank"], photo=PHOTO[r["name"]],
                 specs_in_top10=ens[(r["name"], r["sport"])], clear_steps=round(float(r["clear"]), 2),
                 best_other=other(r))
        import scenes as SC
        c["scene"] = SC.SCENE[c["old_rank"]]
        cards.append(c)
    considered = []
    for i, r in enumerate(ten, 1):
        if r["name"] in names:
            continue
        considered.append({"rank_by_score": i, "athlete": r["name"], "category": r["cat"], "year": r["year"],
                           "season": RC.season(r["sport"], r["year"]), "achievement": achievement(r),
                           "z": round(float(r["z"]), 2), "specs_in_top10": ens[(r["name"], r["sport"])],
                           "photo": PHOTO[r["name"]], "palette": PALETTE.get(r["name"]) or OLD[r["name"]]["rank"],
                           "why": WHY[r["name"]]})
    zcard = {c["athlete"]: c["z_e3"] for c in cards}               # the five show their locked (card) score
    board = [{"rank_by_score": i, "athlete": r["name"], "achievement": achievement(r),
              "season": RC.season(r["sport"], r["year"]), "z": zcard.get(r["name"], round(float(r["z"]), 2)),
              "specs_in_top10": ens[(r["name"], r["sport"])], "in_film": r["name"] in names} for i, r in enumerate(ten, 1)]
    cu = next(r for r in sc if r["cat"] == "nba_3pm" and r["year"] == 2016 and "Curry" in r["name"])
    zs = sorted((float(r["z"]) for r in sc), reverse=True)
    edges = [-4.0 + 0.5 * i for i in range(17)]
    counts = [sum(lo <= z < lo + 0.5 for z in zs) for lo in edges[:-1]]
    card_keys = {(c["category"], c["year"], c["athlete"]) for c in cards}
    tail = [{"name": r["name"], "year": r["year"], "cat": r["cat"], "z": round(float(r["z"]), 2),
             "top10": (r["cat"], r["year"], r["name"]) in card_keys} for r in sorted(sc, key=lambda r: -r["z"]) if r["z"] >= 4]
    F = {"model": "E3_main_env, team sports, clear of the field by one step; the five in the top ten of >= 12 of 16 versions",
         "n_cards": 5, "hook_order": [3, 1, 5, 2, 4],
         "cards": cards, "considered": considered, "board": board,
         "scale": {"n": len(sc), "n_categories": len(D.meta), "median": round(float(zs[len(zs) // 2]), 2),
                   "at_least_4": sum(z >= 4 for z in zs), "at_least_5": sum(z >= 5 for z in zs),
                   "clear": sum(r["clear"] >= RC.CLEAR_STEPS for r in sc), "hist_edges": edges, "hist_counts": counts,
                   "below_hist": sum(z < -4 for z in zs), "tail": tail,
                   "first_season": min(r["year"] for r in sc), "last_season": max(r["year"] for r in sc)},
         "curry": {"z": round(float(cu["z"]), 2), "clear_steps": round(float(cu["clear"]), 2), "best_other": other(cu)},
         "robustness": {"n_specs": len(versions), "agree": RC.AGREE,
                        "in_all_16": [r["name"] for r in five if ens[(r["name"], r["sport"])] == len(versions)]},
         "extras": SEASON_FILM["extras"]}
    json.dump(F, open(os.path.join(HERE, "facts_c.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("facts_c.json:", [(c["rank"], c["athlete"], c["z_e3"], c["specs_in_top10"], c["clear_steps"]) for c in cards])
    print("considered:", [(c["athlete"], c["z"], c["specs_in_top10"]) for c in considered])
    print("scale:", {k: v for k, v in F["scale"].items() if k not in ("tail", "hist_edges", "hist_counts")})
    print("curry:", F["curry"])


if __name__ == "__main__":
    main()
