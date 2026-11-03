"""Score the single-game data with the locked greatest-seasons model (E3_main_env): the best of everyone else in
the 15 seasons either side, a curved trend (quadratic where there are 5+ seasons each side), team-sport values
divided by the league's per-team-game rate^gamma (gamma estimated and partially pooled from these single-game
lists, as run_all.pooled_gamma does for the season lists), the athlete's own other seasons left out, ties counted.

    python score.py [all|corrected|big]   # -> outputs/sg_ranking_<variant>.csv, sg_spec_ensemble_<variant>.csv,
                                          #    sg_by_category.csv, sg_gamma.csv, sg_scale_<variant>.json

Variants (small whole-number stats - home runs, goals - make one extra unit look like many steps, because most
seasons' best games tie at the same number):
  all        every category, as scored
  corrected  a half-unit continuity correction for team-sport counts: z = (x - 0.5/k - base) / sqrt(s^2 + (1/k)^2/12)
  big        only team-sport stats whose season-best game is usually 15 or more (points, rebounds, assists,
             pitcher strikeouts, NFL yards); athletics and swimming always kept

Athletics and swimming are the season lists of followup/data/v2 unchanged: a season best is one performance.
"""
import csv, json, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REC, "followup", "analysis"))
import gsmodel as G                                            # noqa: E402
import context as X                                            # noqa: E402
import run_all as RA                                           # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
LEAGUE_COL = {"sg_nba_pts": ("nba", "pts_per_team_g"), "sg_nba_reb": ("nba", "trb_per_team_g"),
              "sg_nba_ast": ("nba", "ast_per_team_g"), "sg_nba_3pm": ("nba", "fg3_per_team_g"),
              "sg_nhl_goals": ("nhl", "goals_per_team_g"), "sg_nhl_assists": ("nhl", "assists_per_team_g"),
              "sg_nhl_points": ("nhl", "points_per_team_g"), "sg_mlb_h": ("mlb", "h_per_team_g"),
              "sg_mlb_hr": ("mlb", "hr_per_team_g"), "sg_mlb_rbi": ("mlb", "rbi_per_team_g"),
              "sg_mlb_sb": ("mlb", "sb_per_team_g"), "sg_mlb_so": ("mlb", "so_pitch_per_team_g"),
              "sg_nfl_pass_yds": ("nfl", "pass_yds_per_team_g"), "sg_nfl_pass_td": ("nfl", "pass_td_per_team_g"),
              "sg_nfl_rush_yds": ("nfl", "rush_yds_per_team_g"), "sg_nfl_rec_yds": ("nfl", "rec_yds_per_team_g"),
              "sg_nfl_int": ("nfl", "int_per_team_g"), "sg_nfl_sacks": ("nfl", "sacks_per_team_g")}


class SGData(G.Data):
    """the model's Data, with single-game team lists and the v2 athletics and swimming lists"""

    def __init__(self):
        v2 = G.Data("v2")
        self.version = "sg"
        self.cats = {c: m for c, m in v2.cats.items() if m["sport"] in ("Athletics", "Swimming")}
        self.seasons = defaultdict(dict)
        for c in self.cats:
            self.seasons[c] = v2.seasons[c]
        for r in csv.DictReader(open(os.path.join(DATA, "sg_categories.csv"), encoding="utf-8")):
            self.cats[r["category"]] = r
        by = defaultdict(list)
        self.detail = {}
        for r in csv.DictReader(open(os.path.join(DATA, "sg_season_lists.csv"), encoding="utf-8")):
            by[(r["category"], int(r["season"]))].append(r)
            self.detail[(r["category"], int(r["season"]), r["athlete_id"])] = (r["date"], r["detail"])
        for (c, y), rs in by.items():
            rs.sort(key=lambda r: int(r["rank"]))
            self.seasons[c][y] = [G.Entry(r["athlete_id"], r["name_display"], float(r["performance"]), r["raw_value"],
                                          math.nan) for r in rs]
        self.meta = {}
        for c, m in self.cats.items():
            self.meta[c] = {"sport": m["sport"], "title": m["title"],
                            "short": {int(t) for t in m["short_seasons_left_out"].split()},
                            "shift": {int(t) for t in m["level_shift_seasons"].split()},
                            "scaled": m["is_total_scaled_by_schedule"] == "True",
                            "cut": float(m["cutoff_performance"]) if m["cutoff_performance"] else None,
                            "first": int(m["first_season"]), "start": int(m["performances_count_from"])}


def setup():
    D = SGData()
    X.ENV_MAP.clear()
    X.ENV_MAP.update({c: (lg, col, "ratio") for c, (lg, col) in LEAGUE_COL.items() if c in D.meta})
    gam, grows = RA.pooled_gamma(D)
    ctx = X.make(D, env="power", gamma=gam)
    return D, gam, grows, ctx


def variant(D, ctx, recs, kind):
    """apply a variant to scored records (see the docstring)"""
    if kind == "all":
        return recs
    if kind == "corrected":
        out = []
        for r in recs:
            c = r["cat"]
            if c.startswith("sg_"):
                k = ctx.env[c][r["year"]] ** ctx.gamma.get(c, 1.0) if c in ctx.env_cats else 1.0
                r = dict(r, z=(r["x"] - 0.5 / k - r["base"]) / math.sqrt(r["scale"] ** 2 + (1.0 / k) ** 2 / 12.0))
            out.append(r)
        return out
    lvl = {c: float(np.median([lst[0].perf for lst in D.seasons[c].values() if lst])) for c in D.meta if c.startswith("sg_")}
    return [r for r in recs if not r["cat"].startswith("sg_") or lvl[r["cat"]] >= 15]


def main(kind="big"):
    os.makedirs(OUT, exist_ok=True)
    D, gam, grows, ctx = setup()
    spec = G.Spec(ties="all", trend="quad_safe", env="power")
    recs = variant(D, ctx, G.score(D, spec, ctx=ctx), kind)
    page = G.by_athlete(G.athlete_seasons(recs))
    with open(os.path.join(OUT, "sg_gamma.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0]))
        w.writeheader()
        w.writerows(grows)
    with open(os.path.join(OUT, f"sg_ranking_{kind}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "athlete", "sport", "category", "title", "season", "value", "z", "date", "detail",
                    "one_sided_window"])
        for i, r in enumerate(page[:100]):
            d = D.detail.get((r["cat"], r["year"], r["id"]), ("", ""))
            w.writerow([i + 1, r["name"], r["sport"], r["cat"], r["title"], r["year"], r["raw"], round(r["z"], 3), d[0],
                        d[1], r["edge"]])
    best_cat = {}
    for r in recs:
        if r["cat"] not in best_cat or r["z"] > best_cat[r["cat"]]["z"]:
            best_cat[r["cat"]] = r
    with open(os.path.join(OUT, "sg_by_category.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "title", "athlete", "season", "value", "z", "gamma"])
        for c, r in sorted(best_cat.items(), key=lambda kv: -kv[1]["z"]):
            w.writerow([c, r["title"], r["name"], r["year"], r["raw"], round(r["z"], 3), round(gam.get(c, 0), 3) if c in gam else ""])
    zs = np.array([r["z"] for r in recs])
    json.dump({"n": int(len(zs)), "median": round(float(np.median(zs)), 2), "at_least_4": int((zs >= 4).sum()),
               "at_least_5": int((zs >= 5).sum()), "below_1": int((zs < 1).sum()), "n_categories": len(D.meta)},
              open(os.path.join(OUT, f"sg_scale_{kind}.json"), "w"), indent=1)
    # the pre-declared 16-specification ensemble: windows x trends x league adjustment
    ens = defaultdict(int)
    for wdw in (7, 10, 15, 20):
        for tr in ("ols", "quad_safe"):
            for env in ("none", "power"):
                pg = G.by_athlete(G.athlete_seasons(variant(D, ctx, G.score(D, G.Spec(ties="all", window=wdw, trend=tr, env=env),
                                                                            ctx=ctx if env == "power" else None), kind)))
                for r in pg[:10]:
                    ens[(r["name"], r["sport"])] += 1
    with open(os.path.join(OUT, f"sg_spec_ensemble_{kind}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["athlete", "sport", "specs_in_top10", "specs"])
        for (n, s), k in sorted(ens.items(), key=lambda kv: -kv[1]):
            w.writerow([n, s, k, 16])
    print(f"{len(recs)} scored records in {len(D.meta)} categories")
    for i, r in enumerate(page[:25]):
        print(f"{i + 1:3d} {r['z']:5.2f}  {r['name']:26s} {r['year']}  {r['title']:34s} {r['raw']}  "
              f"[{ens.get((r['name'], r['sport']), 0)}/16]" + ("  one-sided" if r["edge"] else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "big")
