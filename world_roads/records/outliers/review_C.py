"""The sniff-test review for video C (season-long individual performances in team sports): writes REVIEW_C.md.

The locked season model of "The 10 Biggest Statistical Outliers" (E3_main_env: the best of everyone else in the 15
seasons either side, a quadratic trend, league adjustment with the locked pooled elasticities, the athlete's own
seasons left out, ties counted) on the 22 team-sport categories only, plus the rule agreed for all three films: a
season makes the list only if it is at least one step clear of every other season on its graph (CLEAR_STEPS).

Inputs: the original season lists (team sports), with the follow-up's identity fixes (aliases merged, same-name
players told apart, ID column of followup/outputs/sources_and_corrections.csv); the league tables in
followup/data/context (Basketball-, Hockey-, Baseball-Reference league pages; NFL from Pro Football Archives and
nflverse); the locked elasticities (followup/outputs/gamma_estimates.csv). Every score is checked against the locked
ranking (rankings_sensitivity.csv, E3_main_env_z) and any difference is listed.

    python review_C.py      # from world_roads/records/outliers/
"""
import csv, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
FU = os.path.join(REC, "followup")
sys.path.insert(0, os.path.join(FU, "analysis"))
sys.path.insert(0, os.path.join(REC, "singlegame"))
import gsmodel as G                                           # noqa: E402
import context as X                                           # noqa: E402
import score as SG                                            # noqa: E402  (margins: the clear-of-field check)

TEAM = ("NBA", "NHL", "MLB", "NFL")
CLEAR_STEPS = SG.CLEAR_STEPS
AGREE = 12            # the film keeps the seasons in the top ten of at least three-quarters of the 16 versions
SEASON_FMT = {"NBA": True, "NHL": True}                       # seasons that span two years are shown as 1985-86


def same_person(key, year):
    """the follow-up's identity fixes (sources_and_corrections.csv, codes ALIAS and ID)"""
    alias = {"henry aaron": "hank aaron", "gavy cravath": "gavvy cravath", "shoeless joe jackson": "joe jackson",
             "j d martinez": "jd martinez", "cal ripken jr": "cal ripken", "b j surhoff": "bj surhoff"}
    key = alias.get(key, key)
    split = {"vladimir guerrero": 2019, "bobby witt": 2022, "eric young": 2009, "mike dunleavy": 2003,
             "syl apps": 1960}                                # the son's career starts here
    for k, y0 in split.items():
        if key in (k, k + " jr") and year >= y0:
            return k + " jr"
        if key == k + " jr" and year < y0:
            return k
    return key


class TeamSeasons(G.Data):
    def __init__(self):
        super().__init__("original")
        for c in list(self.cats):
            if self.cats[c]["sport"] not in TEAM:
                del self.cats[c]
                self.seasons.pop(c, None)
                self.meta.pop(c, None)
                continue
            self.meta[c]["short"] |= G.DEFINITION_BREAKS.get(c, set())
            for y, lst in self.seasons[c].items():
                for e in lst:
                    e.id = same_person(e.id, y)


def setup():
    D = TeamSeasons()
    g = list(csv.DictReader(open(os.path.join(FU, "outputs", "gamma_estimates.csv"), encoding="utf-8")))
    gam = {r["category"]: float(r["gamma_pooled"]) for r in g if r.get("gamma_pooled")}
    return D, gam, X.make(D, env="power", gamma=gam)


def to_own_units(v, r, ctx):
    """a comparison-scale value back into the athlete's own season's units (what a card shows)"""
    e = ctx.env.get(r["cat"], {}).get(r["year"])
    if e is None:
        return v
    g = ctx.gamma.get(r["cat"], 1.0)
    return e ** g / v if ctx.kind.get(r["cat"]) == "era" else v * e ** g


def val(raw):
    """a value as people say it: 262, .406, 18.3"""
    v = float(raw)
    return str(int(v)) if v.is_integer() else (f"{v:.3f}".lstrip("0") if v < 1 else f"{v:g}")


def season(sport, y):
    return f"{y - 1}-{str(y)[2:]}" if SEASON_FMT.get(sport) else str(y)


def ranked(D, ctx, spec):
    sc = SG.margins(D, G.score(D, spec, ctx=ctx, details=True))
    return sc, G.by_athlete(G.athlete_seasons([r for r in sc if r["clear"] >= CLEAR_STEPS]))


FAMOUS = [  # (category, season, name fragment, what it is)
    ("nba_ppg", 1962, "Chamberlain", "Wilt's 50.4 points a game"), ("nba_3pm", 2016, "Curry", "Curry's 402 threes"),
    ("nba_rpg", 1993, "Rodman", "Rodman's 18.3 rebounds"), ("nba_ppg", 1987, "Jordan", "Jordan's 37.1 points a game"),
    ("nba_apg", 1990, "Stockton", "Stockton's 14.5 assists"), ("nba_rpg", 1961, "Chamberlain", "Wilt's 27.2 rebounds"),
    ("nhl_assists", 1986, "Gretzky", "Gretzky's 163 assists"), ("nhl_goals", 1982, "Gretzky", "Gretzky's 92 goals"),
    ("nhl_assists", 1971, "Orr", "Orr's 102 assists"), ("nhl_goals", 1971, "Esposito", "Esposito's 76 goals"),
    ("nhl_points", 1989, "Lemieux", "Lemieux's 199 points"), ("nhl_points", 1953, "Howe", "Howe's 95 points"),
    ("nhl_goals", 1925, "Dye", "Babe Dye's 38 goals"),
    ("mlb_streak", 1941, "DiMaggio", "DiMaggio's 56-game streak"), ("mlb_hr", 1920, "Ruth", "Ruth's 54 home runs"),
    ("mlb_hr", 2001, "Bonds", "Bonds's 73 home runs"), ("mlb_h", 2004, "Suzuki", "Ichiro's 262 hits"),
    ("mlb_so", 1924, "Vance", "Vance's 262 strikeouts"), ("mlb_so", 1946, "Feller", "Feller's 348 strikeouts"),
    ("mlb_so", 1973, "Ryan", "Ryan's 383 strikeouts"), ("mlb_sb", 1962, "Wills", "Wills's 104 steals"),
    ("mlb_sb", 1982, "Henderson", "Henderson's 130 steals"), ("mlb_ba", 1941, "Williams", "Ted Williams's .406"),
    ("mlb_era", 1968, "Gibson", "Gibson's 1.12 ERA"), ("mlb_rbi", 1930, "Wilson", "Hack Wilson's 191 RBI"),
    ("mlb_w", 1968, "McLain", "McLain's 31 wins"),
    ("nfl_pass_td", 1984, "Marino", "Marino's 48 TD passes"), ("nfl_rush_yds", 1963, "Brown", "Jim Brown's 1,863 yards"),
    ("nfl_rush_yds", 1984, "Dickerson", "Dickerson's 2,105 yards"), ("nfl_rec_yds", 1942, "Hutson", "Hutson's 1,211 yards"),
    ("nfl_pass_yds", 2013, "Manning", "Peyton Manning's 5,477 yards"), ("nfl_int", 1952, "Lane", "Night Train Lane's 14 interceptions"),
]


NOTES = {  # the plain-English reason, checked against the per-version facts that follow it in the review
    "Dazzy Vance": "It rests on the league adjustment: strikeouts were scarce in 1924, but without that adjustment "
                   "the big raw totals of 1904-1915 (Waddell 349, Johnson 313) win, and a 20-season window reaches "
                   "Waddell's 1904 even with it.",
    "Babe Ruth": "It rests on the league adjustment: without it, the home-run boom Ruth started catches him "
                 "(Gehrig, Wilson's 56 in 1930, Foxx's 58 in 1932); with it, a 7-season window ranks it 12th or 19th.",
    "Bob Feller": "Always clear of its own era, but with 7- or 10-season windows it ranks 12th to 19th, and the "
                  "widest window without adjustment reaches Koufax's 382 in 1965.",
    "Maury Wills": "Without the league adjustment, the steals boom that followed him wins (Brock's 118 in 1974, "
                   "Henderson's 130 in 1982); with 7-season windows it ranks 11th to 21st.",
    "Ted Williams": "Without the league adjustment, the high averages of the 1920s and 30s (Hornsby's .424, "
                    "Terry's .401) come within a step or ahead; with it, only the 15- and 20-season windows put it in "
                    "the ten.",
}


def why_out(D, versions, r):
    """the versions where this season misses the top ten, grouped by what kept it out"""
    key, beat, below = (r["cat"], r["year"], r["id"]), defaultdict(list), []
    for v, (sc, pg) in versions.items():
        if any(t["name"] == r["name"] and t["sport"] == r["sport"] for t in pg[:10]):
            continue
        rv = next((t for t in sc if (t["cat"], t["year"], t["id"]) == key), None)
        if rv is None:
            below.append((v, None))
        elif rv["clear"] < CLEAR_STEPS:
            s_, who = rv["beaten_by"]
            raw = next((val(e.raw) for e in D.seasons[r["cat"]][s_] if e.name == who), "")
            beat[(who, raw, s_)].append(v)
        else:
            rk = next((i for i, t in enumerate(pg, 1) if t["name"] == r["name"] and t["sport"] == r["sport"]), None)
            below.append((v, rk))

    def desc(vs):
        adj = sorted({v[0] for v in vs})
        w = sorted({v[1] for v in vs})
        tr = {v[2] for v in vs}
        return (("adjusted" if adj == ["power"] else "unadjusted" if adj == ["none"] else "adjusted or not")
                + ", " + "/".join(map(str, w)) + "-season window" + ("s" if len(w) > 1 else "")
                + ("" if len(tr) == 2 else ", " + ("straight" if "ols" in tr else "curved") + " trend"))
    parts = [f"{who} {raw} ({s_}) within a step or ahead in {len(vs)} ({desc(vs)})" for (who, raw, s_), vs in
             sorted(beat.items(), key=lambda kv: -len(kv[1]))]
    if below:
        ranks = sorted(rk for _, rk in below if rk)
        parts.append(f"below the top ten in {len(below)} ({desc([v for v, _ in below])}"
                     + (f"; ranked {', '.join('#' + str(k) for k in ranks)}" if ranks else "") + ")")
    return "Out: " + "; ".join(parts) if parts else ""


def main():
    D, gam, ctx = setup()
    spec = G.Spec(ties="all", env="power", trend="quad_safe")
    sc, top = ranked(D, ctx, spec)
    allpage = G.by_athlete(G.athlete_seasons(sc))
    lock = {}
    for s in csv.DictReader(open(os.path.join(FU, "outputs", "rankings_sensitivity.csv"), encoding="utf-8")):
        if s["E3_main_env_z"] and s["sport"] in TEAM:
            lock[(s["category"], int(s["season"]), s["athlete"])] = float(s["E3_main_env_z"])
    diffs = sorted(((abs(r["z"] - lock[k]), r, lock[k]) for r in sc
                    if (k := (r["cat"], r["year"], r["name"])) in lock), key=lambda t: -t[0])
    exact = sum(d < 0.005 for d, _, _ in diffs)
    ens = defaultdict(int)
    versions = {}
    for env in ("none", "power"):
        for wdw in (7, 10, 15, 20):
            for tr in ("ols", "quad_safe"):
                versions[(env, wdw, tr)] = ranked(D, ctx if env == "power" else None,
                                                  G.Spec(ties="all", window=wdw, trend=tr, env=env))
                for r in versions[(env, wdw, tr)][1][:10]:
                    ens[(r["name"], r["sport"])] += 1
    rank_all = {(r["cat"], r["year"], r["id"]): i for i, r in enumerate(allpage, 1)}
    old = {(r["cat"], r["year"], r["id"]) for r in allpage[:10]}

    def other(r):
        s, who = r["beaten_by"]
        v = dict(r["pts"])[s]
        lst = D.seasons[r["cat"]][s]
        raw = next((e.raw for e in lst if e.name == who), "")
        return f"{who} {raw} ({season(r['sport'], s)}) = {to_own_units(v, r, ctx):.4g}"

    L = ["# Video C review: the biggest season-long outliers in team sports", "",
         "Model: the season film's locked model (E3_main_env: best of everyone else in the 15 seasons either side, "
         "quadratic trend, league adjustment, the athlete's own seasons left out), team sports only, plus the rule "
         f"used in all three films: a season must be at least {CLEAR_STEPS:g} step clear of every other season on its "
         f"graph. {len(sc):,} season-leading performances in {len(D.meta)} categories; "
         f"{sum(r['clear'] >= CLEAR_STEPS for r in sc):,} clear their field.", "",
         f"Reproduction: {exact} of {len(diffs)} scores match the published model to 0.005 ("
         + ", ".join(f"{sp} {sum(d < 0.005 for d, r, _ in diffs if r['sport'] == sp)} of "
                     f"{sum(r['sport'] == sp for _, r, _ in diffs)}" for sp in TEAM)
         + f"; top 30 of the old list: {sum(abs(r['z'] - lock.get((r['cat'], r['year'], r['name']), 99)) < 0.005 for r in allpage[:30])} "
           "of 30). The published model used the follow-up's deeper season lists (top 40, stable IDs), which can't "
           "be rebuilt without the original download cache; the original top-10 lists with the same identity fixes "
           "stand in for them, so a few seasons whose comparison runs past rank 10 or through a merged name score "
           "differently (all listed at the end).", "",
         "Curve, step and *best other* are in the athlete's own season's units (*best other* is the best season by "
         "anyone else on the same graph, converted to that season's league rate).", "",
         f"## The film's five: in the top ten in at least {AGREE} of the 16 versions", "",
         "Every season below was considered; these are the ones the versions agree on (three-quarters or more), in "
         "score order. "
         + f"{sum(ens.get((r['name'], r['sport']), 0) == 16 for r in top)} of them are in the top ten of all 16.", "",
         "| # | Season | Score | In top 10 of 16 versions |", "|---|---|---|---|"]
    for i, r in enumerate([r for r in top if ens.get((r["name"], r["sport"]), 0) >= AGREE], 1):
        L.append(f"| {i} | {r['name']}, {r['title'].lower()}, {season(r['sport'], r['year'])} ({val(r['raw'])}) | "
                 f"{r['z']:.2f} | {ens.get((r['name'], r['sport']), 0)} |")
    five = [r for r in top if ens.get((r["name"], r["sport"]), 0) >= AGREE]
    L += ["", "## Considered, not in the film: the rest of the top ten by score", "",
          f"Each was in the top ten of fewer than {AGREE} of the 16 versions (8 with the league adjustment, 8 "
          "without; windows of 7, 10, 15 and 20 seasons; straight or curved trend). What kept it out in the others:", ""]
    for i, r in enumerate(top[:10], 1):
        if r in five:
            continue
        n = ens.get((r["name"], r["sport"]), 0)
        L.append(f"- **{r['name']}, {r['title'].lower()}, {season(r['sport'], r['year'])} ({val(r['raw'])})**: #{i} by score "
                 f"({r['z']:.2f}), top ten in {n} of 16. " + NOTES.get(r["name"], "") + " " + why_out(D, versions, r) + ".")
    L += ["", "## The list", "",
         "| # | Season | Sport | Value | Curve | Step | Score | Published score | Clear by | Best other on the graph | "
         "In top 10 of 16 versions |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(top[:12], 1):
        lz = lock.get((r["cat"], r["year"], r["name"]))
        L.append(f"| {i} | {r['name']}, {r['title'].lower()}, {season(r['sport'], r['year'])} | {r['sport']} | {r['raw']} | "
                 f"{to_own_units(r['base'], r, ctx):.4g} | "
                 f"{abs(to_own_units(r['base'] + r['scale'], r, ctx) - to_own_units(r['base'], r, ctx)):.3g} | "
                 f"**{r['z']:.2f}** | {lz if lz is not None else '-'} | {r['clear']:.2f} | {other(r)} | "
                 f"{ens.get((r['name'], r['sport']), 0)}{' (one-sided window)' if r['edge'] else ''} |")
    L += ["", "## What the rule removes from the old top ten (team sports)", "",
          "| Season | Score | Clear by | Best other on the graph |", "|---|---|---|---|"]
    for r in allpage[:12]:
        if r["clear"] < CLEAR_STEPS:
            L.append(f"| {r['name']}, {r['title'].lower()}, {season(r['sport'], r['year'])} | {r['z']:.2f} | "
                     f"{r['clear']:.2f} | {other(r)} |")
    L += ["", "## Where the well-known seasons land", "",
          "| Season | Value | Score | Rank among all team seasons | Clear by | Best other on the graph | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for cat, y, nm, what in FAMOUS:
        r = next((r for r in sc if r["cat"] == cat and r["year"] == y and nm in r["name"]), None)
        if r is None:
            L.append(f"| {what} | - | - | - | - | - | not scored (not that season's leader, or no league rate) |")
            continue
        rk = next((i for i, t in enumerate(top, 1) if t is r), None)
        on = r["clear"] >= CLEAR_STEPS
        verdict = (f"**#{rk}**" if rk and rk <= 10 else
                   ("in, below the top ten" if on else
                    ("out: another season on its graph is as good or better" if r["clear"] <= 0
                     else "out: another season on its graph comes within a step")))
        ra = rank_all.get((r["cat"], r["year"], r["id"]))
        L.append(f"| {what} | {r['raw']} | {r['z']:.2f} | {ra if ra else 'not the athlete best'} | {r['clear']:.2f} | "
                 f"{other(r)} | {verdict} |")
    if diffs and diffs[0][0] >= 0.005:
        L += ["", "## Scores that differ from the published model", "",
              "| Season | Score here | Published |", "|---|---|---|"]
        for d, r, lz in diffs:
            if d < 0.005 or len(L) > 400:
                break
            L.append(f"| {r['name']}, {r['title'].lower()}, {r['year']} | {r['z']:.3f} | {lz:.3f} |")
    open(os.path.join(HERE, "REVIEW_C.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print("old top ten:", [(r["name"], r["year"]) for r in allpage[:10]], "removed:",
          [r["name"] for r in allpage[:10] if (r["cat"], r["year"], r["id"]) in old and r["clear"] < CLEAR_STEPS])


if __name__ == "__main__":
    main()
