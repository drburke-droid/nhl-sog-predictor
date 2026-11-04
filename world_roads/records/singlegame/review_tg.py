"""The sniff-test review for video A (single games, team sports): writes REVIEW_A.md from the topk.py model.

    python review_tg.py
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import topk as T                                           # noqa: E402

FAMOUS = [  # (category, season, name fragment, what it is)
    ("sg_nba_pts", 1962, "Chamberlain", "Wilt's 100, the NBA record"), ("sg_nba_pts", 2006, "Bryant", "Kobe's 81"),
    ("sg_nba_pts", 2024, "Doncic", "Doncic's 73"), ("sg_nba_reb", 1961, "Chamberlain", "Wilt's 55 rebounds, the record"),
    ("sg_nba_ast", 1991, "Skiles", "Skiles's 30 assists, the record"), ("sg_nba_ast", 1959, "Cousy", "Cousy's 28 assists"),
    ("sg_nba_3pm", 2019, "Thompson", "Klay Thompson's 14 threes, the record"),
    ("sg_nhl_points", 1976, "Sittler", "Sittler's 10 points, the record"), ("sg_nhl_goals", 1920, "Malone", "Joe Malone's 7 goals, the record"),
    ("sg_nhl_goals", 1969, "Berenson", "Red Berenson's 6 goals (1968-69)"), ("sg_nhl_assists", 1947, "Taylor", "Billy Taylor's 7 assists, the record"),
    ("sg_mlb_rbi", 1924, "Bottomley", "Bottomley's 12 RBI"), ("sg_mlb_rbi", 1993, "Whiten", "Whiten's 12 RBI (and 4 HR)"),
    ("sg_mlb_h", 1975, "Stennett", "Stennett's 7-for-7"), ("sg_mlb_hr", 1976, "Schmidt", "Schmidt's 4 HR (10 innings)"),
    ("sg_mlb_hr", 2012, "Hamilton", "Hamilton's 4 HR"), ("sg_mlb_so", 1998, "Wood", "Kerry Wood's 20 K"),
    ("sg_mlb_so", 1986, "Clemens", "Clemens's 20 K"), ("sg_mlb_so", 2016, "Scherzer", "Scherzer's 20 K"),
    ("sg_mlb_so", 1938, "Feller", "Feller's 18 K"), ("sg_nfl_rush_yds", 2007, "Peterson", "Adrian Peterson's 296, the record"),
    ("sg_nfl_rec_yds", 2013, "Johnson", "Calvin Johnson's 329"), ("sg_nfl_pass_yds", 2021, "Burrow", "Burrow's 525"),
    ("sg_nfl_pass_td", 2013, "Manning", "Peyton Manning's 7 TD passes"), ("sg_nfl_sacks", 2007, "Umenyiora", "Umenyiora's 6 sacks"),
    # before 1999, from the Pro Football Archives box scores
    ("sg_nfl_pass_yds", 1951, "Van Brocklin", "Van Brocklin's 554, the record"),
    ("sg_nfl_pass_yds", 1990, "Moon", "Warren Moon's 527"), ("sg_nfl_pass_yds", 1996, "Esiason", "Esiason's 522 (overtime)"),
    ("sg_nfl_rec_yds", 1989, "Anderson", "Flipper Anderson's 336 (overtime), the record"),
    ("sg_nfl_rec_yds", 1985, "Paige", "Stephone Paige's 309"), ("sg_nfl_rec_yds", 1950, "Box", "Cloyce Box's 302"),
    ("sg_nfl_rush_yds", 1977, "Payton", "Walter Payton's 275"), ("sg_nfl_rush_yds", 1976, "Simpson", "O.J. Simpson's 273"),
    ("sg_nfl_sacks", 1990, "Thomas", "Derrick Thomas's 7 sacks, the record"),
    ("sg_nfl_pass_td", 1954, "Burk", "Adrian Burk's 7 TD passes"), ("sg_nfl_pass_td", 1961, "Blanda", "Blanda's 7 TD passes (AFL)"),
    ("sg_nfl_pass_td", 1962, "Tittle", "Y.A. Tittle's 7 TD passes"), ("sg_nfl_pass_td", 1969, "Kapp", "Joe Kapp's 7 TD passes"),
]


def main():
    D, gam, grows, ctx = T.setup()
    sc = T.score(D, ctx=ctx)
    elig = T.clear_of_field(sc)
    top = T.page(elig)
    allpage = T.page(sc)
    rank_all = {(r["cat"], r["year"], r["id"]): i for i, r in enumerate(sorted(sc, key=lambda r: -r["z"]), 1)}
    ens = {r["athlete"]: int(r["specs_in_top10"]) for r in csv.DictReader(open(os.path.join(T.OUT, "tg_spec_ensemble.csv"), encoding="utf-8"))}
    k = lambda r: ctx.env[r["cat"]][r["year"]] ** ctx.gamma.get(r["cat"], 1.0)
    det = lambda r: D.detail[(r["cat"], r["year"], r["id"])]
    sport = lambda r: {"NBA": "NBA", "NHL": "NHL", "MLB": "MLB", "NFL": "NFL"}[r["sport"]]
    L = ["# Video A review: the biggest single-game outliers in team sports", "",
         f"Model: `topk.py` (curve through each season's top {T.K} other games, ±15 seasons; length-adjusted games; league "
         "adjustment; counts treated as counts; clear of the field by one step). "
         f"{len(sc):,} season-leading games in {len(D.meta)} categories; {len(elig)} clear their field.", "",
         "Numbers are in the athlete's own season: the curve and step are what a card would show; *best other* is the "
         "best game by anyone else on the same graph, converted to that season's scoring rate.", "",
         "## The list", "",
         "| # | Game | Sport | Raw | Length | Curve | Step | Score | Clear by | Best other on the graph | In top 10 of 16 versions |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(top, 1):
        d, bo = det(r), r["beaten_by"]
        L.append(f"| {i} | {r['name']}, {r['title'].lower()}, {d['date'][:10] if '-' in d['date'] else r['year']} | {sport(r)} | "
                 f"{float(r['raw']):g} | {d['length'] or 'n/a'} | {r['base'] * k(r):.1f} | {r['scale'] * k(r):.2f} | "
                 f"**{r['z']:.2f}** | {r['clear']:.2f} | {bo[1]} {float(bo[2]):g} ({bo[0]}) = {bo[3] * k(r):.1f} | "
                 f"{ens.get(r['name'], 0)}{' (one-sided window)' if r['edge'] else ''} |")
    L += ["", "## Where the well-known games land", "",
          "| Game | Raw | Length | Score | Rank among all season-best games | Clear by | Best other on the graph | Verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for cat, y, nm, what in FAMOUS:
        r = next((r for r in sc if r["cat"] == cat and r["year"] == y and nm in r["name"]), None)
        if r is None:
            row = next((x for x in D.seasons[cat].get(y, []) if nm in x.name), None)
            lead = D.seasons[cat].get(y, [None])[0]
            dd = D.detail.get((cat, y, row.id)) if row else None
            L.append(f"| {what} | {dd['raw_value'] if dd else '?'} | {dd['length'] if dd else '?'} | - | - | - | - | "
                     f"not that season's best game once adjusted for length ({lead.name}: {lead.perf:g}) |")
            continue
        d, bo = det(r), r["beaten_by"]
        on = r in elig
        rk = next((i for i, t in enumerate(top, 1) if t is r), None)
        verdict = f"**#{rk}**" if rk else ("in, below the top ten" if on else
                                           ("out: another game on its graph is as good or better" if r["clear"] <= 0
                                            else "out: another game on its graph comes within a step"))
        L.append(f"| {what} | {float(r['raw']):g} | {d['length'] or 'n/a'} | {r['z']:.2f} | {rank_all[(r['cat'], r['year'], r['id'])]} | "
                 f"{r['clear']:.2f} | {bo[1]} {float(bo[2]):g} ({bo[0]}) = {bo[3] * k(r):.1f} | {verdict} |")
    open(os.path.join(HERE, "REVIEW_A.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
