"""The sniff-test review for video B (single performances in athletics and swimming): writes REVIEW_B.md.

The single-day film's model on the individual sports only (speed skating left out: its season lists are not
available to use): each season's best mark in each event, graded against the best mark by anyone else in the 15
seasons either side (quadratic trend, the athlete's own seasons left out, swimming's 2008-09 supersuit seasons as a
level shift), and the clear-of-field rule: a mark makes the list only if it is at least one step clear of every
other mark on its graph. Races are scored as speed and shown as times.

    python review_B.py
"""
import math, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as SG                                         # noqa: E402  (margins, the rule, gsmodel)

G = SG.G
SPORTS = ("Athletics", "Swimming")


class Individual(G.Data):
    def __init__(self):
        super().__init__("v2")
        for c in [c for c in self.cats if self.cats[c]["sport"] not in SPORTS]:
            del self.cats[c]
            self.seasons.pop(c, None)
            self.meta.pop(c, None)


def own(D, cat, v):
    """a comparison value in the event's own units: metres, or seconds for races (scored as speed)"""
    if not D.cats[cat]["performance"].startswith("speed"):
        return v
    return float(re.search(r"(\d+)", D.cats[cat]["performance"]).group(1)) / v


def show(D, cat, v):
    t = own(D, cat, v)
    if not D.cats[cat]["performance"].startswith("speed"):
        return f"{t:.2f} m"
    return f"{int(t // 60)}:{t % 60:05.2f}" if t >= 60 else f"{t:.2f}"


def mark(D, r):
    """the athlete's mark as raced or thrown: times as m:ss.xx"""
    if not D.cats[r["cat"]]["performance"].startswith("speed"):
        return f"{float(r['raw']):.2f} m"
    t = float(r["raw"])
    return f"{int(t // 60)}:{t % 60:05.2f}" if t >= 60 else f"{t:.2f}"


def ranked(D, spec):
    sc = SG.margins(D, G.score(D, spec, details=True))
    return sc, G.by_athlete(G.athlete_seasons(SG.clear_of_field(sc)))


FAMOUS = [("ath_M_LJ", 1968, "Beamon"), ("ath_M_JT", 1996, "Železný"), ("ath_M_PV", 1993, "Bubka"),
          ("ath_W_TJ", 2021, "Rojas"), ("swim_M_200_me", 2025, "Marchand"), ("ath_M_JT", 2020, "Vetter"),
          ("swim_W_100_bu", 2026, "Walsh"), ("swim_W_800_fr", 2016, "Ledecky"), ("ath_M_SP", 1975, "Oldfield"),
          ("ath_W_100m", 1988, "Griffith"), ("ath_W_200m", 1988, "Griffith"), ("ath_W_800m", 1983, "Kratochv"),
          ("ath_W_400m", 1985, "Koch"), ("ath_M_100m", 2009, "Bolt"), ("ath_M_200m", 2009, "Bolt"),
          ("ath_M_LJ", 1991, "Powell"), ("ath_M_HT", 1986, "Sedykh"), ("ath_M_DT", 1986, "Schult"),
          ("ath_M_PV", 2025, "Duplantis"), ("swim_M_50_fr", 2000, "Popov"), ("ath_M_400m", 2016, "van Niekerk")]


def main():
    D = Individual()
    spec = G.Spec(ties="all", trend="quad_safe")
    sc, top = ranked(D, spec)
    allpage = G.by_athlete(G.athlete_seasons(sc))
    ens = defaultdict(int)
    for wdw in (7, 10, 15, 20):
        for tr in ("ols", "quad_safe"):
            for env in ("none", "power"):          # no league adjustment in these sports: the pair is the same
                _, pg = ranked(D, G.Spec(ties="all", window=wdw, trend=tr, env=env))
                for r in pg[:10]:
                    ens[(r["name"], r["sport"])] += 1
    rank_all = {(r["cat"], r["year"], r["id"]): i for i, r in enumerate(allpage, 1)}

    def other(r):
        s, who = r["beaten_by"]
        return f"{who} {show(D, r['cat'], dict(r['pts'])[s])} ({s})"

    L = ["# Video B review: the biggest single-performance outliers in athletics and swimming", "",
         "Model: the single-day film's model on athletics and swimming only (speed skating left out): each season's "
         "best mark, graded against the best mark by anyone else in the 15 seasons either side, plus the rule: a mark "
         f"must be at least {SG.CLEAR_STEPS:g} step clear of every other mark on its graph. {len(sc):,} season-best "
         f"marks in {len(D.meta)} events; {sum(r['clear'] >= SG.CLEAR_STEPS for r in sc):,} clear their field.", "",
         "## The film's ten: the top ten by score (#11 and #12 shown for reference)", "",
         "The last column says how many of the 16 versions (windows of 7, 10, 15 and 20 seasons; straight or curved "
         "trend; the league adjustment does not apply to these sports, so each version is counted twice) also put "
         "the athlete in their top ten.", "",
          "| # | Mark | Sport | Curve | Step | Score | Clear by | Best other on the graph | In top 10 of 16 versions |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(top[:12], 1):
        if i == 11:
            L.append("| | *next* | | | | | | | |")
        step = abs(own(D, r["cat"], r["base"] + r["scale"]) - own(D, r["cat"], r["base"]))
        L.append(f"| {i} | {r['name']}, {r['title']}, {r['year']} ({mark(D, r)}) | {r['sport']} | "
                 f"{show(D, r['cat'], r['base'])} | {step:.2f}{' m' if 'speed' not in D.cats[r['cat']]['performance'] else ' s'} | **{r['z']:.2f}** | "
                 f"{r['clear']:.2f} | {other(r)} | {ens.get((r['name'], r['sport']), 0)}"
                 f"{' (one-sided window)' if r['edge'] else ''} |")
    L += ["", "## What the rule removes from the old top ten", "",
          "| Mark | Score | Clear by | Best other on the graph |", "|---|---|---|---|"]
    for r in allpage[:12]:
        if r["clear"] < SG.CLEAR_STEPS:
            L.append(f"| {r['name']}, {r['title']}, {r['year']} ({mark(D, r)}) | {r['z']:.2f} | {r['clear']:.2f} | {other(r)} |")
    L += ["", "## Where the well-known marks land", "",
          "| Mark | Score | Rank among all season bests | Clear by | Best other on the graph | Verdict |",
          "|---|---|---|---|---|---|"]
    for cat, y, nm in FAMOUS:
        r = next((r for r in sc if r["cat"] == cat and r["year"] == y and nm in r["name"]), None)
        if r is None:
            L.append(f"| {nm}, {cat}, {y} | - | - | - | - | not that season's best, or not on the lists |")
            continue
        rk = next((i for i, t in enumerate(top, 1) if t is r), None)
        verdict = (f"**#{rk}**" if rk and rk <= 10 else
                   ("in, below the top ten" if r["clear"] >= SG.CLEAR_STEPS else
                    ("out: another mark on its graph is as good or better" if r["clear"] <= 0
                     else "out: another mark on its graph comes within a step")))
        L.append(f"| {r['name']}, {r['title']}, {y} ({mark(D, r)}) | {r['z']:.2f} | "
                 f"{rank_all.get((r['cat'], r['year'], r['id']), 'not the athlete best')} | {r['clear']:.2f} | "
                 f"{other(r)} | {verdict} |")
    open(os.path.join(HERE, "REVIEW_B.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
