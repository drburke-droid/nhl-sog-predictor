"""League revenue, 1913-2025, in 2025 dollars, for the panel under the single-season race:  python revenue.py
-> ../media/paid/revenue.csv (league, year, revenue then, revenue in 2025 dollars, documented or between two documented years)

research/revenue_{mlb,nfl,nba,nhl}.csv hold every figure the research found, each with its source and a quote. For each
league and year one figure is used: the league's own total as reported, else the sum of every club's reported revenue,
in the order of BASIS_RANK below; a figure SKIP lists is left out, for the reason given. A rival league's revenue (the
AFL's, the ABA's, the WHA's) is added to its league's in the years both are documented, as the bars count rival
leagues' players in their league's colour. Canadian dollars are converted at the Federal Reserve's H.10 rates (1971 on)
and at par before (the Canadian dollar stayed within about 8% of the US dollar from 1946 to 1970).

Between two documented years revenue grows by the same percentage each year; a league's line begins at its first
documented year and ends at its last. Revenue is converted to 2025 dollars with the CPI-U (salaries.py).
"""
import os, sys, csv, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import salaries as S                                            # noqa: E402

FIRST, LAST = S.FIRST, S.LAST
RIVAL = {"AFL": "NFL", "ABA": "NBA", "WHA": "NHL"}
SKIP = {}                                                       # (league, year, revenue): why


def basis_rank(basis):
    """lower is preferred: a reported league total, then a sum of every club, then a defined subset (BRI, HRR)"""
    b = basis.lower()
    if re.search(r"\b(bri|hrr|basketball related|hockey related)\b", b):
        return 3
    if "sum" in b or "clubs" in b or "teams" in b:
        return 2
    return 1


def fx():
    import curate
    return curate.fx()


def main():
    rates = None
    raw = defaultdict(dict)                                     # league -> year -> (usd, row)
    for lg in ("mlb", "nfl", "nba", "nhl"):
        p = os.path.join(HERE, "research", f"revenue_{lg}.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf8")):
            y, v = int(r["year"]), float(r["revenue_usd"])
            if (r["league"], y, int(v)) in SKIP or v <= 0:
                continue
            if r.get("currency", "USD").strip().upper() == "CAD":
                if y >= 1972:
                    rates = rates or fx()
                    v = v / ((rates[y - 1] + rates[y]) / 2)
            best = raw[r["league"]].get(y)
            if best is None or basis_rank(r["basis"]) < basis_rank(best[1]["basis"]):
                raw[r["league"]][y] = (v, r)
    # rival leagues join their league's total in the years both are documented
    for rival, lg in RIVAL.items():
        for y, (v, r) in raw.pop(rival, {}).items():
            if y in raw[lg]:
                v0, r0 = raw[lg][y]
                raw[lg][y] = (v0 + v, dict(r0, basis=r0["basis"] + f"; plus the {rival}'s {v:,.0f}"))
    c = S.cpi()
    out = []
    for lg, d in sorted(raw.items()):
        ys = sorted(d)
        for y in range(max(ys[0], FIRST), min(ys[-1], LAST) + 1):
            if y in d:
                v, doc = d[y][0], 1
            else:
                a = max(x for x in ys if x < y)
                b = min(x for x in ys if x > y)
                v, doc = d[a][0] * (d[b][0] / d[a][0]) ** ((y - a) / (b - a)), 0
            out.append(dict(league=lg, year=y, revenue=round(v), real_2025=round(v * c[LAST] / c[y]), documented=doc,
                            basis=d[y][1]["basis"] if doc else "between documented years",
                            source=d[y][1]["source_url"] if doc else ""))
    with open(os.path.join(S.OUT, "revenue.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for lg in sorted(raw):
        rs = [r for r in out if r["league"] == lg]
        print(f"{lg}: {rs[0]['year']}-{rs[-1]['year']}, {sum(r['documented'] for r in rs)} documented years of {len(rs)}")


if __name__ == "__main__":
    main()
