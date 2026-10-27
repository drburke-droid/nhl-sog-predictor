"""League revenue, 1913-2025, in 2025 dollars, for the panel under the single-season race:  python revenue.py
-> ../media/paid/revenue.csv (league, year, revenue then, revenue in 2025 dollars, documented or between two documented years)

research/revenue_{mlb,nfl,nba,nhl}.csv hold every figure the research found, each with its source and a quote, often
several per year on different bases (a league's own statement, the sum of every club's revenue as Financial World,
Forbes, Sportico or CNBC estimated it, a CBA's defined subset such as the NBA's BRI or the NHL's HRR). These run 10-25%
apart, so each league's line keeps to one basis wherever it can (choose() below):

* MLB: the one figure per year the research chose: MLB's stated industry revenue (Congressional reports, the Blue Ribbon
  Panel's audited figures, the league's statements) and, where there is none, the sum of all clubs.
* NBA: the sum of all teams' revenue (Financial World 1990-96, Forbes from 1998; Forbes over Sportico); before 1990, the
  1956-57 sums reported to Congress and the league's own totals for 1971 and 1983. Not BRI.
* NHL: the sum of all clubs' revenue from 1990 (Financial World, Forbes, then CNBC or Sportico where Forbes is missing).
  Not HRR; not the 1947-56 figures (gate receipts only, or receipts net of taxes, with an incomplete Toronto).
* NFL: the one figure per year the research found: league totals and club sums from the Congressional hearings
  (1952-57; 1968-69 include the AFL's ten clubs, 1974-75), press figures for 1973, 1980, 1984 and 1988, and the sum of
  all clubs' revenue from Financial World (1990-96) and Forbes (1997 on). Not Sports Illustrated's hedged 1970 guess
  ("may have gone over $120 million").

Between two documented years revenue grows by the same percentage each year (marked; the panel draws a gap of more than
three years dotted); a league's line begins at its first documented year and ends at its last. Revenue is converted to
2025 dollars with the CPI-U (salaries.py).
"""
import os, sys, csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import salaries as S                                            # noqa: E402

FIRST, LAST = S.FIRST, S.LAST
SOURCE_ORDER = ("forbes", "financial world", "cnbc", "sportico")   # for a year with more than one club sum


def choose(lg, rows):
    """year -> the one row used, by the rules above"""
    by = defaultdict(list)
    for r in rows:
        by[int(r["year"])].append(r)
    out = {}
    for y, rs in by.items():
        b = lambda r: r["basis"].lower()
        if lg == "MLB":
            keep = rs
        elif lg == "NBA":
            keep = [r for r in rs if b(r).startswith("team-revenue sum")]
            if not keep and y < 1990:
                keep = [r for r in rs if b(r).startswith("league") and "approximate" not in b(r)]
        elif lg == "NHL":
            keep = [r for r in rs if y >= 1990 and b(r).startswith("sum of all")]
        else:                                         # NFL: one figure a year; not a hedged press guess
            keep = [r for r in rs if "hedged" not in b(r)]
        if keep:
            rank = lambda r: min([i for i, k in enumerate(SOURCE_ORDER) if k in b(r)] or [len(SOURCE_ORDER)])
            out[y] = min(keep, key=rank)
    return out


def fx():
    import curate
    return curate.fx()


def main():
    rates = None
    raw = {}                                                    # league -> year -> (usd, row)
    for lg in ("mlb", "nfl", "nba", "nhl"):
        p = os.path.join(HERE, "research", f"revenue_{lg}.csv")
        if not os.path.exists(p):
            continue
        rows = [r for r in csv.DictReader(open(p, encoding="utf8")) if r["league"].upper() == lg.upper()]
        picked = {}
        for y, r in choose(lg.upper(), rows).items():
            v = float(r["revenue_usd"])
            if r.get("currency", "USD").strip().upper() == "CAD" and y >= 1972:
                rates = rates or fx()
                v = v / ((rates[y - 1] + rates[y]) / 2)
            picked[y] = (v, r)
        raw[lg.upper()] = picked
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
