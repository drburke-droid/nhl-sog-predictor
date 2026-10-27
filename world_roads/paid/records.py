"""The best-paid seasons ever, to date, for the single-season race:  python records.py  -> ../media/paid/season_top.csv

Every documented season's salary in 2025 dollars (salaries.csv, the yearly race's figures), each season its own entry:
Michael Jordan's 1996-97 and his 1997-98 are two. The board of a year is the best seasons of that year and every year
before it, so a season stays until ten better ones have come along. Baseball seasons before 1913 count too (Baseball
Reference's salaries, converted with the Minneapolis Fed's CPI estimates, as in the career race), so the first board
holds the best seasons of the game's first forty years, not only 1913's. Estimated seasons (careers.csv) don't count.
season_top.csv: for every year 1913-2025, the 40 best seasons to date.
"""
import os, sys, csv, heapq
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import salaries as S                                            # noqa: E402
import lifetime as L                                            # noqa: E402

FIRST, LAST = S.FIRST, S.LAST
KEEP = 40


def main():
    rows = list(csv.DictReader(open(os.path.join(S.OUT, "salaries.csv"))))
    for r in rows:
        r["year"], r["nominal"], r["real"] = int(r["year"]), float(r["nominal"]), float(r["real_2025"])
    c = L.cpi()
    for r in S.mlb(first=1871):                               # the seasons before the race begins
        if r["year"] < FIRST:
            rows.append(dict(r, circuit="MLB", real=round(r["nominal"] * c[LAST] / c[r["year"]])))
    when = defaultdict(list)
    for r in rows:
        when[max(r["year"], FIRST - 1)].append(r)
    pool, out = [], []
    for y in range(FIRST - 1, LAST + 1):
        pool = heapq.nlargest(KEEP, pool + when[y], key=lambda r: (r["real"], r["player"], -r["year"]))
        if y < FIRST:
            continue
        for i, r in enumerate(pool):
            out.append(dict(year=y, rank=i + 1, league=r["league"], circuit=r["circuit"], player=r["player"], team=r["team"],
                            pid=r["pid"], season=r["year"], nominal=round(r["nominal"]), real_2025=round(r["real"]), basis=r["basis"]))
    with open(os.path.join(S.OUT, "season_top.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"season_top.csv: {len(out)} rows; the record: {pool[0]['player']} {pool[0]['year']}, ${pool[0]['real'] / 1e6:.1f}M")


if __name__ == "__main__":
    main()
