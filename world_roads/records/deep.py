"""Records far above the field: how far, how rare, and which events have stopped moving.

    python deep.py        # writes tables/seasons.csv, tables/deep_summary.csv; prints the findings

Built on Peter Larsson's all-time lists (alltime.py), which are complete down to a deep cut-off, so for
any season the k-th best athlete is exact whenever k athletes that season cleared the cut-off.

Per event and season (calendar year):
* lead     the season's best legal mark (the world lead);
* tenth    the season's 10th-best athlete (each athlete counted once): depth;
* near     marks within 1% of the world record standing at the time (performance terms).

Per event:
* premium_now   the record against the median world lead of the last ten seasons (2017-2026);
* premium_then  the record against the median world lead of the four seasons before it was set;
* return_years  how often a mark like the record would come up at today's level: a Gumbel
                (extreme-value) distribution fitted by moments to the world leads of 2012-2026,
                and the record's return period in seasons;
* depth trend   the 10th-best athlete, five-season means, and the five seasons when it peaked;
* since_near    seasons since anyone came within 1% of the record.

"Performance" is speed for running events and distance or height for jumps and throws, as in the film.
"""
import os, csv, math, statistics, datetime as dt
from collections import defaultdict
import numpy as np
import analysis as A

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
NOW_YEARS = list(range(2017, 2027))
FIT_YEARS = list(range(2012, 2027))
IMPL = {("M", "JT"): 1986, ("W", "JT"): 1999}


def perf(ev, m):
    return A.RUN_DIST[ev] / m if ev in A.RUN_DIST else m


def load_lists():
    perfs = defaultdict(list)
    wind = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "cache", "alltime_parsed.csv"))):
        k = (r["sex"], r["event"])
        r["mark"] = float(r["mark"])
        r["year"] = int(r["date"][:4])
        (wind if r["list"] == "wind" else perfs)[k].append(r)
    return perfs, wind


def season_table(sex, ev, rows, rec_line):
    field = ev in A.FIELD
    better = (lambda a, b: a > b) if field else (lambda a, b: a < b)
    cut = min(r["mark"] for r in rows) if field else max(r["mark"] for r in rows)
    by = defaultdict(dict)                                   # year -> athlete -> best mark
    for r in rows:
        if (sex, ev) in IMPL and r["year"] < IMPL[(sex, ev)]:
            continue
        cur = by[r["year"]].get(r["athlete"])
        if cur is None or better(r["mark"], cur["mark"]):
            by[r["year"]][r["athlete"]] = r
    out = []
    for y in sorted(by):
        best = sorted(by[y].values(), key=lambda r: -perf(ev, r["mark"]))
        recd = A.record_at(rec_line, dt.date(y, 12, 31))
        rp = perf(ev, recd["mark"]) if recd else None
        near = sum(1 for r in rows if r["year"] == y and rp and perf(ev, r["mark"]) >= 0.99 * rp)
        out.append({"sex": sex, "event": ev, "year": y, "lead": best[0]["mark"], "lead_by": best[0]["athlete"],
                    "tenth": best[9]["mark"] if len(best) >= 10 else None, "athletes": len(best),
                    "near1pct": near, "record": recd["mark"] if recd else None, "cutoff": cut})
    return out


def gumbel_return(values, x):
    """seasons between marks at least as good as x, for a Gumbel fitted by moments to season bests
    (performance terms, so bigger is better)"""
    v = np.array(values, float)
    beta = v.std(ddof=1) * math.sqrt(6) / math.pi
    mu = v.mean() - 0.5772 * beta
    F = math.exp(-math.exp(-(x - mu) / beta))
    return 1 / max(1 - F, 1e-12)


def main():
    rec = A.load_records()
    perfs, wind = load_lists()
    seasons, summary = [], []
    for sex in ("M", "W"):
        for ev in A.ORDER:
            if (sex == "M" and ev == "100mH") or (sex == "W" and ev == "110mH"):
                continue
            line = [r for r in rec[(sex, ev)] if not r["equalled"]]
            wr = line[-1]
            rows = perfs[(sex, ev)]
            st = season_table(sex, ev, rows, line)
            seasons += st
            S = {s["year"]: s for s in st}
            lead = lambda y: perf(ev, S[y]["lead"]) if y in S else None
            pw = perf(ev, wr["mark"])
            now = [lead(y) for y in NOW_YEARS if lead(y)]
            y0 = wr["date"].year
            then = [lead(y) for y in range(y0 - 4, y0) if lead(y)]
            fit = [lead(y) for y in FIT_YEARS if lead(y)]
            # best mark by anyone other than the record holder, ever, and in the last ten seasons
            # the record holder as this list names them: the author of its best legal mark (the women's
            # 100 m list files the official 10.49 under "doubtful wind-reading", so name her directly)
            top = max(rows, key=lambda r: perf(ev, r["mark"]))
            holder = "Florence Griffith-Joyner" if (sex, ev) == ("W", "100m") else top["athlete"]
            others = [r for r in rows if r["athlete"] != holder]
            p2 = max(others, key=lambda r: perf(ev, r["mark"]))
            rec_now = [r for r in rows if r["year"] in NOW_YEARS]
            best_now = max(rec_now, key=lambda r: perf(ev, r["mark"])) if rec_now else None
            tenth = {y: perf(ev, S[y]["tenth"]) for y in S if S[y]["tenth"]}
            win5 = {y: np.mean([tenth[k] for k in range(y - 4, y + 1)]) for y in tenth
                    if all(k in tenth for k in range(y - 4, y + 1))}
            peak = max(win5, key=win5.get) if win5 else None
            near_years = [y for y in S if S[y]["near1pct"] > 0]
            def d5(a, b):
                xs = [tenth[k] for k in range(a, b + 1) if k in tenth]
                return np.mean(xs) if len(xs) >= 3 else None
            s = {"sex": sex, "event": ev, "record": wr["mark"], "record_by": wr["athlete"], "record_year": y0,
                 "premium_now": pw / statistics.median(now) - 1 if now else None,
                 "premium_then": pw / statistics.median(then) - 1 if then else None,
                 "return_seasons": gumbel_return(fit, pw) if len(fit) >= 10 else None,
                 "second": p2["mark"], "second_by": p2["athlete"], "second_year": p2["year"],
                 "premium_second": pw / perf(ev, p2["mark"]) - 1,
                 "best_now": best_now["mark"] if best_now else None,
                 "best_now_by": best_now["athlete"] if best_now else "",
                 "best_now_year": best_now["year"] if best_now else None,
                 "depth_peak_5y_end": peak,
                 "depth_peak": win5[peak] if peak else None,
                 "depth_1984_88": d5(1984, 1988), "depth_1994_98": d5(1994, 1998), "depth_2004_08": d5(2004, 2008),
                 "depth_2014_18": d5(2014, 2018), "depth_2022_26": d5(2022, 2026),
                 "last_near_year": max(near_years) if near_years else None,
                 "near_marks_total": sum(S[y]["near1pct"] for y in S)}
            summary.append(s)
    with open(os.path.join(TABLES, "seasons.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seasons[0].keys()))
        w.writeheader()
        w.writerows(seasons)
    with open(os.path.join(TABLES, "deep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    fmt = lambda ev, v: A.fmt(ev, v) if v else "–"
    print(f"{'event':10} {'record':>9} {'yr':>4}  {'vs now':>7} {'vs then':>7}  {'return (seasons)':>16}  "
          f"{'#2 ever':>22}  {'best 2017-26':>26}  {'depth peak':>10}  {'last <1%':>8}")
    for s in sorted(summary, key=lambda s: -(s["premium_now"] or 0)):
        ev = s["event"]
        rs = s["return_seasons"]
        print(f"{s['sex']} {ev:8} {fmt(ev, s['record']):>9} {s['record_year']:>4}  {s['premium_now']*100:+6.1f}% "
              f"{(s['premium_then'] or 0)*100:+6.1f}%  {('%.0f' % rs) if rs and rs < 1e6 else '>1e6':>16}  "
              f"{fmt(ev, s['second']):>8} {s['second_by'][:13]:>13}  "
              f"{fmt(ev, s['best_now']):>8} {s['best_now_by'][:12]:>12} {s['best_now_year'] or '':>4}  "
              f"{(s['depth_peak_5y_end'] - 4) if s['depth_peak_5y_end'] else '':>5}-{s['depth_peak_5y_end'] or '':<4}  "
              f"{s['last_near_year'] or '':>8}")


def unperf(ev, p):
    return A.RUN_DIST[ev] / p if ev in A.RUN_DIST else p


def depth_report():
    S = list(csv.DictReader(open(os.path.join(TABLES, "deep_summary.csv"))))
    print(f"\n10th-best athlete of the season, five-season means (marks)")
    print(f"{'event':10} {'1984-88':>9} {'1994-98':>9} {'2004-08':>9} {'2014-18':>9} {'2022-26':>9}   peak window")
    for s in S:
        ev = s["event"]
        vals = []
        for k in ("depth_1984_88", "depth_1994_98", "depth_2004_08", "depth_2014_18", "depth_2022_26"):
            vals.append(A.fmt(ev, unperf(ev, float(s[k]))) if s[k] else "–")
        pk = s["depth_peak_5y_end"]
        print(f"{s['sex']} {ev:8} " + " ".join(f"{v:>9}" for v in vals) +
              (f"   {int(pk) - 4}-{pk} ({A.fmt(ev, unperf(ev, float(s['depth_peak'])))})" if pk else ""))


if __name__ == "__main__":
    main()
    depth_report()
