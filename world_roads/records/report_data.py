"""Everything the "records out of reach" report draws, in one file: tables/deep_report.json.

    python deep.py && python report_data.py

Groups (every record falls in exactly one):
* within reach    a mark that would come up at least once in 35 seasons at today's level (Gumbel
                  return period, deep.py);
* era record      out of reach, set in 2000 or earlier, in an event whose depth (the season's 10th-best
                  athlete, five-season means) peaked before 2006 and is still below that peak today;
* lone outlier    out of reach, in an event whose field is as deep as it has ever been (or deeper).
"""
import os, csv, json, math
from collections import defaultdict
import deep, analysis as A

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")

DEPTH = [  # featured events: athletes per season at or beyond a fixed mark
    ("M", "100m", 10.00, "under 10.00"), ("W", "400m", 50.00, "under 50.00"),
    ("W", "800m", 117.00, "under 1:57.00"), ("M", "LJ", 8.40, "8.40 m or farther"),
    ("M", "HJ", 2.35, "2.35 m or higher"), ("M", "HT", 80.0, "80 m or farther"),
    ("W", "LJ", 7.00, "7.00 m or farther"), ("W", "SP", 20.0, "20 m or farther"),
    ("W", "DT", 68.0, "68 m or farther"),
]


def num(x):
    return None if x in ("", None) else float(x)


def main():
    perfs, wind = deep.load_lists()
    S = list(csv.DictReader(open(os.path.join(TABLES, "deep_summary.csv"))))
    # one set of "today" return periods for the whole page: bigfour.py's (all-time lists plus World
    # Athletics' season lists below their cut-offs)
    allp = os.path.join(TABLES, "allsports.json")
    if os.path.exists(allp):
        T = {c["id"]: c["today_T"] for c in json.load(open(allp))["categories"]}
        for s in S:
            k = f"ath_{s['sex']}_{s['event']}"
            if k in T and T[k]:
                s["return_seasons"] = str(T[k])
    seasons = list(csv.DictReader(open(os.path.join(TABLES, "seasons.csv"))))
    score = []
    for s in S:
        ev = s["event"]
        peak_end = num(s["depth_peak_5y_end"])
        peak, now = num(s["depth_peak"]), num(s["depth_2022_26"])
        below_peak = peak is not None and (now is None or now < peak * (0.998 if ev in A.FIELD else 0.999))
        era = int(s["record_year"]) <= 2000 and peak_end is not None and peak_end <= 2005 and below_peak
        ret = num(s["return_seasons"])
        group = "within reach" if not (ret and ret >= 35) else ("era record" if era else "lone outlier")
        score.append({
            "sex": s["sex"], "event": ev, "name": A.NAMES[ev], "short": A.SHORT[ev], "field": ev in A.FIELD,
            "record": A.fmt(ev, float(s["record"])), "record_by": deep_name(s["record_by"]),
            "record_year": int(s["record_year"]),
            "premium_now": num(s["premium_now"]), "premium_then": num(s["premium_then"]),
            "return_seasons": ret, "group": group,
            "second": A.fmt(ev, float(s["second"])), "second_by": s["second_by"], "second_year": int(s["second_year"]),
            "best_now": A.fmt(ev, float(s["best_now"])) if s["best_now"] else None, "best_now_by": s["best_now_by"],
            "best_now_year": int(s["best_now_year"]) if s["best_now_year"] else None,
            "depth_peak": (f"{int(peak_end) - 4}–{int(peak_end)}" if peak_end else None),
            "last_near_year": int(s["last_near_year"]) if s["last_near_year"] else None,
        })
    # depth series for the featured events
    depth = []
    for sex, ev, th, label in DEPTH:
        by = defaultdict(set)
        for r in perfs[(sex, ev)]:
            ok = r["mark"] >= th if ev in A.FIELD else r["mark"] < th
            if ok:
                by[r["year"]].add(r["athlete"])
        depth.append({"sex": sex, "event": ev, "label": f"{'Men' if sex == 'M' else 'Women'}'s {A.SHORT[ev].lower() if ev in A.FIELD else A.SHORT[ev]}",
                      "threshold": label, "series": [[y, len(by.get(y, ()))] for y in range(1976, 2027)]})
    # era fingerprint: decade of each top-100 performer's best
    finger = []
    for sex in ("M", "W"):
        for ev in A.ORDER:
            if (sex, ev) not in perfs or (sex == "M" and ev == "100mH") or (sex == "W" and ev == "110mH"):
                continue
            rows = perfs[(sex, ev)]
            if (sex, ev) in deep.IMPL:
                rows = [r for r in rows if r["year"] >= deep.IMPL[(sex, ev)]]
            best = {}
            for r in rows:
                p = deep.perf(ev, r["mark"])
                if r["athlete"] not in best or p > best[r["athlete"]][0]:
                    best[r["athlete"]] = (p, r["year"])
            top = sorted(best.values(), reverse=True)[:100]
            dec = defaultdict(int)
            for p, y in top:
                dec[y // 10 * 10] += 1
            finger.append({"sex": sex, "event": ev, "short": A.SHORT[ev], "n": len(top),
                           "decades": [dec.get(d, 0) for d in range(1960, 2030, 10)]})
    # the men's long jump
    L, W = perfs[("M", "LJ")], wind[("M", "LJ")]
    leads = {}
    for r in L:
        if r["year"] not in leads or r["mark"] > leads[r["year"]]["mark"]:
            leads[r["year"]] = r
    lj = {
        "leads": [[y, leads[y]["mark"], leads[y]["athlete"], "A" in leads[y]["flags"]] for y in sorted(leads) if y >= 1960],
        "record": [[deep_year(r["date"]), r["mark"], deep_name(r["athlete"])] for r in A.load_records()[("M", "LJ")]
                   if not r["equalled"] and r["date"].year >= 1955],
        "over880": [[r["mark"], r["athlete"], r["venue"], r["date"], "A" in r["flags"], r["wind"]]
                    for r in sorted([r for r in L if r["mark"] >= 8.80], key=lambda r: -r["mark"])],
        "wind880": [[r["mark"], r["athlete"], r["venue"], r["date"], "A" in r["flags"], r["wind"]]
                    for r in sorted([r for r in W if r["mark"] >= 8.80], key=lambda r: -r["mark"])],
        "over850_per_season": [[y, len({r["athlete"] for r in L if r["year"] == y and r["mark"] >= 8.50})]
                               for y in range(1976, 2027)],
    }
    s100 = {r["athlete"] for r in perfs[("M", "100m")]}
    best_lj = {}
    for r in L:
        if r["athlete"] not in best_lj or r["mark"] > best_lj[r["athlete"]][0]:
            best_lj[r["athlete"]] = (r["mark"], r["year"])
    lj["sprint_jumpers"] = sorted([[v[1], a, v[0], min(r["mark"] for r in perfs[("M", "100m")] if r["athlete"] == a)]
                                   for a, v in best_lj.items() if a in s100])
    out = {"score": score, "depth": depth, "fingerprint": finger, "lj": lj,
           "decades": [f"{d}s" for d in range(1960, 2030, 10)]}
    json.dump(out, open(os.path.join(TABLES, "deep_report.json"), "w"), indent=1, default=str)
    # the page: report_page.html with the data written in
    page = open(os.path.join(HERE, "report_page.html"), encoding="utf-8").read()
    allp = os.path.join(TABLES, "allsports.json")
    if os.path.exists(allp):                                 # built by bigfour.py
        out["all"] = json.load(open(allp))
    page = page.replace("/*__DATA__*/null", json.dumps(out, default=str, separators=(",", ":")))
    open(os.path.join(HERE, "report.html"), "w", encoding="utf-8").write(page)
    from collections import Counter
    print(Counter(s["group"] for s in score))
    for g in ("era record", "lone outlier", "within reach"):
        print(g, [f"{s['sex']} {s['event']}" for s in score if s["group"] == g])


def deep_name(s):
    import film
    return film.clean_name(s)


def deep_year(d):
    return round(d.year + (d.timetuple().tm_yday - 0.5) / 365.25, 3)


if __name__ == "__main__":
    main()
