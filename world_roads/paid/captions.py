"""The races' captions, from the data:  python captions.py [--career]  -> captions.json (captions_career.json)

* Milestones: the first salary on record here to reach $100,000, $1 million, $10 million, $25 million and $50 million
  in its own year's money, and what it was worth in 2025's.
* The record: the most paid for a season in 2025 dollars (with the milestone the same player reached, if within two
  years).
* Long reigns: a player who led for five seasons or more in a row, captioned as his run ends.
With --career (-> captions_career.json), from career_top.csv: each new leader who holds the lead three years (or to the
end), the first careers past $10 million, $100 million, $250 million and $500 million in 2025 dollars, reigns of ten
years or more; a pass that is also a milestone, or ends a long reign, shares one caption.
* A few documented moments, each checked against the data before it is used: Babe Ruth's 1930 salary against the
  president's ($75,000 from 1909 to 1949, set by Congress); free agency, after the 1975 Messersmith-McNally
  arbitration ended baseball's reserve clause.
A caption shows for 2.6 years of the race (3.6 s); when the next one is due sooner, it makes room, down to 1.5 years,
and the next one waits for the rest.
"""
import os, sys, csv, json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "paid")
FIRST = 1913
SHOW, SHORTEST = 2.6, 1.5                             # years a caption shows (3.6 s), and at least (2.1 s)


def m(v):
    return f"${v / 1e6:.1f} million" if v >= 1e6 else f"${v:,.0f}"


def annual():
    rows = list(csv.DictReader(open(os.path.join(MEDIA, "salaries.csv"))))
    for r in rows:
        r["year"], r["nominal"], r["real"] = int(r["year"]), float(r["nominal"]), float(r["real_2025"])
    by = defaultdict(list)
    for r in rows:
        by[r["year"]].append(r)
    caps = []
    # the documented moments
    ruth = [r for r in by[1930] if r["player"] == "Babe Ruth"]
    if ruth and ruth[0]["nominal"] == 80000:
        caps.append(dict(year=1930, text=["Babe Ruth's $80,000 tops President Hoover's $75,000",
                                          f"In 2025 dollars: {m(ruth[0]['real'])}"], hit=True))
    caps.append(FREE_AGENCY)
    # the record in 2025 dollars; a milestone the same player reached within two years shares its caption
    rec = max(rows, key=lambda r: r["real"])
    record = [f"Still the record in 2025 dollars: {rec['player']}, {rec['team']}",
              f"{m(rec['nominal'])} in {rec['year']} · {m(rec['real'])} in 2025 dollars"]
    # milestones in nominal money
    for level, label in ((1e5, "$100,000"), (1e6, "$1 million"), (1e7, "$10 million"), (2.5e7, "$25 million"), (5e7, "$50 million")):
        hit = sorted((r for r in rows if r["nominal"] >= level), key=lambda r: (r["year"], -r["nominal"]))
        if hit:
            r = hit[0]
            if record and r["player"] == rec["player"] and abs(r["year"] - rec["year"]) <= 2:
                caps.append(dict(year=rec["year"], text=[f"{r['player']}, {r['team']}: the first {label} salary, still the record in 2025 dollars",
                                                        record[1]], hit=True))
                record = None
                continue
            caps.append(dict(year=r["year"], text=[f"First {label} salary on record: {r['player']}, {r['team']}",
                                                  f"{m(r['nominal'])} in {r['year']} · {m(r['real'])} in 2025 dollars"], hit=True))
    if record:
        caps.append(dict(year=rec["year"], text=record, hit=True))
    # long reigns at No. 1
    lead = {y: max(by[y], key=lambda r: r["real"]) for y in by if FIRST <= y}
    caps += reigns(lead, 5, "the best-paid")
    return caps


def career():
    """the career race: each time a new career takes the lead (and holds it three years), the first careers past
    $10 million, $100 million, $250 million and $500 million in 2025 dollars, and the long reigns"""
    rows = list(csv.DictReader(open(os.path.join(MEDIA, "career_top.csv"))))
    for r in rows:
        r["year"], r["nominal"], r["real"], r["seasons"] = int(r["year"]), float(r["cum_nominal"]), float(r["cum_real"]), int(r["seasons"])
    lead = {r["year"]: r for r in rows if r["rank"] == "1"}
    caps = [FREE_AGENCY]
    years = sorted(lead)
    passes = {}
    for i, y in enumerate(years[1:], 1):
        new, old = lead[y], lead[years[i - 1]]
        ahead = years[i:i + 3]                        # held three years, or to the end
        if new["pid"] != old["pid"] and all(lead[z]["pid"] == new["pid"] for z in ahead):
            passes[new["pid"]] = dict(year=y, new=new, old=old)
    for level, label in ((1e7, "$10 million"), (1e8, "$100 million"), (2.5e8, "$250 million"), (5e8, "$500 million")):
        hit = sorted((r for r in rows if r["real"] >= level), key=lambda r: (r["year"], -r["real"]))
        if not hit:
            continue
        r = hit[0]
        p = passes.get(r["pid"])
        if p and abs(p["year"] - r["year"]) <= 2:     # one caption for both: the new leader is the first past the mark
            caps.append(dict(year=p["year"], text=[f"{r['player']} passes {p['old']['player']}, the first career past {label}",
                                                   f"in 2025 dollars, after {p['new']['seasons']} seasons ({m(p['new']['nominal'])} as paid)"], hit=True))
            del passes[r["pid"]]
        else:
            caps.append(dict(year=r["year"], text=[f"First career past {label} in 2025 dollars: {r['player']}",
                                                  f"after {r['seasons']} seasons ({m(r['nominal'])} as paid)"], hit=True))
    long = {(c["year"], c["pid"]): c for c in reigns(lead, 10, "the biggest career")}
    for p in passes.values():                         # a pass that ends a long reign says so
        end = long.pop((p["year"], p["old"]["pid"]), None)
        what = f", the biggest career for {end['n']} years" if end else ": the biggest career yet"
        caps.append(dict(year=p["year"], text=[f"{p['new']['player']} passes {p['old']['player']}{what}",
                                               f"{m(p['new']['real'])} in 2025 dollars, after {p['new']['seasons']} seasons"], hit=True))
    return caps + list(long.values())


FREE_AGENCY = dict(year=1977, text=["Free agency: the first free agents cash in",
                                    "an arbitrator struck down baseball's reserve clause in 1975"], hit=True)


def reigns(lead, least, what):
    """a player who led for `least` seasons or more in a row, captioned as his run ends"""
    out, run, start = [], None, None
    for y in sorted(lead) + [None]:
        p = (lead[y]["player"], lead[y]["pid"]) if y else None
        if p != run:
            if run and y and y - start >= least:
                out.append(dict(year=y, pid=run[1], n=y - start, text=[f"{run[0]}: {what} for {y - start} straight years ({start}-{y - 1})"]))
            run, start = p, y
    return out


def main():
    is_career = "--career" in sys.argv
    caps = career() if is_career else annual()
    caps.sort(key=lambda c: c["year"])
    out = []
    for c in caps:
        t0 = c["year"] - 0.5
        if out and out[-1]["to"] + 0.3 > t0:           # the one before makes room, down to SHORTEST, then this one waits
            out[-1]["to"] = max(out[-1]["from"] + SHORTEST, t0 - 0.3)
            t0 = max(t0, out[-1]["to"] + 0.3)
        out.append(dict(**{"from": t0, "to": t0 + SHOW}, text=c["text"], hit=c.get("hit", False)))
    name = "captions_career.json" if is_career else "captions.json"
    json.dump(out, open(os.path.join(HERE, name), "w"), indent=1)
    for c in out:
        print(f"{c['from']:7.1f}  {c['text'][0]}  |  {c['text'][1] if len(c['text']) > 1 else ''}")


if __name__ == "__main__":
    main()
