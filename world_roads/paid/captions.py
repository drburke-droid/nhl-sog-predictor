"""The race's captions, from the data:  python captions.py  -> captions.json

* Milestones: the first salary on record here to reach $100,000, $1 million, $10 million, $25 million and $50 million
  in its own year's money, and what it was worth in 2025's.
* The record: the most paid for a season in 2025 dollars (with the milestone the same player reached, if within two
  years).
* Long reigns: a player who led for five seasons or more in a row, captioned as his run ends.
* A few documented moments, each checked against the data before it is used: Babe Ruth's 1930 salary against the
  president's ($75,000 from 1909 to 1949, set by Congress); free agency, after the 1975 Messersmith-McNally
  arbitration ended baseball's reserve clause.
A caption shows for 2.6 years of the race (3.6 s); when the next one is due sooner, it makes room, down to 1.5 years,
and the next one waits for the rest.
"""
import os, csv, json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "paid")
SHOW, SHORTEST = 2.6, 1.5                             # years a caption shows (3.6 s), and at least (2.1 s)


def m(v):
    return f"${v / 1e6:.1f} million" if v >= 1e6 else f"${v:,.0f}"


def main():
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
    caps.append(dict(year=1977, text=["Free agency: the first free agents cash in",
                                      "an arbitrator struck down baseball's reserve clause in 1975"], hit=True))
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
    lead = {y: max(by[y], key=lambda r: r["real"]) for y in by}
    run, start = None, None
    for y in sorted(lead) + [None]:
        p = lead[y]["player"] if y else None
        if p != run:
            if run and y and y - start >= 5:
                caps.append(dict(year=y, text=[f"{run}: the best-paid for {y - start} straight years ({start}-{y - 1})"]))
            run, start = p, y
    caps.sort(key=lambda c: c["year"])
    out = []
    for c in caps:
        t0 = c["year"] - 0.5
        if out and out[-1]["to"] + 0.3 > t0:           # the one before makes room, down to SHORTEST, then this one waits
            out[-1]["to"] = max(out[-1]["from"] + SHORTEST, t0 - 0.3)
            t0 = max(t0, out[-1]["to"] + 0.3)
        out.append(dict(**{"from": t0, "to": t0 + SHOW}, text=c["text"], hit=c.get("hit", False)))
    json.dump(out, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    for c in out:
        print(f"{c['from']:7.1f}  {c['text'][0]}")


if __name__ == "__main__":
    main()
