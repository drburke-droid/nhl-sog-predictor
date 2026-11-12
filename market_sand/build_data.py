"""Merge the four research eras into the film's data, in today's dollars.

    python3 build_data.py            # research/*/ -> data/companies.csv, data/values_today.csv, data/events.csv
    python3 build_data.py --report   # also print every company's peak and life

Rules:
- a company is one legal entity with one id across eras (ids fixed in research/SCHEMA.md);
- every value is converted with money.to_today (dollars at that date's exchange rate, then US CPI);
  where a source gives a low and a high estimate, the value is their geometric mean; where several rows
  give the same date (several sources, or the same figure in two currencies), the median;
- dates: YYYY-MM-DD as given; YYYY-MM mid-month; a bare year is mid-year, or 31 December for year-end values;
- district from the home country/exchange (layout.DISTRICTS); industry, name and fate from the era
  where the company was biggest, unless LABELS / OVERRIDE below say otherwise.
"""
import csv, glob, math, os, re, sys
from collections import defaultdict
import numpy as np

import layout, money

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "research")
OUT = os.path.join(HERE, "data")

LABELS = {   # short names for the glass and the captions
    "voc": "VOC", "wic": "WIC", "eic": "East India Co.", "bank_of_england": "Bank of England",
    "south_sea": "South Sea Co.", "mississippi_co": "Mississippi Co.", "hudsons_bay": "Hudson's Bay",
    "bazacle": "Bazacle", "standard_oil_nj": "Standard Oil · Exxon", "socony": "Socony · Mobil",
    "att": "AT&T", "us_steel": "US Steel", "ge": "GE", "ibm": "IBM", "gm": "GM", "ntt": "NTT",
    "microsoft": "Microsoft", "apple": "Apple", "nvidia": "Nvidia", "aramco": "Aramco",
    "tsmc": "TSMC", "google": "Alphabet (Google)", "facebook": "Meta (Facebook)", "amazon": "Amazon",
    "walmart": "Walmart", "royal_dutch_shell": "Shell", "sbc": "SBC · AT&T Inc.", "aig": "AIG",
    "icbc": "ICBC", "china_construction_bank": "China Construction Bank", "merck": "Merck",
    "johnson_johnson": "Johnson & Johnson", "procter_gamble": "Procter & Gamble", "bhp": "BHP",
    "ntt_docomo": "NTT DoCoMo", "pmi": "Philip Morris Intl.", "philip_morris": "Philip Morris · Altria",
    "aol_time_warner": "AOL Time Warner", "aol": "AOL", "ibj": "Industrial Bank of Japan",
}
# The same legal company under different ids in different eras -> one id (and a factor for its values).
ALIAS = {
    "chevron": ("standard_oil_ca", None),            # Standard Oil of California, renamed Chevron in 1984
    "prr": ("pennsylvania_rr", None),
    "nyc": ("new_york_central", None),
    "second_bus": ("bank_us_second", None),
    "atlantic_richfield": ("atlantic_refining", None),   # Atlantic Refining became ARCO in 1966
    "bp": ("anglo_persian", None),                   # Anglo-Persian -> Anglo-Iranian (1935) -> British Petroleum (1954)
    "national_city_bank": ("citicorp", None),         # National City Bank -> First National City (1955) -> Citicorp (1974)
    # Royal Dutch Petroleum owned 60% of the Royal Dutch/Shell group from the 1907 union until 2005: after 1907
    # its value / 0.6 stands for the whole group (as the 1990-2026 research does); before, it is Royal Dutch alone.
    "royal_dutch": ("royal_dutch_shell", lambda t: 1 / 0.6 if t >= 1907.1 else 1.0),
}
SHELL_TT_MERGED = 1907.1     # Shell Transport's own values stop at the union (from then on it is part of the group)
OVERRIDE = {
    "royal_dutch_shell": {"district": "Amsterdam"},   # Dutch-majority for most of its life (London-only since 2022)
}
NARRATION_ONLY = {"bazacle"}    # worth less than one grain in today's dollars: a line of narration, no vessel   # id -> {field: value} where an era's record needs correcting (say why in a comment)
SUFFIX = re.compile(r",?\s+(Company|Corporation|Corp\.?|Inc\.?|Incorporated|Ltd\.?|Limited|plc|PLC|S\.A\.|AG|N\.V\.|Co\.)$")


def era_rows(name):
    for d in sorted(glob.glob(os.path.join(RES, "*/"))):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
                r["_era"] = os.path.basename(d.rstrip("/"))
                yield r


def dec_year(date, kind=""):
    date = date.strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", date)
    if m:
        y, mo, d = map(int, m.groups())
        return y + (mo - 1) / 12 + (d - 0.5) / 365.25
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", date)
    if m:
        y, mo = map(int, m.groups())
        return y + (mo - 0.5) / 12
    m = re.fullmatch(r"(\d{4})(?:\.\d+)?", date)
    if m:
        y = float(date)
        if "." in date:
            return y
        return y + (0.999 if kind == "year_end" else 0.5)
    raise ValueError(f"bad date {date!r}")


def label_of(cid, name):
    if cid in LABELS:
        return LABELS[cid]
    s = name.split("(")[0].strip()
    for _ in range(2):
        s = SUFFIX.sub("", s).strip().rstrip("&,").strip()
    return s


def main(report=False):
    comps = defaultdict(list)
    for r in era_rows("companies.csv"):
        r["id"] = ALIAS.get(r["id"], (r["id"],))[0]
        r["successors"] = ";".join(ALIAS.get(x.strip(), (x.strip(),))[0] for x in re.split(r"[;,]", r.get("successors", "")) if x.strip())
        comps[r["id"]].append(r)
    # values
    by_date = defaultdict(lambda: defaultdict(list))     # id -> t -> [(kind, today $M, currency, row)]
    skipped = []
    for r in era_rows("values.csv"):
        if not r.get("value"):
            continue
        try:
            t = dec_year(r["date"], r.get("kind", ""))
            v = float(r["value"].replace(",", ""))
            today = money.to_today(v, r["currency"].upper(), t)
        except Exception as e:      # an unknown currency or a bad row: say so, don't guess
            skipped.append((r["id"], r["date"], r.get("currency"), str(e)))
            continue
        cid, fac = ALIAS.get(r["id"], (r["id"], None))
        if fac:
            today *= fac(t)
        if cid == "shell_tt" and t >= SHELL_TT_MERGED:
            continue
        by_date[cid][round(t, 4)].append((r.get("kind", ""), today, r["currency"].upper(), r))
    series = {}
    for cid, dates in by_date.items():
        pts = []
        for t, rows in sorted(dates.items()):
            lows = [x for k, x, _, _ in rows if k == "est_low"]
            highs = [x for k, x, _, _ in rows if k == "est_high"]
            mids = [x for k, x, _, _ in rows if k not in ("est_low", "est_high")]
            if lows and highs:
                mids.append(math.sqrt(np.median(lows) * np.median(highs)))
            elif not mids:
                mids = lows or highs
            pts.append((t, float(np.median(mids)), len(rows)))
        series[cid] = pts
    # companies
    out_c = []
    missing = []
    for cid, recs in comps.items():
        if cid in NARRATION_ONLY:
            continue
        if cid not in series:
            missing.append(cid)
            continue
        pts = series[cid]
        peak_t = max(pts, key=lambda p: p[1])[0]
        # the era record covering the peak decides name/industry; the last era decides the fate
        def covers(r):
            a = r.get("first_value_date") or "0"
            b = r.get("end_date") or "9999"
            try:
                return dec_year(a[:10]) - 1 <= peak_t <= dec_year(b[:10]) + 1
            except ValueError:
                return False
        main_r = next((r for r in recs if covers(r)), recs[-1])
        last_r = max(recs, key=lambda r: r["_era"])
        o = dict(id=cid, name=main_r["name"], label=label_of(cid, main_r["name"]),
                 district=layout.district_of(main_r.get("country", ""), main_r.get("exchange", "")),
                 industry=main_r["industry"], fate=(last_r.get("fate") or "alive").lower(),
                 successors=last_r.get("successors", ""), first=f"{pts[0][0]:.3f}", end=f"{pts[-1][0]:.3f}")
        o.update(OVERRIDE.get(cid, {}))
        out_c.append(o)
    os.makedirs(OUT, exist_ok=True)
    keep = {c["id"] for c in out_c}
    with open(os.path.join(OUT, "companies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_c[0])); w.writeheader(); w.writerows(sorted(out_c, key=lambda c: float(c["first"])))
    with open(os.path.join(OUT, "values_today.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "t", "value_today_musd", "n_rows"])
        for cid in sorted(keep):
            for t, v, n in series[cid]:
                w.writerow([cid, f"{t:.4f}", f"{v:.1f}", n])
    ev = []
    for r in era_rows("events.csv"):
        for f_ in ("from_ids", "to_ids"):
            r[f_] = ";".join(ALIAS.get(x.strip(), (x.strip(),))[0] for x in re.split(r"[;,]", r.get(f_, "")) if x.strip())
        try:
            ev.append(dict(t=f"{dec_year(r['date']):.4f}", type=r["type"].lower(), from_ids=r.get("from_ids", "").replace(",", ";"),
                           to_ids=r.get("to_ids", "").replace(",", ";"), detail=r.get("detail", "")))
        except ValueError:
            skipped.append(("event", r.get("date"), r.get("type"), "bad date"))
    ev.sort(key=lambda e: float(e["t"]))
    with open(os.path.join(OUT, "events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "type", "from_ids", "to_ids", "detail"]); w.writeheader(); w.writerows(ev)
    print(f"{len(out_c)} companies, {sum(len(series[c]) for c in keep)} values, {len(ev)} events")
    if missing:
        print("companies with no usable values:", ", ".join(sorted(missing)))
    if skipped:
        print(f"{len(skipped)} rows skipped, e.g.:")
        for s in skipped[:15]:
            print("  ", s)
    if report:
        for c in sorted(out_c, key=lambda c: -max(v for _, v, _ in series[c["id"]])):
            pts = series[c["id"]]
            t, v, _ = max(pts, key=lambda p: p[1])
            print(f"  {c['label'][:28]:28s} {c['district'][:12]:12s} {c['industry']:13s} peak ${v / 1000:9,.1f}B in {t:7.1f}"
                  f"  ({float(c['first']):.0f}-{float(c['end']):.0f}, {c['fate']})")


if __name__ == "__main__":
    main(report="--report" in sys.argv)
