"""World-record progressions, from the cached Wikipedia pages, into one table.

    python build_records.py           # writes tables/wr_progression.csv and prints each event's line

For each event we choose the tables to use (CONFIG below: page, table indices, and what each table is):
  pre   best marks known before the IAAF's first ratified records (1912 for men; 1920s/30s for women)
  hand  ratified records timed by hand (sprints and hurdles, to 1976)
  auto  ratified records timed electronically (from their first entry, usually 1968 or 1975-77)
  main  ratified records (one table covers the whole history)
Rows that the page's own legend marks as not ratified, or ratified and later rescinded, are dropped.
The record line is the running best of what is left: hand-timed marks until the first electronic one,
then electronic marks only. Each event's page, tables and legend are listed in the output.
"""
import os, re, csv, sys, json, datetime as dt
from wikitables import page_tables

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
FIELD = {"HJ", "PV", "LJ", "TJ", "SP", "DT", "JT", "HT"}

# (sex, event) -> (cache slug, [(table index, kind), ...]); filled in after inspecting each page
CONFIG = {
    ("M", "100m"): ("M_100m", [(0, "pre"), (2, "hand"), (3, "auto")]),
    ("W", "100m"): ("W_100m", [(1, "hand"), (2, "auto")]),
    ("M", "200m"): ("M_200m", [(1, "hand"), (2, "auto")]),
    ("W", "200m"): ("W_200m", [(1, "hand"), (2, "auto")]),
    ("M", "400m"): ("M_400m", [(2, "hand"), (3, "auto")]),
    ("W", "400m"): ("W_400m", [(1, "hand"), (2, "auto")]),
    ("M", "800m"): ("M_800m", [(2, "pre"), (3, "main")]),
    ("W", "800m"): ("W_800m", [(5, "main")]),
    ("M", "1500m"): ("M_1500m", [(0, "pre"), (2, "main")]),
    ("W", "1500m"): ("M_1500m", [(4, "main")]),
    ("M", "5000m"): ("M_5000m", [(0, "pre"), (2, "main")]),
    ("W", "5000m"): ("M_5000m", [(4, "main")]),
    ("M", "10000m"): ("M_10000m", [(0, "pre"), (1, "main")]),
    ("W", "10000m"): ("M_10000m", [(4, "main")]),
    ("M", "110mH"): ("M_110mH", [(0, "hand"), (2, "auto")]),
    ("W", "100mH"): ("W_100mH", [(1, "hand"), (2, "auto")]),
    ("M", "400mH"): ("M_400mH", [(0, "hand"), (2, "auto")]),
    ("W", "400mH"): ("W_400mH", [(1, "main")]),
    ("M", "HJ"): ("M_HJ", [(1, "pre"), (2, "main")]),
    ("W", "HJ"): ("W_HJ", [(1, "main")]),
    ("M", "PV"): ("M_PV", [(1, "main")]),
    ("W", "PV"): ("W_PV", [(0, "pre"), (1, "main")]),
    ("M", "LJ"): ("M_LJ", [(1, "main")]),
    ("W", "LJ"): ("W_LJ", [(1, "main")]),
    ("M", "TJ"): ("W_TJ", [(2, "main")]),                 # the women's title redirects to the combined page
    ("W", "TJ"): ("W_TJ", [(4, "pre"), (5, "main")]),
    ("M", "SP"): ("M_SP", [(2, "main")]),
    ("W", "SP"): ("W_SP", [(2, "main")]),
    ("M", "DT"): ("M_DT", [(1, "main")]),
    ("W", "DT"): ("W_DT", [(1, "main")]),
    # javelins: the implement was redesigned (men 1986, women 1999) and the records restarted
    ("M", "JT"): ("M_JT", [(1, "main"), (2, "main")]),     # marks starred on the page (annulled 1991 tails) dropped
    ("M", "JT_old"): ("M_JT", [(0, "main")]),
    ("W", "JT"): ("W_JT", [(1, "main")]),
    ("W", "JT_old"): ("W_JT", [(0, "main")]),
    ("M", "HT"): ("M_HT", [(0, "main")]),
    ("W", "HT"): ("W_HT", [(0, "main")]),
}
# pages whose key is written in prose rather than as a legend table
LEGEND = {"M_HT": {"#d6ecf3": "ratified", "#ffd0bd": "not ratified"}}
FIELD |= {"JT_old"}

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def parse_date(s):
    s = re.sub(r"\[.*?\]", "", s).strip()
    s = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", s)                # 'March 12th, 2026'
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})", s)
    if m and m[2].lower()[:3] in [k[:3] for k in MONTHS]:
        mon = next(v for k, v in MONTHS.items() if k[:3] == m[2].lower()[:3])
        return dt.date(int(m[3]), mon, int(m[1]))
    m = re.search(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m[1].lower()[:3] in [k[:3] for k in MONTHS]:
        mon = next(v for k, v in MONTHS.items() if k[:3] == m[1].lower()[:3])
        return dt.date(int(m[3]), mon, int(m[2]))
    m = re.search(r"([A-Za-z]+)\s+(\d{4})", s)
    if m and m[1].lower()[:3] in [k[:3] for k in MONTHS]:
        mon = next(v for k, v in MONTHS.items() if k[:3] == m[1].lower()[:3])
        return dt.date(int(m[2]), mon, 15)
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", s)
    if m:
        return dt.date(int(m[1]), 7, 1)
    return None


def parse_mark(s, field):
    s = s.replace(",", ".").replace(" ", "").strip()
    if field:
        m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", s) or re.match(r"^\s*(\d+\.\d+)", s)
        return float(m.group(1)) if m else None
    m = re.match(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", s)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    m = re.match(r"^(\d+):(\d{2}(?:\.\d+)?)", s)
    if m:
        return int(m[1]) * 60 + float(m[2])
    m = re.match(r"^(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def legend(tables):
    """colour -> status, from the page's legend tables (two-column, no header, words like 'ratified')"""
    out = {}
    for tb in tables:
        if tb["headers"]:
            continue
        for r in tb["rows"]:
            words = " ".join(v for k, v in r.items() if not k.startswith("_")).strip().lower()
            if "ratif" in words or "rescind" in words or "pending" in words:
                st = ("rescinded" if "rescind" in words else "pending" if "pending" in words
                      else "not ratified" if "not" in words else "ratified")
                out.setdefault(r["_bg"], st)
    return out


def _rgb(c):
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def status_of(bg, leg):
    """a row's status from its shading; shades not in the legend snap to the nearest legend colour"""
    if bg in leg:
        return leg[bg]
    if bg == "":
        return "ratified"
    rgb = _rgb(bg)
    cands = [(sum((a - b) ** 2 for a, b in zip(rgb, _rgb(k))), v) for k, v in leg.items() if k and _rgb(k) and rgb]
    return min(cands)[1] if cands and min(cands)[0] < 30 ** 2 else "?" + bg


def col(headers, *keys):
    for k in keys:
        for h in headers:
            if h.lower().startswith(k):
                return h
    for k in keys:
        for h in headers:
            if k in h.lower():
                return h
    return None


def build(sex, ev):
    slug, spec = CONFIG[(sex, ev)]
    tables = page_tables(slug)
    leg = LEGEND.get(slug) or legend(tables)
    field = ev in FIELD
    rows = []
    for ti, kind in spec:
        tb = tables[ti]
        h = tb["headers"]
        mc = col(h, "time", "mark", "height", "distance", "result", "performance")
        dc = col(h, "date")
        ac = col(h, "athlete", "name")
        for r in tb["rows"]:
            mark = parse_mark(r.get(mc, ""), field)
            if ev == "JT" and sex == "M" and "*" in r.get(mc, ""):
                mark = None                                       # annulled serrated-tail javelin marks
            date = parse_date(r.get(dc, ""))
            status = "unofficial" if kind == "pre" else status_of(r["_bg"], leg)
            rows.append({"sex": sex, "event": ev, "kind": kind, "table": ti, "status": status, "mark": mark,
                         "raw": r.get(mc, ""), "date": date, "athlete": re.sub(r"\s+", " ", r.get(ac, "")).strip()})
    return rows, leg


def line(rows, field):
    """the record line: the best known marks until the first ratified record, then ratified records only;
    for sprints, hand-timed records until the first electronic one, then electronic records only.
    Each restart begins a new running best (the line can step back, as the IAAF's own lists did)."""
    keep = [r for r in rows if r["mark"] and r["date"] and r["status"] in ("ratified", "pending", "unofficial")]
    official = [r["date"] for r in keep if r["kind"] != "pre"]
    first_official = min(official) if official else None
    autos = [r["date"] for r in keep if r["kind"] == "auto"]
    switch = min(autos) if autos else None
    out, best = [], None
    better = (lambda a, b: a > b) if field else (lambda a, b: a < b)
    for r in sorted(keep, key=lambda r: (r["date"], r["kind"] == "pre", r["kind"] != "auto")):
        if first_official and r["kind"] == "pre" and r["date"] >= first_official:
            continue
        if switch and r["kind"] in ("hand", "pre") and r["date"] >= switch:
            continue
        if out and ((r["kind"] != "pre" and out[-1]["kind"] == "pre") or
                    (r["kind"] == "auto" and out[-1]["kind"] != "auto")):
            best = None                                          # restart: ratified line, or electronic line
        if best is None or better(r["mark"], best) or r["mark"] == best:
            if best is not None and r["mark"] == best and out and out[-1]["mark"] == best:
                r = {**r, "equalled": True}
            out.append(r)
            best = r["mark"]
    return out, switch


def main(only=None):
    os.makedirs(TABLES, exist_ok=True)
    allrows = []
    for (sex, ev) in CONFIG:
        if only and f"{sex}_{ev}" not in only:
            continue
        rows, leg = build(sex, ev)
        ln, switch = line(rows, ev in FIELD)
        dropped = [r for r in rows if r["status"] not in ("ratified", "pending", "unofficial")]
        print(f"== {sex} {ev}  legend={leg}  switch={switch}  rows={len(rows)} line={len(ln)} dropped={len(dropped)}")
        for r in dropped:
            print(f"     dropped [{r['status']}] {r['raw']} {r['athlete']} {r['date']}")
        bad = [r for r in rows if not r["mark"] or not r["date"]]
        for r in bad:
            print(f"     UNPARSED {r['raw']!r} {r['date']} {r['athlete']}")
        for r in ln:
            print(f"     {r['date']}  {r['raw']:>12}  {r['kind']:5}  {r['athlete'][:40]}{'  (=)' if r.get('equalled') else ''}")
        for r in ln:
            allrows.append({k: r.get(k) for k in ("sex", "event", "date", "mark", "raw", "athlete", "kind", "status")} |
                           {"equalled": bool(r.get("equalled"))})
    if not only:
        with open(os.path.join(TABLES, "wr_progression.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        print("rows", len(allrows))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
