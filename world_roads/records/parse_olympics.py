"""Olympic finals (the "field"): the top eight at every Summer Games, from the cached Olympedia pages.

    python parse_olympics.py      # writes tables/olympic_top8.csv and prints what it could not read

Rules
* Placings 1-8 only (ties such as "=3" count; "4 h1 r3/4", AC, DNF, DNS, DQ do not).
* Track: the mark in the "Final" column (or the only time column when the event had one round).
  Margins ("at 2 metres") are not times and are left empty; Olympedia's estimated times ("est") are kept
  but flagged.
* Field events: the final's mark. In early Games the qualifying round counted towards the final
  standings (athletes placed 4-8 have a qualifying mark and no final mark); there, each athlete's best of
  the two rounds is used, as the standings were.
"""
import os, re, json, csv
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "olympedia")
TABLES = os.path.join(HERE, "tables")
FIELD = {"HJ", "PV", "LJ", "TJ", "SP", "DT", "JT", "HT"}


def parse_mark(raw, field):
    """'12', '9.79', '1:52.8', '2-55:18.4', '2:06:32', '13.62' -> seconds or metres (float), est flag"""
    s = raw.replace("\xa0", " ").strip()
    s = re.sub(r"\(.*?\)", "", s).strip()                 # drop "(1 h3)" placings
    if not s or s[0] in "–-" or s.lower().startswith("at "):
        return None, False
    est = "est" in s.lower()
    s = s.replace(",", ".")
    m = re.match(r"^(\d+)[-:](\d{2}):(\d{2}(?:\.\d+)?)", s)          # h-mm:ss or h:mm:ss
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), est
    m = re.match(r"^(\d+):(\d{2}(?:\.\d+)?)", s)                      # m:ss.x
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)), est
    m = re.match(r"^(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), est
    return None, est


def mark_columns(headers):
    """indices of the columns that hold marks (round columns), skipping names, imperial units, splits"""
    out = []
    for i, h in enumerate(headers):
        hl = h.lower()
        if not h or h in ("Pos", "Number", "Competitor", "NOC") or "imperial" in hl or "rank" in hl \
                or "attempts" in hl or "misses" in hl or "throw-off" in hl or re.match(r"^\d+(\.\d+)? km$", hl):
            continue
        out.append(i)
    return out


MARK_HEADS = ("Time", "Distance", "Best Height Cleared", "Height", "Mark")


def _href(td):
    a = td.find("a", href=True)
    return a["href"] if a else td.get_text(strip=True)


def _round_tables(s, n_skip=1):
    """every round table on the page (after the summary): list of {href: (mark, hand)} where mark is the
    automatic time when the table has one, and hand the hand time ('' when there is no separate one)"""
    out = []
    for t in s.find_all("table", class_="table-striped")[n_skip:]:
        heads = [th.get_text(strip=True) for th in t.find("thead").find_all("th")]
        mi = next((heads.index(h) for h in MARK_HEADS if h in heads), None)
        hi = next((i for i, h in enumerate(heads) if h.lower().startswith("time (hand")), None)
        if mi is None:                                         # 'Time (Automatic)', 'Time (Hand)', ...
            cands = [i for i, h in enumerate(heads) if h.split(" (")[0] in MARK_HEADS]
            auto = [i for i in cands if "automatic" in heads[i].lower()]
            mi = (auto or cands or [None])[0]
        if mi is None or "Competitor" not in heads:
            continue
        ci = heads.index("Competitor")
        d = {}
        for tr in t.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) > max(mi, ci):
                hand = tds[hi].get_text(" ", strip=True) if hi is not None and hi != mi and hi < len(tds) else ""
                d[_href(tds[ci])] = (tds[mi].get_text(" ", strip=True), hand)
        out.append(d)
    return out


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


def event_date(s):
    """the last day of the event, from the page's first biodata table ('6 - 10 April 1896', '27 July - 1 August 2012')"""
    import datetime as dt
    bio = s.find("table", class_="biodata")
    for tr in bio.find_all("tr") if bio else []:
        if tr.find("th") and tr.find("th").get_text(strip=True) == "Date":
            txt = tr.find("td").get_text(" ", strip=True).split("—")[0]
            m = re.findall(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", txt)
            if m:
                d, mon, y = m[-1]
                return dt.date(int(y), MONTHS.index(mon.lower()) + 1, int(d)).isoformat()
    return None


def parse_page(path, event):
    s = BeautifulSoup(open(path, encoding="utf-8").read(), "lxml")
    date = event_date(s)
    t = s.find("table", class_="table-striped")
    headers = [th.get_text(strip=True) for th in t.find("thead").find_all("th")]
    cols = mark_columns(headers)
    field = event in FIELD
    rows = []
    for tr in t.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < len(headers) - 3:
            continue
        pos = tds[0].get_text(" ", strip=True)
        m = re.match(r"^=?(\d+)$", pos)
        if not m or int(m.group(1)) > 8:
            continue
        a = tds[headers.index("Competitor")]
        rows.append({"pos": int(m.group(1)), "athlete": a.get_text(strip=True), "href": _href(a),
                     "noc": tds[headers.index("NOC")].get_text(strip=True),
                     "cells": {headers[i]: tds[i].get_text(" ", strip=True) for i in cols if i < len(tds)}})
    hand_h = next((h for h in headers if h.lower().startswith("time (hand")), None)
    auto_h = next((h for h in headers if h.lower().startswith("time (auto")), None)
    if cols and "Final" not in headers and (hand_h or auto_h):  # one-round event with hand and/or automatic times
        for r in rows:
            r["final_raw"] = r["cells"].get(auto_h, "") if auto_h else ""
            r["hand_raw"] = r["cells"].get(hand_h, "") if hand_h else ""
            r["qual_raw"] = []
        source = "summary"
    elif cols:                                                 # marks are in the summary
        named = [headers[i] for i in cols if headers[i] in MARK_HEADS]
        final_h = next((h for h in ("Final Round, Re-run", "Final") if h in headers),
                       named[-1] if named else headers[cols[-1]])
        for r in rows:
            r["final_raw"] = r["cells"].get(final_h, "")
            r["hand_raw"] = ""
            r["qual_raw"] = [v for h, v in r["cells"].items() if h != final_h]
        source = "summary"
    else:                                                      # marks only in the round tables
        rts = _round_tables(s)
        win = rows[0]["href"] if rows else None
        fi = max([i for i, d in enumerate(rts) if win in d] or [len(rts) - 1])
        for r in rows:
            r["final_raw"], r["hand_raw"] = rts[fi].get(r["href"], ("", "")) if rts else ("", "")
            r["qual_raw"] = [d[r["href"]][0] for i, d in enumerate(rts) if i != fi and r["href"] in d]
        source = "rounds"
    # carry-over: some placed athletes have a qualifying mark and no final mark
    carry = field and any(parse_mark(r["final_raw"], field)[0] is None and
                          any(parse_mark(q, field)[0] for q in r["qual_raw"]) for r in rows)
    # the final's own table: when it gives hand and automatic times side by side, keep both
    rts_all = _round_tables(s)
    win = rows[0]["href"] if rows else None
    fis = [i for i, d in enumerate(rts_all) if win in d]
    both = {}
    if fis and not field:
        for href, (m_auto, m_hand) in rts_all[fis[-1]].items():
            if m_hand:
                both[href] = (parse_mark(m_auto, False)[0], parse_mark(m_hand, False)[0])
    out = []
    for r in rows:
        v, est = parse_mark(r["final_raw"], field)
        raw = r["final_raw"]
        hv, _ = parse_mark(r.get("hand_raw", ""), field)
        if carry:
            for q in r["qual_raw"]:
                qv, qest = parse_mark(q, field)
                if qv is not None and (v is None or qv > v):
                    v, est, raw = qv, qest, q
        out.append({"date": date, "pos": r["pos"], "athlete": r["athlete"], "noc": r["noc"], "raw": raw,
                    "mark": v if v is not None else hv,
                    "mark_auto": both[r["href"]][0] if r["href"] in both else (v if r.get("hand_raw") else None),
                    "mark_hand": both[r["href"]][1] if r["href"] in both else hv, "est": est,
                    "carry": bool(carry), "source": source})
    return out, [headers[i] for i in cols]


def main():
    os.makedirs(TABLES, exist_ok=True)
    index = json.load(open(os.path.join(CACHE, os.environ.get("OLY_INDEX", "index.json"))))
    allrows, problems = [], []
    for e in index:
        rows, names = parse_page(os.path.join(CACHE, f"result_{e['result_id']}.html"), e["event"])
        for r in rows:
            allrows.append({"year": e["year"], "sex": e["sex"], "event": e["event"], **r})
            if r["mark"] is None and r["raw"] and not r["raw"].lower().startswith("at ") and r["raw"][0] not in "–-":
                problems.append((e["year"], e["sex"], e["event"], r["pos"], r["raw"]))
    with open(os.path.join(TABLES, "olympic_top8.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print("rows", len(allrows), "events", len(index))
    for p in problems:
        print("UNREAD", p)


if __name__ == "__main__":
    main()
