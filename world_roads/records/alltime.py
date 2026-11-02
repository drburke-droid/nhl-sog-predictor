"""Parse Peter Larsson's all-time lists (alltime-athletics.com, cached by `python fetch.py alltime`).

    python alltime.py        # writes cache/alltime_parsed.csv and prints each list's depth

Which sections count
* running events: the main list only (legal wind, fully automatic timing; hand times, rolling starts,
  indoor marks and en-route splits are in other sections and are left out);
* jumps and throws: the main list plus the indoor list (indoor marks count for world records);
  "ancillary" marks (the other jumps or throws in a series), jump-offs and town-square meetings are
  left out;
* the wind-aided ("no") pages: their main list, kept separately as context.
"""
import os, re, csv, html, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SRC = os.path.join(CACHE, "alltime")
FIELD = {"HJ", "PV", "LJ", "TJ", "SP", "DT", "JT", "HT", "JT_old"}

ROW = re.compile(r"^\s*(\d+)\s+(\d[\d:.]*)([A-Za-z*#+]*)\s+(.*)$")
DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})\s*$")
WIND = re.compile(r"^([+\-±]?\d{1,2}\.\d)\s+")
NAT = re.compile(r"\s{2,}([A-Z]{3})\s+((?:\d\d\.\d\d\.\d\d)|(?:\s{0,8}))\s+")
NAT1 = re.compile(r"\s([A-Z]{3})\s+(\d\d\.\d\d\.\d\d)\s+")      # a name that fills its column
POS = re.compile(r"^(\d+[a-zA-Z]*\d*|[a-zA-Z]{1,2}\d*|-)\s{2,}")


def mark_value(s):
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        v = 0.0
        for p in parts:
            v = v * 60 + p
        return v
    return float(s)


def parse_row(line):
    m = ROW.match(line)
    if not m:
        return None
    rank, mark, flags, rest = m.groups()
    dm = DATE.search(rest)
    if not dm:
        return None
    date = None
    try:
        date = dt.date(int(dm.group(3)), max(1, int(dm.group(2))), max(1, int(dm.group(1))))
    except ValueError:
        date = dt.date(int(dm.group(3)), 7, 1)
    rest = rest[:dm.start()].rstrip()
    wind = ""
    wm = WIND.match(rest)
    if wm:
        wind = wm.group(1).replace("±", "")
        rest = rest[wm.end():]
    nm = NAT.search(rest) or NAT1.search(rest)
    if not nm:
        return None
    athlete = re.sub(r"\s+", " ", rest[:nm.start()]).strip()
    nat, dob = nm.group(1), nm.group(2).strip()
    tail = rest[nm.end():]
    pm = POS.match(tail)
    pos = ""
    if pm:
        pos, tail = pm.group(1), tail[pm.end():]
    venue = re.sub(r"\s+", " ", tail).strip()
    return {"rank": int(rank), "raw": mark + flags, "mark": mark_value(mark), "flags": flags, "wind": wind,
            "athlete": athlete, "nat": nat, "dob": dob, "pos": pos, "venue": venue, "date": date.isoformat()}


def sections(path):
    """-> {section name: [row lines]}: 'main' and, where present, 'indoor'.

    Section names come from the page's "Jump to:" menu. The first block of rows after that menu is the
    main list (some pages head it "All-time ... best", some "main list" or "wind-assisted", some not at
    all); later blocks are named by the menu entry that heads them."""
    s = open(path, encoding="utf-8").read()
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    lines = s.split("\n")
    names, menu_end = [], None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("jump to"):
            j = i + 1
            while j < len(lines) and j < i + 40:
                st = lines[j].strip().lower()
                if st.startswith("all-time") or st.startswith("back to") or (ROW.match(lines[j]) and DATE.search(lines[j])):
                    break
                if st:
                    names.append(st)
                j += 1
            menu_end = j
            break
    out, cur, main_done = {}, "main", False
    for ln in lines[menu_end or 0:]:
        st = ln.strip()
        if not st:
            continue
        if ROW.match(ln) and DATE.search(ln):
            if cur:
                out.setdefault(cur, []).append(ln)
            continue
        low = st.lower()
        if re.match(r"^\d+ total$", low) or low.startswith("jump to"):
            if cur == "main":
                main_done = True
            cur = None
            continue
        if low in names or any(low.startswith(n) for n in names if len(n) > 6) or low.startswith("indoor"):
            if names and low == names[0] and not main_done:
                cur = "main"
            elif (low.startswith("indoor") and "ancillary" not in low and "oversized" not in low
                  and "indoor" not in out):
                if cur == "main":
                    main_done = True
                cur = "indoor"
            else:
                if cur == "main" and out.get("main"):
                    main_done = True
                cur = None
    return out


def main():
    import fetch
    rows = []
    for (sex, ev), page in fetch.ALLTIME.items():
        path = os.path.join(SRC, f"{sex}_{ev}.htm")
        secs = sections(path)
        base = ev.replace("_w", "")
        wind_list = ev.endswith("_w")
        keep = ["main"] + (["indoor"] if base in FIELD and not wind_list else [])
        for sec in keep:
            bad = 0
            for ln in secs.get(sec, []):
                r = parse_row(ln)
                if r is None:
                    bad += 1
                    continue
                rows.append({"sex": sex, "event": base, "list": "wind" if wind_list else sec, **r})
            n = len(secs.get(sec, []))
            if n:
                ms = [parse_row(l)["mark"] for l in secs[sec] if parse_row(l)]
                print(f"{sex} {ev:9} {sec:6} rows={n:5d} unparsed={bad:3d}  best={min(ms) if base not in FIELD else max(ms):g}"
                      f"  cut-off={max(ms) if base not in FIELD else min(ms):g}")
    with open(os.path.join(CACHE, "alltime_parsed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
