"""Read Wikipedia record-progression tables from the cached parse output, keeping what pandas drops:
the section each table sits in, and each row's background colour (these pages shade records that were
never ratified, or ratified and later rescinded).

    python wikitables.py M_100m        # print every table on a page with its section and a preview
"""
import os, re, sys, json
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
WIKI = os.path.join(HERE, "cache", "wikipedia")


def _bg(tag):
    st = (tag.get("style") or "") + ";" + (tag.get("bgcolor") and f"background:{tag.get('bgcolor')}" or "")
    m = re.search(r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,6}|[a-z]+)", st)
    return m.group(1).lower() if m else ""


def _text(td):
    for sup in td.find_all("sup"):
        sup.decompose()
    for s in td.find_all(class_=["sortkey", "mw-ref", "reference"]):
        s.decompose()
    for s in td.find_all(style=re.compile(r"display:\s*none")):
        s.decompose()
    return td.get_text(" ", strip=True).replace("\xa0", " ")


def _header(rows):
    """header names from one or more all-<th> rows (rowspan/colspan aware); a child name replaces its parent"""
    grid, spans = [], {}
    for tr in rows:
        cells = list(tr.find_all(["th", "td"], recursive=False))
        line, col = [], 0
        while cells or col in spans:
            if col in spans:
                rem, txt = spans[col]
                line.append(None)                                   # covered from above
                spans[col] = (rem - 1, txt)
                if rem - 1 <= 0:
                    del spans[col]
                col += 1
                continue
            c = cells.pop(0)
            txt = _text(c)
            rs = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
            cs = int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)
            for k in range(cs):
                line.append(txt)
                if rs > 1:
                    spans[col] = (rs - 1, txt)
                col += 1
        grid.append(line)
    n = max(len(l) for l in grid)
    out = []
    for i in range(n):
        names = [l[i] for l in grid if i < len(l) and l[i]]
        if not names:
            out.append(f"c{i}")
        elif len(names) == 1:
            out.append(names[0])
        else:
            out.append(f"{names[0]} {names[-1]}" if names[0] != names[-1] else names[0])
    return out


def read_table(t):
    """-> headers, rows; each row is a dict of header -> text plus '_bg' (row or first-cell shading)"""
    trs = t.find_all("tr")
    head_rows = []
    while trs and all(c.name == "th" for c in trs[0].find_all(["th", "td"], recursive=False)) and \
            trs[0].find_all(["th", "td"], recursive=False):
        head_rows.append(trs.pop(0))
    headers = _header(head_rows) if head_rows else []
    grid, spans = [], {}                                     # spans: col -> (remaining, text, bg)
    for tr in trs:
        cells = tr.find_all(["th", "td"], recursive=False)
        row, col, bgs = [], 0, [_bg(tr)]
        pending = list(cells)
        while pending or col in spans:
            if col in spans:
                rem, txt, bg = spans[col]
                row.append(txt)
                bgs.append(bg)
                if rem <= 1:
                    del spans[col]
                else:
                    spans[col] = (rem - 1, txt, bg)
                col += 1
                continue
            c = pending.pop(0)
            txt, bg = _text(c), _bg(c)
            rs = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
            cs = int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)
            for k in range(cs):
                row.append(txt)
                bgs.append(bg)
                if rs > 1:
                    spans[col] = (rs - 1, txt, bg)
                col += 1
        if row:
            grid.append((row, [b for b in bgs if b]))
    out = []
    for row, bgs in grid:
        d = {headers[i] if i < len(headers) else f"c{i}": v for i, v in enumerate(row)}
        d["_bg"] = bgs[0] if bgs else ""
        out.append(d)
    return headers, out


def page_tables(slug):
    j = json.load(open(os.path.join(WIKI, f"{slug}.json"), encoding="utf-8"))
    s = BeautifulSoup(j["parse"]["text"], "lxml")
    out = []
    for i, t in enumerate(s.find_all("table", class_="wikitable")):
        h2 = t.find_previous("h2")
        h3 = t.find_previous(["h3"])
        h4 = t.find_previous(["h4"])
        cap = t.find("caption")
        path = [h.get_text(strip=True) for h in (h2, h3, h4) if h is not None]
        # keep only headings that come after the h2 (h3/h4 from an earlier section don't apply)
        sect = [path[0]] if path else []
        for h in (h3, h4):
            if h is not None and h2 is not None and h.sourceline and h2.sourceline and \
                    (h.sourceline, h.sourcepos) > (h2.sourceline, h2.sourcepos):
                sect.append(h.get_text(strip=True))
        headers, rows = read_table(t)
        out.append({"i": i, "section": " > ".join(sect), "caption": cap.get_text(" ", strip=True) if cap else "",
                    "headers": headers, "rows": rows})
    return out


if __name__ == "__main__":
    for tb in page_tables(sys.argv[1]):
        print(f"[{tb['i']}] {tb['section']} | {tb['caption'][:70]} | {tb['headers'][:7]} n={len(tb['rows'])}")
        for r in tb["rows"][:2] + tb["rows"][-2:]:
            print("      ", {k: v[:22] for k, v in list(r.items())[:6]}, r["_bg"])
