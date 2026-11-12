"""The table: districts by home market, one island (lot) per company, canals between them.

Districts sit roughly where their markets are on a world map: the Americas on the left; London,
Amsterdam, Paris and the rest of Europe in the middle; Riyadh; then East Asia on the right. Inside a
district, islands are placed in rows from the back of the table (north, top) to the front, in the order
the companies first appear, so the oldest ones (most of them husks by the end) stand at the back.

Every company keeps its island for the whole film, sized for the widest its pile ever gets, so nothing
moves when a newcomer arrives and the husks stay where they were.

    L = build(companies, peak_radius)   # companies: list of dict(id, district, first); peak_radius: {id: cells}
    L.lots[id] -> (cx, cy, r)            # island centre and radius, in cells
    L.canals                             # list of (x0, y0, x1, y1, width): canal rectangles, in cells
    L.districts[name] -> (x0, y0, x1, y1)
"""
import numpy as np

# district -> (column, order within the column), and which home countries/exchanges belong to it
DISTRICTS = {
    "New York":          dict(col=0, row=0, label="New York", countries=["United States", "USA", "US", "Canada"]),
    "London":            dict(col=1, row=0, label="London", countries=["United Kingdom", "UK", "England", "Great Britain", "Britain"]),
    "Amsterdam":         dict(col=1, row=1, label="Amsterdam", countries=["Netherlands", "Dutch Republic"]),
    "Paris":             dict(col=1, row=2, label="Paris", countries=["France"]),
    "Europe":            dict(col=1, row=3, label="Frankfurt · Zurich · Copenhagen", countries=["Germany", "Switzerland", "Denmark", "Sweden", "Finland", "Italy", "Spain", "Belgium"]),
    "Riyadh":            dict(col=2, row=0, label="Riyadh", countries=["Saudi Arabia"]),
    "Tokyo":             dict(col=3, row=0, label="Tokyo", countries=["Japan"]),
    "Seoul":             dict(col=3, row=1, label="Seoul", countries=["South Korea", "Korea"]),
    "Shanghai · Hong Kong": dict(col=4, row=0, label="Shanghai · Hong Kong", countries=["China", "Hong Kong"]),
    "Taipei":            dict(col=4, row=1, label="Taipei", countries=["Taiwan"]),
}
LOT_PAD = 1.12        # island radius = pile's widest radius x this ...
LOT_MIN = 3.0         # ... and at least this many cells (an empty pedestal still needs room)
CANAL = 2.5           # canal width between islands, cells
RIVER = 9.0           # river width between districts, cells
MARGIN = 14.0         # table edge, cells


def district_of(country, exchange=""):
    s = f"{country} {exchange}".lower()
    for name, d in DISTRICTS.items():
        if any(c.lower() in s for c in d["countries"]):
            return name
    raise ValueError(f"no district for {country!r} / {exchange!r}")


class Layout:
    pass


def _pack(items, width):
    """Shelf packing: items [(id, r)] in order, rows of total width <= width. Returns {id: (x, y)}, (w, h)."""
    pos, x, y, row_h, used_w = {}, 0.0, 0.0, 0.0, 0.0
    for key, r in items:
        d = 2 * r
        if x > 0 and x + d > width:
            y += row_h + CANAL; x, row_h = 0.0, 0.0
        pos[key] = (x + r, y + r)
        x += d + CANAL
        row_h = max(row_h, d)
        used_w = max(used_w, x - CANAL)
    return pos, (used_w, y + row_h)


def build(companies, peak_radius):
    L = Layout()
    by_d = {}
    for c in sorted(companies, key=lambda c: c["first"]):
        by_d.setdefault(c["district"], []).append((c["id"], max(LOT_MIN, peak_radius.get(c["id"], 0) * LOT_PAD)))
    blocks = {}
    for name, items in by_d.items():
        area = sum((2 * r + CANAL) ** 2 for _, r in items)
        widest = max(2 * r for _, r in items)
        width = max(widest, np.sqrt(area) * 1.15)
        blocks[name] = _pack(items, width)
    # columns of districts, left to right; each column stacks its districts top to bottom
    cols = {}
    for name in blocks:
        d = DISTRICTS[name]
        cols.setdefault(d["col"], []).append((d["row"], name))
    L.lots, L.districts, L.canals = {}, {}, []
    x = MARGIN
    col_spans = []
    for col in sorted(cols):
        names = [n for _, n in sorted(cols[col])]
        col_w = max(blocks[n][1][0] for n in names)
        y = MARGIN
        for n in names:
            pos, (w, h) = blocks[n]
            ox = x + (col_w - w) / 2
            for key, (px, py) in pos.items():
                r = dict(by_d[n])[key]
                L.lots[key] = (ox + px, y + py, r)
            L.districts[n] = (ox, y, ox + w, y + h)
            y += h + RIVER
        col_spans.append((x, x + col_w, y - RIVER))
        x += col_w + RIVER
    L.width = x - RIVER + MARGIN
    L.height = max(s[2] for s in col_spans) + MARGIN
    # canals: the gaps between islands. Rivers between columns and districts are drawn as wide channels;
    # inside a district, a canal runs along each row gap and between neighbouring islands in a row.
    for (x0, x1, _), nxt in zip(col_spans, col_spans[1:]):
        L.canals.append((x1, MARGIN * 0.5, nxt[0], L.height - MARGIN * 0.5, RIVER))
    for n, (x0, y0, x1, y1) in L.districts.items():
        L.canals.append((x0 - 1, y1, x1 + 1, y1 + RIVER, RIVER))          # river along the district's front
        rows = {}
        for key, (cx, cy, r) in L.lots.items():
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                rows.setdefault(round(cy - r, 3), []).append((cx, cy, r))
        tops = sorted(rows)
        for top in tops[1:]:
            L.canals.append((x0, top - CANAL, x1, top, CANAL))
        for top, lots in rows.items():
            lots.sort()
            h = 2 * max(r for _, _, r in lots)
            for (ax, _, ar), (bx, _, br) in zip(lots, lots[1:]):
                L.canals.append((ax + ar, top, bx - br, top + h, CANAL))
    return L
