"""The table: districts by home market, one island (lot) per company, canals between them.

Districts sit roughly where their markets are on a world map: the Americas on the left; London,
Amsterdam and Paris side by side in the middle with the rest of Europe in front of them; Moscow and
Riyadh; then East Asia on the right. Inside a district, islands are placed in rows from the back of the
table (north, top) to the front, in the order the companies first appear, so the oldest ones (most of
them husks by the end) stand at the back.

History starts in the middle: each district packs its oldest islands into the back corner nearest
Amsterdam (London and New York from their right-hand edge, the rest from their left), so the VOC, the
East India Company and the Mississippi Company stand a few canals apart and the opening can stay close.

Every company keeps its island for the whole film, sized for the widest its pile ever gets, so nothing
moves when a newcomer arrives and the husks stay where they were.

    L = build(companies, peak_radius)   # companies: list of dict(id, district, first); peak_radius: {id: cells}
    L.lots[id] -> (cx, cy, r)            # island centre and radius, in cells
    L.canals                             # list of (x0, y0, x1, y1, width): canal rectangles, in cells
    L.districts[name] -> (x0, y0, x1, y1)
"""
import re
import numpy as np

# col: left to right; band: back to front within the column; order: left to right within the band;
# align: the edge a district packs its oldest islands against
DISTRICTS = {
    "New York":   dict(col=0, band=0, order=0, align="right", label="New York",
                       countries=["United States", "USA", "US", "U.S.", "American", "Canada", "Canadian"]),
    "São Paulo":  dict(col=0, band=1, order=0, align="right", label="São Paulo", countries=["Brazil"]),
    "London":     dict(col=1, band=0, order=0, align="right", label="London",
                       countries=["United Kingdom", "UK", "England", "English", "Great Britain", "Britain", "British", "Scotland"]),
    "Amsterdam":  dict(col=1, band=0, order=1, align="left", label="Amsterdam",
                       countries=["Netherlands", "Dutch Republic", "Dutch", "Holland"]),
    "Paris":      dict(col=1, band=0, order=2, align="left", label="Paris", countries=["France", "French"]),
    "Europe":     dict(col=1, band=1, order=0, align="left", label="Frankfurt · Zurich · Helsinki",
                       countries=["Germany", "German", "Switzerland", "Swiss", "Denmark", "Sweden", "Finland", "Italy", "Spain", "Belgium"]),
    "Moscow":     dict(col=2, band=0, order=0, align="left", label="Moscow", countries=["Russia", "Russian Empire"]),
    "Riyadh":     dict(col=2, band=1, order=0, align="left", label="Riyadh", countries=["Saudi Arabia"]),
    "Tokyo":      dict(col=3, band=0, order=0, align="left", label="Tokyo", countries=["Japan"]),
    "Seoul":      dict(col=3, band=1, order=0, align="left", label="Seoul", countries=["South Korea", "Korea"]),
    "Shanghai · Hong Kong": dict(col=4, band=0, order=0, align="left", label="Shanghai · Hong Kong", countries=["China", "Hong Kong"]),
    "Taipei":     dict(col=4, band=1, order=0, align="left", label="Taipei", countries=["Taiwan"]),
    "Sydney":     dict(col=4, band=2, order=0, align="left", label="Sydney", countries=["Australia"]),
}
LOT_PAD = 1.12        # island radius = pile's widest radius x this ...
LOT_MIN = 3.0         # ... and at least this many cells (an empty pedestal still needs room)
CANAL = 2.5           # canal width between islands, cells
RIVER = 9.0           # river width between districts, cells
MARGIN = 14.0         # table edge, cells


def district_of(country, exchange=""):
    """District of the first country named (e.g. "Netherlands/United Kingdom" -> Amsterdam)."""
    for tok in [x.strip() for x in re.split(r"[/;,(]| and ", country) if x.strip()]:
        for name, d in DISTRICTS.items():
            if any(tok.lower() == c.lower() or tok.lower().startswith(c.lower() + " ") for c in d["countries"]):
                return name
    raise ValueError(f"no district for {country!r} / {exchange!r}")


class Layout:
    pass


def _pack(items, width, align):
    """Shelf packing: items [(id, r)] in order, rows of width <= width, from the back row forward.
    Returns {id: (x, y)} relative to the block's top-left, the block size, and the rows [(top, h, [ids])]."""
    pos, rows = {}, []
    x, y, row_h, cur = 0.0, 0.0, 0.0, []
    for key, r in items:
        d = 2 * r
        if cur and x + d > width:
            rows.append((y, row_h, cur)); y += row_h + CANAL; x, row_h, cur = 0.0, 0.0, []
        pos[key] = (x + r, y + r)
        cur.append(key)
        x += d + CANAL
        row_h = max(row_h, d)
    rows.append((y, row_h, cur))
    w = max(max(pos[k][0] + r for k, r in items), 0)
    if align == "right":
        pos = {k: (w - px, py) for k, (px, py) in pos.items()}
    return pos, (w, y + row_h), rows


def build(companies, peak_radius):
    L = Layout()
    by_d = {}
    for c in sorted(companies, key=lambda c: c["first"]):
        by_d.setdefault(c["district"], []).append((c["id"], max(LOT_MIN, peak_radius.get(c["id"], 0) * LOT_PAD)))
    blocks = {}
    for name, items in by_d.items():
        area = sum((2 * r + CANAL) ** 2 for _, r in items)
        width = max(max(2 * r for _, r in items), np.sqrt(area) * 1.15)
        blocks[name] = _pack(items, width, DISTRICTS[name]["align"])
    radius_of = {k: r for items in by_d.values() for k, r in items}
    # columns -> bands -> districts side by side
    cols = {}
    for name in blocks:
        d = DISTRICTS[name]
        cols.setdefault(d["col"], {}).setdefault(d["band"], []).append((d["order"], name))
    L.lots, L.districts, L.canals = {}, {}, []
    x = MARGIN
    col_spans = []
    for col in sorted(cols):
        bands = [[n for _, n in sorted(cols[col][b])] for b in sorted(cols[col])]
        band_w = [sum(blocks[n][1][0] for n in names) + RIVER * (len(names) - 1) for names in bands]
        col_w = max(band_w)
        y = MARGIN
        for names, bw in zip(bands, band_w):
            bx = x + (col_w - bw) / 2
            band_h = max(blocks[n][1][1] for n in names)
            for j, n in enumerate(names):
                pos, (w, h), rows = blocks[n]
                for key, (px, py) in pos.items():
                    L.lots[key] = (bx + px, y + py, radius_of[key])
                L.districts[n] = (bx, y, bx + w, y + h)
                # canals inside the district: along each row gap, and between neighbours in a row
                for (top, rh, keys), nxt in zip(rows, rows[1:] + [None]):
                    if nxt is not None:
                        L.canals.append((bx, y + nxt[0] - CANAL, bx + w, y + nxt[0], CANAL))
                    xs = sorted((L.lots[k][0], L.lots[k][2]) for k in keys)
                    for (ax, ar), (cx_, cr) in zip(xs, xs[1:]):
                        L.canals.append((ax + ar, y + top, cx_ - cr, y + top + rh, CANAL))
                if j < len(names) - 1:      # river between districts side by side
                    L.canals.append((bx + w, y - RIVER * 0.5, bx + w + RIVER, y + band_h + RIVER * 0.5, RIVER))
                bx += w + RIVER
            L.canals.append((x - RIVER * 0.5, y + band_h, x + col_w + RIVER * 0.5, y + band_h + RIVER, RIVER))   # river in front of the band
            y += band_h + RIVER
        col_spans.append((x, x + col_w, y - RIVER))
        x += col_w + RIVER
    L.width = x - RIVER + MARGIN
    L.height = max(s[2] for s in col_spans) + MARGIN
    for (x0, x1, _), nxt in zip(col_spans, col_spans[1:]):        # rivers between columns, back to front
        L.canals.append((x1, MARGIN * 0.5, nxt[0], L.height - MARGIN * 0.5, RIVER))
    return L
