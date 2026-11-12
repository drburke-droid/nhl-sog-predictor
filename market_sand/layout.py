"""The table: a city of islands that grows outward from the old centre, one island per company.

The VOC stands at the centre. Every other company gets an island in its home market's district, a wedge
pointing the way that market lies from Amsterdam (the initial great-circle bearing): New York to the west,
London west-south-west, Paris south-west, Johannesburg south, Riyadh south-east, the rest of Europe east-
south-east, Moscow, Shanghai and Tokyo to the north-east. Islands are placed in the order the companies
first appear, each as close to the centre as its wedge allows, so the city grows in rings like Amsterdam's
canals: the oldest companies (most of them husks by the end) form the old core, today's giants the outer
rings. The water between the islands is the market.

Wedges are as wide as their companies need (New York's is the widest) and are spread to avoid each other
while staying as close to their true bearing as they can. Distances from the centre are stretched sideways
(ASPECT) so the finished city fills a 16:9 frame.

    L = build(companies, peak_radius)   # companies: list of dict(id, district, first); peak_radius: {id: cells}
    L.lots[id] -> (cx, cy, r)            # island centre and the widest it gets, in cells (centre of the table = 0, 0)
    L.sectors[district] -> (bearing, half_width) in radians (0 = north/up, clockwise)
"""
import math, re
import numpy as np

# bearing from Amsterdam in degrees (0 = north), and which home countries belong to the district
DISTRICTS = {
    "Amsterdam":  dict(bearing=0, label="Amsterdam", countries=["Netherlands", "Dutch Republic", "Dutch", "Holland"]),
    "Moscow":     dict(bearing=68, label="Moscow", countries=["Russia", "Russian Empire"]),
    "Tokyo":      dict(bearing=33, label="Tokyo", countries=["Japan"]),
    "Seoul":      dict(bearing=42, label="Seoul", countries=["South Korea", "Korea"]),
    "Shanghai · Hong Kong": dict(bearing=52, label="Shanghai · Hong Kong", countries=["China", "Hong Kong"]),
    "Taipei":     dict(bearing=58, label="Taipei", countries=["Taiwan"]),
    "Sydney":     dict(bearing=75, label="Sydney", countries=["Australia"]),
    "Europe":     dict(bearing=125, label="Europe", countries=["Germany", "German", "Austria", "Switzerland", "Swiss", "Denmark", "Sweden", "Finland", "Italy", "Spain", "Belgium"]),
    "Riyadh":     dict(bearing=128, label="Riyadh", countries=["Saudi Arabia"]),
    "Johannesburg": dict(bearing=168, label="Johannesburg", countries=["South Africa"]),
    "Paris":      dict(bearing=207, label="Paris", countries=["France", "French", "Egypt"]),   # Egypt: the Suez Canal Company, run from Paris
    "São Paulo":  dict(bearing=230, label="São Paulo", countries=["Brazil"]),
    "London":     dict(bearing=255, label="London", countries=["United Kingdom", "UK", "England", "English", "Great Britain", "Britain", "British", "Scotland"]),
    "New York":   dict(bearing=287, label="New York", countries=["United States", "USA", "US", "U.S.", "American", "Canada", "Canadian"]),
}
CENTRE = "voc"        # the island at the centre of the table
LOT_PAD = 1.12        # island radius = pile's widest radius x this ...
LOT_MIN = 3.0         # ... and at least this many cells
GAP = 3.0             # water between islands, cells
ASPECT = 1.45         # sideways stretch of distances from the centre (the city fills a wide frame)
TIME_ORDER = 0.82     # a company sits at least this fraction as far out as the one before it in its district


def district_of(country, exchange=""):
    """District of the first country named (e.g. "Netherlands/United Kingdom" -> Amsterdam)."""
    if country.lower().startswith("netherlands antilles"):
        country = country.split("/", 1)[-1]          # Schlumberger: incorporated in Curacao, run from the US
    for tok in [x.strip() for x in re.split(r"[/;,(]| and ", country) if x.strip()]:
        for name, d in DISTRICTS.items():
            if any(tok.lower() == c.lower() or tok.lower().startswith(c.lower() + " ") for c in d["countries"]):
                return name
    raise ValueError(f"no district for {country!r} / {exchange!r}")


class Layout:
    pass


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sectors(area):
    """Angular wedges: width from the district's island area, centred as near its bearing as the others allow."""
    names = sorted(area, key=lambda n: DISTRICTS[n]["bearing"])
    weight = {n: area[n] ** 0.55 for n in names}
    tot = sum(weight.values())
    half = {n: math.pi * 0.94 * weight[n] / tot for n in names}
    cen = {n: math.radians(DISTRICTS[n]["bearing"]) for n in names}
    for _ in range(400):                       # push overlapping neighbours apart (around the circle)
        moved = False
        for a, b in zip(names, names[1:] + names[:1]):
            gap = _wrap(cen[b] - cen[a])
            if gap < 0:
                gap += 2 * math.pi
            need = half[a] + half[b] + 0.02
            if gap < need - 1e-6:
                d = (need - gap) / 2
                cen[a] -= d; cen[b] += d; moved = True
        if not moved:
            break
    return {n: (cen[n], half[n]) for n in names}


def build(companies, peak_radius):
    L = Layout()
    comps = sorted(companies, key=lambda c: (c["id"] != CENTRE, c["first"]))
    rad = {c["id"]: max(LOT_MIN, peak_radius.get(c["id"], 0) * LOT_PAD) for c in comps}
    area = {}
    for c in comps:
        area[c["district"]] = area.get(c["district"], 0) + (2 * rad[c["id"]] + GAP) ** 2
    L.sectors = _sectors(area)
    placed = []            # (x, y, r)
    last_rho = {}
    L.lots = {}
    for c in comps:
        k, r = c["id"], rad[c["id"]]
        if not placed:
            pos = (0.0, 0.0)
        else:
            cen, half = L.sectors[c["district"]]
            P = np.array(placed)
            rho = max(0.0, TIME_ORDER * last_rho.get(c["district"], 0.0))
            pos = None
            while pos is None:
                rho += 0.5
                n = max(3, int(2 * half * rho / 0.7))
                th = cen + np.linspace(-half, half, n)
                th = th[np.argsort(np.abs(th - cen))]            # nearest the wedge's middle first
                xs, ys = rho * ASPECT * np.sin(th), -rho * np.cos(th)
                d = np.hypot(xs[:, None] - P[:, 0], ys[:, None] - P[:, 1]) - P[:, 2] - r - GAP
                ok = np.nonzero((d >= 0).all(axis=1))[0]
                if len(ok):
                    pos = (float(xs[ok[0]]), float(ys[ok[0]]))
            last_rho[c["district"]] = rho
        placed.append((pos[0], pos[1], r))
        L.lots[k] = (pos[0], pos[1], r)
    P = np.array(placed)
    L.x0, L.x1 = float((P[:, 0] - P[:, 2]).min()), float((P[:, 0] + P[:, 2]).max())
    L.y0, L.y1 = float((P[:, 1] - P[:, 2]).min()), float((P[:, 1] + P[:, 2]).max())
    L.width, L.height = L.x1 - L.x0, L.y1 - L.y0
    L.district_of = {c["id"]: c["district"] for c in comps}
    return L
