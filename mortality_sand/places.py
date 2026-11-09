"""Places: where and when, inside each catastrophe (inputs/places/*.csv).

The catalogue spreads a catastrophe over a broad region and evenly over its years. Where history gives more,
a place row takes its deaths out of the catastrophe's own rows and puts them at the right place and month:

  local     the dead lived here (a besieged city's people, a plague-struck province, a COVID hotspot): the
            grains are taken from the piles at the place, in its months.
  converge  the dead lived elsewhere and died here (soldiers at a battle, deportees at a death camp): each
            death still counts against the dead person's home population, so the grains are taken from
            home (the `from` shares, by country) and travel across the map to the place, where they rise.

The totals stay the catalogue's (and, after 1950, the UN's): a place only re-times and re-places deaths
already counted. Where a region's places would take more than its rows hold, they are scaled down to fit.

After 1950 the country-years are also given their months: COVID-19 years follow The Economist's excess-death
model for each country (inputs/covid_monthly_economist.csv), so the waves roll around the world as they did.
"""
import csv, glob, math, os
import numpy as np

import story as S
from proj import to_px

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
DATA = os.path.join(HERE, "data")
CAP = 0.95                   # places may take at most this share of a parent row (the rest stays spread)
CAP_POST = 0.90              # ... of a country-year (after 1950)
KM_PER_DEG = 111.0


# ------------------------------------------------------------------ reading
def _when(s, end=False):
    """'1942-08' -> 1942.5833 (start of month; end=True: end of month); '1942' -> 1942.0 / 1943.0."""
    s = str(s).strip()
    if "-" in s[1:]:
        y, m = s.split("-")[:2]
        y, m = int(y), int(m)
        return y + (m if end else m - 1) / 12.0
    y = int(float(s))
    return float(y + 1 if end else y)


def _from(s):
    out = []
    for part in (s or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        code, _, sh = part.partition(":")
        out.append((code.strip().upper(), float(sh or 1)))
    tot = sum(sh for _, sh in out)
    return [(c, sh / tot) for c, sh in out] if tot > 0 else []


def read_places():
    rows = []
    for path in sorted(glob.glob(os.path.join(INP, "places", "*.csv"))):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            try:
                d = float(r["deaths_central"])
                t0, t1 = _when(r["start"]), _when(r["end"] or r["start"], end=True)
                lon, lat = float(r["lon"]), float(r["lat"])
            except (ValueError, KeyError, TypeError):
                print("  place skipped (unreadable):", os.path.basename(path), r.get("place"))
                continue
            kind = (r.get("kind") or "local").strip().lower()
            fr = _from(r.get("from", "")) if kind == "converge" else []
            if kind == "converge" and not fr:
                kind = "local"
            rows.append(dict(parent=r["parent_id"].strip(), place=r["place"].strip(), kind=kind, lon=lon, lat=lat,
                             sigma=max(float(r.get("radius_km") or 50), 10.0) / KM_PER_DEG,
                             t0=t0, t1=max(t1, t0 + 1 / 24), deaths=d, frm=fr, file=os.path.basename(path)))
    return rows


# ------------------------------------------------------------------ where is a point
_CC = None


def country_at(lon, lat):
    """ISO3 of the country under (lon, lat) on the film's map (the nearest land within ~0.6 deg)."""
    global _CC
    if _CC is None:
        z = np.load(os.path.join(DATA, "country_cells.npz"), allow_pickle=True)
        WC, HC = 1920 // S.CELL, 1080 // S.CELL
        _CC = (z["cc"].reshape(HC, WC), z["names"].tolist())
    cc, names = _CC
    x, y = to_px(lon, lat)
    cx, cy = int(float(np.ravel(x)[0]) // S.CELL), int(float(np.ravel(y)[0]) // S.CELL)
    for r in range(0, 4):
        win = cc[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
        vals = win[win >= 0]
        if len(vals):
            return names[int(np.bincount(vals).argmax())]
    return None


def vessel_of(iso, lon=None, year=1900):
    if iso == "RUS":
        if lon is not None:
            return "SIB" if lon > 60 else "STP"
        return "SIB" if S.siberia_share(year) > 0.5 else "STP"
    return S.ISO_VESSEL.get(iso)


def _home_split(code, year):
    """A home code -> [(vessel, iso or None, share)]. Country codes first (POL is Poland, NAM Namibia, not the
    Polynesia or North America vessels); Russia splits between European Russia and Siberia."""
    if code == "RUS":
        sh = S.siberia_share(year)
        return [("STP", "RUS", 1 - sh), ("SIB", "RUS", sh)]
    v = S.ISO_VESSEL.get(code)
    if v:
        return [(v, code, 1.0)]
    if code in S.VESSELS:
        return [(code, None, 1.0)]
    return []


_CENTRE = None


def country_lonlat(iso):
    """A country's representative point (lon, lat), for matching it to the nearest catalogue row."""
    global _CENTRE
    if _CENTRE is None:
        import vessels as V
        _CENTRE = {}
        for n, g in V.load_countries().items():
            c = g.representative_point()
            _CENTRE[S.NE_TO_ISO.get(n, n)] = (c.x, c.y)
    return _CENTRE.get(iso)


# ------------------------------------------------------------------ before 1950: the catalogue's rows
def _pick_row(rows, vessel, t0, t1, lon, lat):
    """The parent row a piece comes out of: among the event's rows in the same region and overlapping in
    time (else in the same region, else any), the one whose spot best covers (lon, lat): the nearest in
    units of the row's own spread, so a country's row wins over its neighbour's and a small row (the 1938
    Yellow River flood) over the big one around it."""
    def ov(r):
        return min(r["y1"] + 1, t1) - max(r["y0"], t0)

    def fit(r):
        d = math.hypot((r["lon"] - lon) * math.cos(math.radians(lat)), r["lat"] - lat)
        return d / max(r["sigma"], 0.5)
    same = [r for r in rows if r["vessel"] == vessel]
    for cand in ([r for r in same if ov(r) > 0], same, [r for r in rows if ov(r) > 0], rows):
        if cand:
            return min(cand, key=fit)
    return None


def apply_pre1950(spots, places):
    """Split the catalogue's spot lifts (one per row) into place lifts. Returns (lifts, report)."""
    by_id = {}
    for l in spots:
        by_id.setdefault(l["id"], []).append(l)
    pieces = []                      # (row, place, vessel, iso, deaths, kind)
    skipped = []
    for p in places:
        if p["t0"] >= 1950:
            continue
        rows = by_id.get(p["parent"])
        if not rows:
            skipped.append((p["parent"], p["place"], "no such event before 1950"))
            continue
        ymid = 0.5 * (p["t0"] + p["t1"])
        iso_here = country_at(p["lon"], p["lat"])
        v_here = vessel_of(iso_here, p["lon"], ymid) if iso_here else None
        if v_here is None:
            r = _pick_row(rows, None, p["t0"], p["t1"], p["lon"], p["lat"])
            v_here = r["vessel"]
        shares = p["frm"] if p["kind"] == "converge" else [("LOCAL", 1.0)]
        for code, sh in shares:
            if code == "LOCAL":
                homes = [(v_here, None, 1.0)]
                kind = "local"
            else:
                homes = _home_split(code, ymid)
                kind = "converge"
                if not homes:
                    skipped.append((p["parent"], p["place"], f"unknown home {code}"))
                    continue
            for v, iso, f in homes:
                at = (p["lon"], p["lat"])
                if kind == "converge" and iso:
                    at = ((40.0, 55.0) if v == "STP" else (85.0, 56.0)) if iso == "RUS" else \
                        (country_lonlat(iso) or at)              # Russia: its European heartland / Siberia
                r = _pick_row(rows, v, p["t0"], p["t1"], *at)
                pieces.append([r, p, v, iso, p["deaths"] * sh * f, kind])
    # scale down where a row would be overdrawn
    want = {}
    for pc in pieces:
        want[id(pc[0])] = want.get(id(pc[0]), 0) + pc[4]
    scaled = []
    for pc in pieces:
        r = pc[0]
        k = min(1.0, CAP * r["deaths"] / want[id(r)]) if want[id(r)] > 0 else 0
        if k < 0.999:
            scaled.append((r["id"], r["vessel"], want[id(r)], r["deaths"]))
        pc[4] *= k
    taken = {}
    out = []
    for r, p, v, iso, d, kind in pieces:
        if d <= 0:
            continue
        taken[id(r)] = taken.get(id(r), 0) + d
        base = dict(vessel=v, y0=int(math.floor(p["t0"])), y1=int(math.ceil(p["t1"])) - 1, t0=p["t0"], t1=p["t1"],
                    deaths=float(d), cause=r["cause"], name=r["name"], label=r["label"], id=r["id"],
                    place=p["place"], plon=p["lon"], plat=p["lat"])
        if kind == "local":
            base.update(kind="spot", lon=p["lon"], lat=p["lat"], sigma=p["sigma"])
        else:
            base.update(kind="converge", iso=iso, lon=r["lon"], lat=r["lat"], sigma=r["sigma"],
                        to_lon=p["lon"], to_lat=p["lat"], to_sigma=p["sigma"])
        out.append(base)
    rest = []
    for l in spots:
        d = l["deaths"] - taken.get(id(l), 0)
        if d > 1:
            rest.append(dict(l, deaths=float(d)))
    return rest + out, dict(skipped=skipped, scaled=sorted(set(scaled)))


# ------------------------------------------------------------------ after 1950: country-years
def covid_profile():
    """iso -> {year: 12 monthly weights} from The Economist's model (negative months count as 0)."""
    path = os.path.join(INP, "covid_monthly_economist.csv")
    prof = {}
    if not os.path.exists(path):
        return prof
    for r in csv.DictReader(open(path)):
        y, m = int(r["month"][:4]), int(r["month"][5:7])
        prof.setdefault(r["iso3"], {}).setdefault(y, np.zeros(12))[m - 1] += max(float(r["excess"]), 0.0)
    return prof


def apply_post1950(lifts, places, months_from=2020, months_to=2023):
    """Split country-year lifts into place lifts (and, in the COVID years, into months). Returns (lifts, report)."""
    key = {}
    for l in lifts:
        key.setdefault((l["id"], l["iso"], l["vessel"], l["y0"]), []).append(l)
    pieces = []                       # [target lift, place, vessel, iso, deaths, kind, t0, t1]
    skipped = []
    for p in places:
        if p["t1"] <= 1950:
            continue
        ymid = 0.5 * (p["t0"] + p["t1"])
        iso_here = country_at(p["lon"], p["lat"])
        shares = p["frm"] if p["kind"] == "converge" else [("LOCAL", 1.0)]
        for code, sh in shares:
            if code == "LOCAL":
                if iso_here is None:
                    skipped.append((p["parent"], p["place"], "not on land"))
                    continue
                homes = [(vessel_of(iso_here, p["lon"], ymid), iso_here, 1.0)]
                kind = "local"
            else:
                homes = _home_split(code, ymid)
                kind = "converge"
            for v, iso, f in homes:
                for Y in range(int(math.floor(p["t0"])), int(math.ceil(p["t1"]))):
                    a, b = max(p["t0"], Y), min(p["t1"], Y + 1)
                    if b <= a:
                        continue
                    tg = key.get((p["parent"], iso, v, Y))
                    if not tg:
                        skipped.append((p["parent"], f"{p['place']} {Y} ({iso})", "no excess labelled so"))
                        continue
                    d = p["deaths"] * sh * f * (b - a) / (p["t1"] - p["t0"])
                    pieces.append([tg[0], p, v, iso, d, kind, a, b])
    want = {}
    for pc in pieces:
        want[id(pc[0])] = want.get(id(pc[0]), 0) + pc[4]
    scaled = []
    for pc in pieces:
        r = pc[0]
        k = min(1.0, CAP_POST * r["deaths"] / want[id(r)]) if want[id(r)] > 0 else 0
        if k < 0.999:
            scaled.append((r["id"], r["iso"], r["y0"], round(want[id(r)]), round(r["deaths"])))
        pc[4] *= k
    taken, placed_m = {}, {}
    out = []
    for r, p, v, iso, d, kind, a, b in pieces:
        if d <= 0:
            continue
        taken[id(r)] = taken.get(id(r), 0) + d
        if r["y0"] >= months_from:                         # what the place takes, month by month
            pm = placed_m.setdefault(id(r), np.zeros(12))
            for m in range(12):
                lo, hi = r["y0"] + m / 12, r["y0"] + (m + 1) / 12
                pm[m] += d * max(min(b, hi) - max(a, lo), 0) / (b - a)
        base = dict(vessel=v, iso=iso, y0=r["y0"], y1=r["y0"], t0=a, t1=b, deaths=float(d), cause=r["cause"],
                    name=r["name"], label=r["label"], id=r["id"], place=p["place"], plon=p["lon"], plat=p["lat"])
        if kind == "local":
            base.update(kind="spot", lon=p["lon"], lat=p["lat"], sigma=p["sigma"])
        else:
            base.update(kind="converge", to_lon=p["lon"], to_lat=p["lat"], to_sigma=p["sigma"])
        out.append(base)
    prof = covid_profile()
    rest = []
    for l in lifts:
        d = l["deaths"] - taken.get(id(l), 0)
        if d <= 1:
            continue
        w = prof.get(l["iso"], {}).get(l["y0"]) if months_from <= l["y0"] <= months_to else None
        if w is None or w.sum() <= 0:
            rest.append(dict(l, deaths=float(d)))
            continue
        # the country's months, less what its places already took in each month
        mw = w / w.sum() * l["deaths"] - placed_m.get(id(l), np.zeros(12))
        mw = np.maximum(mw, 0)
        if mw.sum() <= 0:
            mw = w.copy()
        mw = mw / mw.sum() * d
        for m in range(12):
            if mw[m] > 1:
                rest.append(dict(l, deaths=float(mw[m]), t0=l["y0"] + m / 12, t1=l["y0"] + (m + 1) / 12))
    return rest + out, dict(skipped=skipped, scaled=sorted(set(scaled)))


# ------------------------------------------------------------------ time helpers
def span(l):
    """(start, end) of a lift in fractional years, end exclusive."""
    return (l.get("t0", l["y0"]), l.get("t1", l["y1"] + 1))


def per_year(l, y_first, n):
    """A lift's deaths in each calendar year y_first .. y_first + n - 1."""
    a, b = span(l)
    out = np.zeros(n)
    for Y in range(int(math.floor(a)), int(math.ceil(b))):
        j = Y - y_first
        if 0 <= j < n:
            out[j] += l["deaths"] * max(min(b, Y + 1) - max(a, Y), 0) / (b - a)
    return out
