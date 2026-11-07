"""Each city's population through history, so the city mountains rise (and fall) when the cities did.

Sources:
  1950-2025  UN World Urbanization Prospects 2018, File 12 (urban agglomerations, every 5 years):
             data/wup2018_cities_1m_series.csv
  AD 1-1949  Tertius Chandler, Four Thousand Years of Urban Growth (1987), geocoded by Reba, Reitsma & Seto,
             "Spatializing 6,000 years of global urbanization", Scientific Data 3:160034 (2016), CC BY 4.0
             (figshare 10.6084/m9.figshare.2059494): data/chandler_city_history.csv (the rows matched here)
Chandler's figures are used as published before 1950 and joined log-linearly to the UN's from 1950 (the
UN's agglomerations are drawn a little wider, so some cities step up faster between Chandler's last figure
and 1950). Before a city's first figure it shrinks back along the world's urbanization curve
(story.urban_frac); a city with no Chandler figure does so from 1950. Every series ends at the city's 2025
size in story.MEGACITIES / CITY_HILLS.

    years, M = histories(cities)       # cities: [(name, lon, lat, millions_2025)]; M[i] = millions at years
"""
import csv, os
import numpy as np
import story as S

HERE = os.path.dirname(os.path.abspath(__file__))
WUP = os.path.join(HERE, "data", "wup2018_cities_1m_series.csv")
CHANDLER = os.path.join(HERE, "data", "chandler_city_history.csv")
GRID = np.unique(np.r_[np.arange(1, 1800, 25), np.arange(1800, 1950, 5), np.arange(1950, 2026, 1)]).astype(float)


def _near(lat, lon, la, lo):
    return np.hypot(lat - la, (lon - lo) * np.cos(np.radians(la)))


def _load_wup():
    rows = list(csv.DictReader(open(WUP, encoding="utf-8")))
    yrs = [int(k[3:7]) for k in rows[0] if k.startswith("pop")]
    lat = np.array([float(r["lat"]) for r in rows]); lon = np.array([float(r["lon"]) for r in rows])
    P = np.array([[float(r[f"pop{y}_thousands"]) / 1000 for y in yrs] for r in rows])
    return rows, np.array(yrs, float), lat, lon, P


def _load_chandler():
    """{(lat, lon, name): [(year, millions)]} for AD years before 1950."""
    out = []
    for r in csv.DictReader(open(CHANDLER, encoding="utf-8")):
        out.append((float(r["lat"]), float(r["lon"]), r["city"], int(r["year"]), float(r["population"]) / 1e6))
    cities = {}
    for la, lo, n, y, p in out:
        cities.setdefault((la, lo, n), []).append((y, p))
    return cities


def histories(cities):
    wrows, wy, wlat, wlon, WP = _load_wup()
    ch = _load_chandler()
    ckeys = list(ch)
    clat = np.array([k[0] for k in ckeys]); clon = np.array([k[1] for k in ckeys])
    uf = np.array([S.urban_frac(y) for y in GRID])
    M = np.zeros((len(cities), len(GRID)))
    info = []
    for i, (name, lo, la, mil) in enumerate(cities):
        d = _near(wlat, wlon, la, lo); j = int(np.argmin(d))
        pts = {}
        if d[j] < 0.5:
            w = WP[j] * (mil / max(WP[j][list(wy).index(2025)], 1e-6))      # end at our 2025 size
            pts.update({float(y): float(v) for y, v in zip(wy, w) if y <= 2025})
            wsrc = wrows[j]["name"]
        else:
            pts[2025.0] = mil; wsrc = None
        csrc = None
        if len(ckeys):
            dc = _near(clat, clon, la, lo); k = int(np.argmin(dc))
            if dc[k] < 0.35:
                hist = sorted((y, p) for y, p in ch[ckeys[k]] if 1 <= y < 1950 and p > 0)
                if hist:
                    csrc = ckeys[k][2]
                    pts.update({float(y): p for y, p in hist})     # as published; joins the UN's series at 1950
        ys = np.array(sorted(pts)); ps = np.array([pts[y] for y in ys])
        lg = np.interp(GRID, ys, np.log(np.maximum(ps, 1e-4)))
        before = GRID < ys[0]                                  # before the first figure: the world's urban curve
        u0 = S.urban_frac(ys[0])
        lg[before] = np.log(max(ps[0], 1e-4) * np.maximum(uf[before], 1e-9) / max(u0, 1e-9))
        M[i] = np.exp(lg)
        info.append(dict(name=name, wup=wsrc, chandler=csrc, first=float(ys[0])))
    return GRID, M, info


def at(years, M, year):
    """Millions per city at a calendar year (log-linear between grid points)."""
    year = float(np.clip(year, years[0], years[-1]))
    k = int(np.clip(np.searchsorted(years, year) - 1, 0, len(years) - 2))
    w = (year - years[k]) / (years[k + 1] - years[k])
    return np.exp(np.log(np.maximum(M[:, k], 1e-9)) * (1 - w) + np.log(np.maximum(M[:, k + 1], 1e-9)) * w)
