"""The whole inhabited world for the most-countries chain:
  * data/mosaic_global.npy   the 18-84N max-pooled grid + the 60S-18N one (python fetch_dem2.py south)
  * data/world_nodes.npz     candidate summits (highest per ~12 km cell) with the country each one is in
  * data/countries.json      the 195 countries (193 UN members + Vatican + Palestine) and their names

Tiny countries with no candidate summit get their own highest point (zoom-13 Terrarium inside the border).

    python world.py
"""
import os, json, math, time, urllib.request
import numpy as np
import cv2
import shapely
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
import chain as CH
import fetch_dem as F1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NE = os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_admin_0_countries.geojson")
NOT_COUNTRIES = {"Antarctica", "Bajo Nuevo Bank (Petrel Is.)", "Bir Tawil", "Brazilian Island", "Cyprus No Mans Area",
                 "Kosovo", "Northern Cyprus", "Scarborough Reef", "Serranilla Bank", "Somaliland",
                 "Southern Patagonian Ice Field", "Spratly Islands", "Taiwan", "Western Sahara"}
SHORT = {"United States of America": "USA", "United Kingdom": "UK", "Republic of Serbia": "Serbia",
         "Democratic Republic of the Congo": "DR Congo", "Republic of the Congo": "Congo",
         "United Republic of Tanzania": "Tanzania", "Bosnia and Herzegovina": "Bosnia & Herzegovina",
         "Central African Republic": "Central African Rep.", "eSwatini": "Eswatini", "The Bahamas": "Bahamas",
         "East Timor": "Timor-Leste", "Ivory Coast": "Côte d'Ivoire", "Saint Vincent and the Grenadines": "St Vincent & Grenadines",
         "Saint Kitts and Nevis": "St Kitts & Nevis", "Saint Lucia": "St Lucia", "Antigua and Barbuda": "Antigua & Barbuda",
         "Trinidad and Tobago": "Trinidad & Tobago", "São Tomé and Principe": "São Tomé & Príncipe",
         "Federated States of Micronesia": "Micronesia", "Papua New Guinea": "Papua New Guinea"}
Z_TINY = 13
# Natural Earth's 10 m outline of Monaco takes in French slopes above it. Its high point (Chemin des
# Révoires, 161 m) sits right on the French border, so Monaco's fire goes on the Rock of Monaco instead
# (zoom-13 height); Vatican Hill is used at its zoom-13 height.
TINY_OVERRIDE = {"Monaco": (60.3, 43.7313, 7.4211), "Vatican": (60.6, 41.9029, 12.4539)}


def countries():
    gj = json.load(open(NE))
    names, geoms, counts = [], [], []
    for f in gj["features"]:
        p = f["properties"]
        is_c = (p["SOVEREIGNT"] == p["ADMIN"] and p["ADMIN"] not in NOT_COUNTRIES) or p["ADMIN"] == "Palestine"
        names.append(SHORT.get(p["ADMIN"], p["ADMIN"]))
        geoms.append(shapely.geometry.shape(f["geometry"]))
        counts.append(is_c)
    return names, geoms, np.array(counts)


def merge_mosaics():
    out = os.path.join(DATA, "mosaic_global.npy")
    mn = json.load(open(os.path.join(DATA, "mosaic_z8max_meta.json")))
    ms = json.load(open(os.path.join(DATA, "mosaic_z8max_south_meta.json")))
    if os.path.exists(os.path.join(DATA, "mosaic_global_meta.json")):       # written last, once complete
        return
    north = np.load(os.path.join(DATA, "mosaic_z8max.npy"), mmap_mode="r")
    south = np.load(os.path.join(DATA, "mosaic_z8max_south.npy"), mmap_mode="r")
    r0n, r0s = mn["row0_px"], ms["row0_px"]
    rows = r0s + south.shape[0] - r0n
    g = np.lib.format.open_memmap(out, "w+", np.int16, (rows, mn["width"]))
    for a in range(0, north.shape[0], 2048):
        blk = np.asarray(north[a:a + 2048])
        g[a:a + len(blk)] = blk
    off = r0s - r0n
    for a in range(0, south.shape[0], 2048):
        blk = np.asarray(south[a:a + 2048])
        np.maximum(g[off + a:off + a + len(blk)], blk, out=g[off + a:off + a + len(blk)])
    g.flush()
    json.dump({"z": mn["z"], "row0_px": r0n, "width": mn["width"], "rows": int(rows), "kind": "max"},
              open(os.path.join(DATA, "mosaic_global_meta.json"), "w"))


def tiny_summit(geom):
    """Highest zoom-13 pixel inside a small country's border."""
    n = 2 ** Z_TINY
    lo0, la0, lo1, la1 = geom.bounds
    def txy(lat, lon):
        s = math.sin(math.radians(lat))
        return (lon + 180) / 360 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    x0, y0 = txy(la1, lo0); x1, y1 = txy(la0, lo1)
    best = (-1e9, 0, 0)
    for ty in range(int(y0), int(y1) + 1):
        for tx in range(int(x0), int(x1) + 1):
            raw = urllib.request.urlopen(F1.URLS[0].format(z=Z_TINY, x=tx, y=ty), timeout=30).read()
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
            h = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
            F1.Z = Z_TINY
            h = F1.despike(h, ty)
            yy, xx = np.mgrid[0:256, 0:256]
            lon = (tx + (xx + 0.5) / 256) / n * 360 - 180
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (ty + (yy + 0.5) / 256) / n))))
            inside = shapely.contains_xy(geom, lon.ravel(), lat.ravel()).reshape(256, 256)
            if inside.any():
                hm = np.where(inside, h, -1e9)
                k = np.unravel_index(np.argmax(hm), hm.shape)
                if hm[k] > best[0]:
                    best = (float(hm[k]), float(lat[k]), float(lon[k]))
    return best


def main():
    t0 = time.time()
    merge_mosaics()
    print(f"mosaic merged {time.time() - t0:.0f}s", flush=True)
    names, geoms, is_c = countries()
    print("countries:", int(is_c.sum()), flush=True)
    pk = np.concatenate([np.load(os.path.join(DATA, "peaks_hr.npy")), np.load(os.path.join(DATA, "peaks_hr_south.npy"))])
    lat, lon, h = pk[:, 0].astype(np.float64), pk[:, 1].astype(np.float64), pk[:, 2].astype(np.float64)
    inv = np.array(json.load(open(os.path.join(DATA, "invalid_nodes.json"))))
    if len(inv):
        t = cKDTree(np.c_[lat, lon])
        drop = np.unique(np.concatenate([t.query_ball_point(p, 2e-4) for p in inv]).astype(int))
        keep = np.setdiff1d(np.arange(len(lat)), drop)
        lat, lon, h = lat[keep], lon[keep], h[keep]
    ok = (h < 9000) & (lat > -60) & (lat < 84)
    lat, lon, h = lat[ok], lon[ok], h[ok]
    cy = np.floor(lat * 111.2 / CH.CELL_KM).astype(np.int64)
    cx = np.floor(lon * 111.2 * np.cos(np.radians((cy + 0.5) * CH.CELL_KM / 111.2)) / CH.CELL_KM).astype(np.int64)
    key = cy * 100000 + cx
    o = np.lexsort((-h, key))
    keep = o[np.r_[True, key[o][1:] != key[o][:-1]]]
    lat, lon, h = lat[keep], lon[keep], h[keep]
    print(f"candidates {len(lat):,}  {time.time() - t0:.0f}s", flush=True)
    # which country is each summit in (-1: sea / no country)
    tree = STRtree(geoms)
    pi, gi = tree.query(shapely.points(lon, lat), predicate="within")
    cid = np.full(len(lat), -1, np.int32)
    cid[pi] = gi
    have = set(np.unique(cid[cid >= 0]).tolist())
    extra = []
    for g in np.nonzero(is_c)[0]:
        if g not in have:
            hm, la, lo = TINY_OVERRIDE.get(names[g]) or tiny_summit(geoms[g])
            print(f"  {names[g]}: no candidate, highest point {hm:.0f} m at {la:.4f},{lo:.4f}", flush=True)
            if hm > -1e8:
                extra.append((la, lo, max(hm, 1.0), g))
    if extra:
        e = np.array(extra)
        lat, lon, h = np.r_[lat, e[:, 0]], np.r_[lon, e[:, 1]], np.r_[h, e[:, 2]]
        cid = np.r_[cid, e[:, 3].astype(np.int32)]
    np.savez(os.path.join(DATA, "world_nodes.npz"), lat=lat, lon=lon, h=h, cid=cid)
    json.dump({"names": names, "counts": is_c.tolist()}, open(os.path.join(DATA, "countries.json"), "w"))
    print(f"nodes {len(lat):,}, in a country {(cid >= 0).sum():,}, countries with a summit "
          f"{len(set(cid[cid >= 0].tolist()) & set(np.nonzero(is_c)[0].tolist()))}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
