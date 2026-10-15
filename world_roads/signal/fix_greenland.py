"""Greenland's ice surface. Terrarium's zoom 5-9 tiles (and many zoom-10 tiles) carry the BEDROCK under
Greenland's ice sheet (often below sea level); zoom 11 has the ice surface (e.g. Summit Camp: zoom 9 says
259 m, zoom 11 says 3,255 m). Rebuild the Greenland box from zoom 11:
  * mosaic_z8max.npy / mosaic_global.npy: max-pooled zoom-11 values (the max with what is there)
  * peaks_hr.npy: Greenland's candidate summits replaced by zoom-11 ones (9x9 maxima, highest per 12 km cell)
  * verify.py uses zoom 11 inside the same box (GREENLAND)

    python fix_greenland.py          (zoom-11 rebuild)
    python fix_greenland.py clamp    (zoom 11 also has corrupt blocks above Greenland's 3,694 m maximum)
    python fix_greenland.py holes    (zoom 11 still has bedrock patches inside the ice sheet: fill them
                                      with the highest surrounding ice; verify.py does the same)
"""
import os, sys, json, time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import fetch_dem as F1
import fetch_dem2 as F2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GREENLAND = (59.0, 84.0, -75.0, -10.0)                 # lat0, lat1, lon0, lon1
Z = 11
CELL_KM = 12.0


def in_box(lat, lon):
    a, b, c, d = GREENLAND
    return (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)


def greenland_tiles():
    jobs = F2.land_tiles(Z, GREENLAND[0], GREENLAND[1])
    n = 2 ** Z
    out = []
    for x, y in jobs:
        lon0 = x / n * 360 - 180; lon1 = (x + 1) / n * 360 - 180
        if lon1 < -73.6 or lon0 > GREENLAND[3]:                   # Greenland's west tip is at 73.1 W
            continue
        lat_c = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 0.5) / n))))
        lon_c = (lon0 + lon1) / 2
        if lat_c < 67.3 and lon_c > -31.5:                       # Iceland, not Greenland
            continue
        if lat_c < 76.0 and lon_c < -60.5 and lat_c > 66.0:      # Baffin Island side of Baffin Bay
            continue
        out.append((x, y))
    return out


def work(job):
    x, y = job
    h = F2.get(Z, x, y)
    F1.Z = Z
    h = np.maximum(F1.despike(h, y), 0).astype(np.float32)
    blk = h.reshape(32, 8, 32, 8).max(axis=(1, 3))
    dil = cv2.dilate(h, np.ones((9, 9), np.uint8))
    yy, xx = np.nonzero((h >= dil) & (h > 15))
    n = 2 ** Z
    lon = (x + (xx + 0.5) / 256) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + (yy + 0.5) / 256) / n))))
    hv = h[yy, xx]
    pk = np.zeros((0, 3), np.float32)
    if len(hv):
        cy = np.floor(lat * 111.2 / CELL_KM).astype(np.int64)
        cx = np.floor(lon * 111.2 * np.cos(np.radians((cy + 0.5) * CELL_KM / 111.2)) / CELL_KM).astype(np.int64)
        key = cy * 100000 + cx
        o = np.lexsort((-hv, key))
        k = o[np.r_[True, key[o][1:] != key[o][:-1]]]
        pk = np.c_[lat[k], lon[k], hv[k]].astype(np.float32)
    return x, y, blk, pk


def main():
    t0 = time.time()
    jobs = greenland_tiles()
    print("zoom-11 Greenland tiles:", len(jobs), flush=True)
    mos = [np.load(os.path.join(DATA, f), mmap_mode="r+") for f in ("mosaic_z8max.npy", "mosaic_global.npy")]
    row0 = json.load(open(os.path.join(DATA, "mosaic_z8max_meta.json")))["row0_px"]
    peaks = []
    with ThreadPoolExecutor(24) as ex:
        for i, (x, y, blk, pk) in enumerate(ex.map(work, jobs)):
            r0 = y * 32 - row0
            for m in mos:
                a, b = max(r0, 0), min(r0 + 32, m.shape[0])
                if b > a:
                    cur = np.asarray(m[a:b, x * 32:(x + 1) * 32])
                    m[a:b, x * 32:(x + 1) * 32] = np.maximum(cur, np.round(blk[a - r0:b - r0]).astype(np.int16))
            if len(pk):
                peaks.append(pk)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    for m in mos:
        m.flush()
    new = np.concatenate(peaks)
    new = new[in_box(new[:, 0], new[:, 1])]
    old = np.load(os.path.join(DATA, "peaks_hr.npy"))
    keep = ~in_box(old[:, 0], old[:, 1])
    if not os.path.exists(os.path.join(DATA, "peaks_hr_pre_greenland.npy")):
        np.save(os.path.join(DATA, "peaks_hr_pre_greenland.npy"), old)
    np.save(os.path.join(DATA, "peaks_hr.npy"), np.concatenate([old[keep], new]))
    print(f"Greenland peaks: {(~keep).sum():,} old -> {len(new):,} new; {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ holes in the ice
INTERIOR_KM = 60.0                 # only deep inside the ice sheet, far from fjords and coastal mountains
HOLE_DROP = 600.0                  # a "hole": this far below the highest ice within ~10 km


def box_rows(meta):
    W = meta["width"]
    my = lambda lat: (0.5 - 0.5 * np.log((1 + np.sin(np.radians(lat))) / (1 - np.sin(np.radians(lat)))) / (2 * np.pi)) * W
    r0 = int(my(GREENLAND[1])) - meta["row0_px"]; r1 = int(my(GREENLAND[0])) - meta["row0_px"]
    c0 = int((-73.6 + 180) / 360 * W); c1 = int((GREENLAND[3] + 180) / 360 * W)
    return max(r0, 0), r1, c0, c1


def interior_mask(meta, r0, r1, c0, c1):
    W = meta["width"]
    land = np.zeros((r1 - r0, c1 - c0), np.uint8)
    gj = json.load(open(os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_land.geojson")))
    for f in gj["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            for k, ring in enumerate(poly):
                c = np.asarray(ring)
                if c[:, 1].max() < GREENLAND[0] or c[:, 0].max() < -80 or c[:, 0].min() > -5:
                    continue
                la = np.clip(c[:, 1], -85, 85)
                y = (0.5 - 0.5 * np.log((1 + np.sin(np.radians(la))) / (1 - np.sin(np.radians(la)))) / (2 * np.pi)) * W - meta["row0_px"] - r0
                x = (c[:, 0] + 180) / 360 * W - c0
                cv2.fillPoly(land, [np.c_[x * 4, y * 4].astype(np.int32)], 1 if k == 0 else 0, cv2.LINE_8, shift=2)
    dist = cv2.distanceTransform(land, cv2.DIST_L2, 5)
    rows = np.arange(r0, r1) + meta["row0_px"] + 0.5
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * rows / W))))
    km_px = 40075.0 / W * np.cos(np.radians(lat))
    return dist * km_px[:, None].astype(np.float32) >= INTERIOR_KM


def fill_holes(passes=((61, HOLE_DROP), (121, 300.0))):
    meta = json.load(open(os.path.join(DATA, "mosaic_z8max_meta.json")))
    r0, r1, c0, c1 = box_rows(meta)
    if os.path.exists(os.path.join(DATA, "greenland_interior.json")):
        inner = np.unpackbits(np.load(os.path.join(DATA, "greenland_interior.npy")), axis=1)[:, :c1 - c0].astype(bool)
    else:
        inner = interior_mask(meta, r0, r1, c0, c1)
        np.save(os.path.join(DATA, "greenland_interior.npy"), np.packbits(inner, axis=1))
        json.dump({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "cols": int(c1 - c0)}, open(os.path.join(DATA, "greenland_interior.json"), "w"))
    import shapely                                     # Greenland only (the box also holds Iceland, Baffin...)
    gj = json.load(open(os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_admin_0_countries.geojson")))
    grl = next(shapely.geometry.shape(f["geometry"]) for f in gj["features"] if f["properties"]["ADMIN"] == "Greenland")
    W = meta["width"]
    rr = np.arange(r0, r1, 16); cc = np.arange(c0, c1, 16)
    la = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (rr + meta["row0_px"] + 0.5) / W))))
    lo = (cc + 0.5) / W * 360 - 180
    LO, LA = np.meshgrid(lo, la)
    g = shapely.contains_xy(grl, LO.ravel(), LA.ravel()).reshape(LA.shape).astype(np.uint8)
    g = cv2.resize(g, (c1 - c0, r1 - r0), interpolation=cv2.INTER_NEAREST).astype(bool)
    inner = inner & g
    for kk, drop in passes:
        fill_pass(r0, r1, c0, c1, inner, kk, drop)


def fill_pass(r0, r1, c0, c1, inner, kk, drop):
    k = np.ones((kk, kk), np.uint8)
    for fn in ("mosaic_z8max.npy", "mosaic_global.npy"):
        mos = np.load(os.path.join(DATA, fn), mmap_mode="r+")
        sub = np.asarray(mos[r0:r1, c0:c1]).astype(np.float32)
        top = cv2.dilate(sub, k)
        hole = inner & (sub < top - drop) & (top > 1800)
        n = int(hole.sum())
        for it in range(6):
            if not hole.any():
                break
            fill = cv2.dilate(np.where(hole, 0, sub).astype(np.float32), np.ones((31 + 30 * it, 31 + 30 * it), np.uint8))
            ok = hole & (fill > 0)
            sub[ok] = fill[ok]
            hole &= ~ok
        mos[r0:r1, c0:c1] = np.round(sub).astype(np.int16)
        mos.flush()
        print(f"{fn}: filled {n:,} hole pixels in the ice sheet (window {kk}, drop {drop:.0f} m)", flush=True)


def restore_neighbours():
    """The Greenland box also holds Iceland and parts of Baffin and Ellesmere: put their zoom-10 terrain
    back exactly as fetch_dem2.py made it (the ice-sheet hole fill must only touch Greenland)."""
    import shapely
    gj = json.load(open(os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_admin_0_countries.geojson")))
    grl = next(shapely.geometry.shape(f["geometry"]) for f in gj["features"] if f["properties"]["ADMIN"] == "Greenland")
    row0 = json.load(open(os.path.join(DATA, "mosaic_z8max_meta.json")))["row0_px"]
    n = 2 ** 10
    zone = grl.buffer(0.3)
    shapely.prepare(zone)
    cand = [(x, y) for x, y in F2.land_tiles(10, GREENLAND[0], GREENLAND[1]) if GREENLAND[2] <= (x + 0.5) / n * 360 - 180 <= GREENLAND[3]]
    lo = np.array([(x + 0.5) / n * 360 - 180 for x, y in cand])
    la = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (np.array([y for x, y in cand]) + 0.5) / n))))
    inside = shapely.contains_xy(zone, lo, la)
    jobs = [c for c, i in zip(cand, inside) if not i]
    print("zoom-10 tiles outside Greenland in the box:", len(jobs), flush=True)
    mos = [np.load(os.path.join(DATA, f), mmap_mode="r+") for f in ("mosaic_z8max.npy", "mosaic_global.npy")]

    def w(j):
        x, y = j
        F1.Z = 10
        h = np.maximum(F1.despike(F2.get(10, x, y), y), 0).astype(np.float32)
        return x, y, h.reshape(64, 4, 64, 4).max(axis=(1, 3))
    with ThreadPoolExecutor(24) as ex:
        for x, y, blk in ex.map(w, jobs):
            r0 = y * 64 - row0
            for m in mos:
                m[r0:r0 + 64, x * 64:(x + 1) * 64] = np.round(blk).astype(np.int16)
    for m in mos:
        m.flush()
    print("restored", flush=True)


GREENLAND_MAX = 3700.0              # Gunnbjørn Fjeld, Greenland's highest point, is 3,694 m


def clamp_tile(h):
    """Corrupt blocks in Greenland's zoom-11 data (up to 6,328 m next to Gunnbjørn Fjeld): anything above
    Greenland's highest point becomes the mean of its valid neighbours."""
    bad = h > GREENLAND_MAX
    if not bad.any():
        return h
    v = np.where(bad, 0, h).astype(np.float32)
    w = cv2.blur((~bad).astype(np.float32), (15, 15))
    m = cv2.blur(v, (15, 15)) / np.maximum(w, 1e-3)
    out = h.copy()
    out[bad] = np.minimum(m[bad], GREENLAND_MAX)
    return out


def clamp():
    import shapely
    meta = json.load(open(os.path.join(DATA, "mosaic_z8max_meta.json")))
    r0, r1, c0, c1 = box_rows(meta)
    for fn in ("mosaic_z8max.npy", "mosaic_global.npy"):
        mos = np.load(os.path.join(DATA, fn), mmap_mode="r+")
        sub = np.asarray(mos[r0:r1, c0:c1]).astype(np.float32)
        n = int((sub > GREENLAND_MAX).sum())
        mos[r0:r1, c0:c1] = np.round(clamp_tile(sub)).astype(np.int16)
        mos.flush()
        print(f"{fn}: {n} pixels above {GREENLAND_MAX:.0f} m replaced", flush=True)
    gj = json.load(open(os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_admin_0_countries.geojson")))
    grl = next(shapely.geometry.shape(f["geometry"]) for f in gj["features"] if f["properties"]["ADMIN"] == "Greenland")
    pk = np.load(os.path.join(DATA, "peaks_hr.npy"))
    over = (pk[:, 2] > GREENLAND_MAX) & shapely.contains_xy(grl.buffer(0.2), pk[:, 1], pk[:, 0])
    # within 3 km of Gunnbjorn Fjeld (68.919 N, 29.898 W) the summit is real: it gets its true 3,694 m
    near = np.hypot((pk[:, 0] - 68.919) * 111.2, (pk[:, 1] + 29.898) * 111.2 * np.cos(np.radians(68.9))) < 3.0
    pk[over & near, 2] = 3694.0
    bad = over & ~near
    np.save(os.path.join(DATA, "peaks_hr.npy"), pk[~bad])
    print(f"capped {int((over & near).sum())} Gunnbjorn Fjeld summits at 3,694 m, dropped {int(bad.sum())} impossible ones", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clamp":
        clamp()
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "holes":
        fill_holes()
    elif len(sys.argv) > 1 and sys.argv[1] == "holes2":
        fill_holes(passes=((121, 300.0),))
    elif len(sys.argv) > 1 and sys.argv[1] == "neighbours":
        restore_neighbours()
    else:
        main()
