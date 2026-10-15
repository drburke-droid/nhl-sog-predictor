"""Better terrain for the search: Terrarium zoom 10 north of 56N (the only zoom where Greenland, arctic
Canada and Siberia come from good sources) and zoom 9 south of it, despiked, reduced to
  * a zoom-8-resolution grid holding each cell's MAXIMUM (so no narrow ridge is smoothed away)
  * the highest summit in every ~12 km cell (candidate fire sites)
    python fetch_dem2.py
"""
import os, json, time, math
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import fetch_dem as F1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NE = os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_land.geojson")
SPLIT = 56.0
LAT_MIN, LAT_MAX = 18.0, 84.0
CELL_KM = 12.0
ZM = 8                                   # mosaic resolution


def merc_y(lat):
    s = math.sin(math.radians(lat))
    return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


def land_tiles(z, la0, la1):
    """Tiles at zoom z touching land (rasterized Natural Earth, grown by one tile)."""
    n = 2 ** z
    img = np.zeros((n, n), np.uint8)
    gj = json.load(open(NE))
    for f in gj["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            c = np.asarray(poly[0])
            la = np.clip(c[:, 1], -85, 85)
            s = np.sin(np.radians(la))
            y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
            x = (c[:, 0] + 180) / 360 * n
            cv2.fillPoly(img, [np.c_[x * 16, y * 16].astype(np.int32)], 1, cv2.LINE_8, shift=4)
            cv2.polylines(img, [np.c_[x * 16, y * 16].astype(np.int32)], True, 1, 1, cv2.LINE_8, shift=4)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    y0, y1 = int(merc_y(la1) * n), int(merc_y(la0) * n)
    ys, xs = np.nonzero(img[y0:y1 + 1])
    return [(int(x), int(y + y0)) for x, y in zip(xs, ys)]


def get(z, x, y):
    for attempt in range(8):
        for u in F1.URLS:
            try:
                import urllib.request
                raw = urllib.request.urlopen(u.format(z=z, x=x, y=y), timeout=30).read()
                im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
                return im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
            except Exception:
                time.sleep(1.5 ** attempt)
    raise RuntimeError(f"tile {z}/{x}/{y}")


def despike(h, z, y):
    F1.Z = z
    return F1.despike(h, y)


def main():
    t0 = time.time()
    jobs = [(10, x, y) for x, y in land_tiles(10, SPLIT, LAT_MAX)] + [(9, x, y) for x, y in land_tiles(9, LAT_MIN, SPLIT)]
    print("tiles:", len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    nM = 2 ** ZM * 256
    row0 = int(merc_y(LAT_MAX) * nM)
    rows = int(merc_y(LAT_MIN) * nM) - row0 + 1
    mosaic = np.zeros((rows, nM), np.int16)
    peaks = []
    kern = np.ones((9, 9), np.uint8)

    def work(job):
        z, x, y = job
        h = np.maximum(despike(get(z, x, y), z, y), 0).astype(np.float32)
        f = 2 ** (z - ZM)
        blk = h.reshape(256 // f, f, 256 // f, f).max(axis=(1, 3))
        # candidate summits: 9x9 local maxima, highest per ~12 km cell
        dil = cv2.dilate(h, kern)
        yy, xx = np.nonzero((h >= dil) & (h > 15))
        n = 2 ** z
        lon = (x + (xx + 0.5) / 256) / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + (yy + 0.5) / 256) / n))))
        hv = h[yy, xx]
        if len(hv):
            cy = np.floor(lat * 111.2 / CELL_KM).astype(np.int64)
            cx = np.floor(lon * 111.2 * np.cos(np.radians((cy + 0.5) * CELL_KM / 111.2)) / CELL_KM).astype(np.int64)
            key = cy * 100000 + cx
            o = np.lexsort((-hv, key))
            k = o[np.r_[True, key[o][1:] != key[o][:-1]]]
            pk = np.c_[lat[k], lon[k], hv[k]].astype(np.float32)
        else:
            pk = np.zeros((0, 3), np.float32)
        return z, x, y, blk, pk

    with ThreadPoolExecutor(24) as ex:
        for i, (z, x, y, blk, pk) in enumerate(ex.map(work, jobs)):
            f = 2 ** (z - ZM)
            s = 256 // f
            r0 = y * s - row0
            c0 = x * s
            r_lo, r_hi = max(r0, 0), min(r0 + s, rows)
            if r_hi > r_lo:
                cur = mosaic[r_lo:r_hi, c0:c0 + s]
                np.maximum(cur, np.round(blk[r_lo - r0:r_hi - r0]).astype(np.int16), out=cur)
            if len(pk):
                peaks.append(pk)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    peaks = np.concatenate(peaks)
    np.save(os.path.join(DATA, "mosaic_z8max.npy"), mosaic)
    np.save(os.path.join(DATA, "peaks_hr.npy"), peaks)
    json.dump({"z": ZM, "row0_px": row0, "width": nM, "rows": rows, "kind": "max"},
              open(os.path.join(DATA, "mosaic_z8max_meta.json"), "w"))
    print("peaks:", len(peaks), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
