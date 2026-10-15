"""Download Terrarium elevation tiles (zoom 8) for all land between 18N and 84N and reduce them to
  * a zoom-7 elevation mosaic (2x2 mean, metres, sea = 0) for line-of-sight checks
  * every local summit (9x9 local maximum at zoom 8) as a candidate fire site
Same elevation source as the VSL Sightline tool.     python fetch_dem.py
"""
import os, json, time, math, urllib.request
import numpy as np
import cv2
import shapely
from shapely.strtree import STRtree
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NE = os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_land.geojson")
URLS = ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"]
Z = 8
LAT_MIN, LAT_MAX = 18.0, 84.0


def tile_y(lat, z):
    s = math.sin(math.radians(lat))
    return int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 2 ** z)


def tile_lat(y, z):
    n = math.pi - 2 * math.pi * y / 2 ** z
    return math.degrees(math.atan(math.sinh(n)))


def land_tiles():
    gj = json.load(open(NE))
    polys = [shapely.geometry.shape(f["geometry"]) for f in gj["features"]]
    tree = STRtree(polys)
    out = []
    n = 2 ** Z
    for y in range(tile_y(LAT_MAX, Z), tile_y(LAT_MIN, Z) + 1):
        la1, la0 = tile_lat(y, Z), tile_lat(y + 1, Z)
        for x in range(n):
            lo0, lo1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
            box = shapely.box(lo0 - 0.05, la0 - 0.05, lo1 + 0.05, la1 + 0.05)
            if len(tree.query(box, predicate="intersects")):
                out.append((x, y))
    return out


def despike(h, y):
    """Clean Terrarium faults, extending the VSL Sightline rule (max(150 m, one pixel width)):
    isolated pits and spikes (beyond lim below/above every neighbour), and row/column streaks (beyond
    lim above all six pixels in the rows above and below, or the columns either side). Pits are
    repaired here too because these faults come as spike+pit pairs and the pit shields the spike.
    Faults take their neighbours' median; repeated until clean."""
    lat = tile_lat(y + 0.5, Z)
    lim = max(150.0, 40075016.7 * math.cos(math.radians(lat)) / (256 * 2 ** Z))
    h = h.astype(np.float32)
    nod = h > 9000                                      # Terrarium no-data (white pixels)
    if nod.any():
        h = h.copy(); h[nod] = np.median(h[~nod]) if (~nod).any() else 0
    for _ in range(6):
        pad = np.pad(h, 1, mode="edge")
        nb = np.stack([pad[1 + dy:257 + dy, 1 + dx:257 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
        rows_nb = nb[[0, 1, 2, 5, 6, 7]]
        cols_nb = nb[[0, 3, 5, 2, 4, 7]]
        fault = (h - lim > nb.max(0)) | (h + lim < nb.min(0)) | (h - lim > rows_nb.max(0)) | (h - lim > cols_nb.max(0))
        if not fault.any():
            break
        h = h.copy()
        h[fault] = np.median(nb[:, fault], axis=0)
    # anything still more than 1,500 m off the median of its 5x5 block is a fault too
    med = cv2.medianBlur(h, 5)
    bad = np.abs(h - med) > 1500
    if bad.any():
        h = h.copy()
        h[bad] = med[bad]
    return h


def get(x, y):
    for attempt in range(6):
        for u in URLS:
            try:
                raw = urllib.request.urlopen(u.format(z=Z, x=x, y=y), timeout=30).read()
                im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
                h = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
                return np.maximum(despike(h, y), 0)
            except Exception:
                time.sleep(1.5 ** attempt)
    raise RuntimeError(f"tile {x},{y}")


def main():
    os.makedirs(DATA, exist_ok=True)
    tiles = land_tiles()
    print("land tiles:", len(tiles), flush=True)
    y_top = tile_y(LAT_MAX, Z) // 2 * 2
    y_bot = tile_y(LAT_MIN, Z) // 2 * 2 + 2
    W7 = 2 ** (Z - 1) * 256
    rows = (y_bot - y_top) // 2 * 256
    mosaic = np.zeros((rows, W7), np.int16)
    peaks = []
    kern = np.ones((9, 9), np.uint8)

    def work(t):
        x, y = t
        h = get(x, y)
        half = cv2.resize(h, (128, 128), interpolation=cv2.INTER_AREA)
        dil = cv2.dilate(h, kern)
        yy, xx = np.nonzero((h >= dil) & (h > 15))
        if len(yy) > 1:                                  # flat summit plateaus: one point per 9x9 block
            key = (yy // 9) * 64 + xx // 9
            _, first = np.unique(key, return_index=True)
            yy, xx = yy[first], xx[first]
        n = 2 ** Z
        lon = (x + (xx + 0.5) / 256) / n * 360 - 180
        ym = (y + (yy + 0.5) / 256) / n
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ym))))
        return x, y, half, np.c_[lat, lon, h[yy, xx]].astype(np.float32)

    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        for i, (x, y, half, pk) in enumerate(ex.map(work, tiles)):
            r0 = (y - y_top) // 2 * 256 + (y % 2) * 128
            c0 = x // 2 * 256 + (x % 2) * 128
            mosaic[r0:r0 + 128, c0:c0 + 128] = np.round(half).astype(np.int16)
            if len(pk):
                peaks.append(pk)
            if i % 1000 == 0:
                print(f"  {i}/{len(tiles)} {time.time() - t0:.0f}s", flush=True)
    peaks = np.concatenate(peaks)
    np.save(os.path.join(DATA, "z7_mosaic.npy"), mosaic)
    np.save(os.path.join(DATA, "peaks_z8.npy"), peaks)
    json.dump({"z": 7, "row0_px": (y_top // 2) * 256, "width": W7, "rows": rows}, open(os.path.join(DATA, "mosaic_meta.json"), "w"))
    print("peaks:", len(peaks), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
