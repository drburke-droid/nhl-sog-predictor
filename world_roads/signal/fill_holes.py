"""Terrarium's zoom-10 tiles are blank (0 m) over large parts of Greenland's ice sheet (and possibly other
Arctic land). Fill those holes from zoom-8 tiles, which do have the ice sheet:
  * data/z8fill/   cached zoom-8 tiles (despiked)
  * mosaic_z8max.npy and mosaic_global.npy: land pixels at <= 1 m where zoom 8 says > 30 m take the zoom-8 value
  * data/fill_peaks.npy: candidate summits (9x9 maxima, highest per 12 km cell) inside the filled areas

    python fill_holes.py
"""
import os, json, math, time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import fetch_dem as F1
import fetch_dem2 as F2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TD = os.path.join(DATA, "z8fill")
Z = 8


def tile8(x, y):
    p = os.path.join(TD, f"{x}_{y}.npy")
    if os.path.exists(p):
        return np.load(p)
    h = F2.get(Z, x, y)
    F1.Z = Z
    h = np.maximum(F1.despike(h, y), 0).astype(np.float32)
    np.save(p, h)
    return h


def main():
    os.makedirs(TD, exist_ok=True)
    t0 = time.time()
    jobs = F2.land_tiles(Z, 56.0, 84.0)
    print("zoom-8 land tiles 56-84N:", len(jobs), flush=True)
    with ThreadPoolExecutor(24) as ex:
        tiles = dict(zip(jobs, ex.map(lambda j: tile8(*j), jobs)))
    print(f"fetched {time.time() - t0:.0f}s", flush=True)
    peaks = []
    total = 0
    for fn, mf in (("mosaic_z8max.npy", "mosaic_z8max_meta.json"), ("mosaic_global.npy", "mosaic_global_meta.json")):
        meta = json.load(open(os.path.join(DATA, mf)))
        mos = np.load(os.path.join(DATA, fn), mmap_mode="r+")
        row0 = meta["row0_px"]
        n = 0
        for (x, y), h8 in tiles.items():
            r0 = y * 256 - row0
            if r0 + 256 <= 0 or r0 >= mos.shape[0]:
                continue
            a, b = max(r0, 0), min(r0 + 256, mos.shape[0])
            cur = np.asarray(mos[a:b, x * 256:(x + 1) * 256])
            src = h8[a - r0:b - r0]
            hole = (cur <= 1) & (src > 30)
            if hole.any():
                # grow the hole a little so its rim (half-blank z10 cells) is filled too
                hole = (cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (src > cur)
                cur = np.where(hole, np.round(src).astype(np.int16), cur)
                mos[a:b, x * 256:(x + 1) * 256] = cur
                n += int(hole.sum())
                if fn == "mosaic_z8max.npy":
                    hh = np.where(hole, src, 0).astype(np.float32)
                    dil = cv2.dilate(hh, np.ones((9, 9), np.uint8))
                    yy, xx = np.nonzero((hh >= dil) & (hh > 15))
                    if len(yy):
                        lon = (x + (xx + 0.5) / 256) / 2 ** Z * 360 - 180
                        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + (yy + 0.5) / 256) / 2 ** Z))))
                        peaks.append(np.c_[lat, lon, hh[yy, xx]].astype(np.float32))
        mos.flush()
        print(f"{fn}: filled {n:,} pixels {time.time() - t0:.0f}s", flush=True)
        total += n
    pk = np.concatenate(peaks) if peaks else np.zeros((0, 3), np.float32)
    np.save(os.path.join(DATA, "fill_peaks.npy"), pk)
    print("fill peaks:", len(pk))


if __name__ == "__main__":
    main()
