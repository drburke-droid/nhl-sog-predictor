"""Baer's law on the named main rivers of all 7,311 systems longer than ~100 km: every OSM line carrying one
of their Wikidata ids, at least 10 km long, smoothed at 1 and 2 km. A larger southern sample than the 1,000
longest main stems.

    python baer_named.py      -> data/baer_named.npy (one row per line)
"""
import os, sys, glob, json, math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RIV = os.path.join(HERE, "..", "rivers", "data")
STEP = 50.0
SCALES = (1000.0, 2000.0)
KMIN = 1 / 50000.0
WD = None


def init(wd):
    global WD
    WD = wd


def one(path):
    z = np.load(path, allow_pickle=True)
    xy, off, wd = z["xy"], z["off"], z["wikidata"]
    rows = []
    for i in range(len(off) - 1):
        w = wd[i]
        if not w or w not in WD:
            continue
        a, b = off[i], off[i + 1]
        if b - a < 5:
            continue
        lon, lat = xy[a:b, 0] / 1e7, xy[a:b, 1] / 1e7
        x = (lon - lon.mean()) * np.cos(np.radians(lat)) * 111320.0
        y = (lat - lat.mean()) * 110540.0
        ds = np.hypot(np.diff(x), np.diff(y))
        L = ds.sum()
        if L < 10000:
            continue
        s = np.r_[0, np.cumsum(ds)]
        g = np.arange(0, L, STEP)
        h = np.unwrap(np.arctan2(np.diff(np.interp(g, s, y)), np.diff(np.interp(g, s, x))))
        row = [lat.mean(), lon.mean(), L, WD[w]]
        for sc in SCALES:
            k = np.diff(gaussian_filter1d(h, sc / STEP, mode="nearest")) / STEP
            row += [(k > KMIN).sum() * STEP, (k < -KMIN).sum() * STEP]
        rows.append(row)
    return np.array(rows, np.float64).reshape(-1, 4 + 2 * len(SCALES))


if __name__ == "__main__":
    systems = json.load(open(os.path.join(RIV, "all_rivers.json")))
    wd = {r["wikidata"]: k for k, r in enumerate(systems) if r.get("wikidata")}
    parts = sorted(glob.glob(os.path.join(RIV, "full", "part-*.npz")))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3, initializer=init, initargs=(wd,)) as ex:
        R = np.vstack(list(ex.map(one, parts)))
    np.save(os.path.join(HERE, "data", "baer_named.npy"), R)
    print(R.shape, "lines,", len(np.unique(R[:, 3])), "rivers")
