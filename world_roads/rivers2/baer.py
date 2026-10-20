"""Baer's law, tested on every river line in OpenStreetMap.

Baer (1860), and Einstein (1926) in the paper on meanders: the Earth's rotation pushes a river's flow toward its
right bank in the northern hemisphere (left in the southern), so right banks erode more. The right bank is the
outer bank of a left-hand bend, so left bends should grow a little more, and a little more of a northern
river's length should curve left; the reverse in the south. The equator flips the sign, which no quirk of
mapping can imitate.

Each OSM river line (drawn downstream, by OSM convention) is resampled every 25 m in a local plane; its heading
is smoothed at a meander scale (250 m and 1 km); the length curving left and right is tallied.

    python baer.py            -> data/baer/part-*.npz, one row per line
"""
import os, sys, glob, math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "rivers", "data", "full")
OUT = os.path.join(HERE, "data", "baer")
STEP = 25.0                                    # m
SCALES = (250.0, 1000.0)                       # smoothing (Gaussian sigma), m
KMIN = 1 / 20000.0                             # curvature below 1/(20 km) counts as straight
COLS = ["lat", "lon", "len_m", "net_deg"] + [f"{k}_{int(s)}" for s in SCALES for k in ("left_m", "right_m", "turnL", "turnR")]


def one(path):
    out = os.path.join(OUT, os.path.basename(path))
    if os.path.exists(out):
        return out
    z = np.load(path, allow_pickle=True)
    xy, off = z["xy"], z["off"]
    rows = []
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        if b - a < 3:
            continue
        lon = xy[a:b, 0] / 1e7
        lat = xy[a:b, 1] / 1e7
        lat0, lon0 = lat.mean(), lon.mean()
        x = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
        y = (lat - lat0) * 110540.0
        ds = np.hypot(np.diff(x), np.diff(y))
        keep = np.r_[True, ds > 0.01]
        x, y = x[keep], y[keep]
        ds = np.hypot(np.diff(x), np.diff(y))
        L = ds.sum()
        if L < 2000:
            continue
        s = np.r_[0, np.cumsum(ds)]
        g = np.arange(0, L, STEP)
        xi, yi = np.interp(g, s, x), np.interp(g, s, y)
        h = np.unwrap(np.arctan2(np.diff(yi), np.diff(xi)))
        row = [lat0, lon0, L, math.degrees(h[-1] - h[0])]
        for sc in SCALES:
            hs = gaussian_filter1d(h, sc / STEP, mode="nearest")
            k = np.diff(hs) / STEP
            left, right = k > KMIN, k < -KMIN
            row += [left.sum() * STEP, right.sum() * STEP, math.degrees(k[left].sum() * STEP), math.degrees(-k[right].sum() * STEP)]
        rows.append(row)
    np.savez_compressed(out, rows=np.array(rows, np.float64), cols=np.array(COLS))
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(SRC, "part-*.npz")))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as ex:
        for k, p in enumerate(ex.map(one, parts)):
            if k % 16 == 0:
                print(k, "/", len(parts), p, flush=True)
    print("done")
