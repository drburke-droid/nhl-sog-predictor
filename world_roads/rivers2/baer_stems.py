"""Baer's law on big rivers: the main stems of the 1,000 longest river systems (from the pi
project), source to mouth, smoothed at 1 km and 5 km. Rotation matters most for large, slow rivers.

    python baer_stems.py      -> data/baer_stems.json
"""
import os, glob, json, math
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
RIV = os.path.join(HERE, "..", "rivers", "data")
STEP = 50.0
SCALES = (1000.0, 2000.0, 5000.0, 10000.0)
KMIN = 1 / 50000.0


def lines():
    xs, offs, vb = [], [np.zeros(1, np.int64)], 0
    for p in sorted(glob.glob(os.path.join(RIV, "full", "part-*.npz"))):
        z = np.load(p, allow_pickle=True)
        xs.append(z["xy"]); offs.append(z["off"][1:] + vb); vb += len(z["xy"])
    return np.concatenate(xs), np.concatenate(offs)


def tally(lon, lat):
    lat0 = lat.mean()
    x = (lon - lon.mean()) * np.cos(np.radians(lat)) * 111320.0          # local scale per vertex (long rivers)
    y = (lat - lat0) * 110540.0
    ds = np.hypot(np.diff(x), np.diff(y))
    keep = np.r_[True, ds > 0.5]
    x, y = x[keep], y[keep]
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0, np.cumsum(ds)]
    g = np.arange(0, s[-1], STEP)
    h = np.unwrap(np.arctan2(np.diff(np.interp(g, s, y)), np.diff(np.interp(g, s, x))))
    out = []
    for sc in SCALES:
        k = np.diff(gaussian_filter1d(h, sc / STEP, mode="nearest")) / STEP
        out += [(k > KMIN).sum() * STEP, (k < -KMIN).sum() * STEP]
    return out


def main():
    xy, off = lines()
    sys_ = json.load(open(os.path.join(RIV, "all_rivers.json")))
    rows = []
    for r in sys_:
        if "pieces" not in r:
            continue                     # only the 1,000 longest systems keep their pieces
        segs = []
        for pid, fwd in r["pieces"]:
            a, b = off[pid], off[pid + 1]
            seg = xy[a:b] / 1e7
            segs.append(seg if fwd else seg[::-1])
        line = np.concatenate(segs).astype(np.float64)
        if len(line) < 10:
            continue
        lat_mid = float(np.median(line[:, 1]))
        lon_mid = float(np.median(line[:, 0]))
        (sx, sy), (mx, my) = r["source"], r["mouth"]
        az = math.degrees(math.atan2((mx - sx) * math.cos(math.radians((sy + my) / 2)), my - sy)) % 360
        rows.append([lat_mid, r["length_km"]] + tally(line[:, 0], line[:, 1]) + [lon_mid, az])
    R = np.array(rows)
    np.save(os.path.join(HERE, "data", "baer_stems_rows.npy"), R)
    res = {"systems": len(R)}
    rng = np.random.default_rng(3)
    for j, sc in enumerate(SCALES):
        l, rr = R[:, 2 + 2 * j], R[:, 3 + 2 * j]
        for name, m in (("north", R[:, 0] >= 5), ("south", R[:, 0] <= -5), ("north_above_45", R[:, 0] >= 45)):
            est = l[m].sum() / (l[m].sum() + rr[m].sum())
            idx = np.nonzero(m)[0]
            bs = []
            for _ in range(1000):
                k = rng.choice(idx, len(idx))
                bs.append(l[k].sum() / (l[k].sum() + rr[k].sum()))
            frac_left = float(np.mean(l[m] > rr[m]))
            res[f"{name}_{int(sc)}"] = {"rivers": int(m.sum()), "share_left": float(est),
                                       "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                       "rivers_mostly_left": frac_left}
    json.dump(res, open(os.path.join(HERE, "data", "baer_stems.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
