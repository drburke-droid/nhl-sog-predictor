"""Every coastline on Earth: OpenStreetMap's land polygons (osmdata.openstreetmap.de,
land-polygons-complete-4326), read straight from the shapefile.

Every ring is one closed coastline: the outer shore of a continent or island, or the shore of a sea
that is a hole in the land (the Caspian). The shapefile draws outer rings clockwise and holes
anticlockwise, so walking every ring in file order keeps the land on the right: the fish's rule.

    python coast.py   -> data/coast_rings.npz  (full-resolution vertices, float64 lon/lat, per-ring
                         offsets, exact haversine length per ring, signed area, hole flag)
"""
import os, sys, json, time
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SHP = os.path.join(DATA, "osm", "land-polygons-complete-4326", "land_polygons.shp")
R_EARTH = 6371.0088


def read_polygons(path):
    """Minimal ESRI shapefile polygon reader: returns vertices (N,2) float64, ring offsets, and the
    record (polygon) each ring belongs to."""
    buf = np.fromfile(path, np.uint8)
    pos = 100
    xs, offs, rec = [], [0], []
    n = 0
    r = 0
    while pos < len(buf):
        clen = int.from_bytes(buf[pos + 4:pos + 8].tobytes(), "big") * 2
        c = pos + 8
        st = int(np.frombuffer(buf[c:c + 4].tobytes(), "<i4")[0])
        if st == 5:
            nparts, npts = np.frombuffer(buf[c + 36:c + 44].tobytes(), "<i4")
            parts = np.frombuffer(buf[c + 44:c + 44 + 4 * nparts].tobytes(), "<i4")
            p0 = c + 44 + 4 * nparts
            pts = np.frombuffer(buf[p0:p0 + 16 * npts].tobytes(), "<f8").reshape(npts, 2)
            bounds = np.r_[parts, npts]
            for a, b in zip(bounds[:-1], bounds[1:]):
                xs.append(pts[a:b])
                n += b - a
                offs.append(n)
                rec.append(r)
        r += 1
        pos = c + clen
    return np.concatenate(xs), np.array(offs, np.int64), np.array(rec, np.int64)


@nb.njit(cache=True)
def ring_stats(xy, off):
    nr = len(off) - 1
    L = np.zeros(nr)
    A = np.zeros(nr)
    for r in range(nr):
        a, b = off[r], off[r + 1]
        s = 0.0
        ar = 0.0
        for i in range(a, b - 1):
            lo1, la1 = xy[i, 0], xy[i, 1]
            lo2, la2 = xy[i + 1, 0], xy[i + 1, 1]
            p1, p2 = math.radians(la1), math.radians(la2)
            h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
            s += 2 * 6371.0088 * math.asin(math.sqrt(min(h, 1.0)))
            ar += (lo2 - lo1) * (la2 + la1)          # shoelace (sign only; >0 = clockwise)
        L[r] = s
        A[r] = ar
    return L, A


import math  # noqa: E402  (numba needs it at module level)

if __name__ == "__main__":
    t0 = time.time()
    xy, off, rec = read_polygons(SHP)
    print(f"{len(off) - 1:,} rings in {rec[-1] + 1:,} polygons, {len(xy):,} vertices ({time.time() - t0:.0f}s)", flush=True)
    L, A = ring_stats(xy, off)
    first = np.r_[True, rec[1:] != rec[:-1]]
    hole = ~first
    print(f"coastline: {L.sum():,.0f} km; outer rings {(~hole).sum():,} ({L[~hole].sum():,.0f} km), holes {hole.sum():,} ({L[hole].sum():,.0f} km)")
    print("outer rings clockwise:", np.mean(A[~hole] > 0), " holes anticlockwise:", np.mean(A[hole] < 0) if hole.any() else None)
    o = np.argsort(-L)[:15]
    for i in o:
        c = xy[off[i]:off[i + 1]]
        print(f"  ring {i}: {L[i]:,.0f} km, {len(c):,} vertices, hole={hole[i]}, lon {c[:,0].min():.1f}..{c[:,0].max():.1f} lat {c[:,1].min():.1f}..{c[:,1].max():.1f}")
    np.savez(os.path.join(DATA, "coast_rings.npz"), xy=xy, off=off, rec=rec, km=L, area=A, hole=hole)
    json.dump({"rings": int(len(off) - 1), "polygons": int(rec[-1] + 1), "vertices": int(len(xy)), "km": float(L.sum()),
               "outer_km": float(L[~hole].sum()), "hole_km": float(L[hole].sum()), "holes": int(hole.sum())},
              open(os.path.join(DATA, "coast_stats.json"), "w"), indent=1)
