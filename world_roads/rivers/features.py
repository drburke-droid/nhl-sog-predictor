"""Features of the 1,000 longest river systems, to ask what the rivers nearest pi have in common.

QA first: of the 1,200 longest systems traced by systems.py, the first 1,000 that are named and are not
  * a duplicate (more than 30% of the course within DUP_KM of a longer system's: a tributary that reaches
    the sea by its own channel down the same lower river),
  * a fragment (a river that stops inland within FRAG_KM of a longer river's main stem, where the mapping
    has a gap), or mostly gap links.

Per river:
  * sinuosity S = length / great-circle distance from source to mouth
  * divider lengths L(r): the main stem walked with a ruler of r km (1, 3, 10, 30, 100 km). S splits into
    meandering  M = L(native) / L(10 km)   (wiggles smaller than 10 km: meanders, braids)
    wandering   C = L(10 km) / straight     (the course itself bending on scales over 10 km)
  * bulge: the farthest the stem strays from the straight line, / straight distance
  * terrain along the stem, every 1 km, from the zoom-8 Terrarium mosaic (signal/data/mosaic_global.npy,
    max-pooled; the 3x3 minimum is taken to keep to the valley floor): source and mouth elevation, relief,
    mean gradient (m/km), share of the length below 200 m ("lowland"), median elevation
  * where: mouth latitude, continent (Natural Earth countries), and terminus (sea, Caspian, inland)

    python features.py   -> data/rivers.json (the 1,000, with features), prints the comparison
"""
import os, sys, json, math
import numpy as np
import numba as nb
import shapely
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from systems import xyz, gc_km, R_EARTH  # noqa: E402

N = 1000
FRAG_KM = 15.0
DUP_KM = 2.0
RULERS = (1.0, 3.0, 10.0, 30.0, 100.0)
MOSAIC = os.path.join(WR, "signal", "data", "mosaic_global.npy")
MOSAIC_META = os.path.join(WR, "signal", "data", "mosaic_global_meta.json")
COUNTRIES = os.path.join(WR, "data", "ne", "ne_10m_admin_0_countries.geojson")


@nb.njit(cache=True)
def divider(P, r):
    """length of a polyline (unit-sphere xyz rows) walked with a ruler of chord r (same units)"""
    n = P.shape[0]
    cx, cy, cz = P[0, 0], P[0, 1], P[0, 2]
    steps = 0
    i = 1
    while i < n:
        bx, by, bz = P[i, 0], P[i, 1], P[i, 2]
        db = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        if db < r:
            i += 1
            continue
        # the point on segment (i-1, i) at distance r from the current point
        ax, ay, az = P[i - 1, 0], P[i - 1, 1], P[i - 1, 2]
        ux, uy, uz = bx - ax, by - ay, bz - az
        wx, wy, wz = ax - cx, ay - cy, az - cz
        a = ux * ux + uy * uy + uz * uz
        b = 2 * (ux * wx + uy * wy + uz * wz)
        c = wx * wx + wy * wy + wz * wz - r * r
        disc = max(b * b - 4 * a * c, 0.0)
        t = (-b + math.sqrt(disc)) / (2 * a) if a > 0 else 1.0
        t = min(max(t, 0.0), 1.0)
        cx, cy, cz = ax + t * ux, ay + t * uy, az + t * uz
        steps += 1
        # stay on this segment: the next point may lie on it too
        P[i - 1, 0], P[i - 1, 1], P[i - 1, 2] = cx, cy, cz
    last = math.sqrt((P[n - 1, 0] - cx) ** 2 + (P[n - 1, 1] - cy) ** 2 + (P[n - 1, 2] - cz) ** 2)
    return steps * r + last


def resample(lonlat, step_km):
    """points every step_km along a lon/lat polyline (and the end), with their distance from the start"""
    seg = gc_km(lonlat[:-1, 0], lonlat[:-1, 1], lonlat[1:, 0], lonlat[1:, 1])
    cum = np.r_[0, np.cumsum(seg)]
    d = np.r_[np.arange(0, cum[-1], step_km), cum[-1]]
    j = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    t = np.where(seg[j] > 0, (d - cum[j]) / np.maximum(seg[j], 1e-12), 0)
    lon = lonlat[j, 0] + t * (lonlat[j + 1, 0] - lonlat[j, 0])
    lat = lonlat[j, 1] + t * (lonlat[j + 1, 1] - lonlat[j, 1])
    return lon, lat, d


class Terrain:
    def __init__(self):
        self.m = np.load(MOSAIC, mmap_mode="r")
        self.meta = json.load(open(MOSAIC_META))

    def __call__(self, lon, lat):
        z, row0 = self.meta["z"], self.meta["row0_px"]
        n = 256 * 2 ** z
        x = ((np.asarray(lon) + 180) / 360 * n).astype(np.int64) % n
        s = np.sin(np.radians(np.clip(lat, -85, 85)))
        y = ((0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n).astype(np.int64) - row0
        y = np.clip(y, 1, self.m.shape[0] - 2)
        best = np.full(len(x), 32767, np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                best = np.minimum(best, self.m[y + dy, (x + dx) % n].astype(np.int32))
        return best


def continent_of(lon, lat):
    gj = json.load(open(COUNTRIES))
    geoms = [shapely.geometry.shape(f["geometry"]) for f in gj["features"]]
    cont = [f["properties"].get("CONTINENT", "") for f in gj["features"]]
    tree = shapely.STRtree(geoms)
    pts = shapely.points(lon, lat)
    out = []
    for p, x in zip(pts, lon):
        c = cont[int(tree.nearest(p))]
        out.append("Asia" if c == "Europe" and x > 60 else c)       # Natural Earth files all of Russia under Europe
    return out


def main():
    S_all = json.load(open(os.path.join(DATA, "systems.json")))
    z = np.load(os.path.join(DATA, "stems.npz"))
    sxy, soff = z["xy"].astype(np.float64), z["off"]
    sid = np.repeat(np.arange(len(soff) - 1), np.diff(soff))
    tree = cKDTree(xyz(sxy[:, 0], sxy[:, 1]))
    # a river whose course mostly runs along a longer one already listed (a tributary reaching the sea by
    # its own channel through the same lower course) is part of that system
    pts, pid = [], []
    for i in range(len(S_all)):
        lon, lat, _ = resample(sxy[soff[i]:soff[i + 1]], 2.0)
        pts.append(xyz(lon, lat)); pid.append(np.full(len(lon), i))
    pts, pid = np.concatenate(pts), np.concatenate(pid)
    ptree = cKDTree(pts)
    shared = np.zeros(len(S_all))
    shared_with = np.full(len(S_all), -1)
    for i in range(len(S_all)):
        m = pid == i
        nb_ = ptree.query_ball_point(pts[m], DUP_KM / R_EARTH)
        hits = [min((pid[k] for k in lst if pid[k] < i), default=-1) for lst in nb_]
        hits = np.array(hits)
        shared[i] = np.mean(hits >= 0)
        if (hits >= 0).any():
            shared_with[i] = np.bincount(hits[hits >= 0]).argmax()
    # ---- QA
    keep, why = [], {}
    for i, r in enumerate(S_all):
        if shared[i] > 0.3:
            why[i] = f"duplicate: {shared[i]:.0%} of its course is {S_all[shared_with[i]]['name']}'s"
        elif not r["name"]:
            why[i] = "unnamed"
        elif r["gap_km"] > 0.15 * r["length_km"]:
            why[i] = "mostly gap links"
        elif r["terminus"] == "inland":
            near = tree.query_ball_point(xyz(np.array([r["mouth"][0]]), np.array([r["mouth"][1]]))[0], FRAG_KM / R_EARTH)
            longer = [sid[k] for k in near if sid[k] < i]
            if longer:
                why[i] = f"fragment: stops {FRAG_KM:.0f} km or less from {S_all[min(longer)]['name']}"
        if i not in why:
            keep.append(i)
    print(f"QA: {len(why)} of the first {len(S_all)} left out "
          f"({sum(1 for k in why if k < keep[N - 1])} of them before the {N}th kept river)")
    for k in sorted(why)[:25]:
        print(f"   {k:4d} {S_all[k]['name'][:28]:28s} {S_all[k]['length_km']:6.0f} km  {why[k]}")
    keep = keep[:N]
    terrain = Terrain()
    out = []
    for rank, i in enumerate(keep):
        r = dict(S_all[i])
        ll = sxy[soff[i]:soff[i + 1]]
        P = xyz(ll[:, 0], ll[:, 1])
        Ls = {}
        for rk in RULERS:
            Ls[rk] = float(divider(P.copy(), 2 * math.sin(rk / R_EARTH / 2)) * R_EARTH)
        L0 = float(np.sum(gc_km(ll[:-1, 0], ll[:-1, 1], ll[1:, 0], ll[1:, 1])))
        D = r["straight_km"]
        # the plane of the great circle through source and mouth: how far the stem strays from it
        a, b = xyz(np.array([r["source"][0]]), np.array([r["source"][1]]))[0], xyz(np.array([r["mouth"][0]]), np.array([r["mouth"][1]]))[0]
        nrm = np.cross(a, b)
        nrm /= max(np.linalg.norm(nrm), 1e-12)
        bulge = float(np.abs(P @ nrm).max() * R_EARTH)
        lon, lat, dkm = resample(ll, 1.0)
        h = terrain(lon, lat).astype(float)
        src_h = float(np.median(h[:3]))
        mouth_h = float(np.median(h[-3:]))
        relief = max(src_h - mouth_h, 0.0)
        r.update(rank=rank + 1, sys_index=i, stem_km=L0, divider_km={str(k): v for k, v in Ls.items()},
                 meander=L0 / Ls[10.0], wander=Ls[10.0] / D, bulge=bulge / D,
                 src_elev_m=src_h, mouth_elev_m=mouth_h, relief_m=relief, gradient_m_per_km=relief / r["length_km"],
                 lowland_share=float(np.mean(h < 200)), median_elev_m=float(np.median(h)),
                 mouth_lat=r["mouth"][1], mean_abs_lat=float(np.mean(np.abs(lat))))
        out.append(r)
        if rank % 200 == 0:
            print(f"  {rank}/{N}", flush=True)
    cont = continent_of(np.array([r["mouth"][0] for r in out]), np.array([r["mouth"][1] for r in out]))
    for r, c in zip(out, cont):
        r["continent"] = c
        mx, my = r["mouth"]
        if r["terminus"] == "sea" and 46 < mx < 55.5 and 36.5 < my < 47.5:
            r["terminus"] = "Caspian"
    json.dump(out, open(os.path.join(DATA, "rivers.json"), "w"), indent=0)
    print(f"wrote {len(out)} rivers to data/rivers.json")


if __name__ == "__main__":
    main()
