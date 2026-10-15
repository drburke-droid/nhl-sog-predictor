"""A fire only counts for a country if it stands at least MARGIN_KM from every other country (Natural
Earth borders are not exact). Vatican City and Monaco are too small for that; their hand-checked high
points (world.py) count. Countries left with no such summit get their highest point >= MARGIN_KM inside
(Terrarium zoom 12/13), with its sightlines added to the graph.

    python border.py    -> world_nodes.npz gains border_ok (+ any new summits), world_edges.npz extended
"""
import os, json, math, time
import numpy as np
import shapely
from shapely.strtree import STRtree
import numba as nb
import chain as CH
import edges as ED
import world as WD

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MARGIN_KM = 1.0
EXEMPT = {"Vatican", "Monaco"}


def border_ok(lat, lon, cid, geoms, names):
    tree = STRtree(geoms)
    dist = MARGIN_KM / 111.2 / np.maximum(np.cos(np.radians(lat)), 0.2)
    pi, gi = tree.query(shapely.points(lon, lat), predicate="dwithin", distance=dist)
    ok = cid >= 0
    clash = gi != cid[pi]
    ok[np.unique(pi[clash])] = False
    for nm in EXEMPT:
        g = names.index(nm)
        ok[cid == g] = True
    return ok


def interior_summit(geom, z=12):
    inner = geom.buffer(-MARGIN_KM / 111.2 * 1.3)
    if inner.is_empty:
        return None
    lo0, la0, lo1, la1 = inner.bounds
    span = max(lo1 - lo0, la1 - la0)
    if span > 1.5:
        z = 10
    elif span > 0.4:
        z = 11
    WD.Z_TINY = z
    return WD.tiny_summit(inner)


def main():
    t0 = time.time()
    nd = dict(np.load(os.path.join(DATA, "world_nodes.npz")))
    names, geoms, is_c = WD.countries()
    ok = border_ok(nd["lat"], nd["lon"], nd["cid"], geoms, names)
    have = set(np.unique(nd["cid"][ok & (nd["cid"] >= 0)]).tolist())
    need = [g for g in np.nonzero(is_c)[0] if g not in have]
    print(f"{ok.sum():,} of {len(ok):,} summits are >= {MARGIN_KM} km from other countries; "
          f"countries without one: {[names[g] for g in need]}  {time.time() - t0:.0f}s", flush=True)
    extra = []
    for g in need:
        r = interior_summit(geoms[g])
        if r is None or r[0] < -1e8:
            print(f"  {names[g]}: no point {MARGIN_KM} km inside", flush=True)
            continue
        print(f"  {names[g]}: interior high point {r[0]:.0f} m at {r[1]:.4f},{r[2]:.4f}", flush=True)
        extra.append((r[1], r[2], max(r[0], 1.0), g))
    n0 = len(nd["lat"])
    if extra:
        e = np.array(extra)
        nd["lat"] = np.r_[nd["lat"], e[:, 0]]; nd["lon"] = np.r_[nd["lon"], e[:, 1]]
        nd["h"] = np.r_[nd["h"], e[:, 2]]; nd["cid"] = np.r_[nd["cid"], e[:, 3].astype(np.int32)]
        ok = np.r_[ok, np.ones(len(e), bool)]
    nd["border_ok"] = ok
    np.savez(os.path.join(DATA, "world_nodes.npz"), **nd)
    if extra:
        add_edges(nd, n0)
    print(f"done {time.time() - t0:.0f}s")


def add_edges(nd, n0):
    """Sightlines from the new summits (index >= n0) to every other summit."""
    mos, row0, W = ED.load_global()
    lat, lon = nd["lat"], nd["lon"]
    H = nd["h"] / 1000 + CH.PYRE_KM
    P = CH.xyz(lat, lon)
    rch = CH.reach(H)
    n = len(lat)
    new = np.arange(n0, n)
    d = np.arccos(np.clip(P[new] @ P.T, -1, 1)) * CH.R
    pu, pv = [], []
    for k, u in enumerate(new):
        v = np.nonzero((d[k] <= rch[u] + rch) & (np.arange(n) != u))[0]
        pu.append(np.full(len(v), u)); pv.append(v)
    pu = np.concatenate(pu).astype(np.int64); pv = np.concatenate(pv).astype(np.int64)
    ok = ED.los_pairs(mos, row0, W, P, H, pu, pv, CH.REFF, 0.5)
    e = np.load(os.path.join(DATA, "world_edges.npz"))
    a, b = np.minimum(pu[ok], pv[ok]), np.maximum(pu[ok], pv[ok])
    np.savez(os.path.join(DATA, "world_edges.npz"), u=np.r_[e["u"], a.astype(np.int32)], v=np.r_[e["v"], b.astype(np.int32)])
    print(f"added {ok.sum():,} sightlines for {len(new)} new summits", flush=True)


if __name__ == "__main__":
    main()
