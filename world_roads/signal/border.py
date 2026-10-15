"""A fire only counts for a country if it stands at least MARGIN_KM from every other country (Natural
Earth borders are not exact). Vatican City and Monaco are too small for that; their hand-checked high
points (world.py) count. Countries left with no such summit get their highest point >= MARGIN_KM inside
(Terrarium zoom 12/13), with its sightlines added to the graph.

    python border.py        -> world_nodes.npz gains border_ok (+ any new summits), world_edges.npz extended
    python border.py gulf   -> zero-margin sightlines around Bahrain and Qatar (flat_region_edges)
    python border.py flat   -> the same for the Gambia, the Palk Strait, the Bahamas and Kuwait
"""
import os, sys, json, math, time
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


@nb.njit(parallel=True, cache=True)
def los_pairs0(mos, row0, W, P, H, pu, pv, reff, step):
    out = np.zeros(len(pu), np.bool_)
    for i in nb.prange(len(pu)):
        out[i] = CH.los(mos, row0, W, P[pu[i]], H[pu[i]], P[pv[i]], H[pv[i]], reff, step, 0.5, 0.0)
    return out


def flat_region_edges(box):
    """Low, flat coasts (the Gulf): the search grid's 25 m margin and ~600 m max-filter block nearly every
    low sightline there. Add zero-margin sightlines for summits in `box` (lat0, lat1, lon0, lon1); the
    zoom-10/12 verification still has the last word on each hop used."""
    nd = np.load(os.path.join(DATA, "world_nodes.npz"))
    lat, lon = nd["lat"], nd["lon"]
    H = nd["h"] / 1000 + CH.PYRE_KM
    P = CH.xyz(lat, lon)
    rch = CH.reach(H)
    a, b, c, d = box
    inb = np.nonzero((lat >= a) & (lat <= b) & (lon >= c) & (lon <= d))[0]
    pu, pv = [], []
    for u in inb:
        dd = np.arccos(np.clip(P[u] @ P.T, -1, 1)) * CH.R
        v = np.nonzero((dd <= rch[u] + rch) & (np.arange(len(lat)) != u))[0]
        pu.append(np.full(len(v), u)); pv.append(v)
    pu = np.concatenate(pu).astype(np.int64); pv = np.concatenate(pv).astype(np.int64)
    mos, row0, W = ED.load_global()
    ok = los_pairs0(mos, row0, W, P, H, pu, pv, CH.REFF, 0.25)
    e = np.load(os.path.join(DATA, "world_edges.npz"))
    u0, v0 = np.minimum(pu[ok], pv[ok]), np.maximum(pu[ok], pv[ok])
    old = set(zip(e["u"][np.isin(e["u"], inb) | np.isin(e["v"], inb)].tolist(), e["v"][np.isin(e["u"], inb) | np.isin(e["v"], inb)].tolist()))
    new = [(x, y) for x, y in zip(u0.tolist(), v0.tolist()) if (x, y) not in old]
    if new:
        nn = np.array(sorted(set(new)), np.int32)
        np.savez(os.path.join(DATA, "world_edges.npz"), u=np.r_[e["u"], nn[:, 0]], v=np.r_[e["v"], nn[:, 1]])
    print(f"{len(inb)} summits in the box: {ok.sum():,} zero-margin sightlines, {len(new):,} new", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gulf":
        flat_region_edges((23.0, 28.5, 47.5, 53.0))
    elif len(sys.argv) > 1 and sys.argv[1] == "flat":
        # other low coasts where the countries' only links are low sightlines
        for box in ((12.4, 14.6, -17.6, -13.0),        # the Gambia (and the Senegal around it)
                    (7.5, 11.0, 78.0, 81.5),            # Palk Strait: India <-> Sri Lanka
                    (20.0, 27.5, -80.5, -72.0),         # the Bahamas
                    (28.0, 30.6, 46.5, 49.2)):          # Kuwait
            flat_region_edges(box)
    else:
        main()
