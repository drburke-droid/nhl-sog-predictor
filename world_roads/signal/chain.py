"""Fewest signal fires from Plymouth eastward around the world (as far as line of sight allows).

Physics (same as the VSL Sightline tool): Earth radius 6371 km, refraction k = 0.13 (effective radius
R/(1-k)), terrain from Terrarium elevation tiles. A fire sits on a 10 m pyre on a summit; two fires
see each other if the straight sight line (in the effective-radius geometry) clears every terrain
sample between them (sea counts as sea level).

    python chain.py      -> data/chain.json (+ data/bfs.npz)
"""
import os, sys, json, time, math
import numpy as np
import numba as nb
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R = 6371.0088
K = 0.13
REFF = R / (1 - K)
PYRE_KM = 0.010
CELL_KM = float(os.environ.get("CELL_KM", 12))
START = ("Plymouth", 50.3640, -4.1420)
SHIP_MAST_KM = 0.030
SKIP_KM = 1.0          # the summit's own max-filtered cell must not "block" its fire
CLEAR_KM = 0.025       # search demands 25 m of clearance (zoom-10 verification uses none)


def reach(h_km):
    return np.sqrt(2 * REFF * np.maximum(h_km, 0))


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def gc(a, b):
    return 2 * R * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))


@nb.njit(cache=True, fastmath=True)
def terrain(mos, row0, W, px, py, pz):
    lon = math.atan2(py, px)
    if pz > 0.9999999:
        pz = 0.9999999
    my = 0.5 * math.log((1 + pz) / (1 - pz))
    X = (lon / (2 * math.pi) + 0.5) * W - 0.5
    Y = (0.5 - my / (2 * math.pi)) * W - row0 - 0.5
    H = mos.shape[0]
    if Y < 0 or Y >= H - 1:
        return 0.0
    xi = int(math.floor(X)); yi = int(Y)
    fx = X - xi; fy = Y - yi
    x0 = xi % W; x1 = (xi + 1) % W
    a = mos[yi, x0]; b = mos[yi, x1]; c = mos[yi + 1, x0]; d = mos[yi + 1, x1]
    return ((a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy) / 1000.0


@nb.njit(cache=True, fastmath=True)
def los(mos, row0, W, pu, hu, pv, hv, reff, step, skip, clear=0.0):
    dot = pu[0] * pv[0] + pu[1] * pv[1] + pu[2] * pv[2]
    if dot > 1.0:
        dot = 1.0
    th = math.acos(dot)
    D = th * 6371.0088
    if D < 1e-6:
        return True
    n = int(D / step) + 2
    st = math.sin(th)
    for i in range(1, n):
        f = i / n
        s = f * D
        if s < skip or D - s < skip:
            continue
        a = math.sin((1 - f) * th) / st; b = math.sin(f * th) / st
        px = a * pu[0] + b * pv[0]; py = a * pu[1] + b * pv[1]; pz = a * pu[2] + b * pv[2]
        line = hu + (hv - hu) * f - s * (D - s) / (2 * reff)
        ramp = min(1.0, max(0.0, (min(s, D - s) - 1.0) / 5.0))     # margin grows from 0 at 1 km to full at 6 km
        if terrain(mos, row0, W, px, py, pz) + clear * ramp > line:
            return False
    return True


@nb.njit(cache=True)
def sea_span(mos, row0, W, pu, pv, step):
    """Along the great circle u->v: distance where the sea starts after leaving u's land, where it ends
    before v's land, and whether any land interrupts the sea in between."""
    dot = min(1.0, pu[0] * pv[0] + pu[1] * pv[1] + pu[2] * pv[2])
    th = math.acos(dot)
    D = th * 6371.0088
    n = int(D / step) + 2
    st = math.sin(th)
    first_sea, last_sea = -1.0, -1.0
    for i in range(1, n):
        f = i / n
        a = math.sin((1 - f) * th) / st; b = math.sin(f * th) / st
        h = terrain(mos, row0, W, a * pu[0] + b * pv[0], a * pu[1] + b * pv[1], a * pu[2] + b * pv[2])
        if h <= 0.001:
            if first_sea < 0:
                first_sea = f * D
            last_sea = f * D
    if first_sea < 0:
        return -1.0, -1.0, True
    for i in range(1, n):
        f = i / n
        s = f * D
        if s <= first_sea or s >= last_sea:
            continue
        a = math.sin((1 - f) * th) / st; b = math.sin(f * th) / st
        if terrain(mos, row0, W, a * pu[0] + b * pv[0], a * pu[1] + b * pv[1], a * pu[2] + b * pv[2]) > 0.001:
            return first_sea, last_sea, True
    return first_sea, last_sea, False


@nb.njit(parallel=True, cache=True)
def los_batch(mos, row0, W, P, H, u, targets, reff, step):
    out = np.zeros(len(targets), np.bool_)
    for i in nb.prange(len(targets)):
        v = targets[i]
        out[i] = los(mos, row0, W, P[u], H[u], P[v], H[v], reff, step, SKIP_KM, CLEAR_KM)
    return out


def load_terrain():
    meta = json.load(open(os.path.join(DATA, "mosaic_z8max_meta.json")))
    mos = np.ascontiguousarray(np.load(os.path.join(DATA, "mosaic_z8max.npy"), mmap_mode="r"))
    return mos, float(meta["row0_px"]), int(meta["width"])


def candidates():
    pk = np.load(os.path.join(DATA, "peaks_hr.npy"))
    lat, lon, h = pk[:, 0].astype(np.float64), pk[:, 1].astype(np.float64), pk[:, 2].astype(np.float64)
    inv_path = os.path.join(DATA, "invalid_nodes.json")
    if os.path.exists(inv_path):                         # summits rejected by verify.py (zoom-10 check)
        inv = np.array(json.load(open(inv_path)))
        if len(inv):
            t = cKDTree(np.c_[lat, lon])
            drop = np.unique(np.concatenate([t.query_ball_point(p, 2e-4) for p in inv]).astype(int))
            keep = np.setdiff1d(np.arange(len(lat)), drop)
            lat, lon, h = lat[keep], lon[keep], h[keep]
            print(f"dropped {len(drop)} rejected summits", flush=True)
    # highest summit per ~CELL_KM cell (equal-area-ish grid)
    cy = np.floor(lat * 111.2 / CELL_KM).astype(np.int64)
    cx = np.floor(lon * 111.2 * np.cos(np.radians((cy + 0.5) * CELL_KM / 111.2)) / CELL_KM).astype(np.int64)
    key = cy * 100000 + cx
    o = np.lexsort((-h, key))
    keep = o[np.r_[True, key[o][1:] != key[o][:-1]]]
    lat, lon, h = lat[keep], lon[keep], h[keep]
    return lat, lon, h


def main():
    t0 = time.time()
    mos, row0, W = load_terrain()
    lat, lon, h = candidates()
    # the start: Plymouth Hoe
    lat = np.r_[START[1], lat]; lon = np.r_[START[2], lon]
    P = xyz(lat, lon)
    h = np.r_[terrain(mos, row0, W, *P[0]) * 1000, h]            # ground height at Plymouth Hoe
    H = h / 1000 + PYRE_KM
    rch = reach(H)
    n = len(H)
    print(f"candidates {n:,} (cell {CELL_KM} km)  {time.time() - t0:.0f}s", flush=True)
    # height classes, each with its own KD-tree, so a query radius = own reach + class max reach
    edges_cls = [0, 0.15, 0.4, 0.8, 1.5, 2.5, 4.0, 9.0]
    cls = np.digitize(H, edges_cls) - 1
    trees, members, cmax = [], [], []
    for c in range(len(edges_cls) - 1):
        m = np.nonzero(cls == c)[0]
        members.append(m)
        trees.append(cKDTree(P[m]) if len(m) else None)
        cmax.append(reach(edges_cls[c + 1]))
    banned = {}
    ban_path = os.path.join(DATA, "banned_edges.json")
    if os.path.exists(ban_path):                         # sightlines rejected by verify.py
        t = cKDTree(np.c_[lat, lon])
        for (a_, b_) in json.load(open(ban_path)):
            da, ia = t.query(a_); db, ib = t.query(b_)
            if da < 2e-4 and db < 2e-4:
                banned.setdefault(int(ia), set()).add(int(ib)); banned.setdefault(int(ib), set()).add(int(ia))
        print(f"banned sightlines: {sum(len(v) for v in banned.values()) // 2}", flush=True)
    hop = np.full(n, -1, np.int32); parent = np.full(n, -1, np.int64)
    hop[0] = 0
    layer = np.array([0])
    k = 0
    tests = 0
    while len(layer):
        nxt = []
        for u in layer:
            found = []
            for c in range(len(trees)):
                if trees[c] is None:
                    continue
                rad = rch[u] + cmax[c]
                chord = 2 * math.sin(min(rad / R, math.pi) / 2)
                idx = members[c][trees[c].query_ball_point(P[u], chord)]
                if not len(idx):
                    continue
                idx = idx[hop[idx] < 0]
                if not len(idx):
                    continue
                d = gc(P[u][None], P[idx])
                idx = idx[d <= rch[u] + rch[idx]]
                if len(idx):
                    found.append(idx)
            if not found:
                continue
            idx = np.unique(np.concatenate(found))
            idx = idx[hop[idx] < 0]
            if not len(idx):
                continue
            ok = los_batch(mos, row0, W, P, H, int(u), idx.astype(np.int64), REFF, 0.5)
            if int(u) in banned:
                ok &= ~np.isin(idx, list(banned[int(u)]))
            tests += len(idx)
            vis = idx[ok]
            hop[vis] = k + 1; parent[vis] = u
            nxt.append(vis)
        layer = np.concatenate(nxt) if nxt else np.array([], np.int64)
        k += 1
        if k % 20 == 0 or not len(layer):
            print(f"hop {k}: +{len(layer):,} reached (total {(hop >= 0).sum():,}), tests {tests:,}, {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(DATA, "bfs.npz"), lat=lat, lon=lon, H=H, hop=hop, parent=parent)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
