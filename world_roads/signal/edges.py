"""Every fire-to-fire sightline in the world: for each pair of candidate summits close enough that
their horizons overlap, the same terrain check as chain.py (25 m clearance, zoom-8-resolution max grid).

    python edges.py      -> data/world_edges.npz (u, v pairs, u < v)
"""
import os, sys, json, math, time
import numpy as np
import numba as nb
import chain as CH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CLS = np.array([0, 0.15, 0.4, 0.8, 1.5, 2.5, 4.0, 9.0])
CELL_DEG = 1.0


def load_global():
    meta = json.load(open(os.path.join(DATA, "mosaic_global_meta.json")))
    mos = np.ascontiguousarray(np.load(os.path.join(DATA, "mosaic_global.npy"), mmap_mode="r"))
    return mos, float(meta["row0_px"]), int(meta["width"])


def grid(lat, lon, members):
    """Nodes of one height class bucketed in 1-degree cells: (order, start offsets)."""
    cy = np.clip(np.floor((lat[members] + 90) / CELL_DEG), 0, 179).astype(np.int64)
    cx = np.clip(np.floor((lon[members] + 180) / CELL_DEG), 0, 359).astype(np.int64)
    cell = cy * 360 + cx
    o = np.argsort(cell, kind="stable")
    start = np.searchsorted(cell[o], np.arange(180 * 360 + 1))
    return members[o].astype(np.int64), start.astype(np.int64)


@nb.njit(parallel=True, cache=True)
def count_pairs(us, lat, lon, P, rch, cls_order, cls_start, cmax):
    out = np.zeros(len(us), np.int64)
    for k in nb.prange(len(us)):
        u = us[k]
        n = 0
        for c in range(len(cmax)):
            order = cls_order[c]; start = cls_start[c]
            Rs = rch[u] + cmax[c]
            dlat = Rs / 111.2
            y0 = max(int(math.floor(lat[u] - dlat + 90)), 0); y1 = min(int(math.floor(lat[u] + dlat + 90)), 179)
            for cy in range(y0, y1 + 1):
                la_far = max(abs(cy - 90.0), abs(cy + 1 - 90.0))
                cosl = math.cos(math.radians(min(la_far, 89.9)))
                dlon = Rs / (111.2 * cosl)
                x0 = int(math.floor(lon[u] - dlon + 180)); x1 = int(math.floor(lon[u] + dlon + 180))
                if dlon >= 180 or x1 - x0 >= 359:
                    x0, x1 = 0, 359
                for xx in range(x0, x1 + 1):
                    cx = xx % 360
                    cid = cy * 360 + cx
                    for j in range(start[cid], start[cid + 1]):
                        v = order[j]
                        if v <= u:
                            continue
                        dot = P[u, 0] * P[v, 0] + P[u, 1] * P[v, 1] + P[u, 2] * P[v, 2]
                        d = math.acos(min(1.0, dot)) * 6371.0088
                        if d <= rch[u] + rch[v]:
                            n += 1
        out[k] = n
    return out


@nb.njit(parallel=True, cache=True)
def fill_pairs(us, offs, lat, lon, P, rch, cls_order, cls_start, cmax, pu, pv):
    for k in nb.prange(len(us)):
        u = us[k]
        n = offs[k]
        for c in range(len(cmax)):
            order = cls_order[c]; start = cls_start[c]
            Rs = rch[u] + cmax[c]
            dlat = Rs / 111.2
            y0 = max(int(math.floor(lat[u] - dlat + 90)), 0); y1 = min(int(math.floor(lat[u] + dlat + 90)), 179)
            for cy in range(y0, y1 + 1):
                la_far = max(abs(cy - 90.0), abs(cy + 1 - 90.0))
                cosl = math.cos(math.radians(min(la_far, 89.9)))
                dlon = Rs / (111.2 * cosl)
                x0 = int(math.floor(lon[u] - dlon + 180)); x1 = int(math.floor(lon[u] + dlon + 180))
                if dlon >= 180 or x1 - x0 >= 359:
                    x0, x1 = 0, 359
                for xx in range(x0, x1 + 1):
                    cx = xx % 360
                    cid = cy * 360 + cx
                    for j in range(start[cid], start[cid + 1]):
                        v = order[j]
                        if v <= u:
                            continue
                        dot = P[u, 0] * P[v, 0] + P[u, 1] * P[v, 1] + P[u, 2] * P[v, 2]
                        d = math.acos(min(1.0, dot)) * 6371.0088
                        if d <= rch[u] + rch[v]:
                            pu[n] = u; pv[n] = v
                            n += 1


@nb.njit(parallel=True, cache=True)
def los_pairs(mos, row0, W, P, H, pu, pv, reff, step):
    out = np.zeros(len(pu), np.bool_)
    for i in nb.prange(len(pu)):
        u = pu[i]; v = pv[i]
        out[i] = CH.los(mos, row0, W, P[u], H[u], P[v], H[v], reff, step, CH.SKIP_KM, CH.CLEAR_KM)
    return out


def main(limit=None):
    t0 = time.time()
    mos, row0, W = load_global()
    nd = np.load(os.path.join(DATA, "world_nodes.npz"))
    lat, lon = nd["lat"], nd["lon"]
    H = nd["h"] / 1000 + CH.PYRE_KM
    P = CH.xyz(lat, lon)
    rch = CH.reach(H)
    n = len(lat)
    cls = np.clip(np.digitize(H, CLS) - 1, 0, len(CLS) - 2)
    cls_order, cls_start = nb.typed.List(), nb.typed.List()
    for c in range(len(CLS) - 1):
        o, s = grid(lat, lon, np.nonzero(cls == c)[0])
        cls_order.append(o); cls_start.append(s)
    cmax = CH.reach(CLS[1:]).astype(np.float64)
    print(f"nodes {n:,}, terrain loaded {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(n) if limit else np.arange(n)       # a random sample for timing runs
    if limit:
        order = order[:limit]
    EU, EV = [], []
    tot_pairs = 0
    CH_N = 20000
    for a in range(0, len(order), CH_N):
        us = np.sort(order[a:a + CH_N]).astype(np.int64)
        cnt = count_pairs(us, lat, lon, P, rch, cls_order, cls_start, cmax)
        offs = np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int64)
        m = int(cnt.sum())
        pu = np.empty(m, np.int64); pv = np.empty(m, np.int64)
        fill_pairs(us, offs, lat, lon, P, rch, cls_order, cls_start, cmax, pu, pv)
        ok = los_pairs(mos, row0, W, P, H, pu, pv, CH.REFF, 0.5)
        EU.append(pu[ok].astype(np.int32)); EV.append(pv[ok].astype(np.int32))
        tot_pairs += m
        done = a + len(us)
        print(f"  {done:,}/{len(order):,} nodes: {tot_pairs:,} pairs tested, {sum(len(e) for e in EU):,} sightlines, "
              f"{time.time() - t0:.0f}s", flush=True)
    eu, ev = np.concatenate(EU), np.concatenate(EV)
    if not limit:
        np.savez(os.path.join(DATA, "world_edges.npz"), u=eu, v=ev)
    print("done", len(eu), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
