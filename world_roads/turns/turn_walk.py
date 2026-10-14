"""Alternate left / right at every intersection: where do you end up?

Rules
  * Public roads only (motorway ... residential, living street); no service roads, paths or parking.
  * An intersection is any point where 3+ road pieces meet. Bends (2 pieces) are just followed.
  * "Left" = the leftmost exit, "right" = the rightmost exit (excluding a U-turn). If only one
    other exit exists, you take it. Dead end: U-turn.
  * Turns alternate L, R, L, R ... at successive intersections. One-way rules are ignored.
The state (current road piece, direction, next turn) is finite, so the walk must eventually repeat
exactly: from then on it circles the same loop forever. The script finds that loop.
"""
import os, sys, json
import numpy as np
import shapely
import pyarrow.parquet as pq

R_EARTH = 6371.0088
DRIVE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street"}
LOOK_M = 12.0


def build(path, lat0):
    t = pq.read_table(path)
    cls = np.array(t.column("class").to_pylist(), dtype=object)
    keep = np.isin(cls, list(DRIVE))
    geoms = shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False)[keep])
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    q = np.round(coords * 2e5).astype(np.int64)
    code = q[:, 0] * 10**9 + q[:, 1]
    nodes, inv = np.unique(code, return_inverse=True)
    same = idx[1:] == idx[:-1]
    a, b = inv[:-1][same], inv[1:][same]
    good = a != b
    a, b = a[good], b[good]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    u = np.unique(lo * len(nodes) + hi)
    ea, eb = u // len(nodes), u % len(nodes)
    lon = (nodes // 10**9) / 2e5; lat = (nodes % 10**9) / 2e5
    kx = np.radians(1) * R_EARTH * 1000 * np.cos(np.radians(lat0)); ky = np.radians(1) * R_EARTH * 1000
    x, y = lon * kx, lat * ky                          # metres, local
    n = len(nodes)
    ends = np.r_[ea, eb]; nb = np.r_[eb, ea]
    o = np.argsort(ends, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(ends, minlength=n))]
    adj = nb[o]
    return dict(lon=lon, lat=lat, x=x, y=y, indptr=indptr, adj=adj, n=n)


def neighbors(G, v):
    return G["adj"][G["indptr"][v]:G["indptr"][v + 1]]


def look_ahead(G, prev, cur):
    """Point ~LOOK_M metres along the branch prev->cur (through bends), for a stable direction."""
    x, y = G["x"], G["y"]
    d = 0.0; p, c = prev, cur
    while True:
        d += np.hypot(x[c] - x[p], y[c] - y[p])
        nb = neighbors(G, c)
        if d >= LOOK_M or len(nb) != 2:
            return c
        nxt = nb[0] if nb[0] != p else nb[1]
        p, c = c, nxt


def heading_into(G, prev, cur):
    """Direction of travel arriving at cur, measured over the last ~LOOK_M metres."""
    back = look_ahead(G, cur, prev)
    return np.array([G["x"][cur] - G["x"][back], G["y"][cur] - G["y"][back]])


def walk(G, start, first, max_steps=5_000_000):
    x, y = G["x"], G["y"]
    prev, cur, parity = start, first, 0          # parity 0 -> next intersection: LEFT
    seen = {}
    path = [start, first]
    turns = []                                   # (node, 'L'/'R'/'U'/'S')
    for step in range(max_steps):
        key = (prev, cur, parity)
        if key in seen:
            return path, turns, seen[key]
        seen[key] = len(path) - 1
        nb = neighbors(G, cur)
        exits = [w for w in nb if w != prev]
        if not exits:                            # dead end
            nxt, kind = prev, "U"
        elif len(nb) <= 2:                       # a bend, not an intersection
            nxt, kind = exits[0], "S"
        else:
            h = heading_into(G, prev, cur)
            angs = []
            for w in exits:
                la = look_ahead(G, cur, w)
                d = np.array([x[la] - x[cur], y[la] - y[cur]])
                angs.append(np.arctan2(h[0] * d[1] - h[1] * d[0], h @ d))   # + = left
            angs = np.array(angs)
            pick = int(np.argmax(angs)) if parity == 0 else int(np.argmin(angs))
            nxt, kind = exits[pick], ("L" if parity == 0 else "R")
            parity ^= 1
        turns.append((cur, kind))
        prev, cur = cur, nxt
        path.append(cur)
    return path, turns, None


def km_of(G, path):
    p = np.array(path)
    return float(np.hypot(np.diff(G["x"][p]), np.diff(G["y"][p])).sum() / 1000)


if __name__ == "__main__":
    lat0, lon0 = 34.0537, -118.2427          # Los Angeles City Hall
    G = build(sys.argv[1] if len(sys.argv) > 1 else "data/la.parquet", lat0)
    kx = np.radians(1) * R_EARTH * 1000 * np.cos(np.radians(lat0)); ky = np.radians(1) * R_EARTH * 1000
    d2 = (G["x"] - lon0 * kx) ** 2 + (G["y"] - lat0 * ky) ** 2
    start = int(np.argmin(d2))
    # first move: the neighbour pointing most to the east (toward New York)
    nb = neighbors(G, start)
    first = int(nb[np.argmax(G["x"][nb] - G["x"][start])])
    path, turns, loop_at = walk(G, start, first)
    total = km_of(G, path)
    lead = km_of(G, path[: loop_at + 1]) if loop_at is not None else None
    loop = km_of(G, path[loop_at:]) if loop_at is not None else None
    lon, lat = G["lon"][np.array(path)], G["lat"][np.array(path)]
    nint = sum(1 for _, k in turns if k in "LR")
    res = {"start": [lon0, lat0], "steps": len(path), "intersections_turned": nint,
           "km_before_loop": lead, "loop_km": loop, "loop_start": [float(G["lon"][path[loop_at]]), float(G["lat"][path[loop_at]])] if loop_at is not None else None,
           "max_dist_from_start_km": float(np.max(np.hypot((lon - lon0) * kx, (lat - lat0) * ky)) / 1000),
           "bbox_walked": [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]}
    if loop is not None:
        res["loops_per_second_at_c"] = 299_792.458 / loop
    print(json.dumps(res, indent=1))
    np.savez("data/walk.npz", lon=lon, lat=lat, loop_at=loop_at if loop_at is not None else -1)
