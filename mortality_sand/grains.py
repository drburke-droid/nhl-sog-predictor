"""Grain geometry shared by the simulation (sand.py), the painter (film.py) and the Cycles project
(blender_cycles.py copies this file into the .blend, where it runs inside Blender with numpy only).

Map pixels throughout: x right, y down, z up (height above the table).
"""
import numpy as np

TG = 0.45              # seconds a lifted grain glows on its pile before it goes


def hash01(a, b, salt):
    """Stable pseudo-random numbers in [0, 1) from integer arrays (per grain jitter)."""
    h = (a.astype(np.uint64) * np.uint64(73856093)) ^ (b.astype(np.uint64) * np.uint64(19349663)) ^ np.uint64(salt * 83492791)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5bd1e995)
    h = h ^ (h >> np.uint64(15))
    return (h % np.uint64(1 << 20)).astype(np.float64) / float(1 << 20)


def visible_grains(h, nb):
    """The grains worth drawing: each pile's top layers, and every grain on a slope or the side of a tower.
    h: grains per cell (all cells); nb: (4, ncell) neighbour cell indices (-1 off the map).
    Returns (cell, lev, hcol): each grain's cell, its level in the pile (0 = bottom) and its pile's height,
    sorted by cell, then level."""
    h = h.astype(np.int64)
    occ = np.where(h > 0)[0]
    if not len(occ):
        e = np.zeros(0, np.int64)
        return e, e, e
    hn = np.full(len(occ), 1 << 20, np.int64)
    for d in range(4):
        n = nb[d][occ]
        hn = np.minimum(hn, np.where(n >= 0, h[np.maximum(n, 0)], 0))
    ho = h[occ]
    lo = np.clip(np.minimum(hn, ho) - 2, 0, None)
    cnt = ho - lo
    cell = np.repeat(occ, cnt)
    lev = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
    return cell, lev, np.repeat(ho, cnt)


def grain_place(cell, lev, cell_xy, gs, grain_dz):
    """Where a resting grain sits (map px; z = the height of its centre) and its radius."""
    jx, jy, jr = hash01(cell, lev, 1), hash01(cell, lev, 2), hash01(cell, lev, 3)
    x = cell_xy[cell, 0] + (jx - 0.5) * 1.3 * gs
    y = cell_xy[cell, 1] + (jy - 0.5) * 1.3 * gs
    r = (1.35 + 0.35 * jr) * gs
    return x, y, lev * grain_dz + r, r


def rise_pos(p, idx, u):
    """Where lifted grain idx of batch p is, u seconds after it was taken (map px; z up). It glows on
    its pile for TG seconds and lifts a little; then it is drawn in toward a column over the place
    the catastrophe struck, turning as it goes, and pulled up faster and faster. A grain whose person
    died far from home (a soldier, a deportee: mv > 0) first travels there in a low arc, for mv
    seconds, and rises from beside the place."""
    x0, y0, z0 = p["x"][idx], p["y"][idx], p["z"][idx]
    cx, cy = p["cx"][idx], p["cy"][idx]
    ph = p["ph"][idx]
    mv = p["mv"][idx] if "mv" in p else np.zeros(len(idx))
    g = np.clip(u / TG, 0, 1)
    lift0 = 2.5 * g * g * (3 - 2 * g)
    trav = mv > 0
    if trav.any():
        # where it lands: a little ring around the place, then it rises from there
        ra = 3.0 + 7.0 * ph[:, 1]
        ax = np.where(trav, cx + ra * np.cos(6.283 * ph[:, 2]), x0)
        ay = np.where(trav, cy + ra * np.sin(6.283 * ph[:, 2]), y0)
        az = np.where(trav, 3.0, z0)
        tm = np.where(trav, np.clip((u - TG) / np.maximum(mv, 1e-6), 0, 1), 1.0)
        e = tm * tm * (3 - 2 * tm)
        dist = np.hypot(ax - x0, ay - y0)
        hop = (8.0 + 0.22 * dist) * 4 * tm * (1 - tm)      # a visible arc: a journey, not a slide
        xs = x0 + (ax - x0) * e
        ys = y0 + (ay - y0) * e
        zs = z0 + (az - z0) * e + hop + lift0 * (1 - e) + 2.5 * e
        bx, by, bz = ax, ay, az + 2.5                      # where the rise starts
    else:
        xs, ys, zs = x0, y0, z0 + lift0
        bx, by, bz = x0, y0, z0 + 2.5
    tau = np.maximum(u - TG - mv, 0.0)
    z = bz + 14.0 * tau + 165.0 * (0.8 + 0.4 * ph[:, 0]) * tau ** 2
    ox, oy = bx - cx, by - cy
    r0 = np.hypot(ox, oy)
    th0 = np.arctan2(oy, ox)
    kk = 1 - np.exp(-tau / 0.85)
    r = r0 * (1 - 0.62 * kk) + 2.0 * np.sin(5.0 * tau + 6.283 * ph[:, 2]) * np.minimum(tau, 1.0)
    th = th0 + (1.1 + 0.9 * ph[:, 1]) * kk + 0.35 * tau
    rising = tau > 0
    x = np.where(rising, cx + r * np.cos(th), xs)
    y = np.where(rising, cy + r * np.sin(th), ys)
    z = np.where(rising, z, zs)
    return x, y, z
