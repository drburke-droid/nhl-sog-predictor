"""The fish's world as one planar network, traced with the right-hand rule.

Pieces:
  * coastline rings (coast.py), walked one way only: clockwise round the land, so the land is on the right
  * lake shores: every lake over 5 km2 that a river reaches. A river that runs into a lake follows its
    shore like any coast, anticlockwise round the water, land on the right
  * river lines (OpenStreetMap waterway=river, from ../rivers/fetch2.py), walked both ways: up one bank and
    down the other
  * links: a river end joined to a coast, a lake shore or another river across a short gap

The rule. Arriving at a junction, leave by the next edge anticlockwise from the one you came in on, so
the land stays on your right. Going upstream that is the rightmost branch; at a dead end (a source) the
only edge is the one you came in on, so you turn round and come down the other bank. Every directed edge
("dart") then belongs to exactly one closed walk, a "loop": the whole shore of one piece of land. A
continent's loop is its coastline plus both banks of every river that reaches it.

    python network.py  -> data/network.npz, data/network_stats.json
"""
import os, re, glob, json, time, math
import numpy as np
import numba as nb
import shapely
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RIV = os.path.join(os.path.dirname(HERE), "rivers", "data", "full")
R_EARTH = 6371.0088
COAST_SNAP_M = 1000.0     # a dangling river end this close to the coastline reaches the sea
LAKE_SNAP_M = 300.0       # ... this close to a lake shore reaches the lake
GAP_M = 1000.0            # ... this close to a separate piece of the river network joins it (a mapping gap, a dam)
BRIDGE_KM = 150.0         # ... the nearest end of a separate piece of the same river (Wikidata) within this
SIMPLIFY_DEG = 0.0008     # drawing and turning geometry (~90 m); every length is measured at full resolution
CANAL = re.compile(r"(\bcanal\b|\bkanal\b|канал|运河|水渠|\bnahar\b|\bcanale\b|\bkanaal\b)", re.I)
K_COAST, K_LAKE, K_RIVER, K_LINK = 0, 1, 2, 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# --------------------------------------------------------------------------- numba kernels
@nb.njit(cache=True)
def hav(lo1, la1, lo2, la2):
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(a, 1.0)))


@nb.njit(cache=True)
def seg_cum(xy, off):
    """cumulative km along each line, restarting at 0 at every line start"""
    out = np.zeros(xy.shape[0])
    for p in range(len(off) - 1):
        for i in range(off[p] + 1, off[p + 1]):
            out[i] = out[i - 1] + hav(xy[i - 1, 0], xy[i - 1, 1], xy[i, 0], xy[i, 1])
    return out


@nb.njit(cache=True)
def dedupe(xy, off):
    keep = np.ones(xy.shape[0], np.bool_)
    for p in range(len(off) - 1):
        for i in range(off[p] + 1, off[p + 1]):
            if xy[i, 0] == xy[i - 1, 0] and xy[i, 1] == xy[i - 1, 1]:
                keep[i] = False
    return keep


@nb.njit(cache=True)
def simplify_ranges(xy, ra, rb, tol):
    """Douglas-Peucker on every range xy[ra[k]..rb[k]] (inclusive; lon scaled by cos lat). Returns the
    kept-vertex mask (range ends always kept)."""
    keep = np.zeros(xy.shape[0], np.bool_)
    stack = np.empty((1 << 22, 2), np.int64)
    for k in range(len(ra)):
        a, b = ra[k], rb[k]
        keep[a] = True
        keep[b] = True
        top = 0
        stack[0, 0] = a; stack[0, 1] = b
        while top >= 0:
            i0, i1 = stack[top, 0], stack[top, 1]
            top -= 1
            if i1 <= i0 + 1:
                continue
            c = math.cos(math.radians(0.5 * (xy[i0, 1] + xy[i1, 1])))
            x0, y0 = xy[i0, 0] * c, xy[i0, 1]
            dx, dy = xy[i1, 0] * c - x0, xy[i1, 1] - y0
            L2 = dx * dx + dy * dy
            best, bi = -1.0, -1
            for i in range(i0 + 1, i1):
                px, py = xy[i, 0] * c - x0, xy[i, 1] - y0
                if L2 > 0:
                    u = (px * dx + py * dy) / L2
                    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
                    ex, ey = px - u * dx, py - u * dy
                else:
                    ex, ey = px, py
                d = ex * ex + ey * ey
                if d > best:
                    best, bi = d, i
            if best > tol * tol:
                keep[bi] = True
                top += 1
                stack[top, 0] = i0; stack[top, 1] = bi
                top += 1
                stack[top, 0] = bi; stack[top, 1] = i1
    return keep


@nb.njit(cache=True)
def end_angles(gxy, goff):
    """direction (radians, anticlockwise from east) of each edge leaving its start, and leaving its end"""
    m = len(goff) - 1
    a0 = np.zeros(m)
    a1 = np.zeros(m)
    for e in range(m):
        s, t = goff[e], goff[e + 1] - 1
        j = s + 1
        while j < t and gxy[j, 0] == gxy[s, 0] and gxy[j, 1] == gxy[s, 1]:
            j += 1
        dl = ((gxy[j, 0] - gxy[s, 0] + 180.0) % 360.0) - 180.0
        a0[e] = math.atan2(gxy[j, 1] - gxy[s, 1], dl * math.cos(math.radians(gxy[s, 1])))
        j = t - 1
        while j > s and gxy[j, 0] == gxy[t, 0] and gxy[j, 1] == gxy[t, 1]:
            j -= 1
        dl = ((gxy[j, 0] - gxy[t, 0] + 180.0) % 360.0) - 180.0
        a1[e] = math.atan2(gxy[j, 1] - gxy[t, 1], dl * math.cos(math.radians(gxy[t, 1])))
    return a0, a1


@nb.njit(cache=True)
def trace_faces(nn, eu, ev, active, slot_start, slot_edge, slot_side):
    """For every active dart (2e: u->v, 2e+1: v->u), the next dart under the right-hand rule; then the
    loops. slot arrays hold each node's edge-ends in anticlockwise order."""
    m = len(eu)
    pos_of = np.full(2 * m, -1, np.int64)
    for n in range(nn):
        for j in range(slot_start[n], slot_start[n + 1]):
            pos_of[2 * slot_edge[j] + slot_side[j]] = j
    nxt = np.full(2 * m, -1, np.int64)
    skipped = 0
    for d in range(2 * m):
        if not active[d]:
            continue
        e = d >> 1
        far_side = 1 if (d & 1) == 0 else 0           # the edge-end we arrive at
        node = ev[e] if (d & 1) == 0 else eu[e]
        j = pos_of[2 * e + far_side]
        a, b = slot_start[node], slot_start[node + 1]
        k = j + 1 if j + 1 < b else a                 # next anticlockwise
        d2 = 2 * slot_edge[k] + slot_side[k]          # leaving: side 0 = edge starts here -> 2e'
        tries = 0
        while not active[d2] and tries < b - a:
            skipped += 1
            k = k + 1 if k + 1 < b else a
            d2 = 2 * slot_edge[k] + slot_side[k]
            tries += 1
        nxt[d] = d2
    loop = np.full(2 * m, -1, np.int64)
    order = np.full(2 * m, -1, np.int64)
    nl = 0
    for d in range(2 * m):
        if not active[d] or loop[d] >= 0:
            continue
        x = d
        i = 0
        while loop[x] < 0:
            loop[x] = nl
            order[x] = i
            i += 1
            x = nxt[x]
        nl += 1
    return nxt, loop, order, nl, skipped


@nb.njit(cache=True)
def mark_shared(keys, order):
    """True for every vertex whose key occurs more than once (order = argsort of keys)"""
    out = np.zeros(len(keys), np.bool_)
    n = len(order)
    i = 0
    while i < n:
        j = i + 1
        while j < n and keys[order[j]] == keys[order[i]]:
            j += 1
        if j - i > 1:
            for k in range(i, j):
                out[order[k]] = True
        i = j
    return out


@nb.njit(cache=True)
def range_km(xy, ra, rb, sc):
    """haversine km along xy[ra[k]..rb[k]] (coordinates times sc = degrees)"""
    out = np.zeros(len(ra))
    for k in range(len(ra)):
        s = 0.0
        for i in range(ra[k], rb[k]):
            s += hav(xy[i, 0] * sc, xy[i, 1] * sc, xy[i + 1, 0] * sc, xy[i + 1, 1] * sc)
        out[k] = s
    return out


@nb.njit(cache=True)
def line_km(xy, off, sc):
    out = np.zeros(len(off) - 1)
    for p in range(len(off) - 1):
        s = 0.0
        for i in range(off[p], off[p + 1] - 1):
            s += hav(xy[i, 0] * sc, xy[i, 1] * sc, xy[i + 1, 0] * sc, xy[i + 1, 1] * sc)
        out[p] = s
    return out


@nb.njit(cache=True)
def simplify_ranges_i(xy, ra, rb, tol, sc):
    """simplify_ranges for int32 e7 coordinates"""
    keep = np.zeros(xy.shape[0], np.bool_)
    stack = np.empty((1 << 22, 2), np.int64)
    for k in range(len(ra)):
        a, b = ra[k], rb[k]
        keep[a] = True
        keep[b] = True
        top = 0
        stack[0, 0] = a; stack[0, 1] = b
        while top >= 0:
            i0, i1 = stack[top, 0], stack[top, 1]
            top -= 1
            if i1 <= i0 + 1:
                continue
            c = math.cos(math.radians(0.5 * (xy[i0, 1] + xy[i1, 1]) * sc))
            x0, y0 = xy[i0, 0] * sc * c, xy[i0, 1] * sc
            dx, dy = xy[i1, 0] * sc * c - x0, xy[i1, 1] * sc - y0
            L2 = dx * dx + dy * dy
            best, bi = -1.0, -1
            for i in range(i0 + 1, i1):
                px, py = xy[i, 0] * sc * c - x0, xy[i, 1] * sc - y0
                if L2 > 0:
                    u = (px * dx + py * dy) / L2
                    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
                    ex, ey = px - u * dx, py - u * dy
                else:
                    ex, ey = px, py
                d = ex * ex + ey * ey
                if d > best:
                    best, bi = d, i
            if best > tol * tol:
                keep[bi] = True
                top += 1
                stack[top, 0] = i0; stack[top, 1] = bi
                top += 1
                stack[top, 0] = bi; stack[top, 1] = i1
    return keep


def in_sorted(a, sorted_u):
    """membership test against a sorted unique array, without np.isin's big sorts"""
    p = np.minimum(np.searchsorted(sorted_u, a), len(sorted_u) - 1)
    return sorted_u[p] == a


def key_of(lon_e7, lat_e7):
    lon = np.where(lon_e7 == -1800000000, 1800000000, lon_e7).astype(np.int64)
    return (lon + 1800000000) * 1800000001 + (np.asarray(lat_e7).astype(np.int64) + 900000000)


def key_deg(xy):
    return key_of(np.rint(xy[:, 0] * 1e7).astype(np.int64), np.rint(xy[:, 1] * 1e7).astype(np.int64))


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def gather(xy, ra, rb, keep):
    """per range [ra, rb], the kept vertices of xy, concatenated; returns (vertices, count per range)"""
    kv = np.nonzero(keep)[0]
    lo = np.searchsorted(kv, ra)
    hi = np.searchsorted(kv, rb, side="right")
    n = hi - lo
    idx = np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(n.sum())
    return xy[kv[idx]], n


# --------------------------------------------------------------------------- loading
def load_rivers():
    xs, offs, nm, en, wd = [], [np.zeros(1, np.int64)], [], [], []
    lx, loffs, la = [], [np.zeros(1, np.int64)], []
    vb = lb = 0
    for p in sorted(glob.glob(os.path.join(RIV, "part-*.npz"))):
        z = np.load(p, allow_pickle=True)
        xs.append(z["xy"]); offs.append(z["off"][1:] + vb); vb += len(z["xy"])
        nm.append(z["name"]); en.append(z["name_en"]); wd.append(z["wikidata"])
        lx.append(z["lake_xy"]); loffs.append(z["lake_off"][1:] + lb); lb += len(z["lake_xy"])
        la.append(z["lake_km2"])
    xy, off = np.concatenate(xs), np.concatenate(offs)
    name, name_en, wd = np.concatenate(nm), np.concatenate(en), np.concatenate(wd)
    # Everything OpenStreetMap tags waterway=river is kept, canal names included: ../rivers/systems.py left out
    # lines named "canal", but a canalised river channel ("Bijlands Kanaal", the Rhine at the Dutch border) is
    # still the river, and leaving it out cuts the Rhine off from the sea. The fish can swim a canal anyway.
    canal = np.zeros(len(name), bool)
    lens = np.diff(off)
    xy = xy[np.repeat(~canal, lens)]
    off = np.r_[0, np.cumsum(lens[~canal])]
    name, name_en, wd = name[~canal], name_en[~canal], wd[~canal]
    keep = dedupe(xy, off)
    lid = np.repeat(np.arange(len(off) - 1), np.diff(off))
    xy, lid = xy[keep], lid[keep]
    cnt = np.bincount(lid, minlength=len(off) - 1)
    ok = cnt >= 2
    xy = xy[ok[lid]]
    off = np.r_[0, np.cumsum(cnt[ok])]
    # lakes and reservoirs only (fetch_lakes.py): ../rivers kept every water polygon, river channels included
    lx, loffs, lb = [], [np.zeros(1, np.int64)], 0
    for p in sorted(glob.glob(os.path.join(DATA, "lakes", "part-*.npz"))):
        z = np.load(p, allow_pickle=True)
        lx.append(z["xy"]); loffs.append(z["off"][1:] + lb); lb += len(z["xy"])
    return xy, off, name[ok], name_en[ok], wd[ok], int(canal.sum()), (np.concatenate(lx), np.concatenate(loffs))


def lake_polygons(lxy, loff):
    """every lake and reservoir over 5 km2, touching parts merged (a reservoir split at a dam is one
    lake); outer shores only (islands in lakes are not traced)"""
    parts = []
    for k in range(len(loff) - 1):
        ring = lxy[loff[k]:loff[k + 1]].astype(np.float64)
        if len(ring) >= 4:
            parts.append(shapely.Polygon(ring))
    parts = shapely.get_parts(shapely.make_valid(np.array(parts, dtype=object)))
    parts = parts[shapely.get_type_id(parts) == 3]
    polys = shapely.get_parts(shapely.union_all(parts))
    polys = polys[shapely.get_type_id(polys) == 3]
    polys = shapely.polygons(shapely.get_exterior_ring(polys))
    lat = shapely.get_coordinates(shapely.centroid(polys))[:, 1]
    km2 = shapely.area(polys) * 111.32 ** 2 * np.cos(np.radians(lat))
    return polys[km2 >= 5.0]


@nb.njit(cache=True)
def line_boxes(xy, off):
    n = len(off) - 1
    b = np.empty((n, 4))
    for p in range(n):
        x0 = y0 = 1e18
        x1 = y1 = -1e18
        for i in range(off[p], off[p + 1]):
            x, y = xy[i, 0], xy[i, 1]
            x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
        b[p, 0], b[p, 1], b[p, 2], b[p, 3] = x0, y0, x1, y1
    return b


def clip(xyi, off, polys, what):
    """cut every line (int32 e7 coordinates) where it runs inside one of `polys`; returns the new lines and,
    per new line, the index of the line it came from"""
    bx = line_boxes(xyi, off) * 1e-7
    tree = shapely.STRtree(polys)
    li, pi = tree.query(shapely.box(bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]), predicate="intersects")
    cand = np.unique(li)
    lens = np.diff(off)
    nl = len(lens)
    is_c = np.zeros(nl, bool); is_c[cand] = True
    sel = np.repeat(is_c, lens)
    lines = shapely.linestrings(xyi[sel] * 1e-7, indices=np.repeat(np.arange(len(cand)), lens[cand]))
    pos = np.searchsorted(cand, li)
    o = np.argsort(pos, kind="stable")
    pos, pi = pos[o], pi[o]
    st = np.r_[0, np.nonzero(np.diff(pos))[0] + 1]
    en = np.r_[st[1:], len(pos)]
    cut = np.empty(len(cand), dtype=object)
    for k, (a, b) in enumerate(zip(st, en)):
        cut[k] = polys[pi[a]] if b - a == 1 else shapely.union_all(polys[pi[a:b]])
    hit = shapely.intersects(lines, cut)
    out = lines.copy()
    out[hit] = shapely.difference(lines[hit], cut[hit])
    parts, pidx = shapely.get_parts(out, return_index=True)
    ok = (shapely.get_type_id(parts) == 1) & (shapely.get_num_coordinates(parts) >= 2)
    parts, pidx = parts[ok], pidx[ok]
    coords, cix = shapely.get_coordinates(parts, return_index=True)
    keepl = np.nonzero(~is_c)[0]
    del lines
    xy2 = np.concatenate([xyi[~sel], np.rint(coords * 1e7).astype(np.int32)])
    n2 = np.r_[lens[keepl], np.bincount(cix, minlength=len(parts))]
    src = np.r_[keepl, cand[pidx]]
    log(f"{what} clip: {len(cand):,} lines near, {hit.sum():,} cut; {nl:,} -> {len(n2):,} lines")
    return xy2, np.r_[0, np.cumsum(n2)], src


def main():
    os.makedirs(DATA, exist_ok=True)
    stats = {}
    # ---------------- coast
    cz = np.load(os.path.join(DATA, "coast_rings.npz"))
    cxy, coff, ckm = cz["xy"], cz["off"], cz["km"]
    nring = len(coff) - 1
    log(f"coast: {nring:,} rings, {len(cxy):,} vertices, {ckm.sum():,.0f} km")
    # ---------------- rivers and lakes
    rxy7, roff, rname, rname_en, rwd, ncanal, (lxy, loff) = load_rivers()
    stats["river_lines"] = int(len(roff) - 1); stats["canals_left_out"] = ncanal
    log(f"rivers: {len(roff) - 1:,} lines, {len(rxy7):,} vertices ({ncanal:,} canals left out)")
    rxy = rxy7                                         # int32, degrees x 1e7, all the way through
    del rxy7
    stats["river_km_raw"] = float(line_km(rxy, roff, 1e-7).sum())
    polys = lake_polygons(lxy, loff)
    log(f"lakes: {len(polys):,} (>= 5 km2, touching parts merged)")
    rxy, roff, src = clip(rxy, roff, polys, "lake")
    rname, rname_en, rwd = rname[src], rname_en[src], rwd[src]
    nline = len(roff) - 1
    stats["river_km_outside_lakes"] = float(line_km(rxy, roff, 1e-7).sum())
    log(f"rivers outside lakes: {stats['river_km_outside_lakes']:,.0f} km")
    rline = np.repeat(np.arange(nline, dtype=np.int32), np.diff(roff))

    # ---------------- vertex identity
    rk = key_of(rxy[:, 0], rxy[:, 1])
    ck = key_deg(cxy)
    o = np.argsort(rk)
    shared = mark_shared(rk, o)
    del o
    on_coast = in_sorted(rk, np.unique(ck))
    isend = np.zeros(len(rk), bool)
    isend[roff[:-1]] = True
    isend[roff[1:] - 1] = True
    # Terminal ends: a line end no other line passes through, where every line meeting there starts there
    # (a source, or the top of a stretch below a gap) or every one ends there (a mouth, or a braid's
    # channels stopping together above a dam). These look for somewhere to connect.
    st_k, en_k = rk[roff[:-1]], rk[roff[1:] - 1]
    through = np.unique(rk[shared & ~isend])
    allk = np.r_[st_k, en_k]
    uk, inv = np.unique(allk, return_inverse=True)
    n_st = np.bincount(inv[:len(st_k)], minlength=len(uk))
    n_en = np.bincount(inv[len(st_k):], minlength=len(uk))
    term_k = uk[((n_st == 0) | (n_en == 0)) & ~in_sorted(uk, through)]
    rep = np.r_[roff[:-1], roff[1:] - 1]
    _, first = np.unique(allk, return_index=True)
    rep_of_k = rep[first]                                  # one vertex per endpoint key (uk order)
    dang = rep_of_k[in_sorted(uk, term_k)]
    dang = dang[~on_coast[dang]]
    del st_k, en_k, through, allk, uk, inv, n_st, n_en, term_k, rep, first, rep_of_k
    log(f"river vertices on a coastline vertex: {on_coast.sum():,}; terminal ends: {len(dang):,}")

    # ---------------- where each end connects: the nearest point of each coast ring within COAST_SNAP_M, of
    # each lake within LAKE_SNAP_M, and of each separate piece of the river network within GAP_M (up to 3)
    tree = cKDTree(xyz(rxy[dang, 0] * 1e-7, rxy[dang, 1] * 1e-7))
    CS, CK, CT, CD, CG = [], [], [], [], []

    def offer(pts, bound_m, kind, group, exclude=None, chunk=4_000_000, sc=1.0):
        for c0 in range(0, len(pts), chunk):
            p = pts[c0:c0 + chunk] * sc
            d, k = tree.query(xyz(p[:, 0], p[:, 1]), k=4, distance_upper_bound=bound_m / 1000 / R_EARTH, workers=-1)
            S_, D_, Q_ = [], [], []
            for col in range(4):
                hit = np.nonzero(np.isfinite(d[:, col]))[0]
                if len(hit):
                    S_.append(k[hit, col]); D_.append(d[hit, col] * R_EARTH * 1000); Q_.append(c0 + hit)
            if not S_:
                continue
            s_, dd, q = np.concatenate(S_), np.concatenate(D_), np.concatenate(Q_)
            if exclude is not None:
                ok = exclude[q] != exclude[dang[s_]]
                s_, dd, q = s_[ok], dd[ok], q[ok]
            g_ = group[q]
            oo = np.lexsort((dd, g_, s_))
            s_, dd, q, g_ = s_[oo], dd[oo], q[oo], g_[oo]
            f = np.r_[True, (s_[1:] != s_[:-1]) | (g_[1:] != g_[:-1])]
            CS.append(s_[f]); CD.append(dd[f]); CT.append(q[f]); CG.append(g_[f]); CK.append(np.full(f.sum(), kind))

    cring = np.repeat(np.arange(nring, dtype=np.int64), np.diff(coff))
    offer(cxy, COAST_SNAP_M, K_COAST, cring)
    del cring
    lrings = shapely.segmentize(shapely.get_exterior_ring(polys), 0.0005)     # shores densified to ~50 m
    lco, lring_of = shapely.get_coordinates(lrings, return_index=True)
    loff_d = np.r_[0, np.cumsum(np.bincount(lring_of, minlength=len(polys)))]
    offer(lco, LAKE_SNAP_M, K_LAKE, lring_of)
    # a gap link only joins two separate pieces of the river network: the nearest line of the end's own piece
    # (a parallel channel at a dam) would add nothing
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    si = np.nonzero(shared)[0]
    oo = np.argsort(rk[si])
    ks, ls = rk[si][oo], rline[si][oo]
    same = ks[1:] == ks[:-1]
    npc, lcomp = connected_components(coo_matrix((np.ones(same.sum()), (ls[:-1][same], ls[1:][same])), shape=(nline, nline)),
                                      directed=False)
    del si, oo, ks, ls, same
    vcomp = lcomp[rline].astype(np.int32)
    log(f"river network: {npc:,} separate pieces before gap links")
    stats["river_pieces_before_links"] = int(npc)
    offer(rxy, GAP_M, K_RIVER, vcomp, exclude=vcomp, sc=1e-7)
    # the same river across a hole in its mapping (a reservoir drawn only as a water area): an end joins the
    # nearest end of a separate piece of the same river (same Wikidata item) within BRIDGE_KM
    wd_u, wd_id = np.unique(rwd[rline[dang]], return_inverse=True)
    has = wd_u[wd_id] != ""
    dcomp = vcomp[dang]
    hi = np.nonzero(has)[0]
    dd_, nb_ = tree.query(xyz(rxy[dang[hi], 0] * 1e-7, rxy[dang[hi], 1] * 1e-7), k=32,
                          distance_upper_bound=BRIDGE_KM / R_EARTH, workers=-1)
    B_i, B_j, B_d = [], [], []
    for col in range(1, 32):
        ok = np.isfinite(dd_[:, col])
        j = np.where(ok, nb_[:, col], 0)
        ok &= (wd_id[j] == wd_id[hi]) & (dcomp[j] != dcomp[hi])
        B_i.append(hi[ok]); B_j.append(j[ok]); B_d.append(dd_[ok, col] * R_EARTH * 1000)
    B_i, B_j, B_d = np.concatenate(B_i), np.concatenate(B_j), np.concatenate(B_d)
    oo = np.lexsort((B_d, B_i))
    B_i, B_j, B_d = B_i[oo], B_j[oo], B_d[oo]
    f = np.r_[True, B_i[1:] != B_i[:-1]]
    B_i, B_j, B_d = B_i[f], B_j[f], B_d[f]
    pk = np.unique(np.c_[np.minimum(B_i, B_j), np.maximum(B_i, B_j)], axis=0, return_index=True)[1]
    B_i, B_j, B_d = B_i[pk], B_j[pk], B_d[pk]
    stats["same_river_bridges"] = int(len(B_i))
    stats["same_river_bridge_km"] = float(B_d.sum() / 1000)
    log(f"same-river bridges: {len(B_i):,}, {B_d.sum() / 1000:,.0f} km")
    del vcomp, dcomp
    CS, CK, CT, CD, CG = (np.concatenate(x) for x in (CS, CK, CT, CD, CG))
    # per end and target piece the nearest point (chunks can repeat a pair); then each end's 3 nearest pieces
    oo = np.lexsort((CD, CG, CK, CS))
    CS, CK, CT, CD, CG = CS[oo], CK[oo], CT[oo], CD[oo], CG[oo]
    f = np.r_[True, (CS[1:] != CS[:-1]) | (CK[1:] != CK[:-1]) | (CG[1:] != CG[:-1])]
    CS, CK, CT, CD = CS[f], CK[f], CT[f], CD[f]
    oo = np.lexsort((CD, CS))
    CS, CK, CT, CD = CS[oo], CK[oo], CT[oo], CD[oo]
    st_ = np.r_[0, np.nonzero(np.diff(CS))[0] + 1]
    rank = np.arange(len(CS)) - np.repeat(st_, np.diff(np.r_[st_, len(CS)]))
    keep = rank < 3
    a_end, a_kind, a_tgt, a_d = dang[CS[keep]], CK[keep], CT[keep], CD[keep]
    a_end = np.r_[a_end, dang[B_i]]
    a_kind = np.r_[a_kind, np.full(len(B_i), K_RIVER)]
    a_tgt = np.r_[a_tgt, dang[B_j]]
    a_d = np.r_[a_d, B_d]
    del CS, CK, CT, CD, CG, tree
    for k_, nm_ in ((K_COAST, "coast"), (K_LAKE, "lake_shore"), (K_RIVER, "river_gap")):
        stats[f"links_to_{nm_}"] = int((a_kind == k_).sum())
    stats["terminal_ends"] = int(len(dang))
    stats["terminal_ends_unlinked"] = int(len(dang) - len(np.unique(a_end)))
    log("links:", {k: v for k, v in stats.items() if k.startswith("links_to") or k.startswith("terminal")})

    # ---------------- nodes on each kind of line
    rnode = shared | isend | on_coast
    rnode[a_tgt[a_kind == K_RIVER]] = True
    cnode = in_sorted(ck, np.unique(rk[rnode]))
    cnode[a_tgt[a_kind == K_COAST]] = True
    lnode = np.zeros(len(lco), bool)
    lnode[a_tgt[a_kind == K_LAKE]] = True
    del shared
    # artificial coast segments: the antimeridian cut and the South Pole edge
    lon, lat = cxy[:, 0], cxy[:, 1]
    art = np.zeros(len(cxy), bool)                  # segment i -> i+1 is artificial
    same_ring = np.ones(len(cxy) - 1, bool)
    same_ring[coff[1:-1] - 1] = False
    art[:-1] = same_ring & (((np.abs(lon[:-1]) == 180.0) & (lon[:-1] == lon[1:])) | ((lat[:-1] <= -89.99999) & (lat[1:] <= -89.99999)))
    cnode[np.nonzero(art)[0]] = True
    cnode[np.nonzero(art)[0] + 1] = True
    stats["artificial_coast_segments"] = int(art.sum())
    # a ring's last vertex repeats its first
    first_node = cnode[coff[:-1]] | cnode[coff[1:] - 1]
    cnode[coff[:-1]] = first_node
    cnode[coff[1:] - 1] = False
    lnode[loff_d[1:] - 1] = False

    # ---------------- chains of a ring between consecutive nodes (unrolled across the ring's closing vertex)
    def chains(xy, off, node_mask, skip_art=None):
        ring_of = np.repeat(np.arange(len(off) - 1), np.diff(off))
        nidx = np.nonzero(node_mask)[0]
        nr = ring_of[nidx]
        st = np.r_[0, np.nonzero(np.diff(nr))[0] + 1]
        en = np.r_[st[1:], len(nidx)]
        pieces, ring_ids, va, vb = [], [], [], []
        for s0, s1 in zip(st, en):
            r = nr[s0]
            vs = nidx[s0:s1]
            last = off[r + 1] - 1
            for j in range(len(vs)):
                a = vs[j]
                if skip_art is not None and skip_art[a]:
                    continue
                if j + 1 < len(vs):
                    pieces.append(xy[a:vs[j + 1] + 1]); vb.append(vs[j + 1])
                else:
                    pieces.append(np.concatenate([xy[a:last + 1], xy[off[r] + 1:vs[0] + 1]])); vb.append(vs[0])
                ring_ids.append(r); va.append(a)
        n = np.array([len(p) for p in pieces], np.int64)
        flat = np.concatenate(pieces) if pieces else np.zeros((0, 2))
        go = np.r_[0, np.cumsum(n)]
        km = seg_cum(flat, go)[go[1:] - 1] if len(n) else np.zeros(0)
        return flat, go, np.array(ring_ids, np.int64), km, ring_of, np.array(va, np.int64), np.array(vb, np.int64)

    c_flat, c_go, c_ring, c_km, ring_of_v, _, _ = chains(cxy, coff, cnode, art)
    has_node = np.zeros(nring, bool)
    has_node[np.unique(ring_of_v[cnode])] = True
    pure = np.nonzero(~has_node)[0]
    log(f"coast: {len(c_km):,} chains on {has_node.sum():,} rings ({c_km.sum():,.0f} km); "
        f"{len(pure):,} rings no river reaches ({ckm[pure].sum():,.0f} km)")
    l_flat, l_go, l_ring, l_km, _, l_va, l_vb = chains(lco, loff_d, lnode)
    lake_ccw = shapely.is_ccw(lrings)
    log(f"lake shores: {len(l_km):,} chains on {len(np.unique(l_ring)):,} lakes, {l_km.sum():,.0f} km")
    stats.update(coast_km=float(c_km.sum() + ckm[pure].sum()), coast_rings=int(nring), coast_rings_untouched=int(len(pure)),
                 lake_shore_km=float(l_km.sum()), lakes_reached=int(len(np.unique(l_ring))))

    # ---------------- river edges: each line cut at its nodes
    nidx = np.nonzero(rnode)[0]
    lid_n = rline[nidx]
    same = lid_n[1:] == lid_n[:-1]
    ra, rb, r_line = nidx[:-1][same], nidx[1:][same], lid_n[:-1][same]
    r_km = range_km(rxy, ra, rb, 1e-7)
    ok = r_km > 0
    ra, rb, r_line, r_km = ra[ok], rb[ok], r_line[ok], r_km[ok]
    # the same stretch mapped twice (same ends, same length) is walked once
    lo_, hi_ = np.minimum(rk[ra], rk[rb]), np.maximum(rk[ra], rk[rb])
    _, uq = np.unique(np.c_[lo_, hi_, np.rint(r_km * 1000).astype(np.int64)], axis=0, return_index=True)
    uq.sort()
    stats["river_duplicates_dropped"] = int(len(ra) - len(uq))
    ra, rb, r_line, r_km = ra[uq], rb[uq], r_line[uq], r_km[uq]
    stats["river_km"] = float(r_km.sum())
    log(f"river edges: {len(ra):,}, {r_km.sum():,.0f} km ({stats['river_duplicates_dropped']:,} duplicates dropped)")

    # ---------------- simplified geometry for every edge, one array
    del rk
    kr = simplify_ranges_i(rxy, ra, rb, SIMPLIFY_DEG, 1e-7)
    g_r, n_r = gather(rxy, ra, rb, kr)
    g_r = g_r * 1e-7
    kc = simplify_ranges(c_flat, c_go[:-1], c_go[1:] - 1, SIMPLIFY_DEG)
    g_c, n_c = gather(c_flat, c_go[:-1], c_go[1:] - 1, kc)
    pa, pb = coff[pure], coff[pure + 1] - 1
    kp = simplify_ranges(cxy, pa, pb, SIMPLIFY_DEG)
    g_p, n_p = gather(cxy, pa, pb, kp)
    kl = simplify_ranges(l_flat, l_go[:-1], l_go[1:] - 1, SIMPLIFY_DEG)
    g_l, n_l = gather(l_flat, l_go[:-1], l_go[1:] - 1, kl)
    # links: river end -> target point
    tgt_xy = np.where((a_kind == K_COAST)[:, None], cxy[np.where(a_kind == K_COAST, a_tgt, 0)],
                      np.where((a_kind == K_LAKE)[:, None], lco[np.where(a_kind == K_LAKE, a_tgt, 0)],
                               rxy[np.where(a_kind == K_RIVER, a_tgt, 0)] * 1e-7))
    g_k = np.stack([rxy[a_end] * 1e-7, tgt_xy], 1).reshape(-1, 2)
    n_k = np.full(len(a_end), 2, np.int64)
    gxy = np.concatenate([g_r, g_c, g_p, g_l, g_k])
    gn = np.r_[n_r, n_c, n_p, n_l, n_k]
    goff = np.r_[0, np.cumsum(gn)].astype(np.int64)
    m = len(gn)
    kind = np.r_[np.full(len(n_r), K_RIVER), np.full(len(n_c) + len(n_p), K_COAST), np.full(len(n_l), K_LAKE),
                 np.full(len(n_k), K_LINK)].astype(np.int8)
    km = np.r_[r_km, c_km, ckm[pure], l_km, a_d / 1000]
    line = np.r_[r_line, np.full(m - len(r_line), -1)]
    ring = np.r_[np.full(len(n_r), -1), c_ring, pure, l_ring, np.full(len(n_k), -1)]
    log(f"edges: {m:,}; geometry {len(gxy):,} vertices")
    del rxy, c_flat, l_flat, kr, kc, kp, kl

    # ---------------- nodes: one per distinct location; rings no river reaches get a private node
    k0 = key_deg(gxy[goff[:-1]])
    k1 = key_deg(gxy[goff[1:] - 1])
    # a lake shore can touch itself (reservoir parts merged at a point): every shore vertex is its own node,
    # so the shore never passes through one node twice
    li_ = np.nonzero(kind == K_LAKE)[0]
    k0[li_] = -(l_va + 1)
    k1[li_] = -(l_vb + 1)
    lk_ = np.nonzero(kind == K_LINK)[0][a_kind == K_LAKE]
    k1[lk_] = -(a_tgt[a_kind == K_LAKE] + 1)
    ukeys, inv = np.unique(np.r_[k0, k1], return_inverse=True)
    eu, ev = inv[:m].copy(), inv[m:].copy()
    nn = len(ukeys)
    ip = np.nonzero(kind == K_COAST)[0][len(n_c):]   # the untouched rings
    eu[ip] = ev[ip] = nn + np.arange(len(ip))
    nn += len(ip)
    # zero-length links (end already on its target) are dropped: their two ends are one node anyway
    zl = (kind == K_LINK) & (eu == ev)
    stats["links"] = int((kind == K_LINK).sum() - zl.sum())
    log(f"nodes: {nn:,}")

    # ---------------- darts: rivers and links both ways; shores one way, land on the right
    active = np.ones(2 * m, bool)
    shore = (kind == K_COAST) | (kind == K_LAKE)
    lake_rev = np.zeros(m, bool)
    li = np.nonzero(kind == K_LAKE)[0]
    lake_rev[li] = ~lake_ccw[ring[li]]               # walk lake shores anticlockwise round the water
    active[2 * np.nonzero(shore & ~lake_rev)[0] + 1] = False
    active[2 * np.nonzero(shore & lake_rev)[0]] = False
    active[2 * np.nonzero(zl)[0]] = False
    active[2 * np.nonzero(zl)[0] + 1] = False

    # ---------------- rotation system: each node's edge-ends sorted anticlockwise
    a0, a1 = end_angles(gxy, goff)
    E = np.r_[np.arange(m), np.arange(m)]
    S = np.r_[np.zeros(m, np.int64), np.ones(m, np.int64)]
    Nn = np.r_[eu, ev]
    ang = np.r_[a0, a1]
    keep_end = ~np.r_[zl, zl]
    E, S, Nn, ang = E[keep_end], S[keep_end], Nn[keep_end], ang[keep_end]
    # A shore node: the shore arrives and leaves; rivers there must turn off on the land side. The shore
    # leaving gets key +10, the shore arriving -10, and each river its angle measured from the arriving
    # shore, folded into the land side (a river drawn a little out to sea sorts next to the nearer shore).
    sh = shore[E]
    out_dart = 2 * E + S                              # dart leaving the node along this edge-end
    in_dart = 2 * E + (1 - S)                         # dart arriving along it
    is_out = sh & active[out_dart]
    is_in = sh & active[in_dart]
    n_out = np.bincount(Nn[is_out], minlength=nn)
    n_in = np.bincount(Nn[is_in], minlength=nn)
    n_sh = np.bincount(Nn[sh], minlength=nn)
    simple = (n_out == 1) & (n_in == 1) & (n_sh == 2)
    th_in = np.zeros(nn); th_out = np.zeros(nn)
    th_in[Nn[is_in]] = ang[is_in]
    th_out[Nn[is_out]] = ang[is_out]
    key = ang.copy()
    sn = simple[Nn]
    two = 2 * np.pi
    w = np.mod(th_out - th_in, two)
    phi = np.mod(ang - th_in[Nn], two)
    wv = w[Nn]
    fold = np.where(phi < wv, phi, np.where(phi < wv + (two - wv) / 2, phi, phi - two))
    key[sn] = fold[sn]
    key[sn & is_out] = 10.0
    key[sn & is_in] = -10.0
    stats["shore_nodes"] = int(simple.sum())
    stats["shore_nodes_complex"] = int(((n_sh > 0) & ~simple & (np.bincount(Nn, minlength=nn) > n_sh)).sum())
    o = np.lexsort((key, Nn))
    slot_edge, slot_side = E[o].astype(np.int64), S[o].astype(np.int64)
    slot_start = np.r_[0, np.cumsum(np.bincount(Nn, minlength=nn))].astype(np.int64)
    nxt, loop, order, nl, skipped = trace_faces(nn, eu.astype(np.int64), ev.astype(np.int64), active,
                                               slot_start, slot_edge, slot_side)
    pred = np.bincount(nxt[active], minlength=2 * m)
    badd = np.nonzero(active & (pred != 1))[0]
    if len(badd):
        bn = np.unique(np.r_[eu[badd >> 1], ev[badd >> 1]])
        for n_ in bn[:12]:
            inc = np.nonzero((eu == n_) | (ev == n_))[0]
            log(f"  awkward node {n_} at {gxy[goff[inc[0]]] if eu[inc[0]] == n_ else gxy[goff[inc[0] + 1] - 1]}: "
                f"edges {[(int(e), int(kind[e]), round(float(km[e]), 3)) for e in inc]}")
    stats["darts"] = int(active.sum())
    stats["loops"] = int(nl)
    stats["skipped_turns"] = int(skipped)
    stats["darts_not_permutation"] = int((pred[active] != 1).sum())
    log(f"loops: {nl:,} from {active.sum():,} darts; turns skipped onto the sea side: {skipped:,}; "
        f"non-permutation darts: {stats['darts_not_permutation']:,}")
    walk_km = float(np.where(shore, 1, 2)[~zl] @ km[~zl])
    stats["walk_km_without_crossings"] = walk_km
    log(f"every dart once: {walk_km:,.0f} km")
    np.savez(os.path.join(DATA, "network.npz"), gxy=gxy.astype(np.float64), goff=goff, eu=eu, ev=ev, kind=kind, km=km,
             line=line, ring=ring, active=active, nxt=nxt, loop=loop, order=order)
    np.savez(os.path.join(DATA, "river_names.npz"), name=rname, name_en=rname_en, wikidata=rwd)
    json.dump(stats, open(os.path.join(DATA, "network_stats.json"), "w"), indent=1)
    log(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
