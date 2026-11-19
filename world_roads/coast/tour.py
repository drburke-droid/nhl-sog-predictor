"""The fish's whole journey: every loop of network.py, joined into one walk.

A loop is the shore of one piece of land. The fish can get from one loop to another three ways:
  * cross a river (free): the two banks of a river channel belong to different loops when the river is
    part of a braid, a delta or a lake chain; the far bank is right there
  * swim across the sea: from one ocean coastline to another (an island). The swim is a straight line,
    counted at full distance, out and back
  * leap: a river basin that no water connects to the sea (the Aral, Lake Eyre, the Okavango, or a river
    whose mapping has a gap) can only be reached over land. Leaps are hypothetical: instantaneous, no
    distance, drawn as rings, like the jumps between road networks in the roads film

The joins form a tree: a minimum spanning tree with crossings first, then swims by length, then leaps
by length (a leap costs more than any swim). The fish walks it depth first. Going round a loop with the
land on its right, wherever a join leaves the loop it goes over, does that loop (and everything behind
it) the same way, and comes back to carry on where it left. So islands are swum out to as the fish
passes the nearest point of the coast.

    python tour.py   -> data/tour.npz (per geometry vertex: time of the pass each way), data/tour_stats.json
"""
import os, json, time, math
import numpy as np
import numba as nb
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R_EARTH = 6371.0088
C_KMS = 299_792.458
SWIM_SAMPLE_KM = 3.0          # coast points for the swim tree, about this far apart
LEAP_SAMPLE_KM = 6.0          # all shore points for the leap tree
LEAP_PENALTY = 1e7            # a leap is chosen only when no water route exists
START = (float(os.environ.get("START_LON", "-114.075")), float(os.environ.get("START_LAT", "51.054")))
T0 = time.time()
K_COAST, K_LAKE, K_RIVER, K_LINK = 0, 1, 2, 3


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


@nb.njit(cache=True)
def hav(lo1, la1, lo2, la2):
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(a, 1.0)))


@nb.njit(cache=True)
def vertex_fracs(gxy, goff):
    """for every geometry vertex, its fraction of the way along its edge (by simplified length)"""
    f = np.zeros(gxy.shape[0])
    for e in range(len(goff) - 1):
        a, b = goff[e], goff[e + 1]
        s = 0.0
        for i in range(a + 1, b):
            s += hav(gxy[i - 1, 0], gxy[i - 1, 1], gxy[i, 0], gxy[i, 1])
            f[i] = s
        if s > 0:
            for i in range(a, b):
                f[i] /= s
        else:
            for i in range(a, b):
                f[i] = (i - a) / max(b - a - 1, 1)
    return f


@nb.njit(cache=True)
def sample_darts(darts, gxy, goff, frac, km, dpos, step):
    """points along each dart about `step` km apart (always its start): lon, lat, dart index, position
    on the loop axis (km)"""
    cap = 0
    for i in range(len(darts)):
        cap += 1 + int(km[darts[i] >> 1] / step)
    out = np.empty((cap, 2))
    od = np.empty(cap, np.int64)
    op = np.empty(cap)
    n = 0
    for i in range(len(darts)):
        d = darts[i]
        e = d >> 1
        a, b = goff[e], goff[e + 1]
        L = km[e]
        nxt = 0.0
        for k in range(b - a):
            j = a + k if (d & 1) == 0 else b - 1 - k
            fr = frac[j] if (d & 1) == 0 else 1.0 - frac[j]
            if fr * L >= nxt and n < cap:
                out[n, 0], out[n, 1] = gxy[j, 0], gxy[j, 1]
                od[n] = i
                op[n] = dpos[i] + fr * L
                n += 1
                nxt = (math.floor(fr * L / step) + 1) * step
    return out[:n], od[:n], op[:n]


@nb.njit(cache=True)
def find(par, x):
    r = x
    while par[r] != r:
        r = par[r]
    while par[x] != r:
        par[x], x = r, par[x]
    return r


@nb.njit(cache=True)
def kruskal(n, a, b, order):
    par = np.arange(n)
    chosen = np.zeros(len(a), np.bool_)
    for k in order:
        ra, rb = find(par, a[k]), find(par, b[k])
        if ra != rb:
            par[ra] = rb
            chosen[k] = True
    return chosen


@nb.njit(cache=True)
def euler_tour(nl, Lkm, adj_start, adj_edge, ea, eb, pa, pb, ekm, root, entry):
    """Depth-first walk of the loop tree. Emits intervals (loop, s0, s1, t0) on each loop's own axis and
    the joins (edge, direction, t0). t is distance walked so far (km); leaps add nothing."""
    cap_i = 4 * nl + 16
    I_loop = np.empty(cap_i, np.int64); I_s0 = np.empty(cap_i); I_s1 = np.empty(cap_i); I_t = np.empty(cap_i)
    J_e = np.empty(2 * nl, np.int64); J_dir = np.empty(2 * nl, np.int8); J_t = np.empty(2 * nl)
    ni = 0
    nj = 0
    # explicit stack: loop, entry position, incoming tree edge, children (sorted) as a slice of a buffer
    st_loop = np.empty(nl, np.int64); st_entry = np.empty(nl); st_in = np.empty(nl, np.int64)
    st_c0 = np.empty(nl, np.int64); st_c1 = np.empty(nl, np.int64); st_k = np.empty(nl, np.int64)
    st_pos = np.empty(nl)                      # position on the loop reached so far (relative to entry)
    cbuf = np.empty(2 * len(ea) + 16, np.int64)
    cpos = np.empty(2 * len(ea) + 16)
    cb = 0
    t = 0.0
    top = 0

    # frame for root
    L = root
    st_loop[0] = L; st_entry[0] = entry; st_in[0] = -1; st_pos[0] = 0.0
    n0 = cb
    for j in range(adj_start[L], adj_start[L + 1]):
        k = adj_edge[j]
        p = pa[k] if ea[k] == L else pb[k]
        cbuf[cb] = k
        cpos[cb] = (p - entry) % Lkm[L] if Lkm[L] > 0 else 0.0
        cb += 1
    o = np.argsort(cpos[n0:cb])
    tmpk = cbuf[n0:cb][o].copy(); tmpp = cpos[n0:cb][o].copy()
    cbuf[n0:cb] = tmpk; cpos[n0:cb] = tmpp
    st_c0[0] = n0; st_c1[0] = cb; st_k[0] = n0
    while top >= 0:
        L = st_loop[top]
        if st_k[top] < st_c1[top]:
            c = st_k[top]
            st_k[top] += 1
            k = cbuf[c]
            rel = cpos[c]
            # walk the loop up to the join
            if rel > st_pos[top]:
                I_loop[ni] = L; I_s0[ni] = st_pos[top]; I_s1[ni] = rel; I_t[ni] = t
                t += rel - st_pos[top]
                ni += 1
                st_pos[top] = rel
            # go over
            fwd = ea[k] == L
            J_e[nj] = k; J_dir[nj] = 0 if fwd else 1; J_t[nj] = t
            nj += 1
            t += ekm[k]
            C = eb[k] if fwd else ea[k]
            pc = pb[k] if fwd else pa[k]
            top += 1
            st_loop[top] = C; st_entry[top] = pc; st_in[top] = k; st_pos[top] = 0.0
            n0 = cb
            for j in range(adj_start[C], adj_start[C + 1]):
                k2 = adj_edge[j]
                if k2 == k:
                    continue
                p = pa[k2] if ea[k2] == C else pb[k2]
                cbuf[cb] = k2
                cpos[cb] = (p - pc) % Lkm[C] if Lkm[C] > 0 else 0.0
                cb += 1
            if cb - n0 > 1:
                o = np.argsort(cpos[n0:cb])
                tmpk = cbuf[n0:cb][o].copy(); tmpp = cpos[n0:cb][o].copy()
                cbuf[n0:cb] = tmpk; cpos[n0:cb] = tmpp
            st_c0[top] = n0; st_c1[top] = cb; st_k[top] = n0
        else:
            # finish the loop, then come back over the join we arrived by
            if Lkm[L] > st_pos[top]:
                I_loop[ni] = L; I_s0[ni] = st_pos[top]; I_s1[ni] = Lkm[L]; I_t[ni] = t
                t += Lkm[L] - st_pos[top]
                ni += 1
            k = st_in[top]
            cb = st_c0[top]
            top -= 1
            if k >= 0:
                fwd = ea[k] == st_loop[top]
                J_e[nj] = k; J_dir[nj] = 1 if fwd else 0; J_t[nj] = t
                nj += 1
                t += ekm[k]
    return I_loop[:ni], I_s0[:ni], I_s1[:ni], I_t[:ni], J_e[:nj], J_dir[:nj], J_t[:nj], t


@nb.njit(cache=True)
def vertex_times(darts, dloop, dpos, visited, goff, frac, km, entry, Lkm, gstart, iv_key, iv_s0, iv_t, tf, tb):
    """time (km walked) at which the fish passes every vertex of every dart"""
    for i in range(len(darts)):
        d = darts[i]
        e = d >> 1
        a, b = goff[e], goff[e + 1]
        L = dloop[i]
        for j in range(a, b):
            fr = frac[j] if (d & 1) == 0 else 1.0 - frac[j]
            p = dpos[i] + fr * km[e]
            if not visited[L]:
                continue
            rel = (p - entry[L]) % Lkm[L] if Lkm[L] > 0 else 0.0
            if rel >= Lkm[L]:
                rel = 0.0
            key = gstart[L] + rel
            # interval with the largest start <= key
            lo, hi = 0, len(iv_key)
            while lo < hi:
                mid = (lo + hi) // 2
                if iv_key[mid] <= key:
                    lo = mid + 1
                else:
                    hi = mid
            q = lo - 1
            tt = iv_t[q] + (key - iv_key[q])
            if (d & 1) == 0:
                tf[j] = tt
            else:
                tb[j] = tt


def sphere_pairs(U):
    """neighbour pairs of points on the unit sphere: the edges of their convex hull (= spherical Delaunay),
    plus each point qhull set aside as coplanar (a near-duplicate) paired with its nearest hull vertex"""
    hull = ConvexHull(U, qhull_options="Qc")
    s = hull.simplices
    pairs = np.r_[s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]]
    if len(hull.coplanar):
        cp = hull.coplanar
        pairs = np.r_[pairs, cp[:, [0, 2]], np.c_[cp[:, 0], s[cp[:, 1], 0]], np.c_[cp[:, 0], s[cp[:, 1], 1]],
                      np.c_[cp[:, 0], s[cp[:, 1], 2]]]
    return np.unique(np.sort(pairs, axis=1), axis=0), len(hull.coplanar)


def connect_rest(nl, ja, jb, Ql, Qp, Q, chosen_pairs):
    """Borůvka on nearest neighbours: any loops the tree still misses are joined, by leap, to the nearest
    point outside their piece, until one piece remains"""
    par = np.arange(nl)
    for a, b in zip(*chosen_pairs):
        ra, rb = find(par, a), find(par, b)
        if ra != rb:
            par[ra] = rb
    tree = cKDTree(xyz(Q[:, 0], Q[:, 1]))
    U = xyz(Q[:, 0], Q[:, 1])
    extra = []
    for it in range(40):
        roots = np.empty(nl, np.int64)
        for x in range(nl):
            roots[x] = find(par, x)
        pc = roots[Ql]
        npieces = len(np.unique(roots))
        if npieces == 1:
            break
        main = np.bincount(pc).argmax()
        k = 8
        d, nb_ = tree.query(U, k=k, workers=-1)
        R, D, I, J = [], [], [], []
        for col in range(1, k):
            other = (pc[nb_[:, col]] != pc) & (pc != main)
            i = np.nonzero(other)[0]
            R.append(pc[i]); D.append(d[i, col]); I.append(i); J.append(nb_[i, col])
        R, D, I, J = (np.concatenate(x) for x in (R, D, I, J))
        if len(R) == 0:
            break
        o = np.lexsort((D, R))
        f = np.r_[True, R[o][1:] != R[o][:-1]]
        sel = o[f]
        nj = 0
        for i, j, dd in zip(I[sel], J[sel], D[sel]):
            ra, rb = find(par, Ql[i]), find(par, Ql[j])
            if ra != rb:
                par[ra] = rb
                nj += 1
                extra.append((Ql[i], Ql[j], Qp[i], Qp[j], dd * R_EARTH, Q[i, 0], Q[i, 1], Q[j, 0], Q[j, 1]))
        best = sel
        log(f"  fallback round {it}: {npieces:,} pieces, {len(best):,} joined")
    return extra


def main():
    z = np.load(os.path.join(DATA, "network.npz"))
    gxy, goff, eu, ev, kind, km, ring, active, loop, order = (z[k] for k in
                                                             ("gxy", "goff", "eu", "ev", "kind", "km", "ring", "active", "loop", "order"))
    m = len(kind)
    nl = int(loop.max()) + 1
    frac = vertex_fracs(gxy, goff)
    act = np.nonzero(active)[0]
    o = np.lexsort((order[act], loop[act]))
    darts = act[o]
    dloop = loop[darts]
    dkm = km[darts >> 1]
    lstart = np.searchsorted(dloop, np.arange(nl + 1))
    gcum = np.cumsum(dkm) - dkm                           # global start of each dart
    gstart = gcum[lstart[:-1]]                            # global start of each loop
    dpos = gcum - gstart[dloop]                           # dart start on its loop's axis
    Lkm = np.bincount(dloop, weights=dkm, minlength=nl)
    dart_at = np.full(2 * m, -1, np.int64)
    dart_at[darts] = np.arange(len(darts))
    log(f"{nl:,} loops, {len(darts):,} darts, {Lkm.sum():,.0f} km")

    # ---------------- joins 1: crossings (loops that share an edge)
    both = active[0::2] & active[1::2]
    e_sh = np.nonzero(both & (loop[0::2] != loop[1::2]))[0]
    ia, ib = dart_at[2 * e_sh], dart_at[2 * e_sh + 1]
    X_a, X_b = dloop[ia], dloop[ib]
    X_pa = dpos[ia] + 0.5 * km[e_sh]
    X_pb = dpos[ib] + 0.5 * km[e_sh]
    mid = gxy[goff[e_sh]]
    log(f"crossings: {len(e_sh):,} shared edges")
    # water components
    g = coo_matrix((np.ones(len(e_sh)), (X_a, X_b)), shape=(nl, nl))
    nwc, wc = connected_components(g, directed=False)
    # ---------------- which loops touch the open ocean: coast darts not on the Caspian (the land's one hole)
    cz = np.load(os.path.join(DATA, "coast_rings.npz"))
    hole = np.nonzero(cz["hole"])[0]
    caspian = np.zeros(len(cz["off"]) - 1, bool)
    caspian[hole] = True
    import shapely
    for h in hole:
        poly = shapely.Polygon(cz["xy"][cz["off"][h]:cz["off"][h + 1]])
        first = cz["xy"][cz["off"][:-1]]
        bb = poly.bounds
        cand = np.nonzero((first[:, 0] > bb[0]) & (first[:, 0] < bb[2]) & (first[:, 1] > bb[1]) & (first[:, 1] < bb[3]))[0]
        caspian[cand[shapely.contains_xy(poly, first[cand, 0], first[cand, 1])]] = True
    log(f"rings in a sea with no outlet (the Caspian and its islands): {caspian.sum()}")
    ek = kind[darts >> 1]
    er = ring[darts >> 1]
    ocean_dart = (ek == K_COAST) & ~caspian[np.maximum(er, 0)]
    wc_ocean = np.zeros(nwc, bool)
    wc_ocean[wc[dloop[ocean_dart]]] = True
    log(f"water bodies: {nwc:,}; reach the ocean: {wc_ocean.sum():,} "
        f"({np.bincount(wc, weights=Lkm, minlength=nwc)[wc_ocean].sum():,.0f} km); "
        f"cut off: {(~wc_ocean).sum():,} ({np.bincount(wc, weights=Lkm, minlength=nwc)[~wc_ocean].sum():,.0f} km)")

    # ---------------- joins 2: swims between ocean coasts (minimum spanning tree candidates from a spherical Delaunay)
    sd = np.nonzero(ocean_dart)[0]
    P, Pd, Pp = sample_darts(darts[sd], gxy, goff, frac, km, dpos[sd], SWIM_SAMPLE_KM)
    Pl = dloop[sd][Pd]
    U = xyz(P[:, 0], P[:, 1])
    _, ui = np.unique(np.round(U, 7), axis=0, return_index=True)
    U = U[ui]
    P, Pl, Pp = P[ui], Pl[ui], Pp[ui]
    pairs, ncop = sphere_pairs(U)
    log(f"swim sample: {len(P):,} coast points ({ncop:,} near-duplicates)")
    pairs = pairs[wc[Pl[pairs[:, 0]]] != wc[Pl[pairs[:, 1]]]]
    S_d = np.linalg.norm(U[pairs[:, 0]] - U[pairs[:, 1]], axis=1)
    S_d = 2 * R_EARTH * np.arcsin(np.minimum(S_d / 2, 1.0))
    log(f"swim candidates: {len(pairs):,}")
    S_a, S_b, S_pa, S_pb = Pl[pairs[:, 0]], Pl[pairs[:, 1]], Pp[pairs[:, 0]], Pp[pairs[:, 1]]
    S_xy = np.c_[P[pairs[:, 0]], P[pairs[:, 1]]]
    del U

    # ---------------- joins 3: leaps (any shore point; one dart per river edge is enough)
    first_of_edge = (darts & 1) == 0
    lone = ~active[darts ^ 1]
    ld = np.nonzero(first_of_edge | lone)[0]
    Q, Qd, Qp = sample_darts(darts[ld], gxy, goff, frac, km, dpos[ld], LEAP_SAMPLE_KM)
    Ql = dloop[ld][Qd]
    V = xyz(Q[:, 0], Q[:, 1])
    _, vi = np.unique(np.round(V, 7), axis=0, return_index=True)
    V = V[vi]
    Q, Ql, Qp = Q[vi], Ql[vi], Qp[vi]
    pairs, ncop = sphere_pairs(V)
    log(f"leap sample: {len(Q):,} points ({ncop:,} near-duplicates)")
    pairs = pairs[wc[Ql[pairs[:, 0]]] != wc[Ql[pairs[:, 1]]]]
    L_d = np.linalg.norm(V[pairs[:, 0]] - V[pairs[:, 1]], axis=1)
    L_d = 2 * R_EARTH * np.arcsin(np.minimum(L_d / 2, 1.0))
    L_a, L_b, L_pa, L_pb = Ql[pairs[:, 0]], Ql[pairs[:, 1]], Qp[pairs[:, 0]], Qp[pairs[:, 1]]
    L_xy = np.c_[Q[pairs[:, 0]], Q[pairs[:, 1]]]
    log(f"leap candidates: {len(pairs):,}")
    del V

    # ---------------- the tree
    J_a = np.r_[X_a, S_a, L_a]; J_b = np.r_[X_b, S_b, L_b]
    J_pa = np.r_[X_pa, S_pa, L_pa]; J_pb = np.r_[X_pb, S_pb, L_pb]
    J_type = np.r_[np.zeros(len(X_a)), np.ones(len(S_a)), np.full(len(L_a), 2)].astype(np.int8)
    J_km = np.r_[np.zeros(len(X_a)), S_d, L_d]
    J_xy = np.r_[np.c_[mid, mid], S_xy, L_xy]
    w = np.r_[np.zeros(len(X_a)), S_d, L_d + LEAP_PENALTY]
    chosen = kruskal(nl, J_a, J_b, np.argsort(w, kind="stable"))
    J_a, J_b, J_pa, J_pb, J_type, J_km, J_xy = (x[chosen] for x in (J_a, J_b, J_pa, J_pb, J_type, J_km, J_xy))
    if len(J_a) < nl - 1:
        log(f"tree has {len(J_a):,} of {nl - 1:,} joins; joining the rest by nearest neighbour")
        ex = connect_rest(nl, J_a, J_b, Ql, Qp, Q, (J_a, J_b))
        if ex:
            ex = np.array(ex)
            J_a = np.r_[J_a, ex[:, 0].astype(np.int64)]; J_b = np.r_[J_b, ex[:, 1].astype(np.int64)]
            J_pa = np.r_[J_pa, ex[:, 2]]; J_pb = np.r_[J_pb, ex[:, 3]]
            J_type = np.r_[J_type, np.full(len(ex), 2, np.int8)]; J_km = np.r_[J_km, ex[:, 4]]
            J_xy = np.r_[J_xy, ex[:, 5:9]]
    walk_km = np.where(J_type == 1, J_km, 0.0)            # leaps: no distance; crossings: zero
    log(f"tree: {len(J_a):,} joins (should be {nl - 1:,}): crossings {np.sum(J_type == 0):,}, swims {np.sum(J_type == 1):,} "
        f"({J_km[J_type == 1].sum():,.0f} km each way), leaps {np.sum(J_type == 2):,}")

    # ---------------- where to start: the point of a dart nearest START, on the dart heading closest to
    # START_HEADING (degrees clockwise from north); default: the Bow River in downtown Calgary, heading downstream
    hd = math.radians(float(os.environ.get("START_HEADING", "100")))
    vx = gxy[:, 0]
    near = np.nonzero((np.abs(vx - START[0]) < 0.05) & (np.abs(gxy[:, 1] - START[1]) < 0.05))[0]
    eid = np.searchsorted(goff, near, side="right") - 1
    best = None
    for v, e in zip(near, eid):
        for d in (2 * e, 2 * e + 1):
            if not active[d]:
                continue
            j = v + 1 if (d & 1) == 0 else v - 1
            if j < goff[e] or j >= goff[e + 1]:
                continue
            dx = (gxy[j, 0] - gxy[v, 0]) * math.cos(math.radians(gxy[v, 1]))
            dy = gxy[j, 1] - gxy[v, 1]
            dist = math.hypot((gxy[v, 0] - START[0]) * math.cos(math.radians(START[1])), gxy[v, 1] - START[1])
            cosang = (dx * math.sin(hd) + dy * math.cos(hd)) / max(math.hypot(dx, dy), 1e-12)
            score = dist * 111.2 + 5 * (1 - cosang)
            if best is None or score < best[0]:
                fr = frac[v] if (d & 1) == 0 else 1 - frac[v]
                best = (score, d, fr, v)
    _, d0, fr0, v0 = best
    i0 = int(dart_at[d0])
    root, entry = int(dloop[i0]), float(dpos[i0] + fr0 * km[d0 >> 1])
    log(f"start: loop {root} ({Lkm[root]:,.0f} km) at {gxy[v0]}, dart {d0}")
    # adjacency
    ends = np.r_[J_a, J_b]
    eidx = np.r_[np.arange(len(J_a)), np.arange(len(J_a))]
    oo = np.argsort(ends, kind="stable")
    adj_edge = eidx[oo]
    adj_start = np.searchsorted(ends[oo], np.arange(nl + 1))
    Il, Is0, Is1, It, Je, Jdir, Jt, total = euler_tour(nl, Lkm, adj_start, adj_edge, J_a, J_b, J_pa, J_pb, walk_km,
                                                      root, entry)
    log(f"tour: {len(Il):,} loop stretches, {len(Je):,} joins walked, {total:,.0f} km = {total / C_KMS:.2f} s at c")
    assert abs(np.sum(Is1 - Is0) - Lkm.sum()) < 1e-3 * Lkm.sum(), "every loop walked once"

    # ---------------- the time the fish passes every vertex, each way
    entry_all = np.zeros(nl)
    entry_all[root] = entry
    # each loop's entry: the join it was first reached by
    seen = np.zeros(len(J_a), bool)
    for k, dr in zip(Je, Jdir):
        if not seen[k]:
            seen[k] = True
            C = J_b[k] if dr == 0 else J_a[k]
            entry_all[C] = J_pb[k] if dr == 0 else J_pa[k]
    key = gstart[Il] + Is0
    oi = np.argsort(key)
    tf = np.full(len(gxy), np.inf)
    tb = np.full(len(gxy), np.inf)
    visited = np.zeros(nl, np.bool_)
    visited[Il] = True
    log(f"loops not reached by the tour: {(~visited).sum()} ({Lkm[~visited].sum():.1f} km)")
    vertex_times(darts, dloop, dpos, visited, goff, frac, km, entry_all, Lkm, gstart, key[oi], Is0[oi], It[oi], tf, tb)
    stats = {"loops": int(nl), "water_bodies": int(nwc), "water_bodies_reaching_ocean": int(wc_ocean.sum()),
             "shore_km": float(Lkm.sum()), "swims": int(np.sum(J_type == 1)), "swim_km_each_way": float(J_km[J_type == 1].sum()),
             "leaps": int(np.sum(J_type == 2)), "crossings": int(np.sum(J_type == 0)), "total_km": float(total),
             "seconds_at_c": float(total / C_KMS), "start": [float(v) for v in gxy[v0]]}
    json.dump(stats, open(os.path.join(DATA, "tour_stats.json"), "w"), indent=1)
    np.savez(os.path.join(DATA, "tour.npz"), tf=tf, tb=tb, J_type=J_type, J_km=J_km, J_xy=J_xy, Je=Je, Jdir=Jdir, Jt=Jt,
             total=total)
    log(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
