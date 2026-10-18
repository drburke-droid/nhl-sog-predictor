"""River systems from OpenStreetMap's river lines: for every mouth, the longest path upstream (the main
stem, as hydrologists measure a river's length), and the 1,000 longest systems.

1. Junctions: every vertex shared by two river lines (an OSM node), and every line end. Lines are cut at
   junctions into edges, directed downstream (OSM draws waterways in the direction of flow).
2. Lakes: a river ending inside or at a lake (> 5 km2) is joined across the lake to each river leaving it.
   Lakes with no outflow (Caspian, Aral, Chad, ...) end their systems.
3. Gaps: a river that stops within SNAP_M of another river's line (SNAP_WIDE_M inland) is joined to it.
4. Cycles (braids, lines drawn against the flow) are broken, then the longest upstream path to every node
   is found in one topological pass, and each outlet's main stem traced.
5. Distributaries: outlets whose main stems mostly overlap are one system (the longest is kept).

    python systems.py [--relink]  -> data/systems.json (top 1000; main-stem geometry in data/stems.npz)
"""
import os, re, sys, glob, json, time, math
import numpy as np
import numba as nb
import shapely
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R_EARTH = 6371.0088
SNAP_M = 300.0
SNAP_WIDE_M = 10000.0
BRIDGE_KM = 150.0
STUB_KM = 20.0
SHORE_DEG = 0.03                                              # ~3 km
SNAP_LAKE_M = 25000.0                                         # from the shore of a reservoir to the river drawn through it
TOP = 1200                                                    # traced; the first 1,000 that pass QA are used
# lakes with no outlet: anything that seems to leave them is an inflow drawn the wrong way
ENDORHEIC = re.compile(r"(Каспий|Caspian|Hazar|خزر|Арал|\bAral\b|Балқаш|Балхаш|Balkhash|Tchad|Lake Chad|Eyre|Lake Torrens|"
                       r"Gairdner|Turkana|Ысык|Issyk|Urmia|ارومیه|Van Gölü|青海湖|Qinghai|Tuz Gölü|Dead Sea|البحر الميت|"
                       r"Salton|Great Salt Lake|Mar Chiquita|Etosha|Sarygamy|Sarıqamıs|دریاچه نمک|Увс|Uvs|Rukwa|Natron|Pyramid Lake)", re.I)
CANAL = re.compile(r"(\bcanal\b|\bkanal\b|канал|运河|水渠|\bnahar\b|\bcanale\b|\bkanaal\b)", re.I)
NE_LAND = os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_land.geojson")


def load():
    xs, offs, nm, en, wd = [], [np.zeros(1, np.int64)], [], [], []
    lx, loffs, ln, la = [], [np.zeros(1, np.int64)], [], []
    vb = lb = 0
    for p in sorted(glob.glob(os.path.join(DATA, "full", "part-*.npz"))):
        z = np.load(p, allow_pickle=True)
        xs.append(z["xy"]); offs.append(z["off"][1:] + vb); vb += len(z["xy"])
        nm.append(z["name"]); en.append(z["name_en"]); wd.append(z["wikidata"])
        lx.append(z["lake_xy"]); loffs.append(z["lake_off"][1:] + lb); lb += len(z["lake_xy"])
        ln.append(z["lake_name"]); la.append(z["lake_km2"])
    return (np.concatenate(xs), np.concatenate(offs), np.concatenate(nm), np.concatenate(en), np.concatenate(wd),
            np.concatenate(lx), np.concatenate(loffs), np.concatenate(ln), np.concatenate(la))


@nb.njit(cache=True)
def seg_km(xy, off):
    """haversine length of each vertex's segment to the next vertex of the same line (0 at line ends)"""
    n = xy.shape[0]
    out = np.zeros(n)
    for p in range(len(off) - 1):
        for i in range(off[p], off[p + 1] - 1):
            lo1, la1 = xy[i, 0] * 1e-7, xy[i, 1] * 1e-7
            lo2, la2 = xy[i + 1, 0] * 1e-7, xy[i + 1, 1] * 1e-7
            p1, p2 = math.radians(la1), math.radians(la2)
            a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
            out[i] = 2 * 6371.0088 * math.asin(math.sqrt(min(a, 1.0)))
    return out


@nb.njit(cache=True)
def out_csr(n, eu):
    cnt = np.zeros(n + 1, np.int64)
    for k in range(len(eu)):
        cnt[eu[k] + 1] += 1
    for i in range(n):
        cnt[i + 1] += cnt[i]
    pos = cnt[:-1].copy()
    oe = np.empty(len(eu), np.int64)
    for k in range(len(eu)):
        oe[pos[eu[k]]] = k
        pos[eu[k]] += 1
    return cnt, oe


@nb.njit(cache=True)
def drop_back_edges(n, eu, ev):
    """Braided channels and lines drawn against the flow make cycles. A depth-first search downstream
    from the sources marks every edge that returns to a node still on the search path; dropping those
    leaves a DAG. Only edges on cycles can be marked."""
    cnt, oe = out_csr(n, eu)
    indeg = np.zeros(n, np.int64)
    for k in range(len(eu)):
        indeg[ev[k]] += 1
    color = np.zeros(n, np.int8)
    keep = np.ones(len(eu), np.bool_)
    sv = np.empty(n, np.int64)
    sp_ = np.empty(n, np.int64)
    for pas in range(2):
        for r in range(n):
            if color[r] != 0 or (pas == 0 and indeg[r] != 0):
                continue
            top = 0
            sv[0] = r; sp_[0] = cnt[r]; color[r] = 1
            while top >= 0:
                u = sv[top]
                p = sp_[top]
                if p < cnt[u + 1]:
                    sp_[top] = p + 1
                    k = oe[p]
                    v = ev[k]
                    if color[v] == 1:
                        keep[k] = False
                    elif color[v] == 0:
                        color[v] = 1
                        top += 1
                        sv[top] = v; sp_[top] = cnt[v]
                else:
                    color[u] = 2
                    top -= 1
    return keep


@nb.njit(cache=True)
def longest_upstream(n, eu, ev, ew):
    """Kahn's topological order on the (acyclic) downstream graph; best[v] = longest path from any
    source to v, via[v] = the edge it arrives by (-1 at sources)."""
    cnt, oe = out_csr(n, eu)
    indeg = np.zeros(n, np.int64)
    for k in range(len(eu)):
        indeg[ev[k]] += 1
    best = np.zeros(n)
    via = np.full(n, -1, np.int64)
    q = np.empty(n, np.int64)
    h = t = 0
    for i in range(n):
        if indeg[i] == 0:
            q[t] = i; t += 1
    while h < t:
        u = q[h]; h += 1
        for j in range(cnt[u], cnt[u + 1]):
            k = oe[j]
            v = ev[k]
            if best[u] + ew[k] > best[v]:
                best[v] = best[u] + ew[k]
                via[v] = k
            indeg[v] -= 1
            if indeg[v] == 0:
                q[t] = v; t += 1
    return best, via, t, q


@nb.njit(cache=True)
def choose_snaps(sinks, cstart, cend, ct, rank, icnt, ie, eu, limit):
    """for each dangling end, the nearest candidate (in order) that is not upstream of it: joining to
    an upstream point would make a loop. A target ranked after the sink in topological order cannot
    be upstream; otherwise search upstream from the sink (up to `limit` nodes; if the search is cut
    off, the candidate is refused)."""
    pick = np.full(len(sinks), -1, np.int64)
    stamp = np.zeros(len(rank), np.int32)
    queue = np.empty(limit, np.int64)
    for i in range(len(sinks)):
        s = sinks[i]
        searched = False
        complete = False
        for c in range(cstart[i], cend[i]):
            t = ct[c]
            if t < 0 or t == s:
                continue
            if rank[t] > rank[s]:
                pick[i] = c
                break
            if not searched:
                searched = True
                stamp[s] = i + 1
                queue[0] = s
                h, tl = 0, 1
                complete = True
                while h < tl:
                    u = queue[h]; h += 1
                    for j in range(icnt[u], icnt[u + 1]):
                        w = eu[ie[j]]
                        if stamp[w] != i + 1:
                            if tl >= limit:
                                complete = False
                                break
                            stamp[w] = i + 1
                            queue[tl] = w; tl += 1
                    if not complete:
                        break
            if complete and stamp[t] != i + 1:
                pick[i] = c
                break
    return pick


def gc_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def coast_dist():
    """distance (km) from any lon/lat to the Natural Earth 1:10m coastline, from its vertices
    densified to ~500 m"""
    land = shapely.union_all([shapely.geometry.shape(f["geometry"]) for f in json.load(open(NE_LAND))["features"]])
    c = shapely.get_coordinates(shapely.segmentize(shapely.boundary(land), 0.005))
    tree = cKDTree(xyz(c[:, 0], c[:, 1]))
    shapely.prepare(land)

    def f(lon, lat):
        d, _ = tree.query(xyz(np.atleast_1d(lon), np.atleast_1d(lat)), workers=-1)
        return d * R_EARTH
    f.on_land = lambda lon, lat: shapely.contains_xy(land, lon, lat)
    return f


def links(xy, off, piece, cum, node_of_j, jpos, nlon, nlat, eu, ev, ew, epiece, nn, keyof, wd, lxy, loff, lname, lkm2, coast_km, t0):
    """the extra downstream links: across lakes, and over gaps in the mapping"""
    nv, npc = len(xy), len(off) - 1
    outdeg = np.bincount(eu, minlength=nn)
    indeg = np.bincount(ev, minlength=nn)
    sinks = np.nonzero(outdeg == 0)[0]
    sources = np.nonzero(indeg == 0)[0]
    sink_piece = np.full(nn, -1, np.int64)
    sink_piece[ev] = epiece                                       # a line ending at the node
    src_piece = np.full(nn, -1, np.int64)
    src_piece[eu[::-1]] = epiece[::-1]                            # a line starting at the node
    # ---- lakes. A lake mapped as several polygons (split at a bridge or dam) is one lake.
    parts = []
    for k in range(len(lname)):
        ring = lxy[loff[k]:loff[k + 1]].astype(np.float64)
        parts.append(shapely.Polygon(ring) if len(ring) >= 4 else shapely.Point(0, 0).buffer(1e-9))
    parts = shapely.make_valid(np.array(parts, dtype=object))
    new = np.r_[True, (lname[1:] != lname[:-1]) | (lkm2[1:] != lkm2[:-1])]
    ng = int(new.sum())
    starts = np.nonzero(new)[0]
    ends = np.r_[starts[1:], len(parts)]
    polys = np.array([parts[a_] if b_ - a_ == 1 else shapely.union_all(parts[a_:b_]) for a_, b_ in zip(starts, ends)], dtype=object)
    gkm2 = lkm2[new]
    tree = shapely.STRtree(shapely.buffer(polys, 0.002))
    tree_in = shapely.STRtree(polys)
    # river drawn through a lake (a reservoir with its centreline mapped): tributaries join that line
    mid = (jpos[:-1] + jpos[1:]) // 2
    same = piece[jpos[:-1]] == piece[jpos[1:]]
    mid = mid[same]
    mi, ml = tree_in.query(shapely.points(xy[mid, 0] * 1e-7, xy[mid, 1] * 1e-7), predicate="intersects")
    inside_km = np.bincount(ml, weights=ew[mi], minlength=ng)
    env = shapely.get_coordinates(shapely.oriented_envelope(polys)).reshape(ng, -1, 2)[:, :4]
    side = np.hypot(np.diff(env[:, :3, 0], axis=1) * np.cos(np.radians(env[:, :2, 1])), np.diff(env[:, :3, 1], axis=1))
    long_km = side.max(1) * 111.2                                 # the lake's length
    through = inside_km > 0.6 * long_km
    si, sl = tree.query(shapely.points(nlon[sinks], nlat[sinks]), predicate="intersects")
    oi, ol = tree.query(shapely.points(nlon[sources], nlat[sources]), predicate="intersects")
    # an inflow comes from outside the lake; an outflow leaves it (lines lying inside the lake are neither)
    def outside(vidx, lakes):
        return ~shapely.contains_xy(polys[lakes], xy[vidx, 0] * 1e-7, xy[vidx, 1] * 1e-7)
    far_s = off[sink_piece[sinks[si]]]                            # where an inflow's line starts
    far_o = off[src_piece[sources[oi]] + 1] - 1                   # where an outflow's line ends
    # A line that starts inside the lake but is fed from upstream is an inflow too, and one that ends inside
    # it but flows on is an outflow; so is a short first line, such as the stubs below a dam.
    plen = cum[off[1:] - 1] - cum[off[:-1]]
    fed = indeg[node_of_j[far_s]] > 0
    flows_on = outdeg[node_of_j[far_o]] > 0
    ok_s = outside(far_s, sl) | fed | (plen[sink_piece[sinks[si]]] < 2.0)
    ok_o = outside(far_o, ol) | flows_on | (plen[src_piece[sources[oi]]] < 2.0)
    # the lake outlines are outer shores only, so islands count as lake: a river on an island (Isle Royale
    # in Lake Superior) is neither. Inflows and outflows meet the lake within SHORE_DEG of its outer shore.
    shore = shapely.boundary(polys)
    ok_s &= shapely.distance(shore[sl], shapely.points(nlon[sinks[si]], nlat[sinks[si]])) < SHORE_DEG
    ok_o &= shapely.distance(shore[ol], shapely.points(nlon[sources[oi]], nlat[sources[oi]])) < SHORE_DEG
    # an outflow runs downhill: a line drawn the wrong way (an inflow mapped from its mouth to its source)
    # climbs away from the lake. Its elevation 15 km downstream must not be 30 m above the lake's.
    from features import Terrain
    terrain = Terrain()
    cnt_o, oe_o = out_csr(nn, eu)
    cand = sources[oi[ok_o]]
    ahead = cand.copy()
    reach = np.zeros(len(cand))
    for n_, o in enumerate(cand):
        v, d = o, 0.0
        while d < 300.0 and cnt_o[v + 1] > cnt_o[v]:
            k = oe_o[cnt_o[v]]
            d += ew[k]
            v = ev[k]
            if d < 15.0:
                ahead[n_] = v
        reach[n_] = d
    climb = terrain(nlon[ahead], nlat[ahead]).astype(float) - terrain(nlon[cand], nlat[cand])
    idx_o = np.nonzero(ok_o)[0]
    ok_o[idx_o[climb > 30]] = False
    print(f"  {np.sum(climb > 30):,} lake 'outflows' climb away from their lake: drawn the wrong way, left out", flush=True)
    # the longest path takes whichever outflow is farthest away, so a stub at a river mouth on the far shore
    # would capture a whole lake. An outflow that stops within STUB_KM, where the lake has one that runs at
    # least twice as far, is left out. (Mapping gaps stop the walk too, so longer ones are kept.)
    best_reach = {}
    for n_, l in enumerate(ol[idx_o]):
        best_reach[l] = max(best_reach.get(l, 0.0), reach[n_] if ok_o[idx_o[n_]] else 0.0)
    small = np.array([reach[n_] < min(0.5 * best_reach[l], STUB_KM) for n_, l in enumerate(ol[idx_o])], bool)
    ok_o[idx_o[small]] = False
    print(f"  {small.sum():,} lake 'outflows' end within half the distance of their lake's main outflow, left out", flush=True)
    endo = np.array([bool(ENDORHEIC.search(n or "")) for n in lname[starts]])
    ok_o &= ~endo[ol]
    lake_in, lake_out = {}, {}
    for i, l in zip(si[ok_s], sl[ok_s]):
        lake_in.setdefault(int(l), []).append(int(sinks[i]))
    for i, l in zip(oi[ok_o], ol[ok_o]):
        lake_out.setdefault(int(l), []).append(int(sources[i]))
    # an inflow crosses to the lake's outflow: the river that also flows in, if one does (a reservoir), else
    # every outflow (the longest path picks among them). In a lake with a river drawn through it, the
    # crossing is only a candidate: tried first for the river's own outflow, otherwise in order of
    # distance with the gap links.
    xu, xv, xw, xkind = [], [], [], []
    LX, LO, LW, LP = [], [], [], []
    terminal = np.zeros(nn, bool)
    for l, ins in lake_in.items():
        outs = lake_out.get(l, [])
        if not outs:
            continue
        kin = {keyof(sink_piece[x]) for x in ins} - {""}
        main = [o for o in outs if keyof(src_piece[o]) in kin]
        for x in ins:
            for o in (main or outs):
                if x == o:
                    continue
                w_ = float(gc_km(nlon[x], nlat[x], nlon[o], nlat[o]))
                if through[l]:
                    LX.append(x); LO.append(o); LW.append(w_); LP.append(0 if keyof(sink_piece[x]) == keyof(src_piece[o]) else 1)
                else:
                    xu.append(x); xv.append(o); xw.append(w_); xkind.append(1)
                    terminal[x] = True
    LX, LO, LW, LP = np.array(LX, np.int64), np.array(LO, np.int64), np.array(LW), np.array(LP, np.int8)
    in_through = np.zeros(nn, bool)
    in_through[sinks[si][through[sl]]] = True
    print(f"lakes: {ng:,} ({len(lname):,} polygons); {len(set(lake_in) & set(lake_out)):,} with outflows "
          f"({len(xu):,} crossings, {len(LX):,} more possible), {through.sum():,} with a river drawn through "
          f"({time.time() - t0:.0f}s)", flush=True)
    # ---- the topological rank of every node, for the no-loop test
    EU0 = np.r_[eu, np.array(xu, np.int64)]
    EV0 = np.r_[ev, np.array(xv, np.int64)]
    keep = drop_back_edges(nn, EU0, EV0)
    _, _, _, order = longest_upstream(nn, EU0[keep], EV0[keep], np.r_[ew, np.array(xw)][keep])
    rank = np.empty(nn, np.int64)
    rank[order] = np.arange(nn)
    # ---- gaps: a river that ends inland, not in a lake, joins the nearest other river that is not
    # upstream of it: within SNAP_M near the coast, SNAP_WIDE_M inland (tributaries often stop at the bank
    # of a wide river whose centreline runs mid-channel, or in a dead-end side channel)
    dang = sinks[~terminal[sinks]]
    radius = np.where(coast_km(nlon[dang], nlat[dang]) < 3.0, SNAP_M, SNAP_WIDE_M)
    radius = np.where(in_through[dang], SNAP_LAKE_M, radius) / 1000 / R_EARTH
    dpiece = sink_piece[dang]
    tree_s = cKDTree(xyz(nlon[dang], nlat[dang]))
    KS, KQ, KD = [], [], []
    for c0 in range(0, nv, 5_000_000):
        c1 = min(nv, c0 + 5_000_000)
        dl, kl = tree_s.query(xyz(xy[c0:c1, 0] * 1e-7, xy[c0:c1, 1] * 1e-7), k=4,
                              distance_upper_bound=SNAP_LAKE_M / 1000 / R_EARTH, workers=-1)
        S, Q, D = [], [], []
        for col in range(4):
            hit = np.nonzero(np.isfinite(dl[:, col]))[0]
            if len(hit) == 0:
                continue
            sv, d = kl[hit, col], dl[hit, col]
            q = c0 + hit
            ok = (piece[q] != dpiece[sv]) & (d <= radius[sv])
            S.append(sv[ok]); Q.append(q[ok]); D.append(d[ok])
        if S:
            S, Q, D = np.concatenate(S), np.concatenate(Q), np.concatenate(D)
            key = S.astype(np.int64) * npc + piece[Q]
            o = np.lexsort((D, key))
            first = np.r_[True, key[o][1:] != key[o][:-1]]
            KS.append(S[o][first]); KQ.append(Q[o][first]); KD.append(D[o][first])
        if c0 % 50_000_000 == 0:
            print(f"  gap search {c1:,} of {nv:,} vertices ({time.time() - t0:.0f}s)", flush=True)
    S, Q, D = np.concatenate(KS), np.concatenate(KQ), np.concatenate(KD)
    key = S.astype(np.int64) * npc + piece[Q]
    o = np.lexsort((D, key))
    first = np.r_[True, key[o][1:] != key[o][:-1]]
    S, Q, D = S[o][first], Q[o][first], D[o][first]
    o = np.lexsort((D, S))                                        # each sink's candidate lines, nearest first
    S, Q, D = S[o], Q[o], D[o]
    gstart = np.r_[0, np.nonzero(S[1:] != S[:-1])[0] + 1]
    rk = np.arange(len(S)) - np.repeat(gstart, np.diff(np.r_[gstart, len(S)]))
    m = rk < 12
    S, Q, D = S[m], Q[m], D[m]
    j = np.searchsorted(jpos, Q, side="left")                     # the next junction at or after q on its line
    j = np.minimum(j, len(jpos) - 1)
    T = np.where(piece[jpos[j]] == piece[Q], node_of_j[jpos[j]], -1).astype(np.int64)
    W = D * R_EARTH + cum[jpos[j]] - cum[Q]
    # with the lake crossings, in the order they are tried
    LS = np.searchsorted(dang, LX)
    LS = np.where(dang[np.minimum(LS, len(dang) - 1)] == LX, LS, -1)
    LO, LW, LP, LS = LO[LS >= 0], LW[LS >= 0], LP[LS >= 0], LS[LS >= 0]
    S = np.r_[S, LS]
    T = np.r_[T, LO]
    W = np.r_[W, LW]
    K0 = np.r_[np.full(len(Q), 2, np.int8), np.ones(len(LS), np.int8)]
    dkm = np.r_[D * R_EARTH, LW]
    # a link longer than 1 km must not point back the way the river came (more than ~107 degrees from its
    # direction of flow over its last 20 vertices)
    i1 = off[dpiece + 1] - 1
    i0 = np.maximum(off[dpiece], i1 - 20)
    coslat = np.cos(np.radians(nlat[dang]))
    fx = (xy[i1, 0] - xy[i0, 0]) * 1e-7 * coslat
    fy = (xy[i1, 1] - xy[i0, 1]) * 1e-7
    tx = np.r_[xy[Q, 0] * 1e-7, nlon[LO]]
    ty = np.r_[xy[Q, 1] * 1e-7, nlat[LO]]
    gx = (tx - nlon[dang[S]]) * coslat[S]
    gy = ty - nlat[dang[S]]
    cosang = (fx[S] * gx + fy[S] * gy) / np.maximum(np.hypot(fx[S], fy[S]) * np.hypot(gx, gy), 1e-12)
    ok = (dkm <= 1.0) | (cosang > -0.3)
    pri = np.r_[np.ones(len(Q), np.int8), LP]                    # the same river's own outflow first
    S, T, W, K0, dkm, pri = S[ok], T[ok], W[ok], K0[ok], dkm[ok], pri[ok]
    o = np.lexsort((dkm, pri, S))
    S, T, W, K0 = S[o], T[o], W[o], K0[o]
    cptr = np.searchsorted(S, np.arange(len(dang) + 1))
    icnt, ie = out_csr(nn, EV0)                                   # in-edges, for the upstream search
    pick = choose_snaps(dang, cptr[:-1], cptr[1:], T, rank, icnt, ie, EU0, 400_000)
    print(f"ends: {(pick >= 0).sum():,} of {len(dang):,} joined across a lake or a gap ({time.time() - t0:.0f}s)", flush=True)
    # ---- a river with a hole in its mapping: an inland end joins the nearest start of a line of the same
    # river (same Wikidata item) within BRIDGE_KM, ahead of it (within 90 degrees of its direction of flow)
    # and not upstream of it
    rest = dang[(pick < 0) & (radius > SNAP_M / 1000 / R_EARTH * 1.01)]
    rwd = np.array([wd[sink_piece[x]] for x in rest], dtype=object)
    rest = rest[rwd != ""]
    rwd = rwd[rwd != ""]
    starts_p = np.nonzero(wd != "")[0]
    st_node = node_of_j[off[starts_p]].astype(np.int64)
    tree_b = cKDTree(xyz(nlon[st_node], nlat[st_node]))
    dist, nb_ = tree_b.query(xyz(nlon[rest], nlat[rest]), k=64, distance_upper_bound=BRIDGE_KM / R_EARTH, workers=-1)
    cs, ct, cw = [], [], []
    for i in range(len(rest)):
        x = rest[i]
        p = sink_piece[x]
        i1 = off[p + 1] - 1
        i0 = max(off[p], i1 - 20)
        fx = (xy[i1, 0] - xy[i0, 0]) * 1e-7 * np.cos(np.radians(nlat[x]))
        fy = (xy[i1, 1] - xy[i0, 1]) * 1e-7
        for d, j_ in zip(dist[i], nb_[i]):
            if not np.isfinite(d):
                break
            if wd[starts_p[j_]] != rwd[i]:
                continue
            tnode = st_node[j_]
            gx = (nlon[tnode] - nlon[x]) * np.cos(np.radians(nlat[x]))
            gy = nlat[tnode] - nlat[x]
            if fx * gx + fy * gy <= 0:
                continue
            cs.append(i); ct.append(tnode); cw.append(d * R_EARTH)
    cs, ct, cw = np.array(cs, np.int64), np.array(ct, np.int64), np.array(cw)
    bptr = np.searchsorted(cs, np.arange(len(rest) + 1))
    pick2 = choose_snaps(rest, bptr[:-1], bptr[1:], ct, rank, icnt, ie, EU0, 400_000)
    print(f"bridges: {(pick2 >= 0).sum():,} of {len(rest):,} remaining inland ends joined to their own river downstream "
          f"({time.time() - t0:.0f}s)", flush=True)
    # ---- links chosen one at a time can still close loops between them (two ends each joined to the
    # other's line). In each loop keep the shortest link, refuse the rest and choose again, until none.
    ends = np.r_[dang, rest]
    cstart = np.r_[cptr[:-1], len(T) + bptr[:-1]]
    cend = np.r_[cptr[1:], len(T) + bptr[1:]]
    CT, CW = np.r_[T, ct], np.r_[W, cw]
    CK = np.r_[K0, np.full(len(ct), 3, np.int8)]
    pick = np.r_[pick, np.where(pick2 >= 0, pick2 + len(T), -1)]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    def strong(U, V):
        return connected_components(coo_matrix((np.ones(len(U), np.int8), (U, V)), shape=(nn, nn)).tocsr(),
                                    directed=True, connection="strong")[1]
    lab0 = strong(EU0, EV0)
    for it in range(20):
        sel = np.nonzero(pick >= 0)[0]
        lu, lv = ends[sel], CT[pick[sel]]
        lab = strong(np.r_[EU0, lu], np.r_[EV0, lv])
        cyc = (lab[lu] == lab[lv]) & (lab0[lu] != lab0[lv])
        if not cyc.any():
            break
        idx, comp, w = sel[cyc], lab[lu[cyc]], CW[pick[sel[cyc]]]
        o = np.lexsort((w, comp))
        idx, comp = idx[o], comp[o]
        first = np.r_[True, comp[1:] != comp[:-1]]
        single = np.r_[comp[1:] != comp[:-1], True] & first
        drop = idx[~first | single]
        CT[pick[drop]] = -1
        pick[drop] = -1
        sel = np.nonzero(pick >= 0)[0]
        U2, V2 = np.r_[EU0, ends[sel]], np.r_[EV0, CT[pick[sel]]]
        keep = drop_back_edges(nn, U2, V2)
        _, _, _, order = longest_upstream(nn, U2[keep], V2[keep], np.zeros(keep.sum()))
        rank[order] = np.arange(nn)
        icnt2, ie2 = out_csr(nn, V2)
        pick[drop] = choose_snaps(ends[drop], cstart[drop], cend[drop], CT, rank, icnt2, ie2, U2, 400_000)
        print(f"  loops: {cyc.sum():,} links on loops, {len(drop):,} chosen again ({time.time() - t0:.0f}s)", flush=True)
    for i in np.nonzero(pick >= 0)[0]:
        c = pick[i]
        xu.append(int(ends[i])); xv.append(int(CT[c])); xw.append(float(CW[c])); xkind.append(int(CK[c]))
    print(f"links: {(CK[pick[pick >= 0]] == 1).sum():,} more lake crossings, {(CK[pick[pick >= 0]] == 2).sum():,} gap links, "
          f"{(CK[pick[pick >= 0]] == 3).sum():,} bridges "
          f"({time.time() - t0:.0f}s)", flush=True)
    return xu, xv, xw, xkind


def main():
    t0 = time.time()
    xy, off, name, name_en, wd, lxy, loff, lname, lkm2 = load()
    # irrigation canals mapped as rivers (Kachhi Canal, Rohri Canal...) are not rivers. A compound name
    # ("L'Yonne - Canal du Nivernais") is a river with a canalised stretch, and stays.
    canal = np.array([bool(CANAL.search(a or "")) or bool(CANAL.search(b or "")) for a, b in zip(name, name_en)])
    canal &= np.array([not any(c in (a or "") + (b or "") for c in (" - ", "(", "/")) for a, b in zip(name, name_en)])
    if canal.any():
        lens = np.diff(off)
        xy = xy[np.repeat(~canal, lens)]
        off = np.r_[0, np.cumsum(lens[~canal])]
        name, name_en, wd = name[~canal], name_en[~canal], wd[~canal]
    nv, npc = len(xy), len(off) - 1
    print(f"{npc:,} river lines ({canal.sum():,} canals left out), {nv:,} vertices; {len(lname):,} lakes "
          f"({time.time() - t0:.0f}s)", flush=True)
    piece = np.repeat(np.arange(npc), np.diff(off)).astype(np.int32)
    seg = seg_km(xy, off)
    total_km = seg.sum()
    cum = np.cumsum(seg)
    cum -= seg                                                  # along-line distance at each vertex (global running sum)
    del seg
    print(f"{total_km:,.0f} km of river ({time.time() - t0:.0f}s)", flush=True)
    # junctions: line ends, and any vertex that coincides with a line end (where a tributary or side
    # channel meets a line part-way along). Two lines crossing at shared interior nodes are not joined.
    def keys(lo, hi):
        return (xy[lo:hi, 0].astype(np.int64) << 32) | (xy[lo:hi, 1].astype(np.int64) & 0xFFFFFFFF)
    end_idx = np.r_[off[:-1], off[1:] - 1]
    ek = (xy[end_idx, 0].astype(np.int64) << 32) | (xy[end_idx, 1].astype(np.int64) & 0xFFFFFFFF)
    E_u = np.unique(ek)
    del ek, end_idx
    node_of_j = np.full(nv, -1, np.int32)
    for c0 in range(0, nv, 20_000_000):
        c1 = min(nv, c0 + 20_000_000)
        k = keys(c0, c1)
        pos = np.minimum(np.searchsorted(E_u, k), len(E_u) - 1)
        hit = E_u[pos] == k
        node_of_j[c0:c1][hit] = pos[hit]
    jpos = np.nonzero(node_of_j >= 0)[0]
    print(f"{len(jpos):,} junction vertices ({time.time() - t0:.0f}s)", flush=True)
    # edges: consecutive junction vertices on the same line
    same = piece[jpos[:-1]] == piece[jpos[1:]]
    a, b = jpos[:-1][same], jpos[1:][same]
    eu, ev = node_of_j[a].astype(np.int64), node_of_j[b].astype(np.int64)
    ew = cum[b] - cum[a]
    epiece = piece[a]
    nn = len(E_u)
    nlon = (E_u >> 32).astype(np.int32) * 1e-7
    nlat = (E_u & 0xFFFFFFFF).astype(np.uint32).astype(np.int32) * 1e-7
    print(f"graph: {nn:,} nodes, {len(eu):,} edges ({time.time() - t0:.0f}s)", flush=True)
    coast_km = coast_dist()
    keyof = lambda p: (wd[p] or name_en[p] or name[p]) if p >= 0 else ""
    cache = os.path.join(DATA, "links.npz")
    if os.path.exists(cache) and "--relink" not in sys.argv:
        z = np.load(cache)
        xu, xv, xw, xkind = z["xu"].tolist(), z["xv"].tolist(), z["xw"].tolist(), z["xkind"].tolist()
        print(f"links: {len(xu):,} lake and gap links from {cache}", flush=True)
    else:
        xu, xv, xw, xkind = links(xy, off, piece, cum, node_of_j, jpos, nlon, nlat, eu, ev, ew, epiece, nn, keyof, wd,
                                  lxy, loff, lname, lkm2, coast_km, t0)
        np.savez(cache, xu=np.array(xu, np.int64), xv=np.array(xv, np.int64), xw=np.array(xw), xkind=np.array(xkind, np.int8))
    EU = np.r_[eu, np.array(xu, np.int64)]
    EV = np.r_[ev, np.array(xv, np.int64)]
    EW = np.r_[ew, np.array(xw)]
    EP = np.r_[epiece, np.full(len(xu), -1, np.int32)]
    EK = np.r_[np.zeros(len(eu), np.int8), np.array(xkind, np.int8)]
    outlets = np.nonzero(np.bincount(EU, minlength=nn) == 0)[0]
    keep = drop_back_edges(nn, EU, EV)
    print(f"cycles: {(~keep).sum():,} edges dropped to make the network acyclic ({time.time() - t0:.0f}s)", flush=True)
    best, via, reached, _ = longest_upstream(nn, EU[keep], EV[keep], EW[keep])
    kidx = np.nonzero(keep)[0]
    via = np.where(via >= 0, kidx[np.maximum(via, 0)], -1)
    print(f"longest paths: {reached:,} of {nn:,} nodes ordered ({time.time() - t0:.0f}s)", flush=True)
    outlets = outlets[np.argsort(-best[outlets])]
    np.savez(os.path.join(DATA, "graph.npz"), nlon=nlon, nlat=nlat, EU=EU, EV=EV, EW=EW, EK=EK, EP=EP, keep=keep, best=best, via=via)
    # trace main stems. Distributaries of one river share most of their main stem: keep the longest
    used = np.zeros(len(EU), bool)
    systems = []
    for o in outlets:
        if best[o] < 100 or len(systems) >= TOP:
            break
        path_e = []
        v = o
        while via[v] >= 0:
            path_e.append(via[v])
            v = EU[via[v]]
        pe = np.array(path_e[::-1], np.int64)
        if EW[pe][used[pe]].sum() > 0.3 * best[o]:
            continue
        used[pe] = True
        systems.append((o, v, pe))
    print(f"{len(systems):,} systems traced ({time.time() - t0:.0f}s)", flush=True)
    out = []
    coords, coffs = [], [0]
    for o, src, pe in systems:
        sx, sy, mx, my = nlon[src], nlat[src], nlon[o], nlat[o]
        L = float(best[o])
        D = float(gc_km(sx, sy, mx, my))
        # names: lines of one river share a Wikidata item (or failing that a name). Rivers are named for
        # their lower course: walking up from the mouth, the first river to run 15% of the length (or 400 km).
        lake_km = float(EW[pe][EK[pe] == 1].sum())
        gap_km = float(EW[pe][EK[pe] >= 2].sum())
        km_from_mouth, labels, along = {}, {}, []
        main = ""
        wd_of = {}
        for k in pe:
            p = EP[k]
            if EK[k] == 0 and wd[p]:
                wd_of.setdefault(name_en[p], wd[p]); wd_of.setdefault(name[p], wd[p])
        wd_of.pop("", None)
        for k in pe[::-1]:
            if EK[k] != 0:
                continue
            p = EP[k]
            key = wd[p] or wd_of.get(name_en[p]) or wd_of.get(name[p]) or name_en[p] or name[p]
            if not key:
                continue
            labels.setdefault(key, {}).setdefault(name_en[p] or name[p], 0.0)
            labels[key][name_en[p] or name[p]] += EW[k]
            km_from_mouth[key] = km_from_mouth.get(key, 0.0) + EW[k]
            if not main and km_from_mouth[key] > min(0.15 * L, 400.0):
                main = key
            if not along or along[-1] != key:
                along.append(key)
        along = along[::-1]
        if not main and km_from_mouth:
            main = max(km_from_mouth, key=km_from_mouth.get)
        lab = {k: max(v, key=v.get) for k, v in labels.items()}
        dist_coast = float(coast_km(mx, my)[0])
        out.append(dict(name=lab.get(main, ""), wikidata=main if main.startswith("Q") else "",
                        names_along=[lab[k] for k in along], length_km=L, straight_km=D,
                        sinuosity=L / max(D, 1e-6), source=[float(sx), float(sy)], mouth=[float(mx), float(my)],
                        lake_km=lake_km, gap_km=gap_km, n_edges=len(pe), mouth_coast_km=dist_coast,
                        mouth_on_land=bool(coast_km.on_land(mx, my)),
                        terminus="sea" if dist_coast < 25 or not coast_km.on_land(mx, my) else "inland"))
        # full-resolution vertices of the main stem (for drawing and for scale analysis), simplified to ~30 m
        seq = []
        for k in pe:
            u_, v_ = EU[k], EV[k]
            if EK[k] != 0:
                seq.append([[nlon[u_], nlat[u_]], [nlon[v_], nlat[v_]]])
                continue
            p = EP[k]
            vs = np.arange(off[p], off[p + 1])
            ia = vs[node_of_j[vs] == u_]
            ib = vs[node_of_j[vs] == v_]
            if len(ia) and len(ib):
                i0 = ia[0]
                i1 = ib[ib > i0][0] if (ib > i0).any() else ib[-1]
                seq.append((xy[i0:i1 + 1] * 1e-7).tolist())
        line = np.concatenate([np.asarray(s_, np.float64) for s_ in seq]) if seq else np.zeros((0, 2))
        line = shapely.get_coordinates(shapely.simplify(shapely.LineString(line), 0.0003)) if len(line) > 1 else line
        coords.append(line.astype(np.float32))
        coffs.append(coffs[-1] + len(line))
    np.savez(os.path.join(DATA, "stems.npz"), xy=np.concatenate(coords), off=np.array(coffs))
    json.dump(out, open(os.path.join(DATA, "systems.json"), "w"), indent=0)
    s = np.array([r["sinuosity"] for r in out])
    Ls = np.array([r["length_km"] for r in out])
    print(f"top {len(out)}: lengths {Ls.min():,.0f}-{Ls.max():,.0f} km; sinuosity mean {s.mean():.3f}, median {np.median(s):.3f}, "
          f"ratio of totals {Ls.sum() / sum(r['straight_km'] for r in out):.3f}  ({time.time() - t0:.0f}s)")
    for r in out[:60]:
        print(f"  {r['name'][:28]:28s} {r['length_km']:7,.0f} km  straight {r['straight_km']:6,.0f}  S={r['sinuosity']:.2f}  "
              f"lake {r['lake_km']:4.0f}  gap {r['gap_km']:4.0f}  {r['terminus']:6s} via {' > '.join(r['names_along'][:6])[:90]}")


if __name__ == "__main__":
    main()
