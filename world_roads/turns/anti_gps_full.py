"""Anti-GPS on the FULL street network (residential and unclassified side roads included).

Same rule as anti_gps.py: at every intersection the GPS recomputes the fastest route to New York and
you take any other road (no U-turns except at dead ends; GPS-says-U-turn means any road forward).

A full continental street graph is too big, so this works on a large box around Southern California
with every drivable street in it, collapsed to intersections. GPS distances inside the box are exact
shortest paths to the box edge plus, from there, the continental major-road distance to NYC (the
network used in lost_photon.py). If the set of places reachable by wrong turns never touches the
edge of the box, the answer holds for the whole continent: you can never leave.

    python anti_gps_full.py fetch W S E N [tag]   # stream Overture segments -> data/<tag>_edges.npz
    python anti_gps_full.py solve [tag] [classes] # reachability + random wrong-turn drivers
"""
import os, sys, json, time
import numpy as np
import numba as nb
import shapely
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra, connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cities"))
DATA = os.path.join(HERE, "data")
R_EARTH = 6371.0088
CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
           "living_street", "service", "unknown"]
UTURN = os.environ.get("UTURN", "0") == "1"           # allow a U-turn at an intersection as a "wrong" choice
DRIVE = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street"]


# ------------------------------------------------------------------------------------------ fetch
def fetch(bbox, tag):
    import fetch_cities as FC
    keys, tab = FC.rowgroups()
    w, s, e, n = bbox
    hit = tab[(tab[:, 2] <= e) & (tab[:, 3] >= w) & (tab[:, 4] <= n) & (tab[:, 5] >= s)]
    jobs = [(keys[int(r[0])], int(r[1])) for r in hit]
    print("row groups:", len(jobs), flush=True)

    def work(job):
        key, rg = job
        for attempt in range(6):
            try:
                pf = pq.ParquetFile(FC.fs.open(FC.BUCKET + key, block_size=8 << 20))
                t = pf.read_row_group(rg, columns=["subtype", "class", "geometry", "bbox"])
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        b = t.column("bbox").combine_chunks()
        xmin = b.field("xmin").to_numpy(); xmax = b.field("xmax").to_numpy()
        ymin = b.field("ymin").to_numpy(); ymax = b.field("ymax").to_numpy()
        sub = t.column("subtype").to_numpy(zero_copy_only=False)
        cls = np.array(t.column("class").to_pylist(), dtype=object)
        cls = np.where(cls == None, "unknown", cls)  # noqa: E711
        m = (sub == "road") & np.isin(cls, CLASSES) & (xmin <= e) & (xmax >= w) & (ymin <= n) & (ymax >= s)
        if not m.any():
            return None
        geoms = shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False)[m])
        cid = np.array([CLASSES.index(c) for c in cls[m]], np.int8)
        coords, idx = shapely.get_coordinates(geoms, return_index=True)
        q = np.round(coords * 2e5).astype(np.int64)
        code = q[:, 0] * 10**9 + (q[:, 1] + 20_000_000)
        same = idx[1:] == idx[:-1]
        la = np.radians(coords[:, 1]); lo = np.radians(coords[:, 0])
        h = np.sin((la[1:] - la[:-1]) / 2) ** 2 + np.cos(la[1:]) * np.cos(la[:-1]) * np.sin((lo[1:] - lo[:-1]) / 2) ** 2
        km = 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        a, bb = code[:-1][same], code[1:][same]
        good = a != bb
        return a[good], bb[good], km[same][good].astype(np.float32), cid[idx[:-1][same]][good]

    parts = []
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            if r is not None:
                parts.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    a, b, km, c = (np.concatenate([p[k] for p in parts]) for k in range(4))
    np.savez(os.path.join(DATA, f"{tag}_edges.npz"), a=a, b=b, km=km, c=c, bbox=np.array(bbox))
    print("edges:", len(a), f"{time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------ graph
@nb.njit(cache=True)
def contract(indptr, nbr, nlen, deg):
    """Collapse chains of degree-2 vertices. Returns contracted edges (u, v, km) between key vertices."""
    n = len(deg)
    is_key = deg != 2
    eu = np.empty(len(nbr), np.int64); ev = np.empty(len(nbr), np.int64); el = np.empty(len(nbr), np.float64)
    m = 0
    seen = np.zeros(len(nbr), np.bool_)                      # half-edges already walked
    for u in range(n):
        if not is_key[u]:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            if seen[j]:
                continue
            seen[j] = True
            prev, cur, L = u, nbr[j], nlen[j]
            while not is_key[cur]:
                a = indptr[cur]
                jn = a if nbr[a] != prev else a + 1
                if nbr[a] == prev and nbr[a + 1] == prev:     # 2-cycle through cur
                    jn = a + 1
                seen[jn] = True
                L += nlen[jn]
                prev, cur = cur, nbr[jn]
            # mark the reverse half-edge of the last step as walked
            for k in range(indptr[cur], indptr[cur + 1]):
                if nbr[k] == prev and not seen[k]:
                    seen[k] = True
                    break
            eu[m] = u; ev[m] = cur; el[m] = L; m += 1
    return eu[:m], ev[:m], el[:m]


def build(tag, classes):
    z = np.load(os.path.join(DATA, f"{tag}_edges.npz"))
    keep = np.isin(z["c"], [CLASSES.index(c) for c in classes])
    a, b, km = z["a"][keep], z["b"][keep], z["km"][keep].astype(np.float64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    _, first = np.unique(np.stack([lo, hi], 1), axis=0, return_index=True)
    lo, hi, km = lo[first], hi[first], km[first]
    nodes, inv = np.unique(np.r_[lo, hi], return_inverse=True)
    m = len(lo)
    ea, eb = inv[:m], inv[m:]
    n = len(nodes)
    ends = np.r_[ea, eb]; nb_ = np.r_[eb, ea]; ln = np.r_[km, km]
    o = np.argsort(ends, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(ends, minlength=n))].astype(np.int64)
    deg = np.diff(indptr)
    cu, cv, cl = contract(indptr, nb_[o].astype(np.int64), ln[o], deg)
    keyv = np.unique(np.r_[cu, cv])
    remap = np.full(n, -1, np.int64); remap[keyv] = np.arange(len(keyv))
    lon = (nodes[keyv] // 10**9) / 2e5
    lat = (nodes[keyv] % 10**9 - 20_000_000) / 2e5
    print(f"vertices {n:,} edges {m:,} -> intersections/dead ends {len(keyv):,}, roads {len(cu):,}", flush=True)
    return dict(lon=lon, lat=lat, u=remap[cu], v=remap[cv], km=cl, bbox=z["bbox"])


# ------------------------------------------------------------------------------------------ solve
@nb.njit(cache=True)
def options(hstart, hhead, hlen, twin, dist, h, out, start_node, uturn):
    """Allowed half-edges after arriving by half-edge h (h < 0: at start_node, no previous road)."""
    v = hhead[h] if h >= 0 else start_node
    a, b = hstart[v], hstart[v + 1]
    tw = twin[h] if h >= 0 else -1
    n_fwd = 0
    for j in range(a, b):
        if j != tw:
            n_fwd += 1
    k = 0
    if n_fwd == 0:
        for j in range(a, b):
            out[k] = j; k += 1
        return k
    g, best = -1, 1e18                                   # GPS: next road on the fastest route to NYC
    for j in range(a, b):
        d = hlen[j] + dist[hhead[j]]
        if d < best:
            best, g = d, j
    for j in range(a, b):
        if (j != tw or uturn) and j != g:
            out[k] = j; k += 1
    if k == 0:                                           # only the GPS road leads on (cannot happen at
        for j in range(a, b):                            # a real intersection; kept for safety)
            if j != tw:
                out[k] = j; k += 1
    return k


@nb.njit(cache=True)
def transitions(hstart, hhead, hlen, twin, dist, uturn):
    H = len(hhead)
    rows = np.empty(H * 3, np.int64); cols = np.empty(H * 3, np.int64)
    m = 0
    buf = np.empty(256, np.int64)
    for h in range(H):
        k = options(hstart, hhead, hlen, twin, dist, h, buf, -1, uturn)
        for i in range(k):
            if m >= len(rows):
                r2 = np.empty(len(rows) * 2, np.int64); c2 = np.empty(len(rows) * 2, np.int64)
                r2[:m] = rows[:m]; c2[:m] = cols[:m]; rows, cols = r2, c2
            rows[m] = h; cols[m] = buf[i]; m += 1
    return rows[:m], cols[:m]


@nb.njit(cache=True)
def walk(hstart, hhead, hlen, twin, dist, start, seed, steps, edge_node, uturn):
    np.random.seed(seed)
    buf = np.empty(256, np.int64)
    k = options(hstart, hhead, hlen, twin, dist, -1, buf, start, uturn)
    h = buf[np.random.randint(k)]
    km = hlen[h]
    visits = np.zeros(len(hstart) - 1, np.int32)
    for i in range(steps):
        v = hhead[h]
        visits[v] += 1
        if edge_node[v]:
            return True, i, km, v, visits
        k = options(hstart, hhead, hlen, twin, dist, h, buf, -1, uturn)
        h = buf[np.random.randint(k)]
        km += hlen[h]
    return False, steps, km, hhead[h], visits


def solve(tag, classes):
    import lost_photon as LP
    t0 = time.time()
    Gf = build(tag, classes)
    lon, lat, u, v, km = Gf["lon"], Gf["lat"], Gf["u"], Gf["v"], Gf["km"]
    n = len(lon)
    w, s, e, nn = Gf["bbox"]
    # keep the piece of the network that contains LA
    comp = connected_components(coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)), directed=False)[1]
    start = int(np.argmin(LP.gc_km(lon, lat, *LP.LA)))
    # box edge: anything within 3 km of the box boundary counts as "got out"
    kx = np.radians(1) * R_EARTH * np.cos(np.radians(lat)); ky = np.radians(1) * R_EARTH
    edge_km = np.minimum.reduce([(lon - w) * kx, (e - lon) * kx, (lat - s) * ky, (nn - lat) * ky])
    edge_node = edge_km < 3.0
    # GPS distance to NYC: exact inside the box, continental major-road distance from the box edge
    import gimmicks as GM
    Gc = GM.prepare(LP.load())
    from scipy.spatial import cKDTree
    cxyz = lambda lo_, la_: np.stack([np.cos(np.radians(la_)) * np.cos(np.radians(lo_)),
                                      np.cos(np.radians(la_)) * np.sin(np.radians(lo_)), np.sin(np.radians(la_))], 1)
    tree = cKDTree(cxyz(Gc["lon"], Gc["lat"]))
    en = np.nonzero(edge_node)[0]
    dd, ii = tree.query(cxyz(lon[en], lat[en]), k=8)
    exit_cost = np.min(Gc["dist"][ii] + 1.25 * dd * R_EARTH, axis=1)
    S = n                                                     # super-sink
    A = csr_matrix((np.r_[km, km, exit_cost], (np.r_[u, v, en], np.r_[v, u, np.full(len(en), S)])), shape=(n + 1, n + 1))
    dist = dijkstra(A, indices=S, directed=False)[:n]
    # half-edge structure
    hu = np.r_[u, v]; hv = np.r_[v, u]; hl = np.r_[km, km]
    M = len(u)
    twin0 = np.r_[np.arange(M) + M, np.arange(M)]
    o = np.argsort(hu, kind="stable")
    inv = np.empty_like(o); inv[o] = np.arange(len(o))
    hstart = np.r_[0, np.cumsum(np.bincount(hu, minlength=n))].astype(np.int64)
    hhead = hv[o].astype(np.int64); hlen = hl[o]; twin = inv[twin0[o]].astype(np.int64)
    rows, cols = transitions(hstart, hhead, hlen, twin, dist, UTURN)
    H = len(hhead)
    T = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(H, H))
    buf = np.empty(256, np.int64)
    k = options(hstart, hhead, hlen, twin, dist, -1, buf, start, UTURN)
    reach = np.zeros(H, bool)
    for h in buf[:k]:
        if not reach[h]:
            reach[breadth_first_order(T, int(h), directed=True, return_predecessors=False)] = True
    rnodes = np.unique(hhead[reach])
    escaped = bool(edge_node[rnodes].any())
    d_ny = LP.gc_km(lon[rnodes], lat[rnodes], *LP.NYC)
    d_la = LP.gc_km(lon[rnodes], lat[rnodes], *LP.LA)
    res = dict(tag=tag, classes=classes, intersections=int(n), roads=int(M), la_component_size=int((comp == comp[start]).sum()),
               reachable_intersections=int(len(rnodes)), reachable_share=float(len(rnodes) / n),
               escaped_box=escaped, box=[float(x) for x in Gf["bbox"]],
               reach_bbox=[float(lon[rnodes].min()), float(lat[rnodes].min()), float(lon[rnodes].max()), float(lat[rnodes].max())],
               closest_to_nyc=[float(lon[rnodes][np.argmin(d_ny)]), float(lat[rnodes][np.argmin(d_ny)]), float(d_ny.min())],
               farthest_from_la_km=float(d_la.max()), gps_km_la_to_nyc=float(dist[start]))
    print(json.dumps(res, indent=1), f"{time.time() - t0:.0f}s", flush=True)
    # random wrong-turn drivers: where do they end up?
    walks = []
    for seed in range(int(os.environ.get("WALKS", 4))):
        ok, steps, kmw, end, visits = walk(hstart, hhead, hlen, twin, dist, start, seed, 300_000_000, edge_node, UTURN)
        late = np.argsort(-visits)[:5]
        walks.append(dict(seed=seed, escaped=bool(ok), km=float(kmw), end=[float(lon[end]), float(lat[end])],
                          most_visited=[[round(float(lon[x]), 3), round(float(lat[x]), 3)] for x in late]))
        print(json.dumps(walks[-1]), f"{time.time() - t0:.0f}s", flush=True)
    res["walks"] = walks
    sfx = f"{tag}_{len(classes)}" + ("_uturn" if UTURN else "")
    res["uturn_allowed"] = UTURN
    json.dump(res, open(os.path.join(DATA, f"anti_gps_full_{sfx}.json"), "w"), indent=1)
    np.save(os.path.join(DATA, f"anti_gps_full_{sfx}_reach.npy"), np.c_[lon[rnodes], lat[rnodes]].astype(np.float32))


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(tuple(map(float, sys.argv[2:6])), sys.argv[6] if len(sys.argv) > 6 else "socal")
    else:
        tag = sys.argv[2] if len(sys.argv) > 2 else "socal"
        cl = sys.argv[3].split(",") if len(sys.argv) > 3 else DRIVE
        solve(tag, cl)
