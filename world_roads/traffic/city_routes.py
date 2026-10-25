"""Real city roads for the traffic-equilibrium video: a directed street graph timed at the speed limits,
and genuinely different routes between the trip's two ends (cities.py; TRAFFIC_CITY=nyc|calgary).

    python city_routes.py fetch     # Overture segments in the city's box, with names, speed limits, one-ways
    python city_routes.py build     # graph + routes -> data/<city>/graph.npz, ../media/traffic/routes_<city>.json

Free-flow time = length / speed limit: OpenStreetMap's maxspeed where it is mapped, a default for the road
class elsewhere. No traffic, no traffic lights, no turn delays: the 3 a.m. time of an ideal driver.
Alternatives use the via-node method of route planners: for every junction v, the fastest route from the
origin to v and on from v to the destination; keep the quickest ones that share little road with each other.
"""
import os, sys, json, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cities import C, CITY                         # noqa: E402
DATA = os.path.join(HERE, "data", CITY)
MEDIA = os.path.join(ROOT, "media", "traffic")
BBOX = C["bbox"]
DRIVE = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street"]
DEFAULT_KMH = C["default_kmh"]                    # where OSM has no maxspeed
LINK_KMH = C["link_kmh"]                          # ramps without a mapped limit
CENTRE = C["centre"]
KX = math.radians(1) * 6371.0088 * math.cos(math.radians(CENTRE[1]))
KY = math.radians(1) * 6371.0088
ORIGIN, DEST = C["origin"], C["dest"]


def xy(lon, lat):
    return (np.asarray(lon) - CENTRE[0]) * KX, (np.asarray(lat) - CENTRE[1]) * KY


# ------------------------------------------------------------------ fetch
def fetch():
    import pyarrow.parquet as pq, pyarrow as pa, fsspec
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, ROOT)
    from fetch_overture import list_files, BUCKET
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    cols = ["id", "names", "class", "subclass", "geometry", "bbox", "subtype", "access_restrictions", "speed_limits"]

    def hits(key):
        md = pq.ParquetFile(fs.open(BUCKET + key, block_size=1 << 20)).metadata
        names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)]
        ix = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}
        out = []
        for r in range(md.num_row_groups):
            st = {k: md.row_group(r).column(i).statistics for k, i in ix.items()}
            if st["bbox.xmin"].min <= BBOX[2] and st["bbox.xmax"].max >= BBOX[0] and \
               st["bbox.ymin"].min <= BBOX[3] and st["bbox.ymax"].max >= BBOX[1]:
                out.append(r)
        return key, out

    def read(key, rgs):
        pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
        out = []
        for r in rgs:
            for attempt in range(6):
                try:
                    t = pf.read_row_group(r, columns=cols)
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
                    pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
            b = t.column("bbox").combine_chunks()
            m = (t.column("subtype").to_numpy(zero_copy_only=False) == "road") & \
                (b.field("xmin").to_numpy() <= BBOX[2]) & (b.field("xmax").to_numpy() >= BBOX[0]) & \
                (b.field("ymin").to_numpy() <= BBOX[3]) & (b.field("ymax").to_numpy() >= BBOX[1])
            if m.any():
                out.append(t.filter(pa.array(m)).drop(["bbox", "subtype"]))
        return out
    os.makedirs(DATA, exist_ok=True)
    with ThreadPoolExecutor(32) as ex:
        hs = [h for h in ex.map(hits, list_files()) if h[1]]
    with ThreadPoolExecutor(8) as ex:
        tabs = [t for res in ex.map(lambda h: read(*h), hs) for t in res]
    tb = pa.concat_tables(tabs)
    pq.write_table(tb, os.path.join(DATA, "roads.parquet"))
    print("segments:", tb.num_rows)


# ------------------------------------------------------------------ graph
def _kmh(limits):
    """the unconditional max speed of a segment (km/h), or None"""
    for s in limits or []:
        w = s.get("when") or {}
        if any(w.get(k) for k in ("during", "using", "recognized", "mode", "vehicle")):
            continue
        mx = s.get("max_speed")
        if mx and mx.get("value"):
            v = float(mx["value"])
            return v * 1.609344 if (mx.get("unit") or "km/h") == "mph" else v
    return None


def _oneway(restr):
    """(forward allowed, backward allowed, open at all) from Overture access restrictions for cars"""
    fwd = bwd = ok = True
    for r in restr or []:
        if r.get("access_type") != "denied" or r.get("between"):
            continue
        w = r.get("when") or {}
        if any(w.get(k) for k in ("during", "using", "recognized", "vehicle")):
            continue
        modes = w.get("mode") or []
        if modes and not any(m in ("vehicle", "motor_vehicle", "car") for m in modes):
            continue
        h = w.get("heading")
        if h == "backward":
            bwd = False
        elif h == "forward":
            fwd = False
        elif not h:
            ok = False
    return fwd, bwd, ok


def build_graph():
    import pyarrow.parquet as pq, shapely
    t = pq.read_table(os.path.join(DATA, "roads.parquet"))
    cls = np.array(t.column("class").to_pylist(), dtype=object)
    sub = np.array(t.column("subclass").to_pylist(), dtype=object)
    keep = np.isin(cls, DRIVE)
    idx_keep = np.nonzero(keep)[0]
    names = t.column("names").to_pylist()
    lim = t.column("speed_limits").to_pylist()
    acc = t.column("access_restrictions").to_pylist()
    geoms = shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False)[keep])
    coords, own = shapely.get_coordinates(geoms, return_index=True)
    q = np.round(coords * 2e5).astype(np.int64)                 # ~0.5 m snapping joins segments at connectors
    code = q[:, 0] * 10**9 + q[:, 1]
    nodes, inv = np.unique(code, return_inverse=True)
    same = own[1:] == own[:-1]
    a, b, seg = inv[:-1][same], inv[1:][same], own[:-1][same]
    good = a != b
    a, b, seg = a[good], b[good], seg[good]
    nlon, nlat = (nodes // 10**9) / 2e5, (nodes % 10**9) / 2e5
    nx, ny = xy(nlon, nlat)
    elen = np.hypot(nx[a] - nx[b], ny[a] - ny[b])               # km (local projection, < 0.2 % error here)
    S = len(idx_keep)
    kmh = np.zeros(S)
    fw, bw = np.ones(S, bool), np.ones(S, bool)
    nm = np.empty(S, dtype=object)
    mapped = np.zeros(S, bool)
    for j, i in enumerate(idx_keep):
        v = _kmh(lim[i])
        mapped[j] = v is not None
        if v is None:
            v = LINK_KMH if (sub[i] == "link" and cls[i] in ("motorway", "trunk", "primary")) else DEFAULT_KMH[cls[i]]
        kmh[j] = max(v, 5.0)
        f, bk, ok = _oneway(acc[i])
        fw[j], bw[j] = f and ok, bk and ok
        nm[j] = (names[i] or {}).get("primary") or ""
    ecls = cls[idx_keep][seg]
    emin = elen / kmh[seg] * 60.0
    np.savez_compressed(os.path.join(DATA, "graph.npz"), nx=nx, ny=ny, a=a, b=b, seg=seg, elen=elen, emin=emin,
                        fw=fw[seg], bw=bw[seg], ecls=ecls.astype("U16"), seg_name=nm.astype("U64"),
                        seg_kmh=kmh, seg_mapped=mapped, seg_cls=cls[idx_keep].astype("U16"))
    print(f"nodes {len(nodes):,}  edges {len(a):,}  km {elen.sum():,.0f}  maxspeed mapped on "
          f"{(elen * mapped[seg]).sum() / elen.sum():.0%} of road length; one-way edges {(~(fw[seg] & bw[seg])).sum():,}")


def load_graph():
    return dict(np.load(os.path.join(DATA, "graph.npz"), allow_pickle=False))


# ------------------------------------------------------------------ routes
def nearest(g, lon, lat, classes):
    x, y = xy(lon, lat)
    ok = np.isin(g["ecls"], classes)
    nodes = np.unique(np.r_[g["a"][ok], g["b"][ok]])
    i = nodes[np.argmin(np.hypot(g["nx"][nodes] - x, g["ny"][nodes] - y))]
    return int(i)


def directed(g):
    """arcs (tail, head, minutes, edge id) honouring one-way rules"""
    a, b, w = g["a"], g["b"], g["emin"]
    e = np.arange(len(a))
    fw, bw = g["fw"], g["bw"]
    return (np.r_[a[fw], b[bw]], np.r_[b[fw], a[bw]], np.r_[w[fw], w[bw]], np.r_[e[fw], e[bw]])


def route_search(g, o, d, k_all=20, slack=1.7, max_overlap=0.8, max_tries=150000, core_km=0.0):
    """via-node alternatives, quickest first, each sharing at most `max_overlap` of its length with every route
    kept before it. core_km > 0 measures that overlap only away from the trip's two ends (every route has to
    leave the airport, or arrive downtown, the same way)"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n = len(g["nx"])
    tl, hd, w, eid = directed(g)
    # parallel arcs: keep the fastest (csr would add them up)
    key = tl.astype(np.int64) * n + hd
    order = np.lexsort((w, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    sel = order[first]
    tl, hd, w, eid = tl[sel], hd[sel], w[sel], eid[sel]
    A = csr_matrix((w, (tl, hd)), shape=(n, n))
    arc_e = {}                                              # (tail, head) -> edge id
    for t_, h_, e_ in zip(tl, hd, eid):
        arc_e[(int(t_), int(h_))] = int(e_)
    dO, pO = dijkstra(A, indices=o, return_predecessors=True)
    dD, pD = dijkstra(A.T.tocsr(), indices=d, return_predecessors=True)
    best = dO[d]
    print(f"fastest at free flow: {best:.1f} min")

    def path_via(v):
        left = [v]
        while left[-1] != o:
            p = pO[left[-1]]
            if p < 0:
                return None
            left.append(int(p))
        left = left[::-1]
        right = [v]
        while right[-1] != d:
            p = pD[right[-1]]
            if p < 0:
                return None
            right.append(int(p))
        nodes_ = left + right[1:]
        if len(set(nodes_)) != len(nodes_):
            return None                                    # the two halves meet again: a detour with a U-turn
        return nodes_

    tot = dO + dD
    cand = np.nonzero(np.isfinite(tot) & (tot <= best * slack))[0]
    cand = cand[np.argsort(tot[cand], kind="stable")]
    routes, on_route = [], np.zeros(n, bool)

    def edges_of(nodes_):
        return np.array([arc_e[(u, v)] for u, v in zip(nodes_[:-1], nodes_[1:])])

    mx, my = (g["nx"][g["a"]] + g["nx"][g["b"]]) / 2, (g["ny"][g["a"]] + g["ny"][g["b"]]) / 2
    far = (np.hypot(mx - g["nx"][o], my - g["ny"][o]) > core_km) & (np.hypot(mx - g["nx"][d], my - g["ny"][d]) > core_km)
    wlen = g["elen"] * far

    def overlap(e1, e2):
        s = np.intersect1d(e1, e2)
        return wlen[s].sum() / max(1e-9, min(wlen[e1].sum(), wlen[e2].sum()))

    def local_ok(nodes_, cost):
        """reject 'alternatives' that detour just to touch v: the stretch around v must itself be near-optimal
        (a T-test as in Abraham et al. 2013): the route's own time between points ~15 % of the trip either side
        of v must be within 20 % of the fastest time between them"""
        cum = np.r_[0, np.cumsum([A[u, v] for u, v in zip(nodes_[:-1], nodes_[1:])])]
        T = cum[-1]
        iv = nodes_.index(v_cur)
        lo = int(np.searchsorted(cum, max(0, cum[iv] - 0.15 * T)))
        hi = int(min(len(nodes_) - 1, np.searchsorted(cum, cum[iv] + 0.15 * T)))
        if hi <= lo:
            return True
        dd = dijkstra(A, indices=nodes_[lo], limit=(cum[hi] - cum[lo]) * 1.25)[nodes_[hi]]
        return (cum[hi] - cum[lo]) <= 1.2 * dd + 1e-9

    tried = 0
    for v in cand:
        if on_route[v]:
            continue
        tried += 1
        if tried > max_tries or len(routes) >= k_all:
            break
        nodes_ = path_via(int(v))
        if nodes_ is None:
            continue
        v_cur = int(v)
        e = edges_of(nodes_)
        ov = max((overlap(e, r["edges"]) for r in routes), default=0.0)
        if ov > max_overlap:
            on_route[nodes_] = True                        # nearby via nodes would mostly give this route again
            continue
        if not local_ok(nodes_, tot[v]):
            continue
        routes.append(dict(nodes=nodes_, edges=e, minutes=float(tot[v]), via=v_cur, overlap=float(ov)))
        on_route[nodes_] = True
    print(f"tried {tried} via nodes, kept {len(routes)} routes")
    return routes


def via_sample(g, o, d, n=400, seed=7):
    """routes forced through n random junctions on major roads across the city (fastest to the junction, then
    fastest on): a picture of how many slower ways across town there are"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    N = len(g["nx"])
    tl, hd, w, _ = directed(g)
    A = csr_matrix((w, (tl, hd)), shape=(N, N))            # duplicates add up, so take min first
    key = tl.astype(np.int64) * N + hd
    order = np.lexsort((w, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    sel = order[first]
    A = csr_matrix((w[sel], (tl[sel], hd[sel])), shape=(N, N))
    dO, pO = dijkstra(A, indices=o, return_predecessors=True)
    dD, pD = dijkstra(A.T.tocsr(), indices=d, return_predecessors=True)
    maj = np.isin(g["ecls"], ["motorway", "trunk", "primary", "secondary"])
    cand = np.unique(np.r_[g["a"][maj], g["b"][maj]])
    r = np.hypot(g["nx"][cand], g["ny"][cand])
    cand = cand[(r < C["via_radius_km"]) & np.isfinite(dO[cand] + dD[cand])]
    rng = np.random.default_rng(seed)
    pick = rng.choice(cand, size=min(n * 3, len(cand)), replace=False)
    out = []
    for v in pick:
        left = [int(v)]
        while left[-1] != o:
            left.append(int(pO[left[-1]]))
        right = [int(v)]
        while right[-1] != d:
            right.append(int(pD[right[-1]]))
        nodes_ = left[::-1] + right[1:]
        if len(set(nodes_)) != len(nodes_):
            continue                                       # would have to double back on itself
        out.append((float(dO[v] + dD[v]), nodes_))
        if len(out) >= n:
            break
    return out


def simplify(g, nodes_, tol=0.03):
    import shapely
    ls = shapely.LineString(np.c_[g["nx"][nodes_], g["ny"][nodes_]])
    return np.round(np.asarray(shapely.simplify(ls, tol).coords), 3).tolist()


def describe(g, r):
    e = r["edges"]
    km = float(g["elen"][e].sum())
    seg = g["seg"][e]
    names = g["seg_name"][seg]
    by = {}
    for nm_, L, c in zip(names, g["elen"][e], g["ecls"][e]):
        if nm_ and c in ("motorway", "trunk", "primary", "secondary"):
            by[nm_] = by.get(nm_, 0.0) + float(L)
    main = [k for k, v in sorted(by.items(), key=lambda kv: -kv[1]) if v > 1.5][:3]
    cls_km = {}
    for c, L in zip(g["ecls"][e], g["elen"][e]):
        cls_km[c] = cls_km.get(c, 0.0) + float(L)
    mapped = float((g["elen"][e] * g["seg_mapped"][seg]).sum() / max(km, 1e-9))
    return km, main, cls_km, mapped


def crossing_routes(g, o, d, crossings):
    """the fastest route through each named crossing (a bridge or tunnel: a list of its segment names). Only
    nodes inside the crossing count, so a route has to cross it, not just touch the junction at one end."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n = len(g["nx"])
    tl, hd, w, eid = directed(g)
    key = tl.astype(np.int64) * n + hd
    order = np.lexsort((w, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    sel = order[first]
    tl, hd, w, eid = tl[sel], hd[sel], w[sel], eid[sel]
    A = csr_matrix((w, (tl, hd)), shape=(n, n))
    arc = {(int(x), int(y)): int(e) for x, y, e in zip(tl, hd, eid)}
    dO, pO = dijkstra(A, indices=o, return_predecessors=True)
    dD, pD = dijkstra(A.T.tocsr(), indices=d, return_predecessors=True)
    names = g["seg_name"][g["seg"]]
    out = []
    for group in crossings:
        m = np.isin(names, group)
        inside = np.setdiff1d(np.unique(np.r_[g["a"][m], g["b"][m]]), np.unique(np.r_[g["a"][~m], g["b"][~m]]))
        tot = dO[inside] + dD[inside]
        v = int(inside[np.argmin(tot)])
        left, right = [v], [v]
        while left[-1] != o:
            left.append(int(pO[left[-1]]))
        while right[-1] != d:
            right.append(int(pD[right[-1]]))
        nodes_ = left[::-1] + right[1:]
        assert len(set(nodes_)) == len(nodes_), group
        e = np.array([arc[(x, y)] for x, y in zip(nodes_[:-1], nodes_[1:])])
        out.append(dict(nodes=nodes_, edges=e, minutes=float(tot.min()), via=v, overlap=float("nan"), crossing=group[0]))
        print(f"via {group[0]}: {tot.min():.1f} min")
    return out


def pick_three(g, routes, within=1.15):
    """the three quick routes that share the least road with each other"""
    import itertools
    best = routes[0]["minutes"]
    ok = [i for i, r in enumerate(routes) if r["minutes"] <= within * best]

    def ov(i, j):
        s = np.intersect1d(routes[i]["edges"], routes[j]["edges"])
        return g["elen"][s].sum() / min(g["elen"][routes[i]["edges"]].sum(), g["elen"][routes[j]["edges"]].sum())
    trip = min(itertools.combinations(ok, 3),
               key=lambda c: (round(max(ov(i, j) for i, j in itertools.combinations(c, 2)), 2), sum(c)))
    return list(trip), {f"{i}-{j}": round(float(ov(i, j)), 3) for i, j in itertools.combinations(trip, 2)}


def build():
    g = load_graph()
    o = nearest(g, ORIGIN[0], ORIGIN[1], C["snap_origin"])
    d = nearest(g, DEST[0], DEST[1], C["snap_dest"])
    routes = route_search(g, o, d)
    if C.get("crossings"):
        main = []
        for r in crossing_routes(g, o, d, C["crossings"]):
            same = [k for k, q in enumerate(routes) if len(q["edges"]) == len(r["edges"]) and np.array_equal(np.sort(q["edges"]), np.sort(r["edges"]))]
            if same:
                routes[same[0]]["crossing"] = r["crossing"]
                main.append(same[0])
            else:
                routes.append(r)
                main.append(len(routes) - 1)
        main.sort(key=lambda k: routes[k]["minutes"])

        def ov(i, j):
            s_ = np.intersect1d(routes[i]["edges"], routes[j]["edges"])
            return g["elen"][s_].sum() / min(g["elen"][routes[i]["edges"]].sum(), g["elen"][routes[j]["edges"]].sum())
        import itertools
        pair_overlap = {f"{i}-{j}": round(float(ov(i, j)), 3) for i, j in itertools.combinations(main, 2)}
    else:
        main, pair_overlap = pick_three(g, routes)
    out = dict(origin=dict(name=ORIGIN[2], x=float(g["nx"][o]), y=float(g["ny"][o])),
               dest=dict(name=DEST[2], x=float(g["nx"][d]), y=float(g["ny"][d])),
               city=CITY, centre_lonlat=CENTRE, km_per_deg=[KX, KY], main=[int(i) for i in main], main_overlap=pair_overlap, routes=[])
    for k, r in enumerate(routes):
        km, via, cls_km, mapped = describe(g, r)
        out["routes"].append(dict(rank=k, minutes=round(r["minutes"], 3), km=round(km, 3), via=via, crossing=r.get("crossing"),
                                  max_overlap=round(r["overlap"], 3), km_by_class={c: round(v, 2) for c, v in cls_km.items()},
                                  speed_limit_mapped_share=round(mapped, 3),
                                  xy=simplify(g, r["nodes"], 0.004)))
        if k in main:                                      # full geometry + time along the route, for the race
            em = g["emin"][r["edges"]]
            out["routes"][-1]["detail"] = dict(xy=np.round(np.c_[g["nx"][r["nodes"]], g["ny"][r["nodes"]]], 3).tolist(),
                                               tmin=np.round(np.r_[0, np.cumsum(em)], 4).tolist())
        print(f"{k:2d} {r['minutes']:6.1f} min {km:5.1f} km overlap {r['overlap']:.2f} via {', '.join(via)}")
    vs = via_sample(g, o, d)
    out["via_sample"] = [dict(minutes=round(t, 3), xy=simplify(g, nd)) for t, nd in vs]
    ts = np.array([t for t, _ in vs])
    print(f"via sample: {len(ts)} routes, {ts.min():.1f}-{ts.max():.1f} min, median {np.median(ts):.1f}")
    os.makedirs(MEDIA, exist_ok=True)
    json.dump(out, open(os.path.join(MEDIA, f"routes_{CITY}.json"), "w"))
    print("main three:", main, pair_overlap)
    print("wrote", os.path.join(MEDIA, f"routes_{CITY}.json"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        fetch()
    elif cmd == "graph":
        build_graph()
    else:
        if not os.path.exists(os.path.join(DATA, "graph.npz")):
            build_graph()
        build()
