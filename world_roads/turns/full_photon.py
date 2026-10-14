"""The lost photon on EVERY street: coin-flip driving from LA to New York on the full North American
street network (motorway ... residential and living streets; no service roads, tracks or paths).

The network is built from Overture connectors (the points where road segments meet), so each graph
edge is a stretch of road between two junction points, with its true geodesic length.

    python full_photon.py fetch          # stream North America -> data/na_full_{edges,nodes}.npy
    python full_photon.py sim [N]        # N coin-flip walks LA -> NYC (same rule as lost_photon.py)
"""
import os, sys, json, time
import numpy as np
import numba as nb
import shapely
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cities"))
import lost_photon as LP  # noqa: E402

DATA = os.path.join(HERE, "data")
NA = (-170.0, 7.0, -52.0, 72.0)
DRIVE = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street"]
R_EARTH = 6371.0088
HEX = np.full(256, 0, np.uint64)
for i, ch in enumerate(b"0123456789abcdef"):
    HEX[ch] = i
for i, ch in enumerate(b"ABCDEF"):
    HEX[ch] = 10 + i


def uuid64(arr):
    """64-bit key from a UUID string array: first 8 + last 8 hex digits."""
    s = np.asarray(arr.to_numpy(zero_copy_only=False), dtype="S36").view(np.uint8).reshape(-1, 36)
    sel = np.r_[0:8, 28:36]
    v = HEX[s[:, sel]]
    out = np.zeros(len(s), np.uint64)
    for k in range(16):
        out = (out << np.uint64(4)) | v[:, k]
    return out


def fetch():
    import fetch_cities as FC
    keys, tab = FC.rowgroups()
    w, s, e, n = NA
    hit = tab[(tab[:, 2] <= e) & (tab[:, 3] >= w) & (tab[:, 4] <= n) & (tab[:, 5] >= s)]
    jobs = [(keys[int(r[0])], int(r[1])) for r in hit]
    if os.environ.get("LIMIT"):
        jobs = jobs[:int(os.environ["LIMIT"])]
    print("row groups:", len(jobs), flush=True)

    def work(job):
        key, rg = job
        for attempt in range(8):
            try:
                pf = pq.ParquetFile(FC.fs.open(FC.BUCKET + key, block_size=8 << 20))
                t = pf.read_row_group(rg, columns=["subtype", "class", "geometry", "connectors", "bbox"])
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)
        b = t.column("bbox").combine_chunks()
        xmin = b.field("xmin").to_numpy(); xmax = b.field("xmax").to_numpy()
        ymin = b.field("ymin").to_numpy(); ymax = b.field("ymax").to_numpy()
        sub = t.column("subtype").to_numpy(zero_copy_only=False)
        cls = np.array(t.column("class").to_pylist(), dtype=object)
        m = (sub == "road") & np.isin(cls, DRIVE) & (xmin <= e) & (xmax >= w) & (ymin <= n) & (ymax >= s)
        if not m.any():
            return None
        t = t.filter(pa.array(m))
        cid = np.array([DRIVE.index(c) for c in cls[m]], np.int8)
        geoms = shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False))
        coords, idx = shapely.get_coordinates(geoms, return_index=True)
        la = np.radians(coords[:, 1]); lo = np.radians(coords[:, 0])
        same = idx[1:] == idx[:-1]
        h = np.sin((la[1:] - la[:-1]) / 2) ** 2 + np.cos(la[1:]) * np.cos(la[:-1]) * np.sin((lo[1:] - lo[:-1]) / 2) ** 2
        seg_km = np.bincount(idx[:-1][same], weights=2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h[same], 0, 1))),
                             minlength=len(geoms))
        con = t.column("connectors").combine_chunks()
        offs = con.offsets.to_numpy()
        flat = con.flatten()
        ids = uuid64(flat.field("connector_id"))
        at = flat.field("at").to_numpy(zero_copy_only=False)
        parent = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
        nxt = np.r_[parent[1:] == parent[:-1], False]
        i0 = np.nonzero(nxt)[0]
        u, v = ids[i0], ids[i0 + 1]
        km = ((at[i0 + 1] - at[i0]) * seg_km[parent[i0]]).astype(np.float32)
        ok = u != v
        pts = shapely.line_interpolate_point(geoms[parent], at, normalized=True)
        xy = shapely.get_coordinates(pts).astype(np.float32)
        uid, first = np.unique(ids, return_index=True)
        return u[ok], v[ok], km[ok], cid[parent[i0]][ok], uid, xy[first]

    E, N = [], []
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            if r is not None:
                E.append(r[:4]); N.append(r[4:])
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  edges so far {sum(len(x[0]) for x in E):,}  {time.time() - t0:.0f}s", flush=True)
    u, v, km, c = (np.concatenate([x[k] for x in E]) for k in range(4))
    nid = np.concatenate([x[0] for x in N]); nxy = np.concatenate([x[1] for x in N])
    nid, first = np.unique(nid, return_index=True)
    nxy = nxy[first]
    np.save(os.path.join(DATA, "na_full_u.npy"), u); np.save(os.path.join(DATA, "na_full_v.npy"), v)
    np.save(os.path.join(DATA, "na_full_km.npy"), km); np.save(os.path.join(DATA, "na_full_cls.npy"), c)
    np.save(os.path.join(DATA, "na_full_nid.npy"), nid); np.save(os.path.join(DATA, "na_full_nxy.npy"), nxy)
    print(f"edges {len(u):,}  nodes {len(nid):,}  road km {km.sum():,.0f}  {time.time() - t0:.0f}s")


def load():
    u = np.load(os.path.join(DATA, "na_full_u.npy")); v = np.load(os.path.join(DATA, "na_full_v.npy"))
    km = np.load(os.path.join(DATA, "na_full_km.npy")).astype(np.float64)
    nid = np.load(os.path.join(DATA, "na_full_nid.npy")); nxy = np.load(os.path.join(DATA, "na_full_nxy.npy"))
    a = np.searchsorted(nid, u); b = np.searchsorted(nid, v)
    del u, v
    n = len(nid)
    comp = connected_components(coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)), directed=False)[1]
    lon, lat = nxy[:, 0].astype(np.float64), nxy[:, 1].astype(np.float64)
    big = int(np.argmax(np.bincount(comp)))                     # the continental network
    keep = comp[a] == big
    a, b, km = a[keep], b[keep], km[keep]
    used, inv = np.unique(np.r_[a, b], return_inverse=True)
    m = len(a)
    a, b = inv[:m], inv[m:]
    lon, lat = lon[used], lat[used]
    n = len(used)
    ends = np.r_[a, b]; nbr = np.r_[b, a]; ln = np.r_[km, km]
    o = np.argsort(ends, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(ends, minlength=n))].astype(np.int64)
    G = dict(lon=lon, lat=lat, indptr=indptr, adj=nbr[o].astype(np.int64), alen=ln[o], n=n, m=m, km=float(km.sum()))
    G["start"] = int(np.argmin(LP.gc_km(lon, lat, *LP.LA)))
    G["goal"] = LP.gc_km(lon, lat, *LP.NYC) < LP.ARRIVE_KM
    return G


def sim(n_walks):
    t0 = time.time()
    G = load()
    names, place_of = LP.place_index(G)
    print(f"network: {G['n']:,} junctions, {G['m']:,} road pieces, {G['km']:,.0f} km  "
          f"goal junctions {int(G['goal'].sum()):,}  ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for seed in range(n_walks):
        steps, km, ok, visits, _ = LP._walk(G["indptr"], G["adj"], G["alen"], G["goal"], G["start"], seed,
                                            20_000_000_000, place_of, len(names), False)
        rows.append(dict(seed=seed, arrived=bool(ok), steps=int(steps), km=float(km), seconds=float(km / LP.C_KMS),
                         visits={nm: int(x) for nm, x in zip(names, visits)}))
        print(seed, ok, f"{km:,.0f} km", f"{km / LP.C_KMS:,.1f} s", f"{steps:,} steps", f"{time.time() - t0:.0f}s", flush=True)
        s = np.array([r["seconds"] for r in rows])
        summary = dict(n=len(rows), arrived=int(sum(r["arrived"] for r in rows)), median_s=float(np.median(s)),
                       mean_s=float(s.mean()), min_s=float(s.min()), max_s=float(s.max()),
                       network_km=G["km"], junctions=G["n"], road_pieces=G["m"])
        json.dump(dict(summary=summary, walks=rows), open(os.path.join(DATA, "full_photon_sim.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))





# ------------------------------------------------------------------ faster: spatially ordered graph, parallel walks
def morton(lon, lat):
    x = np.clip(((lon + 180) / 360 * 65535), 0, 65535).astype(np.uint64)
    y = np.clip(((lat + 90) / 180 * 65535), 0, 65535).astype(np.uint64)
    def spread(v):
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
        v = (v | (v << np.uint64(1))) & np.uint64(0x55555555)
        return v
    return spread(x) | (spread(y) << np.uint64(1))


def load_fast():
    cache = os.path.join(DATA, "na_full_graph.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    G = load()
    order = np.argsort(morton(G["lon"], G["lat"]), kind="stable")    # neighbours end up near each other in memory
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    n = G["n"]
    src = np.repeat(np.arange(n), np.diff(G["indptr"]))
    a, b, ln = rank[src], rank[G["adj"]], G["alen"]
    o = np.lexsort((b, a))
    indptr = np.r_[0, np.cumsum(np.bincount(a, minlength=n))].astype(np.int64)
    F = dict(indptr=indptr, adj=b[o].astype(np.int32), alen=ln[o].astype(np.float32),
             lon=G["lon"][order].astype(np.float32), lat=G["lat"][order].astype(np.float32),
             goal=G["goal"][order], start=np.int64(rank[G["start"]]), km=np.float64(G["km"]), m=np.int64(G["m"]))
    np.savez(cache, **F)
    return F


@nb.njit(parallel=True, cache=True)
def walks(indptr, adj, alen, goal, start, seeds, max_steps, every, n_samp):
    nw = len(seeds)
    steps_out = np.zeros(nw, np.int64); km_out = np.zeros(nw); ok_out = np.zeros(nw, np.bool_)
    samp = np.full((nw, n_samp), -1, np.int64)
    for w in nb.prange(nw):
        np.random.seed(seeds[w])
        prev = -1; cur = start; km = 0.0; ns = 0
        st = 0
        while st < max_steps:
            if goal[cur]:
                ok_out[w] = True
                break
            if st % every == 0 and ns < n_samp:
                samp[w, ns] = cur; ns += 1
            a = indptr[cur]; b = indptr[cur + 1]
            d = b - a
            n_ex = d
            for j in range(a, b):
                if adj[j] == prev:
                    n_ex -= 1
                    break
            if n_ex <= 0:
                j_pick = a
            else:
                k = np.random.randint(n_ex)
                j_pick = a
                skipped = False
                for j in range(a, b):
                    if not skipped and adj[j] == prev and n_ex < d:
                        skipped = True
                        continue
                    if k == 0:
                        j_pick = j
                        break
                    k -= 1
            km += alen[j_pick]
            prev = cur; cur = adj[j_pick]
            st += 1
        steps_out[w] = st; km_out[w] = km
    return steps_out, km_out, ok_out, samp


def sim_fast(n_walks, max_steps=60_000_000_000, every=10_000_000):
    t0 = time.time()
    F = load_fast()
    print(f"graph ready ({time.time() - t0:.0f}s): {len(F['lon']):,} junctions, goal {int(F['goal'].sum()):,}", flush=True)
    out = os.path.join(DATA, "full_photon_fast.json")
    rows = json.load(open(out))["walks"] if os.path.exists(out) else []
    while len(rows) < n_walks:
        seeds = np.arange(len(rows), len(rows) + 4, dtype=np.int64)
        st, km, ok, samp = walks(F["indptr"], F["adj"], F["alen"], F["goal"], int(F["start"]), seeds, max_steps, every, 6000)
        for i in range(len(seeds)):
            s = samp[i][samp[i] >= 0]
            rows.append(dict(seed=int(seeds[i]), arrived=bool(ok[i]), steps=int(st[i]), km=float(km[i]),
                             seconds=float(km[i] / LP.C_KMS),
                             track=[[round(float(F["lon"][x]), 3), round(float(F["lat"][x]), 3)] for x in s]))
            print(seeds[i], bool(ok[i]), f"{km[i]:,.0f} km", f"{km[i] / LP.C_KMS / 60:,.1f} min", f"{st[i]:,} steps",
                  f"{time.time() - t0:.0f}s", flush=True)
        sec = np.array([r["seconds"] for r in rows])
        summary = dict(n=len(rows), arrived=int(sum(r["arrived"] for r in rows)), median_s=float(np.median(sec)),
                       mean_s=float(sec.mean()), min_s=float(sec.min()), max_s=float(sec.max()),
                       network_km=float(F["km"]), junctions=int(len(F["lon"])), road_pieces=int(F["m"]))
        json.dump(dict(summary=summary, walks=rows), open(out, "w"))
        print(json.dumps(summary), flush=True)


@nb.njit(cache=True)
def replay(indptr, adj, alen, goal, start, seed, max_steps, place_of, n_places):
    """Same walk as walks() for one seed (identical RNG use), counting entries into named places."""
    np.random.seed(seed)
    prev = -1; cur = start; km = 0.0
    visits = np.zeros(n_places, np.int64); last = -1
    st = 0
    while st < max_steps:
        if goal[cur]:
            break
        a = indptr[cur]; b = indptr[cur + 1]
        d = b - a
        n_ex = d
        for j in range(a, b):
            if adj[j] == prev:
                n_ex -= 1
                break
        if n_ex <= 0:
            j_pick = a
        else:
            k = np.random.randint(n_ex)
            j_pick = a
            skipped = False
            for j in range(a, b):
                if not skipped and adj[j] == prev and n_ex < d:
                    skipped = True
                    continue
                if k == 0:
                    j_pick = j
                    break
                k -= 1
        km += alen[j_pick]
        prev = cur; cur = adj[j_pick]
        st += 1
        pl = place_of[cur]
        if pl >= 0 and pl != last:
            visits[pl] += 1
        last = pl
    return st, km, visits


if __name__ == "__main__" and sys.argv[1] == "fast":
    sim_fast(int(sys.argv[2]) if len(sys.argv) > 2 else 8)


if __name__ == "__main__" and sys.argv[1] in ("fetch", "sim"):
    if sys.argv[1] == "fetch":
        fetch()
    else:
        sim(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
