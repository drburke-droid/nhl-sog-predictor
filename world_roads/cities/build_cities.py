"""Street graph + light route for every city (same method as the Calgary film).

    python build_cities.py [name ...]      -> data/<slug>.npz, data/<slug>.json
Every public street within RADIUS_KM of the centre; the walk spreads out in rings from the centre.
"""
import os, sys, json, time
import numpy as np
import shapely
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "calgary"))
sys.path.insert(0, os.path.dirname(HERE))
from cities import CITIES, RADIUS_KM, slug  # noqa: E402
import city as calgary  # noqa: E402  (walk() is reused as is)

DATA = os.path.join(HERE, "data")
STREETS = calgary.STREETS
R_EARTH = 6371.0088


def circle(lon0, lat0, r_km, n=720):
    kx = np.radians(1) * R_EARTH * np.cos(np.radians(lat0)); ky = np.radians(1) * R_EARTH
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return shapely.Polygon(np.c_[lon0 + r_km * np.cos(a) / kx, lat0 + r_km * np.sin(a) / ky])


def build(name, center):
    lon0, lat0 = center
    kx = np.radians(1) * R_EARTH * np.cos(np.radians(lat0)); ky = np.radians(1) * R_EARTH
    t = pq.read_table(os.path.join(DATA, f"{slug(name)}.parquet"))
    cls = np.array(t.column("class").to_pylist(), dtype=object)
    keep = np.isin(cls, STREETS)
    geoms = shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False)[keep])
    cls = cls[keep]
    geoms = shapely.intersection(geoms, circle(lon0, lat0, RADIUS_KM))
    lines, gi = shapely.get_parts(geoms, return_index=True)
    ok = shapely.get_type_id(lines) == 1
    lines, gi = lines[ok], gi[ok]
    coords, idx = shapely.get_coordinates(lines, return_index=True)
    q = np.round(coords * 2e5).astype(np.int64)                 # ~0.5 m snapping merges shared connectors
    code = q[:, 0] * 10**9 + (q[:, 1] + 20_000_000)                # latitude offset keeps the low part positive
    nodes, inv = np.unique(code, return_inverse=True)
    same = idx[1:] == idx[:-1]
    a, b = inv[:-1][same], inv[1:][same]
    ecls = cls[gi[idx[:-1][same]]]
    good = a != b
    a, b, ecls = a[good], b[good], ecls[good]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    _, first = np.unique(lo * len(nodes) + hi, return_index=True)
    ea, eb, ecls = lo[first], hi[first], ecls[first]
    # drop isolated vertices left over from deduplication
    used, inv2 = np.unique(np.r_[ea, eb], return_inverse=True)
    m = len(ea)
    ea, eb = inv2[:m], inv2[m:]
    nodes = nodes[used]
    nlon = (nodes // 10**9) / 2e5
    nlat = (nodes % 10**9 - 20_000_000) / 2e5
    nx, ny = (nlon - lon0) * kx, (nlat - lat0) * ky
    elen = np.hypot(nx[ea] - nx[eb], ny[ea] - ny[eb])
    n = len(nodes)
    comp = connected_components(coo_matrix((np.ones(m), (ea, eb)), shape=(n, n)), directed=False)[1]
    key = np.hypot((nx[ea] + nx[eb]) / 2, (ny[ea] + ny[eb]) / 2)
    by = {c: float(elen[ecls == c].sum()) for c in STREETS}
    return dict(nx=nx, ny=ny, ea=ea.astype(np.int64), eb=eb.astype(np.int64), elen=elen, key=key,
                comp=comp, ecls=ecls, by_class=by)


def run(item):
    name, center = item
    out = os.path.join(DATA, f"{slug(name)}.npz")
    if os.path.exists(out):
        return name, json.load(open(out[:-4] + ".json"))
    t0 = time.time()
    G = build(name, center)
    rn, re_, cum, stats = calgary.walk(G, band=3.0)
    stats.update(name=name, center=list(center), radius_km=RADIUS_KM, by_class_km=G["by_class"],
                 nodes=int(len(G["nx"])), edges=int(len(G["ea"])), build_s=time.time() - t0)
    cid = np.array([STREETS.index(c) for c in G["ecls"]], np.int8)
    np.savez_compressed(out, nx=G["nx"].astype(np.float32), ny=G["ny"].astype(np.float32), ea=G["ea"].astype(np.int32),
                        eb=G["eb"].astype(np.int32), ecls=cid, rn=rn.astype(np.int32), re=re_.astype(np.int32),
                        cum=cum)
    json.dump(stats, open(out[:-4] + ".json", "w"), indent=1)
    print(f"{name}: {stats['street_km']:,.0f} km streets, walk {stats['walk_km']:,.0f} km, "
          f"{stats['seconds_at_c'] * 1000:.1f} ms, {time.time() - t0:.0f}s", flush=True)
    return name, stats


if __name__ == "__main__":
    only = sys.argv[1:]
    items = [(n, c) for n, c in CITIES if (not only or n in only)
             and os.path.exists(os.path.join(DATA, f"{slug(n)}.parquet"))]
    with ProcessPoolExecutor(int(os.environ.get("JOBS", 2))) as ex:
        for name, st in ex.map(run, items):
            pass
