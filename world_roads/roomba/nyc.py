"""The Roomba photon in New York City: every street inside city limits (the five boroughs).

The street network is the USA network's (usa.py: Overture, motorway through residential and living
streets) cut to the City of New York (Overture divisions: the city's land, plus New York State water
within ~2.5 km so the bridges between boroughs stay), keeping the largest connected network. Roads that leave the city (to New Jersey, Westchester, Nassau) are cut, as Calgary's were.
The photon starts in Times Square.

    python nyc.py boundary          -> data/nyc_boundary.wkt
    python nyc.py build             -> data/nyc_graph.npz
    python nyc.py many 200          -> data/nyc_cover_many.json (summaries)
    python nyc.py feature SEED      -> data/nyc_first|log|path_<seed>.npy
"""
import os, re, sys, json, time, urllib.request
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cover as CV  # noqa: E402

DATA = os.path.join(HERE, "data")
BBOX = (-74.28, 40.48, -73.68, 40.93)
START = (-73.98555, 40.75800)                     # Times Square (7th Ave and Broadway at W 45th St)
LON0, LAT0 = START
KX, KY = 111.320 * np.cos(np.radians(LAT0)), 110.574    # km per degree here


def boundary():
    import pyarrow.parquet as pq
    import fsspec
    from fetch_overture import BUCKET, RELEASE
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    pre = f"release/{RELEASE}/theme=divisions/type=division_area/"
    xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={pre}").read().decode()
    keys = [k for k in re.findall(r"<Key>([^<]*)</Key>", xml) if k.endswith(".parquet")]
    found = []
    for key in keys:
        pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=1 << 20))
        md = pf.metadata
        names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)]
        ix = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}
        for r in range(md.num_row_groups):
            g = md.row_group(r)
            st = {k: g.column(i).statistics for k, i in ix.items()}
            if not (st["bbox.xmin"].min <= BBOX[2] and st["bbox.xmax"].max >= BBOX[0] and
                    st["bbox.ymin"].min <= BBOX[3] and st["bbox.ymax"].max >= BBOX[1]):
                continue
            t = pf.read_row_group(r, columns=["names", "subtype", "class", "geometry", "region"])
            for i, n in enumerate(t.column("names").to_pylist()):
                nm = n["primary"] if n else None
                if nm in ("New York", "New York City", "City of New York"):
                    found.append((nm, t.column("subtype")[i].as_py(), t.column("class")[i].as_py(), t.column("region")[i].as_py(),
                                  shapely.from_wkb(t.column("geometry")[i].as_py())))
    for nm, s, c, reg, g in found:
        print(nm, s, c, reg, g.geom_type, round(g.area, 4), [round(b, 3) for b in g.bounds])
    land = max((g for nm, s, c, reg, g in found if s == "locality" and c == "land" and reg == "US-NY"), key=lambda g: g.area)
    st_land = [g for nm, s, c, reg, g in found if s == "region" and c == "land"][0]
    st_sea = [g for nm, s, c, reg, g in found if s == "region" and c == "maritime"][0]
    # the city's land, plus New York State water within ~2.5 km of it: keeps the bridges between the boroughs,
    # while New Jersey (another state) and Nassau / Westchester (land outside the city) stay out
    box = shapely.box(*BBOX)
    water = shapely.difference(shapely.intersection(st_sea, box), shapely.intersection(st_land, box))
    geom = shapely.union(land, shapely.intersection(water, land.buffer(0.03)))
    open(os.path.join(DATA, "nyc_boundary.wkt"), "w").write(geom.wkt)
    open(os.path.join(DATA, "nyc_land.wkt"), "w").write(land.wkt)
    print("saved", geom.bounds, round(geom.area, 4))


def build():
    t0 = time.time()
    geom = shapely.from_wkt(open(os.path.join(DATA, "nyc_boundary.wkt")).read())
    lon, lat = np.load(os.path.join(DATA, "usa_lon.npy")), np.load(os.path.join(DATA, "usa_lat.npy"))
    cand = np.nonzero((lon > BBOX[0]) & (lon < BBOX[2]) & (lat > BBOX[1]) & (lat < BBOX[3]))[0]
    shapely.prepare(geom)
    inside = np.zeros(len(lon), bool)
    inside[cand[shapely.contains_xy(geom, lon[cand].astype(np.float64), lat[cand].astype(np.float64))]] = True
    ea, eb = np.load(os.path.join(DATA, "usa_ea.npy")), np.load(os.path.join(DATA, "usa_eb.npy"))
    km = np.load(os.path.join(DATA, "usa_elen.npy"))
    k = np.nonzero(inside[ea] & inside[eb])[0]
    used, inv = np.unique(np.r_[ea[k], eb[k]], return_inverse=True)
    m = len(k)
    a, b, ln = inv[:m], inv[m:], km[k].astype(np.float64)
    n = len(used)
    comp = connected_components(coo_matrix((np.ones(m, np.int8), (a, b)), shape=(n, n)), directed=False)[1]
    big = np.bincount(comp[a], weights=ln).argmax()
    tot = float(ln.sum())
    keep = comp[a] == big
    used2, inv2 = np.unique(np.r_[a[keep], b[keep]], return_inverse=True)
    m2 = int(keep.sum())
    a2, b2 = inv2[:m2], inv2[m2:]
    nodes = used[used2]
    nlon, nlat = lon[nodes].astype(np.float64), lat[nodes].astype(np.float64)
    nx, ny = (nlon - LON0) * KX, (nlat - LAT0) * KY
    start = int(np.argmin(np.hypot(nx, ny)))
    np.savez(os.path.join(DATA, "nyc_graph.npz"), ea=a2, eb=b2, elen=ln[keep], nx=nx, ny=ny, lon=nlon, lat=nlat,
             start=start, usa_road=k[keep], total_km=tot)
    ip, ad, ei, al = CV.csr(len(nodes), a2, b2, ln[keep])
    blk = CV.block_ids(ip, ei, m2)
    print(f"NYC: {ln[keep].sum():,.1f} km in the main network ({ln[keep].sum() / tot:.1%} of {tot:,.1f} km inside), "
          f"{m2:,} segments, {len(np.unique(blk)):,} blocks, {len(nodes):,} points; start {nx[start]:.3f},{ny[start]:.3f} km "
          f"{time.time() - t0:.0f}s", flush=True)


def graph():
    z = np.load(os.path.join(DATA, "nyc_graph.npz"))
    m, n = len(z["ea"]), len(z["nx"])
    ip, ad, ei, al = CV.csr(n, z["ea"], z["eb"], z["elen"])
    return z, m, ip, ad, ei, al


def many(n_walks, kinds=(False,)):
    z, m, ip, ad, ei, al = graph()
    out = dict(network_km=float(z["elen"].sum()), segments=m)
    for uturn in kinds:
        s, lasts = [], []
        t0 = time.time()
        for seed in range(n_walks):
            steps, km, first, log, last = CV.cover_walk(ip, ad, ei, al, m, int(z["start"]), seed, uturn, 10_000_000, 50_000_000_000)
            assert np.isfinite(first).all()
            s.append(km / CV.C_KMS)
            lasts.append(int(last))
        key = "plain" if uturn else "nonback"
        out["seconds_" + key] = s
        out["last_" + key] = lasts
        q = np.percentile(s, [0, 10, 50, 90, 100])
        out["pct_" + key] = dict(zip(("min", "p10", "median", "p90", "max"), map(float, q)))
        print(key, out["pct_" + key], f"{time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(os.path.join(DATA, "nyc_cover_many.json"), "w"), indent=1)


def feature(seed):
    z, m, ip, ad, ei, al = graph()
    steps, km, first, log, last = CV.cover_walk(ip, ad, ei, al, m, int(z["start"]), seed, False, 100_000, 50_000_000_000)
    path = CV.walk_path(ip, ad, ei, al, m, int(z["start"]), seed, False, int(steps) + 10)
    assert len(path) == steps and abs(z["elen"][path].sum() - km) < 1e-3
    np.save(os.path.join(DATA, f"nyc_first_{seed}.npy"), first)
    np.save(os.path.join(DATA, f"nyc_log_{seed}.npy"), log)
    np.save(os.path.join(DATA, f"nyc_path_{seed}.npy"), path)
    e = int(last)
    lo = (z["lon"][z["ea"][e]] + z["lon"][z["eb"][e]]) / 2
    la = (z["lat"][z["ea"][e]] + z["lat"][z["eb"][e]]) / 2
    print(f"seed {seed}: {km:,.0f} km = {CV.fmt_time(km / CV.C_KMS)}, {steps:,} steps; last segment {e} at {la:.6f} {lo:.6f}, "
          f"{z['elen'][e] * 1000:.0f} m")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "boundary":
        boundary()
    elif cmd == "build":
        build()
    elif cmd == "many":
        many(int(sys.argv[2]))
    elif cmd == "feature":
        feature(int(sys.argv[2]))
