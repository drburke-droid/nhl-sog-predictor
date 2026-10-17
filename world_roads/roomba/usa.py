"""The Roomba photon on every road in the contiguous United States.

Network: the full North American street network of turns/full_photon.py (Overture, motorway through
residential and living streets; no service roads, tracks or paths), cut to the lower 48 states (Natural
Earth 1:10m borders, rasterised at 0.005 deg), keeping the largest connected network. The photon starts at
the geographic centre of the contiguous US, near Lebanon, Kansas.

    python usa.py build            -> data/usa_*.npy (the graph, in junction order)
    python usa.py walk 4           -> 4 cover walks at once (one per core) -> data/usa_cover.json
"""
import os, sys, json, time
import numpy as np
import numba as nb
import cv2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cover as CV  # noqa: E402

DATA = os.path.join(HERE, "data")
NA_GRAPH = os.path.join(ROOT, "turns", "data", "na_full_graph.npz")
NE = os.path.join(ROOT, "data", "ne", "ne_10m_admin_0_countries.geojson")
CENTER = (-98.5795, 39.8283)             # geographic centre of the contiguous US (near Lebanon, Kansas)
BOX = (-125.5, 24.0, -66.0, 49.6)
RES = 0.005


def us_mask():
    import json as J
    gj = J.load(open(NE))
    f = [f for f in gj["features"] if f["properties"]["ADM0_A3"] == "USA"][0]
    polys = f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]]
    W = int((BOX[2] - BOX[0]) / RES) + 1
    H = int((BOX[3] - BOX[1]) / RES) + 1
    mask = np.zeros((H, W), np.uint8)
    n = 0
    for poly in polys:
        ring = np.array(poly[0])
        cx, cy = ring[:, 0].mean(), ring[:, 1].mean()
        if not (BOX[0] < cx < BOX[2] and BOX[1] < cy < BOX[3]):
            continue                                              # Alaska, Hawaii, territories
        pts = np.c_[(ring[:, 0] - BOX[0]) / RES, (BOX[3] - ring[:, 1]) / RES].round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        for hole in poly[1:]:
            h = np.array(hole)
            cv2.fillPoly(mask, [np.c_[(h[:, 0] - BOX[0]) / RES, (BOX[3] - h[:, 1]) / RES].round().astype(np.int32)], 0)
        n += 1
    print(f"US polygons in the lower-48 box: {n}", flush=True)
    return mask


def build():
    t0 = time.time()
    z = np.load(NA_GRAPH)
    indptr, adj, alen = z["indptr"], z["adj"], z["alen"]
    lon, lat = z["lon"], z["lat"]
    n = len(lon)
    mask = us_mask()
    c = ((lon - BOX[0]) / RES).round().astype(np.int64)
    r = ((BOX[3] - lat) / RES).round().astype(np.int64)
    ok = (c >= 0) & (c < mask.shape[1]) & (r >= 0) & (r < mask.shape[0])
    inside = np.zeros(n, bool)
    inside[ok] = mask[r[ok], c[ok]] > 0
    del c, r, ok
    print(f"junctions inside: {inside.sum():,} of {n:,}  {time.time() - t0:.0f}s", flush=True)
    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    keep = inside[src] & inside[adj]
    src, dst, ln = src[keep], adj[keep], alen[keep]
    del keep, adj, alen
    # pair the two half-edges of each road: sort by (low end, high end, length)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    q = np.round(ln * 1e4).astype(np.int64)
    o = np.lexsort((q, hi, lo))
    lo, hi, ln = lo[o][0::2], hi[o][0::2], ln[o][0::2].astype(np.float64)
    del src, dst, q, o
    print(f"roads: {len(lo):,}  {time.time() - t0:.0f}s", flush=True)
    comp = connected_components(coo_matrix((np.ones(len(lo), np.int8), (lo, hi)), shape=(n, n)), directed=False)[1]
    big = np.bincount(comp[lo], weights=ln).argmax()
    tot_km = float(ln.sum())
    k = comp[lo] == big
    lo, hi, ln = lo[k], hi[k], ln[k]
    used, inv = np.unique(np.r_[lo, hi], return_inverse=True)       # keeps the space-filling order
    m = len(lo)
    ea, eb = inv[:m].astype(np.int32), inv[m:].astype(np.int32)
    ip, ad, ei, al = CV.csr(len(used), ea, eb, ln)
    os.makedirs(DATA, exist_ok=True)
    for name, arr in (("indptr", ip), ("adj", ad), ("eid", ei), ("alen", al.astype(np.float32)), ("ea", ea), ("eb", eb),
                      ("elen", ln.astype(np.float32)), ("lon", lon[used]), ("lat", lat[used])):
        np.save(os.path.join(DATA, f"usa_{name}.npy"), arr)
    start = int(np.argmin((lon[used] - CENTER[0]) ** 2 * np.cos(np.radians(CENTER[1])) ** 2 + (lat[used] - CENTER[1]) ** 2))
    meta = dict(junctions=int(len(used)), roads=int(m), km=float(ln.sum()), km_all_inside=tot_km, start=start,
                start_lonlat=[float(lon[used][start]), float(lat[used][start])])
    json.dump(meta, open(os.path.join(DATA, "usa_meta.json"), "w"), indent=1)
    print(meta, f"{time.time() - t0:.0f}s", flush=True)


@nb.njit(parallel=True, cache=True)
def walks(indptr, adj, eid, alen, m, start, seeds, log_every, max_steps):
    k = len(seeds)
    steps = np.zeros(k, np.int64)
    kms = np.zeros(k)
    lasts = np.zeros(k, np.int64)
    firsts = np.zeros((k, m), np.float32)
    nlog = max_steps // log_every + 2
    logs = np.zeros((k, nlog, 4))
    nlogs = np.zeros(k, np.int64)
    for i in nb.prange(k):
        st, km, first, log, last = CV.cover_walk(indptr, adj, eid, alen, m, start, seeds[i], False, log_every, max_steps)
        steps[i] = st; kms[i] = km; lasts[i] = last
        firsts[i] = first.astype(np.float32)
        logs[i, :len(log)] = log
        nlogs[i] = len(log)
    return steps, kms, firsts, logs, nlogs, lasts


def walk(k):
    meta = json.load(open(os.path.join(DATA, "usa_meta.json")))
    L = {nm: np.load(os.path.join(DATA, f"usa_{nm}.npy")) for nm in ("indptr", "adj", "eid", "alen")}
    alen = L["alen"].astype(np.float64)
    seeds = np.arange(k, dtype=np.int64)
    print(f"USA: {meta['roads']:,} roads, {meta['km']:,.0f} km; {k} walks from {meta['start_lonlat']}", flush=True)
    t0 = time.time()
    steps, kms, firsts, logs, nlogs, lasts = walks(L["indptr"], L["adj"], L["eid"], alen, meta["roads"], meta["start"],
                                                   seeds, 20_000_000, 400_000_000_000)
    rows = []
    for i in range(k):
        done = bool(np.isfinite(firsts[i]).all())
        rows.append(dict(seed=int(seeds[i]), done=done, steps=int(steps[i]), km=float(kms[i]), seconds=float(kms[i] / CV.C_KMS),
                         x_network=float(kms[i] / meta["km"]), last=int(lasts[i])))
        np.save(os.path.join(DATA, f"usa_first_{i}.npy"), firsts[i])
        np.save(os.path.join(DATA, f"usa_log_{i}.npy"), logs[i, :nlogs[i]])
        print(f"seed {i}: done={done} {kms[i]:,.0f} km = {CV.fmt_time(kms[i] / CV.C_KMS)} at c, {steps[i]:,} steps", flush=True)
    json.dump(dict(meta=meta, walks=rows, wall_s=time.time() - t0), open(os.path.join(DATA, "usa_cover.json"), "w"), indent=1)
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        walk(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
