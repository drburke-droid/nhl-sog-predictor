"""Stream the Overture Maps transportation/segment layer and reduce it to compact per-file products.

Never loads the global dataset: each parquet row group (~20k segments) is read with only the
`subtype`, `class` and `geometry` columns over HTTP range requests, reduced, and discarded.

Per input file the output npz holds
  * len_km[class]           : exact geodesic length (haversine on full-resolution vertices)
  * dens_<group>            : length (km) per pixel of a 2x-supersampled (3840x2160) canvas
  * edges, edge_cls         : graph edges for GRAPH_CLASSES, vertices snapped to a 4x canvas grid
                              (used for routing/rendering only; distance is measured above)
"""
import os, re, sys, time, urllib.request
import numpy as np
import pyarrow.parquet as pq
import fsspec
import shapely
from concurrent.futures import ProcessPoolExecutor
from proj import to_px, W, H

RELEASE = "2026-09-23.1"
BUCKET = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/"
PREFIX = f"release/{RELEASE}/theme=transportation/type=segment/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "overture")

CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
           "living_street", "service", "track", "pedestrian", "footway", "steps", "path", "cycleway",
           "bridleway", "unknown"]
CID = {c: i for i, c in enumerate(CLASSES)}
OTHER = len(CLASSES)  # any class string not listed above
GRAPH_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary"}
# density groups (2x canvas): 0 major (graph classes), 1 minor vehicular, 2 tracks/unknown, 3 non-motorized
GROUP = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 2])
DW, DH = W * 2, H * 2
GW, GH = W * 4, H * 4
R_EARTH = 6371.0088


def list_files():
    xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={PREFIX}").read().decode()
    return sorted(k for k in re.findall(r"<Key>([^<]*)</Key>", xml) if k.endswith(".parquet"))


def process_file(key):
    name = os.path.basename(key).split("-")[1]
    out = os.path.join(OUT, f"part-{name}.npz")
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    for attempt in range(5):
        try:
            pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
            break
        except Exception:
            time.sleep(2 ** attempt)
    len_km = np.zeros(len(CLASSES) + 1)
    dens = [np.zeros(DW * DH, np.float32) for _ in range(4)]
    edges, ecls = [], []
    for rg in range(pf.metadata.num_row_groups):
        for attempt in range(6):
            try:
                tb = pf.read_row_group(rg, columns=["subtype", "class", "geometry"])
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        sub = tb.column("subtype").to_numpy(zero_copy_only=False)
        keep = np.nonzero(sub == "road")[0]
        if keep.size == 0:
            continue
        cls_s = tb.column("class").to_numpy(zero_copy_only=False)[keep]
        cls = np.array([CID.get(c, OTHER) for c in cls_s], np.int16)
        geoms = shapely.from_wkb(tb.column("geometry").take(keep).to_numpy(zero_copy_only=False))
        coords, idx = shapely.get_coordinates(geoms, return_index=True)
        if len(coords) < 2:
            continue
        lon, lat = coords[:, 0], coords[:, 1]
        same = idx[1:] == idx[:-1]
        # exact geodesic length of every consecutive vertex pair within a feature
        la1, la2 = np.radians(lat[:-1]), np.radians(lat[1:])
        dlo = np.radians(lon[1:] - lon[:-1])
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
        seg = 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        seg = np.where(same, seg, 0.0)
        pcls = cls[idx[:-1]]
        len_km += np.bincount(pcls, weights=seg, minlength=len(CLASSES) + 1)
        # density: attribute each pair's length to its midpoint pixel on the 2x canvas
        mx, my = to_px((lon[:-1] + lon[1:]) / 2, (lat[:-1] + lat[1:]) / 2, ss=2)
        # pairs that straddle the antimeridian get a bogus midpoint; use the first vertex instead
        wrap = np.abs(lon[1:] - lon[:-1]) > 180
        if wrap.any():
            fx, fy = to_px(lon[:-1][wrap], lat[:-1][wrap], ss=2)
            mx[wrap], my[wrap] = fx, fy
        ix, iy = mx.astype(np.int64), my.astype(np.int64)
        ok = same & (ix >= 0) & (ix < DW) & (iy >= 0) & (iy < DH)
        pix = iy[ok] * DW + ix[ok]
        grp = GROUP[pcls[ok]]
        for g in range(4):
            m = grp == g
            if m.any():
                dens[g] += np.bincount(pix[m], weights=seg[ok][m], minlength=DW * DH).astype(np.float32)
        # graph edges on the 4x grid for major classes
        gmask = same & (pcls <= CID["tertiary"])
        if gmask.any():
            gx, gy = to_px(lon, lat, ss=4)
            qx = np.clip(np.rint(gx), 0, GW - 1).astype(np.int64)
            qy = np.clip(np.rint(gy), 0, GH - 1).astype(np.int64)
            node = qy * GW + qx
            a_, b_ = node[:-1][gmask], node[1:][gmask]
            dx = np.abs(qx[1:][gmask] - qx[:-1][gmask])
            good = (a_ != b_) & (dx < GW // 2)
            lo, hi = np.minimum(a_, b_)[good], np.maximum(a_, b_)[good]
            edges.append(lo * (GW * GH) + hi)
            ecls.append(pcls[gmask][good])
    if edges:
        e = np.concatenate(edges)
        c = np.concatenate(ecls)
        order = np.lexsort((c, e))
        e, c = e[order], c[order]
        first = np.r_[True, e[1:] != e[:-1]]  # keep the most important class per unique edge
        e, c = e[first], c[first]
    else:
        e, c = np.zeros(0, np.int64), np.zeros(0, np.int16)
    sparse = {}
    for g in range(4):
        nz = np.nonzero(dens[g])[0]
        sparse[f"dens{g}_idx"] = nz.astype(np.int32)
        sparse[f"dens{g}_val"] = dens[g][nz]
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, len_km=len_km, edges=e, edge_cls=c, **sparse)
    os.replace(tmp, out)
    return out, time.time() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = list_files()
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]
    nproc = int(os.environ.get("NPROC", 6))
    t0 = time.time()
    with ProcessPoolExecutor(nproc) as ex:
        for i, (o, dt) in enumerate(ex.map(process_file, files)):
            print(f"[{i+1}/{len(files)}] {os.path.basename(o)} {dt:.0f}s  elapsed {time.time()-t0:.0f}s", flush=True)
