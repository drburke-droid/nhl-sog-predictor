"""Every river centreline on Earth at full resolution, plus every big lake, for building river systems.

Rivers (OpenStreetMap waterway=river lines): all vertices as int32 (degrees x 1e7, so pieces that share an
OSM node share exact coordinates), the primary and English names, the Wikidata id.
Lakes and reservoirs larger than LAKE_KM2: outline simplified to ~50 m, name.

    python fetch2.py   -> data/full/part-*.npz   (resumable; 6 files at a time)
"""
import os, sys, time
import numpy as np
import pyarrow.parquet as pq
import fsspec
import shapely
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fetch_overture import BUCKET  # noqa: E402
from fetch import list_files       # noqa: E402

OUT = os.path.join(HERE, "data", "full")
LAKE_KM2 = 5.0


def en_name(names):
    if not names:
        return ""
    for k, v in (names.get("common") or []):
        if k == "en":
            return v
    return ""


def process_file(key):
    tag = os.path.basename(key).split("-")[1]
    out = os.path.join(OUT, f"part-{tag}.npz")
    if os.path.exists(out):
        return out, 0, 0
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    for attempt in range(6):
        try:
            pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
            break
        except Exception:
            time.sleep(2 ** attempt)
    rn, ren, rwd, rxy, roff = [], [], [], [], [0]
    ln, lxy, loff, larea = [], [], [0], []
    for rg in range(pf.metadata.num_row_groups):
        t = None
        for attempt in range(6):
            try:
                cls = pf.read_row_group(rg, columns=["class"]).column(0).to_numpy(zero_copy_only=False)
                idx = np.nonzero(np.isin(cls, ["river", "lake", "reservoir"]))[0]
                if len(idx):
                    t = pf.read_row_group(rg, columns=["class", "names", "wikidata", "geometry"]).take(idx).to_pylist()
                break
            except Exception:
                time.sleep(2 ** attempt)
        if not t:
            continue
        for r in t:
            g = shapely.from_wkb(r["geometry"])
            if r["class"] == "river":
                if g.geom_type not in ("LineString", "MultiLineString"):
                    continue
                for part in (g.geoms if g.geom_type == "MultiLineString" else [g]):
                    c = shapely.get_coordinates(part)
                    if len(c) < 2:
                        continue
                    rxy.append(np.round(c * 1e7).astype(np.int32))
                    roff.append(roff[-1] + len(c))
                    rn.append((r["names"] or {}).get("primary") or "")
                    ren.append(en_name(r["names"]))
                    rwd.append(r["wikidata"] or "")
            elif g.geom_type in ("Polygon", "MultiPolygon"):
                lat = (g.bounds[1] + g.bounds[3]) / 2
                km2 = g.area * 111.32 ** 2 * np.cos(np.radians(lat))
                if km2 < LAKE_KM2:
                    continue
                for part in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                    ring = shapely.get_coordinates(shapely.simplify(part.exterior, 0.0005))
                    lxy.append(ring.astype(np.float32))
                    loff.append(loff[-1] + len(ring))
                    ln.append((r["names"] or {}).get("primary") or en_name(r["names"]) or "")
                    larea.append(km2)
    np.savez(out, name=np.array(rn, dtype=object), name_en=np.array(ren, dtype=object), wikidata=np.array(rwd, dtype=object),
             xy=np.concatenate(rxy) if rxy else np.zeros((0, 2), np.int32), off=np.array(roff, np.int64),
             lake_name=np.array(ln, dtype=object), lake_xy=np.concatenate(lxy) if lxy else np.zeros((0, 2), np.float32),
             lake_off=np.array(loff, np.int64), lake_km2=np.array(larea))
    return out, len(rn), len(ln)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    keys = list_files()
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for k, (out, nr, nl) in enumerate(ex.map(process_file, keys)):
            print(f"{k + 1}/{len(keys)} {os.path.basename(out)}: {nr} rivers, {nl} lakes  {time.time() - t0:.0f}s", flush=True)
