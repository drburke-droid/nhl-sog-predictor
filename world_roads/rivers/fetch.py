"""Every river centreline on Earth (OpenStreetMap waterway=river, via Overture's base/water layer).

Each water file is streamed one row group at a time; only river lines are kept. Per line: the exact length
(haversine over every vertex), both end points (OSM draws waterways downstream, so start = upstream), the
name and Wikidata id, and a copy simplified to ~20 m for drawing.

    python fetch.py        -> data/parts/part-*.npz   (resumable; 6 files at a time)
"""
import os, re, sys, time, urllib.request
import numpy as np
import pyarrow.parquet as pq
import fsspec
import shapely
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fetch_overture import BUCKET, RELEASE  # noqa: E402

PREFIX = f"release/{RELEASE}/theme=base/type=water/"
OUT = os.path.join(HERE, "data", "parts")
R_EARTH = 6371.0088


def list_files():
    xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={PREFIX}").read().decode()
    return sorted(k for k in re.findall(r"<Key>([^<]*)</Key>", xml) if k.endswith(".parquet"))


def hav_km(lon, lat):
    p = np.radians(lat); l = np.radians(lon)
    a = np.sin(np.diff(p) / 2) ** 2 + np.cos(p[:-1]) * np.cos(p[1:]) * np.sin(np.diff(l) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def process_file(key):
    name = os.path.basename(key).split("-")[1]
    out = os.path.join(OUT, f"part-{name}.npz")
    if os.path.exists(out):
        return out, 0
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    for attempt in range(6):
        try:
            pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
            break
        except Exception:
            time.sleep(2 ** attempt)
    names, wd, lens, ends, coords, offs = [], [], [], [], [], [0]
    for rg in range(pf.metadata.num_row_groups):
        for attempt in range(6):
            try:
                cls = pf.read_row_group(rg, columns=["class"]).column(0).to_numpy(zero_copy_only=False)
                idx = np.nonzero(cls == "river")[0]
                if len(idx) == 0:
                    break
                t = pf.read_row_group(rg, columns=["names", "wikidata", "geometry"]).take(idx).to_pylist()
                break
            except Exception:
                time.sleep(2 ** attempt)
                t = None
        if len(idx) == 0 or t is None:
            continue
        for r in t:
            g = shapely.from_wkb(r["geometry"])
            if g.geom_type not in ("LineString", "MultiLineString"):
                continue                                      # riverbank areas: skip, we want centrelines
            nm = (r["names"] or {}).get("primary") or ""
            for part in (g.geoms if g.geom_type == "MultiLineString" else [g]):
                c = shapely.get_coordinates(part)
                if len(c) < 2:
                    continue
                lens.append(float(hav_km(c[:, 0], c[:, 1]).sum()))
                ends.append((c[0, 0], c[0, 1], c[-1, 0], c[-1, 1]))
                s = shapely.get_coordinates(shapely.simplify(part, 0.0002))
                coords.append(s.astype(np.float32))
                offs.append(offs[-1] + len(s))
                names.append(nm)
                wd.append(r["wikidata"] or "")
    np.savez(out, name=np.array(names, dtype=object), wikidata=np.array(wd, dtype=object), len_km=np.array(lens),
             ends=np.array(ends, dtype=np.float64).reshape(-1, 4),
             coords=np.concatenate(coords) if coords else np.zeros((0, 2), np.float32), offs=np.array(offs, np.int64))
    return out, len(lens)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    keys = list_files()
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(6) as ex:
        for out, n in ex.map(process_file, keys):
            done += 1
            print(f"{done}/{len(keys)} {os.path.basename(out)}: {n} river lines  {time.time() - t0:.0f}s", flush=True)
