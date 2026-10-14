"""Fetch Overture road segments around each city centre (row groups pruned by bbox statistics).

    python fetch_cities.py            # all cities; skips ones already on disk
The bbox statistics of every row group are read once and cached in data/rowgroups.npz.
"""
import os, sys, time
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq, fsspec
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fetch_overture import list_files, BUCKET  # noqa: E402
from cities import CITIES, RADIUS_KM, slug  # noqa: E402

DATA = os.path.join(HERE, "data")
fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
COLS = ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")


def rg_stats(key):
    for attempt in range(6):
        try:
            md = pq.ParquetFile(fs.open(BUCKET + key, block_size=1 << 20)).metadata
            break
        except Exception:
            time.sleep(2 ** attempt)
    names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)]
    ix = [names.index(c) for c in COLS]
    out = []
    for r in range(md.num_row_groups):
        g = md.row_group(r)
        s = [g.column(i).statistics for i in ix]
        out.append((r, s[0].min, s[1].max, s[2].min, s[3].max))
    return key, out


def rowgroups():
    path = os.path.join(DATA, "rowgroups.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return list(z["keys"]), z["tab"]
    files = list_files()
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(rg_stats, files))
    keys = [k for k, _ in res]
    tab = np.array([(fi, *row) for fi, (_, rows) in enumerate(res) for row in rows], np.float64)
    np.savez(path, keys=np.array(keys, dtype=object), tab=tab)
    return keys, tab


def bbox_of(lon, lat, r_km):
    dlat = r_km / 111.2
    dlon = r_km / (111.2 * np.cos(np.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def read(key, rgs, bb):
    pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
    out = []
    for r in rgs:
        for attempt in range(6):
            try:
                t = pf.read_row_group(int(r), columns=["id", "subtype", "class", "geometry", "bbox"])
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
                pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
        b = t.column("bbox").combine_chunks()
        xmin = b.field("xmin").to_numpy(); xmax = b.field("xmax").to_numpy()
        ymin = b.field("ymin").to_numpy(); ymax = b.field("ymax").to_numpy()
        sub = t.column("subtype").to_numpy(zero_copy_only=False)
        m = (sub == "road") & (xmin <= bb[2]) & (xmax >= bb[0]) & (ymin <= bb[3]) & (ymax >= bb[1])
        if m.any():
            out.append(t.filter(pa.array(m)).select(["id", "class", "geometry"]))
    return out


def fetch_city(name, center, keys, tab):
    out = os.path.join(DATA, f"{slug(name)}.parquet")
    if os.path.exists(out):
        return
    t0 = time.time()
    bb = bbox_of(*center, RADIUS_KM + 0.5)
    hit = tab[(tab[:, 2] <= bb[2]) & (tab[:, 3] >= bb[0]) & (tab[:, 4] <= bb[3]) & (tab[:, 5] >= bb[1])]
    jobs = {}
    for row in hit:
        jobs.setdefault(int(row[0]), []).append(int(row[1]))
    with ThreadPoolExecutor(8) as ex:
        tables = [t for res in ex.map(lambda kv: read(keys[kv[0]], kv[1], bb), jobs.items()) for t in res]
    tb = pa.concat_tables(tables)
    pq.write_table(tb, out)
    print(f"{name}: {len(hit)} row groups, {tb.num_rows:,} segments, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    keys, tab = rowgroups()
    print("row groups:", len(tab), flush=True)
    only = sys.argv[1:]
    for name, center in CITIES:
        if only and name not in only:
            continue
        fetch_city(name, center, keys, tab)
