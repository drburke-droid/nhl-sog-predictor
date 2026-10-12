"""Fetch every Overture road segment inside a small bounding box (row groups pruned by bbox stats)."""
import sys, os, re, urllib.request, json
import numpy as np, pyarrow.parquet as pq, pyarrow as pa, fsspec, shapely
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_overture import list_files, BUCKET

BBOX = (-114.33, 50.83, -113.85, 51.22)   # Calgary city limits, with a margin
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})


def rg_hits(key):
    pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=1 << 20))
    md = pf.metadata
    names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)]
    ix = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}
    hits = []
    for r in range(md.num_row_groups):
        g = md.row_group(r)
        st = {k: g.column(i).statistics for k, i in ix.items()}
        if st["bbox.xmin"].min <= BBOX[2] and st["bbox.xmax"].max >= BBOX[0] and \
           st["bbox.ymin"].min <= BBOX[3] and st["bbox.ymax"].max >= BBOX[1]:
            hits.append(r)
    return key, hits


def read(key, rgs):
    pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
    out = []
    for r in rgs:
        for attempt in range(6):
            try:
                t = pf.read_row_group(r, columns=["id", "subtype", "class", "geometry", "bbox"])
                break
            except Exception:
                if attempt == 5:
                    raise
                import time; time.sleep(2 ** attempt)
                pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
        b = t.column("bbox").combine_chunks()
        xmin = b.field("xmin").to_numpy(); xmax = b.field("xmax").to_numpy()
        ymin = b.field("ymin").to_numpy(); ymax = b.field("ymax").to_numpy()
        sub = t.column("subtype").to_numpy(zero_copy_only=False)
        m = (sub == "road") & (xmin <= BBOX[2]) & (xmax >= BBOX[0]) & (ymin <= BBOX[3]) & (ymax >= BBOX[1])
        if m.any():
            out.append(t.filter(pa.array(m)).select(["id", "class", "geometry"]))
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = list_files()
    with ThreadPoolExecutor(32) as ex:
        hits = [h for h in ex.map(rg_hits, files) if h[1]]
    print("row groups:", sum(len(h[1]) for h in hits), "in", len(hits), "files")
    with ThreadPoolExecutor(8) as ex:
        tables = [t for res in ex.map(lambda h: read(*h), hits) for t in res]
    tb = pa.concat_tables(tables)
    pq.write_table(tb, os.path.join(OUT, "calgary_segments.parquet"))
    print("segments:", tb.num_rows)
