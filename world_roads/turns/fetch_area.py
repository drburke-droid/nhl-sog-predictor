"""Fetch Overture road segments for a bounding box (row groups pruned by bbox statistics)."""
import os, sys
import pyarrow as pa, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "calgary"))
sys.path.insert(0, os.path.dirname(HERE))
import fetch_city as fc
from fetch_overture import list_files


def fetch(bbox, out):
    fc.BBOX = bbox
    files = list_files()
    with ThreadPoolExecutor(32) as ex:
        hits = [h for h in ex.map(fc.rg_hits, files) if h[1]]
    print("row groups:", sum(len(h[1]) for h in hits), "in", len(hits), "files", flush=True)
    with ThreadPoolExecutor(6) as ex:
        tables = [t for res in ex.map(lambda h: fc.read(*h), hits) for t in res]
    tb = pa.concat_tables(tables)
    pq.write_table(tb, out)
    print("segments:", tb.num_rows)


if __name__ == "__main__":
    w, s, e, n = map(float, sys.argv[1:5])
    fetch((w, s, e, n), sys.argv[5])
