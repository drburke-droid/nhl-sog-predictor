"""Every lake and reservoir over 5 km2 (Overture base/water, classes `lake` and `reservoir`), with its class.

../rivers/fetch2.py kept every water polygon over 5 km2, which includes the areas OpenStreetMap draws for
wide rivers (class `river`). Those are river channels, not lakes: the fish follows a river's centre line
both ways, so they are left out here.

    python fetch_lakes.py   -> data/lakes/part-*.npz   (resumable; 6 files at a time)
"""
import os, sys, time
import numpy as np
import pyarrow.parquet as pq
import fsspec
import shapely
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rivers"))
from fetch_overture import BUCKET  # noqa: E402
from fetch import list_files       # noqa: E402

OUT = os.path.join(HERE, "data", "lakes")
LAKE_KM2 = 5.0
CLASSES = ["lake", "reservoir"]


def process_file(key):
    tag = os.path.basename(key).split("-")[1]
    out = os.path.join(OUT, f"part-{tag}.npz")
    if os.path.exists(out):
        return out, 0
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    for attempt in range(6):
        try:
            pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=8 << 20))
            break
        except Exception:
            time.sleep(2 ** attempt)
    ln, lc, lxy, loff, larea = [], [], [], [0], []
    for rg in range(pf.metadata.num_row_groups):
        t = None
        for attempt in range(6):
            try:
                cls = pf.read_row_group(rg, columns=["class"]).column(0).to_numpy(zero_copy_only=False)
                idx = np.nonzero(np.isin(cls, CLASSES))[0]
                if len(idx):
                    t = pf.read_row_group(rg, columns=["class", "names", "geometry"]).take(idx).to_pylist()
                break
            except Exception:
                time.sleep(2 ** attempt)
        if not t:
            continue
        for r in t:
            g = shapely.from_wkb(r["geometry"])
            if g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            lat = (g.bounds[1] + g.bounds[3]) / 2
            km2 = g.area * 111.32 ** 2 * np.cos(np.radians(lat))
            if km2 < LAKE_KM2:
                continue
            for part in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                ring = shapely.get_coordinates(shapely.simplify(part.exterior, 0.0005))
                lxy.append(ring.astype(np.float32))
                loff.append(loff[-1] + len(ring))
                ln.append((r["names"] or {}).get("primary") or "")
                lc.append(r["class"])
                larea.append(km2)
    np.savez(out, name=np.array(ln, dtype=object), cls=np.array(lc, dtype=object),
             xy=np.concatenate(lxy) if lxy else np.zeros((0, 2), np.float32), off=np.array(loff, np.int64),
             km2=np.array(larea))
    return out, len(ln)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    keys = list_files()
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for k, (out, n) in enumerate(ex.map(process_file, keys)):
            print(f"{k + 1}/{len(keys)} {os.path.basename(out)}: {n} lakes  {time.time() - t0:.0f}s", flush=True)
