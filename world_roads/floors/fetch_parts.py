"""Overture building parts (OpenStreetMap 3D: setbacks, podiums, spires) for Manhattan and nearby waterfronts.

The city's footprint data has one polygon and one roof height per building, so a tower with setbacks
(the Empire State Building) would render as a box. Where OpenStreetMap maps building parts, the X-ray
view draws those instead. Only row groups whose bbox overlaps the area are read, over HTTP.

    python fetch_parts.py      -> data/nyc/parts.npz
"""
import os, re, sys, time, urllib.request
import numpy as np
import pyarrow.parquet as pq
import fsspec
import shapely
from concurrent.futures import ThreadPoolExecutor

RELEASE = "2026-09-23.1"
BUCKET = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/"
BOX = (-74.03, 40.68, -73.90, 40.83)            # lon0, lat0, lon1, lat1: Manhattan, Downtown Brooklyn, LIC
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nyc", "parts.npz")


def list_files(prefix):
    keys, token = [], None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        xml = urllib.request.urlopen(url).read().decode()
        keys += re.findall(r"<Key>([^<]*\.parquet)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
        if not m:
            return sorted(keys)
        token = m.group(1)


def rg_overlaps(md, rg, names):
    st = {}
    r = md.row_group(rg)
    for c in range(r.num_columns):
        col = r.column(c)
        nm = col.path_in_schema
        if nm in names and col.statistics is not None and col.statistics.has_min_max:
            st[nm] = (col.statistics.min, col.statistics.max)
    try:
        xmin, xmax = st["bbox.xmin"][0], st["bbox.xmax"][1]
        ymin, ymax = st["bbox.ymin"][0], st["bbox.ymax"][1]
    except KeyError:
        return True
    return not (xmax < BOX[0] or xmin > BOX[2] or ymax < BOX[1] or ymin > BOX[3])


def one(key, cols):
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    for attempt in range(5):
        try:
            pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=4 << 20))
            break
        except Exception:
            time.sleep(2 ** attempt)
    md = pf.metadata
    names = {"bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax"}
    out = []
    for rg in range(md.num_row_groups):
        if not rg_overlaps(md, rg, names):
            continue
        tb = pf.read_row_group(rg, columns=cols)
        bb = tb.column("bbox").combine_chunks()
        x0 = bb.field("xmin").to_numpy(); x1 = bb.field("xmax").to_numpy()
        y0 = bb.field("ymin").to_numpy(); y1 = bb.field("ymax").to_numpy()
        k = np.nonzero((x1 >= BOX[0]) & (x0 <= BOX[2]) & (y1 >= BOX[1]) & (y0 <= BOX[3]))[0]
        if k.size:
            out.append(tb.take(k))
    return out


def main():
    prefix = f"release/{RELEASE}/theme=buildings/type=building_part/"
    keys = list_files(prefix)
    print(len(keys), "building_part files", flush=True)
    cols = ["id", "bbox", "geometry", "height", "min_height", "num_floors", "min_floor", "building_id"]
    tabs = []
    with ThreadPoolExecutor(16) as ex:
        for i, r in enumerate(ex.map(lambda k: one(k, cols), keys)):
            tabs += r
    import pyarrow as pa
    tb = pa.concat_tables(tabs)
    print(len(tb), "parts in the box", flush=True)
    geoms = shapely.from_wkb(tb.column("geometry").to_numpy(zero_copy_only=False))
    h = tb.column("height").to_numpy(zero_copy_only=False).astype(float)
    mh = tb.column("min_height").to_numpy(zero_copy_only=False).astype(float)
    nf = tb.column("num_floors").to_numpy(zero_copy_only=False).astype(float)
    mf = tb.column("min_floor").to_numpy(zero_copy_only=False).astype(float)
    # polygons -> exterior rings (largest polygon of a multipolygon)
    rings, off = [], [0]
    keep = []
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        if g.geom_type != "Polygon":
            continue
        c = np.asarray(g.exterior.coords)[:, :2]
        rings.append(c); off.append(off[-1] + len(c)); keep.append(i)
    keep = np.array(keep)
    np.savez_compressed(OUT, xy=np.concatenate(rings), off=np.array(off), height=h[keep], min_height=mh[keep],
                        num_floors=nf[keep], min_floor=mf[keep],
                        building_id=np.array(tb.column("building_id").to_numpy(zero_copy_only=False))[keep].astype(str))
    print("wrote", OUT, len(keep), "parts", flush=True)


if __name__ == "__main__":
    import urllib.parse
    main()
