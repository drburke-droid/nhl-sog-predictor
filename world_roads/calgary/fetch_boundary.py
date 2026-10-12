"""Fetch the City of Calgary boundary polygon from Overture divisions (division_area)."""
import os, re, urllib.request
import pyarrow.parquet as pq, pyarrow as pa, fsspec, shapely
from fetch_city import BBOX, OUT, fs
from fetch_overture import BUCKET, RELEASE

PREFIX = f"release/{RELEASE}/theme=divisions/type=division_area/"
xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={PREFIX}").read().decode()
keys = [k for k in re.findall(r"<Key>([^<]*)</Key>", xml) if k.endswith(".parquet")]
found = []
for key in keys:
    pf = pq.ParquetFile(fs.open(BUCKET + key, block_size=1 << 20))
    md = pf.metadata
    names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)]
    ix = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}
    for r in range(md.num_row_groups):
        g = md.row_group(r); st = {k: g.column(i).statistics for k, i in ix.items()}
        if not (st["bbox.xmin"].min <= BBOX[2] and st["bbox.xmax"].max >= BBOX[0] and
                st["bbox.ymin"].min <= BBOX[3] and st["bbox.ymax"].max >= BBOX[1]):
            continue
        t = pf.read_row_group(r, columns=["names", "subtype", "class", "geometry", "country"])
        nm = [n["primary"] if n else None for n in t.column("names").to_pylist()]
        for i, n in enumerate(nm):
            if n == "Calgary":
                found.append((t.column("subtype")[i].as_py(), t.column("class")[i].as_py(),
                              shapely.from_wkb(t.column("geometry")[i].as_py())))
for s, c, g in found:
    print(s, c, g.geom_type, round(g.area, 4), g.bounds)
loc = [g for s, c, g in found if s in ("locality", "county")]
geom = max(loc, key=lambda g: g.area)
open(os.path.join(OUT, "calgary_boundary.wkt"), "w").write(geom.wkt)
print("saved")
