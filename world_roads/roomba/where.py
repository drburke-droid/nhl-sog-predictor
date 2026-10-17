"""Name the place at a point: the nearest named road segments and towns, from Overture (same release as the network).

Only row groups whose bounding-box statistics cover the point are read, over HTTP range requests.

    python where.py LAT LON [LAT LON ...]   -> data/where.json
"""
import os, re, sys, json, math, urllib.request
import numpy as np
import pyarrow.parquet as pq
import fsspec
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fetch_overture import BUCKET, RELEASE  # noqa: E402

FS = fsspec.filesystem("https", client_kwargs={"trust_env": True})


def files(theme, typ):
    pre = f"release/{RELEASE}/theme={theme}/type={typ}/"
    xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={pre}").read().decode()
    return sorted(k for k in re.findall(r"<Key>([^<]*)</Key>", xml) if k.endswith(".parquet"))


def row_groups(pf, lon, lat, pad):
    """Row groups whose bbox column statistics overlap the padded point."""
    names = pf.schema_arrow.names
    md = pf.metadata
    cols = {md.row_group(0).column(j).path_in_schema: j for j in range(md.num_columns)}
    jx0, jx1, jy0, jy1 = (cols["bbox.xmin"], cols["bbox.xmax"], cols["bbox.ymin"], cols["bbox.ymax"])
    out = []
    for r in range(md.num_row_groups):
        g = md.row_group(r)
        s = [g.column(j).statistics for j in (jx0, jx1, jy0, jy1)]
        if any(x is None or not x.has_min_max for x in s):
            continue
        if s[0].min <= lon + pad and s[1].max >= lon - pad and s[2].min <= lat + pad and s[3].max >= lat - pad:
            out.append(r)
    return out


def dist_m(lon, lat, lon2, lat2):
    k = math.cos(math.radians(lat))
    return np.hypot((lon2 - lon) * k, lat2 - lat) * 111_320.0


def scan(keys, pts, pad, columns, keep):
    """For each file whose footer overlaps any point, read the matching row groups and collect rows near the points."""
    found = [[] for _ in pts]
    for key in keys:
        pf = pq.ParquetFile(FS.open(BUCKET + key, block_size=1 << 20))
        for i, (lat, lon) in enumerate(pts):
            for r in row_groups(pf, lon, lat, pad):
                tb = pf.read_row_group(r, columns=columns).to_pylist()
                for row in tb:
                    b = row["bbox"]
                    if b["xmin"] <= lon + pad and b["xmax"] >= lon - pad and b["ymin"] <= lat + pad and b["ymax"] >= lat - pad:
                        found[i].append(keep(row, lat, lon))
    return found


def seg_row(row, lat, lon):
    g = shapely.from_wkb(row["geometry"])
    p = shapely.Point(lon, lat)
    q = shapely.ops.nearest_points(g, p)[0]
    nm = (row.get("names") or {}).get("primary")
    return dict(name=nm, cls=row.get("class"), m=float(dist_m(lon, lat, q.x, q.y)))


def place_row(row, lat, lon):
    g = shapely.from_wkb(row["geometry"]).representative_point()
    nm = (row.get("names") or {}).get("primary")
    return dict(name=nm, subtype=row.get("subtype"), region=row.get("region"), km=float(dist_m(lon, lat, g.x, g.y) / 1000))


def main(pts):
    import shapely.ops  # noqa: F401
    segs = scan(files("transportation", "segment"), pts, 0.004, ["names", "class", "geometry", "bbox"], seg_row)
    places = scan(files("divisions", "division"), pts, 0.25, ["names", "subtype", "region", "geometry", "bbox"], place_row)
    out = []
    for (lat, lon), s, p in zip(pts, segs, places):
        s = sorted((x for x in s if x["m"] < 400), key=lambda x: x["m"])[:8]
        p = sorted((x for x in p if x["subtype"] in ("locality", "localadmin", "county", "neighborhood")), key=lambda x: x["km"])[:8]
        out.append(dict(lat=lat, lon=lon, segments=s, places=p))
        print(json.dumps(out[-1], indent=1), flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "where.json"), "w"), indent=1)


if __name__ == "__main__":
    a = list(map(float, sys.argv[1:]))
    main(list(zip(a[0::2], a[1::2])))
