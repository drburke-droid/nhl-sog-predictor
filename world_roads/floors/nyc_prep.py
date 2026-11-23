"""New York City's buildings, for the X-ray view and the floor-area numbers.

Footprints: NYC Building Footprints (data.cityofnewyork.us 5zhs-2jue): 1,082,928 polygons with roof height
and ground elevation (feet). Floor area: PLUTO (64uk-42ks), the city's tax-lot table, whose bldgarea is
each lot's gross floor area (sq ft). Each footprint gets the floor count of its lot when the lot has one
building, else roof height / 3.4 m; a lot's floor area is shared among its footprints by footprint area x
floors.

    python nyc_prep.py      -> data/nyc/buildings.npz, data/nyc/stats.json
"""
import os, glob, json, csv, sys
import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "nyc")
FT = 0.3048
SQFT = FT * FT
LON0, LAT0 = -73.95, 40.70                    # local metric frame: x east, y north (m)
KX = 111320.0 * np.cos(np.radians(LAT0))
KY = 110574.0
STOREY = 3.4                                  # m per floor where the lot's floor count can't be used


def to_xy(lon, lat):
    return (lon - LON0) * KX, (lat - LAT0) * KY


def main():
    csv.field_size_limit(1 << 30)
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "fp_*.csv")), key=lambda p: int(p.split("_")[-1][:-4])):
        with open(f) as fh:
            rd = csv.DictReader(fh)
            rows += [r for r in rd]
    # mappluto_bbl (fetched separately, same order): the PLUTO lot a footprint belongs to, condos included
    maps = []
    for f in sorted(glob.glob(os.path.join(D, "map_*.csv")), key=lambda p: int(p.split("_")[-1][:-4])):
        with open(f) as fh:
            maps += [r for r in csv.DictReader(fh)]
    assert len(maps) == len(rows) and all(a["bin"] == b["bin"] for a, b in zip(rows, maps))
    for r, m in zip(rows, maps):
        r["mappluto_bbl"] = m["mappluto_bbl"] or m["base_bbl"]
    print(len(rows), "footprints")
    lots = {}
    with open(os.path.join(D, "pluto.csv")) as fh:
        for r in csv.DictReader(fh):
            try:
                bbl = str(int(float(r["bbl"])))
            except ValueError:
                continue
            f = lambda k: float(r[k]) if r[k] not in ("", None) else 0.0
            lots[bbl] = (f("bldgarea") * SQFT, f("numfloors"), f("numbldgs"), r["borough"], r["address"])
    print(len(lots), "lots")
    keep = [r for r in rows if r["feature_code"] in ("2100", "5110") and r["last_status_type"] != "Demolition"]
    geoms = shapely.from_wkt(np.array([r["the_geom"] for r in keep], dtype=object))
    # largest polygon of each multipolygon, exterior ring
    rings, off, area = [], [0], []
    for g in geoms:
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        c = np.asarray(g.exterior.coords)[:, :2]
        x, y = to_xy(c[:, 0], c[:, 1])
        rings.append(np.stack([x, y], 1).astype(np.float32))
        off.append(off[-1] + len(c))
    xy = np.concatenate(rings)
    off = np.array(off, np.int64)
    n = len(keep)
    # shoelace area in the metric frame
    x, y = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    cs = np.concatenate([[0], np.cumsum(cross)])
    a = np.abs(cs[off[1:] - 1] - cs[off[:-1]]) / 2
    hr = np.array([float(r["height_roof"] or 0) for r in keep]) * FT
    ge = np.array([float(r["ground_elevation"] or 0) for r in keep]) * FT
    bbl = np.array([r["mappluto_bbl"].split(".")[0] if r["mappluto_bbl"] else "" for r in keep])
    binn = np.array([r["bin"] for r in keep])
    garage = np.array([r["feature_code"] == "5110" for r in keep])
    # footprints per lot
    uniq, inv, cnt = np.unique(bbl, return_inverse=True, return_counts=True)
    nfp = cnt[inv]
    floors = np.maximum(1, np.round(hr / STOREY)).astype(np.int32)
    lot_fl = np.array([lots.get(b, (0, 0, 0, "", ""))[1] for b in bbl])
    use_lot = (nfp == 1) & (lot_fl >= 1) & ~garage
    floors[use_lot] = np.round(lot_fl[use_lot]).astype(np.int32)
    floors[garage] = 1
    # floor area: share each lot's PLUTO floor area by footprint area x floors
    w = a * floors
    wsum = np.bincount(inv, weights=w)
    lot_area = np.array([lots.get(b, (0,))[0] for b in uniq])
    fa = np.where(lot_area[inv] > 0, lot_area[inv] * w / np.maximum(wsum[inv], 1e-9), w)
    matched = lot_area[inv] > 0
    cx = np.array([r[:, 0].mean() for r in rings]); cy = np.array([r[:, 1].mean() for r in rings])
    np.savez_compressed(os.path.join(D, "buildings.npz"), xy=xy, off=off, area=a, height=hr, ground=ge, floors=floors,
                        floor_area=fa, bbl=bbl, bin=binn, garage=garage, cx=cx, cy=cy)
    pl_total = sum(v[0] for v in lots.values())
    boro = {}
    for b, v in lots.items():
        boro[v[3]] = boro.get(v[3], 0) + v[0]
    st = dict(footprints=n, pluto_lots=len(lots), pluto_floor_area_m2=pl_total, floor_area_from_footprints_m2=float(fa.sum()),
              share_of_footprints_matched=float(matched.mean()), floors_total=int(floors.sum()),
              floor_area_by_borough_m2=boro, footprint_area_m2=float(a.sum()))
    esb = np.nonzero(binn == "1015862")[0]
    if esb.size:
        i = esb[0]
        st["esb"] = dict(index=int(i), footprint_m2=float(a[i]), height_m=float(hr[i]), floors=int(floors[i]), floor_area_m2=float(fa[i]))
    json.dump(st, open(os.path.join(D, "stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
