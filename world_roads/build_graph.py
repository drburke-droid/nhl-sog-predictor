"""Merge per-file Overture reductions into one routable road graph + density rasters.

Outputs data/graph.npz and data/density.npz and prints a provenance summary (data/stats.json).
"""
import glob, json, os
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import cv2
from proj import W, H, LON0, SCALE, CX, CY, to_px
from fetch_overture import CLASSES, GRAPH_CLASSES, DW, DH, GW, GH, R_EARTH
from waypoints import sweep_key

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MIN_LAT = -60.0           # Antarctic roads are omitted from the map and the route (documented)
GAP_PX = 2.5              # bridge class-filter gaps up to 2.5 px on the 4x grid (~13 km at the equator)


def inv_y(y):
    """Inverse of the Natural Earth projection's y(phi) by Newton iteration."""
    phi = y / 1.007226
    for _ in range(12):
        p2 = phi * phi; p4 = p2 * p2
        f = phi * (1.007226 + p2 * (0.015085 + p4 * (-0.044475 + 0.028874 * p2 - 0.005916 * p4))) - y
        df = 1.007226 + p2 * (3 * 0.015085 + p4 * (-7 * 0.044475 + 9 * 0.028874 * p2 - 11 * 0.005916 * p4))
        phi = phi - f / df
    return phi


def px4_to_lonlat(px, py):
    x = (px / 4 - CX) / SCALE
    y = (CY - py / 4) / SCALE
    phi = inv_y(y)
    p2 = phi * phi; p4 = p2 * p2
    lam = x / (0.8707 - 0.131979 * p2 + p4 * (-0.013791 + p4 * p2 * (0.003971 - 0.001529 * p2)))
    lon = (np.degrees(lam) + LON0 + 180) % 360 - 180
    return lon, np.degrees(phi)


def haversine(lon1, lat1, lon2, lat2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def land_mask(ss):
    """Rasterize Natural Earth 10m land at supersample ss (Antarctica excluded)."""
    gj = json.load(open(os.path.join(DATA, "ne", "ne_10m_land.geojson")))
    m = np.zeros((H * ss, W * ss), np.uint8)
    cut = LON0 - 180
    for f in gj["features"]:
        g = shapely.geometry.shape(f["geometry"])
        parts = [shapely.clip_by_rect(g, -180, MIN_LAT, cut, 90), shapely.clip_by_rect(g, cut, MIN_LAT, 180, 90)]
        for p in parts:
            for poly in getattr(p, "geoms", [p]):
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                rings = [poly.exterior] + list(poly.interiors)
                pts = []
                for r in rings:
                    c = np.asarray(r.coords)
                    # nudge points sitting exactly on the cut to the correct side
                    lon = c[:, 0].copy()
                    side = np.median(lon) < cut
                    lon[np.abs(lon - cut) < 1e-9] = cut - 1e-7 if side else cut + 1e-7
                    x, y = to_px(lon, c[:, 1], ss=ss)
                    pts.append(np.round(np.c_[x, y] * 8).astype(np.int32))
                cv2.fillPoly(m, pts[:1], 255, lineType=cv2.LINE_AA, shift=3)
                if len(pts) > 1:
                    cv2.fillPoly(m, pts[1:], 0, lineType=cv2.LINE_AA, shift=3)
    return m


def main():
    files = sorted(glob.glob(os.path.join(DATA, "overture", "part-*.npz")))
    print(len(files), "parts")
    len_km = 0
    dens = np.zeros((4, DH * DW), np.float32)
    E, C = [], []
    for f in files:
        d = np.load(f)
        len_km = len_km + d["len_km"]
        for g in range(4):
            dens[g][d[f"dens{g}_idx"]] += d[f"dens{g}_val"]
        E.append(d["edges"]); C.append(d["edge_cls"])
    e = np.concatenate(E); c = np.concatenate(C)
    o = np.lexsort((c, e)); e, c = e[o], c[o]
    first = np.r_[True, e[1:] != e[:-1]]; e, c = e[first], c[first]
    a, b = e // (GW * GH), e % (GW * GH)
    ax, ay, bx, by = a % GW, a // GW, b % GW, b // GW
    ylim = to_px(0, MIN_LAT, ss=4)[1]
    keep = (ay < ylim) & (by < ylim)
    a, b, c = a[keep], b[keep], c[keep]
    nodes, inv = np.unique(np.r_[a, b], return_inverse=True)
    ea, eb = inv[: len(a)], inv[len(a):]
    nx, ny = (nodes % GW).astype(np.float64), (nodes // GW).astype(np.float64)
    lon, lat = px4_to_lonlat(nx, ny)
    elen = haversine(lon[ea], lat[ea], lon[eb], lat[eb])
    ekind = np.zeros(len(ea), np.int8)  # 0 road, 1 gap bridge
    print(f"nodes {len(nodes):,} edges {len(ea):,}  graph length {elen.sum()/1e6:.2f} M km")

    # ---- bridge class-filter gaps (dead ends close to another component, over land only)
    n = len(nodes)
    def comps(ea, eb):
        g = coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n, n))
        return connected_components(g, directed=False)[1]
    comp = comps(ea, eb)
    print("components before bridging:", comp.max() + 1)
    land4 = land_mask(4)
    deg = np.bincount(np.r_[ea, eb], minlength=n)
    tree = cKDTree(np.c_[nx, ny])
    dead = np.nonzero(deg == 1)[0]
    pairs = tree.query_ball_point(np.c_[nx[dead], ny[dead]], r=GAP_PX)
    ba, bb = [], []
    for u, cand in zip(dead, pairs):
        best, bd = -1, 1e9
        for v in cand:
            if comp[v] != comp[u]:
                d2 = (nx[v] - nx[u]) ** 2 + (ny[v] - ny[u]) ** 2
                if d2 < bd:
                    best, bd = v, d2
        if best >= 0:
            ts = np.linspace(0, 1, 6)
            sx = np.clip(np.rint(nx[u] + (nx[best] - nx[u]) * ts), 0, GW - 1).astype(int)
            sy = np.clip(np.rint(ny[u] + (ny[best] - ny[u]) * ts), 0, GH - 1).astype(int)
            if (land4[sy, sx] > 128).all():
                ba.append(u); bb.append(best)
    ba, bb = np.array(ba, np.int64), np.array(bb, np.int64)
    if len(ba):
        lo, hi = np.minimum(ba, bb), np.maximum(ba, bb)
        k = np.unique(lo * n + hi); ba, bb = k // n, k % n
    ea = np.r_[ea, ba]; eb = np.r_[eb, bb]
    c = np.r_[c, np.full(len(ba), CLASSES.index("tertiary"), c.dtype)]
    elen = np.r_[elen, haversine(lon[ba], lat[ba], lon[bb], lat[bb])]
    ekind = np.r_[ekind, np.ones(len(ba), np.int8)]
    comp = comps(ea, eb)
    ncomp = comp.max() + 1
    clen = np.bincount(comp[ea], weights=elen, minlength=ncomp)
    print(f"bridged {len(ba):,} gaps ({elen[ekind==1].sum():,.0f} km); components now {ncomp:,}")
    big = np.sort(clen)[::-1]
    print("largest components (M km):", np.round(big[:8] / 1e6, 3), " share of top1/top10:",
          round(big[0] / clen.sum(), 3), round(big[:10].sum() / clen.sum(), 3))

    mlon = np.degrees(np.arctan2(np.sin(np.radians(lon[ea])) + np.sin(np.radians(lon[eb])),
                                 np.cos(np.radians(lon[ea])) + np.cos(np.radians(lon[eb]))))
    key = sweep_key(mlon, (lat[ea] + lat[eb]) / 2)
    np.savez(os.path.join(DATA, "graph.npz"), nx=nx, ny=ny, lon=lon, lat=lat, ea=ea, eb=eb, ecls=c,
             elen=elen, ekind=ekind, comp=comp, key=key)
    np.savez_compressed(os.path.join(DATA, "density.npz"), dens=dens.reshape(4, DH, DW))
    cv2.imwrite(os.path.join(DATA, "land2x.png"), land_mask(2))
    stats = {
        "overture_release": "2026-09-23.1",
        "length_km_by_class": {k: float(v) for k, v in zip(CLASSES + ["other"], len_km)},
        "graph_classes": sorted(GRAPH_CLASSES),
        "graph_nodes": int(n), "graph_edges": int(len(ea)),
        "graph_length_km": float(elen.sum()), "gap_bridges": int(len(ba)),
        "gap_bridge_km": float(elen[ekind == 1].sum()), "components": int(ncomp),
    }
    json.dump(stats, open(os.path.join(DATA, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
