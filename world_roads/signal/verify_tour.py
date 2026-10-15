"""Zoom-10 check of the most-countries chain (data/tour.json), as verify.py does for the loop:
  * every fire's summit must be there (zoom-10 max within 600 m >= search height - 100 m)
  * every sightline must clear zoom-10 terrain sampled every 150 m
  * every country's first fire must lie inside that country, at least BORDER_KM from its border
Failures go to data/tour_invalid.json / data/tour_banned.json for the next tour.py run.

    python verify_tour.py    -> exit code 1 if anything failed
"""
import os, sys, json, time
import numpy as np
import shapely
import verify as V
import world as WD

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BORDER_KM = 1.0


def main():
    os.makedirs(V.TILE_DIR, exist_ok=True)
    T = json.load(open(os.path.join(DATA, "tour.json")))
    path, lat, lon, h = T["path"], T["lat"], T["lon"], T["h_m"]
    inv_p, ban_p = os.path.join(DATA, "tour_invalid.json"), os.path.join(DATA, "tour_banned.json")
    inv = json.load(open(inv_p)) if os.path.exists(inv_p) else []
    ban = json.load(open(ban_p)) if os.path.exists(ban_p) else []
    bad_nodes, bad_edges = [], []
    t0 = time.time()
    hz = []
    names_c = json.load(open(os.path.join(DATA, "countries.json")))["names"]
    exempt = {names_c.index(n) for n in ("Vatican", "Monaco")}
    for i in range(len(path)):
        if T["cid"][i] in exempt:                    # tiny countries: the fire stays on its hand-picked spot
            m13 = float(V.elev(np.array([lat[i]]), np.array([lon[i]]), 13)[0])
            hz.append(h[i])
            if abs(m13 - h[i]) > 15:
                bad_nodes.append(path[i])
                print(f"  fire {i}: {names_c[T['cid'][i]]} at {h[i]:.0f} m but zoom 13 says {m13:.0f} m -> invalid", flush=True)
            continue
        m = V.summit_max(lat[i], lon[i])
        m12 = V.summit_hr(lat[i], lon[i])
        hz.append(m12)                               # the fire on the true (zoom-12) summit within 300 m
        if m < h[i] - 100 or m12 < h[i] - V.summit_tol(h[i]):
            bad_nodes.append(path[i])
            print(f"  fire {i}: search height {h[i]:.0f} m, zoom-10 max {m:.0f} m, zoom-12 {m12:.0f} m -> invalid", flush=True)
        if i % 100 == 0:
            print(f"  summits {i}/{len(path)} {len(V.CACHE)} tiles {time.time() - t0:.0f}s", flush=True)
    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            continue
        ok, m_m, at = V.hop_clear((lat[i - 1], lon[i - 1]), (lat[i], lon[i]), hz[i - 1] / 1000 + 0.010, hz[i] / 1000 + 0.010)
        if not ok:
            bad_edges.append(sorted([path[i - 1], path[i]]))
            print(f"  hop {i}: blocked by {-m_m:.0f} m at {at:.0f} km", flush=True)
        if i % 100 == 0:
            print(f"  hops {i}/{len(path)} {len(V.CACHE)} tiles {time.time() - t0:.0f}s", flush=True)
    # each country's first fire is really in it
    names, geoms, _ = WD.countries()
    seen = set()
    near = []
    for i in range(len(path)):
        c = T["counts_for"][i]
        if c < 0 or c in seen:
            continue
        seen.add(c)
        p = shapely.Point(lon[i], lat[i])
        d_km = geoms[c].exterior.distance(p) * 111.2 if geoms[c].geom_type == "Polygon" else \
            min(g.exterior.distance(p) for g in geoms[c].geoms) * 111.2
        if not geoms[c].contains(p) or d_km < BORDER_KM:
            near.append((names[c], i, round(d_km, 2)))
    print("first fires within", BORDER_KM, "km of a border:", near, flush=True)
    print(f"checked {len(path)} fires: {len(bad_nodes)} bad summits, {len(bad_edges)} blocked sightlines, "
          f"{len(V.CACHE)} tiles, {time.time() - t0:.0f}s", flush=True)
    json.dump(inv + bad_nodes, open(inv_p, "w"))
    json.dump(ban + bad_edges, open(ban_p, "w"))
    T["verified_heights_m"] = hz
    T["border_flags"] = near
    json.dump(T, open(os.path.join(DATA, "tour.json"), "w"), indent=1)
    sys.exit(1 if bad_nodes or bad_edges else 0)


if __name__ == "__main__":
    main()
