"""The order the robot cleans New York, and when (in light time) each building is done.

It starts at the Empire State Building and spirals outward: rings 150 m wide around it, each ring taken in
order of bearing, alternating direction ring to ring. Within a building it cleans floor by floor from the
bottom, in rows 25 cm apart, so a building takes floor_area / 0.25 m / c. Between buildings it travels
straight from centroid to centroid (also at c); between floors it takes the stairs (counted as the storey
height).

    python nyc_route.py     -> data/nyc/route.npz, data/nyc/route_stats.json
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "nyc")
C_MS = 299792458.0
LANE = 0.25
RING = 150.0


def main():
    b = np.load(os.path.join(D, "buildings.npz"), allow_pickle=True)
    cx, cy, fa, fl = b["cx"], b["cy"], b["floor_area"], b["floors"]
    h = np.where(b["height"] > 1, b["height"], fl * 3.4)
    esb = int(np.nonzero(b["bin"] == "1015862")[0][0])
    dx, dy = cx - cx[esb], cy - cy[esb]
    r = np.hypot(dx, dy)
    ring = (r // RING).astype(np.int64)
    ang = np.arctan2(dy, dx)
    ang = np.where(ring % 2 == 0, ang, -ang)
    order = np.lexsort((ang, ring))
    order = np.concatenate([[esb], order[order != esb]])          # the Empire State Building first
    clean_m = fa[order] / LANE                                    # rows driven in each building
    stairs = h[order]                                             # up through every floor
    hop = np.concatenate([[0.0], np.hypot(np.diff(cx[order]), np.diff(cy[order]))])
    hop_down = np.concatenate([[0.0], h[order][:-1]])             # back down to the street
    travel = hop + hop_down + stairs
    t_start = np.cumsum(np.concatenate([[0.0], (clean_m + travel)[:-1]])) / C_MS + travel / C_MS
    t_end = t_start + clean_m / C_MS
    ts = np.empty(len(order)); te = np.empty(len(order))
    ts[order] = t_start; te[order] = t_end
    np.savez_compressed(os.path.join(D, "route.npz"), order=order, t_start=ts, t_end=te)
    st = dict(buildings=int(len(order)), floor_area_m2=float(fa.sum()), rows_km=float(clean_m.sum() / 1e3),
              travel_km=float(travel.sum() / 1e3), total_light_s=float(t_end[-1]), cleaning_light_s=float(clean_m.sum() / C_MS),
              esb_light_s=float(te[esb] - ts[esb]), esb_rows_km=float(fa[esb] / LANE / 1e3), esb_floor_area_m2=float(fa[esb]),
              last_building=int(order[-1]))
    for q in (0.25, 0.5, 0.75, 0.9):
        st[f"t_{int(q * 100)}pct_floor_s"] = float(np.interp(q, np.cumsum(fa[order]) / fa.sum(), t_end))
    json.dump(st, open(os.path.join(D, "route_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
