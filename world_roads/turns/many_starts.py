"""Run the alternating-turn walk from many random starts; summarise loop lengths and how far walks get."""
import json, sys
import numpy as np
import turn_walk as T

lat0 = 34.0537
G = T.build("data/la.parquet", lat0)
BB = (-118.60, 33.80, -117.90, 34.30)
rng = np.random.default_rng(1)
inner = np.nonzero((G["lon"] > -118.45) & (G["lon"] < -118.05) & (G["lat"] > 33.92) & (G["lat"] < 34.18) &
                   (np.diff(G["indptr"]) >= 3))[0]
res = []
for s in rng.choice(inner, 300, replace=False):
    nb = T.neighbors(G, s)
    first = int(nb[rng.integers(len(nb))])
    path, turns, loop_at = T.walk(G, int(s), first, max_steps=400_000)
    p = np.array(path)
    lon, lat = G["lon"][p], G["lat"][p]
    edge = (lon.min() < BB[0] + 0.01) | (lon.max() > BB[2] - 0.01) | (lat.min() < BB[1] + 0.01) | (lat.max() > BB[3] - 0.01)
    kx = np.radians(1) * 6371.0088 * np.cos(np.radians(lat0)); ky = np.radians(1) * 6371.0088
    far = float(np.max(np.hypot((lon - lon[0]) * kx, (lat - lat[0]) * ky)))
    loop = T.km_of(G, path[loop_at:]) if loop_at is not None else None
    res.append(dict(loop_km=loop, far_km=far, hit_edge=bool(edge), lead_km=T.km_of(G, path[: (loop_at or 0) + 1])))
L = np.array([r["loop_km"] for r in res if r["loop_km"] is not None and not r["hit_edge"]])
F = np.array([r["far_km"] for r in res])
print(json.dumps({"runs": len(res), "hit_map_edge": sum(r["hit_edge"] for r in res),
                  "loop_km_median": float(np.median(L)), "loop_km_p10": float(np.percentile(L, 10)), "loop_km_p90": float(np.percentile(L, 90)),
                  "loop_km_min": float(L.min()), "loop_km_max": float(L.max()),
                  "farthest_from_start_km_median": float(np.median(F)), "farthest_km_max": float(F.max())}, indent=1))
