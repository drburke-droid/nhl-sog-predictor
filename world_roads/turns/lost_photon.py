"""The lost photon: leave Los Angeles at the speed of light and flip a coin at every intersection.
How long until you reach New York?

Network: the highways and main roads (motorway ... tertiary) of the world road film, the connected
piece that contains Los Angeles (Alaska and Canada to Panama; the Darien Gap ends it). At every intersection the driver picks one of the exits
uniformly at random (never a U-turn unless it is a dead end); bends are simply followed.
"Arrived" = within ARRIVE_KM of New York City Hall.

    python lost_photon.py sim [n_walks]     # distribution of arrival times
    python lost_photon.py path <seed>       # record one full walk for the video
"""
import os, sys, json, time
import numpy as np
import numba as nb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(os.path.dirname(HERE), "data", "graph.npz")
DATA = os.path.join(HERE, "data")
C_KMS = 299_792.458
R = 6371.0088
LA = (-118.2427, 34.0537)                 # Los Angeles City Hall
NYC = (-74.0060, 40.7128)                 # New York City Hall
ARRIVE_KM = 15.0
# places counted as "drove through" (within PASS_KM of the centre)
PLACES = {"Calgary": (-114.0631, 51.0461), "Chicago": (-87.6298, 41.8781), "Miami": (-80.1918, 25.7617),
          "Mexico City": (-99.1332, 19.4326), "Seattle": (-122.3321, 47.6062), "Toronto": (-79.3832, 43.6532),
          "Las Vegas": (-115.1398, 36.1699), "Denver": (-104.9903, 39.7392), "Dallas": (-96.7970, 32.7767),
          "Anchorage": (-149.9003, 61.2181), "Los Angeles": LA, "Boston": (-71.0589, 42.3601),
          "Washington": (-77.0369, 38.9072), "Montreal": (-73.5673, 45.5017), "Vancouver": (-123.1207, 49.2827)}
PASS_KM = 25.0


def gc_km(lon1, lat1, lon2, lat2):
    a = np.sin(np.radians(lat2 - lat1) / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * \
        np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load():
    g = np.load(GRAPH)
    lon, lat, ea, eb, elen, ecls = g["lon"], g["lat"], g["ea"], g["eb"], g["elen"], g["ecls"]
    n = len(lon)
    m = len(ea)
    comp = connected_components(coo_matrix((np.ones(m), (ea, eb)), shape=(n, n)), directed=False)[1]
    s = int(np.argmin(gc_km(lon, lat, *LA)))
    keep = comp[ea] == comp[s]
    ea, eb, elen, ecls = ea[keep], eb[keep], elen[keep], ecls[keep]
    used, inv = np.unique(np.r_[ea, eb], return_inverse=True)
    m = len(ea)
    ea, eb = inv[:m], inv[m:]
    lon, lat = lon[used], lat[used]
    n = len(used)
    ends = np.r_[ea, eb]; nbr = np.r_[eb, ea]; ln = np.r_[elen, elen]
    o = np.argsort(ends, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(ends, minlength=n))].astype(np.int64)
    G = dict(lon=lon, lat=lat, indptr=indptr, adj=nbr[o].astype(np.int64), alen=ln[o], ea=ea, eb=eb, elen=elen,
             ecls=ecls, n=n)
    G["start"] = int(np.argmin(gc_km(lon, lat, *LA)))
    G["goal"] = (gc_km(lon, lat, *NYC) < ARRIVE_KM)
    return G


@nb.njit(cache=True)
def _walk(indptr, adj, alen, goal, start, seed, max_steps, place_of, n_places, record):
    """Non-backtracking random walk. Returns (steps, km, arrived, visits_per_place, path)."""
    np.random.seed(seed)
    prev, cur = -1, start
    km = 0.0
    visits = np.zeros(n_places, np.int64)
    last_place = -1
    path = np.empty(max_steps + 1 if record else 1, np.int32)
    if record:
        path[0] = cur
    for step in range(max_steps):
        if goal[cur]:
            return step, km, True, visits, path[:step + 1] if record else path
        a, b = indptr[cur], indptr[cur + 1]
        d = b - a
        # exits excluding the road we came in on
        n_ex = d
        if prev >= 0:
            for j in range(a, b):
                if adj[j] == prev:
                    n_ex -= 1
                    break
        if n_ex <= 0:                                  # dead end: U-turn
            j_pick = a
            for j in range(a, b):
                if adj[j] == prev:
                    j_pick = j
                    break
        else:
            k = np.random.randint(n_ex)
            j_pick = -1
            for j in range(a, b):
                if prev >= 0 and adj[j] == prev and n_ex < d:
                    continue
                if k == 0:
                    j_pick = j
                    break
                k -= 1
        nxt = adj[j_pick]
        km += alen[j_pick]
        prev, cur = cur, nxt
        if record:
            path[step + 1] = cur
        p = place_of[cur]
        if p >= 0 and p != last_place:
            visits[p] += 1
        last_place = p
    return max_steps, km, False, visits, path


def place_index(G):
    names = list(PLACES)
    place_of = np.full(G["n"], -1, np.int32)
    for i, nm in enumerate(names):
        place_of[gc_km(G["lon"], G["lat"], *PLACES[nm]) < PASS_KM] = i
    return names, place_of


def sim(n_walks=200, max_steps=2_000_000_000):
    G = load()
    names, place_of = place_index(G)
    print("nodes", G["n"], "edges", len(G["ea"]), "km", round(G["elen"].sum()), "goal nodes", G["goal"].sum(), flush=True)
    rows = []
    t0 = time.time()
    for seed in range(n_walks):
        steps, km, ok, visits, _ = _walk(G["indptr"], G["adj"], G["alen"], G["goal"], G["start"], seed,
                                         max_steps, place_of, len(names), False)
        rows.append(dict(seed=seed, steps=int(steps), km=float(km), arrived=bool(ok),
                         seconds=float(km / C_KMS), visits={nm: int(v) for nm, v in zip(names, visits)}))
        print(seed, ok, f"{km:,.0f} km", f"{km / C_KMS:.2f} s", f"{steps:,} steps", f"{time.time() - t0:.0f}s", flush=True)
    s = np.array([r["seconds"] for r in rows])
    summary = dict(n=len(rows), arrived=int(sum(r["arrived"] for r in rows)), mean_s=float(s.mean()),
                   median_s=float(np.median(s)), p10_s=float(np.percentile(s, 10)), p90_s=float(np.percentile(s, 90)),
                   min_s=float(s.min()), max_s=float(s.max()), network_km=float(G["elen"].sum()),
                   direct_light_ms=float(gc_km(*LA, *NYC) / C_KMS * 1000))
    print(json.dumps(summary, indent=1))
    os.makedirs(DATA, exist_ok=True)
    json.dump(dict(summary=summary, walks=rows), open(os.path.join(DATA, "lost_photon_sim.json"), "w"), indent=1)


def record(seed, max_steps=2_000_000_000):
    G = load()
    names, place_of = place_index(G)
    steps, km, ok, visits, path = _walk(G["indptr"], G["adj"], G["alen"], G["goal"], G["start"], seed,
                                        max_steps, place_of, len(names), True)
    print(seed, ok, f"{km:,.0f} km", f"{km / C_KMS:.3f} s", f"{steps:,} steps")
    np.save(os.path.join(DATA, f"lost_photon_path_{seed}.npy"), path)
    return path


if __name__ == "__main__":
    if sys.argv[1] == "sim":
        sim(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else:
        record(int(sys.argv[2]))
