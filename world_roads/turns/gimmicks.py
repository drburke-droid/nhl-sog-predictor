"""Driving personalities for the lost photon, LA -> New York, on the same network as lost_photon.py.

Modes (p = probability of a random exit at an intersection, non-backtracking):
  compass  head for the exit pointing most directly at New York (no map, perfect sense of direction)
  signs    follow the road signs (the exit on the shortest remaining route); misread with prob p
  straight always the straightest exit; dead ends reflect (U-turn)
  frost    the road less traveled: the exit used least so far (ties: random)
  tron     never visit the same intersection twice; among free exits, the one pointing most at NYC
           (or random with prob p); dies when boxed in
"""
import os, sys, json, time
import numpy as np
import numba as nb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lost_photon as LP  # noqa: E402

MODES = {"compass": 0, "signs": 1, "straight": 2, "frost": 3, "tron": 4}


def prepare(G):
    lon, lat = G["lon"], G["lat"]
    kx = np.radians(1) * LP.R * np.cos(np.radians(lat)); ky = np.radians(1) * LP.R
    G["x"] = np.radians(lon) * LP.R * np.cos(np.radians(lat))   # only used for local differences
    # unit vector toward NYC at every node (local east/north)
    tx = (LP.NYC[0] - lon) * kx; ty = (LP.NYC[1] - lat) * ky
    nrm = np.hypot(tx, ty) + 1e-9
    G["tx"], G["ty"] = tx / nrm, ty / nrm
    G["kx"] = kx
    n = G["n"]
    A = coo_matrix((np.r_[G["elen"], G["elen"]], (np.r_[G["ea"], G["eb"]], np.r_[G["eb"], G["ea"]])),
                   shape=(n, n)).tocsr()
    G["dist"] = dijkstra(A, indices=np.nonzero(G["goal"])[0], min_only=True)
    return G


@nb.njit(cache=True)
def _dir(indptr, adj, lon, lat, kx, u, j, look):
    """east/north unit vector from u along half-edge j, looking `look` pieces through bends."""
    p, c = u, adj[j]
    for _ in range(look - 1):
        a, b = indptr[c], indptr[c + 1]
        if b - a != 2:
            break
        nx_ = adj[a] if adj[a] != p else adj[a + 1]
        p, c = c, nx_
    dx = (lon[c] - lon[u]) * kx[u]; dy = (lat[c] - lat[u]) * 111.19
    n = np.sqrt(dx * dx + dy * dy) + 1e-9
    return dx / n, dy / n


@nb.njit(cache=True)
def run(mode, p, indptr, adj, alen, goal, start, seed, max_steps, lon, lat, kx, tx, ty, dist, record):
    np.random.seed(seed)
    n = len(indptr) - 1
    prev, cur = -1, start
    km = 0.0
    seen_he = np.zeros(len(adj), np.bool_)
    used = np.zeros(len(adj), np.int32)
    visited = np.zeros(n if mode == 4 else 1, np.bool_)
    if mode == 4:
        visited[cur] = True
    hx, hy = 1.0, 0.0                                     # heading (east at the start)
    path = np.empty(max_steps + 1 if record else 1, np.int32)
    if record:
        path[0] = cur
    status = 0                                            # 0 running/gave up, 1 arrived, 2 trapped, 3 boxed in
    steps = 0
    for step in range(max_steps):
        steps = step
        if goal[cur]:
            status = 1
            break
        a, b = indptr[cur], indptr[cur + 1]
        # candidate exits (not back the way we came)
        best = -1
        n_ex = 0
        for j in range(a, b):
            if adj[j] != prev:
                n_ex += 1
        j_pick = -1
        if n_ex == 0:
            if mode == 4:
                status = 3
                break
            for j in range(a, b):
                if adj[j] == prev:
                    j_pick = j
        elif mode == 4:
            free = 0
            for j in range(a, b):
                if adj[j] != prev and not visited[adj[j]]:
                    free += 1
            if free == 0:
                status = 3
                break
            if np.random.random() < p:
                k = np.random.randint(free)
                for j in range(a, b):
                    if adj[j] != prev and not visited[adj[j]]:
                        if k == 0:
                            j_pick = j
                            break
                        k -= 1
            else:
                bs = -2.0
                for j in range(a, b):
                    if adj[j] != prev and not visited[adj[j]]:
                        dx, dy = _dir(indptr, adj, lon, lat, kx, cur, j, 3)
                        s = dx * tx[cur] + dy * ty[cur]
                        if s > bs:
                            bs = s; j_pick = j
        elif (mode == 0 or mode == 1) and n_ex >= 1 and np.random.random() < p:
            k = np.random.randint(n_ex)
            for j in range(a, b):
                if adj[j] != prev:
                    if k == 0:
                        j_pick = j
                        break
                    k -= 1
        elif mode == 0:
            bs = -2.0
            for j in range(a, b):
                if adj[j] != prev:
                    dx, dy = _dir(indptr, adj, lon, lat, kx, cur, j, 3)
                    s = dx * tx[cur] + dy * ty[cur]
                    if s > bs:
                        bs = s; j_pick = j
        elif mode == 1:
            bd = 1e18
            for j in range(a, b):
                d = alen[j] + dist[adj[j]]
                if d < bd:
                    bd = d; j_pick = j
        elif mode == 2:
            bs = -2.0
            for j in range(a, b):
                if adj[j] != prev:
                    dx, dy = _dir(indptr, adj, lon, lat, kx, cur, j, 1)
                    s = dx * hx + dy * hy
                    if s > bs:
                        bs = s; j_pick = j
        else:                                             # frost: least used exit, ties random
            mu = 1 << 30
            cnt = 0
            for j in range(a, b):
                if adj[j] != prev:
                    if used[j] < mu:
                        mu = used[j]; cnt = 1
                    elif used[j] == mu:
                        cnt += 1
            k = np.random.randint(cnt)
            for j in range(a, b):
                if adj[j] != prev and used[j] == mu:
                    if k == 0:
                        j_pick = j
                        break
                    k -= 1
        deterministic = mode == 2 or (mode == 0 and p == 0.0) or (mode == 1 and p == 0.0)
        if deterministic:
            if seen_he[j_pick]:
                status = 2
                break
            seen_he[j_pick] = True
        used[j_pick] += 1
        nxt = adj[j_pick]
        # heading for the straight rule: direction of the piece just driven
        dx = (lon[nxt] - lon[cur]) * kx[cur]; dy = (lat[nxt] - lat[cur]) * 111.19
        nn = np.sqrt(dx * dx + dy * dy)
        if nn > 0:
            hx, hy = dx / nn, dy / nn
        km += alen[j_pick]
        prev, cur = cur, nxt
        if mode == 4:
            visited[cur] = True
        if record:
            path[step + 1] = cur
        steps = step + 1
    if goal[cur]:
        status = 1
    return status, steps, km, cur, path[:steps + 1] if record else path


def go(G, mode, p, seed, max_steps=50_000_000, record=False):
    return run(MODES[mode], float(p), G["indptr"], G["adj"], G["alen"], G["goal"], G["start"], seed, max_steps,
               G["lon"], G["lat"], G["kx"], G["tx"], G["ty"], G["dist"], record)


def summarize(G, mode, p, seeds):
    res = []
    for s in seeds:
        st, steps, km, end, _ = go(G, mode, p, s)
        res.append((st, km, end))
    st = np.array([r[0] for r in res]); km = np.array([r[1] for r in res])
    ok = st == 1
    out = dict(mode=mode, p=p, n=len(res), arrived=int(ok.sum()), trapped=int((st == 2).sum()),
               boxed=int((st == 3).sum()), gave_up=int((st == 0).sum()))
    if ok.any():
        out.update(median_ms=float(np.median(km[ok]) / LP.C_KMS * 1000), mean_ms=float(km[ok].mean() / LP.C_KMS * 1000),
                   best_ms=float(km[ok].min() / LP.C_KMS * 1000), worst_ms=float(km[ok].max() / LP.C_KMS * 1000))
    if (~ok).any():
        ends = [r[2] for r in res if r[0] != 1]
        d_ny = LP.gc_km(G["lon"][ends], G["lat"][ends], *LP.NYC)
        d_la = LP.gc_km(G["lon"][ends], G["lat"][ends], *LP.LA)
        out.update(fail_median_km_driven=float(np.median(km[~ok])), fail_end_median_km_from_ny=float(np.median(d_ny)),
                   fail_end_median_km_from_la=float(np.median(d_la)))
    return out


if __name__ == "__main__":
    G = prepare(LP.load())
    t0 = time.time()
    rows = []
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    plan = [("straight", 0.0, [0]), ("signs", 0.0, [0])]
    plan += [("compass", p, range(N)) for p in (0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0)]
    plan += [("signs", p, range(N)) for p in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)]
    plan += [("frost", 0.0, range(min(N, 100)))]
    plan += [("tron", p, range(N)) for p in (0.0, 0.1, 0.3, 1.0)]
    for mode, p, seeds in plan:
        r = summarize(G, mode, p, list(seeds))
        rows.append(r)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(HERE, "data", "gimmicks.json"), "w"), indent=1)
