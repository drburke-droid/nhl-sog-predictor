"""Anti-GPS: drive from LA to New York doing the opposite of the GPS at every intersection.

At every intersection (3+ roads) the GPS recomputes the fastest route to NYC and suggests a road;
you take any OTHER road (never a U-turn, unless the GPS itself says to turn around, in which case
any road forward is "wrong"). Bends are followed; dead ends force a U-turn. Same network and goal as
lost_photon.py (major roads of North + Central America; within 15 km of NYC City Hall).

The rule defines a Markov chain on (previous node, current node) states. This script answers
  1. is NYC reachable at all (some sequence of wrong turns)?   -> graph reachability
  2. picking wrong turns at random, do you get there for sure? -> can every reachable state still
     reach NYC (no trap you can never leave)?
  3. how long would it take?                                    -> simulation + the shortest lucky route
"""
import os, sys, json, time
import numpy as np
import numba as nb
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lost_photon as LP  # noqa: E402
import gimmicks as GM  # noqa: E402


@nb.njit(cache=True)
def gps_pick(indptr, adj, alen, dist, cur):
    best, bj = 1e18, -1
    for j in range(indptr[cur], indptr[cur + 1]):
        d = alen[j] + dist[adj[j]]
        if d < best:
            best, bj = d, j
    return bj


@nb.njit(cache=True)
def options(indptr, adj, alen, dist, prev, cur, out):
    """Write the allowed half-edges out of `cur` (arrived from `prev`) into out; return count."""
    a, b = indptr[cur], indptr[cur + 1]
    n_fwd = 0
    for j in range(a, b):
        if adj[j] != prev:
            n_fwd += 1
    k = 0
    if n_fwd == 0:                                    # dead end: turn around
        for j in range(a, b):
            out[k] = j; k += 1
        return k
    if n_fwd == 1 and prev >= 0:                      # a bend: no suggestion, follow the road
        for j in range(a, b):
            if adj[j] != prev:
                out[k] = j; k += 1
        return k
    g = gps_pick(indptr, adj, alen, dist, cur)        # intersection (or the start): GPS speaks
    for j in range(a, b):
        if adj[j] != prev and j != g:
            out[k] = j; k += 1
    return k


@nb.njit(cache=True)
def build_transitions(indptr, adj, alen, dist, src_of):
    H = len(adj)
    rows = np.empty(H * 4, np.int64); cols = np.empty(H * 4, np.int64)
    m = 0
    buf = np.empty(64, np.int64)
    for h in range(H):
        k = options(indptr, adj, alen, dist, src_of[h], adj[h], buf)
        for i in range(k):
            if m >= len(rows):
                rows2 = np.empty(len(rows) * 2, np.int64); cols2 = np.empty(len(cols) * 2, np.int64)
                rows2[:m] = rows[:m]; cols2[:m] = cols[:m]; rows, cols = rows2, cols2
            rows[m] = h; cols[m] = buf[i]; m += 1
    return rows[:m], cols[:m]


@nb.njit(cache=True)
def simulate(indptr, adj, alen, dist, goal, start, seed, max_steps, record):
    np.random.seed(seed)
    buf = np.empty(64, np.int64)
    prev, cur = -1, start
    km = 0.0
    path = np.empty(max_steps + 1 if record else 1, np.int32)
    if record:
        path[0] = cur
    for step in range(max_steps):
        if goal[cur]:
            return True, step, km, cur, path[:step + 1] if record else path
        k = options(indptr, adj, alen, dist, prev, cur, buf)
        j = buf[np.random.randint(k)]
        km += alen[j]
        prev, cur = cur, adj[j]
        if record:
            path[step + 1] = cur
    return False, max_steps, km, cur, path


if __name__ == "__main__":
    t0 = time.time()
    G = GM.prepare(LP.load())
    indptr, adj, alen, dist, goal = G["indptr"], G["adj"], G["alen"], G["dist"], G["goal"]
    n, H = G["n"], len(adj)
    src_of = np.repeat(np.arange(n), np.diff(indptr)).astype(np.int64)
    rows, cols = build_transitions(indptr, adj, alen, dist, src_of)
    T = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(H, H))
    # start: from LA with no previous road -> any road except the GPS one
    buf = np.empty(64, np.int64)
    k = options(indptr, adj, alen, dist, -1, G["start"], buf)
    first = buf[:k]
    reach = np.zeros(H, bool)
    for h in first:
        reach[breadth_first_order(T, int(h), directed=True, return_predecessors=False)] = True
    goal_state = goal[adj]
    res = {"states": int(H), "reachable_states": int(reach.sum()), "goal_reachable": bool((reach & goal_state).any())}
    # which reachable states can still get to NYC? (reverse search from goal states)
    can = np.zeros(H, bool)
    gs = np.nonzero(goal_state)[0]
    TT = T.T.tocsr()
    sup = csr_matrix((np.ones(len(gs)), (np.zeros(len(gs), int), gs)), shape=(1, H))
    from scipy.sparse import vstack, hstack
    # super-source -> every goal state, BFS over reversed edges
    A = vstack([hstack([csr_matrix((1, 1)), sup]), hstack([csr_matrix((H, 1)), TT])]).tocsr()
    can[breadth_first_order(A, 0, directed=True, return_predecessors=False)[1:] - 1] = True
    trapped = reach & ~can
    res["reachable_but_doomed_states"] = int(trapped.sum())
    if trapped.any():
        tn = adj[np.nonzero(trapped)[0]]
        res["doomed_region_lonlat_sample"] = [[float(G["lon"][v]), float(G["lat"][v])] for v in tn[:: max(1, len(tn) // 8)][:8]]
    # luckiest possible wrong-turn route (shortest path through the state graph)
    W = csr_matrix((alen[cols], (rows, cols)), shape=(H, H))
    d0 = np.min([dijkstra(W, indices=int(h)) + alen[h] for h in first], axis=0)
    best = float(d0[gs].min()) if len(gs) else float("inf")
    res["luckiest_route_km"] = best
    res["luckiest_route_ms"] = best / LP.C_KMS * 1000
    res["gps_route_km"] = float(dist[G["start"]])
    print(json.dumps(res, indent=1), f"{time.time() - t0:.0f}s", flush=True)
    # random wrong turns
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = []
    for s in range(N):
        ok, steps, km, end, _ = simulate(indptr, adj, alen, dist, goal, G["start"], s, 200_000_000, False)
        out.append(dict(seed=s, arrived=bool(ok), km=float(km), seconds=float(km / LP.C_KMS),
                        end=[float(G["lon"][end]), float(G["lat"][end])],
                        end_km_from_nyc=float(LP.gc_km(G["lon"][end], G["lat"][end], *LP.NYC))))
        print(json.dumps(out[-1]), f"{time.time() - t0:.0f}s", flush=True)
    res["walks"] = out
    json.dump(res, open(os.path.join(HERE, "data", "anti_gps.json"), "w"), indent=1)
