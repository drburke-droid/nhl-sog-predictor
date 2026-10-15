"""If an AI took over every phone, cable, satellite and radio, how long would it take to warn every
capital on Earth by fire? The first fire is lit at Bletchley Park; every fire that sees a lit fire lights
its own, so the warning spreads as a wave and branches wherever it can.

  1. Fewest relays from Bletchley Park to every capital: breadth-first search over the sightline graph
     (world_nodes + capitals.py beacons). Time = relays x 5 minutes; the last capital sets the clock.
  2. The fires actually needed: among all the fastest routes, a tree that shares trunks as much as possible
     (top-down greedy on the shortest-path DAG, using per-summit sets of the capitals below it).
  3. Every fire and sightline in the tree is verified (zoom 10 sightlines every 150 m; summits as in
     verify.summit_check). Failures are banned and the tree recomputed until clean.

    python capitals_tree.py    -> data/capitals_tree.json
"""
import os, sys, json, math, time
import numpy as np
import numba as nb
from scipy.sparse import csr_matrix
import tour as TR
import verify as V
import chain as CH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MIN_PER_RELAY = 5.0
INF = TR.INF


def load(extra_bans):
    nd = np.load(os.path.join(DATA, "world_nodes.npz"))
    e = np.load(os.path.join(DATA, "world_edges.npz"))
    u, v = e["u"].astype(np.int64), e["v"].astype(np.int64)
    n = len(nd["lat"])
    bad_nodes, bad_edges = set(), set()
    for f in ("tour_invalid.json", "cap_invalid.json"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            bad_nodes |= set(json.load(open(p)))
    for f in ("tour_banned.json", "cap_banned.json"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            bad_edges |= set(tuple(sorted(x)) for x in json.load(open(p)))
    keep = ~(np.isin(u, list(bad_nodes)) | np.isin(v, list(bad_nodes))) if bad_nodes else np.ones(len(u), bool)
    if bad_edges:
        key = u * n + v
        bk = np.array([a * n + b for a, b in bad_edges], np.int64)
        keep &= ~np.isin(key, bk)
    u, v = u[keep], v[keep]
    A = csr_matrix((np.ones(2 * len(u), np.int8), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    A.sum_duplicates()
    return nd, A.indptr.astype(np.int64), A.indices.astype(np.int64), len(bad_nodes), len(bad_edges)


@nb.njit(cache=True)
def capital_sets(indptr, indices, dist, order, own):
    """cs[v] = bitset of the capitals reachable from v going down the shortest-path DAG."""
    n, w = own.shape
    cs = own.copy()
    for k in range(len(order) - 1, -1, -1):               # deepest first
        v = order[k]
        dv = dist[v]
        if dv <= 0 or dv >= INF:
            continue
        for j in range(indptr[v], indptr[v + 1]):
            u = indices[j]
            if dist[u] == dv - 1:
                for b in range(w):
                    cs[u, b] |= cs[v, b]
    return cs


def popc(x):
    return np.array([bin(int(a)).count("1") for a in x.ravel()]).reshape(x.shape).sum(-1) if x.ndim > 1 else \
        sum(bin(int(a)).count("1") for a in x)


def build_tree(indptr, indices, dist, cs, root, cap_nodes):
    """Top-down: each active fire hands its capitals to as few children as possible."""
    w = cs.shape[1]
    parent = {root: -1}
    need = {root: cs[root].copy()}
    frontier = [root]
    capset = set(cap_nodes)
    while frontier:
        nxt = []
        for u in frontier:
            rem = need[u].copy()
            if u in capset:                                 # this capital is served by its own fire
                b = cap_nodes.index(u)
                rem[b // 64] &= ~np.uint64(1 << (b % 64))
            if not rem.any():
                continue
            ch = indices[indptr[u]:indptr[u + 1]]
            ch = ch[dist[ch] == dist[u] + 1]
            ch = np.unique(ch)
            cand = cs[ch]
            while rem.any():
                inter = cand & rem[None, :]
                score = popc(inter)
                k = int(np.argmax(score))
                if score[k] == 0:
                    break
                v = int(ch[k])
                give = inter[k].copy()
                if v in need:
                    need[v] |= give
                else:
                    need[v] = give
                    parent[v] = u
                    nxt.append(v)
                rem &= ~give
        frontier = nxt
    return parent


def solve():
    t0 = time.time()
    nd, indptr, indices, nb_, ne_ = load(None)
    caps = json.load(open(os.path.join(DATA, "capitals.json")))
    names = json.load(open(os.path.join(DATA, "countries.json")))["names"]
    root = json.load(open(os.path.join(DATA, "capitals_start.json")))["node"]
    d0 = np.full(len(nd["lat"]), INF, np.int64); d0[root] = 0
    dist, _ = TR.bfs_multi(indptr, indices, d0)
    reach = [c for c in caps if dist[c["node"]] < INF]
    unreach = [c for c in caps if dist[c["node"]] >= INF]
    cap_nodes = [c["node"] for c in reach]
    w = (len(cap_nodes) + 63) // 64
    own = np.zeros((len(dist), w), np.uint64)
    for b, v in enumerate(cap_nodes):
        own[v, b // 64] |= np.uint64(1 << (b % 64))
    order = np.argsort(dist, kind="stable")
    order = order[dist[order] < INF]
    cs = capital_sets(indptr, indices, dist, order, own)
    parent = build_tree(indptr, indices, dist, cs, root, cap_nodes)
    nodes = sorted(parent, key=lambda v: (dist[v], v))
    print(f"bans: {nb_} summits, {ne_} sightlines | {len(reach)} capitals reachable, {len(unreach)} not | "
          f"last at {max(dist[c] for c in cap_nodes)} relays | tree {len(nodes)} fires  {time.time() - t0:.0f}s", flush=True)
    return nd, nodes, parent, dist, reach, unreach, names


OK_PATH = os.path.join(DATA, "cap_verified.json")


def verify_tree(nd, nodes, parent, caps_nodes):
    """Zoom-10/12 check of every fire and sightline in the tree; returns (bad nodes, bad edges, heights).
    Fires and sightlines that already passed are remembered (cap_verified.json), so a round only checks new ones."""
    okd = json.load(open(OK_PATH)) if os.path.exists(OK_PATH) else {"nodes": {}, "edges": []}
    ok_n = {int(k): v for k, v in okd["nodes"].items()}
    ok_e = set(tuple(e) for e in okd["edges"])
    hz, bad_n, bad_e = {}, [], []
    n_new = 0
    for v in nodes:
        la, lo, h = float(nd["lat"][v]), float(nd["lon"][v]), float(nd["h"][v])
        if v in caps_nodes:                                   # capital beacons: their height is zoom-12 ground
            hz[v] = h
            continue
        if int(v) in ok_n:
            hz[v] = ok_n[int(v)]
            continue
        n_new += 1
        ok, h_ = V.summit_check(la, lo, h)
        hz[v] = h_
        if ok:
            ok_n[int(v)] = float(h_)
        else:
            bad_n.append(int(v))
    for v in nodes:
        u = parent[v]
        if u < 0:
            continue
        key = tuple(sorted([int(u), int(v)]))
        if key in ok_e:
            continue
        n_new += 1
        ok, m, at = V.hop_clear((float(nd["lat"][u]), float(nd["lon"][u])), (float(nd["lat"][v]), float(nd["lon"][v])),
                                hz[u] / 1000 + CH.PYRE_KM, hz[v] / 1000 + CH.PYRE_KM)
        if ok:
            ok_e.add(key)
        else:
            bad_e.append(list(key))
    json.dump({"nodes": {str(k): v for k, v in ok_n.items()}, "edges": sorted(ok_e)}, open(OK_PATH, "w"))
    print(f"  {n_new} new summits and sightlines checked", flush=True)
    return bad_n, bad_e, hz


def main():
    for rnd in range(1, 61):
        print(f"=== round {rnd}", flush=True)
        nd, nodes, parent, dist, reach, unreach, names = solve()
        capn = {c["node"] for c in reach}
        t0 = time.time()
        bad_n, bad_e, hz = verify_tree(nd, nodes, parent, capn)
        print(f"checked {len(nodes)} fires: {len(bad_n)} bad summits, {len(bad_e)} blocked sightlines "
              f"{time.time() - t0:.0f}s", flush=True)
        if not bad_n and not bad_e:
            break
        for f, add in (("cap_invalid.json", bad_n), ("cap_banned.json", bad_e)):
            p = os.path.join(DATA, f)
            old = json.load(open(p)) if os.path.exists(p) else []
            json.dump(old + add, open(p, "w"))
    idx = {v: i for i, v in enumerate(nodes)}
    st = json.load(open(os.path.join(DATA, "capitals_start.json")))
    out = dict(start=st["name"], min_per_relay=MIN_PER_RELAY,
               fires=[dict(lat=float(nd["lat"][v]), lon=float(nd["lon"][v]), h_m=float(hz[v]), hop=int(dist[v]),
                           parent=idx[parent[v]] if parent[v] >= 0 else -1,
                           cid=int(nd["cid"][v])) for v in nodes],
               capitals=[dict(country=c["country"], capital=c["capital"], fire=idx[c["node"]], hop=int(dist[c["node"]]),
                              hours=round(int(dist[c["node"]]) * MIN_PER_RELAY / 60, 2), lat=c["lat"], lon=c["lon"])
                         for c in sorted(reach, key=lambda c: dist[c["node"]])],
               unreachable=[dict(country=c["country"], capital=c["capital"], lat=c["lat"], lon=c["lon"]) for c in unreach],
               verified=not bad_n and not bad_e)
    json.dump(out, open(os.path.join(DATA, "capitals_tree.json"), "w"), indent=1)
    last = out["capitals"][-1]
    print(f"{len(out['capitals'])} capitals in {last['hop']} relays = {last['hop'] * MIN_PER_RELAY / 60:.1f} h "
          f"(last: {last['capital']}, {last['country']}); {len(nodes)} fires; verified={out['verified']}")


if __name__ == "__main__":
    main()
