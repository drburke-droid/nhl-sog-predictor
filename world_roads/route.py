"""Greedy approximate coverage walk (a rural-postman heuristic) over the routable road graph.

Rules, applied at every step from the dot's current node:
 1. Eligible roads are unvisited edges whose sweep key <= (smallest unvisited key + BAND_KM).
 2. If an eligible edge touches the current node, drive it (prefer the straightest continuation).
 3. Otherwise, if the current connected component still has eligible edges, take the shortest
    road path (Dijkstra) to the nearest node with one. These deadhead kilometres are real repeated
    travel and are counted in the total distance (this is the "backtracking" overhead).
 4. Otherwise the component is exhausted for now: make a hypothetical instantaneous jump to the
    nearest eligible edge elsewhere. Jumps add zero distance and zero time.
The result is deterministic. It is NOT an optimal Chinese-postman tour.
"""
import os, json, time
import numpy as np
import numba as nb
import heapq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ALPHA = float(os.environ.get("ALPHA", 150))
BAND_KM = float(os.environ.get("BAND_KM", 1500))
MATCH_KM = float(os.environ.get("MATCH_KM", 300))


@nb.njit(cache=True)
def hierholzer(starts, ea, eb, key, nx, ny, indptr, inc, alpha):
    """Euler circuit of every component (graph must be even), started at `starts[c]`.
    Edge choice at a node: lowest sweep key + alpha * turn cost (keeps the circuit sweeping).
    Returns concatenated (node, arrival edge) sequences; arrival edge -1 marks a component start."""
    m = ea.shape[0]
    n = nx.shape[0]
    used = np.zeros(m, np.bool_)
    out_n = np.empty(m + starts.shape[0] + 1, np.int64)
    out_e = np.empty(m + starts.shape[0] + 1, np.int64)
    no = 0
    st_n = np.empty(m + 2, np.int64)
    st_e = np.empty(m + 2, np.int64)
    tmp_n = np.empty(m + 2, np.int64)
    tmp_e = np.empty(m + 2, np.int64)
    for s in starts:
        sp = 0
        st_n[0] = s; st_e[0] = -1; sp = 1
        nt = 0
        while sp > 0:
            v = st_n[sp - 1]
            ae = st_e[sp - 1]
            dxp = 0.0; dyp = 0.0
            if ae >= 0:
                u = eb[ae] if ea[ae] == v else ea[ae]
                dxp = nx[v] - nx[u]; dyp = ny[v] - ny[u]
            pn = np.sqrt(dxp * dxp + dyp * dyp)
            best = -1; bs = 1e300
            for j in range(indptr[v], indptr[v + 1]):
                e = inc[j]
                if used[e]:
                    continue
                o = eb[e] if ea[e] == v else ea[e]
                tc = 0.0
                if pn > 0:
                    dx = nx[o] - nx[v]; dy = ny[o] - ny[v]
                    tc = 1.0 - (dx * dxp + dy * dyp) / ((np.sqrt(dx * dx + dy * dy) + 1e-9) * pn)
                sc = key[e] + alpha * tc
                if sc < bs:
                    bs = sc; best = e
            if best >= 0:
                used[best] = True
                st_n[sp] = eb[best] if ea[best] == v else ea[best]
                st_e[sp] = best
                sp += 1
            else:
                tmp_n[nt] = v; tmp_e[nt] = ae; nt += 1
                sp -= 1
        # reverse popped order; reversed node i (i>0) is reached via the arrival edge of popped[K-1-i]
        for i in range(nt):
            out_n[no + i] = tmp_n[nt - 1 - i]
            out_e[no + i] = tmp_e[nt - 1 - i] if i > 0 else -1
        no += nt
    return out_n[:no], out_e[:no]


@nb.njit(cache=True)
def augment(n, ea, eb, elen, indptr, inc, maxd, odd):
    """Greedy T-join: pair each node flagged in `odd` with its nearest unpaired flagged node (shortest
    road path, within maxd km) and return the edges along those paths (to be traversed twice).
    `odd` is updated in place; nodes still flagged afterwards could not be paired."""
    dist = np.full(n, np.inf)
    pedge = np.full(n, -1, np.int64)
    touched = np.empty(n, np.int64)
    out = np.empty(ea.shape[0] * 2, np.int64)
    no = 0
    for s in range(n):
        if not odd[s]:
            continue
        nt = 0
        h = [(0.0, s)]
        dist[s] = 0.0; touched[nt] = s; nt += 1
        target = -1
        while len(h) > 0:
            d, u = heapq.heappop(h)
            if d > dist[u]:
                continue
            if u != s and odd[u]:
                target = u
                break
            if d > maxd:
                break
            for j in range(indptr[u], indptr[u + 1]):
                e = inc[j]
                v = eb[e] if ea[e] == u else ea[e]
                nd = d + elen[e]
                if nd < dist[v]:
                    if dist[v] == np.inf:
                        touched[nt] = v; nt += 1
                    dist[v] = nd; pedge[v] = e
                    heapq.heappush(h, (nd, v))
        if target >= 0:
            odd[s] = False; odd[target] = False
            v = target
            while v != s:
                e = pedge[v]
                if no < out.shape[0]:
                    out[no] = e; no += 1
                v = eb[e] if ea[e] == v else ea[e]
        for i in range(nt):
            dist[touched[i]] = np.inf; pedge[touched[i]] = -1
    return out[:no]

@nb.njit(cache=True)
def nearest_target(src, indptr, inc, ea, eb, elen, target):
    """Dijkstra from src to the closest node with target[node] >= 0; returns (node, path edges)."""
    n = indptr.shape[0] - 1
    dist = {}
    pedge = {}
    dist[src] = 0.0
    pedge[src] = -1
    h = [(0.0, src)]
    found = -1
    while len(h) > 0:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        if target[u] >= 0:
            found = u
            break
        for j in range(indptr[u], indptr[u + 1]):
            e = inc[j]
            v = eb[e] if ea[e] == u else ea[e]
            nd = d + elen[e]
            if v not in dist or nd < dist[v]:
                dist[v] = nd; pedge[v] = e
                heapq.heappush(h, (nd, v))
    path = np.empty(0, np.int64)
    if found >= 0:
        k = 0
        v = found
        while v != src:
            e = pedge[v]; k += 1
            v = eb[e] if ea[e] == v else ea[e]
        path = np.empty(k, np.int64)
        v = found
        while v != src:
            e = pedge[v]; k -= 1; path[k] = e
            v = eb[e] if ea[e] == v else ea[e]
    return found, path


def csr(n, ea, eb):
    mm = len(ea)
    ends = np.r_[ea, eb]; eid = np.r_[np.arange(mm), np.arange(mm)]
    o = np.argsort(ends, kind="stable")
    return (np.r_[0, np.cumsum(np.bincount(ends, minlength=n))].astype(np.int64), eid[o].astype(np.int64))


def euler_piece(eids, s_glob, ea, eb, elen, key, nx, ny):
    """Exact Euler path over the piece's edges (+ greedy T-join repeats) from s_glob to the
    piece's highest-key node. Returns (nodes, edges) in global ids; edge -1 = virtual jump."""
    nodes, inv = np.unique(np.r_[ea[eids], eb[eids]], return_inverse=True)
    k = len(eids)
    la, lb = inv[:k].astype(np.int64), inv[k:].astype(np.int64)
    n = len(nodes)
    s = int(np.searchsorted(nodes, s_glob))
    nk = np.full(n, -np.inf); np.maximum.at(nk, la, key[eids]); np.maximum.at(nk, lb, key[eids])
    t = int(np.argmax(nk))
    deg = np.bincount(np.r_[la, lb], minlength=n)
    odd = deg % 2 == 1
    if s != t:
        odd[s] ^= True; odd[t] ^= True
    ll = elen[eids]
    ip, ic = csr(n, la, lb)
    dup = augment(n, la, lb, ll, ip, ic, MATCH_KM, odd) if odd.any() else np.zeros(0, np.int64)
    left = np.nonzero(odd)[0]
    left = left[np.argsort(nk[left])]
    va, vb = left[0::2], left[1::2]
    A = np.r_[la, la[dup], va]; B = np.r_[lb, lb[dup], vb]
    K = np.r_[key[eids], key[eids][dup], nk[va]]
    G = np.r_[eids, eids[dup], -np.ones(len(va), np.int64)]
    ip, ic = csr(n, A, B)
    rn, re_ = hierholzer(np.array([s], np.int64), A, B, K, nx[nodes], ny[nodes], ip, ic, ALPHA)
    assert len(re_) == len(A) + 1
    ge = np.where(re_ >= 0, G[np.maximum(re_, 0)], -1)
    return nodes[rn], ge


def main():
    g = dict(np.load(os.path.join(DATA, "graph.npz")))
    ea, eb, elen, key, comp = g["ea"].astype(np.int64), g["eb"].astype(np.int64), g["elen"], g["key"], g["comp"]
    nx, ny = g["nx"], g["ny"]
    n = len(nx); m = len(ea)
    t0 = time.time()
    indptr, inc = csr(n, ea, eb)
    band = np.floor(key / BAND_KM).astype(np.int64)
    out_n, out_e = [], []
    cur = int(ea[np.argmin(key)])
    out_n.append(np.array([cur])); out_e.append(np.array([-2]))
    target = np.full(n, -1, np.int64)
    n_pieces = 0
    for b in np.unique(band):
        be = np.nonzero(band == b)[0]
        bn, binv = np.unique(np.r_[ea[be], eb[be]], return_inverse=True)
        kk = len(be)
        gph = coo_matrix((np.ones(kk), (binv[:kk], binv[kk:])), shape=(len(bn), len(bn)))
        lab = connected_components(gph, directed=False)[1]
        elab = lab[binv[:kk]]
        order = np.argsort(elab, kind="stable")
        cuts = np.r_[0, np.nonzero(np.diff(elab[order]))[0] + 1, kk]
        pieces = [be[order[a:c]] for a, c in zip(cuts[:-1], cuts[1:])]
        pending = set(range(len(pieces)))
        target[bn] = lab
        pcomp = np.array([comp[ea[p[0]]] for p in pieces])
        n_pieces += len(pieces)
        while pending:
            same = [p for p in pending if pcomp[p] == comp[cur]]
            if same:
                v, path = nearest_target(cur, indptr, inc, ea, eb, elen, target)
                assert v >= 0
                if len(path):
                    pn = np.empty(len(path), np.int64)
                    u = cur
                    for i, e in enumerate(path):
                        u = eb[e] if ea[e] == u else ea[e]; pn[i] = u
                    out_n.append(pn); out_e.append(path)
                p = int(target[v])
            else:
                cand = bn[np.isin(lab, list(pending))]
                d2 = (nx[cand] - nx[cur]) ** 2 + (ny[cand] - ny[cur]) ** 2
                v = int(cand[np.argmin(d2)])
                p = int(target[v])
                out_n.append(np.array([v])); out_e.append(np.array([-1]))
            rn, re_ = euler_piece(pieces[p], v, ea, eb, elen, key, nx, ny)
            out_n.append(rn[1:]); out_e.append(re_[1:])
            cur = int(rn[-1])
            pending.discard(p)
            target[bn[lab == p]] = -1
        target[bn] = -1
    rn = np.concatenate(out_n); re_ = np.concatenate(out_e)
    dt = time.time() - t0
    step = np.where(re_ >= 0, elen[np.maximum(re_, 0)], 0.0)
    cum = np.cumsum(step)
    jumps = np.nonzero(re_ == -1)[0]
    jd = np.hypot(nx[rn[jumps]] - nx[rn[jumps - 1]], ny[rn[jumps]] - ny[rn[jumps - 1]]) / 4
    seen = np.zeros(m, bool); seen[re_[re_ >= 0]] = True
    # continuity check: every driven step must use an edge touching the previous node
    e_ok = re_[1:] >= 0
    prev = rn[:-1][e_ok]; cur_ = rn[1:][e_ok]; ee = re_[1:][e_ok]
    assert (((ea[ee] == prev) & (eb[ee] == cur_)) | ((eb[ee] == prev) & (ea[ee] == cur_))).all()
    stats = {
        "band_km": BAND_KM, "match_km": MATCH_KM, "alpha": ALPHA, "pieces": int(n_pieces),
        "route_steps": int(len(rn)), "jumps": int(len(jumps)), "jumps_over_20px": int((jd > 20).sum()),
        "network_km": float(elen.sum()), "walk_km": float(cum[-1]),
        "repeat_overhead": float(cum[-1] / elen.sum() - 1), "edges_covered": float(seen.mean()),
        "seconds": dt,
    }
    print(json.dumps(stats, indent=1))
    np.savez(os.path.join(DATA, "route.npz"), node=rn, edge=re_, cum=cum)
    json.dump(stats, open(os.path.join(DATA, "route_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
