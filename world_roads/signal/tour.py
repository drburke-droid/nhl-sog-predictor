"""The most countries one chain of signal fires can pass through.

Graph: every candidate summit in the world (world.py) and every fire-to-fire sightline (edges.py).
  1. The largest connected piece of the world, and the countries in it (a fire must stand in a country
     for it to count: 193 UN members + Vatican + Palestine).
  2. An order for visiting them: countries are split into ~6-degree regions, the hop distance between
     regions comes from multi-source BFS, and a region tour (one region per country) is improved with
     2-opt / or-opt / region swaps.
  3. The fewest fires that visit the countries in that order: a layered shortest path (each layer one
     country, any summit in it), so travel inside big countries is counted exactly.
  4. Hops rejected by the zoom-10 check (verify_tour.py) are banned and the path is recomputed.

    python tour.py     -> data/tour.json
"""
import os, sys, json, math, time
import numpy as np
import numba as nb
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REG_DEG = 6.0
INF = 1 << 29


def load_graph():
    nd = np.load(os.path.join(DATA, "world_nodes.npz"))
    e = np.load(os.path.join(DATA, "world_edges.npz"))
    u, v = e["u"].astype(np.int64), e["v"].astype(np.int64)
    n = len(nd["lat"])
    ban_p = os.path.join(DATA, "tour_banned.json")
    if os.path.exists(ban_p):
        ban = set(tuple(sorted(x)) for x in json.load(open(ban_p)))
        keep = np.array([(a, b) not in ban for a, b in zip(u.tolist(), v.tolist())]) if ban else np.ones(len(u), bool)
        u, v = u[keep], v[keep]
        print(f"banned sightlines: {len(ban)}", flush=True)
    inv_p = os.path.join(DATA, "tour_invalid.json")
    if os.path.exists(inv_p):
        bad = np.array(json.load(open(inv_p)), np.int64)
        if len(bad):
            keep = ~(np.isin(u, bad) | np.isin(v, bad))
            u, v = u[keep], v[keep]
            print(f"invalid summits: {len(bad)}", flush=True)
    A = csr_matrix((np.ones(2 * len(u), np.int8), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    return nd, A.indptr.astype(np.int64), A.indices.astype(np.int64), A


@nb.njit(cache=True)
def bfs_multi(indptr, indices, d0):
    """Fewest hops from sources with integer start costs d0 (INF = not a source), level by level."""
    n = len(d0)
    dist = d0.copy()
    par = np.full(n, -1, np.int64)
    order = np.argsort(d0)
    nsrc = 0
    while nsrc < n and d0[order[nsrc]] < INF:
        nsrc += 1
    if nsrc == 0:
        return dist, par
    fr = np.empty(n, np.int64); nx = np.empty(n, np.int64)
    nf = 0; ptr = 0
    cur = d0[order[0]]
    while True:
        while ptr < nsrc and d0[order[ptr]] == cur:
            s = order[ptr]; ptr += 1
            if dist[s] == cur:
                fr[nf] = s; nf += 1
        if nf == 0:
            if ptr < nsrc:
                cur = d0[order[ptr]]
                continue
            break
        nn = 0
        for a in range(nf):
            x = fr[a]
            if dist[x] != cur:
                continue
            for j in range(indptr[x], indptr[x + 1]):
                y = indices[j]
                if dist[y] > cur + 1:
                    dist[y] = cur + 1
                    par[y] = x
                    nx[nn] = y; nn += 1
        fr, nx = nx, fr
        nf = nn
        cur += 1
    return dist, par


def main():
    t0 = time.time()
    nd, indptr, indices, A = load_graph()
    lat, lon = nd["lat"], nd["lon"]
    cid = np.where(nd["border_ok"], nd["cid"], -1)              # only fires >= 1 km inside count (border.py)
    cj = json.load(open(os.path.join(DATA, "countries.json")))
    names, counts = cj["names"], np.array(cj["counts"])
    n = len(lat)
    ncomp, lab = connected_components(A, directed=False)
    is_c = np.zeros(n, bool)
    is_c[cid >= 0] = counts[cid[cid >= 0]]
    # the component with the most countries
    best, best_set = -1, set()
    for c in np.unique(lab[is_c]):
        cs = set(cid[(lab == c) & is_c].tolist())
        if len(cs) > len(best_set):
            best, best_set = c, cs
    comp = lab == best
    T = sorted(best_set)
    print(f"{ncomp:,} components; the largest by countries holds {comp.sum():,} summits and {len(T)} countries "
          f"{time.time() - t0:.0f}s", flush=True)
    missing = sorted(names[g] for g in np.nonzero(counts)[0] if g not in best_set)
    print("unreachable:", ", ".join(missing), flush=True)
    # regions: country x 6-degree cell (inside the component)
    m = comp & is_c
    idx = np.nonzero(m)[0]
    rk = cid[idx].astype(np.int64) * 100000 + np.floor((lat[idx] + 90) / REG_DEG).astype(np.int64) * 100 + \
        np.floor((lon[idx] + 180) / REG_DEG).astype(np.int64)
    ukeys, rid = np.unique(rk, return_inverse=True)
    reg_country = (ukeys // 100000).astype(np.int64)
    nr = len(ukeys)
    print(f"regions: {nr}", flush=True)
    # region-to-region hop distances (multi-source BFS from each region)
    D = np.full((nr, nr), INF, np.int64)
    node_reg = np.full(n, -1, np.int64); node_reg[idx] = rid
    order = np.argsort(node_reg[idx]); idx_s = idx[order]; rid_s = rid[order]
    starts = np.searchsorted(rid_s, np.arange(nr + 1))
    for r in range(nr):
        d0 = np.full(n, INF, np.int64)
        d0[idx_s[starts[r]:starts[r + 1]]] = 0
        dist, _ = bfs_multi(indptr, indices, d0)
        D[r] = np.minimum.reduceat(dist[idx_s], starts[:-1])
        if r % 50 == 0:
            print(f"  BFS {r}/{nr} {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(DATA, "tour_D.npy"), D)
    json.dump({"T": [int(x) for x in T], "reg_country": reg_country.tolist(), "ukeys": ukeys.tolist()},
              open(os.path.join(DATA, "tour_regions.json"), "w"))
    print(f"distance matrix done {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ the order (one region per country)
@nb.njit(cache=True)
def path_cost(D, seq):
    c = 0
    for i in range(len(seq) - 1):
        c += D[seq[i], seq[i + 1]]
    return c


@nb.njit(cache=True)
def improve(D, seq, regs_of, reg_ptr, seq_c):
    """2-opt, or-opt (segments of 1-3) and region swaps until no move helps. seq: regions, seq_c: their countries."""
    n = len(seq)
    better = True
    while better:
        better = False
        # region swap
        for i in range(n):
            c = seq_c[i]
            best = 0; br = seq[i]
            for j in range(reg_ptr[c], reg_ptr[c + 1]):
                r = regs_of[j]
                d = 0
                if i > 0:
                    d += D[seq[i - 1], r] - D[seq[i - 1], seq[i]]
                if i < n - 1:
                    d += D[r, seq[i + 1]] - D[seq[i], seq[i + 1]]
                if d < best:
                    best = d; br = r
            if best < 0:
                seq[i] = br
                better = True
        # 2-opt (open path: reversing a segment that touches an end is allowed)
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                a0 = D[seq[i - 1], seq[i]] if i > 0 else 0
                b0 = D[seq[j], seq[j + 1]] if j < n - 1 else 0
                a1 = D[seq[i - 1], seq[j]] if i > 0 else 0
                b1 = D[seq[i], seq[j + 1]] if j < n - 1 else 0
                if a1 + b1 < a0 + b0:
                    seq[i:j + 1] = seq[i:j + 1][::-1].copy()
                    seq_c[i:j + 1] = seq_c[i:j + 1][::-1].copy()
                    better = True
        # or-opt: move a segment of length L to another place (either orientation)
        for L in range(1, 4):
            i = 0
            while i + L <= n:
                seg = seq[i:i + L].copy(); segc = seq_c[i:i + L].copy()
                rem = np.concatenate((seq[:i], seq[i + L:])); remc = np.concatenate((seq_c[:i], seq_c[i + L:]))
                base = path_cost(D, seq)
                cut = 0
                if i > 0:
                    cut += D[seq[i - 1], seq[i]]
                if i + L < n:
                    cut += D[seq[i + L - 1], seq[i + L]]
                if i > 0 and i + L < n:
                    cut -= D[seq[i - 1], seq[i + L]]
                best = 0; bp = -1; brev = False
                m = len(rem)
                for pos in range(m + 1):
                    for rev in (False, True):
                        f = seg[-1] if rev else seg[0]
                        l = seg[0] if rev else seg[-1]
                        add = 0
                        if pos > 0:
                            add += D[rem[pos - 1], f]
                        if pos < m:
                            add += D[l, rem[pos]]
                        if pos > 0 and pos < m:
                            add -= D[rem[pos - 1], rem[pos]]
                        if add - cut < best:
                            best = add - cut; bp = pos; brev = rev
                if bp >= 0:
                    if brev:
                        seg = seg[::-1].copy(); segc = segc[::-1].copy()
                    seq[:] = np.concatenate((rem[:bp], seg, rem[bp:]))
                    seq_c[:] = np.concatenate((remc[:bp], segc, remc[bp:]))
                    better = True
                i += 1
    return seq, seq_c


def order(D, reg_country, T, restarts=60, seed=0):
    T = np.array(T)
    cmap = {c: i for i, c in enumerate(T)}
    rc = np.array([cmap[c] for c in reg_country])
    o = np.argsort(rc, kind="stable")
    regs_of = o.astype(np.int64)
    reg_ptr = np.searchsorted(rc[o], np.arange(len(T) + 1)).astype(np.int64)
    rng = np.random.default_rng(seed)
    best = None
    for k in range(restarts):
        # randomized nearest-neighbour construction
        c0 = rng.integers(len(T))
        r = regs_of[reg_ptr[c0] + rng.integers(reg_ptr[c0 + 1] - reg_ptr[c0])]
        seq, seqc, used = [r], [c0], {c0}
        while len(used) < len(T):
            d = D[seq[-1]].astype(float) + rng.random(len(rc)) * 0.9
            d[np.isin(rc, list(used))] = np.inf
            r = int(np.argmin(d))
            seq.append(r); seqc.append(rc[r]); used.add(rc[r])
        seq, seqc = improve(D, np.array(seq, np.int64), regs_of, reg_ptr, np.array(seqc, np.int64))
        cost = path_cost(D, seq)
        if best is None or cost < best[0]:
            best = (cost, seq.copy(), seqc.copy())
            print(f"  restart {k}: region-tour cost {cost}", flush=True)
    return best[1], T[best[2]]


# ------------------------------------------------------------------ the fewest fires for that order
def layered(indptr, indices, cid, comp, seq_countries):
    """Fewest hops visiting the countries in order (any summit of each). A country that can no longer be
    reached from the previous one (after zoom-10/12 bans) is dropped; returns (path, hops, dropped)."""
    n = len(cid)
    seq = [c for c in seq_countries if (comp & (cid == c)).any()]
    dropped = [c for c in seq_countries if c not in seq]
    cost = np.full(n, INF, np.int64)
    cost[comp & (cid == seq[0])] = 0
    pars = []
    for k in range(1, len(seq)):
        dist, par = bfs_multi(indptr, indices, cost)
        nxt = comp & (cid == seq[k])
        if dist[nxt].min() >= INF:
            dropped.append(seq[k])
            continue
        pars.append(par.astype(np.int32))
        cost = np.full(n, INF, np.int64)
        cost[nxt] = dist[nxt]
    y = int(np.argmin(cost))
    total = int(cost[y])
    path = [y]
    for k in range(len(pars) - 1, -1, -1):
        par = pars[k]
        while par[path[-1]] >= 0:
            path.append(int(par[path[-1]]))
    return path[::-1], total, dropped


def main2():
    t0 = time.time()
    nd, indptr, indices, A = load_graph()
    lat, lon = nd["lat"], nd["lon"]
    cid = np.where(nd["border_ok"], nd["cid"], -1)              # only fires >= 1 km inside count (border.py)
    cj = json.load(open(os.path.join(DATA, "countries.json")))
    names, counts = cj["names"], np.array(cj["counts"])
    D = np.load(os.path.join(DATA, "tour_D.npy"))
    R = json.load(open(os.path.join(DATA, "tour_regions.json")))
    T, reg_country = R["T"], np.array(R["reg_country"])
    ncomp, lab = connected_components(A, directed=False)
    is_c = np.zeros(len(lat), bool); is_c[cid >= 0] = counts[cid[cid >= 0]]
    comp_lab = np.bincount(lab[is_c & np.isin(cid, T)]).argmax()
    comp = lab == comp_lab
    seq, seqc = order(D, reg_country, T)
    seqc = [int(c) for c in seqc]
    print(f"order found {time.time() - t0:.0f}s", flush=True)
    for it in range(6):
        path, total, dropped = layered(indptr, indices, cid, comp, seqc)
        if dropped:
            print("  no longer reachable:", [names[c] for c in dropped], flush=True)
        visited = []
        for v in path:
            c = int(cid[v])
            if c >= 0 and counts[c] and c not in visited:
                visited.append(c)
        print(f"  pass {it}: {len(path)} fires, {len(visited)} countries {time.time() - t0:.0f}s", flush=True)
        if visited == seqc:
            break
        seqc = visited                    # countries passed on the way are visited where they are passed
    json.dump({"path": [int(v) for v in path], "countries": [names[c] for c in visited],
               "lat": [float(lat[v]) for v in path], "lon": [float(lon[v]) for v in path],
               "h_m": [float(nd["h"][v]) for v in path], "cid": [int(nd["cid"][v]) for v in path],
               "counts_for": [int(cid[v]) for v in path]},
              open(os.path.join(DATA, "tour.json"), "w"), indent=1)
    print(f"tour: {len(path)} fires through {len(visited)} countries")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "order":
        main2()
    else:
        main()
