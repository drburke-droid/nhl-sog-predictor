"""The Roomba photon: a photon drives the road network at random, at the speed of light, until it has
driven every road at least once. How long does that take?

Rule (as the lost photon, turns/lost_photon.py): at every junction it takes one of the other roads at
random, with equal odds; it never turns back, except at a dead end. Bends are followed; one-way rules
are ignored. 'uturn=True' runs the plain random walk instead (any road, including the one it came in on).

The graph: CSR over half-edges (indptr, adj = the node at the other end, eid = the road's id, alen = km).
"""
import math, time
import numpy as np
import numba as nb

C_KMS = 299_792.458


@nb.njit(cache=True)
def choose(indptr, eid, cur, came, uturn):
    """The half-edge taken at junction cur, having arrived on road `came` (-1: none)."""
    a, b = indptr[cur], indptr[cur + 1]
    d = b - a
    if uturn or came < 0:
        return a + np.random.randint(d)
    n_back = 0
    for k in range(a, b):
        if eid[k] == came:
            n_back += 1
    n_ex = d - n_back if n_back < d else 0
    if n_ex <= 0 or (n_back > 1 and d == n_back):        # dead end (or a loop's own ends): turn back
        return a + np.random.randint(d)
    r = np.random.randint(n_ex)
    skipped = 0
    for k in range(a, b):
        if eid[k] == came and skipped < 1:
            skipped += 1
            continue
        if r == 0:
            return k
        r -= 1
    return -1


@nb.njit(cache=True)
def cover_walk(indptr, adj, eid, alen, m, start, seed, uturn, log_every, max_steps):
    """Returns (steps, km, first_km per road (inf if never), log rows [steps, km, roads covered, km covered],
    the last road covered)."""
    np.random.seed(seed)
    first = np.full(m, np.inf)
    covered = 0
    covered_km = 0.0
    km = 0.0
    nlog = max_steps // log_every + 2
    log = np.zeros((nlog, 4))
    li = 0
    cur = start
    came = -1
    last = -1
    step = 0
    while step < max_steps:
        if indptr[cur + 1] == indptr[cur]:
            break
        j = choose(indptr, eid, cur, came, uturn)
        e = eid[j]
        km += alen[j]
        if first[e] == np.inf:
            first[e] = km
            covered += 1
            covered_km += alen[j]
            last = e
            if covered == m:
                step += 1
                log[li, 0] = step; log[li, 1] = km; log[li, 2] = covered; log[li, 3] = covered_km
                li += 1
                return step, km, first, log[:li], last
        came = e
        cur = adj[j]
        step += 1
        if step % log_every == 0:
            log[li, 0] = step; log[li, 1] = km; log[li, 2] = covered; log[li, 3] = covered_km
            li += 1
    return step, km, first, log[:li], last


@nb.njit(cache=True)
def walk_path(indptr, adj, eid, alen, m, start, seed, uturn, max_steps):
    """The same walk as cover_walk (same seed, same choices), recording every road driven, until all are covered."""
    np.random.seed(seed)
    seen = np.zeros(m, np.bool_)
    covered = 0
    path = np.empty(max_steps, np.int32)
    cur = start
    came = -1
    step = 0
    while step < max_steps:
        j = choose(indptr, eid, cur, came, uturn)
        e = eid[j]
        path[step] = e
        step += 1
        if not seen[e]:
            seen[e] = True
            covered += 1
            if covered == m:
                break
        came = e
        cur = adj[j]
    return path[:step]


def csr(n, ea, eb, elen):
    """Half-edge CSR for an undirected graph given as edge lists (road id = position in the lists)."""
    m = len(ea)
    ends = np.r_[ea, eb]
    nbr = np.r_[eb, ea]
    ids = np.r_[np.arange(m), np.arange(m)]
    ln = np.r_[elen, elen]
    o = np.argsort(ends, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(ends, minlength=n))].astype(np.int64)
    return indptr, nbr[o].astype(np.int32), ids[o].astype(np.int32), ln[o].astype(np.float64)


def fmt_time(s):
    if s < 1:
        return f"{s * 1000:.1f} ms"
    if s < 120:
        return f"{s:.2f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"
