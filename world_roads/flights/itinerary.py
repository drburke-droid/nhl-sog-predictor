"""Chain every flight of the modeled day into one itinerary for a single light-speed plane, ROUTE BY ROUTE.

A route is an airport pair. The plane takes one route at a time and shuttles it (A->B->A->...) until
it has flown every flight scheduled on that route today; then the route is done (and is erased on
screen). Each flight departs from the airport where the previous one landed.

Route order follows the same continental sweep as the road film (waypoints.sweep_key): a route's
key is the lower of its two airports' keys. At its current airport the plane takes the unfinished
route with the lowest key (within BAND_KM of the lowest unfinished key anywhere); if there is none,
it flies an empty repositioning leg (great circle) to the nearest airport that has one. If a route's
two directions are unbalanced, the plane repositions to the other end to finish it. Repositioning
legs count as distance.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from waypoints import sweep_key  # noqa: E402
from schedule import gc_km  # noqa: E402

DATA = os.path.join(HERE, "data")
BAND_KM = 2500.0


def pairs_of(src, dst):
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    pairs, pid = np.unique(a * 100000 + b, return_inverse=True)
    return pairs // 100000, pairs % 100000, pid


def main():
    S = dict(np.load(os.path.join(DATA, "schedule.npz")))
    src, dst, km, dep = S["src"], S["dst"], S["km"], S["dep"]
    lat, lon = S["lat"], S["lon"]
    n, m = len(lat), len(src)
    pa, pb, pid = pairs_of(src, dst)
    P = len(pa)
    dirn = (src != pa[pid]).astype(np.int64)            # 0: pa->pb, 1: pb->pa
    order = np.lexsort((dep, dirn, pid))                 # flights grouped by (route, direction), by time
    key2 = pid[order] * 2 + dirn[order]
    st2 = np.r_[0, np.cumsum(np.bincount(key2, minlength=2 * P))]
    ptr2 = st2[:-1].copy()
    akey = sweep_key(lon, lat)
    pkey = np.minimum(akey[pa], akey[pb])
    inc_air = np.r_[pa, pb]; inc_pair = np.r_[np.arange(P), np.arange(P)]
    o = np.lexsort((pkey[inc_pair], inc_air))
    inc_air, inc_pair = inc_air[o], inc_pair[o]
    ist = np.r_[0, np.cumsum(np.bincount(inc_air, minlength=n))]
    iptr = ist[:-1].copy()
    done = np.zeros(P, bool)
    gorder = np.argsort(pkey, kind="stable")
    g = 0
    xyz = np.stack([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                    np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))], 1)

    def head(a):
        while iptr[a] < ist[a + 1] and done[inc_pair[iptr[a]]]:
            iptr[a] += 1
        return pkey[inc_pair[iptr[a]]] if iptr[a] < ist[a + 1] else np.inf

    head_key = np.array([head(a) for a in range(n)])
    cur = int(pa[gorder[0]])
    legs, leg_route = [], []                             # (flight id or -1, from, to), route of each leg
    left = P
    while left:
        while done[gorder[g]]:
            g += 1
        thr = pkey[gorder[g]] + BAND_KM
        hk = head(cur)
        if hk <= thr:
            p = int(inc_pair[iptr[cur]])
            while True:                                  # shuttle until the route is used up
                d = 0 if cur == pa[p] else 1
                if ptr2[2 * p + d] < st2[2 * p + d + 1]:
                    f = int(order[ptr2[2 * p + d]]); ptr2[2 * p + d] += 1
                    legs.append((f, cur, int(dst[f]))); leg_route.append(p)
                    cur = int(dst[f])
                elif ptr2[2 * p + 1 - d] < st2[2 * p + 2 - d]:
                    other = int(pb[p] if cur == pa[p] else pa[p])
                    legs.append((-1, cur, other)); leg_route.append(p)
                    cur = other
                else:
                    break
            done[p] = True
            left -= 1
            head_key[pa[p]] = head(pa[p]); head_key[pb[p]] = head(pb[p])
            continue
        cand = np.nonzero(head_key <= thr)[0]
        for a in cand:                                   # refresh stale heads
            head_key[a] = head(a)
        cand = cand[head_key[cand] <= thr]
        nxt = int(cand[np.argmax(xyz[cand] @ xyz[cur])])
        legs.append((-1, cur, nxt)); leg_route.append(-1)
        cur = nxt
    legs = np.array(legs, np.int64)
    leg_route = np.array(leg_route, np.int64)
    assert (np.sort(legs[legs[:, 0] >= 0, 0]) == np.arange(m)).all()
    leg_km = np.where(legs[:, 0] >= 0, km[np.maximum(legs[:, 0], 0)],
                      gc_km(lat[legs[:, 1]], lon[legs[:, 1]], lat[legs[:, 2]], lon[legs[:, 2]]))
    dead = legs[:, 0] < 0
    stats = {"flights": int(m), "routes": int(P), "flight_km": float(km.sum()), "reposition_legs": int(dead.sum()),
             "reposition_km": float(leg_km[dead].sum()), "total_km": float(leg_km.sum()),
             "overhead": float(leg_km[dead].sum() / km.sum()), "seconds_at_c": float(leg_km.sum() / 299_792.458)}
    np.savez(os.path.join(DATA, "itinerary.npz"), legs=legs, leg_km=leg_km, leg_route=leg_route)
    json.dump(stats, open(os.path.join(DATA, "itinerary_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
