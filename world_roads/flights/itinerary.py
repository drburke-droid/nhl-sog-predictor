"""Chain every flight of the modeled day into one itinerary for a single light-speed plane.

Each flight must depart from the airport where the previous one landed. Flights are taken in the
same continental sweep as the road film (waypoints.sweep_key): a flight's key is the lower of its two
airports' keys, so an intercontinental flight is flown, and usually flown back, while the plane is
working the earlier region. When the plane's airport has no eligible flight left, it flies an empty
repositioning leg (great-circle) to the nearest airport that has one; those legs count as distance.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from waypoints import sweep_key  # noqa: E402
from schedule import gc_km  # noqa: E402

DATA = os.path.join(HERE, "data")
BAND_KM = 2500.0


def main():
    S = dict(np.load(os.path.join(DATA, "schedule.npz")))
    src, dst, km, dep = S["src"], S["dst"], S["km"], S["dep"]
    lat, lon = S["lat"], S["lon"]
    n, m = len(lat), len(src)
    akey = sweep_key(lon, lat)
    fkey = np.minimum(akey[src], akey[dst])
    order = np.lexsort((dep, fkey, src))                 # per airport: lowest key, then earliest departure
    starts = np.r_[0, np.cumsum(np.bincount(src, minlength=n))]
    ptr = starts[:-1].copy()
    gorder = np.argsort(fkey, kind="stable")
    used = np.zeros(m, bool)
    g = 0
    cur = int(src[gorder[0]])
    legs = []                                            # (flight id or -1, from, to)
    xyz = np.stack([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                    np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))], 1)
    head_key = np.full(n, np.inf)                        # lowest unused flight key per airport
    for a in range(n):
        if starts[a] < starts[a + 1]:
            head_key[a] = fkey[order[starts[a]]]
    done = 0
    while done < m:
        while used[gorder[g]]:
            g += 1
        thr = fkey[gorder[g]] + BAND_KM
        while ptr[cur] < starts[cur + 1] and used[order[ptr[cur]]]:
            ptr[cur] += 1
        if ptr[cur] < starts[cur + 1] and fkey[order[ptr[cur]]] <= thr:
            f = order[ptr[cur]]; ptr[cur] += 1
            used[f] = True; done += 1
            legs.append((f, cur, int(dst[f])))
            cur = int(dst[f])
            head_key[src[f]] = fkey[order[ptr[src[f]]]] if ptr[src[f]] < starts[src[f] + 1] else np.inf
            continue
        # reposition: nearest airport (great circle) with an eligible unused flight
        for a in np.nonzero(head_key <= thr)[0]:         # refresh stale heads
            while ptr[a] < starts[a + 1] and used[order[ptr[a]]]:
                ptr[a] += 1
            head_key[a] = fkey[order[ptr[a]]] if ptr[a] < starts[a + 1] else np.inf
        cand = np.nonzero(head_key <= thr)[0]
        nxt = int(cand[np.argmax(xyz[cand] @ xyz[cur])])
        legs.append((-1, cur, nxt))
        cur = nxt
    legs = np.array(legs, np.int64)
    leg_km = np.where(legs[:, 0] >= 0, km[np.maximum(legs[:, 0], 0)],
                      gc_km(lat[legs[:, 1]], lon[legs[:, 1]], lat[legs[:, 2]], lon[legs[:, 2]]))
    dead = legs[:, 0] < 0
    stats = {"flights": int(m), "flight_km": float(km.sum()), "reposition_legs": int(dead.sum()),
             "reposition_km": float(leg_km[dead].sum()), "total_km": float(leg_km.sum()),
             "overhead": float(leg_km[dead].sum() / km.sum()), "seconds_at_c": float(leg_km.sum() / 299_792.458)}
    np.savez(os.path.join(DATA, "itinerary.npz"), legs=legs, leg_km=leg_km)
    json.dump(stats, open(os.path.join(DATA, "itinerary_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
