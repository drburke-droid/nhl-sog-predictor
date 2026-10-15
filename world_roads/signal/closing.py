"""Close the loop: the fewest fires (mountain fires + fire ships) to get the signal from Greenland back
to Plymouth, possibly stepping across islands no land chain can reach (Iceland, Faroes, Jan Mayen,
Svalbard, Bear Island). A fire ship carries its fire 30 m up the mast.

    python closing.py   -> data/loop.json (full loop: every station in order, with type)
"""
import os, json, heapq, math
import numpy as np
import chain as CH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RS = float(CH.reach(CH.SHIP_MAST_KM))                 # a fire ship's horizon, km
ISLANDS = {"Iceland": (63.2, 66.7, -24.6, -13.4), "Faroes": (61.3, 62.5, -7.8, -6.2), "Jan Mayen": (70.7, 71.3, -9.3, -7.8),
           "Svalbard": (76.0, 81.0, 10.0, 34.0), "Bear Island": (74.2, 74.6, 18.6, 19.5)}


def thin(idx, lat, lon, H, cell=15.0):
    cy = np.floor(lat[idx] * 111.2 / cell).astype(np.int64)
    cx = np.floor(lon[idx] * 111.2 * np.cos(np.radians(lat[idx])) / cell).astype(np.int64)
    key = cy * 100000 + cx
    o = np.lexsort((-H[idx], key))
    return idx[o][np.r_[True, key[o][1:] != key[o][:-1]]]


def ships_needed(d, ra, rb):
    gap = d - ra - rb
    return 0 if gap <= 0 else int(math.ceil(gap / (2 * RS)))


def path_to(par, v):
    p = [v]
    while par[p[-1]] >= 0:
        p.append(int(par[p[-1]]))
    return p[::-1]


def main():
    z = np.load(os.path.join(DATA, "bfs.npz"))
    lat, lon, H, hop, par = z["lat"], z["lon"], z["H"], z["hop"], z["parent"]
    mos, row0, W = CH.load_terrain()
    P = CH.xyz(lat, lon)
    rch = CH.reach(H)
    reached = hop >= 0
    far = np.nonzero(reached & (lat > 59.5) & (lon > -50) & (lon < -11))[0]          # Greenland, east/south
    eu = np.nonzero(reached & (lat > 54) & (lat < 72) & (lon > -11) & (lon < 20))[0]       # N Atlantic coast of Europe
    isl = {nm: np.nonzero((~reached) & (lat > b[0]) & (lat < b[1]) & (lon > b[2]) & (lon < b[3]))[0] for nm, b in ISLANDS.items()}
    far = far[lon[far] > -45]                                       # Greenland's east side, facing Europe
    far, eu = thin(far, lat, lon, H, 25.0), thin(eu, lat, lon, H, 30.0)
    isl = {k: thin(v, lat, lon, H, 10.0) if len(v) else v for k, v in isl.items()}
    nodes = np.r_[far, eu, np.concatenate(list(isl.values()))].astype(np.int64)
    group = np.r_[np.full(len(far), -1), np.full(len(eu), -2),
                  np.concatenate([np.full(len(v), i) for i, v in enumerate(isl.values())])]
    N = len(nodes)
    print("far", len(far), "eu", len(eu), {k: len(v) for k, v in isl.items()}, flush=True)
    Pn = P[nodes]
    rn = rch[nodes]
    dist = np.full(N, np.inf); prev = np.full(N, -1); prev_ships = np.zeros(N, int)
    pq = []
    for i in range(N):
        if group[i] == -1:
            dist[i] = hop[nodes[i]] + 1
            heapq.heappush(pq, (dist[i], i))
    los_cache = {}

    def los_ok(i, j):
        k = (min(i, j), max(i, j))
        if k not in los_cache:
            a, b = nodes[i], nodes[j]
            los_cache[k] = bool(CH.los(mos, row0, W, P[a], H[a], P[b], H[b], CH.REFF, 0.5, CH.SKIP_KM, CH.CLEAR_KM))
        return los_cache[k]

    sea_cache = {}

    def sea_ok(i, j):
        """ships need open sea between the two shores: no island in the middle of the crossing"""
        k = (min(i, j), max(i, j))
        if k not in sea_cache:
            fs, ls, land = CH.sea_span(mos, row0, W, P[nodes[i]], P[nodes[j]], 0.5)
            sea_cache[k] = (fs >= 0) and not land
        return sea_cache[k]

    final = (np.inf, -1)
    done = np.zeros(N, bool)
    while pq:
        c, i = heapq.heappop(pq)
        if done[i] or c > dist[i]:
            continue
        done[i] = True
        if group[i] == -2:
            tot = c + hop[nodes[i]] - 1
            if tot < final[0]:
                final = (tot, i)
            continue
        if c >= final[0]:
            break
        d = CH.gc(Pn[i][None], Pn)
        sh = np.ceil(np.maximum(d - rn[i] - rn, 0) / (2 * RS)).astype(int)
        same = (group == group[i]) & (group >= 0)
        ok = (~done) & (group != -1) & (d <= 900) & (np.arange(N) != i) & (~same | (sh == 0))
        nc = c + sh + 1
        for j in np.nonzero(ok & (nc < dist))[0]:
            s_ = sh[j]
            if s_ == 0 and not los_ok(i, j):
                if same[j]:
                    continue
                s_ = 1
            if s_ > 0 and not sea_ok(i, j):
                continue
            if c + s_ + 1 < dist[j]:
                dist[j] = c + s_ + 1; prev[j] = i; prev_ships[j] = s_
                heapq.heappush(pq, (dist[j], j))
    D = None
    tot, y = final
    # unroll the crossing
    seq = []
    j = y
    while j >= 0:
        seq.append((j, int(prev_ships[j])))
        j = prev[j]
    seq = seq[::-1]
    x = seq[0][0]
    out_path = path_to(par, int(nodes[x]))                 # Plymouth ... X (the eastward chain)
    back_path = path_to(par, int(nodes[y]))[::-1]          # Y ... Plymouth
    stations = []
    for k, v in enumerate(out_path):
        stations.append(dict(kind="fire", leg="out", lat=float(lat[v]), lon=float(lon[v]), h_m=float((H[v] - CH.PYRE_KM) * 1000)))
    for (a, _), (b, sh) in zip(seq[:-1], seq[1:]):
        pa, pb = P[nodes[a]], P[nodes[b]]
        d = float(CH.gc(pa[None], pb[None])[0])
        ra, rb = rch[nodes[a]], rch[nodes[b]]
        if sh:
            th = d / CH.R

            def at(s):
                f = s / d
                return (np.sin((1 - f) * th) * pa + np.sin(f * th) * pb) / np.sin(th)

            def lim(p_land, h_land, from_a):
                """farthest distance along the crossing at which a ship's mast fire is visible from the shore fire"""
                lo_, hi_ = 0.0, (ra if from_a else rb) + RS
                for _ in range(24):
                    mid = (lo_ + hi_) / 2
                    ps = at(mid if from_a else d - mid)
                    ok = CH.los(mos, row0, W, p_land, h_land, ps, CH.SHIP_MAST_KM, CH.REFF, 0.3, 0.5, CH.CLEAR_KM)
                    lo_, hi_ = (mid, hi_) if ok else (lo_, mid)
                return lo_
            a_lim = lim(pa, H[nodes[a]], True)
            b_lim = lim(pb, H[nodes[b]], False)
            fs, ls, _ = CH.sea_span(mos, row0, W, pa, pb, 0.3)             # ships stay on the water
            a_lim = max(min(a_lim, ls), fs)
            b_lim = max(min(b_lim, d - fs), d - ls)
            span = d - b_lim - a_lim
            sh = 1 if span <= 0 else int(math.ceil(span / (2 * RS * 0.995))) + 1
            x0, x1 = (a_lim, d - b_lim) if span > 0 else ((a_lim + d - b_lim) / 2,) * 2
            xs = np.linspace(x0, x1, sh) if sh > 1 else np.array([x0])
            for s in xs:
                p = at(s)
                stations.append(dict(kind="ship", leg="cross", lat=float(np.degrees(np.arcsin(p[2]))),
                                     lon=float(np.degrees(np.arctan2(p[1], p[0]))), h_m=CH.SHIP_MAST_KM * 1000 - 10))
        v = nodes[b]
        stations.append(dict(kind="fire", leg="cross" if group[b] >= 0 else "back", lat=float(lat[v]), lon=float(lon[v]),
                             h_m=float((H[v] - CH.PYRE_KM) * 1000), island=list(ISLANDS)[group[b]] if group[b] >= 0 else None))
    for v in back_path[1:]:
        stations.append(dict(kind="fire", leg="back", lat=float(lat[v]), lon=float(lon[v]), h_m=float((H[v] - CH.PYRE_KM) * 1000)))
    n_out = len(out_path)
    n_ship = sum(s["kind"] == "ship" for s in stations)
    n_isl = sum(1 for s in stations if s.get("island"))
    res = dict(total_stations=len(stations) - 1, out_fires=n_out, ships=n_ship, island_fires=n_isl,
               back_fires=len(back_path) - 1, last_land_fire=dict(lat=float(lat[out_path[-1]]), lon=float(lon[out_path[-1]])),
               ship_reach_km=RS, stations=stations)
    json.dump(res, open(os.path.join(DATA, "loop.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "stations"}, indent=1))
    print("crossing:", [(s["kind"], s.get("island"), round(s["lat"], 2), round(s["lon"], 2)) for s in stations[n_out - 1:n_out + n_ship + n_isl + 1]])


if __name__ == "__main__":
    main()
