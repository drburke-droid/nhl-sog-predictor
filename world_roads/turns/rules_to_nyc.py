"""Which driving rule gets you from Los Angeles to New York? Tested on the major-road graph
(motorway ... tertiary, from the world road film) of North America.

Rules, applied at every intersection (3+ road pieces); bends are followed, dead ends U-turn:
  alt_lr   alternate leftmost / rightmost exit
  konami   ↑ ↑ ↓ ↓ ← → ← → B A, repeating: ↑ straightest, ↓ U-turn, ← leftmost, → rightmost,
           B "brake" = smallest road class, A "accelerate" = biggest road class
  compass  the exit pointing most directly at New York
  tremaux  compass, but never drive the same road piece in the same direction twice
           (if every exit is used, take the least-used one) - a maze-solver's memory
A walk is "trapped" when its state repeats (it will loop forever), "arrived" within 15 km of NYC.
"""
import os, sys, json, heapq
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(os.path.dirname(HERE), "data", "graph.npz")
LA = (-118.2427, 34.0537)
NYC = (-74.0060, 40.7128)
R = 6371.0088


def load():
    g = dict(np.load(GRAPH))
    lon, lat = g["lon"], g["lat"]
    ea, eb, ecls, elen = g["ea"], g["eb"], g["ecls"], g["elen"]
    sel = (lon[ea] > -170) & (lon[ea] < -50) & (lat[ea] > 15) & (lat[ea] < 72)   # North America
    ea, eb, ecls, elen = ea[sel], eb[sel], ecls[sel], elen[sel]
    n = len(lon)
    m = len(ea)
    ends = np.r_[ea, eb]; nb = np.r_[eb, ea]; eid = np.r_[np.arange(m), np.arange(m)]
    o = np.argsort(ends, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(ends, minlength=n))]
    return dict(lon=lon, lat=lat, indptr=indptr, adj=nb[o], eid=eid[o], ecls=ecls, elen=elen, n=n, m=m)


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def gc_km(lon1, lat1, lon2, lat2):
    a = np.sin(np.radians(lat2 - lat1) / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def local_vec(G, a, b):
    """east/north vector from node a to node b (km, local tangent plane)"""
    lat0 = np.radians(G["lat"][a])
    return np.array([np.radians(G["lon"][b] - G["lon"][a]) * np.cos(lat0) * R, np.radians(G["lat"][b] - G["lat"][a]) * R])


def look(G, prev, cur, steps=3):
    """a node a few pieces along the branch prev->cur, through bends"""
    p, c = prev, cur
    for _ in range(steps):
        nb = G["adj"][G["indptr"][c]:G["indptr"][c + 1]]
        if len(nb) != 2:
            break
        nxt = nb[0] if nb[0] != p else nb[1]
        p, c = c, nxt
    return c


def turn_angle(G, prev, cur, w):
    h = local_vec(G, look(G, cur, prev), cur)
    d = local_vec(G, cur, look(G, cur, w))
    return float(np.arctan2(h[0] * d[1] - h[1] * d[0], h @ d))          # + = left


def bearing_err(G, cur, w, target):
    d = local_vec(G, cur, look(G, cur, w))
    t = np.array([np.radians(target[0] - G["lon"][cur]) * np.cos(np.radians(G["lat"][cur])), np.radians(target[1] - G["lat"][cur])])
    return float(np.arccos(np.clip(d @ t / (np.linalg.norm(d) * np.linalg.norm(t) + 1e-12), -1, 1)))


KONAMI = ["S", "S", "U", "U", "L", "R", "L", "R", "B", "A"]


def run(G, rule, start, first, target=NYC, max_steps=3_000_000):
    adj, ip, eid, ecls, elen = G["adj"], G["indptr"], G["eid"], G["ecls"], G["elen"]
    tgt_xyz = xyz(*target)
    prev, cur, k = start, first, 0
    path = [start, first]
    km = 0.0
    used = {}
    seen = set()
    for _ in range(max_steps):
        if gc_km(G["lon"][cur], G["lat"][cur], *target) < 15:
            return dict(result="arrived", path=path, km=km)
        state = (prev, cur, k % (len(KONAMI) if rule == "konami" else 2))
        if rule != "tremaux":
            if state in seen:
                return dict(result="trapped", path=path, km=km)
            seen.add(state)
        nbr = adj[ip[cur]:ip[cur + 1]]; eids = eid[ip[cur]:ip[cur + 1]]
        exits = [(w, e) for w, e in zip(nbr, eids) if w != prev]
        back = [(w, e) for w, e in zip(nbr, eids) if w == prev]
        if not exits:
            nxt, e = back[0]
        elif len(nbr) <= 2:
            nxt, e = exits[0]
        else:
            if rule == "alt_lr":
                angs = [turn_angle(G, prev, cur, w) for w, _ in exits]
                nxt, e = exits[int(np.argmax(angs)) if k % 2 == 0 else int(np.argmin(angs))]
                k += 1
            elif rule == "konami":
                c = KONAMI[k % len(KONAMI)]
                angs = np.array([turn_angle(G, prev, cur, w) for w, _ in exits])
                if c == "S":
                    nxt, e = exits[int(np.argmin(np.abs(angs)))]
                elif c == "U":
                    nxt, e = back[0]
                elif c == "L":
                    nxt, e = exits[int(np.argmax(angs))]
                elif c == "R":
                    nxt, e = exits[int(np.argmin(angs))]
                else:
                    cl = np.array([ecls[e_] for _, e_ in exits])
                    best = cl.max() if c == "B" else cl.min()
                    cand = [i for i in range(len(exits)) if cl[i] == best]
                    nxt, e = exits[min(cand, key=lambda i: abs(angs[i]))]
                k += 1
            else:
                errs = [bearing_err(G, cur, w, target) for w, _ in exits]
                if rule == "compass":
                    nxt, e = exits[int(np.argmin(errs))]
                else:   # tremaux
                    order = sorted(range(len(exits)), key=lambda i: (used.get((cur, exits[i][0]), 0), errs[i]))
                    nxt, e = exits[order[0]]
        used[(cur, nxt)] = used.get((cur, nxt), 0) + 1
        km += elen[e]
        prev, cur = cur, nxt
        path.append(cur)
    return dict(result="gave up", path=path, km=km)


def shortest_km(G, s, t):
    dist = {s: 0.0}; h = [(0.0, s)]
    while h:
        d, u = heapq.heappop(h)
        if u == t:
            return d
        if d > dist[u]:
            continue
        for v, e in zip(G["adj"][G["indptr"][u]:G["indptr"][u + 1]], G["eid"][G["indptr"][u]:G["indptr"][u + 1]]):
            nd = d + G["elen"][e]
            if nd < dist.get(v, 1e18):
                dist[v] = nd; heapq.heappush(h, (nd, v))
    return None


if __name__ == "__main__":
    G = load()
    deg = np.diff(G["indptr"])
    cand = np.nonzero(deg >= 2)[0]
    s = int(cand[np.argmin(gc_km(G["lon"][cand], G["lat"][cand], *LA))])
    t = int(cand[np.argmin(gc_km(G["lon"][cand], G["lat"][cand], *NYC))])
    nb = G["adj"][G["indptr"][s]:G["indptr"][s + 1]]
    first = int(nb[np.argmax(G["lon"][nb])])                      # set off eastward
    out = {"shortest_route_km": shortest_km(G, s, t), "straight_line_km": float(gc_km(*LA, *NYC))}
    for rule in ("alt_lr", "konami", "compass", "tremaux"):
        r = run(G, rule, s, first)
        p = np.array(r["path"])
        lon, lat = G["lon"][p], G["lat"][p]
        closest = float(gc_km(lon, lat, *NYC).min())
        far = float(gc_km(lon, lat, *LA).max())
        out[rule] = dict(result=r["result"], km=round(r["km"]), steps=len(p), closest_to_nyc_km=round(closest),
                         farthest_from_la_km=round(far), end=[round(float(lon[-1]), 2), round(float(lat[-1]), 2)],
                         light_ms=round(r["km"] / 299_792.458 * 1000, 2))
        np.savez(os.path.join(HERE, "data", f"rule_{rule}.npz"), lon=lon, lat=lat)
        print(rule, json.dumps(out[rule]), flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "rules_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))
