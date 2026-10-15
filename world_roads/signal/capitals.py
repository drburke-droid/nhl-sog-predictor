"""Every capital city as a beacon site, for "how long until every capital knows?".

Capitals: Natural Earth 10 m populated places ("Admin-0 capital"), with the choices below where it lists
several. Each capital's beacon stands on the city's highest ground within RADIUS_KM of its centre, inside
the country and at least 0.5 km from any other country (Vatican City and Monaco use their existing
summits). Heights are Terrarium zoom 12, taken as the median of 9 samples within ~100 m so one bad pixel
or building cannot fake a hill.

    python capitals.py         -> data/capitals.json; the beacons join world_nodes.npz, with their sightlines
                                  (zero margin, like the flat coasts: the zoom-10/12 verification decides)
    python capitals.py start   -> the first fire, at Bletchley Park (data/capitals_start.json)
    python capitals.py plains  -> ground-fire corridors to capitals that stand on plains too flat for the
                                  summit grid (one candidate per ~12 km is too sparse there)
"""
import os, json, math, time
import numpy as np
import shapely
from shapely.strtree import STRtree
import chain as CH
import edges as ED
import world as WD
import verify as V
import border as BD

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NE = os.path.join(os.path.dirname(HERE), "data", "ne")
RADIUS_KM = 10.0
CHOICE = {"Bolivia": "La Paz", "Tanzania": "Dodoma", "South Africa": "Pretoria", "Côte d'Ivoire": "Yamoussoukro",
          "Benin": "Porto-Novo", "Myanmar": "Naypyidaw", "Eswatini": "Mbabane", "Netherlands": "Amsterdam",
          "Nigeria": "Abuja", "Philippines": "Manila", "Japan": "Tokyo", "Chile": "Santiago", "Malaysia": "Kuala Lumpur",
          "Israel": "Jerusalem", "Morocco": "Rabat", "Sri Lanka": "Sri Jayawardenepura Kotte"}
MANUAL = {"Palestine": ("Ramallah", 31.9029, 35.2062), "South Sudan": ("Juba", 4.8594, 31.5713),
          "Nauru": ("Yaren", -0.5477, 166.9209)}
EXISTING = {"Vatican": "Vatican City", "Monaco": "Monaco"}


def capital_list(names, counts):
    gj = json.load(open(os.path.join(NE, "ne_10m_admin_0_countries.geojson")))
    a3 = [f["properties"]["ADM0_A3"] for f in gj["features"]]
    pp = [f["properties"] for f in json.load(open(os.path.join(NE, "ne_10m_populated_places_simple.geojson")))["features"]]
    caps = [p for p in pp if p["featurecla"] in ("Admin-0 capital", "Admin-0 capital alt")]
    out = {}
    for g, nm in enumerate(names):
        if not counts[g]:
            continue
        if nm in MANUAL:
            out[g] = MANUAL[nm]
            continue
        c = [p for p in caps if p["adm0_a3"] == a3[g]]
        if nm in CHOICE:
            c = [p for p in c if p["name"] == CHOICE[nm]]
        else:
            c = [p for p in c if p["featurecla"] == "Admin-0 capital"] or c
        p = c[0]
        out[g] = (p["name"], float(p["latitude"]), float(p["longitude"]))
    return out


def beacon_site(lat, lon, geom, others):
    """Highest robust ground within RADIUS_KM, inside geom, >= 0.5 km from the other countries."""
    rs = np.arange(0.0, RADIUS_KM + 0.01, 0.35)
    pts = [(lat, lon)]
    for r in rs[1:]:
        k = max(6, int(2 * math.pi * r / 0.35))
        for t in np.linspace(0, 2 * math.pi, k, endpoint=False):
            pts.append((lat + r * math.sin(t) / 111.2, lon + r * math.cos(t) / (111.2 * math.cos(math.radians(lat)))))
    p = np.array(pts)
    ok = shapely.contains_xy(geom, p[:, 1], p[:, 0])
    if others is not None:
        d = 0.5 / 111.2 / max(math.cos(math.radians(lat)), 0.2)
        near = shapely.dwithin(others, shapely.points(p[:, 1], p[:, 0]), d)
        ok &= ~near
    p = p[ok]
    if not len(p):
        return None
    h = V.elev(p[:, 0], p[:, 1], 12)
    best = []
    for i in np.argsort(-h)[:12]:                      # robust height of the 12 highest samples
        la, lo = p[i]
        off = np.array([(a, b) for a in (-0.06, 0, 0.06) for b in (-0.06, 0, 0.06)])
        q = np.c_[la + off[:, 0] / 111.2, lo + off[:, 1] / (111.2 * math.cos(math.radians(la)))]
        best.append((float(np.median(V.elev(q[:, 0], q[:, 1], 12))), float(la), float(lo)))
    return max(best)


def main():
    t0 = time.time()
    names, geoms, is_c = WD.countries()
    cj = json.load(open(os.path.join(DATA, "countries.json")))
    caps = capital_list(cj["names"], cj["counts"])
    nd = dict(np.load(os.path.join(DATA, "world_nodes.npz")))
    n0 = len(nd["lat"])
    rec, new = [], []
    for g, (cname, la, lo) in sorted(caps.items(), key=lambda x: cj["names"][x[0]]):
        nm = cj["names"][g]
        if nm in EXISTING:
            k = int(np.nonzero(nd["cid"] == g)[0][0])
            rec.append(dict(cid=g, country=nm, capital=EXISTING[nm], lat=la, lon=lo, b_lat=float(nd["lat"][k]),
                            b_lon=float(nd["lon"][k]), b_h=float(nd["h"][k]), node=k))
            continue
        others = shapely.union_all([geoms[j] for j in STRtree(geoms).query(geoms[g].buffer(0.2)) if j != g]) \
            if len(STRtree(geoms).query(geoms[g].buffer(0.2))) > 1 else None
        site = beacon_site(la, lo, geoms[g], others)
        if site is None:
            print(f"  {nm}: no ground inside the country near {cname}", flush=True)
            continue
        h, bla, blo = site
        rec.append(dict(cid=g, country=nm, capital=cname, lat=la, lon=lo, b_lat=bla, b_lon=blo, b_h=max(h, 1.0),
                        node=n0 + len(new)))
        new.append((bla, blo, max(h, 1.0), g))
        print(f"  {nm:24s} {cname:26s} beacon {max(h, 1.0):6.0f} m, "
              f"{math.hypot((bla - la) * 111.2, (blo - lo) * 111.2 * math.cos(math.radians(la))):4.1f} km from the centre "
              f"{time.time() - t0:.0f}s", flush=True)
    e = np.array(new)
    nd["lat"] = np.r_[nd["lat"], e[:, 0]]; nd["lon"] = np.r_[nd["lon"], e[:, 1]]
    nd["h"] = np.r_[nd["h"], e[:, 2]]; nd["cid"] = np.r_[nd["cid"], e[:, 3].astype(np.int32)]
    nd["border_ok"] = np.r_[nd["border_ok"], np.ones(len(e), bool)]
    np.savez(os.path.join(DATA, "world_nodes.npz"), **nd)
    json.dump(rec, open(os.path.join(DATA, "capitals.json"), "w"), indent=1)
    # sightlines from every capital beacon (zero margin; verification decides)
    lat, lon = nd["lat"], nd["lon"]
    H = nd["h"] / 1000 + CH.PYRE_KM
    P = CH.xyz(lat, lon)
    rch = CH.reach(H)
    pu, pv = [], []
    for r in rec:
        u = r["node"]
        d = np.arccos(np.clip(P[u] @ P.T, -1, 1)) * CH.R
        v = np.nonzero((d <= rch[u] + rch) & (np.arange(len(lat)) != u))[0]
        pu.append(np.full(len(v), u)); pv.append(v)
    pu = np.concatenate(pu).astype(np.int64); pv = np.concatenate(pv).astype(np.int64)
    mos, row0, W = ED.load_global()
    ok = BD.los_pairs0(mos, row0, W, P, H, pu, pv, CH.REFF, 0.25)
    ed = np.load(os.path.join(DATA, "world_edges.npz"))
    a, b = np.minimum(pu[ok], pv[ok]), np.maximum(pu[ok], pv[ok])
    pairs = np.unique(np.c_[a, b], axis=0)
    np.savez(os.path.join(DATA, "world_edges.npz"), u=np.r_[ed["u"], pairs[:, 0].astype(np.int32)],
             v=np.r_[ed["v"], pairs[:, 1].astype(np.int32)])
    print(f"{len(rec)} capitals, {len(new)} new beacons, {len(pairs):,} sightlines  {time.time() - t0:.0f}s")


def add_plains(step_km=3.0, offsets_km=(-4.0, 0.0, 4.0), max_km=400.0):
    """For each capital the fire cannot reach although it reaches land within max_km: candidate ground fires
    every step_km along the great circle from the nearest lit fire to the capital's beacon, on three parallel
    lines. A 10 m fire on flat ground sees another about 24 km away, but the summit grid holds only the
    highest point of each ~12 km cell, and on a plain that is often too sparse. Heights are zoom-12 ground
    (median of 9 samples); points off land (Natural Earth 10 m) are skipped. Every sightline to or between
    them is checked on zoom-10 terrain here, so they join the graph verified."""
    import capitals_tree as CT
    import tour as TR
    from shapely.geometry import shape
    nd, indptr, indices, _, _ = CT.load(None)
    caps = json.load(open(os.path.join(DATA, "capitals.json")))
    root = json.load(open(os.path.join(DATA, "capitals_start.json")))["node"]
    d0 = np.full(len(nd["lat"]), CT.INF, np.int64); d0[root] = 0
    dist, _ = TR.bfs_multi(indptr, indices, d0)
    land = shapely.union_all([shape(f["geometry"]) for f in json.load(open(os.path.join(NE, "ne_10m_land.geojson")))["features"]])
    shapely.prepare(land)
    names, geoms, _ = WD.countries()
    tree = STRtree(geoms)
    P = CH.xyz(nd["lat"], nd["lon"])
    okd = json.load(open(CT.OK_PATH))
    capn = {c["node"] for c in caps}
    known = {int(k_) for k_ in okd["nodes"]} | capn | {root}
    lit = np.array([v for v in np.nonzero(dist < CT.INF)[0] if int(v) in known])      # lit and verified
    new_n, new_e = [], []
    n0 = len(nd["lat"])
    for c in caps:
        if dist[c["node"]] < CT.INF:
            continue
        b = CH.xyz(np.array([c["b_lat"]]), np.array([c["b_lon"]]))[0]
        dd = np.arccos(np.clip(P[lit] @ b, -1, 1)) * CH.R
        k = int(np.argmin(dd))
        if dd[k] > max_km:
            continue
        a = int(lit[k])
        A, B = (float(nd["lat"][a]), float(nd["lon"][a])), (c["b_lat"], c["b_lon"])
        D = float(dd[k])
        pts = []
        for f in np.arange(step_km, D - 1.0, step_km) / D:
            q = slerp(A, B, f)
            h_ = heading(q, B)
            for off in offsets_km:
                la = q[0] + off * math.cos(h_ + math.pi / 2) / 111.2                  # off km to the right
                lo = q[1] + off * math.sin(h_ + math.pi / 2) / (111.2 * math.cos(math.radians(q[0])))
                pts.append((la, lo))
        pts = [p_ for p_ in pts if land.contains(shapely.Point(p_[1], p_[0]))]
        off9 = np.array([(x, y) for x in (-0.06, 0, 0.06) for y in (-0.06, 0, 0.06)])
        for la, lo in pts:
            q = np.c_[la + off9[:, 0] / 111.2, lo + off9[:, 1] / (111.2 * math.cos(math.radians(la)))]
            h = float(np.median(V.elev(q[:, 0], q[:, 1], 12)))
            if h < 1.0:
                continue
            hit = [j for j in tree.query(shapely.Point(lo, la)) if geoms[j].contains(shapely.Point(lo, la))]
            new_n.append((la, lo, h, hit[0] if hit else -1))
        print(f"  {c['country']}: {D:.0f} km from the nearest lit fire, {len(pts)} corridor points", flush=True)
    if not new_n:
        print("no corridors needed")
        return
    e = np.array(new_n)
    for k_, arr in (("lat", e[:, 0]), ("lon", e[:, 1]), ("h", e[:, 2])):
        nd[k_] = np.r_[nd[k_], arr]
    nd["cid"] = np.r_[nd["cid"], e[:, 3].astype(np.int32)]
    nd["border_ok"] = np.r_[nd["border_ok"], np.ones(len(e), bool)]
    nd = dict(nd)
    # sightlines from every new point to every point (old or new) whose horizon overlaps, checked on zoom 10
    H = nd["h"] / 1000 + CH.PYRE_KM
    P = CH.xyz(nd["lat"], nd["lon"])
    rch = CH.reach(H)
    hz = {int(k_): v for k_, v in okd["nodes"].items()}
    hz.update({v: float(nd["h"][v]) for v in capn | {root}})          # capital beacons and Bletchley: zoom-12 ground
    t0 = time.time()
    for i in range(n0, len(nd["lat"])):
        d = np.arccos(np.clip(P[i] @ P.T, -1, 1)) * CH.R
        cand = np.nonzero((d <= rch[i] + rch) & (np.arange(len(d)) < i) & (d < 60.0))[0]
        for j in cand:
            j = int(j)
            hj = hz.get(j, nd["h"][j] if j >= n0 else None)
            if hj is None:                                 # an unchecked summit: let the tree search check it later
                continue
            ok, m, at = V.hop_clear((nd["lat"][i], nd["lon"][i]), (nd["lat"][j], nd["lon"][j]),
                                    nd["h"][i] / 1000 + CH.PYRE_KM, hj / 1000 + CH.PYRE_KM)
            if ok:
                new_e.append((j, i))
    print(f"{len(new_n)} ground fires, {len(new_e)} verified sightlines  {time.time() - t0:.0f}s")
    base = dict(np.load(os.path.join(DATA, "world_nodes.npz")))
    np.savez(os.path.join(DATA, "world_nodes.npz"), **{k_: nd[k_] for k_ in base})
    ed = np.load(os.path.join(DATA, "world_edges.npz"))
    ne = np.array(new_e, np.int64)
    np.savez(os.path.join(DATA, "world_edges.npz"), u=np.r_[ed["u"], ne[:, 0].astype(np.int32)], v=np.r_[ed["v"], ne[:, 1].astype(np.int32)])
    for i in range(n0, len(nd["lat"])):
        okd["nodes"][str(i)] = float(nd["h"][i])
    okd["edges"] = sorted(set(tuple(x) for x in okd["edges"]) | {tuple(sorted(x)) for x in new_e})
    okd["demoted"] = sorted(set(okd.get("demoted", [])) | set(range(n0, len(nd["lat"]))))
    okd["pruned"] = sorted(set(okd.get("pruned", [])) | set(range(n0, len(nd["lat"]))))
    json.dump(okd, open(CT.OK_PATH, "w"))
    json.dump(dict(first=n0, n=len(new_n)), open(os.path.join(DATA, "capitals_plains.json"), "w"))


def slerp(a, b, f):
    pa, pb = CH.xyz(np.array([a[0]]), np.array([a[1]]))[0], CH.xyz(np.array([b[0]]), np.array([b[1]]))[0]
    th = math.acos(min(1.0, float(pa @ pb)))
    p = (math.sin((1 - f) * th) * pa + math.sin(f * th) * pb) / math.sin(th)
    return math.degrees(math.asin(p[2])), math.degrees(math.atan2(p[1], p[0]))


def heading(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return math.atan2(y, x)                                 # radians clockwise from north


START = ("Bletchley Park", 51.9977, -0.7407)      # the WWII codebreakers' home; the 2023 AI Safety Summit


def add_start():
    """The warning's first fire, lit at Bletchley Park (zoom-12 ground, median of 9 samples)."""
    nm, la, lo = START
    off = np.array([(a, b) for a in (-0.06, 0, 0.06) for b in (-0.06, 0, 0.06)])
    q = np.c_[la + off[:, 0] / 111.2, lo + off[:, 1] / (111.2 * math.cos(math.radians(la)))]
    h = max(float(np.median(V.elev(q[:, 0], q[:, 1], 12))), 1.0)
    nd = dict(np.load(os.path.join(DATA, "world_nodes.npz")))
    u = len(nd["lat"])
    names = json.load(open(os.path.join(DATA, "countries.json")))["names"]
    nd["lat"] = np.r_[nd["lat"], la]; nd["lon"] = np.r_[nd["lon"], lo]; nd["h"] = np.r_[nd["h"], h]
    nd["cid"] = np.r_[nd["cid"], np.int32(names.index("UK"))]; nd["border_ok"] = np.r_[nd["border_ok"], True]
    np.savez(os.path.join(DATA, "world_nodes.npz"), **nd)
    lat, lon = nd["lat"], nd["lon"]
    H = nd["h"] / 1000 + CH.PYRE_KM
    P = CH.xyz(lat, lon)
    rch = CH.reach(H)
    d = np.arccos(np.clip(P[u] @ P.T, -1, 1)) * CH.R
    v = np.nonzero((d <= rch[u] + rch) & (np.arange(len(lat)) != u))[0].astype(np.int64)
    mos, row0, W = ED.load_global()
    ok = BD.los_pairs0(mos, row0, W, P, H, np.full(len(v), u, np.int64), v, CH.REFF, 0.25)
    ed = np.load(os.path.join(DATA, "world_edges.npz"))
    np.savez(os.path.join(DATA, "world_edges.npz"), u=np.r_[ed["u"], v[ok].astype(np.int32)],
             v=np.r_[ed["v"], np.full(int(ok.sum()), u, np.int32)])
    json.dump(dict(name=nm, lat=la, lon=lo, h=h, node=int(u)), open(os.path.join(DATA, "capitals_start.json"), "w"), indent=1)
    print(f"{nm}: {h:.0f} m, {int(ok.sum())} sightlines")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        add_start()
    elif len(sys.argv) > 1 and sys.argv[1] == "plains":
        add_plains()
    else:
        main()
