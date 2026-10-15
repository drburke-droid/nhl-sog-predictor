"""Every capital city as a beacon site, for "how long until every capital knows?".

Capitals: Natural Earth 10 m populated places ("Admin-0 capital"), with the choices below where it lists
several. Each capital's beacon stands on the city's highest ground within RADIUS_KM of its centre, inside
the country and at least 0.5 km from any other country (Vatican City and Monaco use their existing
summits). Heights are Terrarium zoom 12, taken as the median of 9 samples within ~100 m so one bad pixel
or building cannot fake a hill.

    python capitals.py         -> data/capitals.json; the beacons join world_nodes.npz, with their sightlines
                                  (zero margin, like the flat coasts: the zoom-10/12 verification decides)
    python capitals.py start   -> the first fire, at Bletchley Park (data/capitals_start.json)
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
    else:
        main()
