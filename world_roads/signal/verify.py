"""Verify the loop at high resolution (Terrarium zoom 10, ~4x finer than the search; zoom 11 on
Greenland, where zoom 10 often shows the bedrock under the ice sheet):
  * each fire's summit must really be there: zoom-10 max within 600 m >= search height - 100 m
  * each fire-to-fire sightline must clear zoom-10 terrain sampled every 150 m
Failures are appended to data/invalid_nodes.json / data/banned_edges.json for the next search.

    python verify.py    -> prints failures, exit code 1 if any
"""
import os, sys, json, math, time, urllib.request
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import chain as CH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
Z = 10
TILE_DIR = os.path.join(DATA, f"z{Z}")
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CACHE = {}
# Terrarium's zoom 10 shows bedrock under much of Greenland's ice sheet; zoom 11 has the ice surface
GREENLAND = (59.0, 84.0, -73.6, -10.0)


def zoom_for(lat, lon):
    a, b, c, d = GREENLAND
    g = (np.asarray(lat) >= a) & (np.asarray(lat) <= b) & (np.asarray(lon) >= c) & (np.asarray(lon) <= d)
    g &= ~((np.asarray(lat) < 67.3) & (np.asarray(lon) > -31.5))           # Iceland
    return np.where(g, 11, Z)


def tile_xy(lat, lon, z=Z):
    n = 2.0 ** np.asarray(z)
    s = np.sin(np.radians(lat))
    fx = (np.asarray(lon) + 180) / 360 * n
    fy = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    return fx, fy


def fetch(t):
    t, h = _fetch(t)
    if t[0] == 11:                                    # zoom 11 is only used on Greenland: cap corrupt blocks
        import fix_greenland as FG
        h = FG.clamp_tile(h)
    return t, h


def _fetch(t):
    z, x, y = t
    d = os.path.join(DATA, f"z{z}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{x}_{y}.npy")
    if os.path.exists(path):
        return t, np.load(path)
    for a in range(6):
        try:
            raw = urllib.request.urlopen(URL.format(z=z, x=x % 2 ** z, y=y), timeout=30).read()
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
            h = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
            import fetch_dem as F1
            F1.Z = z
            h = np.maximum(F1.despike(h, y), 0).astype(np.float32)
            np.save(path, h)
            return t, h
        except Exception:
            time.sleep(1.5 ** a)
    raise RuntimeError(t)


def ensure(tiles):
    need = [t for t in set(tiles) if t not in CACHE]
    with ThreadPoolExecutor(32) as ex:
        for t, h in ex.map(fetch, need):
            CACHE[t] = h


_GI = {}


def greenland_floor(lat, lon):
    """Inside Greenland's ice sheet zoom 11 still has bedrock patches: the hole-filled grid
    (fix_greenland.py holes) gives the ice surface there (NaN elsewhere)."""
    if not _GI:
        meta = json.load(open(os.path.join(DATA, "mosaic_z8max_meta.json")))
        gi = json.load(open(os.path.join(DATA, "greenland_interior.json")))
        _GI.update(meta=meta, gi=gi, mask=np.load(os.path.join(DATA, "greenland_interior.npy")),
                   mos=np.load(os.path.join(DATA, "mosaic_z8max.npy"), mmap_mode="r"))
    meta, gi = _GI["meta"], _GI["gi"]
    Wm = meta["width"]
    s_ = np.sin(np.radians(lat))
    r = ((0.5 - 0.5 * np.log((1 + s_) / (1 - s_)) / (2 * np.pi)) * Wm - meta["row0_px"]).astype(int)
    c = ((np.asarray(lon) + 180) / 360 * Wm).astype(int)
    out = np.full(len(r), np.nan, np.float32)
    inb = (r >= gi["r0"]) & (r < gi["r1"]) & (c >= gi["c0"]) & (c < gi["c1"])
    for i in np.nonzero(inb)[0]:
        rr, cc = r[i] - gi["r0"], c[i] - gi["c0"]
        if (_GI["mask"][rr, cc >> 3] >> (7 - (cc & 7))) & 1:
            out[i] = _GI["mos"][r[i], c[i]]
    return out


def elev(lat, lon, z=None):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    out = _elev(lat, lon, z)
    g = zoom_for(lat, lon) == 11
    if g.any() and os.path.exists(os.path.join(DATA, "greenland_interior.json")):
        fl = greenland_floor(lat[g], lon[g])
        out[g] = np.where(np.isnan(fl), out[g], np.maximum(out[g], np.nan_to_num(fl)))
    return out


def _elev(lat, lon, zz=None):
    z = zoom_for(lat, lon) if zz is None else np.full(len(lat), zz)
    fx, fy = tile_xy(lat, lon, z)
    tx, ty = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ensure(list(zip(z.tolist(), tx.tolist(), ty.tolist())))
    out = np.empty(len(fx), np.float32)
    px, py = (fx - tx) * 256 - 0.5, (fy - ty) * 256 - 0.5
    for i in range(len(fx)):
        h = CACHE[(int(z[i]), tx[i], ty[i])]
        x0 = min(max(int(px[i]), 0), 254); y0 = min(max(int(py[i]), 0), 254)
        ax = min(max(px[i] - x0, 0), 1); ay = min(max(py[i] - y0, 0), 1)
        out[i] = (h[y0, x0] * (1 - ax) + h[y0, x0 + 1] * ax) * (1 - ay) + (h[y0 + 1, x0] * (1 - ax) + h[y0 + 1, x0 + 1] * ax) * ay
    return out


def summit_hr(lat, lon, r_km=0.3, z=12):
    """The summit at zoom 12 (~30 m): highest point within r_km. Catches fake low hills (data spikes under
    the despiker's 150 m threshold) that zoom 9/10 shows in flat country."""
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = [(lat, lon)] + [(lat + r * np.sin(t) / 111.2, lon + r * np.cos(t) / (111.2 * np.cos(np.radians(lat))))
                          for r in (0.05, 0.1, 0.15, 0.2, 0.25, r_km) for t in a]
    p = np.array(pts)
    return float(elev(p[:, 0], p[:, 1], z).max())


def summit_tol(h):
    return 25.0 if h < 400 else 100.0


LOW_KM = 0.600                       # summits below this are re-checked at zoom 12


def summit_check(lat, lon, h):
    """(ok, fire height). Low summits (fake hills are always low: the despiker removes spikes > 150 m)
    are re-checked at zoom 12, and the fire burns at the zoom-12 summit within 600 m. Higher summits use
    zoom 10 (11 on Greenland), because zoom 12 shows the bedrock under ice caps."""
    m10 = summit_max(lat, lon)
    if h < LOW_KM * 1000:
        m12 = summit_hr(lat, lon, r_km=0.6)
        return (m10 >= h - 100) and (m12 >= h - (25.0 if h < 400 else 60.0)), m12
    return m10 >= h - 100, max(h, m10)


def summit_max(lat, lon, r_km=0.6):
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = [(lat, lon)] + [(lat + r * np.sin(t) / 111.2, lon + r * np.cos(t) / (111.2 * np.cos(np.radians(lat))))
                          for r in (0.2, 0.4, r_km) for t in a]
    p = np.array(pts)
    return float(elev(p[:, 0], p[:, 1]).max())


def hop_clear(a, b, ha, hb, step=0.15):
    pa, pb = CH.xyz(np.array([a[0]]), np.array([a[1]]))[0], CH.xyz(np.array([b[0]]), np.array([b[1]]))[0]
    th = math.acos(min(1.0, float(pa @ pb)))
    D = th * CH.R
    n = max(int(D / step), 2)
    f = np.arange(1, n) / n
    s = f * D
    keep = (s > 0.5) & (D - s > 0.5)
    f, s = f[keep], s[keep]
    p = (np.sin((1 - f) * th)[:, None] * pa + np.sin(f * th)[:, None] * pb) / math.sin(th)
    lat = np.degrees(np.arcsin(p[:, 2])); lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    terr = elev(lat, lon) / 1000.0
    line = ha + (hb - ha) * f - s * (D - s) / (2 * CH.REFF)
    margin = line - terr
    i = int(np.argmin(margin))
    return bool(margin[i] >= 0), float(margin[i] * 1000), float(s[i])


def main():
    os.makedirs(TILE_DIR, exist_ok=True)
    L = json.load(open(os.path.join(DATA, "loop.json")))
    st = L["stations"]
    inv = json.load(open(os.path.join(DATA, "invalid_nodes.json"))) if os.path.exists(os.path.join(DATA, "invalid_nodes.json")) else []
    ban = json.load(open(os.path.join(DATA, "banned_edges.json"))) if os.path.exists(os.path.join(DATA, "banned_edges.json")) else []
    bad_nodes, bad_edges, bad_ships = [], [], []
    shp = os.path.join(DATA, "banned_ships.json")
    bans = json.load(open(shp)) if os.path.exists(shp) else []
    t0 = time.time()
    hz = []
    for i, s in enumerate(st):
        if s["kind"] != "fire" or i == 0 or i == len(st) - 1:
            hz.append(s["h_m"])
            continue
        ok, h_ = summit_check(s["lat"], s["lon"], s["h_m"])
        hz.append(h_)                            # the fire goes on the true summit within 600 m
        if not ok:
            bad_nodes.append([round(s["lat"], 5), round(s["lon"], 5)])
            print(f"  fire {i}: search height {s['h_m']:.0f} m, checked summit {h_:.0f} m -> invalid", flush=True)
    for i in range(1, len(st)):
        a, b = st[i - 1], st[i]
        ha, hb = hz[i - 1] / 1000 + CH.PYRE_KM, hz[i] / 1000 + CH.PYRE_KM
        ok, m_m, at = hop_clear((a["lat"], a["lon"]), (b["lat"], b["lon"]), ha, hb)
        if a["kind"] != "fire" or b["kind"] != "fire":
            if not ok:
                # the whole ship crossing between these two shore fires is rejected
                fa = max(k for k in range(i) if st[k]["kind"] == "fire")
                fb = min(k for k in range(i, len(st)) if st[k]["kind"] == "fire")
                bad_ships.append([[round(st[fa]["lat"], 5), round(st[fa]["lon"], 5)], [round(st[fb]["lat"], 5), round(st[fb]["lon"], 5)]])
                print(f"  ship hop {i}: blocked by {-m_m:.0f} m at {at:.0f} km", flush=True)
            continue
        if not ok:
            bad_edges.append([[round(a["lat"], 5), round(a["lon"], 5)], [round(b["lat"], 5), round(b["lon"], 5)]])
            print(f"  hop {i}: blocked by {-m_m:.0f} m at {at:.0f} km", flush=True)
    json.dump(bans + bad_ships, open(shp, "w"))
    print(f"checked {len(st)} stations: {len(bad_nodes)} bad summits, {len(bad_edges)} blocked sightlines, {len(bad_ships)} blocked ship hops, "
          f"{len(CACHE)} tiles, {time.time() - t0:.0f}s", flush=True)
    json.dump(inv + bad_nodes, open(os.path.join(DATA, "invalid_nodes.json"), "w"))
    json.dump(ban + bad_edges, open(os.path.join(DATA, "banned_edges.json"), "w"))
    L["verified_heights_m"] = hz
    json.dump(L, open(os.path.join(DATA, "loop.json"), "w"), indent=1)
    sys.exit(1 if bad_nodes or bad_edges or bad_ships else 0)


if __name__ == "__main__":
    main()
