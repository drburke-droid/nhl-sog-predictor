"""Verify the loop at high resolution (Terrarium zoom 10, ~4x finer than the search):
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


def tile_xy(lat, lon):
    n = 2 ** Z
    s = np.sin(np.radians(lat))
    fx = (np.asarray(lon) + 180) / 360 * n
    fy = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    return fx, fy


def fetch(t):
    x, y = t
    path = os.path.join(TILE_DIR, f"{x}_{y}.npy")
    if os.path.exists(path):
        return t, np.load(path)
    for a in range(6):
        try:
            raw = urllib.request.urlopen(URL.format(z=Z, x=x % 2 ** Z, y=y), timeout=30).read()
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
            h = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
            import fetch_dem as F1
            F1.Z = Z
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


def elev(lat, lon):
    fx, fy = tile_xy(lat, lon)
    tx, ty = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ensure(list(zip(tx.tolist(), ty.tolist())))
    out = np.empty(len(fx), np.float32)
    px, py = (fx - tx) * 256 - 0.5, (fy - ty) * 256 - 0.5
    for i in range(len(fx)):
        h = CACHE[(tx[i], ty[i])]
        x0 = min(max(int(px[i]), 0), 254); y0 = min(max(int(py[i]), 0), 254)
        ax = min(max(px[i] - x0, 0), 1); ay = min(max(py[i] - y0, 0), 1)
        out[i] = (h[y0, x0] * (1 - ax) + h[y0, x0 + 1] * ax) * (1 - ay) + (h[y0 + 1, x0] * (1 - ax) + h[y0 + 1, x0 + 1] * ax) * ay
    return out


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
    bad_nodes, bad_edges = [], []
    t0 = time.time()
    hz = []
    for i, s in enumerate(st):
        if s["kind"] != "fire" or i == 0 or i == len(st) - 1:
            hz.append(s["h_m"])
            continue
        m = summit_max(s["lat"], s["lon"])
        hz.append(max(s["h_m"], m))              # the fire goes on the true (zoom-10) summit within 600 m
        if m < s["h_m"] - 100:
            bad_nodes.append([round(s["lat"], 5), round(s["lon"], 5)])
            print(f"  fire {i}: search height {s['h_m']:.0f} m, zoom-10 max {m:.0f} m -> invalid", flush=True)
    for i in range(1, len(st)):
        a, b = st[i - 1], st[i]
        ha, hb = hz[i - 1] / 1000 + CH.PYRE_KM, hz[i] / 1000 + CH.PYRE_KM
        ok, m_m, at = hop_clear((a["lat"], a["lon"]), (b["lat"], b["lon"]), ha, hb)
        if a["kind"] != "fire" or b["kind"] != "fire":
            if m_m < -1:
                print(f"  ship hop {i}: blocked by {-m_m:.0f} m at {at:.0f} km", flush=True)
            continue
        if not ok:
            bad_edges.append([[round(a["lat"], 5), round(a["lon"], 5)], [round(b["lat"], 5), round(b["lon"], 5)]])
            print(f"  hop {i}: blocked by {-m_m:.0f} m at {at:.0f} km", flush=True)
    print(f"checked {len(st)} stations: {len(bad_nodes)} bad summits, {len(bad_edges)} blocked sightlines, "
          f"{len(CACHE)} tiles, {time.time() - t0:.0f}s", flush=True)
    json.dump(inv + bad_nodes, open(os.path.join(DATA, "invalid_nodes.json"), "w"))
    json.dump(ban + bad_edges, open(os.path.join(DATA, "banned_edges.json"), "w"))
    L["verified_heights_m"] = hz
    json.dump(L, open(os.path.join(DATA, "loop.json"), "w"), indent=1)
    sys.exit(1 if bad_nodes or bad_edges else 0)


if __name__ == "__main__":
    main()
