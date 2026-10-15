"""Heightfields for the first-person beacon sequence (fpv.py): the real terrain around fires 18-20 of the
loop (Mont Blanc -> northern Apennines -> Dolomites), in a local azimuthal-equidistant frame (km).

  * a coarse grid over the whole view (Terrarium zoom 11, ~55 m) at 70 m
  * a fine patch around each summit (zoom 13, ~14 m) at 12 m

    python fpv_terrain.py    -> data/fpv_terrain.npz
"""
import os, json, math, time, urllib.request
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import fetch_dem as F1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TILES = os.path.join(DATA, "fpv_tiles")
R = 6371.0088
C = (45.4, 9.4)                                   # frame centre
COARSE = dict(x0=-430.0, x1=340.0, y0=-270.0, y1=260.0, d=0.070, z=11)
FINE_HALF, FINE_D, FINE_Z = 5.0, 0.012, 13
STATIONS = (18, 19, 20)


def fwd(lat, lon):
    """lat/lon -> local x, y (km), azimuthal equidistant around C."""
    p0, l0 = math.radians(C[0]), math.radians(C[1])
    p, l = np.radians(lat), np.radians(lon)
    cc = np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(l - l0)
    c = np.arccos(np.clip(cc, -1, 1))
    k = np.where(c > 1e-12, c / np.maximum(np.sin(c), 1e-12), 1.0)
    x = R * k * np.cos(p) * np.sin(l - l0)
    y = R * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(l - l0))
    return x, y


def inv(x, y):
    p0, l0 = math.radians(C[0]), math.radians(C[1])
    rho = np.hypot(x, y)
    c = rho / R
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.cos(c) * np.sin(p0) + np.where(rho > 0, y * np.sin(c) * np.cos(p0) / rho, 0))
        lon = l0 + np.arctan2(x * np.sin(c), rho * np.cos(p0) * np.cos(c) - y * np.sin(p0) * np.sin(c))
    return np.degrees(lat), np.degrees(lon)


def merc(lat, lon, z):
    n = 2 ** z
    s = np.sin(np.radians(lat))
    return (np.asarray(lon) + 180) / 360 * n * 256, (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n * 256


def tile(z, x, y):
    d = os.path.join(TILES, f"z{z}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{x}_{y}.npy")
    if os.path.exists(path):
        return np.load(path)
    for a in range(8):
        try:
            raw = urllib.request.urlopen(F1.URLS[0].format(z=z, x=x, y=y), timeout=30).read()
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
            h = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
            F1.Z = z
            h = np.maximum(F1.despike(h, y), -50).astype(np.float32)
            np.save(path, h)
            return h
        except Exception:
            time.sleep(1.5 ** a)
    raise RuntimeError((z, x, y))


def grid(x0, x1, y0, y1, d, z):
    xs = np.arange(x0, x1 + d / 2, d)
    ys = np.arange(y0, y1 + d / 2, d)
    # tiles covering the grid's lat/lon footprint
    bx, by = np.meshgrid(np.linspace(x0, x1, 60), np.linspace(y0, y1, 60))
    la, lo = inv(bx, by)
    mx, my = merc(la, lo, z)
    tx0, tx1 = int(mx.min() // 256), int(mx.max() // 256)
    ty0, ty1 = int(my.min() // 256), int(my.max() // 256)
    jobs = [(z, tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    print(f"  zoom {z}: {len(jobs)} tiles, grid {len(xs)}x{len(ys)}", flush=True)
    mos = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
    with ThreadPoolExecutor(24) as ex:
        for (zz, tx, ty), h in zip(jobs, ex.map(lambda j: tile(*j), jobs)):
            mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = h
    out = np.empty((len(ys), len(xs)), np.float32)
    for a in range(0, len(ys), 512):
        gx, gy = np.meshgrid(xs, ys[a:a + 512])
        la, lo = inv(gx, gy)
        mx, my = merc(la, lo, z)
        out[a:a + 512] = cv2.remap(mos, (mx - tx0 * 256 - 0.5).astype(np.float32), (my - ty0 * 256 - 0.5).astype(np.float32),
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out / 1000.0                                           # km


def main():
    t0 = time.time()
    L = json.load(open(os.path.join(DATA, "loop.json")))
    st = L["stations"]
    c = COARSE
    coarse = grid(c["x0"], c["x1"], c["y0"], c["y1"], c["d"], c["z"])
    print(f"coarse done {time.time() - t0:.0f}s", flush=True)
    fines, fpos = [], []
    for i in STATIONS:
        x, y = fwd(st[i]["lat"], st[i]["lon"])
        x, y = float(x), float(y)
        f = grid(x - FINE_HALF, x + FINE_HALF, y - FINE_HALF, y + FINE_HALF, FINE_D, FINE_Z)
        fines.append(f); fpos.append((x - FINE_HALF, y - FINE_HALF))
        print(f"  fine patch for fire {i} done {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(DATA, "fpv_terrain.npz"), coarse=coarse, cx0=c["x0"], cy0=c["y0"], cd=c["d"],
             fine=np.stack(fines), fpos=np.array(fpos), fd=FINE_D)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
