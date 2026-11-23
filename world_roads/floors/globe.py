"""Every floor on Earth: where it is (GHSL building volume, scaled to the published world total), the order
the robot takes it in, and a night-globe renderer in the style of the road films.

Floor area per 5-arcminute cell = 273 billion m^2 x (the cell's share of GHSL building volume). The robot
starts in New York, sweeps North America, Latin America, Europe, Africa, Asia and Oceania along hand-placed
waypoints (as in the world-roads film), and saves Antarctica for last.

    python globe.py      -> data/ghsl/sweep.npz, data/ghsl/countries.json
"""
import os, json, math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "ghsl")
NE = os.path.join(HERE, "data", "ne")
WORLD_FLOOR_M2 = 273e9
R = 6371.0

SEQUENCES = [
    # from New York down the east, round the Great Lakes, out to Atlantic Canada
    [(-74.0, 40.7), (-77, 39), (-80, 35), (-82, 28), (-84, 34), (-90, 35), (-88, 41), (-79, 44), (-73, 46),
     (-71, 43), (-64, 46), (-56, 49)],
    # the middle and west, up to Alaska
    [(-90, 45), (-97, 47), (-97, 38), (-97, 30), (-106, 32), (-105, 40), (-112, 41), (-112, 33), (-118, 34),
     (-122, 38), (-123, 49), (-135, 60), (-150, 64)],
    # Mexico, Central America, the Caribbean
    [(-110, 29), (-104, 22), (-99, 19), (-92, 17), (-88, 14), (-84, 10), (-80, 8.5), (-78, 21), (-72, 19), (-66, 18)],
    # South America
    [(-75, 8), (-67, 10), (-60, 4), (-60, -3), (-48, -1), (-38, -6), (-35, -9), (-39, -14), (-43, -20),
     (-47, -16), (-56, -15), (-63, -17), (-70, -16), (-77, -12), (-79, -2), (-70, -33), (-65, -26),
     (-57, -25), (-50, -25), (-46, -23), (-56, -34), (-58, -34.6), (-62, -39), (-68, -46), (-69, -53)],
    # Europe
    [(-17, 65), (-9, 39), (-3, 40), (2, 47), (-2, 53), (-8, 53), (5, 52), (10, 51), (10, 60), (25, 62),
     (20, 52), (15, 48), (12, 43), (20, 44), (24, 38), (30, 48), (38, 55)],
    # Turkey, the Levant, Africa, Arabia
    [(33, 39), (36, 33), (31, 30), (10, 34), (-6, 33), (-15, 14), (3, 8), (20, 0), (25, -15), (28, -26),
     (20, -33), (32, -25), (37, -5), (38, 8), (32, 15), (45, 22), (52, 30)],
    # Asia
    [(62, 40), (60, 56), (85, 55), (105, 52), (133, 45), (127, 37), (138, 36), (125, 44), (116, 40),
     (105, 47), (85, 43), (90, 31), (70, 30), (77, 28), (78, 12), (81, 7), (90, 23), (104, 30),
     (120, 31), (113, 23), (121, 24), (121, 14), (106, 16), (100, 14), (102, 3), (100, 0), (110, -7),
     (115, 0), (120, -3)],
    # Oceania
    [(145, -6), (131, -13), (153, -27), (151, -34), (145, -38), (138, -35), (116, -32), (147, -42),
     (178, -18), (175, -37), (170, -45)],
    # Antarctica, last: McMurdo, then the South Pole station
    [(166.7, -77.85), (0.0, -89.9)],
]


def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def densified(step_km=25.0, gap_km=2000.0):
    pts, keys, k0 = [], [], 0.0
    for seq in SEQUENCES:
        v = _xyz(*np.array(seq, float).T)
        for a, b in zip(v[:-1], v[1:]):
            ang = np.arccos(np.clip(a @ b, -1, 1))
            n = max(2, int(ang * R / step_km))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            s = np.sin(ang)
            p = (np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / s if s > 1e-9 else np.repeat(a[None], n, 0)
            pts.append(p); keys.append(k0 + t[:, 0] * ang * R)
            k0 += ang * R
        pts.append(v[-1:]); keys.append(np.array([k0]))
        k0 += gap_km
    return np.concatenate(pts), np.concatenate(keys)


def country_raster(shape, x0, y0, cell):
    feats = json.load(open(os.path.join(NE, "ne_50m_admin_0_countries.geojson")))["features"]
    h, w = shape
    im = Image.new("I", (w, h), 0)
    d = ImageDraw.Draw(im)
    names = [""]
    for i, f in enumerate(feats, 1):
        names.append(f["properties"].get("NAME_EN") or f["properties"]["NAME"])
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            d.polygon([((x - x0) / cell, (y0 - y) / cell) for x, y in p[0]], fill=i)
    return np.asarray(im, np.int32), names


def main():
    g = np.load(os.path.join(D, "agg10.npz"))
    V = g["v"].astype(np.float64)
    x0, y0, cell = float(g["x0"]), float(g["y0"]), float(g["cell"])
    F = WORLD_FLOOR_M2 * V / V.sum()
    rr, cc = np.nonzero(F > 0)
    lon = x0 + (cc + 0.5) * cell
    lat = y0 - (rr + 0.5) * cell
    pts, keys = densified()
    _, i = cKDTree(pts).query(_xyz(lon, lat))
    key = keys[i]
    order = np.argsort(key, kind="stable")
    cum = np.cumsum(F[rr, cc][order])
    np.savez_compressed(os.path.join(D, "sweep.npz"), rr=rr[order].astype(np.int32), cc=cc[order].astype(np.int32),
                        key=key[order].astype(np.float32), cum=cum, F=F.astype(np.float32), x0=x0, y0=y0, cell=cell)
    CR, names = country_raster(F.shape, x0, y0, cell)
    share = np.bincount(CR.ravel(), weights=F.ravel(), minlength=len(names))
    top = np.argsort(-share)[:15]
    out = {names[k] or "(sea / unassigned)": float(share[k]) for k in top}
    json.dump(dict(top_countries_m2=out, total_m2=float(F.sum())), open(os.path.join(D, "countries.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    # floor area near a few cities (within 40 km)
    cities = {"Tokyo": (139.69, 35.69), "Shanghai": (121.47, 31.23), "Delhi": (77.21, 28.61), "São Paulo": (-46.63, -23.55),
              "Lagos": (3.38, 6.52), "London": (-0.13, 51.51), "Mumbai": (72.88, 19.08), "Cairo": (31.24, 30.04),
              "New York": (-74.0, 40.71), "Mexico City": (-99.13, 19.43), "Beijing": (116.40, 39.90)}
    cl = {}
    for nm, (lo, la) in cities.items():
        d = np.degrees(np.arccos(np.clip(_xyz(lon, lat) @ _xyz(lo, la), -1, 1))) * 111.2
        cl[nm] = float(F[rr, cc][d < 40].sum())
    json.dump(cl, open(os.path.join(D, "cities.json"), "w"), indent=1)
    print(json.dumps(cl, indent=1))


# ---------------------------------------------------------------------------- rendering
def ortho_inverse(px, py, g):
    x = (px - g["cx"]) / g["r"]
    y = (g["cy"] - py) / g["r"]
    rho = np.hypot(x, y)
    inside = rho <= 1.0
    c = np.arcsin(np.clip(rho, 0, 1))
    p0, l0 = math.radians(g["lat0"]), math.radians(g["lon0"])
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.clip(np.cos(c) * math.sin(p0) + np.where(rho > 0, y * np.sin(c) * math.cos(p0) / rho, 0), -1, 1))
        lon = l0 + np.arctan2(x * np.sin(c), rho * np.cos(c) * math.cos(p0) - y * np.sin(c) * math.sin(p0))
    return np.degrees(lon), np.degrees(lat), np.cos(c), rho, inside


def ortho_forward(lon, lat, g):
    p0, l0 = math.radians(g["lat0"]), math.radians(g["lon0"])
    p, l = np.radians(lat), np.radians(lon)
    x = np.cos(p) * np.sin(l - l0)
    y = math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(l - l0)
    vis = math.sin(p0) * np.sin(p) + math.cos(p0) * np.cos(p) * np.cos(l - l0) > 0
    return g["cx"] + g["r"] * x, g["cy"] - g["r"] * y, vis


class Globe:
    def __init__(self):
        s = np.load(os.path.join(D, "sweep.npz"))
        self.F = s["F"]
        self.x0, self.y0, self.cell = float(s["x0"]), float(s["y0"]), float(s["cell"])
        self.rr, self.cc, self.cum = s["rr"], s["cc"], s["cum"]
        self.rank = np.full(self.F.shape, np.inf, np.float64)
        self.rank[self.rr, self.cc] = self.cum                        # floor done when this cell is finished
        self.land = self._land()

    def _land(self, res=0.05):
        feats = json.load(open(os.path.join(NE, "ne_50m_land.geojson")))["features"]
        w, h = int(360 / res), int(180 / res)
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        for f in feats:
            gg = f["geometry"]
            polys = gg["coordinates"] if gg["type"] == "MultiPolygon" else [gg["coordinates"]]
            for p in polys:
                for k, ring in enumerate(p):
                    d.polygon([((x + 180) / res, (90 - y) / res) for x, y in ring], fill=0 if k else 255)
        return np.asarray(im, np.float32) / 255.0, res

    def render(self, g, done_m2, W=1920, H=1080, ss=2, glow=1.0, fire=0.0):
        """g: dict(cx, cy, r, lon0, lat0) in output px; done_m2: floor area cleaned so far (cells in sweep order)"""
        h, w = H * ss, W * ss
        gg = dict(g, cx=g["cx"] * ss, cy=g["cy"] * ss, r=g["r"] * ss)
        py, px = np.mgrid[0:h, 0:w].astype(np.float32)
        lon, lat, mu, rho, inside = ortho_inverse(px + 0.5, py + 0.5, gg)
        lonw = ((lon + 180) % 360) - 180
        img = np.zeros((h, w, 3), np.float32)
        rng = np.random.default_rng(7)
        st = np.zeros((h, w), np.float32)
        n = 380
        sx, sy = rng.integers(0, w, n), rng.integers(0, h, n)
        st[sy, sx] = rng.uniform(0.1, 0.6, n) ** 2
        import cv2
        st = cv2.GaussianBlur(st, (0, 0), 1.0 * ss) * 6
        img += st[..., None] * np.array([0.8, 0.85, 1.0], np.float32) * (~inside)[..., None]
        land, res = self.land
        li = map_coordinates(land, [np.clip((90 - lat) / res, 0, land.shape[0] - 1),
                                    np.clip((lonw + 180) / res, 0, land.shape[1] - 1)], order=1)
        shade = np.clip(mu, 0, 1) ** 0.55
        ocean = np.array([0.010, 0.020, 0.045], np.float32)
        ground = np.array([0.028, 0.042, 0.070], np.float32)
        base = (ocean * (1 - li[..., None]) + ground * li[..., None]) * (0.45 + 0.55 * shade[..., None])
        # floor density, nearest cell
        ri = np.clip(((self.y0 - lat) / self.cell).astype(np.int64), 0, self.F.shape[0] - 1)
        ci = np.clip(((lonw - self.x0) / self.cell).astype(np.int64), 0, self.F.shape[1] - 1)
        f = self.F[ri, ci]
        cellkm2 = (self.cell * 111.32) ** 2 * np.cos(np.radians(lat))
        dens = f / np.maximum(cellkm2, 1e-3) / 1e6                   # floor m^2 per m^2 of ground
        it = (1 - np.exp(-dens * 30.0)) ** 0.8 * 1.7 * glow
        clean = self.rank[ri, ci] <= done_m2
        col = np.where(clean[..., None], np.array([1.0, 0.70, 0.28], np.float32) * 1.25, np.array([0.30, 0.74, 1.0], np.float32))
        lit = it[..., None] * col * (0.35 + 0.65 * shade[..., None])
        img = np.where(inside[..., None], base + lit, img)
        if fire > 0:
            hot = np.array([1.0, 0.55, 0.18], np.float32)
            img = np.where(inside[..., None], img + fire * (0.25 + 0.6 * li[..., None]) * hot * (0.4 + 0.6 * shade[..., None]), img)
        lum = img.max(axis=2)
        bright = img * (lum > 0.08)[..., None]
        for sg, k in ((3 * ss, 0.30), (14 * ss, 0.18)):
            img += k * cv2.GaussianBlur(bright, (0, 0), sg)
        Rr = gg["r"]
        dist = np.hypot(px - gg["cx"], py - gg["cy"])
        rim_in = np.where(inside, np.clip(rho, 0, 1) ** 14, 0)
        halo = np.where(~inside, np.exp(-(dist - Rr) / (16 * ss)), 0) + 0.35 * np.where(~inside, np.exp(-(dist - Rr) / (70 * ss)), 0)
        sky = np.array([0.30, 0.55, 1.0], np.float32) if fire <= 0 else (1 - min(fire, 1)) * np.array([0.30, 0.55, 1.0], np.float32) + min(fire, 1) * np.array([1.0, 0.55, 0.2], np.float32)
        img += (0.55 * rim_in + 0.45 * halo * (1 + 2 * fire))[..., None] * sky * 0.6
        return img, gg


if __name__ == "__main__":
    main()
