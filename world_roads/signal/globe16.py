"""16:9 night-globe renderer for the long video: world texture (globe_texture_world.py), optional
high-resolution relief patches for close-ups, fires / sightlines with bloom, country tints.
"""
import os, json, math
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NE = os.path.join(os.path.dirname(HERE), "data", "ne")
W, H = 1920, 1080
FIRE = np.array([1.0, 0.55, 0.15], np.float32)
CORE = np.array([1.0, 0.92, 0.75], np.float32)
SHIP = np.array([1.0, 0.80, 0.45], np.float32)
R_EARTH = 6371.0088


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def slerp_ll(a, b, u):
    """Great-circle interpolation between (lat, lon) a and b."""
    pa, pb = unit(*a), unit(*b)
    th = math.acos(min(1.0, max(-1.0, float(pa @ pb))))
    if th < 1e-9:
        return a
    p = (math.sin((1 - u) * th) * pa + math.sin(u * th) * pb) / math.sin(th)
    return math.degrees(math.asin(p[2])), math.degrees(math.atan2(p[1], p[0]))


def gc_km(a, b):
    return 2 * R_EARTH * math.asin(min(1.0, float(np.linalg.norm(unit(*a) - unit(*b))) / 2))


def sprite(sig):
    n = int(sig * 4) * 2 + 1
    k = np.arange(n, dtype=np.float32) - n // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig)).astype(np.float32)


def make_patch(name, la0, la1, lo0, lo1, ppd=150):
    """A high-resolution relief texture (equirectangular) for a close-up, cached in data/."""
    path = os.path.join(DATA, f"patch_{name}.png")
    if os.path.exists(path):
        return cv2.imread(path)[..., ::-1].copy(), (la0, la1, lo0, lo1)
    meta = json.load(open(os.path.join(DATA, "mosaic_global_meta.json")))
    mos = np.load(os.path.join(DATA, "mosaic_global.npy"), mmap_mode="r")
    Wm, row0 = meta["width"], meta["row0_px"]
    th, tw = int((la1 - la0) * ppd), int((lo1 - lo0) * ppd)
    lat = la1 - (np.arange(th) + 0.5) / ppd
    lon = lo0 + (np.arange(tw) + 0.5) / ppd
    s = np.sin(np.radians(lat))
    my = (0.5 - 0.5 * np.log((1 + s) / (1 - s)) / (2 * np.pi)) * Wm - row0 - 0.5
    mx = (lon + 180) / 360 * Wm - 0.5
    r0, r1 = int(my.min()) - 2, int(my.max()) + 3
    c0, c1 = int(mx.min()) - 2, int(mx.max()) + 3
    sub = np.asarray(mos[r0:r1, c0:c1], np.float32)
    MX, MY = np.meshgrid((mx - c0).astype(np.float32), (my - r0).astype(np.float32))
    elev = cv2.remap(sub, MX, MY, cv2.INTER_LINEAR)
    elev = cv2.GaussianBlur(elev, (0, 0), 0.8)
    km_px = 111.2 / ppd
    dx = km_px * np.cos(np.radians(lat))[:, None] * 1000
    gx = (np.roll(elev, -1, 1) - np.roll(elev, 1, 1)) / (2 * dx)
    gy = (np.roll(elev, -1, 0) - np.roll(elev, 1, 0)) / (2 * km_px * 1000)
    az, alt = np.radians(315), np.radians(40)
    slope = np.arctan(np.hypot(gx, gy) * 3.0)
    aspect = np.arctan2(-gx, gy)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    hn = np.clip(elev / 3000, 0, 1) ** 0.7
    land = np.zeros((th, tw), np.uint8)
    border = np.zeros((th, tw), np.uint8)

    def pxl(c):
        return (np.c_[(c[:, 0] - lo0) * ppd, (la1 - c[:, 1]) * ppd] * 16).astype(np.int32)
    for fn, draw in (("ne_10m_land.geojson", "fill"), ("ne_10m_admin_0_countries.geojson", "line")):
        gj = json.load(open(os.path.join(NE, fn)))
        for f in gj["features"]:
            g = f["geometry"]
            for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                for k, ring in enumerate(poly):
                    c = np.asarray(ring)
                    if c[:, 0].max() < lo0 - 1 or c[:, 0].min() > lo1 + 1 or c[:, 1].max() < la0 - 1 or c[:, 1].min() > la1 + 1:
                        continue
                    if draw == "fill":
                        cv2.fillPoly(land, [pxl(c)], 255 if k == 0 else 0, cv2.LINE_AA, shift=4)
                    else:
                        cv2.polylines(border, [pxl(c)], True, 255, 1, cv2.LINE_AA, shift=4)
    land = land.astype(np.float32) / 255
    base = np.array([0.075, 0.095, 0.13]); hi = np.array([0.20, 0.23, 0.28]); ocean = np.array([0.012, 0.022, 0.045])
    tex = np.empty((th, tw, 3), np.float32)
    for ch in range(3):
        col = (base[ch] + (hi[ch] - base[ch]) * hn) * (0.55 + 0.75 * shade)
        tex[..., ch] = ocean[ch] * (1 - land) + col * land
    coast = cv2.morphologyEx((land > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    tex += coast[..., None] * np.array([0.05, 0.07, 0.10], np.float32)
    tex += ((border.astype(np.float32) / 255) * land)[..., None] * np.array([0.06, 0.075, 0.10], np.float32)
    out = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, out[..., ::-1])
    return out, (la0, la1, lo0, lo1)


class Globe16:
    def __init__(self, patches=()):
        self.tex = cv2.imread(os.path.join(DATA, "globe_tex_world.png"))[..., ::-1].copy()
        self.TH, self.TW = self.tex.shape[:2]
        self.ys, self.xs = np.mgrid[0:H, 0:W].astype(np.float32)
        self.patches = [make_patch(*p) for p in patches]

    @staticmethod
    def basis(lat0, lon0):
        c = unit(lat0, lon0)
        east = np.array([-math.sin(math.radians(lon0)), math.cos(math.radians(lon0)), 0.0])
        north = np.cross(c, east)
        return c, east, north

    def render(self, lat0, lon0, rpx, cx=W / 2, cy=H / 2):
        c, e, n = self.basis(lat0, lon0)
        x = (self.xs - cx) / rpx
        y = -(self.ys - cy) / rpx
        r2 = x * x + y * y
        inside = r2 <= 1
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        P2 = np.clip(x * e[2] + y * n[2] + z * c[2], -1, 1)
        P0 = x * e[0] + y * n[0] + z * c[0]
        P1 = x * e[1] + y * n[1] + z * c[1]
        lat = np.degrees(np.arcsin(P2))
        lon = np.degrees(np.arctan2(P1, P0))
        mx = ((lon + 180) / 360 * self.TW - 0.5).astype(np.float32)
        my = ((90 - lat) / 180 * self.TH - 0.5).astype(np.float32)
        img = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32) / 255
        for tex, (la0, la1, lo0, lo1) in self.patches:
            th, tw = tex.shape[:2]
            if lat.max() < la0 or lat.min() > la1:
                continue
            px = ((lon - lo0) / (lo1 - lo0) * tw - 0.5).astype(np.float32)
            py = ((la1 - lat) / (la1 - la0) * th - 0.5).astype(np.float32)
            wgt = np.clip(np.minimum(np.minimum(px, tw - 1 - px), np.minimum(py, th - 1 - py)) / 40.0, 0, 1)
            if wgt.max() <= 0:
                continue
            ptex = cv2.remap(tex, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
            img = img * (1 - wgt[..., None]) + ptex * wgt[..., None]
        img = img * (0.85 + 0.25 * z[..., None]) * inside[..., None]
        rr = np.sqrt(r2)
        halo = np.exp(-np.maximum(rr - 1, 0) * rpx / 22) * (rr > 0.985)
        img += (halo + np.clip(1 - z, 0, 1) ** 6 * 0.6 * inside)[..., None] * np.array([0.10, 0.22, 0.45], np.float32) * 0.6
        return img

    def project(self, lat, lon, lat0, lon0, rpx, cx=W / 2, cy=H / 2):
        c, e, n = self.basis(lat0, lon0)
        P = unit(np.asarray(lat, float), np.asarray(lon, float))
        return cx + (P @ e) * rpx, cy - (P @ n) * rpx, P @ c


class Layer:
    """Emission (fires, sightlines) for one frame, then bloom + tonemap over a base image."""
    SPR = {s: sprite(s) for s in (0.9, 1.3, 2.0, 3.0, 5.0, 8.0, 14.0)}

    def __init__(self):
        self.E = np.zeros((H, W, 3), np.float32)
        self.L = np.zeros((H, W), np.float32)
        self.Lc = []

    def blit(self, x, y, sig, amp, col):
        spr = self.SPR[sig]
        h = spr.shape[0] // 2
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - h), min(H, yi + h + 1), max(0, xi - h), min(W, xi + h + 1)
        if y1 > y0 and x1 > x0:
            self.E[y0:y1, x0:x1] += spr[y0 - (yi - h):y1 - (yi - h), x0 - (xi - h):x1 - (xi - h), None] * (np.asarray(col, np.float32) * amp)

    def fire(self, x, y, amp, scale=1.0, col=FIRE):
        self.blit(x, y, 0.9, amp * 1.5, CORE)
        self.blit(x, y, 2.0 if scale < 1.3 else 3.0, amp * 0.8, col)
        self.blit(x, y, 5.0 if scale < 1.3 else 8.0, amp * 0.22, col)

    def line(self, p0, p1, val=0.55, th=1):
        cv2.line(self.L, (int(p0[0] * 16), int(p0[1] * 16)), (int(p1[0] * 16), int(p1[1] * 16)), float(val), th, cv2.LINE_AA, shift=4)

    def compose(self, img, exposure=1.15, line_col=FIRE):
        E = self.E + self.L[..., None] * line_col * 0.8
        small = cv2.resize(E, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3) * 0.6 + cv2.GaussianBlur(small, (0, 0), 10) * 0.5, (W, H))
        out = 1 - np.exp(-(img + E + bloom) * exposure)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
