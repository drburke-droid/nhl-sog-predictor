"""Map renderers for the Roomba-photon video (1920x1080): Calgary with the photon's trail, and the USA.

Both show the share of each pixel's road length driven so far (gold) over a dim map of every road.
"""
import os, sys, json, math
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
W, H = 1920, 1080
BG = np.array([5, 8, 15], np.float32)
DIM = np.array([38, 52, 78], np.float32)
GOLD = np.array([255, 190, 90], np.float32)
HOT = np.array([255, 245, 220], np.float32)


def splat_segments(x0, y0, x1, y1, weights=None, max_px=60):
    """Pixel samples along segments (~1 per pixel): returns (pixel x, pixel y, owner index)."""
    L = np.hypot(x1 - x0, y1 - y0)
    n = np.clip(np.ceil(L).astype(np.int64), 1, max_px)
    owner = np.repeat(np.arange(len(x0)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    f = (k + 0.5) / np.repeat(n, n)
    xs = x0[owner] + (x1 - x0)[owner] * f
    ys = y0[owner] + (y1 - y0)[owner] * f
    return np.round(xs).astype(np.int64), np.round(ys).astype(np.int64), owner


class RoadMap:
    """Roads as pixel samples; each sample carries its road's length share and first-driven distance."""

    def __init__(self, x0, y0, x1, y1, km, first, w=W, h=H, max_px=60):
        px, py, own = splat_segments(x0, y0, x1, y1, max_px=max_px)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px, py, own = px[ok], py[ok], own[ok]
        cnt = np.bincount(own, minlength=len(km))
        wgt = km[own] / np.maximum(cnt[own], 1)
        self.w, self.h = w, h
        pix = py * w + px
        self.total = np.bincount(pix, weights=wgt, minlength=w * h).reshape(h, w)
        o = np.argsort(first[own], kind="stable")
        self.pix, self.wgt, self.t = pix[o], wgt[o], first[own][o].astype(np.float64)
        self.done = np.zeros(w * h)
        self.i = 0
        t = self.total[self.total > 0]
        self.norm = np.percentile(t, 99.5) if len(t) else 1.0

    def advance(self, D):
        j = int(np.searchsorted(self.t, D, side="right"))
        new = self.pix[self.i:j]
        if j > self.i:
            self.done += np.bincount(new, weights=self.wgt[self.i:j], minlength=self.w * self.h)
            self.i = j
        return new

    def reset(self):
        self.done[:] = 0
        self.i = 0

    def image(self, glow_pix=None, dim=1.0, gamma=0.5):
        dens = np.clip(self.total / self.norm, 0, 1) ** gamma
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(self.total > 0, self.done.reshape(self.h, self.w) / self.total, 0.0)
        img = BG + dens[..., None] * (DIM * dim * (1 - frac[..., None]) + GOLD * frac[..., None])
        return img


def tonemap(img, glow=None):
    """img float (0..255 scale); glow: extra emission, bloomed."""
    if glow is not None and glow.max() > 0:
        b = cv2.GaussianBlur(glow, (0, 0), 2.0) * 0.8 + cv2.GaussianBlur(glow, (0, 0), 7.0) * 0.6
        img = img + glow + b
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ Calgary
class Calgary:
    KM_PX = 25.5                                    # pixels per km

    def __init__(self, seed=4, cx=0.0, cy=0.0, ox=W / 2, oy=H / 2):
        g = np.load(os.path.join(DATA, "calgary_graph.npz"))
        self.nx, self.ny, self.ea, self.eb, self.elen = g["nx"], g["ny"], g["ea"], g["eb"], g["elen"]
        self.start = int(g["start"])
        self.first = np.load(os.path.join(DATA, f"calgary_first_{seed}.npy")).astype(np.float64)
        self.path = np.load(os.path.join(DATA, f"calgary_path_{seed}.npy"))
        self.cum = np.cumsum(self.elen[self.path])                    # km driven after each step
        self.cx, self.cy, self.ox, self.oy = cx, cy, ox, oy
        X, Y = self.to_px(self.nx, self.ny)
        self.X, self.Y = X, Y
        self.map = RoadMap(X[self.ea], Y[self.ea], X[self.eb], Y[self.eb], self.elen, self.first)
        self.mx, self.my = (X[self.ea] + X[self.eb]) / 2, (Y[self.ea] + Y[self.eb]) / 2
        self.total_km = float(self.cum[-1])
        self.trail = np.zeros((H, W), np.float32)
        self.k = 0

    def to_px(self, x, y):
        return self.ox + (np.asarray(x) - self.cx) * self.KM_PX, self.oy - (np.asarray(y) - self.cy) * self.KM_PX

    def step_at(self, D):
        return int(np.searchsorted(self.cum, D, side="right"))

    MAX_TRAIL = 25_000                              # steps drawn per frame (the most recent ones)

    def frame(self, D, trail_decay=0.72, head=True):
        """The map after D km of driving; the trail keeps the last frames' roads glowing."""
        k1 = min(self.step_at(D), len(self.path) - 1)
        self.map.advance(D)
        img = self.map.image()
        self.trail *= trail_decay
        if k1 > self.k:
            seg = self.path[max(self.k, k1 + 1 - self.MAX_TRAIL):k1 + 1]
            px, py, own = splat_segments(self.X[self.ea[seg]], self.Y[self.ea[seg]], self.X[self.eb[seg]], self.Y[self.eb[seg]], max_px=40)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            self.trail += np.bincount(py[ok] * W + px[ok], minlength=W * H).reshape(H, W).astype(np.float32) * 0.35
            self.k = k1
        glow = np.minimum(self.trail, 1.0)[..., None] * (HOT * 0.45)
        if head:
            e = self.path[k1]
            hx, hy = int(self.mx[e]), int(self.my[e])
            if 0 <= hx < W and 0 <= hy < H:
                g = np.zeros((H, W), np.float32)
                g[hy, hx] = 1.0
                g = cv2.GaussianBlur(g, (0, 0), 2.0) * 60 + cv2.GaussianBlur(g, (0, 0), 7.0) * 250
                glow = glow + np.minimum(g, 1.6)[..., None] * HOT
        return tonemap(img, glow)

    def head_xy(self, D):
        e = self.path[min(self.step_at(D), len(self.path) - 1)]
        return float(self.mx[e]), float(self.my[e])


# ------------------------------------------------------------------ USA
def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5, R=6371.0):
    lon, lat = np.radians(lon), np.radians(lat)
    l0, f0, f1, f2 = map(math.radians, (lon0, lat0, p1, p2))
    n = (math.sin(f1) + math.sin(f2)) / 2
    C = math.cos(f1) ** 2 + 2 * n * math.sin(f1)
    rho0 = R * math.sqrt(C - 2 * n * math.sin(f0)) / n
    rho = R * np.sqrt(C - 2 * n * np.sin(lat)) / n
    th = n * (lon - l0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)              # km, x east, y north


class USA:
    def __init__(self, seed=0, scale=0.36, ox=W / 2, oy=H / 2 + 20):
        lon, lat = np.load(os.path.join(DATA, "usa_lon.npy")), np.load(os.path.join(DATA, "usa_lat.npy"))
        ea, eb = np.load(os.path.join(DATA, "usa_ea.npy")), np.load(os.path.join(DATA, "usa_eb.npy"))
        km = np.load(os.path.join(DATA, "usa_elen.npy")).astype(np.float64)
        x, y = albers(lon, lat)
        self.scale, self.ox, self.oy = scale, ox, oy
        X, Y = ox + x * scale, oy - y * scale
        self.X, self.Y = X.astype(np.float32), Y.astype(np.float32)
        first = np.load(os.path.join(DATA, f"usa_first_{seed}.npy")).astype(np.float64)
        tp = os.path.join(DATA, f"usa_tail_{seed}.npz")
        if os.path.exists(tp):                         # float32 can't order the last few hundred km: use the exact replay
            tl = np.load(tp)
            first[tl["road"][tl["new"]]] = tl["km"][tl["new"]]
        mx, my = (X[ea] + X[eb]) / 2, (Y[ea] + Y[eb]) / 2
        self.map = RoadMap(mx, my, mx, my, km, first, max_px=1)       # roads are shorter than a pixel here
        self.first = first
        self.ea, self.eb = ea, eb

    def frame(self, D, flash_decay=0.8):
        i0 = self.map.i
        new = self.map.advance(D)
        img = self.map.image(gamma=0.42)
        if not hasattr(self, "flash"):
            self.flash = np.zeros(W * H, np.float32)
            self.tot = np.maximum(self.map.total.ravel(), 1e-9)
        self.flash *= flash_decay
        if len(new):                                   # flash by the share of each pixel's roads driven this frame
            add = np.bincount(new, weights=self.map.wgt[i0:i0 + len(new)], minlength=W * H) / self.tot
            self.flash += np.minimum(add, 1.0).astype(np.float32) * 3.0
        glow = np.minimum(self.flash.reshape(H, W), 1.0)[..., None] * HOT * 0.55
        return tonemap(img, glow)

    def to_px(self, lon, lat):
        x, y = albers(np.asarray(lon), np.asarray(lat))
        return self.ox + x * self.scale, self.oy - y * self.scale


# ------------------------------------------------------------------ the zoom to one road
class Zoom:
    """Every USA road at any scale, zooming about one target road. Roads are kept nearest-first from the
    target, so the roads that can be on screen are a prefix of the arrays."""

    def __init__(self, target):
        lon, lat = np.load(os.path.join(DATA, "usa_lon.npy")), np.load(os.path.join(DATA, "usa_lat.npy"))
        x, y = albers(lon, lat)
        del lon, lat
        x, y = x.astype(np.float32), y.astype(np.float32)
        ea, eb = np.load(os.path.join(DATA, "usa_ea.npy")), np.load(os.path.join(DATA, "usa_eb.npy"))
        km = np.load(os.path.join(DATA, "usa_elen.npy"))
        self.tgt = (float(x[ea[target]]), float(y[ea[target]]), float(x[eb[target]]), float(y[eb[target]]))
        self.tx, self.ty = (self.tgt[0] + self.tgt[2]) / 2, (self.tgt[1] + self.tgt[3]) / 2
        self.target_km = float(km[target])
        d = np.hypot((x[ea] + x[eb]) / 2 - self.tx, (y[ea] + y[eb]) / 2 - self.ty)
        d[target] = np.inf                                      # drawn on its own
        o = np.argsort(d)[:-1]
        self.d = d[o]
        self.ax, self.ay, self.bx, self.by = x[ea[o]], y[ea[o]], x[eb[o]], y[eb[o]]
        self.km = km[o].astype(np.float32)
        self.norm = None

    def screen(self, s, px, py):
        """target at screen (px, py), s px/km: world -> screen"""
        return lambda X, Y: (px + (X - self.tx) * s, py - (Y - self.ty) * s)

    def density(self, s, px, py):
        corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], np.float64)
        r = np.hypot(corners[:, 0] - px, corners[:, 1] - py).max() / s + 1.0
        k = int(np.searchsorted(self.d, r))
        f = self.screen(s, px, py)
        x0, y0 = f(self.ax[:k], self.ay[:k])
        x1, y1 = f(self.bx[:k], self.by[:k])
        km = self.km[:k]
        if s < 2.0:                                             # roads shorter than a pixel: bin by midpoint
            qx = np.round((x0 + x1) / 2).astype(np.int64); qy = np.round((y0 + y1) / 2).astype(np.int64)
            wgt = km
        else:
            qx, qy, own = splat_segments(x0, y0, x1, y1, max_px=400)
            cnt = np.bincount(own, minlength=k)
            wgt = km[own] / np.maximum(cnt[own], 1)
        ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
        tot = np.bincount(qy[ok] * W + qx[ok], weights=wgt[ok], minlength=W * H).reshape(H, W)
        nz = tot[tot > 0]
        n = np.percentile(nz, 99.5) if len(nz) else 1.0
        self.norm = n if self.norm is None else math.exp(0.7 * math.log(self.norm) + 0.3 * math.log(n))
        dens = np.clip(tot / self.norm, 0, 1) ** 0.42
        if s > 40:                                              # thicker roads close up
            kk = 1 + 2 * int(round(min(3.0, math.log2(s / 40))))
            dens = cv2.dilate(dens.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        return dens, f
