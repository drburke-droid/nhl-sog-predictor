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

    def image(self, glow_pix=None, dim=1.0, gamma=0.5, fgamma=1.0):
        """fgamma > 1 keeps a pixel blue until nearly all its roads are driven (shows the last pockets)"""
        dens = np.clip(self.total / self.norm, 0, 1) ** gamma
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(self.total > 0, self.done.reshape(self.h, self.w) / self.total, 0.0) ** fgamma
        img = BG + dens[..., None] * (DIM * dim * (1 - frac[..., None]) + GOLD * frac[..., None])
        return img


def tonemap(img, glow=None):
    """img float (0..255 scale); glow: extra emission, bloomed."""
    if glow is not None and glow.max() > 0:
        b = cv2.GaussianBlur(glow, (0, 0), 2.0) * 0.8 + cv2.GaussianBlur(glow, (0, 0), 7.0) * 0.6
        img = img + glow + b
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ the Roomba
_SPRITES = {}


def roomba_sprite(diam, angle):
    """A top-down Roomba, bumper facing `angle` (radians, screen coordinates: 0 = right, y down). RGBA."""
    key = (int(diam), int(round(math.degrees(angle) / 6)) % 60)
    if key in _SPRITES:
        return _SPRITES[key]
    from PIL import Image, ImageDraw
    S = 4 * int(diam) + 8
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, R = S / 2, 2 * int(diam)
    d.ellipse([c - R, c - R, c + R, c + R], fill=(58, 62, 72, 255), outline=(196, 202, 214, 255), width=max(2, R // 10))
    d.arc([c - R, c - R, c + R, c + R], 270 - 75, 270 + 75, fill=(120, 128, 142, 255), width=max(3, R // 5))   # bumper (front = up)
    r2 = R * 0.62
    d.ellipse([c - r2, c - r2, c + r2, c + r2], fill=(34, 37, 44, 255), outline=(90, 96, 108, 255), width=max(1, R // 16))
    r3 = R * 0.17
    d.ellipse([c - r3, c - r3, c + r3, c + r3], fill=(255, 205, 120, 255))                        # the button, lit
    w = R * 0.22
    d.rounded_rectangle([c - w, c - R * 0.9, c + w, c - R * 0.74], radius=max(1, R // 20), fill=(30, 32, 38, 255))   # sensor window
    im = im.rotate(-math.degrees(angle) - 90, resample=Image.BICUBIC, expand=False)   # front from up to `angle`
    im = im.resize((S // 4, S // 4), Image.LANCZOS)
    _SPRITES[key] = im
    return im


def draw_roomba(im, x, y, angle, diam=26):
    """paste a Roomba sprite centred at (x, y) on a PIL image"""
    sp = roomba_sprite(diam, angle)
    im.alpha_composite(sp, (int(round(x - sp.width / 2)), int(round(y - sp.height / 2)))) if im.mode == "RGBA" else \
        im.paste(sp, (int(round(x - sp.width / 2)), int(round(y - sp.height / 2))), sp)


# ------------------------------------------------------------------ a city
class City:
    """One city's streets and one walk on them: coverage, a fading trail of the last frames' driving, the
    Roomba's position and heading, and how many blocks are still undriven."""
    MAX_TRAIL = 25_000                              # steps drawn per frame (the most recent ones)

    def __init__(self, prefix, seed, km_px, cx=0.0, cy=0.0, ox=W / 2, oy=H / 2, w=W, h=H):
        import cover as CV
        g = np.load(os.path.join(DATA, f"{prefix}_graph.npz"))
        self.nx, self.ny, self.ea, self.eb, self.elen = g["nx"], g["ny"], g["ea"], g["eb"], g["elen"]
        self.start = int(g["start"])
        self.first = np.load(os.path.join(DATA, f"{prefix}_first_{seed}.npy")).astype(np.float64)
        self.path = np.load(os.path.join(DATA, f"{prefix}_path_{seed}.npy"))
        self.cum = np.cumsum(self.elen[self.path])                    # km driven after each step
        self.km_px, self.cx, self.cy, self.ox, self.oy, self.w, self.h = km_px, cx, cy, ox, oy, w, h
        X, Y = self.to_px(self.nx, self.ny)
        self.X, self.Y = X, Y
        self.map = RoadMap(X[self.ea], Y[self.ea], X[self.eb], Y[self.eb], self.elen, self.first, w=w, h=h)
        self.mx, self.my = (X[self.ea] + X[self.eb]) / 2, (Y[self.ea] + Y[self.eb]) / 2
        self.total_km = float(self.cum[-1])
        self.blk, self.nblk = CV.blocks(len(self.nx), self.ea, self.eb, self.elen)
        self.block_t = np.sort(CV.block_max(self.blk, self.first, self.nblk))
        self.trail = np.zeros((h, w), np.float32)
        self.k = 0

    def to_px(self, x, y):
        return self.ox + (np.asarray(x) - self.cx) * self.km_px, self.oy - (np.asarray(y) - self.cy) * self.km_px

    def step_at(self, D):
        return int(np.searchsorted(self.cum, D, side="right"))

    def blocks_left(self, D):
        return self.nblk - int(np.searchsorted(self.block_t, D, side="right"))

    def frame(self, D, trail_decay=0.72, head=True):
        """The map after D km of driving; the trail keeps the last frames' roads glowing."""
        W_, H_ = self.w, self.h
        k1 = min(self.step_at(D), len(self.path) - 1)
        self.map.advance(D)
        img = self.map.image()
        self.trail *= trail_decay
        if k1 > self.k:
            seg = self.path[max(self.k, k1 + 1 - self.MAX_TRAIL):k1 + 1]
            px, py, own = splat_segments(self.X[self.ea[seg]], self.Y[self.ea[seg]], self.X[self.eb[seg]], self.Y[self.eb[seg]], max_px=40)
            ok = (px >= 0) & (px < W_) & (py >= 0) & (py < H_)
            self.trail += np.bincount(py[ok] * W_ + px[ok], minlength=W_ * H_).reshape(H_, W_).astype(np.float32) * 0.35
            self.k = k1
        glow = np.minimum(self.trail, 1.0)[..., None] * (HOT * 0.45)
        if head:                                       # a soft halo; the Roomba itself is drawn on top
            hx, hy, _ = self.head(D)
            if 0 <= hx < W_ and 0 <= hy < H_:
                g = np.zeros((H_, W_), np.float32)
                g[int(hy), int(hx)] = 1.0
                g = cv2.GaussianBlur(g, (0, 0), 9.0) * 500
                glow = glow + np.minimum(g, 1.0)[..., None] * HOT * 0.8
        return tonemap(img, glow)

    def head(self, D):
        """screen position and heading of the Roomba after D km (along its current segment)"""
        k = min(self.step_at(D), len(self.path) - 1)
        e = self.path[k]
        a, b = self.ea[e], self.eb[e]
        if k > 0:                                      # it entered this segment from the end it shares with the last one
            p = self.path[k - 1]
            if a not in (self.ea[p], self.eb[p]):
                a, b = b, a
        f = 1.0 - (self.cum[k] - D) / max(self.elen[e], 1e-9)
        f = min(max(f, 0.0), 1.0)
        x = self.X[a] + (self.X[b] - self.X[a]) * f
        y = self.Y[a] + (self.Y[b] - self.Y[a]) * f
        # heading: over the last ~0.4 km, so it doesn't jitter on short segments
        k0 = max(0, int(np.searchsorted(self.cum, D - 0.4)))
        e0 = self.path[k0]
        x0, y0 = self.mx[e0], self.my[e0]
        ang = math.atan2(y - y0, x - x0) if (x - x0) ** 2 + (y - y0) ** 2 > 1 else math.atan2(self.Y[b] - self.Y[a], self.X[b] - self.X[a])
        return float(x), float(y), ang


def Calgary(seed=31, cx=0.0, cy=0.0, ox=W / 2, oy=H / 2, km_px=25.5, w=W, h=H):
    return City("calgary", seed, km_px, cx, cy, ox, oy, w, h)


# ------------------------------------------------------------------ USA
BLUE = np.array([110, 170, 255], np.float32)
SPARK_FROM = 80_000


def spark(mask):
    """small blue glows on a 0/1 mask"""
    g = cv2.GaussianBlur(mask, (0, 0), 1.5) * (2 * math.pi * 1.5 ** 2) \
        + cv2.GaussianBlur(mask, (0, 0), 6.0) * (2 * math.pi * 6.0 ** 2) * 0.3        # a lone road: peak 1.3
    return np.minimum(g, 1.5)[..., None] * BLUE


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
        import cover as CV
        blk = np.load(os.path.join(DATA, "usa_block.npy"))              # junction-to-junction blocks
        self.nblk = int(blk.max()) + 1
        self.block_t = np.sort(CV.block_max(blk, first, self.nblk))
        del blk
        mx, my = (X[ea] + X[eb]) / 2, (Y[ea] + Y[eb]) / 2
        self.map = RoadMap(mx, my, mx, my, km, first, max_px=1)       # roads are shorter than a pixel here
        self.first = first
        self.ea, self.eb = ea, eb

    def frame(self, D, flash_decay=0.8):
        i0 = self.map.i
        new = self.map.advance(D)
        img = self.map.image(gamma=0.42, dim=1.5, fgamma=4.0)
        if not hasattr(self, "flash"):
            self.flash = np.zeros(W * H, np.float32)
            self.tot = np.maximum(self.map.total.ravel(), 1e-9)
        self.flash *= flash_decay
        if len(new):                                   # flash by the share of each pixel's roads driven this frame
            add = np.bincount(new, weights=self.map.wgt[i0:i0 + len(new)], minlength=W * H) / self.tot
            self.flash += np.minimum(add, 1.0).astype(np.float32) * 3.0
        glow = np.minimum(self.flash.reshape(H, W), 1.0)[..., None] * HOT * 0.55
        left = len(self.map.pix) - self.map.i
        if left < SPARK_FROM:                          # the last roads, too few to see: one blue spark each
            a = min(1.0, (SPARK_FROM - left) / 30_000)
            rem = np.minimum(np.bincount(self.map.pix[self.map.i:], minlength=W * H), 1).reshape(H, W).astype(np.float32)
            glow = glow + spark(rem) * a
        return tonemap(img, glow)

    def blocks_left(self, D):
        return self.nblk - int(np.searchsorted(self.block_t, D, side="right"))

    def to_px(self, lon, lat):
        x, y = albers(np.asarray(lon), np.asarray(lat))
        return self.ox + x * self.scale, self.oy - y * self.scale


# ------------------------------------------------------------------ the zoom to one road
class Zoom:
    """Every USA road at any scale, zooming about one point. Roads are kept nearest-first from that point,
    so the roads that can be on screen are a prefix of the arrays. `late` roads (not yet driven when the zoom
    starts) are left out of the gold map, to be drawn one by one."""

    def __init__(self, late, centre=None):
        lon, lat = np.load(os.path.join(DATA, "usa_lon.npy")), np.load(os.path.join(DATA, "usa_lat.npy"))
        x, y = albers(lon, lat)
        del lon, lat
        self.x, self.y = x.astype(np.float32), y.astype(np.float32)
        del x, y
        x, y = self.x, self.y
        ea, eb = np.load(os.path.join(DATA, "usa_ea.npy")), np.load(os.path.join(DATA, "usa_eb.npy"))
        self.ea, self.eb = ea, eb
        km = np.load(os.path.join(DATA, "usa_elen.npy"))
        self.road_km = km
        late = np.asarray(late)
        if centre is None:
            centre = ((x[ea[late[-1]]] + x[eb[late[-1]]]) / 2, (y[ea[late[-1]]] + y[eb[late[-1]]]) / 2)
        self.tx, self.ty = float(centre[0]), float(centre[1])
        d = np.hypot((x[ea] + x[eb]) / 2 - self.tx, (y[ea] + y[eb]) / 2 - self.ty)
        d[late] = np.inf                                        # drawn on their own
        o = np.argsort(d)[:len(d) - len(np.unique(late))]
        self.d = d[o]
        self.ax, self.ay, self.bx, self.by = x[ea[o]], y[ea[o]], x[eb[o]], y[eb[o]]
        self.km = km[o].astype(np.float32)
        del d, o
        self.norm = None

    def ends(self, roads):
        """world endpoints (x0, y0, x1, y1) of roads"""
        r = np.asarray(roads)
        return self.x[self.ea[r]], self.y[self.ea[r]], self.x[self.eb[r]], self.y[self.eb[r]]

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
