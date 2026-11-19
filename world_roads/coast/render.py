"""Render "Every shore on Earth at light speed": a fish of light swims every coastline, lake shore and both banks
of every river, keeping the land on its right.

    python render.py prepare            # raster layers + the fish's path from network.npz and tour.npz
    python render.py still 0 10 40 ...  # world-view stills at these light-times (s)

Like the roads film (../render.py): every lit pixel is sorted by the time the fish reaches it, so turning the
map gold is an incremental scatter-add per frame. Coasts and lake shores turn gold in one pass; a river turns
pale gold when one bank is done and full gold when the fish comes back down the other.
"""
import os, sys, json, math, time
import numpy as np
import numba as nb
import cv2
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cproj import W, H, RES, to_px, LON0  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
C_KMS = 299_792.458
SS = 2                                    # raster supersampling
K_COAST, K_LAKE, K_RIVER, K_LINK = 0, 1, 2, 3
INTEN = np.array([1.0, 0.8, 0.6, 0.6], np.float32)       # coast, lake shore, river, link
THICK = np.array([2, 2, 1, 1], np.int64)                  # line width on the supersampled canvas
LAYERS = os.path.join(DATA, f"layers_r{RES}.npz")


# --------------------------------------------------------------------------- rasterising
@nb.njit(cache=True)
def raster_segments(px, py, segj, t1, t2, inten, thick, W2, H2, I2, T1, T2):
    """thin DDA lines on the supersampled canvas: max intensity, earliest first-pass and both-pass time"""
    for k in range(len(segj)):
        j = segj[k]
        x0, y0, x1, y1 = px[j], py[j], px[j + 1], py[j + 1]
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) > W2 * 0.5:
            continue                                    # crosses the map's edge (the Bering Strait cut)
        n = int(max(abs(dx), abs(dy))) + 1
        th = thick[k]
        for i in range(n + 1):
            x = int(x0 + dx * i / n + 0.5)
            y = int(y0 + dy * i / n + 0.5)
            for oy in range(th):
                for ox in range(th):
                    xx, yy = x + ox - th // 2, y + oy - th // 2
                    if 0 <= xx < W2 and 0 <= yy < H2:
                        if inten[k] > I2[yy, xx]:
                            I2[yy, xx] = inten[k]
                        if t1[k] < T1[yy, xx]:
                            T1[yy, xx] = t1[k]
                        if t2[k] < T2[yy, xx]:
                            T2[yy, xx] = t2[k]


@nb.njit(cache=True)
def decimate_path(x, y, t, min_px):
    """keep a path point when it has moved min_px from the last kept one, or when the next step is a leap
    (no time passes over a long distance)"""
    n = len(x)
    keep = np.zeros(n, np.bool_)
    keep[0] = True
    lx, ly = x[0], y[0]
    for i in range(1, n):
        d = math.hypot(x[i] - lx, y[i] - ly)
        leap_next = i + 1 < n and (t[i + 1] - t[i]) * 299792.458 < 1e-6 and math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) > 3
        leap_here = (t[i] - t[i - 1]) * 299792.458 < 1e-6 and math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) > 3
        if d >= min_px or leap_next or leap_here or i == n - 1:
            keep[i] = True
            lx, ly = x[i], y[i]
    return keep


def land_rings_px(ss):
    """every coastline ring as a pixel polygon (rings crossing the map's edge at 170 W split there)"""
    import shapely
    cz = np.load(os.path.join(DATA, "coast_rings.npz"))
    xy, off = cz["xy"], cz["off"]
    out = []
    cut = LON0 - 180.0                                  # -170
    for r in range(len(off) - 1):
        c = xy[off[r]:off[r + 1]]
        if len(c) < 4:
            continue
        lo = c[:, 0]
        if c[:, 1].min() <= -89.99:
            out.append(antarctica_px(c, cut, ss))
            continue
        if lo.min() < cut < lo.max() and (lo.min() < cut - 1e-9):
            poly = shapely.Polygon(c)
            parts = [shapely.clip_by_rect(poly, -181, -91, cut, 91), shapely.clip_by_rect(poly, cut, -91, 181, 91)]
            for p in parts:
                for g in shapely.get_parts(p):
                    if g.geom_type == "Polygon":
                        cc = shapely.get_coordinates(g.exterior)
                        X, Y = to_px(cc[:, 0] + np.where(cc[:, 0] <= cut + 1e-9, -1e-7, 1e-7), cc[:, 1], ss)
                        out.append(np.c_[X, Y])
            continue
        X, Y = to_px(lo, c[:, 1], ss)
        if X.max() - X.min() < 1.0 and Y.max() - Y.min() < 1.0:
            continue
        out.append(np.c_[X, Y])
    holes = [r for r in np.nonzero(cz["hole"])[0]]
    hole_px = []
    for r in holes:
        c = xy[off[r]:off[r + 1]]
        X, Y = to_px(c[:, 0], c[:, 1], ss)
        hole_px.append(np.c_[X, Y])
    return out, hole_px


def antarctica_px(c, cut, ss):
    """Antarctica reaches the pole: its coast as one chain from the map's left edge to its right edge, closed
    along the edge and the pole line"""
    real = ~((np.abs(c[:, 0]) == 180.0) | (c[:, 1] <= -89.99))
    ch = c[real]
    lam = ((ch[:, 0] - LON0 + 180.0) % 360.0) - 180.0
    k = int(np.argmax(np.abs(np.diff(lam)))) + 1          # where the chain crosses the map's edge
    ch = np.r_[ch[k:], ch[:k]]
    lam = ((ch[:, 0] - LON0 + 180.0) % 360.0) - 180.0
    if lam[0] > lam[-1]:
        ch = ch[::-1]
    lat0, lat1 = ch[0, 1], ch[-1, 1]
    n = 200
    right = np.c_[np.full(n, cut - 1e-7), np.linspace(lat1, -90, n)]
    pole = np.c_[cut - 1e-7 - np.linspace(0, 360 - 2e-7, n), np.full(n, -90.0)]
    left = np.c_[np.full(n, cut + 1e-7), np.linspace(-90, lat0, n)]
    P = np.r_[ch, right, pole, left]
    X, Y = to_px(P[:, 0], P[:, 1], ss)
    return np.c_[X, Y]


def fill(canvas, polys, value):
    pts = [np.rint(p * 4).astype(np.int32) for p in polys if len(p) >= 3]
    for i in range(0, len(pts), 20000):
        cv2.fillPoly(canvas, pts[i:i + 20000], value, cv2.LINE_AA, shift=2)


# --------------------------------------------------------------------------- prepare
def prepare():
    t0 = time.time()
    z = np.load(os.path.join(DATA, "network.npz"))
    gxy, goff, kind = z["gxy"], z["goff"], z["kind"]
    tz = np.load(os.path.join(DATA, "tour.npz"))
    tf, tb = tz["tf"] / C_KMS, tz["tb"] / C_KMS                    # seconds
    total_s = float(tz["total"]) / C_KMS
    m = len(kind)
    W2, H2 = W * SS, H * SS
    px, py = to_px(gxy[:, 0], gxy[:, 1], SS)
    px, py = px.astype(np.float32), py.astype(np.float32)
    ekind = np.repeat(kind, np.diff(goff))
    last = np.zeros(len(gxy), bool)
    last[goff[1:] - 1] = True
    segj = np.nonzero(~last)[0]
    fwd, bwd = tf[segj], tb[segj + 1]
    one = (ekind[segj] == K_COAST) | (ekind[segj] == K_LAKE)
    t1 = np.minimum(fwd, bwd)
    t2 = np.where(one, t1, np.maximum(fwd, bwd))
    sk = ekind[segj]
    I2 = np.zeros((H2, W2), np.float32)
    T1 = np.full((H2, W2), np.inf, np.float32)
    T2 = np.full((H2, W2), np.inf, np.float32)
    # draw dim classes first is not needed: max intensity, min times
    raster_segments(px, py, segj, t1.astype(np.float32), t2.astype(np.float32), INTEN[sk], THICK[sk], W2, H2, I2, T1, T2)
    print(f"rasterised {len(segj):,} segments ({time.time() - t0:.0f}s)", flush=True)
    # swims across open sea: a faint line, gold once swum (from the tour's joins)
    J_type, J_xy, Je, Jdir, Jt = tz["J_type"], tz["J_xy"], tz["Je"], tz["Jdir"], tz["Jt"] / C_KMS
    first_t = np.full(len(J_type), np.inf)
    np.minimum.at(first_t, Je, Jt)
    sw = np.nonzero((J_type == 1) & (tz["J_km"] > 40))[0]
    S_I = np.zeros((H2, W2), np.float32)
    S_T = np.full((H2, W2), np.inf, np.float32)
    ax, ay = to_px(J_xy[sw, 0], J_xy[sw, 1], SS)
    bx, by = to_px(J_xy[sw, 2], J_xy[sw, 3], SS)
    sp_x = np.c_[ax, bx].ravel().astype(np.float32)
    sp_y = np.c_[ay, by].ravel().astype(np.float32)
    sj = np.arange(0, 2 * len(sw), 2)
    ts = first_t[sw].astype(np.float32)
    raster_segments(sp_x, sp_y, sj, ts, ts, np.full(len(sw), 0.22, np.float32), np.ones(len(sw), np.int64), W2, H2, S_I, S_T, S_T.copy())
    # lit pixels -> two streams on the 1x canvas: at T1 the cyan goes and gold comes (half for a river with
    # one bank done), at T2 the rest of the gold
    lit = I2 > 0.01
    yy, xx = np.nonzero(lit)
    a1, a2, ii = T1[yy, xx], T2[yy, xx], I2[yy, xx] / (SS * SS)
    p1 = ((yy // SS) * W + (xx // SS)).astype(np.int32)
    half = np.isfinite(a2) & (a2 > a1)
    dg1 = np.where(half, 0.5 * ii, ii)
    s_ok = np.isfinite(a1)
    # stream entries: time, pixel, cyan removed, gold added
    st_t = np.r_[a1[s_ok], a2[half & s_ok]]
    st_p = np.r_[p1[s_ok], p1[half & s_ok]]
    st_c = np.r_[ii[s_ok], np.zeros((half & s_ok).sum(), np.float32)]
    st_g = np.r_[dg1[s_ok], 0.5 * ii[half & s_ok]]
    # swim pixels: gold only
    sy, sx = np.nonzero(S_I > 0)
    st_t = np.r_[st_t, S_T[sy, sx]]
    st_p = np.r_[st_p, ((sy // SS) * W + (sx // SS)).astype(np.int32)]
    st_c = np.r_[st_c, np.zeros(len(sy), np.float32)]
    st_g = np.r_[st_g, S_I[sy, sx] / (SS * SS)]
    o = np.argsort(st_t, kind="stable")
    st_t, st_p, st_c, st_g = st_t[o].astype(np.float32), st_p[o], st_c[o].astype(np.float32), st_g[o].astype(np.float32)
    Itot = np.bincount(p1, weights=ii, minlength=W * H).astype(np.float32)
    unreached = np.bincount(p1[~s_ok], weights=ii[~s_ok], minlength=W * H).astype(np.float32)
    print(f"lit pixels {lit.sum():,}; stream entries {len(st_t):,} ({time.time() - t0:.0f}s)", flush=True)
    del I2, T1, T2, S_I, S_T
    # background: near-black ocean, dark navy land, lakes back to water
    land2 = np.zeros((H2, W2), np.uint8)
    rings, holes = land_rings_px(SS)
    fill(land2, rings, 255)
    fill(land2, holes, 0)
    lakes = lake_px(SS)
    fill(land2, lakes, 0)
    land1 = cv2.resize(land2, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    print(f"land filled ({time.time() - t0:.0f}s)", flush=True)
    yy1, xx1 = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1 - 0.40 * (((xx1 - W / 2) / (W * 0.62)) ** 2 + ((yy1 - H * 0.52) / (H * 0.75)) ** 2)
    ocean = np.array([0.006, 0.014, 0.032], np.float32)
    landc = np.array([0.026, 0.040, 0.070], np.float32)
    bg = ocean[None, None] * np.clip(vign, 0.35, 1)[..., None] + land1[..., None] * (landc - ocean)[None, None]
    # the fish's path: every vertex pass in time order, decimated on screen
    T = np.r_[tf, tb]
    V = np.r_[np.arange(len(tf)), np.arange(len(tb))]
    ok = np.isfinite(T)
    T, V = T[ok], V[ok]
    o = np.argsort(T, kind="stable")
    T, V = T[o], V[o]
    fx, fy = to_px(gxy[V, 0], gxy[V, 1], 1)
    keep = decimate_path(fx, fy, T, 0.3 * RES)
    rx, ry, rt = fx[keep].astype(np.float32), fy[keep].astype(np.float32), T[keep]
    # leaps (instantaneous) for the rings
    lp = np.nonzero(J_type[Je] == 2)[0]
    k = Je[lp]
    a_is_from = Jdir[lp] == 0
    x0, y0 = to_px(np.where(a_is_from, J_xy[k, 0], J_xy[k, 2]), np.where(a_is_from, J_xy[k, 1], J_xy[k, 3]), 1)
    x1, y1 = to_px(np.where(a_is_from, J_xy[k, 2], J_xy[k, 0]), np.where(a_is_from, J_xy[k, 3], J_xy[k, 1]), 1)
    jd = np.hypot(x1 - x0, y1 - y0)
    big = (jd > 10 * RES) & (jd < W / 2)
    np.savez(LAYERS, st_t=st_t, st_p=st_p, st_c=st_c, st_g=st_g, Itot=Itot, unreached=unreached, bg=bg.astype(np.float32),
             rx=rx, ry=ry, rt=rt, j_t=Jt[lp][big], j_x0=x0[big], j_y0=y0[big], j_x1=x1[big], j_y1=y1[big], j_d=jd[big],
             total_s=total_s)
    print(f"path {len(rx):,} points; leaps drawn {big.sum():,} of {len(lp):,}; total {total_s:.3f} s "
          f"({time.time() - t0:.0f}s)", flush=True)


def lake_px(ss):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, "lakes", "part-*.npz"))):
        z = np.load(p, allow_pickle=True)
        xy, off = z["xy"], z["off"]
        for k in range(len(off) - 1):
            c = xy[off[k]:off[k + 1]].astype(np.float64)
            X, Y = to_px(c[:, 0], c[:, 1], ss)
            if X.max() - X.min() > W * ss / 2:
                continue
            out.append(np.c_[X, Y])
    return out


# --------------------------------------------------------------------------- frames
CYAN = np.array([0.14, 0.62, 1.00], np.float32)
GOLD = np.array([1.00, 0.56, 0.13], np.float32) * 0.95
HOT = np.array([1.00, 0.93, 0.80], np.float32)
FLASH_S = 0.35            # newly traced shore glows white-hot this long (light-seconds at real time)
FONT_DIR = "/usr/share/fonts/opentype/inter"


def font(name, size):
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(FONT_DIR, name), int(size * RES))


def gauss(sig, n):
    k = np.arange(n, dtype=np.float32) - n // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig))


def fish_sprite(angle, scale=1.0):
    """a small fish of light pointing along `angle` (radians, screen coords): white-hot body, tail fin, glow"""
    n = int(150 * RES * scale) | 1
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - c
    ca, sa = math.cos(angle), math.sin(angle)
    u = (xx * ca + yy * sa) / (RES * scale)          # along the fish (head at +u)
    v = (-xx * sa + yy * ca) / (RES * scale)
    body = np.exp(-((u - 1.0) / 5.0) ** 2 - (v / 1.9) ** 2)
    core = np.exp(-((u - 2.5) / 2.2) ** 2 - (v / 1.0) ** 2)
    # tail: a forked fin behind the body
    tu = -(u + 5.2)
    tail = np.exp(-((tu - 1.4) / 1.5) ** 2) * (np.exp(-((v - 0.9 * tu - 0.2) / 0.7) ** 2) + np.exp(-((v + 0.9 * tu + 0.2) / 0.7) ** 2))
    tail *= (tu > -0.5)
    glow = np.exp(-(u ** 2 + v ** 2) / (2 * 6.0 ** 2))
    halo = np.exp(-(u ** 2 + v ** 2) / (2 * 15.0 ** 2))
    edge = np.clip((n // 2 - np.hypot(xx, yy)) / (0.18 * n), 0, 1)     # fade to zero before the sprite's edge
    halo *= edge
    glow *= edge
    spr = (core[..., None] * np.array([1.0, 0.98, 0.92]) * 2.4 + body[..., None] * np.array([1.0, 0.88, 0.62]) * 1.5
           + tail[..., None] * np.array([1.0, 0.80, 0.45]) * 1.1 + glow[..., None] * np.array([1.0, 0.70, 0.30]) * 0.55
           + halo[..., None] * np.array([1.0, 0.52, 0.14]) * 0.30)
    return spr.astype(np.float32)


class World:
    """the whole-map view at light-time t (seconds)"""

    def __init__(self):
        L = dict(np.load(LAYERS))
        self.L = L
        self.total = float(L["total_s"])
        self.Cg = np.zeros(W * H, np.float32)          # gold so far
        self.Cc = np.zeros(W * H, np.float32)          # cyan removed so far
        self.k = 0
        self.fish_cache = {}
        self.heading = 0.0
        self.sx = self.sy = None

    def reset_to(self, t):
        L = self.L
        k = int(np.searchsorted(L["st_t"], t, "right"))
        self.Cg = np.bincount(L["st_p"][:k], weights=L["st_g"][:k], minlength=W * H).astype(np.float32)
        self.Cc = np.bincount(L["st_p"][:k], weights=L["st_c"][:k], minlength=W * H).astype(np.float32)
        self.k = k

    def advance(self, t):
        L = self.L
        k = int(np.searchsorted(L["st_t"], t, "right"))
        if k > self.k:
            np.add.at(self.Cg, L["st_p"][self.k:k], L["st_g"][self.k:k])
            np.add.at(self.Cc, L["st_p"][self.k:k], L["st_c"][self.k:k])
            self.k = k
        elif k < self.k:
            self.reset_to(t)

    def fish_at(self, t):
        rt, rx, ry = self.L["rt"], self.L["rx"], self.L["ry"]
        i = int(np.searchsorted(rt, t, "left"))
        i = min(max(i, 1), len(rt) - 1)
        t0, t1 = rt[i - 1], rt[i]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return float(rx[i - 1] + u * (rx[i] - rx[i - 1])), float(ry[i - 1] + u * (ry[i] - ry[i - 1])), i

    def emission(self, t, flash_s=FLASH_S, cyan_level=0.42):
        """linear-light map image (H, W, 3) at time t"""
        L = self.L
        self.advance(t)
        k = self.k
        kf = int(np.searchsorted(L["st_t"], t - flash_s, "right"))
        F = np.zeros(W * H, np.float32)
        if k > kf:
            age = (t - L["st_t"][kf:k]) / flash_s
            np.add.at(F, L["st_p"][kf:k], L["st_g"][kf:k] * (1 - age) ** 2 * 2.4)
        Ic = (L["Itot"] - self.Cc).reshape(H, W)
        Ig = self.Cg.reshape(H, W)
        F2 = F.reshape(H, W)
        E = cv2.merge([cv2.scaleAdd(F2, float(HOT[c]), cv2.addWeighted(Ic, float(CYAN[c] * cyan_level), Ig, float(GOLD[c]), 0))
                       for c in range(3)])
        return E

    def add_fish(self, E, t, trail_s=0.012, scale=1.0, xform=None):
        L = self.L
        x, y, i = self.fish_at(t)
        # heading: smoothed screen velocity over the last moments
        j0 = int(np.searchsorted(L["rt"], t - 0.02, "left"))
        if i - j0 >= 2:
            dx, dy = x - float(np.mean(L["rx"][j0:i])), y - float(np.mean(L["ry"][j0:i]))
            if math.hypot(dx, dy) > 0.5 * RES:
                target = math.atan2(dy, dx)
                d = (target - self.heading + math.pi) % (2 * math.pi) - math.pi
                self.heading += 0.35 * d
        # trail
        i0 = int(np.searchsorted(L["rt"], t - trail_s, "left"))
        if i0 < i:
            sx = np.r_[L["rx"][i0:i], x]
            sy = np.r_[L["ry"][i0:i], y]
            if xform is not None:
                sx, sy = xform(sx, sy)
            st = np.r_[L["rt"][i0:i], t]
            jump = np.r_[False, (np.diff(st) * C_KMS < 1e-6) | (np.abs(np.diff(sx)) > W / 3)]
            tr = np.zeros((H, W), np.uint8)
            pts = np.round(np.c_[sx, sy] * 16).astype(np.int32)
            nb_ = 6
            nseg = len(sx) - 1
            for b in range(nb_):
                a_, b_ = b * nseg // nb_, (b + 1) * nseg // nb_
                seg = [pts[j:j + 2] for j in range(a_, b_) if not jump[j + 1]]
                if seg:
                    cv2.polylines(tr, seg, False, int(50 + 205 * (b + 1) / nb_), max(1, RES), cv2.LINE_AA, shift=4)
            ys, xs = np.nonzero(tr)
            if len(ys):
                E[ys, xs] += tr[ys, xs, None].astype(np.float32) / 255 * np.array([1.0, 0.82, 0.5], np.float32) * 1.3
        if xform is not None:
            x, y = xform(np.array([x]), np.array([y]))
            x, y = float(x[0]), float(y[0])
        key = (round(self.heading * 16) % 101, round(scale * 8))
        if key not in self.fish_cache:
            self.fish_cache[key] = fish_sprite(round(self.heading * 16) / 16, scale)
        add_sprite(E, x, y, self.fish_cache[key])
        return x, y

    def rings(self, E, t, win=0.6):
        L = self.L
        jt = L["j_t"]
        a, b = np.searchsorted(jt, t - win), np.searchsorted(jt, t, "right")
        if b <= a:
            return []
        ring = np.zeros((H, W), np.uint8)
        labels = []
        for j in range(a, b):
            u = (t - jt[j]) / win
            big = min(1.0, L["j_d"][j] / (300 * RES))
            for (cx, cy, base, amp) in ((L["j_x0"][j], L["j_y0"][j], 4, 0.35), (L["j_x1"][j], L["j_y1"][j], 3, 0.8)):
                rad = (base + (10 + 20 * big) * (1 - (1 - u) ** 2)) * RES
                cv2.circle(ring, (int(cx * 16), int(cy * 16)), int(rad * 16), int(255 * amp * (0.5 + 0.5 * big) * (1 - u) ** 1.5),
                           max(1, RES), cv2.LINE_AA, shift=4)
            if L["j_d"][j] > 60 * RES:
                labels.append((L["j_x1"][j], L["j_y1"][j], (1 - u) ** 0.7))
        ys, xs = np.nonzero(ring)
        E[ys, xs] += ring[ys, xs, None].astype(np.float32) / 255 * np.array([0.85, 0.92, 1.0], np.float32) * 0.9
        return labels[-1:]


def add_sprite(D, x, y, spr):
    n = spr.shape[0]; h = n // 2
    xi, yi = int(round(x)), int(round(y))
    x0, y0 = xi - h, yi - h
    Hh, Ww = D.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(Ww, x0 + n), min(Hh, y0 + n)
    if dx1 > dx0 and dy1 > dy0:
        D[dy0:dy1, dx0:dx1] += spr[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]


def finish(img_lin, bg=None, exposure=1.15):
    """bloom + filmic tone map -> float RGB 0..1"""
    em = img_lin
    Hh, Ww = em.shape[:2]
    small = cv2.resize(em, (Ww // 2, Hh // 2), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(small, (0, 0), 2.2 * RES)
    tiny = cv2.resize(small, (Ww // 8, Hh // 8), interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), 3.0), (Ww // 2, Hh // 2), interpolation=cv2.INTER_LINEAR)
    bloom = cv2.resize(cv2.addWeighted(b1, 0.5, b2, 0.6, 0), (Ww, Hh), interpolation=cv2.INTER_LINEAR)
    img = cv2.add(em, bloom)
    if bg is not None:
        img = cv2.add(img, bg)
    cv2.multiply(img, -exposure, dst=img)
    cv2.exp(img, dst=img)
    cv2.subtract((1.0, 1.0, 1.0, 0.0), img, dst=img)
    return img


def overlay(img, rgba):
    """alpha-composite a PIL RGBA image (only its bounding box) onto float img"""
    bb = rgba.getbbox()
    if not bb:
        return img
    x0, y0, x1, y1 = bb
    a = np.asarray(rgba.crop(bb)).astype(np.float32) / 255
    reg = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = reg * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]
    return img


def to8(img):
    return cv2.convertScaleAbs(img, alpha=255.0, beta=0.5)


def world_still(times):
    from PIL import Image, ImageDraw
    os.makedirs(OUT, exist_ok=True)
    Wd = World()
    bg = Wd.L["bg"]
    for t in times:
        Wd.reset_to(t - 0.05)
        for tt in np.linspace(t - 0.05, t, 4):
            Wd.heading = Wd.heading
            E = Wd.emission(float(tt))
            Wd.add_fish(E, float(tt))
        Wd.rings(E, t)
        img = finish(E, bg)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((56 * RES, 34 * RES), "Every shore on Earth at light speed", font=font("Inter-SemiBold.otf", 42), fill=(236, 242, 250, 255))
        d.text((58 * RES, 88 * RES), "One fish of light · land always on its right", font=font("Inter-Regular.otf", 23), fill=(150, 176, 205, 255))
        right = W - 58 * RES
        mm, ss = int(t // 60), t % 60
        d.text((right, 30 * RES), "ELAPSED", font=font("Inter-Medium.otf", 15), fill=(130, 150, 175, 255), anchor="ra")
        d.text((right, 48 * RES), f"{mm}:{ss:05.2f}", font=font("InterDisplay-Medium.otf", 58), fill=(246, 236, 214, 255), anchor="ra")
        img = overlay(img, im)
        path = os.path.join(OUT, f"world_{t:07.3f}.png")
        cv2.imwrite(path, to8(img)[..., ::-1])
        print(path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "still":
        world_still([float(x) for x in sys.argv[2:]])
