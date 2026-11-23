"""The film: a robot vacuum at the speed of light cleans every floor in New York, then on Earth (1920x1080, 30 fps,
captions and music, no voice).

    python video.py render NAME [workers]     render one section (in parallel chunks, resumable) -> out/video/NAME.mp4
    python video.py still NAME t [t ...]      single frames -> out/video/NAME_t.png
    NAME: hook, room, rules, esb, nyc, earth1, far, earth2, dust, uturn, spotless

Every slow-motion or speed-up is labelled on screen, and the light time shown is the true time.
"""
import os, sys, json, math, time, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
sys.path.insert(0, os.path.join(ROOT, "roomba"))
import style as ST                                                    # noqa: E402
import xray as X                                                      # noqa: E402
import cityfilm as CF                                                 # noqa: E402
import frames as FR                                                   # noqa: E402
import globe as G                                                     # noqa: E402

FR.SS = 1
W, H, FPS = 1920, 1080, 30
OUT = os.path.join(HERE, "out", "video")
INK, INK2, MUTED, GOLD = ST.INK, ST.INK2, ST.MUTED, ST.GOLD
C_MS = 299792458.0
LANE = 0.25
WORLD_FLOOR_M2 = 273e9
WORLD_CLEAN_S = WORLD_FLOOR_M2 / LANE / C_MS                         # 3,642.5 s
WORLD_TRAVEL_S = 410.0
WORLD_T = WORLD_CLEAN_S + WORLD_TRAVEL_S
NYC = json.load(open(os.path.join(HERE, "data", "nyc", "route_stats.json")))
NYS = json.load(open(os.path.join(HERE, "data", "nyc", "stats.json")))
NYC_LL = (-73.95, 40.72)


# ======================================================================== overlays
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, fin=0.35, fout=0.35):
    return ST.fade(t, t0, t1, fin, fout)


def text(d, xy, s, size, col, a=1.0, weight="SemiBold", anchor="la"):
    if a > 0.003:
        d.text(xy, s, font=ST.font(size, weight), fill=tuple(int(c) for c in col) + (int(255 * a),), anchor=anchor)


def captions(fr, t, caps):
    """caps: [(t0, t1, lines)] with lines [(text, size, colour, weight[, t_in])]; a line with t_in fades in later,
    inside a panel already sized for it"""
    act = [(c, fade(t, c[0], c[1])) for c in caps]
    act = [(c, a) for c, a in act if a > 0]
    if not act:
        return fr
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    for (t0, t1, lines), a in act:
        pad, gap = 30, 14
        fs = [ST.font(l[1], l[3]) for l in lines]
        ws = [d.textlength(l[0], font=f) for l, f in zip(lines, fs)]
        hs = [int(l[1] * 1.18) for l in lines]
        bw = int(max(ws) + 2 * pad)
        bh = int(sum(hs) + gap * (len(lines) - 1) + 2 * pad)
        x, y = (W - bw) // 2, H - bh - 70
        d.rounded_rectangle([x, y, x + bw, y + bh], 22, fill=ST.PANEL + (int(205 * a),))
        yy = y + pad
        for l, f, hh in zip(lines, fs, hs):
            la = a * (fade(t, l[4], 1e9, 0.4, 0.4) if len(l) > 4 else 1.0)
            text(d, (x + bw / 2, yy), l[0], l[1], l[2], la, l[3], "ma")
            yy += hh + gap
    return np.asarray(im)


def hud(fr, rows, a=1.0, x0=60, y0=56, title=None):
    if a <= 0:
        return fr
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    y = y0
    if title:
        text(d, (x0, y), title, 26, INK2, a); y += 44
    for lab, val, col, sz in rows:
        text(d, (x0, y), lab, 22, MUTED, a); y += 28
        text(d, (x0, y), val, sz, col, a, "Bold"); y += int(sz * 1.25) + 18
    return np.asarray(im)


def tag(fr, s, a=1.0, x=W - 60, y=56, col=INK2, size=26, anchor="ra", weight="SemiBold"):
    if a <= 0 or not s:
        return fr
    im = Image.fromarray(fr)
    text(ImageDraw.Draw(im, "RGBA"), (x, y), s, size, col, a, weight, anchor)
    return np.asarray(im)


def factor_label(F, kind="SLOW MOTION"):
    if F < 1.6:
        return "REAL TIME"
    e = int(math.floor(math.log10(F)))
    q = 10 ** max(0, e - 1)
    return f"{kind}  ×{int(round(F / q) * q):,}"


def fmt_s(s):
    return FR.fmt_s(max(s, 0.0))


def blend(a, b, u):
    return (a.astype(np.float32) * (1 - u) + b.astype(np.float32) * u + 0.5).astype(np.uint8)


# ======================================================================== sections
class Section:
    name, dur = "section", 1.0
    chunk = 150

    def prepare(self, t):
        """state for a render starting at t (e.g. cleaned buildings so far)"""

    def frame(self, t):
        raise NotImplementedError


def nyc_hud(fr, R, L, a=1.0):
    n, area = R.R.progress(L)
    return hud(fr, [("LIGHT TIME", fmt_s(L), GOLD, 64),
                    ("BUILDINGS CLEANED", f"{n:,} of {NYC['buildings']:,}", INK, 40),
                    ("FLOOR CLEANED", f"{area / 1e6:,.1f} million m²" if area > 1e6 else f"{area:,.0f} m²", INK, 40)],
               a=a, title="NEW YORK CITY · EVERY FLOOR")


def city_cam(s, tx, ty, tz, ox, oy):
    return X.Cam(CF.PSI, CF.EL, s, tx, ty, tz, ox=ox, oy=oy, W=W, H=H, ss=1)


def pchip_cols(times, rows):
    rows = np.asarray(rows, np.float64)
    return [PchipInterpolator(times, rows[:, j]) for j in range(rows.shape[1])]


def slow_map(knots, T, L0, L1=None, t1=None, n_per_s=2000):
    """light time vs video time from slow-motion knots (t, log10 factor); scaled so L(t1) = L1 if given"""
    v = np.linspace(0, T, int(T * n_per_s) + 1)
    f = PchipInterpolator([k[0] for k in knots], [k[1] for k in knots])(v)
    dL = np.concatenate([[0], np.cumsum(np.diff(v) / 10 ** f[:-1])])
    c = 1.0
    if L1 is not None:
        c = (L1 - L0) / np.interp(t1, v, dL)
    return v, L0 + c * dL, 10 ** f / c


class Globe:
    _g = None

    @classmethod
    def get(cls):
        if cls._g is None:
            cls._g = G.Globe()
        return cls._g


def globe_img(g, done, light=None, fire=0.0, glow=1.0, smooth=False, light_size=0.8):
    img, gg = Globe.get().render(g, done, ss=1, fire=fire, glow=glow, smooth=smooth)
    if light is not None:
        u, v, vis = G.ortho_forward(np.array([light[0]]), np.array([light[1]]), gg)
        if vis[0]:
            CF.light_win(img, float(u[0]), float(v[0]), light_size)
    return (np.clip(X.tonemap(img), 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------ 1. the hook
class Hook(Section):
    """One floor, the Empire State Building, Midtown, Manhattan, then off the planet: one continuous pull-back."""
    name, dur = "hook", 15.0
    chunk = 150
    KN = [(0, 9), (2.4, 9), (3.2, 5.0), (3.9, 3.0), (5.0, 2.1), (8.0, 1.25), (11, 0.8), (15, 0.7)]
    G_FINAL = dict(cx=1100, cy=470, r=430, lon0=-62.0, lat0=30.0)

    def __init__(self):
        self.R = CF.CityRenderer()
        R, C = self.R, self.R.C
        self.tower = self._tower()
        self.i14 = [i for i, q in enumerate(R.esb_slabs) if q[1] == self.tower][14]
        self.zf = R.esb_slabs[self.i14][0]
        self.L0 = R.esb_time_at_slab(self.i14, 0.46)
        self.v, self.Lv, self.Fv = slow_map(self.KN, self.dur, self.L0)
        ex, ey = C.cx[R.esb], C.cy[R.esb]
        mx, my = float(C.cx.mean()), float(C.cy.mean())
        au, av, _ = G.ortho_forward(np.array([NYC_LL[0]]), np.array([NYC_LL[1]]), self.G_FINAL)
        self.anchor = (float(au[0]), float(av[0]))
        keys = [(0.0, math.log(2.05), ex, ey, 215, 1180, 500),
                (3.6, math.log(1.9), ex, ey, 215, 1180, 500),
                (8.0, math.log(0.55), ex + 120, ey + 250, 0, 1060, 640),
                (10.8, math.log(0.105), ex + 1500, ey + 2500, 0, 1150, 560),
                (12.6, math.log(0.0009), mx, my, 0, self.anchor[0], self.anchor[1]),
                (14.6, math.log(430 / 6.371e6), mx, my, 0, self.anchor[0], self.anchor[1]),
                (15.0, math.log(430 / 6.371e6), mx, my, 0, self.anchor[0], self.anchor[1])]
        self.cam_f = pchip_cols([k[0] for k in keys], [k[1:] for k in keys])
        # ground plane: oblique 3/4 view -> top-down, north up (for lifting off toward the globe)
        p, e = math.radians(CF.PSI), math.radians(CF.EL)
        A = np.array([[math.cos(p), -math.sin(p)], [-math.sin(p) * math.sin(e), -math.cos(p) * math.sin(e)]])
        self.Mtd = np.array([[1, 0], [0, -1]]) @ np.linalg.inv(A)
        ring = C.rxy[C.ps[self.tower]:C.pe[self.tower]][:-1]
        segs = FR.rows_in(ring)
        self.rows_len = sum(np.linalg.norm(q - p) for p, q in segs)

    def _tower(self):
        C, R = self.R.C, self.R
        ks = R.esb_k
        return ks[np.argmax([(C.pz1[k] - C.pz0[k]) * (C.pnfl[k] > 0) for k in ks])]

    def L(self, t):
        return float(np.interp(t, self.v, self.Lv))

    def F(self, t):
        return float(np.interp(t, self.v, self.Fv))

    def cam(self, t):
        ls, tx, ty, tz, ox, oy = (float(f(t)) for f in self.cam_f)
        return city_cam(math.exp(ls), tx, ty, tz, ox, oy)

    def prepare(self, t):
        self.R.advance(self.L(t))

    def city(self, t, L):
        R, C = self.R, self.R.C
        R.advance(L)
        cam = self.cam(t)
        img = R.frame(cam, L)
        z, i, f = R.esb_state(L)
        pos = None
        if i is not None and cam.s > 1.0:
            _, k, j, _ = R.esb_slabs[i]
            x, y, _ = FR.active_floor(C, k, R.esb_slabs[i][0], f, cam.s, cam, img, dust_n=0, wake=False, rows_gain=0.9)
            pos = (x, y)
        if pos is None:
            x, y, zz = R.frontier_point(L)
            u, v = cam.project(np.array([x]), np.array([y]), np.array([zz]))
            pos = (float(u[0]), float(v[0]))
        CF.light_win(img, pos[0], pos[1], size=CF.interp_log(cam.s, [(2.05, 0.5), (0.5, 0.7), (0.01, 0.9)]))
        lam = ease((t - 11.3) / 1.3)
        if lam > 0:
            T = (1 - lam) * np.eye(2) + lam * self.Mtd
            a = np.array(self.anchor)
            M = np.hstack([T, (a - T @ a)[:, None]])
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=tuple(float(q) for q in X.BG))
        return CF.finish(img), pos

    def globe(self, t, L):
        ls = float(self.cam_f[0](t))
        r = math.exp(ls) * 6.371e6
        lam = ease((t - 12.4) / 2.2)
        lon0 = NYC_LL[0] + lam * (self.G_FINAL["lon0"] - NYC_LL[0])
        lat0 = NYC_LL[1] + lam * (self.G_FINAL["lat0"] - NYC_LL[1])
        u, v, _ = G.ortho_forward(np.array([NYC_LL[0]]), np.array([NYC_LL[1]]), dict(cx=0, cy=0, r=r, lon0=lon0, lat0=lat0))
        g = dict(cx=self.anchor[0] - float(u[0]), cy=self.anchor[1] - float(v[0]), r=r, lon0=lon0, lat0=lat0)
        done = self.R.R.progress(L)[1]
        return globe_img(g, done, light=NYC_LL, smooth=r > 900)

    def inset(self, L, pos):
        R, C = self.R, self.R.C
        dist = 0.46 * self.rows_len + (L - self.L0) * C_MS
        f = min(max(dist / self.rows_len, 0.0), 1.0)
        r = FR.robot_at(C, self.tower, min(f, 0.999))
        Ri, si = 230, 185.0
        ci = X.Cam(CF.PSI, CF.EL, si, r[0], r[1], self.zf, ox=Ri, oy=Ri + 20, W=2 * Ri, H=2 * Ri, ss=1)
        im2 = np.zeros((2 * Ri, 2 * Ri, 3), np.float32) + X.BG
        ix, iy, ih = FR.active_floor(C, self.tower, self.zf, f, si, ci, im2, dust_n=200000, wake=True, rows_gain=0.30)
        ins = CF.finish(im2, bloom=(0.12, 0.06, 0.02))
        ins = FR.paste_rgba(ins, FR.robot_sprite(int(0.35 * si * 1.5), ih), ix, iy)
        return ins

    def frame(self, t):
        L = self.L(t)
        fr, pos = (None, None)
        if t < 12.85:
            fr, pos = self.city(t, L)
        if t > 12.2:
            gfr = self.globe(t, L)
            fr = gfr if fr is None else blend(fr, gfr, ease((t - 12.2) / 0.6))
        a_in = fade(t, 0.25, 3.75, 0.4, 0.45)
        if a_in > 0 and pos is not None:
            m = FR.magnifier(fr, self.inset(L, pos), 470, 470, 230, pos[0], pos[1],
                             "the robot: 35 cm across")
            fr = blend(fr, m, a_in)
        F = self.F(t)
        fr = tag(fr, factor_label(F), fade(t, 0.3, 11.6, 0.4, 0.5))
        fr = tag(fr, "EMPIRE STATE BUILDING · X-RAY VIEW", fade(t, 0.3, 3.9, 0.4, 0.4), y=96, col=MUTED, size=22)
        fr = nyc_hud(fr, self.R, L, fade(t, 4.3, 11.4, 0.5, 0.5))
        fr = captions(fr, t, [(0.35, 3.95, [("This robot vacuum moves at the speed of light.", 46, INK, "SemiBold")]),
                              (4.4, 10.7, [("It's going to clean every floor of every building in New York.", 42, INK, "SemiBold")]),
                              (11.2, 15.2, [("Then every floor on Earth.", 48, INK, "SemiBold"),
                                            ("How long will it take?", 40, GOLD, "SemiBold", 12.7)])])
        return fr


# ------------------------------------------------------------------------ 2. then and now
class Room(Section):
    name, dur = "room", 10.0
    PX = 112.0
    T_LAPSE = 60.0

    def __init__(self):
        import room as RM
        import maps
        self.RM, self.maps = RM, maps
        self.paths = FR.modern_route(RM)
        seg_len = [np.linalg.norm(np.diff(p, axis=0), axis=1) for p in self.paths]
        self.T = (sum(s.sum() for s in seg_len) + sum(np.linalg.norm(self.paths[i + 1][0] - self.paths[i][-1])
                                                      for i in range(len(self.paths) - 1))) / RM.V
        self.rand = RM.simulate(seconds=self.T + 5, dt=0.02)
        # the modern robot's track as one polyline with cumulative distance
        pts = np.concatenate(self.paths)
        self.mpts = pts
        self.breaks = np.cumsum([len(p) for p in self.paths])[:-1]
        self.mcum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        self.ox_l, self.ox_r, self.oy = 130, W - 130 - RM.RW * self.PX, 290
        self.base = self._base()
        self.free = self._free()

    def _base(self):
        RM, PX = self.RM, self.PX
        im = Image.new("RGB", (W, H), (5, 8, 15))
        d = ImageDraw.Draw(im, "RGBA")
        for ox in (self.ox_l, self.ox_r):
            px = lambda x, y: (ox + x * PX, self.oy + y * PX)
            d.rectangle([*px(0, 0), *px(RM.RW, RM.RH)], fill=(18, 22, 32))
        return np.asarray(im).astype(np.float32)

    def _free(self):
        RM, PX = self.RM, self.PX
        out = []
        for ox in (self.ox_l, self.ox_r):
            Wr, Hr = int(RM.RW * PX), int(RM.RH * PX)
            yy, xx = np.mgrid[self.oy:self.oy + Hr, int(ox):int(ox) + Wr]
            mx, my = (xx - ox) / PX, (yy - self.oy) / PX
            obst = np.zeros_like(mx, bool)
            for _, (a0, b0, a1, b1) in RM.BOXES:
                obst |= (mx >= a0) & (mx <= a1) & (my >= b0) & (my <= b1)
            obst |= np.hypot(mx - RM.PLANT[0], my - RM.PLANT[1]) <= RM.PLANT[2]
            f = np.zeros((H, W), bool)
            f[self.oy:self.oy + Hr, int(ox):int(ox) + Wr] = ~obst
            out.append(f)
        return out

    def sim_t(self, t):
        return min(max(t - 0.5, 0.0) * self.T_LAPSE, self.T)

    def frame(self, t):
        RM, PX = self.RM, self.PX
        st = self.sim_t(t)
        img = self.base.copy()
        cov = np.zeros((H, W), np.uint8)
        tr = np.zeros((H, W), np.float32)
        wpx = int(2 * RM.R * PX * 0.92)
        k = int(st / 0.02)
        P = self.rand[:k + 1]
        pl = np.c_[self.ox_l + P[:, 0] * PX, self.oy + P[:, 1] * PX].astype(np.int32)
        if len(pl) > 1:
            cv2.polylines(cov, [pl], False, 1, wpx, cv2.LINE_8)
            cv2.polylines(tr, [pl[-400:]], False, 1.0, 2, cv2.LINE_AA)
        dist = st * RM.V
        j = int(np.searchsorted(self.mcum, dist))
        Q = self.mpts[:max(j, 1)]
        if j > 0 and j < len(self.mpts):
            u = (dist - self.mcum[j - 1]) / max(self.mcum[j] - self.mcum[j - 1], 1e-9)
            Q = np.vstack([Q, self.mpts[j - 1] + u * (self.mpts[j] - self.mpts[j - 1])])
        ql = np.c_[self.ox_r + Q[:, 0] * PX, self.oy + Q[:, 1] * PX].astype(np.int32)
        pieces = [q for q in np.split(ql, [b for b in self.breaks if b < len(ql)]) if len(q) > 1]
        if pieces:
            cv2.polylines(cov, pieces, False, 1, wpx, cv2.LINE_8)
            cv2.polylines(tr, pieces[-1:], False, 0.7, 1, cv2.LINE_AA)
        g = np.array(GOLD, np.float32)
        covered = []
        for side, fmask in enumerate(self.free):
            c = (cov.astype(bool) & fmask)
            covered.append(c.sum() / fmask.sum())
        c = cov.astype(np.float32)[..., None] * (self.free[0] | self.free[1])[..., None]
        img = img * (1 - 0.45 * c) + g * 0.45 * c + tr[..., None] * np.array([255, 235, 190], np.float32) * 0.5
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        for ox in (self.ox_l, self.ox_r):
            px = lambda x, y: (ox + x * PX, self.oy + y * PX)
            for nm, (a0, b0, a1, b1) in RM.BOXES:
                d.rounded_rectangle([*px(a0, b0), *px(a1, b1)], 8, fill=(40, 46, 62), outline=(86, 96, 118), width=2)
                d.text(px((a0 + a1) / 2, (b0 + b1) / 2), nm, font=ST.font(19), fill=(130, 140, 160), anchor="mm")
            x, y, r = RM.PLANT
            d.ellipse([*px(x - r, y - r), *px(x + r, y + r)], fill=(34, 64, 48), outline=(70, 120, 90), width=2)
            d.rectangle([*px(0, 0), *px(RM.RW, RM.RH)], outline=(150, 160, 180), width=5)
        hl = math.atan2(*(P[-1] - P[max(0, len(P) - 8)])[::-1]) if len(P) > 8 else -0.35
        self.maps.draw_roomba(im, pl[-1][0], pl[-1][1], hl, diam=int(2 * RM.R * PX))
        hr = math.atan2(*(Q[-1] - Q[max(0, len(Q) - 2)])[::-1]) if len(Q) > 1 else 0.0
        self.maps.draw_roomba(im, ql[-1][0], ql[-1][1], hr, diam=int(2 * RM.R * PX))
        d = ImageDraw.Draw(im, "RGBA")
        a = fade(t, 0.2, 1e9)
        text(d, (self.ox_l, 190), "LAST TIME · A 2002 ROOMBA", 26, MUTED, a)
        text(d, (self.ox_l, 226), "No map. Bump, turn, repeat.", 34, INK2, a)
        text(d, (self.ox_r, 190), "THIS TIME · A MODERN ROBOT VACUUM", 26, MUTED, a)
        text(d, (self.ox_r, 226), "Maps the room. Rows, then the edges.", 34, INK2, a)
        yb = self.oy + RM.RH * PX + 24
        mm, ss = divmod(int(st), 60)
        text(d, (self.ox_l, yb), f"{mm}:{ss:02d} · {covered[0]:.0%} of the floor", 30, INK, a)
        done = st >= self.T - 1e-6
        text(d, (self.ox_r, yb), f"{mm}:{ss:02d} · " + (f"done ({covered[1]:.0%})" if done else f"{covered[1]:.0%} of the floor"),
             30, GOLD if done else INK, a)
        text(d, (W - 60, 56), f"TIME-LAPSE  ×{int(self.T_LAPSE)}", 26, INK2, a, "SemiBold", "ra")
        fr = np.asarray(im)
        return captions(fr, t, [(0.3, 4.9, [("Last time: a 2002 Roomba with no map, at random.", 42, INK, "SemiBold")]),
                                (5.1, 9.9, [("This one has a map. Rows 25 cm apart, every square metre once.", 40, INK, "SemiBold")])])


# ------------------------------------------------------------------------ 3. the rules
class Rules(Section):
    name, dur = "rules", 10.0

    def __init__(self):
        import shapely
        # the tower floor from OpenStreetMap, as in frame 06
        Cc = self._city()
        ks = np.nonzero(Cc.pb == Cc.esb)[0]
        tower = ks[np.argmax([(Cc.pz1[k] - Cc.pz0[k]) * (Cc.pnfl[k] > 0) for k in ks])]
        self.ring = Cc.rxy[Cc.ps[tower]:Cc.pe[tower]][:-1]
        self.area = shapely.Polygon(self.ring).area
        self.segs = FR.rows_in(self.ring)
        self.rows_m = sum(np.linalg.norm(q - p) for p, q in self.segs)

    def _city(self):
        return X.City()

    def frame(self, t):
        ring, segs = self.ring, self.segs
        mn, mx = ring.min(0), ring.max(0)
        s = 640 / max(mx - mn)
        ox, oy = 560, 540
        tf = lambda xy: (ox + (xy[0] - (mn[0] + mx[0]) / 2) * s, oy - (xy[1] - (mn[1] + mx[1]) / 2) * s)
        im = Image.new("RGB", (W, H), (5, 8, 15))
        d = ImageDraw.Draw(im, "RGBA")
        d.polygon([tf(p) for p in ring], fill=(14, 26, 44))
        q = ease((t - 0.6) / 7.6)
        n = len(segs)
        k_now = int(n * q)
        for i, (p, qq) in enumerate(segs[:k_now]):
            d.line([tf(p), tf(qq)], fill=(255, 186, 80, 150 if i % 2 else 205), width=max(1, int(LANE * s)))
        if 0 < k_now < n:
            p, qq = segs[k_now]
            u = n * q - k_now
            m = p + u * (qq - p)
            d.line([tf(p), tf(m)], fill=(255, 240, 210, 255), width=max(2, int(LANE * s)))
        d.polygon([tf(p) for p in ring], outline=(150, 200, 240), width=3)
        a = fade(t, 0.3, 1e9)
        text(d, (ox, oy + 360), f"One floor of the Empire State Building's tower · {self.area:,.0f} m²", 26, INK2, a, "SemiBold", "ma")
        x0 = 1060
        text(d, (x0, 150), "THE RULES", 26, MUTED, a)
        rules = [("Rows 25 cm apart.", "Every square metre of floor, once."),
                 ("Every floor of every building.", "Bottom to top, then on to the next building."),
                 ("Light speed the whole way.", "No slowing down for corners. Or stairs."),
                 ("Floors only.", "Not walls, not ceilings, not furniture.")]
        y = 200
        for i, (a_, b_) in enumerate(rules):
            aa = fade(t, 0.6 + 1.4 * i, 1e9, 0.45)
            text(d, (x0, y), a_, 38, INK, aa)
            text(d, (x0, y + 50), b_, 28, INK2, aa, "Regular")
            y += 128
        aa = fade(t, 7.0, 1e9, 0.45)
        text(d, (x0, y + 20), f"This floor: {self.rows_m / 1e3:.1f} km of rows", 34, GOLD, aa)
        text(d, (x0, y + 66), f"Light needs {self.rows_m / C_MS * 1e6:.0f} microseconds.", 34, GOLD, aa)
        return np.asarray(im)


# ------------------------------------------------------------------------ 4. the Empire State Building
class ESB(Section):
    name, dur = "esb", 15.0
    T0, T1 = 0.7, 10.3

    def __init__(self):
        self.R = CF.CityRenderer(levels=("P1",))
        R = self.R
        self.cam = city_cam(2.05, R.C.cx[R.esb], R.C.cy[R.esb], 215, 1180, 500)
        self.F = (self.T1 - self.T0) / (R.esb_t1 - R.esb_t0)

    def L(self, t):
        R = self.R
        if t <= self.T0:
            return 0.0
        return R.esb_t0 + min((t - self.T0) / (self.T1 - self.T0), 1.0) * (R.esb_t1 - R.esb_t0)

    def frame(self, t):
        R, C = self.R, self.R.C
        L = self.L(t)
        img = R.frame(self.cam, L)
        z, i, f = R.esb_state(L)
        if i is not None:
            _, k, j, _ = R.esb_slabs[i]
            x, y, _ = FR.active_floor(C, k, R.esb_slabs[i][0], f, self.cam.s, self.cam, img, dust_n=0, wake=False, rows_gain=0.9)
            CF.light_win(img, x, y, 0.55)
        fr = CF.finish(img)
        q = 0.0 if L <= R.esb_t0 else min(1.0, (L - R.esb_t0) / (R.esb_t1 - R.esb_t0))
        area = q * NYC["esb_floor_area_m2"]
        fr = hud(fr, [("LIGHT TIME", fmt_s(L), GOLD, 72),
                      ("FLOOR CLEANED", f"{area:,.0f} of {NYC['esb_floor_area_m2']:,.0f} m²", INK, 40),
                      ("ROWS DRIVEN", f"{area / LANE / 1e3:,.0f} km", INK, 40)], a=fade(t, 0.3, 1e9),
                 title="EMPIRE STATE BUILDING · 102 FLOORS")
        fr = tag(fr, factor_label(self.F), fade(t, 0.4, self.T1 + 0.3))
        blink = 0.1 / NYC["esb_light_s"]
        return captions(fr, t, [(0.4, 6.0, [("Back to the start, slowed right down.", 42, INK, "SemiBold")]),
                                (self.T1 + 0.2, 15.2, [("The whole Empire State Building: 3.5 milliseconds.", 44, INK, "SemiBold"),
                                                       (f"In the blink of an eye (about 0.1 s), it could clean {blink:.0f} of them.",
                                                        34, GOLD, "SemiBold", self.T1 + 1.6)])])


# ------------------------------------------------------------------------ 5. every floor in New York
class NYCRun(Section):
    name, dur = "nyc", 40.0
    chunk = 200
    END = 33.0
    KN = [(0, 2.1), (4, 1.45), (10, 0.9), (20, 0.7), (33, 0.62), (40, 0.62)]

    def __init__(self):
        self.R = CF.CityRenderer(levels=("P2", "P3"))
        R, C = self.R, self.R.C
        L0 = R.esb_t1
        self.Ltot = NYC["total_light_s"]
        self.v, self.Lv, self.Fv = slow_map(self.KN, self.dur, L0, self.Ltot, self.END)
        ex, ey = C.cx[R.esb], C.cy[R.esb]
        keys = [(0.0, math.log(0.55), ex + 120, ey + 250, 0, 1060, 640),
                (3.0, math.log(0.55), ex + 120, ey + 250, 0, 1060, 640),
                (12.0, math.log(0.105), ex + 1500, ey + 2500, 0, 1150, 560),
                (20.0, math.log(0.105), ex + 1500, ey + 2500, 0, 1150, 560),
                (30.0, math.log(0.036), ex + 6000, ey - 5000, 0, 1240, 560),
                (40.0, math.log(0.034), ex + 6000, ey - 5000, 0, 1240, 560)]
        self.cam_f = pchip_cols([k[0] for k in keys], [k[1:] for k in keys])
        mn = NYS["floor_area_by_borough_m2"]["MN"]
        self.mn = mn

    def L(self, t):
        return float(min(np.interp(t, self.v, self.Lv), self.Ltot + 1e-6)) if t < self.END else self.Ltot + 1e-6

    def prepare(self, t):
        self.R.advance(self.L(t))

    def frame(self, t):
        R = self.R
        L = self.L(t)
        R.advance(L)
        ls, tx, ty, tz, ox, oy = (float(f(t)) for f in self.cam_f)
        cam = city_cam(math.exp(ls), tx, ty, tz, ox, oy)
        img = R.frame(cam, L)
        if t < self.END:
            x, y, z = R.frontier_point(L)
            u, v = cam.project(np.array([x]), np.array([y]), np.array([z]))
            CF.light_win(img, float(u[0]), float(v[0]), 0.65)
        fr = CF.finish(img, bloom=(0.24, 0.15, 0.09))
        a_run = fade(t, -1, self.END + 0.2, 0.3, 0.4)
        fr = nyc_hud(fr, R, min(L, self.Ltot), a_run)
        if t > self.END:
            moon = NYC["rows_km"] / 384400
            fr = hud(fr, [("LIGHT TIME", fmt_s(NYC["total_light_s"]), GOLD, 72),
                          ("BUILDINGS", f"{NYC['buildings']:,}", INK, 40),
                          ("FLOOR", f"{NYC['floor_area_m2'] / 1e6:,.0f} million m²", INK, 40),
                          ("ROWS DRIVEN", f"{NYC['rows_km'] / 1e6:.2f} million km", INK, 40)],
                     a=fade(t, self.END + 0.1, 1e9, 0.5), title="NEW YORK CITY · EVERY FLOOR")
        fr = tag(fr, factor_label(float(np.interp(t, self.v, self.Fv))), fade(t, -1, self.END, 0.3, 0.4))
        moon = NYC["rows_km"] / 384400
        return captions(fr, t, [(12.5, 19.6, [(f"Manhattan alone: {self.mn / 1e6:,.0f} million m² of floor.", 42, INK, "SemiBold"),
                                              (f"Light needs {self.mn / LANE / C_MS:.1f} seconds to vacuum it.", 34, GOLD, "SemiBold", 13.8)]),
                                (self.END + 0.3, 40.2, [(f"Every floor in New York: {NYC['total_light_s']:.1f} seconds.", 46, GOLD, "Bold"),
                                                        (f"{NYC['rows_km'] / 1e6:.1f} million km of rows: {moon:.1f} times the distance to the Moon.",
                                                         32, INK2, "Regular", self.END + 1.6)])])


# ------------------------------------------------------------------------ 6/8. every floor on Earth
class Sweep:
    """the world sweep in video time: E1 seconds in earth1 then E2 in earth2, at one constant speed-up"""
    E1, E2 = 46.0, 14.0
    SP = WORLD_T / (E1 + E2)

    def __init__(self):
        s = np.load(os.path.join(HERE, "data", "ghsl", "sweep.npz"))
        self.cum, self.rr, self.cc = s["cum"], s["rr"], s["cc"]
        x0, y0, cell = float(s["x0"]), float(s["y0"]), float(s["cell"])
        self.lon = x0 + (self.cc + 0.5) * cell
        self.lat = y0 - (self.rr + 0.5) * cell
        n = int((self.E1 + self.E2) * FPS) + 1
        u = np.arange(n) / FPS
        idx = np.clip(np.searchsorted(self.cum, self.done(u)), 0, len(self.cum) - 1)
        v = G._xyz(self.lon[idx], self.lat[idx])
        self.flon, self.flat = self.lon[idx], self.lat[idx]
        vs = gaussian_filter1d(v, 2.5 * FPS, axis=0, mode="nearest")
        vs /= np.linalg.norm(vs, axis=1, keepdims=True)
        self.clon = np.degrees(np.arctan2(vs[:, 1], vs[:, 0]))
        self.clat = np.degrees(np.arcsin(vs[:, 2]))
        cities = {"Tokyo": (139.69, 35.69), "Beijing": (116.40, 39.90), "Shanghai": (121.47, 31.23), "Delhi": (77.21, 28.61)}
        cj = json.load(open(os.path.join(HERE, "data", "ghsl", "cities.json")))
        self.cities = {}
        X3 = G._xyz(self.lon, self.lat)
        for nm, (lo, la) in cities.items():
            d = np.degrees(np.arccos(np.clip(X3 @ G._xyz(lo, la), -1, 1))) * 111.2
            k = np.nonzero(d < 40)[0]
            done_at = self.cum[k.max()] if len(k) else 0
            self.cities[nm] = (lo, la, self.u_of(done_at), cj[nm] / LANE / C_MS)

    def done(self, u):
        Lw = np.minimum(np.asarray(u) * self.SP, WORLD_T)
        return Lw / WORLD_T * WORLD_FLOOR_M2

    def u_of(self, done):
        return done / WORLD_FLOOR_M2 * WORLD_T / self.SP

    def at(self, u):
        i = int(np.clip(round(u * FPS), 0, len(self.clon) - 1))
        return self.flon[i], self.flat[i], self.clon[i], self.clat[i]


def world_hud(fr, done, a=1.0):
    t = done / WORLD_FLOOR_M2 * WORLD_T
    return hud(fr, [("LIGHT TIME", fmt_s(t), GOLD, 64),
                    ("FLOOR CLEANED", f"{done / 1e9:,.1f} billion m²" if done > 1e9 else f"{done / 1e6:,.0f} million m²", INK, 40),
                    ("OF ABOUT", "273 billion m²", INK2, 34)], a=a, title="EVERY FLOOR ON EARTH")


class Earth1(Section):
    name, dur = "earth1", Sweep.E1
    chunk = 180

    def __init__(self):
        self.S = Sweep()
        bj = self.S.cities["Beijing"][2]
        tk = self.S.cities["Tokyo"][2]
        self.caps = [(1.0, 8.2, [("Now every floor on Earth: about 273 billion m².", 44, INK, "SemiBold"),
                                 ("About the area of New Zealand, mostly in homes.", 34, GOLD, "SemiBold", 2.4)]),
                     (bj - 2.5, bj + 4.5, [("China has about a fifth of Earth's floor (20%).", 42, INK, "SemiBold")]),
                     (max(tk, bj + 4.8), max(tk, bj + 4.8) + 6.5,
                      [("Greater Tokyo is the biggest single city: 24 seconds of light.", 40, GOLD, "SemiBold")])]

    def u(self, t):
        return t

    def frame(self, t):
        S = self.S
        u = self.u(t)
        done = float(S.done(u))
        flon, flat, clon, clat = S.at(u)
        g = dict(cx=1240, cy=500, r=440, lon0=clon, lat0=float(np.clip(0.75 * clat, -62, 50)))
        fr = globe_img(g, done, light=(flon, flat))
        fr = world_hud(fr, done)
        fr = tag(fr, f"SPED UP  ×{S.SP:.0f}", 1.0)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        for nm, (lo, la, uc, ts) in S.cities.items():
            a = fade(u, uc, uc + 6.0, 0.4, 0.6)
            if a <= 0:
                continue
            x, y, vis = G.ortho_forward(np.array([lo]), np.array([la]), g)
            if not vis[0]:
                continue
            x, y = float(x[0]), float(y[0])
            side = 1 if nm in ("Tokyo", "Shanghai") else -1
            d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(240, 244, 250, int(230 * a)), width=2)
            text(d, (x + side * 26, y - 16), f"{nm} · {ts:.0f} s", 26, INK, a, "SemiBold", "la" if side > 0 else "ra")
        return captions(np.asarray(im), t, self.caps)


class Earth2(Earth1):
    name, dur = "earth2", Sweep.E2 + 6.0
    chunk = 200
    POLE = (0.0, -89.99)

    def __init__(self):
        self.S = Sweep()

    def u(self, t):
        return Sweep.E1 + min(t, Sweep.E2)

    def frame(self, t):
        S = self.S
        u = self.u(t)
        done = float(S.done(u)) if t < Sweep.E2 else WORLD_FLOOR_M2 * 1.0001
        flon, flat, clon, clat = S.at(u)
        k = ease((t - (Sweep.E2 - 2.0)) / 2.0)                    # the light heads for the pole at the end
        if k > 0:
            v = G._xyz(flon, flat) * (1 - k) + G._xyz(*self.POLE) * k
            v /= np.linalg.norm(v)
            flon, flat = math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(v[2]))
        lat_c = float(np.clip(0.75 * clat, -64, 50))
        e = ease((t - (Sweep.E2 - 4.5)) / 4.0)                      # swing south for the last floor
        dl = ((150.0 - clon + 180) % 360) - 180
        g = dict(cx=1240, cy=500 - 60 * e, r=430, lon0=clon + e * dl, lat0=lat_c + e * (-62.0 - lat_c))
        fr = globe_img(g, done, light=(flon, flat))
        T = WORLD_T
        if t < Sweep.E2 + 0.3:
            fr = world_hud(fr, min(done, WORLD_FLOOR_M2), fade(t, -1, Sweep.E2 + 0.3, 0.3, 0.4))
            fr = tag(fr, f"SPED UP  ×{S.SP:.0f}", fade(t, -1, Sweep.E2, 0.3, 0.4))
        fr = hud(fr, [("LIGHT TIME", "1 h 08 min", GOLD, 72),
                      ("VACUUMING", fmt_s(WORLD_CLEAN_S), INK, 40),
                      ("GETTING THERE", "≈ 7 min", INK, 40),
                      ("BUILDINGS", "≈ 2.5 billion", INK, 40)], a=fade(t, Sweep.E2 + 0.2, 1e9, 0.5), title="EVERY FLOOR ON EARTH")
        a = fade(t, Sweep.E2 - 0.6, 1e9, 0.5)
        if a > 0:
            u_, v_, vis = G.ortho_forward(np.array([self.POLE[0]]), np.array([self.POLE[1]]), g)
            if vis[0]:
                fr = ST.ring_label(fr, int(u_[0]), int(v_[0]), "Amundsen–Scott South Pole Station: the last floor", a, side=-1)
        return captions(fr, t, [(Sweep.E2 + 0.3, self.dur + 0.2,
                                 [("Every floor on Earth: about 1 hour 8 minutes.", 46, GOLD, "Bold"),
                                  ("An hour of vacuuming, and 7 minutes getting between 2.5 billion buildings.", 32, INK2, "Regular",
                                   Sweep.E2 + 1.8)])])


# ------------------------------------------------------------------------ 7. how far
class Far(Section):
    name, dur = "far", 12.0

    def frame(self, t):
        AU = 149597870.7
        rows_km = WORLD_FLOOR_M2 / LANE / 1e3
        im = Image.new("RGB", (W, H), (5, 8, 15))
        d = ImageDraw.Draw(im, "RGBA")
        x0, x1, y = 170, 1760, 520
        sc = (x1 - x0) / 10.0
        a = fade(t, 0.2, 1e9)
        d.line([(x0, y), (x1, y)], fill=(60, 74, 96, int(255 * a)), width=2)
        for k in range(1, 11):
            d.line([(x0 + k * sc, y - 8), (x0 + k * sc, y + 8)], fill=(60, 74, 96, int(255 * a)), width=2)
            text(d, (x0 + k * sc, y + 22), f"{k} AU", 20, MUTED, a, "Regular", "ma")
        d.ellipse([x0 - 34, y - 34, x0 + 34, y + 34], fill=(255, 210, 120, int(255 * a)))
        text(d, (x0, y - 70), "Sun", 26, INK2, a, "SemiBold", "ma")
        for nm, au, r in (("Mercury", 0.387, 5), ("Venus", 0.723, 8), ("Earth", 1.0, 8), ("Mars", 1.524, 6),
                          ("Jupiter", 5.204, 20), ("Saturn", 9.537, 17)):
            x = x0 + au * sc
            d.ellipse([x - r, y - r, x + r, y + r], fill=(170, 186, 206, int(255 * a)))
            if nm in ("Earth", "Jupiter", "Saturn", "Mars"):
                text(d, (x, y - r - 40), nm, 26, INK2, a, "SemiBold", "ma")
        q = ease((t - 1.0) / 5.0)
        dist = q * rows_km
        xe = x0 + dist / AU * sc
        if q > 0:
            d.line([(x0, y - 120), (xe, y - 120)], fill=(255, 186, 80, 255), width=10)
            d.ellipse([xe - 12, y - 132, xe + 12, y - 108], fill=(255, 240, 210, 255))
            text(d, (xe, y - 190), f"{dist / 1e9:.2f} billion km", 40, GOLD, 1.0, "Bold", "ma")
            text(d, ((x0 + xe) / 2, y - 100), "every row on every floor, end to end", 26, INK2, fade(t, 1.5, 1e9), "Regular", "ma")
        return captions(np.asarray(im), t, [(6.4, 12.2, [(f"{rows_km / 1e9:.2f} billion km of rows: farther than Jupiter is from the Sun.",
                                                          42, INK, "SemiBold")])])


# ------------------------------------------------------------------------ 9. the dustbin
class Dust(Section):
    name, dur = "dust", 7.0

    def __init__(self):
        g = dict(cx=1240, cy=500, r=440, lon0=20, lat0=20)
        self.base = (globe_img(g, WORLD_FLOOR_M2 * 1.0001, glow=0.6).astype(np.float32) * 0.55).astype(np.uint8)

    def frame(self, t):
        q = ease((t - 0.4) / 2.4)
        dust_t = WORLD_FLOOR_M2 * 0.3 / 1e6
        bins = WORLD_FLOOR_M2 * 0.3 / 150
        yrs = bins * 10 / 3.156e7
        fr = hud(self.base.copy(), [("DUST PICKED UP", f"≈ {q * dust_t:,.0f} tonnes", INK, 52),
                                    ("BIN EMPTIED", f"≈ {q * bins / 1e6:,.0f} million times", INK, 52),
                                    ("AT 10 S PER EMPTYING", f"≈ {q * yrs:,.0f} years", GOLD, 52)],
                 a=fade(t, 0.2, 1e9), title="THE DUSTBIN", y0=300, x0=110)
        return captions(fr, t, [(0.3, 7.2, [("Oh, and the dust: about half a billion bin-fulls.", 42, INK, "SemiBold"),
                                            ("The robot is fast. The dock that empties it is not.", 34, GOLD, "SemiBold", 2.6)])])


# ------------------------------------------------------------------------ 10. the U-turn
class UTurn(Section):
    name, dur = "uturn", 16.0
    HIT = 3.4
    SPEED = 3.0                       # m/s on screen: slowed down 100 million times

    def __init__(self):
        self.C = X.City()
        C = self.C
        ks = np.nonzero(C.pb == C.esb)[0]
        self.tower = ks[np.argmax([(C.pz1[k] - C.pz0[k]) * (C.pnfl[k] > 0) for k in ks])]
        j = 14
        self.zf = C.pz0[self.tower] + j * (C.pz1[self.tower] - C.pz0[self.tower]) / C.pnfl[self.tower]
        ring = C.rxy[C.ps[self.tower]:C.pe[self.tower]][:-1]
        segs = FR.rows_in(ring)
        lens = np.array([np.linalg.norm(q - p) for p, q in segs])
        self.cum = np.cumsum(lens)
        k = int(len(segs) * 0.40)
        self.k = k
        self.f_end = (self.cum[k] - 0.2) / self.cum[-1]
        r = segs[k][1]
        self.cam = X.Cam(CF.PSI, 40, 120.0, r[0], r[1], self.zf, ox=960, oy=500, W=W, H=H, ss=1)
        m, c2 = 4.0, C_MS ** 2
        gam = 1 / math.sqrt(1 - 0.99 ** 2)
        self.tsar = (gam - 1) * m * c2 / (50 * 4.184e15)
        self.turns = WORLD_FLOOR_M2 / LANE / 4.0

    def frame(self, t):
        C = self.C
        d_left = max(0.0, (self.HIT - t) * self.SPEED)                  # metres before the wall
        f = self.f_end - d_left / self.cum[-1]
        img = np.zeros((H, W, 3), np.float32) + X.BG
        x, y, hd = FR.active_floor(C, self.tower, self.zf, f, self.cam.s, self.cam, img, dust_n=250000, wake=True, rows_gain=0.30)
        glow = 0.6 + 2.0 * ease((t - (self.HIT - 1.2)) / 1.2)
        boom = math.exp(-max(0.0, t - self.HIT - 0.5) / 0.9) * (1.0 if t >= self.HIT else 0.0)
        CF.light_win(img, x, y, 1.2 + 6.0 * boom, gain=glow + 25 * boom)
        img += boom * 2.2 * np.array([1.0, 0.85, 0.6], np.float32)
        fr = CF.finish(img, bloom=(0.3, 0.25, 0.2))
        if t < self.HIT + 0.15:
            fr = FR.paste_rgba(fr, FR.robot_sprite(int(0.35 * self.cam.s * 1.5), hd), x, y)
        fr = tag(fr, "SLOW MOTION  ×100,000,000", fade(t, 0.2, self.HIT, 0.3, 0.2))
        fr = hud(fr, [("ONE STOP AT 99% OF LIGHT SPEED", f"{self.tsar:.0f} Tsar Bombas", GOLD, 56),
                      ("U-TURNS, EVERY FLOOR ON EARTH", f"≈ {self.turns / 1e9:,.0f} billion", INK, 44)],
                 a=fade(t, 5.6, 1e9, 0.5), title="PHYSICS FOOTNOTE")
        return captions(fr, t, [(0.3, self.HIT - 0.1, [("Physics footnote: nothing with mass can actually reach light speed.", 38, INK, "SemiBold")]),
                                (6.0, 11.0, [("At 99% of it, every U-turn at the end of a row", 40, INK, "SemiBold"),
                                             ("releases about 10 Tsar Bombas.", 40, GOLD, "SemiBold", 7.2)]),
                                (11.2, 16.2, [(f"And it makes about {self.turns / 1e9:.0f} billion U-turns.", 42, INK, "SemiBold")])])


# ------------------------------------------------------------------------ 11. spotless
class Spotless(Section):
    name, dur = "spotless", 12.0

    def __init__(self):
        turns = WORLD_FLOOR_M2 / LANE / 4.0
        gam = 1 / math.sqrt(1 - 0.99 ** 2)
        self.sun_min = turns * (gam - 1) * 4.0 * C_MS ** 2 / 3.828e26 / 60
        self.turns = turns

    def frame(self, t):
        g = dict(cx=960, cy=500, r=400, lon0=-40 + 2.0 * t, lat0=25)
        fire = 1.6 * ease((t - 0.8) / 5.5)
        fr = globe_img(g, WORLD_FLOOR_M2 * 1.0001, fire=fire)
        fr = tag(fr, f"{self.turns / 1e9:.0f} billion U-turns ≈ {self.sun_min:.0f} minutes of the Sun's entire output",
                 fade(t, 1.0, 1e9), x=960, y=40, size=28, anchor="ma")
        return captions(fr, t, [(6.0, 12.2, [("The floors would be spotless. There just wouldn't be any floors.", 42, INK, "SemiBold"),
                                             ("What should it clean next?", 40, GOLD, "Bold", 8.6)])])


SECTIONS = {c.name: c for c in (Hook, Room, Rules, ESB, NYCRun, Earth1, Far, Earth2, Dust, UTurn, Spotless)}
ORDER = ["hook", "room", "rules", "esb", "nyc", "earth1", "far", "earth2", "dust", "uturn", "spotless"]


# ======================================================================== rendering
def encode(path, frames_iter):
    tmp = path + ".tmp.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                          "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for fr in frames_iter:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    os.replace(tmp, path)


def render_chunk(name, i0, i1):
    sec = SECTIONS[name]()
    sec.prepare(i0 / FPS)
    cdir = os.path.join(OUT, "chunks")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{name}_{i0:05d}.mp4")
    t0 = time.time()

    def gen():
        for i in range(i0, i1):
            yield sec.frame(i / FPS)
            if (i - i0) % 30 == 29:
                print(f"{name} {i + 1}/{i1} ({time.time() - t0:.0f}s)", flush=True)
    encode(path, gen())
    open(path + ".ok", "w").close()


def render(name, workers=3):
    cls = SECTIONS[name]
    n = int(round(cls.dur * FPS))
    step = cls.chunk
    ranges = [(i, min(n, i + step)) for i in range(0, n, step)]
    cdir = os.path.join(OUT, "chunks")
    os.makedirs(cdir, exist_ok=True)
    todo = [r for r in ranges if not os.path.exists(os.path.join(cdir, f"{name}_{r[0]:05d}.mp4.ok"))]
    procs = []
    while todo or procs:
        while todo and len(procs) < workers:
            i0, i1 = todo.pop(0)
            log = open(os.path.join(OUT, f"{name}_{i0:05d}.log"), "w")
            procs.append(subprocess.Popen([sys.executable, __file__, "chunk", name, str(i0), str(i1)], stdout=log, stderr=subprocess.STDOUT))
        time.sleep(2)
        for p in procs[:]:
            if p.poll() is not None:
                procs.remove(p)
                assert p.returncode == 0, f"{name} chunk failed"
    lst = os.path.join(cdir, f"{name}.txt")
    open(lst, "w").write("".join(f"file '{os.path.join(cdir, f'{name}_{a:05d}.mp4')}'\n" for a, _ in ranges))
    out = os.path.join(OUT, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    print("wrote", out, flush=True)


def stills(name, ts):
    sec = SECTIONS[name]()
    os.makedirs(OUT, exist_ok=True)
    for t in ts:
        sec.prepare(t)
        Image.fromarray(sec.frame(t)).save(os.path.join(OUT, f"{name}_{t:05.2f}.png"))
        print("still", name, t, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "chunk":
        render_chunk(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "render":
        render(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif cmd == "still":
        stills(sys.argv[2], [float(x) for x in sys.argv[3:]])
