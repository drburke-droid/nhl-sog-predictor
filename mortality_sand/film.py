"""Paint the film: the sandpile simulation (sand.py) composited into the Blender-rendered glass world
(blender_scene.py plates), with the lifted grains rising out of the frame.

    python film.py still 60 120 180        # PNG stills at those video seconds -> out/
    python film.py video --workers 4       # the film -> out/two_thousand_years_of_catastrophe.mp4
    python film.py video --from 120 --to 150   # a stretch of it, for checking

Per frame: the sand surface is shaded in map space (pile lighting from the height field, ambient occlusion,
cast shadows), every visible grain splatted as a small lit ball through the camera, then laid under the
glass (the plates' ratio = glass tint, difference = glass highlights). A lifted grain first glows in the
colour of its cause where it lay, then is drawn up into a turning column over the place it came from and
pulled up out of the frame, faster and faster, toward the camera.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import story as S
import sand as Sd
import places as PL
import camera as C
from sand import smooth, year_at, s_at, hash01, CELL, WC, HC, GRAIN_DZ
from proj import W, H, to_px

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
FPS = 30
FONT_DIR = "/usr/share/fonts/opentype/inter"
SUPERSAMPLE = int(os.environ.get("SAND_SS", "2"))       # grain splat supersampling (1: quick previews)
FILM_NAME = "two_thousand_years_of_catastrophe"


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


SANDCOL = hex_rgb(S.SAND_COLOUR)
CCOL = np.stack([hex_rgb(c[2]) for c in S.CAUSES])
INK = hex_rgb("#e8eef8")
MUTED = hex_rgb("#93a1b8")
TG = 0.45              # seconds a lifted grain glows on its pile before it goes
LABEL_MIN = 2.0e6      # catastrophes of at least this many deaths get a name on the map
LABEL_HOLD = 2.6       # seconds a name stays up at least
LABEL_MAX = 9.0        # ...and at most
LABEL_RATE = 5.0e4     # deaths a year at its worst: below this an event is a drizzle, not named
PLACE_MIN = 1.5e5      # a place (battle, camp, hotspot) with at least this many deaths is tagged on the map
PLACE_HOLD = 1.6       # seconds a place tag stays up at least
PLACE_MAX = 6.0        # ...and at most
PLACE_LIVE = 6         # tags on screen at once, at most (the biggest places first)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def load_plate(name):
    p = name if os.path.isabs(name) else os.path.join(DATA, name)
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
    if im.shape[1] != W:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
    return im


def project_P(P, x, y, z):
    q = P @ np.stack([x, y, z, np.ones_like(x)])
    w = np.where(q[2] > 1e-6, q[2], np.nan)
    return np.nan_to_num(q[0] / w, nan=-1e6), np.nan_to_num(q[1] / w, nan=-1e6)


class Plates:
    """The Blender glass for the wide shot: the table alone (bare) and with the glass; their ratio is the
    glass's tint and refraction darkening, their difference its highlights. Without plates (before
    blender_scene.py has run) the printed table is used with no glass."""

    def __init__(self):
        g, b = os.path.join(DATA, "plate_glass.png"), os.path.join(DATA, "plate_bare.png")
        if os.path.exists(g) and os.path.exists(b):
            gl, bare = load_plate(g), load_plate(b)
            self.bare = bare
            self.ratio = np.clip((gl + 0.004) / (bare + 0.004), 0.42, 1.0)
            self.spec = np.clip(gl - bare, 0, None)
        else:
            print("  (no glass plates yet: table only)", flush=True)
            t = cv2.cvtColor(cv2.imread(os.path.join(DATA, "table.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            M = C.hom(C.P_of(C.WIDE), 0.0) @ np.diag([0.5, 0.5, 1.0])
            self.bare = cv2.warpPerspective(t, M, (W, H), flags=cv2.INTER_AREA) ** 2.2 * 0.8
            self.ratio = np.ones((H, W, 3), np.float32)
            self.spec = np.zeros((H, W, 3), np.float32)

    def at(self, s, P):
        return self.bare, self.ratio, self.spec


class Painter:
    def __init__(self, hud_scale=1, sim=None):
        self.k = k = hud_scale              # the HUD drawn k times larger (2: for a 3840x2160 overlay)
        self.plates = Plates()
        self.set_camera(0.0)
        self.vis = None
        self.prev_hall = np.zeros(Sd.NCELL, np.int64)
        rng = np.random.default_rng(9)
        self.cell_tint = (0.86 + 0.24 * rng.random((HC, WC))).astype(np.float32)
        self.disp_t = None
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.illum = (0.80 + 0.32 * (1 - (xx / W) * 0.55 - (yy / H) * 0.45))[..., None]   # key light upper left
        r2 = ((xx / W - 0.5) / 0.78) ** 2 + ((yy / H - 0.5) / 0.8) ** 2
        self.vign = (1 - 0.32 * np.clip(r2, 0, 1.5) ** 1.6)[..., None].astype(np.float32)
        self.f_label = font("Inter-SemiBold.otf", 14 * k)
        self.f_year = font("InterDisplay-SemiBold.otf", 58 * k)
        self.f_year_sub = font("Inter-Medium.otf", 20 * k)
        self.f_count = font("Inter-Medium.otf", 17 * k)
        self.f_leg = font("Inter-Medium.otf", 14 * k)
        self.f_leg_h = font("Inter-SemiBold.otf", 12 * k)
        self.f_title = font("InterDisplay-SemiBold.otf", 84 * k)
        self.f_sub = font("Inter-Regular.otf", 28 * k)
        self.f_note = font("Inter-Regular.otf", 15 * k)
        self.f_event = font("InterDisplay-SemiBold.otf", 21 * k)
        self.f_event_sub = font("Inter-Medium.otf", 15 * k)
        self.f_end = font("InterDisplay-SemiBold.otf", 44 * k)
        self.f_end_sub = font("Inter-Medium.otf", 24 * k)
        self.cache = {}
        L = sim.L if sim is not None else json.load(open(os.path.join(DATA, "lifts.json")))
        self.events = event_groups(L)
        self.places = place_groups(L)
        self.f_place = font("Inter-SemiBold.otf", 13 * k)

    # ---- the camera for this frame
    def set_camera(self, s):
        self.pose = C.pose_at(s)
        self.P = C.P_of(self.pose)
        self.loc, self.cr, self.cu, self.cf = C.basis(self.pose)

    def project(self, x, y, z):
        return project_P(self.P, x, y, z)

    # ---- text helpers
    def text_sprite(self, text, fnt, rgb, max_w=None, spacing=6, align="left", halo=False):
        key = (text, id(fnt), tuple(rgb), max_w, align, halo)
        if key in self.cache:
            return self.cache[key]
        lines = [text]
        if max_w:
            words, lines, cur = text.split(), [], ""
            for wd in words:
                t = (cur + " " + wd).strip()
                if fnt.getlength(t) > max_w and cur:
                    lines.append(cur); cur = wd
                else:
                    cur = t
            lines.append(cur)
        asc, desc = fnt.getmetrics()
        lh = asc + desc + spacing * self.k
        pad = (7 if halo else 2) * self.k
        ww = int(max(fnt.getlength(l) for l in lines)) + 2 * pad
        hh = lh * len(lines) + 2 * pad
        im = Image.new("L", (ww, hh), 0)
        dr = ImageDraw.Draw(im)
        for i, l in enumerate(lines):
            x = pad if align == "left" else (ww - fnt.getlength(l)) / 2
            dr.text((x, pad + i * lh), l, font=fnt, fill=255)
        a = np.asarray(im, np.float32) / 255
        spr = np.zeros((hh, ww, 4), np.float32)
        spr[..., :3] = rgb
        spr[..., 3] = a
        if halo:
            hal = np.clip(cv2.GaussianBlur(cv2.dilate(a, np.ones((2 * self.k + 1,) * 2, np.uint8)), (0, 0), 2.2 * self.k)
                          * 1.5, 0, 0.85)
            out_a = a + hal * (1 - a)
            spr[..., :3] = (rgb * a[..., None] + 0.02 * (hal * (1 - a))[..., None]) / np.maximum(out_a, 1e-4)[..., None]
            spr[..., 3] = out_a
        self.cache[key] = spr
        return spr

    @staticmethod
    def blit(img, spr, x, y, alpha=1.0):
        if alpha <= 0.003:
            return
        h, w = spr.shape[:2]
        x, y = int(round(x)), int(round(y))
        xa, ya, xb, yb = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
        if xa >= xb or ya >= yb:
            return
        s = spr[ya - y:yb - y, xa - x:xb - x]
        a = s[..., 3:4] * alpha
        reg = img[ya:yb, xa:xb]
        reg *= (1 - a)
        reg += s[..., :3] * a

    def panel(self, img, x0, y0, x1, y1, a):
        """A soft dark backing behind HUD text, so it reads over sand in the close-ups."""
        key = ("panel", int(x0), int(y0), int(x1), int(y1))
        pad = 40 * self.k
        if key not in self.cache:
            m = np.zeros((int(y1 - y0) + 2 * pad, int(x1 - x0) + 2 * pad), np.float32)
            m[pad:-pad, pad:-pad] = 1.0
            self.cache[key] = cv2.GaussianBlur(m, (0, 0), 16 * self.k)[..., None]
        m = self.cache[key]
        xa, ya = int(x0) - pad, int(y0) - pad
        h, w = m.shape[:2]
        xb, yb = min(img.shape[1], xa + w), min(img.shape[0], ya + h)
        sx, sy = max(0, -xa), max(0, -ya)
        xa, ya = max(0, xa), max(0, ya)
        img[ya:yb, xa:xb] *= 1 - a * m[sy:sy + yb - ya, sx:sx + xb - xa]

    @staticmethod
    def shade(img, x, y, w, h, a):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
        if a > 0.003 and x0 < x1 and y0 < y1:
            img[y0:y1, x0:x1] *= (1 - a)

    # ---- the sand: every grain a small lit ball, as in the Blender renders

    # ---- the sand: every grain a small lit ball, as in the Blender renders
    def update_disp(self, sim):
        """Choose the grains worth drawing (each pile's top layers, and every grain on a slope or the
        side of a tower) and ease their colours toward the true ones, so swaps fade instead of
        flickering and new grains swell into place. Call every frame."""
        if self.disp_t == sim.s:
            return
        self.disp_t = sim.s
        h = sim.h.astype(np.int64)
        occ = np.where(h > 0)[0]
        if not len(occ):
            self.vis = None
            self.prev_hall = h
            return
        hn = np.full(len(occ), 1 << 20, np.int64)
        for d in range(4):
            nb = Sd.NB[d][occ]
            hn = np.minimum(hn, np.where(nb >= 0, h[np.maximum(nb, 0)], 0))
        ho = h[occ]
        lo = np.clip(np.minimum(hn, ho) - 2, 0, None)
        cnt = ho - lo
        cell = np.repeat(occ, cnt)
        lev = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        hcol = np.repeat(ho, cnt)
        tgt = SANDCOL[None, :] * (self.cell_tint.ravel()[cell] * (0.88 + 0.24 * hash01(cell, lev, 5)))[:, None]
        key = cell * Sd.HMAX + lev                      # sorted: cells ascending, levels ascending
        col = tgt.astype(np.float32)
        scale = np.ones(len(key), np.float32)
        if self.vis is not None:
            pk = self.vis["key"]
            j = np.clip(np.searchsorted(pk, key), 0, len(pk) - 1)
            hit = pk[j] == key
            pc, ps = self.vis["col"][j[hit]], self.vis["scale"][j[hit]]
            col[hit] = pc + (tgt[hit] - pc) * 0.18
            scale[hit] = np.minimum(ps + 0.22, 1.0)
            fresh = (~hit) & (lev >= self.prev_hall[cell])            # just added on top: swell in
            scale[fresh] = 0.3
        self.vis = dict(key=key, cell=cell, lev=lev, hcol=hcol, col=col, scale=scale)
        self.prev_hall = h

    def shadows(self, sim):
        """Shadows on the table (map space, cell resolution; light from the upper left): long ones
        cast by towers, and soft contact shadows around every pile."""
        h = sim.h
        Z = (h * GRAIN_DZ).astype(np.float32).reshape(HC, WC)
        sh = np.ones((HC, WC), np.float32)
        if Z.max() > 4:
            for j in range(1, int(Z.max() / CELL) + 2):
                occ = Z[:-j, :-j] - CELL * j > 1.0
                sh[j:, j:][occ] = np.minimum(sh[j:, j:][occ], 0.58 + 0.25 * min(1.0, j / 40.0))
            sh = cv2.GaussianBlur(sh, (0, 0), 0.9)
        self.tower_shadow = sh
        cov = np.clip(h.reshape(HC, WC).astype(np.float32) * 0.45, 0, 1)
        contact = np.zeros_like(cov)
        contact[1:, 1:] = cov[:-1, :-1]
        contact = cv2.GaussianBlur(np.maximum(contact, cov * 0.6), (0, 0), 1.1)
        tab = sh * (1 - 0.42 * contact)
        M = C.hom(self.P, 0.0) @ np.array([[CELL, 0, CELL / 2], [0, CELL, CELL / 2], [0, 0, 1.0]])
        self.table_shadow = cv2.warpPerspective(tab, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=1.0)[..., None]

    def grains(self, sim, s):
        """Splat the visible grains as shaded spheres at 2x, far to near. Returns premultiplied colour,
        coverage, the share of covered pixels below the glass rim, and depth."""
        V = self.vis
        if V is None:
            return None
        cell, lev, hcol, col = V["cell"], V["lev"], V["hcol"], V["col"]
        jx, jy, jr = hash01(cell, lev, 1), hash01(cell, lev, 2), hash01(cell, lev, 3)
        x = Sd.CELL_XY[cell, 0] + (jx - 0.5) * 1.3 * Sd.GS
        y = Sd.CELL_XY[cell, 1] + (jy - 0.5) * 1.3 * Sd.GS
        r = (1.35 + 0.35 * jr) * Sd.GS * V["scale"]
        z = lev * GRAIN_DZ + r
        Xw = np.stack([x / 100.0 - 9.6, 5.4 - y / 100.0, z / 100.0], 1) - self.loc
        d = Xw @ self.cf
        u = W / 2 + C.FOCAL_PX * (Xw @ self.cr) / d
        v = H / 2 - C.FOCAL_PX * (Xw @ self.cu) / d
        rho = (r / 100.0) * C.FOCAL_PX / d
        vis = (d > 0.1) & (u > -12) & (u < W + 12) & (v > -12) & (v < H + 12)
        if not vis.any():
            return None
        sel = np.where(vis)[0][np.argsort(-d[vis], kind="stable")]     # far to near: nearer grains paint over
        # light: ambient occlusion down the sides of piles and towers, cast shadows near the ground
        below = (hcol[sel] - 1 - lev[sel]).astype(np.float32)
        ao = 0.58 + 0.42 * np.clip(1 - below / np.maximum(6.0, 0.6 * hcol[sel]), 0, 1)
        sh = np.where(z[sel] < 12.0, self.tower_shadow.ravel()[cell[sel]], 1.0)
        base = col[sel] * (ao * sh)[:, None]
        SS = SUPERSAMPLE
        cu, cv_, rs = u[sel] * SS, v[sel] * SS, np.maximum(rho[sel] * SS, 0.7)
        zs, ds = z[sel], d[sel].astype(np.float32)
        R = int(np.ceil(rs.max())) + 1
        oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
        ox, oy = ox.ravel(), oy.ravel()
        HH, WW = H * SS, W * SS
        rgb = np.zeros((HH * WW, 3), np.float32)
        cov = np.zeros(HH * WW, np.float32)
        low = np.zeros(HH * WW, np.float32)
        dep = np.zeros(HH * WW, np.float32)
        L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
        chunk = max(1, 3_000_000 // len(ox))
        for a in range(0, len(sel), chunk):
            b = slice(a, a + chunk)
            px = np.floor(cu[b])[:, None] + ox[None, :]
            py = np.floor(cv_[b])[:, None] + oy[None, :]
            dx = (px + 0.5 - cu[b][:, None]) / rs[b][:, None]
            dy = (py + 0.5 - cv_[b][:, None]) / rs[b][:, None]
            d2 = dx * dx + dy * dy
            m = (d2 < 1.0) & (px >= 0) & (px < WW) & (py >= 0) & (py < HH)
            nz = np.sqrt(np.clip(1 - d2[m], 0, 1))
            lam = np.clip(L[0] * dx[m] + L[1] * dy[m] + L[2] * nz, 0, 1)
            shd = 0.32 + 0.78 * lam + 0.28 * lam ** 16
            idx = (py[m] * WW + px[m]).astype(np.int64)
            gi = np.nonzero(m)[0] + a
            rgb[idx] = base[gi] * shd[:, None]
            cov[idx] = 1.0
            low[idx] = (zs[gi] <= C.WALL_H)
            dep[idx] = ds[gi]
        rgb = cv2.resize(rgb.reshape(HH, WW, 3), (W, H), interpolation=cv2.INTER_AREA)
        cov2 = cov.reshape(HH, WW)
        dep = cv2.resize(dep.reshape(HH, WW) * cov2, (W, H), interpolation=cv2.INTER_AREA)
        cov = cv2.resize(cov2, (W, H), interpolation=cv2.INTER_AREA)
        low = cv2.resize(low.reshape(HH, WW), (W, H), interpolation=cv2.INTER_AREA)[..., None]
        dep = dep / np.maximum(cov, 1e-4)
        return rgb, cov[..., None], low, dep

    def plane_depth(self):
        """Camera depth of the table under every pixel."""
        F = C.FOCAL_PX
        uu = (np.arange(W, dtype=np.float32) - W / 2 + 0.5) / F
        vv = (np.arange(H, dtype=np.float32) - H / 2 + 0.5) / F
        dz = self.cf[2] + uu[None, :] * self.cr[2] - vv[:, None] * self.cu[2]
        return np.where(dz < -1e-4, -self.loc[2] / np.minimum(dz, -1e-4), 1e3).astype(np.float32)

    def dof(self, img, depth):
        """Depth of field of the same lens as the Blender close-ups (f/0.9, focused on the target)."""
        df = self.pose[3]
        f = C.LENS
        coc = (f / 0.9) * np.abs(depth - df) / depth * f / (df * 1000.0 - f)        # mm on the sensor
        sig = np.clip(coc / C.SENSOR * W * 0.5, 0, 6.0)
        if sig.max() < 0.35:
            return img
        levels = [0.0, 0.9, 1.8, 3.0, 4.5, 6.0]
        out = np.zeros_like(img)
        for i, sl in enumerate(levels):
            lo_ = levels[i - 1] if i else None
            hi_ = levels[i + 1] if i + 1 < len(levels) else None
            w = np.zeros_like(sig)
            if lo_ is not None:
                w = np.where((sig > lo_) & (sig <= sl), (sig - lo_) / (sl - lo_), w)
            if hi_ is not None:
                w = np.where((sig > sl) & (sig < hi_), (hi_ - sig) / (hi_ - sl), w)
            if i == 0:
                w = np.where(sig <= 0, 1.0, w)
            if i == len(levels) - 1:
                w = np.where(sig >= sl, 1.0, w)
            if not w.any():
                continue
            bl = img if sl == 0 else cv2.GaussianBlur(img, (0, 0), sl)
            out += bl * w[..., None]
        return out


    # ---- lifted grains
    @staticmethod
    def rise_pos(p, idx, u):
        """Where lifted grain idx of batch p is, u seconds after it was taken (map px; z up). It glows on
        its pile for TG seconds and lifts a little; then it is drawn in toward a column over the place
        the catastrophe struck, turning as it goes, and pulled up faster and faster. A grain whose person
        died far from home (a soldier, a deportee: mv > 0) first travels there in a low arc, for mv
        seconds, and rises from beside the place."""
        x0, y0, z0 = p["x"][idx], p["y"][idx], p["z"][idx]
        cx, cy = p["cx"][idx], p["cy"][idx]
        ph = p["ph"][idx]
        mv = p["mv"][idx] if "mv" in p else np.zeros(len(idx))
        g = np.clip(u / TG, 0, 1)
        lift0 = 2.5 * g * g * (3 - 2 * g)
        trav = mv > 0
        if trav.any():
            # where it lands: a little ring around the place, then it rises from there
            ra = 3.0 + 7.0 * ph[:, 1]
            ax = np.where(trav, cx + ra * np.cos(6.283 * ph[:, 2]), x0)
            ay = np.where(trav, cy + ra * np.sin(6.283 * ph[:, 2]), y0)
            az = np.where(trav, 3.0, z0)
            tm = np.where(trav, np.clip((u - TG) / np.maximum(mv, 1e-6), 0, 1), 1.0)
            e = tm * tm * (3 - 2 * tm)
            dist = np.hypot(ax - x0, ay - y0)
            hop = (6.0 + 0.10 * dist) * 4 * tm * (1 - tm)
            xs = x0 + (ax - x0) * e
            ys = y0 + (ay - y0) * e
            zs = z0 + (az - z0) * e + hop + lift0 * (1 - e) + 2.5 * e
            bx, by, bz = ax, ay, az + 2.5                      # where the rise starts
        else:
            xs, ys, zs = x0, y0, z0 + lift0
            bx, by, bz = x0, y0, z0 + 2.5
        tau = np.maximum(u - TG - mv, 0.0)
        z = bz + 14.0 * tau + 165.0 * (0.8 + 0.4 * ph[:, 0]) * tau ** 2
        ox, oy = bx - cx, by - cy
        r0 = np.hypot(ox, oy)
        th0 = np.arctan2(oy, ox)
        kk = 1 - np.exp(-tau / 0.85)
        r = r0 * (1 - 0.62 * kk) + 2.0 * np.sin(5.0 * tau + 6.283 * ph[:, 2]) * np.minimum(tau, 1.0)
        th = th0 + (1.1 + 0.9 * ph[:, 1]) * kk + 0.35 * tau
        rising = tau > 0
        x = np.where(rising, cx + r * np.cos(th), xs)
        y = np.where(rising, cy + r * np.sin(th), ys)
        z = np.where(rising, z, zs)
        return x, y, z

    def particles(self, sim, s, top, glow):
        """Draw every lifted grain still in view: a ball glowing in its cause's colour on the pile, then a
        streak with a fading tail as it rises (motion blur), growing as it nears the camera."""
        cut = 0
        for p in sim.particles:
            u = s - p["t0"]
            act = (u >= 0) & (u < Sd.LIFE)
            if not act.any():
                continue
            idx = np.where(act)[0]
            uu = u[idx]
            x, y, z = self.rise_pos(p, idx, uu)
            Xw = np.stack([x / 100.0 - 9.6, 5.4 - y / 100.0, z / 100.0], 1) - self.loc
            d = Xw @ self.cf
            sx, sy = self.project(x, y, z)
            ok = (d > 1.5) & (sx > -60) & (sx < W + 60) & (sy > -60) & (sy < H + 60)
            if not ok.any():
                continue
            idx, uu, x, y, z, d, sx, sy = idx[ok], uu[ok], x[ok], y[ok], z[ok], d[ok], sx[ok], sy[ok]
            rho = (p["r"][idx] / 100.0) * C.FOCAL_PX / np.maximum(d, 1e-3)
            g = np.clip(uu / TG, 0, 1)
            col = SANDCOL[None, :] * (1 - g[:, None]) + CCOL[p["cause"][idx]] * g[:, None]
            bright = (1.0 + 0.2 * g) * np.clip(1.1 - (uu - TG) / 8.0, 0.7, 1.1)
            bright = np.minimum(bright, 1.0 / np.maximum(col.max(1), 1e-3))     # no channel past 1: hue holds
            # the head (half a frame of blur), then a tail that fades: (time back, alpha, width)
            pts = [(sx, sy)]
            for back in (0.5 / FPS, 2.0 / FPS, 4.0 / FPS):
                xb, yb, zb = self.rise_pos(p, idx, np.maximum(uu - back, 0))
                pts.append(self.project(xb, yb, zb))
            passes = ((0, 1, 1.0, 1.0), (1, 2, 0.55, 0.75), (2, 3, 0.25, 0.55))
            for i in range(len(idx)):
                w0 = max(1, int(round(rho[i] * 1.7)))
                for (pa, pb, al, wk) in passes:
                    if pa > 0 and uu[i] < TG + 0.05:
                        break                                   # no tail while it is still on the pile
                    c = tuple(float(v) for v in col[i] * bright[i] * al)
                    p0 = (int(pts[pb][0][i] * 8), int(pts[pb][1][i] * 8))
                    p1 = (int(pts[pa][0][i] * 8), int(pts[pa][1][i] * 8))
                    if pa == 0 and abs(p1[0] - p0[0]) + abs(p1[1] - p0[1]) < 6:
                        cv2.circle(top, p1, max(int(rho[i] * 8), 6), c, -1, cv2.LINE_AA, shift=3)
                    else:
                        cv2.line(top, p0, p1, c, max(1, int(round(w0 * wk))), cv2.LINE_AA, shift=3)
                gc = tuple(float(v) * 0.5 * float(g[i]) for v in col[i])
                cv2.circle(glow, (int(sx[i] * 4), int(sy[i] * 4)), int(max(4.0, rho[i] * 2.4) * 2), gc, -1,
                           cv2.LINE_AA, shift=3)
                cut += 1
        return cut

    # ---- frame
    def frame(self, sim, s):
        self.set_camera(s)
        bare, ratio, spec = self.plates.at(s, self.P)
        self.update_disp(sim)
        self.shadows(sim)
        top = np.zeros((H, W, 3), np.float32)
        glow = np.zeros((H // 2, W // 2, 3), np.float32)
        self.particles(sim, s, top, glow)
        img = bare * self.table_shadow * ratio + spec
        depth = self.plane_depth()
        g = self.grains(sim, s)
        if g is not None:
            grgb, gcov, glow_, gdep = g
            grgb = grgb * self.illum
            f = glow_ / np.maximum(gcov, 1e-4)
            img = img * (1 - gcov) + grgb * (ratio * f + (1 - f)) + spec * glow_
            depth = np.where(gcov[..., 0] > 0.5, gdep, depth)
        tm = np.clip(top.max(2, keepdims=True) * 1.4, 0, 1)
        img = img * (1 - tm) + top                                 # lifted grains, above the glass
        gl = cv2.GaussianBlur(glow, (0, 0), 2.5) * 0.7 + cv2.GaussianBlur(glow, (0, 0), 9.0) * 0.6
        img += cv2.resize(gl, (W, H), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        lum = small.max(2, keepdims=True)                     # bloom keeps each colour's hue
        bloom = cv2.GaussianBlur(small * (np.maximum(lum - 0.55, 0) / np.maximum(lum, 1e-4)), (0, 0), 4.0)
        img += cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR) * 0.5
        img *= self.vign
        img = self.dof(img, depth)
        img = img * 255.0
        self.hud(img, sim, s)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---- on-screen text
    def chip(self, img, x, y, col, a):
        k = self.k
        reg = img[y + 4 * k:y + 17 * k, x + 1 * k:x + 14 * k]
        chip = np.zeros(reg.shape[:2], np.float32)
        cv2.circle(chip, (6 * k * 16, 6 * k * 16), 6 * k * 16, 1.0, -1, cv2.LINE_AA, shift=4)
        m = chip[..., None] * a
        reg *= (1 - m); reg += col * 255 * m

    def hud(self, img, sim, s, title=True):
        k = self.k
        IW, IH = img.shape[1], img.shape[0]
        title_a = 1.0 - smooth((s - 4.6) / 1.6)
        end0 = S.TODAY_S + 2.0
        end_a = smooth((s - end0) / 1.5)
        live_a = 1.0 - 0.75 * end_a
        # ---- event names on the map
        self.place_tags(img, s, (1 - title_a) * (1 - end_a))
        self.event_labels(img, sim, s, (1 - title_a) * (1 - end_a))
        # ---- the year, the scale and the running count
        yr = int(math.floor(year_at(s)))
        yr = min(yr, S.YEAR1)
        spr = self.text_sprite(f"{yr}", self.f_year, INK * 255)
        self.blit(img, spr, 44 * k, 26 * k, 0.95 * live_a)
        self.blit(img, self.text_sprite("CE", self.f_year_sub, MUTED * 255), 48 * k, 26 * k + spr.shape[0] + 2 * k,
                  0.9 * live_a)
        y = 26 * k + spr.shape[0] + 30 * k
        self.blit(img, self.text_sprite(f"1 grain = {S.GRAIN:,} people", self.f_year_sub, INK * 255), 48 * k, y,
                  0.85 * live_a)
        if PACK_NOTE:                                          # the packed regions, said plainly
            self.blit(img, self.text_sprite(PACK_NOTE, self.f_note, MUTED * 255, halo=True), 42 * k, y + 50 * k,
                      0.75 * live_a * (1 - title_a))
        lifted = sim.lifted.sum() * S.GRAIN
        if lifted > 0:
            txt = f"Lifted so far: {fmt_people(lifted)}"
            self.blit(img, self.text_sprite(txt, self.f_count, INK * 255), 48 * k, y + 30 * k,
                      0.85 * live_a * smooth((s - S.INTRO_S) / 1.0))
        # ---- legend: the causes, with what each has taken so far
        lx, ly = 40 * k, 870 * k
        self.blit(img, self.text_sprite("LIFTED SAND: DEATHS ABOVE NORMAL, BY CAUSE", self.f_leg_h, MUTED * 255),
                  lx, ly - 24 * k, 0.8 * live_a)
        for c, (key, lab, _) in enumerate(S.CAUSES):
            yy = ly + c * 19 * k
            self.chip(img, lx, yy, CCOL[c], live_a)
            n = sim.lifted[c] * S.GRAIN
            t = lab + (f"   {fmt_people(n, short=True)}" if n > 0 else "")
            self.blit(img, self.text_sprite(t, self.f_leg, INK * 255), lx + 20 * k, yy, 0.85 * live_a)
        # ---- sources
        n1 = self.text_sprite("Excess deaths: UN World Population Prospects 2024 from 1950; published estimates of "
                              "each catastrophe before.", self.f_note, MUTED * 255)
        n2 = self.text_sprite("Population: HYDE 3.3, Gapminder, UN WPP 2024. Normal = each place's trend death rate.",
                              self.f_note, MUTED * 255)
        self.blit(img, n1, IW - n1.shape[1] - 40 * k, IH - 40 * k - 2 * n1.shape[0], 0.6 * live_a * (1 - title_a))
        self.blit(img, n2, IW - n2.shape[1] - 40 * k, IH - 40 * k - n1.shape[0], 0.6 * live_a * (1 - title_a))
        # ---- title
        if title and title_a > 0.003:
            self.shade(img, 0, 0, IW, IH, 0.55 * title_a)
            t = self.text_sprite(S.TITLE, self.f_title, INK * 255)
            self.blit(img, t, IW / 2 - t.shape[1] / 2, 420 * k, title_a)
            if S.SUBTITLE:
                st = self.text_sprite(S.SUBTITLE, self.f_sub, MUTED * 255)
                self.blit(img, st, IW / 2 - st.shape[1] / 2, 530 * k, title_a)
        # ---- the end: what was lifted, in all
        if end_a > 0.003:
            self.shade(img, 0, 0, IW, IH, 0.5 * end_a)
            tot = sim.lifted.sum() * S.GRAIN
            t = self.text_sprite(f"{fmt_people(tot)} deaths above normal since year 1", self.f_end, INK * 255)
            self.blit(img, t, IW / 2 - t.shape[1] / 2, 380 * k, end_a)
            parts = "   ".join(f"{lab.split(',')[0].split(' &')[0]} {fmt_people(sim.lifted[c] * S.GRAIN, short=True)}"
                               for c, (key, lab, _) in enumerate(S.CAUSES) if sim.lifted[c] > 0)
            t2 = self.text_sprite(parts, self.f_end_sub, MUTED * 255)
            self.blit(img, t2, IW / 2 - t2.shape[1] / 2, 460 * k, end_a)
            t3 = self.text_sprite("Before 1950, only catastrophes big enough to have been counted.", self.f_end_sub,
                                  MUTED * 255)
            self.blit(img, t3, IW / 2 - t3.shape[1] / 2, 510 * k, end_a * smooth((s - end0 - 1.5) / 1.2))

    def place_tags(self, img, s, a_all):
        """A small name under each big place while its sand rises: "Stalingrad", "Auschwitz", "Lombardy"."""
        self._tags = []
        if a_all <= 0.003:
            return
        k = self.k
        placed = self._tags = []
        for pg in self.places:
            if len(placed) >= PLACE_LIVE:
                break
            if not (pg["s0"] - 0.2 <= s <= pg["s1"] + 0.5):
                continue
            a = smooth((s - pg["s0"] + 0.2) / 0.4) * smooth((pg["s1"] + 0.5 - s) / 0.5) * a_all
            if a < 0.01:
                continue
            sx, sy = self.project(np.array([pg["xy"][0]]), np.array([pg["xy"][1]]), np.array([0.0]))
            x, y = float(sx[0]) * k, float(sy[0]) * k + 9 * k
            txt = f"{pg['name']}  {fmt_people(pg['d'], short=True)}"
            t = self.text_sprite(txt, self.f_place, CCOL[pg["cause"]] * 255 * 0.45 + INK * 255 * 0.55, halo=True)
            bx, by, bw, bh = x - t.shape[1] / 2, y, t.shape[1], t.shape[0]
            if any(bx < b[2] and b[0] < bx + bw and by < b[3] and b[1] < by + bh for b in placed):
                continue
            placed.append((bx, by, bx + bw, by + bh))
            self.blit(img, t, bx, by, 0.92 * a)

    def event_labels(self, img, sim, s, a_all):
        """Name each big catastrophe while it lifts sand (and a little after), above where it struck."""
        if a_all <= 0.003:
            return
        k = self.k
        yr = year_at(s)
        placed = list(getattr(self, "_tags", []))            # step aside from place tags too
        for ev in self.events:
            s0, s1 = ev["s0"], max(ev["s1"], ev["s0"] + LABEL_HOLD)
            if not (s0 - 0.3 <= s <= s1 + 0.6):
                continue
            a = smooth((s - s0 + 0.3) / 0.5) * smooth((s1 + 0.6 - s) / 0.6) * a_all
            if a < 0.01:
                continue
            # where: the deaths-weighted centre of its sources striking now (else all of them)
            w = np.array([max(min(yr, y1) - max(yr - 1, y0), 0) * d / (y1 - y0) for (y0, y1, d) in ev["spans"]])
            if w.sum() <= 0:
                w = np.array([d for (_, _, d) in ev["spans"]])
            if ev["world"]:
                x, y = img.shape[1] / 2, 150 * k
            else:
                j = int(np.argmax(w))                      # over the region it is hitting hardest now
                near = np.hypot(ev["cx"] - ev["cx"][j], ev["cy"] - ev["cy"][j]) < 120
                wn = w * near
                cx = float((wn * ev["cx"]).sum() / wn.sum()); cy = float((wn * ev["cy"]).sum() / wn.sum())
                sx, sy = self.project(np.array([cx]), np.array([cy]), np.array([0.0]))
                x, y = float(sx[0]) * k, float(sy[0]) * k - 34 * k
            t1 = self.text_sprite(ev["name"], self.f_event, CCOL[ev["cause"]] * 255 * 0.35 + INK * 255 * 0.65, halo=True)
            t2 = self.text_sprite(ev["sub"], self.f_event_sub, INK * 255, halo=True)
            bw, bh = max(t1.shape[1], t2.shape[1]), t1.shape[0] + t2.shape[0] - 8 * k
            bx, by = x - bw / 2, y - bh
            for _ in range(6):                                  # step aside from names already placed
                hit = [b for b in placed if bx < b[2] and b[0] < bx + bw and by < b[3] and b[1] < by + bh]
                if not hit:
                    break
                by = max(b[3] for b in hit) + 2 * k
            else:
                continue
            placed.append((bx, by, bx + bw, by + bh))
            self.blit(img, t1, x - t1.shape[1] / 2, by, a)
            self.blit(img, t2, x - t2.shape[1] / 2, by + t1.shape[0] - 8 * k, 0.9 * a)


def _pack_note():
    parts = [(S.VESSELS[v]["label"], S.GRAIN / k) for v, k in zip(Sd.ORDER, Sd.PACK) if k < 0.999]
    if not parts:
        return ""
    txt = ", ".join(f"{lab} {round(n, -2):,.0f}" for lab, n in parts)
    return f"Piles packed to fit their glass (people per grain): {txt}"


PACK_NOTE = _pack_note()


def fmt_people(n, short=False):
    if n >= 1e9:
        return f"{n / 1e9:.2f} billion" if not short else f"{n / 1e9:.2f} bn"
    if n >= 1e6:
        return f"{n / 1e6:,.0f} million" if not short else f"{n / 1e6:,.0f} M"
    return f"{n / 1e3:,.0f} thousand" if not short else f"{n / 1e3:,.0f} k"


def lift_xy(l):
    """Where a lift's sand rises (map px): its place, its spot, or its country's centre."""
    if l["kind"] == "converge":
        x, y = to_px(l["to_lon"], l["to_lat"])
    elif l["kind"] == "spot":
        x, y = to_px(l["lon"], l["lat"])
    else:
        return Sd.CC_CENTRE.get(l["iso"]) or tuple(Sd.META[l["vessel"]]["anchor"])
    return (float(np.ravel(x)[0]), float(np.ravel(y)[0]))


def tag_name(place):
    """A place's on-screen tag: short ("Leningrad: hunger winter" -> "Leningrad", "Rhineland, northern
    Germany and the Low Countries" -> "Rhineland", "Central Mexico, 1522-44" -> "Central Mexico")."""
    import re
    n = place.split(":")[0]
    n = re.sub(r"\s*\([^)]*\)", "", n)
    n = re.sub(r",?\s*\d{3,4}(\s*[-\u2013]\s*\d{2,4})?$", "", n).strip()
    n = re.sub(r"\s+(of|in|on)$", "", n)
    if len(n) > 30 and "," in n:
        n = n.split(",")[0]
    if len(n) > 30 and " and " in n:
        n = n.split(" and ")[0]
    return n.strip() or place


def place_groups(L):
    """Named places (a battle, a camp, a hotspot) big enough to be tagged while their sand rises."""
    groups = {}
    for l in L:
        if not l.get("place"):
            continue
        g = groups.setdefault((l["id"], l["place"]), dict(name=tag_name(l["place"]), cause=l["cause"], d=0.0, a=1e9,
                                                          b=-1e9))
        g["d"] += l["deaths"]
        a, b = PL.span(l)
        g["a"], g["b"] = min(g["a"], a), max(g["b"], b)
        x, y = to_px(l["plon"], l["plat"])
        g["xy"] = (float(np.ravel(x)[0]), float(np.ravel(y)[0]))
    out = []
    for (gid, _), g in groups.items():
        if g["d"] < PLACE_MIN:
            continue
        s0, s1 = s_at(g["a"]), s_at(g["b"])
        s1 = min(max(s1, s0 + PLACE_HOLD), s0 + PLACE_MAX)
        out.append(dict(id=gid, name=g["name"], d=g["d"], s0=s0, s1=s1, xy=g["xy"],
                        cause=S.CAUSE_KEYS.index(g["cause"]) if g["cause"] in S.CAUSE_KEYS else 4))
    out.sort(key=lambda e: -e["d"])
    return out


def event_groups(L):
    """The named catastrophes big enough to be labelled: each event's sources (all its regions, or all its
    country-years), its total, its years and where it struck."""
    groups = {}
    for i, l in enumerate(L):
        if not l.get("id") or not l.get("name"):
            continue
        g = groups.setdefault(l["id"], dict(name=l.get("label") or l["name"], cause=l["cause"], spans=[], xy=[],
                                            vessels=set()))
        xy = lift_xy(l)
        if l["deaths"] > g.get("biggest", 0) and not l.get("place"):     # named after its largest part
            g["biggest"] = l["deaths"]
            g["name"] = l.get("label") or l["name"]
        g["spans"].append((*PL.span(l), l["deaths"]))
        g["xy"].append(xy)
        g["vessels"].add(l["vessel"])
    out = []
    for gid, g in groups.items():
        tot = sum(d for _, _, d in g["spans"])
        y0 = int(math.floor(min(a for a, _, _ in g["spans"]))); y1 = int(math.ceil(max(b for _, b, _ in g["spans"])))
        rate = np.zeros(y1 - y0)                                # deaths per calendar year
        for a, b, d in g["spans"]:
            rate += PL.per_year(dict(t0=a, t1=b, y0=0, y1=0, deaths=d), y0, y1 - y0)
        if tot < LABEL_MIN or rate.max() < LABEL_RATE:
            continue                                            # small, or a slow drizzle over centuries
        hot = np.where(rate >= 0.35 * rate.max())[0]            # named while it is near its worst
        ya, yb = y0 + hot[0], y0 + hot[-1] + 1
        s0 = s_at(ya)
        yrs = f"{y0}" if y1 - 1 == y0 else f"{y0}\u2013{y1 - 1}"
        xy = np.array(g["xy"])
        out.append(dict(id=gid, name=g["name"], cause=S.CAUSE_KEYS.index(g["cause"]) if g["cause"] in S.CAUSE_KEYS else 4,
                        spans=g["spans"], cx=xy[:, 0], cy=xy[:, 1], total=tot,
                        world=g["cause"] == "epidemic" and len(g["vessels"]) >= 10,   # pandemics: top centre
                        s0=s0, s1=min(s_at(yb), s0 + LABEL_MAX), sub=f"{yrs}  \u00b7  {fmt_people(tot)}"))
    out.sort(key=lambda e: -e["total"])
    return out


# ============================================================== drivers
def checkpoint_key():
    """Simulation checkpoints are valid while the data and the simulation code are unchanged."""
    import hashlib
    h = hashlib.sha1()
    for p in ("data/lifts.json", "data/series.npz", "data/clock.json", "sand.py", "story.py"):
        h.update(open(os.path.join(HERE, p), "rb").read())
    return h.hexdigest()[:12]


def resume(before_s):
    """The latest saved simulation state at or before video second before_s (or a fresh one)."""
    import pickle, glob
    d = os.path.join(DATA, "ckpt", checkpoint_key())
    best = None
    for p in glob.glob(os.path.join(d, "*.pkl")):
        t = float(os.path.basename(p)[:-4])
        if t <= before_s + 1e-9 and (best is None or t > best[0]):
            best = (t, p)
    if best:
        return pickle.load(open(best[1], "rb"))
    return Sd.Sand()


def save_ckpt(sim):
    import pickle
    d = os.path.join(DATA, "ckpt", checkpoint_key())
    os.makedirs(d, exist_ok=True)
    sim.L_w = {}                                    # (cache: rebuilt on demand)
    pickle.dump(sim, open(os.path.join(d, f"{sim.s:09.3f}.pkl"), "wb"), protocol=4)


def cmd_still(args):
    os.makedirs(OUT, exist_ok=True)
    want = sorted(float(v) for v in args.times)
    sim = resume(want[0] - 3.0)
    painter = Painter(sim=sim)
    k = 0
    f_start = int(round(sim.s * FPS)) + (1 if sim.s > 0 else 0)
    next_ck = (math.floor(sim.s / 10.0) + 1) * 10.0
    for f in range(f_start, int(round(S.DURATION * FPS)) + 1):
        s = f / FPS
        sim.step(s)
        if s >= next_ck - 1e-9:
            save_ckpt(sim)
            next_ck += 10.0
        if want and s > want[k] - 2.5:
            painter.update_disp(sim)
        while k < len(want) and want[k] <= s + 1e-9:
            t0 = time.time()
            im = painter.frame(sim, s)
            p = os.path.join(OUT, f"still_{want[k]:07.2f}.png")
            cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            print("wrote", p, f"(year {year_at(s):.1f}, {time.time() - t0:.2f}s to paint)", flush=True)
            k += 1
        if k >= len(want):
            return


def render_chunk(job):
    f0, f1, path = job
    sim = Sd.Sand()
    painter = Painter(sim=sim)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "18", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f1):
        s = f / FPS
        sim.step(s)
        if f >= f0 - 60:
            painter.update_disp(sim)
        if f >= f0:
            ff.stdin.write(painter.frame(sim, s).tobytes())
            if (f - f0) % 300 == 0:
                print(f"  [{f0}-{f1}] frame {f}  {time.time() - t0:5.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    return path


def cmd_video(args):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    out = args.out or os.path.join(OUT, FILM_NAME + ".mp4")
    a = int(round(args.from_s * FPS))
    n = int(round((args.to_s if args.to_s else S.DURATION) * FPS))
    nw = args.workers
    # every worker re-simulates from the start, so later chunks paint fewer frames
    w = np.array([1.0 / (1 + 0.15 * k) for k in range(nw)])
    edges = np.concatenate([[a], a + np.round(np.cumsum(w / w.sum()) * (n - a))]).astype(int)
    jobs = [(int(edges[k]), int(edges[k + 1]), os.path.join(OUT, f"chunk_{k}.mp4")) for k in range(nw)]
    t0 = time.time()
    with Pool(nw) as pool:
        parts = pool.map(render_chunk, jobs)
    lst = os.path.join(OUT, "chunks.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.basename(p)}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out], check=True)
    for p in parts + [lst]:
        os.remove(p)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("still"); a.add_argument("times", nargs="+")
    b = sp.add_parser("video"); b.add_argument("--workers", type=int, default=4)
    b.add_argument("--from", dest="from_s", type=float, default=0.0)
    b.add_argument("--to", dest="to_s", type=float, default=None)
    b.add_argument("--out", default=None)
    args = ap.parse_args()
    {"still": cmd_still, "video": cmd_video}[args.cmd](args)
