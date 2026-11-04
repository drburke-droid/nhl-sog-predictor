"""Paint "Grains of Ancestry" (glass edition): the sandpile simulation (sand.py) composited into the
Blender-rendered glass world (blender_scene.py plates), with wind-blown sand flying between vessels.

    python film.py still 40 75 165         # PNG stills -> out/
    python film.py video --workers 4       # the film -> out/grains_of_ancestry_glass.mp4

Per frame: the sand surface is shaded in map space at 2x (pile lighting from the height field, ambient
occlusion, cast shadows, grain texture), projected into the camera with the plane homography,
then laid under the glass (the plates' ratio = glass tint, difference = glass highlights). Flying
grains follow 3D arcs with turbulence, projected with the full camera matrix, and cast shadows.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import story as S
import sand as Sd
from sand import smooth, ybp_at, CELL, WC, HC, K
from proj import W, H

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
FPS = 30
FONT_DIR = "/usr/share/fonts/opentype/inter"
UP = 2 * CELL                     # sand texture pixels per cell (2x map resolution)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


COL = np.stack([hex_rgb(c[2]) for c in S.COMPONENTS])
INK = hex_rgb("#e8eef8")
MUTED = hex_rgb("#93a1b8")


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def load_plate(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        p = p.replace(".png", "_test.png")
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
    if im.shape[1] != W:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
    return im


CAM = np.array(json.load(open(os.path.join(DATA, "camera.json")))["P"])


def project(x, y, z):
    """Map px (x, y) at height z (map px) -> screen px."""
    q = CAM @ np.stack([x, y, z, np.ones_like(x)])
    return q[0] / q[2], q[1] / q[2]


# homography from the 2x sand texture (z = 0) to the screen
HOM = CAM[:, [0, 1, 3]] @ np.diag([1.0 / 2, 1.0 / 2, 1.0])
HOM = HOM / HOM[2, 2]


# ============================================================== static sand texture
def grain_texture(seed=5):
    """Randomly scattered grains of varied size (2x map resolution), each lit from the upper left.
    Returns (shade, coverage, owner): owner = the cell under each grain's centre, which gives the
    grain its colour, so grains are whole even where they straddle cells."""
    rng = np.random.default_rng(seed)
    hh, ww = HC * UP, WC * UP
    n = int(hh * ww / (UP * UP) * 1.9)
    xs = rng.uniform(0, ww, n); ys = rng.uniform(0, hh, n)
    rs = rng.gamma(6.0, 0.42, n) + 1.0                     # radii ~ 1.6-4 px, a few larger
    order = np.argsort(rs)[::-1]                           # big grains first, small ones on top
    owner_g = np.full((hh, ww), -1, np.int32)
    shade = np.full((hh, ww), 0.50, np.float32)
    cov = np.zeros((hh, ww), np.float32)
    for i in order:
        x0, y0, r = xs[i], ys[i], rs[i]
        xa, xb = int(max(0, x0 - r - 1)), int(min(ww, x0 + r + 2))
        ya, yb = int(max(0, y0 - r - 1)), int(min(hh, y0 + r + 2))
        if xa >= xb or ya >= yb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        dx, dy = (xx + 0.5 - x0) / r, (yy + 0.5 - y0) / r
        d2 = dx * dx + dy * dy
        c = np.clip((1.0 - d2) * r * 0.9, 0, 1)
        if not c.any():
            continue
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        lam = np.clip(-0.55 * dx - 0.6 * dy + 0.58 * nz, 0, 1)
        sh = 0.60 + 0.55 * lam + 0.30 * lam ** 14
        reg = shade[ya:yb, xa:xb]
        reg[:] = reg * (1 - c) + sh * c
        cov[ya:yb, xa:xb] = np.maximum(cov[ya:yb, xa:xb], c)
        owner_g[ya:yb, xa:xb][c > 0.5] = i
    cell = (np.minimum(ys // UP, HC - 1) * WC + np.minimum(xs // UP, WC - 1)).astype(np.int64)
    own = np.where(owner_g >= 0, cell[np.maximum(owner_g, 0)], 
                   (np.arange(hh)[:, None] // UP) * WC + np.arange(ww)[None, :] // UP)
    return shade, cov, own.astype(np.int32)


# ============================================================== painter
class Painter:
    def __init__(self):
        self.bare = load_plate("plate_bare.png")
        self.glass = load_plate("plate_glass.png")
        self.ratio = np.clip((self.glass + 0.004) / (self.bare + 0.004), 0.42, 1.0)
        self.spec = np.clip(self.glass - self.bare, 0, None)
        cache = os.path.join(DATA, "grain_texture.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            self.tex, self.cov, self.owner = z["tex"], z["cov"], z["own"]
        else:
            self.tex, self.cov, self.owner = grain_texture()
            np.savez(cache, tex=self.tex, cov=self.cov, own=self.owner)
        self.tex8 = cv2.merge([np.clip(self.tex * 128, 0, 255).astype(np.uint8)] * 3)
        self.cov8 = np.clip(self.cov * 255, 0, 255).astype(np.uint8)
        self.covpile8 = np.clip(np.maximum(self.cov, 0.85) * 255, 0, 255).astype(np.uint8)
        rng = np.random.default_rng(9)
        self.cell_tint = (0.86 + 0.24 * rng.random((HC, WC))).astype(np.float32)
        self.disp, self.disp_t = None, None
        self.prev_h = np.zeros((HC, WC), np.int16)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.illum = (0.80 + 0.32 * (1 - (xx / W) * 0.55 - (yy / H) * 0.45))[..., None]   # key light upper left
        r2 = ((xx / W - 0.5) / 0.78) ** 2 + ((yy / H - 0.5) / 0.8) ** 2
        self.vign = (1 - 0.32 * np.clip(r2, 0, 1.5) ** 1.6)[..., None].astype(np.float32)
        self.f_label = font("Inter-SemiBold.otf", 14)
        self.f_year = font("InterDisplay-SemiBold.otf", 58)
        self.f_year_sub = font("Inter-Medium.otf", 20)
        self.f_cap = font("Inter-Medium.otf", 29)
        self.f_leg = font("Inter-Medium.otf", 14)
        self.f_leg_h = font("Inter-SemiBold.otf", 12)
        self.f_title = font("InterDisplay-SemiBold.otf", 84)
        self.f_sub = font("Inter-Regular.otf", 28)
        self.f_note = font("Inter-Regular.otf", 15)
        self.cache = {}
        self.label_pos = {}
        for v in Sd.ORDER:
            m = Sd.META[v]
            if m["inset"]:
                bx0, by0, bx1, by1 = m["bbox"]
                x, y = project(np.array([(bx0 + bx1) / 2]), np.array([by1 + 12.0]), np.array([0.0]))
            else:
                x, y = project(np.array([m["anchor"][0]]), np.array([m["anchor"][1]]), np.array([0.0]))
            self.label_pos[v] = (float(x[0]), float(y[0]))

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
        lh = asc + desc + spacing
        pad = 7 if halo else 2
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
            hal = np.clip(cv2.GaussianBlur(cv2.dilate(a, np.ones((3, 3), np.uint8)), (0, 0), 2.2) * 1.5, 0, 0.85)
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

    @staticmethod
    def shade(img, x, y, w, h, a):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
        if a > 0.003 and x0 < x1 and y0 < y1:
            img[y0:y1, x0:x1] *= (1 - a)

    # ---- the sand surface
    def update_disp(self, sim):
        """Per-cell displayed colour, eased toward the current top grain (call every frame)."""
        if self.disp_t == sim.s:
            return
        top = sim.top_colors().reshape(HC, WC)
        target = COL[np.maximum(top, 0)] * self.cell_tint[..., None]
        if self.disp is None:
            self.disp = target.astype(np.float32)
        else:
            new = (self.prev_h == 0) & (top >= 0)              # freshly covered cells take their colour at once
            self.disp += (target - self.disp) * 0.18
            self.disp[new] = target[new]
        self.prev_h = sim.h.reshape(HC, WC).copy()
        self.disp_t = sim.s

    def sand_layer(self, sim, s):
        hgt = sim.h.reshape(HC, WC).astype(np.float32)
        top = sim.top_colors().reshape(HC, WC)
        occ = hgt > 0
        alb = COL[np.maximum(top, 0)] * self.cell_tint[..., None]
        self.update_disp(sim)
        # lighting from the pile surface (cell resolution, then smooth upsampling)
        hs = cv2.GaussianBlur(hgt, (0, 0), 0.8)
        gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3) / 8
        k = 1.4
        nx, ny, nz = -k * gx, -k * gy, np.ones_like(gx)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        L = np.array([-0.55, -0.6, 0.58]); L /= np.linalg.norm(L)
        lam = np.clip((nx * L[0] + ny * L[1] + nz * L[2]) / nn, 0, 1)
        ao = np.clip(1 - 0.16 * np.maximum(cv2.GaussianBlur(hs, (0, 0), 2.5) - hs, 0), 0.55, 1)
        shadow = np.ones_like(hs)
        for j in range(1, 6):
            sh = np.roll(np.roll(hs, j, 0), j, 1)           # neighbour toward the light
            shadow = np.minimum(shadow, np.where(sh - 0.55 * j > hs + 0.4, 0.72, 1.0))
        shadow = cv2.GaussianBlur(shadow, (0, 0), 0.8)
        light = (0.38 + 0.82 * lam) * ao * shadow
        # colours ease toward the top grain's colour, so stirring and drift fade in instead of flickering
        alb = self.disp if self.disp is not None else alb
        cell_rgb = alb * light[..., None]
        # the closing ripple of light from East Africa
        tw0 = S.TODAY_S + 9.4
        if tw0 <= s <= tw0 + 6.5:
            ax, ay = Sd.META["EAF"]["anchor"]
            yy, xx = np.mgrid[0:HC, 0:WC]
            dd = np.hypot((xx + 0.5) * CELL - ax, (yy + 0.5) * CELL - ay)
            wv = np.exp(-((dd - (s - tw0) * 330.0) / 60.0) ** 2) * 0.85 * smooth((tw0 + 6.5 - s) / 1.5)
            cell_rgb = cell_rgb * (1 - wv[..., None]) + np.array([1.0, 0.96, 0.84]) * wv[..., None] * 1.25
        # 8-bit from here: gather each grain's colour from its cell, light it with the grain texture
        rgb8 = np.clip(cell_rgb.reshape(-1, 3) * 200.0, 0, 255).astype(np.uint8)    # 200 = 1.0 (headroom)
        flag = np.minimum(hgt.ravel(), 2).astype(np.uint8)
        px = np.take(rgb8, self.owner, axis=0)
        fl = np.take(flag, self.owner)
        alpha = np.where(fl == 2, self.covpile8, self.cov8)
        alpha[fl == 0] = 0
        px = cv2.multiply(px, self.tex8, scale=1 / 128.0)
        px = cv2.multiply(px, cv2.merge([alpha] * 3), scale=1 / 255.0)
        rgba = cv2.merge([px[..., 0], px[..., 1], px[..., 2], alpha])
        w = cv2.warpPerspective(rgba, HOM, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
        w[..., :3] *= 1 / 200.0
        w[..., 3] *= 1 / 255.0
        return w

    # ---- flying sand
    def particles(self, sim, s, img, shadow):
        glow = np.zeros((H // 2, W // 2, 3), np.float32)
        for p in sim.particles:
            u = (s - p["t0"]) / p["dur"]
            act = (u > 0) & (u < 1)
            if not act.any():
                continue
            idx = np.where(act)[0]
            for back, alpha_k, width_k in ((0.0, 1.0, 1.0), (0.05, 0.5, 0.6), (0.1, 0.25, 0.45)):
                uu = np.clip(u[idx] - back, 0, 1)
                x, y, z, ok, pc = self.flight_pos(p, idx, uu)
                xp, yp, zp, okp, pcp = self.flight_pos(p, idx, np.clip(uu - 0.6 / FPS / p["dur"][idx], 0, 1))
                okp = okp & (pcp == pc)            # never streak across the map edge
                sx, sy = project(x, y, z)
                px_, py_ = project(xp, yp, zp)
                if back == 0.0:
                    shx, shy = project(x, y, np.zeros_like(x))
                    for i in range(len(idx)):
                        if ok[i]:
                            a = 0.32 * math.exp(-z[i] / 70.0)
                            cv2.circle(shadow, (int(shx[i] * 4), int(shy[i] * 4)), int((2.0 + z[i] * 0.02) * 4),
                                       a, -1, cv2.LINE_AA, shift=2)
                col = COL[p["col"][idx]]
                fade = np.clip(np.minimum(uu / 0.15, (1 - uu) / 0.05 + 0.3), 0, 1)
                for i in range(len(idx)):
                    if not (ok[i] and okp[i]):
                        continue
                    c = col[i] * (1.45 * alpha_k * fade[i])
                    w = max(1, int(round(p["size"][idx[i]] * width_k * 1.3)))
                    pt0 = (int(px_[i] * 8), int(py_[i] * 8)); pt1 = (int(sx[i] * 8), int(sy[i] * 8))
                    cv2.line(img, pt0, pt1, tuple(float(v) for v in c), w, cv2.LINE_AA, shift=3)
                    if back == 0.0:
                        cv2.circle(glow, (int(sx[i] * 4), int(sy[i] * 4)), 7, tuple(float(v) * 0.9 * fade[i] ** 2 for v in col[i]),
                                   -1, cv2.LINE_AA, shift=3)
        for (xy, s0, k) in sim.splash:          # a puff of dust where the founders land
            t = s - s0
            if 0 <= t < 1.5:
                sx, sy = project(np.array([xy[0]]), np.array([xy[1]]), np.array([0.0]))
                r = 4 + 28 * t
                a = 0.35 * (1 - t / 1.5)
                cv2.circle(glow, (int(sx[0] * 4), int(sy[0] * 4)), int(r * 2), (a, a * 0.95, a * 0.85), 2,
                           cv2.LINE_AA, shift=2)
        return glow

    @staticmethod
    def flight_pos(p, idx, u):
        path = p["path"]
        ue = u * u * (3 - 2 * u) * 0.7 + u * 0.3
        pts, piece = path.at(ue)
        nxt, piece2 = path.at(np.minimum(ue + 0.01, 1))
        d = nxt - pts
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
        q0 = np.clip(u / 0.22, 0, 1); w0 = 1 - q0 * q0 * (3 - 2 * q0)
        q1 = np.clip((u - 0.75) / 0.25, 0, 1); w1 = q1 * q1 * (3 - 2 * q1)
        p0, p1 = path.at(np.array([0.0, 1.0]))[0]
        xy = pts + (p["start"][idx] - p0) * w0[:, None] + (p["end"][idx] - p1) * w1[:, None]
        L = path.total
        env = np.sin(np.pi * u) ** 0.8
        ph = p["ph"][idx]
        lat = (p["lat"][idx] * (5 + 0.03 * L) + 6 * p["amp"][idx] * np.sin(u * 7.0 + ph[:, 0])
               + 3 * np.sin(u * 17.0 + ph[:, 1])) * env
        xy = xy + nrm * lat[:, None]
        apex = np.clip(0.13 * L, 22, 115)
        z = apex * np.sin(np.pi * u) * (0.85 + 0.3 * p["amp"][idx]) + 4 * env * np.sin(u * 11 + ph[:, 2]) + 1.5
        ok = (piece == piece2) & (xy[:, 0] > -20) & (xy[:, 0] < W + 20)
        return xy[:, 0], xy[:, 1], np.maximum(z, 0), ok, piece

    # ---- frame
    def frame(self, sim, s):
        sand = self.sand_layer(sim, s)
        a = np.clip(sand[..., 3:4], 0, 1)
        srgb = sand[..., :3] * self.illum
        shadow = np.zeros((H, W), np.float32)
        top = np.zeros((H, W, 3), np.float32)
        glow = self.particles(sim, s, top, shadow)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 2.0)[..., None]
        base = self.bare * (1 - a) + srgb
        base *= (1 - shadow)
        img = base * self.ratio + self.spec
        img = img + top                                            # flying grains, above the glass
        g = cv2.GaussianBlur(glow, (0, 0), 2.0) * 0.6 + cv2.GaussianBlur(glow, (0, 0), 6.0) * 0.5
        img += cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bloom = cv2.GaussianBlur(np.maximum(small - 0.55, 0), (0, 0), 4.0)
        img += cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR) * 0.5
        img *= self.vign
        img = img * 255.0
        self.hud(img, sim, s)
        return np.clip(img, 0, 255).astype(np.uint8)

    def hud(self, img, sim, s):
        title_a = 1.0 - smooth((s - 4.2) / 1.6)
        for v in Sd.ORDER:
            vi = Sd.VI[v]
            has = sim.counts[vi].sum() > 0
            spr = self.text_sprite(S.VESSELS[v]["label"], self.f_label, INK * 255, halo=True)
            x, y = self.label_pos[v]
            self.blit(img, spr, x - spr.shape[1] / 2, y - spr.shape[0] / 2, (0.8 if has else 0.4) * (1 - title_a * 0.7))
        ybp = ybp_at(s)
        if ybp >= 500:
            v = int(round(ybp / 100.0) * 100) if ybp >= 10_000 else (int(round(ybp / 10.0) * 10) if ybp >= 1_000 else int(round(ybp)))
            big, small = f"{v:,}", "years ago"
        else:
            yr = int(round(2025 - ybp))
            big, small = (f"{yr}", "CE") if yr < 2024 else ("Today", "")
        spr = self.text_sprite(big, self.f_year, INK * 255)
        self.blit(img, spr, 44, 26, 0.95)
        if small:
            self.blit(img, self.text_sprite(small, self.f_year_sub, MUTED * 255), 48, 26 + spr.shape[0] + 2, 0.9)
        lx, ly = 40, 700
        if sim.first_seen:
            self.blit(img, self.text_sprite("ANCESTRY  (COLOUR OF SAND)", self.f_leg_h, MUTED * 255), lx, ly - 24, 0.8)
        row = 0
        for c in range(K):
            if c not in sim.first_seen:
                continue
            a = smooth((s - sim.first_seen[c]) / 0.8)
            yy = ly + row * 19
            reg = img[yy + 4:yy + 17, lx + 1:lx + 14]
            chip = np.zeros(reg.shape[:2], np.float32)
            cv2.circle(chip, (6 * 16, 6 * 16), 6 * 16, 1.0, -1, cv2.LINE_AA, shift=4)
            m = chip[..., None] * a
            reg *= (1 - m); reg += COL[c] * 255 * m
            self.blit(img, self.text_sprite(S.COMPONENTS[c][1], self.f_leg, INK * 255), lx + 20, yy, 0.85 * a)
            row += 1
        for (s0, s1, text) in CAP_SCHEDULE:
            if s0 - 0.5 <= s <= s1 + 0.5:
                a = smooth((s - s0 + 0.5) / 0.5) * smooth((s1 + 0.5 - s) / 0.5)
                spr = self.text_sprite(text, self.f_cap, INK * 255, max_w=1300, align="center")
                x = W / 2 - spr.shape[1] / 2 + 90
                y = 990 - spr.shape[0] / 2
                self.shade(img, x - 30, y - 14, spr.shape[1] + 60, spr.shape[0] + 24, 0.55 * a)
                self.blit(img, spr, x, y, a)
        if title_a > 0.003:
            self.shade(img, 0, 0, W, H, 0.55 * title_a)
            t = self.text_sprite("Grains of Ancestry", self.f_title, INK * 255)
            self.blit(img, t, W / 2 - t.shape[1] / 2, 420, title_a)
            st = self.text_sprite("200,000 years of human migration and mixing, as sand", self.f_sub, MUTED * 255)
            self.blit(img, st, W / 2 - st.shape[1] / 2, 530, title_a)
        if s > S.TODAY_S + 9.0:
            a = smooth((s - S.TODAY_S - 9.0) / 1.2)
            n = self.text_sprite("Proportions are approximate, simplified from published ancient-DNA studies. "
                                 "Vessels group neighbouring countries.", self.f_note, MUTED * 255)
            self.blit(img, n, W - n.shape[1] - 40, 40, 0.8 * a)
        t_i0, t_i1 = S.TODAY_S + 1.0, S.TODAY_S + 8.2
        if t_i0 <= s <= t_i1:
            a = smooth((s - t_i0) / 0.8) * smooth((t_i1 - s) / 0.8)
            for v in ("AND", "PNG", "SAR", "ADS"):
                bx0, by0, bx1, by1 = Sd.META[v]["bbox"]
                rr = 1.0 + 0.04 * math.sin((s - t_i0) * 3)
                t = np.linspace(0, 2 * np.pi, 90)
                ex = (bx0 + bx1) / 2 + ((bx1 - bx0) / 2 + 26) * rr * np.cos(t)
                ey = (by0 + by1) / 2 + ((by1 - by0) / 2 + 26) * rr * np.sin(t)
                sx, sy = project(ex, ey, np.zeros_like(ex))
                pts = np.round(np.stack([sx, sy], 1) * 16).astype(np.int32)
                col = tuple(float(v) for v in np.array([255, 235, 190]) * 0.8 * a)
                ov = np.zeros_like(img)
                cv2.polylines(ov, [pts], True, col, 2, cv2.LINE_AA, shift=4)
                img += ov


# ============================================================== captions
def caption_schedule():
    items = sorted((c[1], c[2]) if c[0] == "s" else (Sd.s_at(c[0]), c[1]) for c in S.CAPTIONS)
    out = []
    for i, (s0, text) in enumerate(items):
        s1 = s0 + max(3.6, len(text.split()) / 3.1)
        s1 = min(s1, items[i + 1][0] - 1.0) if i + 1 < len(items) else S.DURATION - 0.6
        out.append((s0, s1, text))
    return out


CAP_SCHEDULE = caption_schedule()


# ============================================================== drivers
def cmd_still(args):
    os.makedirs(OUT, exist_ok=True)
    sim, painter = Sd.Sand(), Painter()
    want = sorted(float(v) for v in args.times)
    k = 0
    for f in range(int(round(S.DURATION * FPS)) + 1):
        s = f / FPS
        sim.step(s)
        if want and s > want[k] - 2.5:
            painter.update_disp(sim)
        while k < len(want) and want[k] <= s + 1e-9:
            t0 = time.time()
            im = painter.frame(sim, s)
            p = os.path.join(OUT, f"glass_{want[k]:07.2f}.png")
            cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            print("wrote", p, f"({time.time() - t0:.2f}s to paint)", flush=True)
            k += 1
        if k >= len(want):
            return


def render_chunk(job):
    f0, f1, path = job
    sim, painter = Sd.Sand(), Painter()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "16", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
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
    out = os.path.join(OUT, "grains_of_ancestry_glass.mp4")
    n = int(round(S.DURATION * FPS))
    nw = args.workers
    # every worker re-simulates from the start, so later chunks paint fewer frames
    w = np.array([1.0 / (1 + 0.25 * k) for k in range(nw)])
    edges = np.concatenate([[0], np.round(np.cumsum(w / w.sum()) * n)]).astype(int)
    jobs = [(int(edges[k]), int(edges[k + 1]), os.path.join(OUT, f"gchunk_{k}.mp4")) for k in range(nw)]
    t0 = time.time()
    with Pool(nw) as pool:
        parts = pool.map(render_chunk, jobs)
    lst = os.path.join(OUT, "gchunks.txt")
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
    args = ap.parse_args()
    {"still": cmd_still, "video": cmd_video}[args.cmd](args)
