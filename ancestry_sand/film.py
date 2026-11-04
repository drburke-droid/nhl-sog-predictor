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
import camera as C
from sand import smooth, ybp_at, CELL, WC, HC, K
from proj import W, H

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
FPS = 30
FONT_DIR = "/usr/share/fonts/opentype/inter"


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


COL = np.stack([hex_rgb(c[2]) for c in S.COMPONENTS])
INK = hex_rgb("#e8eef8")
MUTED = hex_rgb("#93a1b8")


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def load_plate(name):
    p = name if os.path.isabs(name) else os.path.join(DATA, name)
    if not os.path.exists(p):
        p = p.replace(".png", "_test.png")
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
    if im.shape[1] != W:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
    return im


def project_P(P, x, y, z):
    """Map px (x, y) at height z (map px) -> screen px, through camera matrix P."""
    q = P @ np.stack([x, y, z, np.ones_like(x)])
    return q[0] / q[2], q[1] / q[2]


WALL_MID = C.WALL_H / 2          # glass plates are reprojected through the plane halfway up the walls
GRAIN_DZ = 0.55                  # height of one grain level in a pile (map px): dense places rise as towers


class Plates:
    """The Blender glass for any camera pose: rendered at keyframes along the path, reprojected
    (through the plane halfway up the walls) and cross-faded in between."""

    def __init__(self):
        d = os.path.join(DATA, "plates")
        idx = json.load(open(os.path.join(DATA, "plate_poses.json")))
        self.keys = [(float(t), int(i)) for t, i in idx["keys"]]
        self.poses = [tuple(p) for p in idx["poses"]]
        self.dir = d
        self.cache = {}

    def load(self, i):
        if i not in self.cache:
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            j = i
            if not os.path.exists(os.path.join(self.dir, f"b_{i:03d}.png")):      # still rendering: preview only
                j = 0
                print(f"  (plate {i} not rendered yet; reprojecting the wide plate)", flush=True)
            g = load_plate(os.path.join(self.dir, f"g_{j:03d}.png"))
            bp = os.path.join(self.dir, f"b_{j:03d}.png")
            b = load_plate(bp if os.path.exists(bp) else os.path.join(DATA, "plate_bare.png"))
            ratio = np.clip((g + 0.004) / (b + 0.004), 0.42, 1.0)
            spec = np.clip(g - b, 0, None)
            P = C.P_of(self.poses[j])
            self.cache[i] = (b, ratio, spec, P)
        return self.cache[i]

    def at(self, s, P):
        ts = [k[0] for k in self.keys]
        j = int(np.searchsorted(ts, s, side="right")) - 1
        j = min(max(j, 0), len(ts) - 1)
        if j == len(ts) - 1 or self.keys[j][1] == self.keys[j + 1][1]:
            pairs = [(self.keys[j][1], 1.0)]
        else:
            w = (s - ts[j]) / (ts[j + 1] - ts[j])
            pairs = [(self.keys[j][1], 1 - w), (self.keys[j + 1][1], w)]
        acc = [np.zeros((H, W, 3), np.float32) for _ in range(3)]
        wsum = np.zeros((H, W, 1), np.float32)
        for i, w in pairs:
            if w <= 1e-4:
                continue
            b, ratio, spec, Pk = self.load(i)
            if np.allclose(Pk, P, atol=1e-9):
                acc[0] += w * b; acc[1] += w * ratio; acc[2] += w * spec; wsum += w
                continue
            Mg = C.hom(P, WALL_MID) @ np.linalg.inv(C.hom(Pk, WALL_MID))
            Mt = C.hom(P, 0.0) @ np.linalg.inv(C.hom(Pk, 0.0))
            ok = cv2.warpPerspective(np.ones((H, W), np.float32), Mg, (W, H), flags=cv2.INTER_LINEAR)[..., None]
            acc[0] += w * ok * cv2.warpPerspective(b, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            acc[1] += w * ok * cv2.warpPerspective(ratio, Mg, (W, H), flags=cv2.INTER_LINEAR, borderValue=(1, 1, 1))
            acc[2] += w * ok * cv2.warpPerspective(spec, Mg, (W, H), flags=cv2.INTER_LINEAR)
            wsum += w * ok
        wsum = np.maximum(wsum, 1e-4)
        b = acc[0] / wsum
        ratio = np.where(wsum > 2e-4, acc[1] / wsum, 1.0)
        spec = acc[2] / wsum
        return b, ratio, spec


def hash01(a, b, salt):
    """Stable pseudo-random numbers in [0, 1) from integer arrays (per grain jitter)."""
    h = (a.astype(np.uint64) * np.uint64(73856093)) ^ (b.astype(np.uint64) * np.uint64(19349663)) ^ np.uint64(salt * 83492791)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5bd1e995)
    h = h ^ (h >> np.uint64(15))
    return (h % np.uint64(1 << 20)).astype(np.float64) / float(1 << 20)


# ============================================================== painter
class Painter:
    def __init__(self):
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
        self.label_map = {}
        for v in Sd.ORDER:
            m = Sd.META[v]
            if m["inset"]:
                f = np.asarray(m["frame"])
                self.label_map[v] = ((f[:, 0].min() + f[:, 0].max()) / 2, f[:, 1].max() + 11.0)
            else:
                self.label_map[v] = (m["anchor"][0], m["anchor"][1])

    # ---- the camera for this frame
    def set_camera(self, s):
        """Camera pose, projection and plane homography for video time s."""
        self.pose = C.pose_at(s)
        self.P = C.P_of(self.pose)
        self.loc, self.cr, self.cu, self.cf = C.basis(self.pose)
        self.panel_a = 0.55 * float(np.clip((C.WIDE[3] - self.pose[3]) / 9.0, 0, 1))   # HUD backing in close-ups

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

    def panel(self, img, x0, y0, x1, y1, a):
        """A soft dark backing behind HUD text, so it reads over sand in the close-ups."""
        key = ("panel", int(x0), int(y0), int(x1), int(y1))
        if key not in self.cache:
            pad = 40
            m = np.zeros((int(y1 - y0) + 2 * pad, int(x1 - x0) + 2 * pad), np.float32)
            m[pad:-pad, pad:-pad] = 1.0
            self.cache[key] = cv2.GaussianBlur(m, (0, 0), 16)[..., None]
        m = self.cache[key]
        xa, ya = int(x0) - 40, int(y0) - 40
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
        tgt = COL[sim.stack[cell, lev].astype(np.int64)] * self.cell_tint.ravel()[cell][:, None]
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
            for j in range(1, int(Z.max() / 3.0) + 2):
                occ = Z[:-j, :-j] - 3.0 * j > 1.0
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
        x = Sd.CELL_XY[cell, 0] + (jx - 0.5) * 1.3
        y = Sd.CELL_XY[cell, 1] + (jy - 0.5) * 1.3
        r = (1.35 + 0.35 * jr) * V["scale"]
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
        tw0 = S.TODAY_S + 9.4                               # the closing ripple of light from East Africa
        if tw0 <= s <= tw0 + 6.5:
            ax, ay = Sd.META["EAF"]["anchor"]
            dd = np.hypot(x[sel] - ax, y[sel] - ay)
            wv = (np.exp(-((dd - (s - tw0) * 330.0) / 60.0) ** 2) * 0.85 * smooth((tw0 + 6.5 - s) / 1.5))[:, None]
            base = base * (1 - wv) + np.array([1.0, 0.96, 0.84]) * wv * 1.25
        SS = 2
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
                sx, sy = self.project(x, y, z)
                px_, py_ = self.project(xp, yp, zp)
                if back == 0.0:
                    shx, shy = self.project(x, y, np.zeros_like(x))
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
                sx, sy = self.project(np.array([xy[0]]), np.array([xy[1]]), np.array([0.0]))
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
        self.set_camera(s)
        bare, ratio, spec = self.plates.at(s, self.P)
        self.update_disp(sim)
        self.shadows(sim)
        shadow = np.zeros((H, W), np.float32)
        top = np.zeros((H, W, 3), np.float32)
        glow = self.particles(sim, s, top, shadow)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 2.0)[..., None]
        img = bare * self.table_shadow * (1 - shadow) * ratio + spec
        depth = self.plane_depth()
        g = self.grains(sim, s)
        if g is not None:
            grgb, gcov, glow_, gdep = g
            grgb = grgb * self.illum
            # grains below the rim stand behind the glass; above it, nothing is in front of them
            f = glow_ / np.maximum(gcov, 1e-4)
            img = img * (1 - gcov) + grgb * (ratio * f + (1 - f)) + spec * glow_
            depth = np.where(gcov[..., 0] > 0.5, gdep, depth)
        img = img + top                                            # flying grains, above the glass
        g = cv2.GaussianBlur(glow, (0, 0), 2.0) * 0.6 + cv2.GaussianBlur(glow, (0, 0), 6.0) * 0.5
        img += cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bloom = cv2.GaussianBlur(np.maximum(small - 0.55, 0), (0, 0), 4.0)
        img += cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR) * 0.5
        img *= self.vign
        img = self.dof(img, depth)
        img = img * 255.0
        self.hud(img, sim, s)
        return np.clip(img, 0, 255).astype(np.uint8)

    def hud(self, img, sim, s):
        title_a = 1.0 - smooth((s - 4.2) / 1.6)
        for v in Sd.ORDER:
            vi = Sd.VI[v]
            has = sim.counts[vi].sum() > 0
            mag = Sd.META[v]["mag"]
            lab = S.VESSELS[v]["label"] + (f"  \u00d7{mag:g}" if mag else "")
            spr = self.text_sprite(lab, self.f_label, INK * 255, halo=True)
            x, y = self.project(np.array([self.label_map[v][0]]), np.array([self.label_map[v][1]]), np.array([0.0]))
            x, y = float(x[0]), float(y[0])
            self.blit(img, spr, x - spr.shape[1] / 2, y - spr.shape[0] / 2, (0.8 if has else 0.4) * (1 - title_a * 0.7))
        ybp = ybp_at(s)
        if ybp >= 500:
            v = int(round(ybp / 100.0) * 100) if ybp >= 10_000 else (int(round(ybp / 10.0) * 10) if ybp >= 1_000 else int(round(ybp)))
            big, small = f"{v:,}", "years ago"
        else:
            yr = int(round(2025 - ybp))
            big, small = (f"{yr}", "CE") if yr < 2024 else ("Today", "")
        spr = self.text_sprite(big, self.f_year, INK * 255)
        self.panel(img, 30, 22, 330, 170, self.panel_a)
        self.blit(img, spr, 44, 26, 0.95)
        if small:
            self.blit(img, self.text_sprite(small, self.f_year_sub, MUTED * 255), 48, 26 + spr.shape[0] + 2, 0.9)
        tot = max(int(sim.counts.sum()), 1)
        ppl = Sd.world_pop(ybp) / tot
        ppl = float(f"{ppl:.2g}") if ppl < 1000 else round(ppl, -3)
        line = f"1 grain \u2248 {ppl:,.0f} people"
        self.blit(img, self.text_sprite(line, self.f_year_sub, INK * 255), 48, 26 + spr.shape[0] + 30, 0.85)
        lx, ly = 40, 700
        if sim.first_seen:
            self.blit(img, self.text_sprite("ANCESTRY  (COLOUR OF SAND)", self.f_leg_h, MUTED * 255), lx, ly - 24, 0.8)
        nrow = sum(1 for c in range(K) if c in sim.first_seen)
        if nrow:
            self.panel(img, lx - 8, ly - 30, lx + 250, ly + nrow * 19 + 4, self.panel_a)
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
            n = self.text_sprite("Ancestry from published ancient-DNA studies; population from long-run estimates. "
                                 "Magnified islands keep a minimum of sand to stay visible.", self.f_note, MUTED * 255)
            self.blit(img, n, W - n.shape[1] - 40, 40, 0.8 * a)
        t_i0, t_i1 = S.TODAY_S + 1.0, S.TODAY_S + 8.2
        if t_i0 <= s <= t_i1:
            a = smooth((s - t_i0) / 0.8) * smooth((t_i1 - s) / 0.8)
            for v in ("PNG", "ADS", "MDG"):
                bx0, by0, bx1, by1 = Sd.META[v]["bbox"]
                rr = 1.0 + 0.04 * math.sin((s - t_i0) * 3)
                t = np.linspace(0, 2 * np.pi, 90)
                ex = (bx0 + bx1) / 2 + ((bx1 - bx0) / 2 + 26) * rr * np.cos(t)
                ey = (by0 + by1) / 2 + ((by1 - by0) / 2 + 26) * rr * np.sin(t)
                sx, sy = self.project(ex, ey, np.zeros_like(ex))
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
