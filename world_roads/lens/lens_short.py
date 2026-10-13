"""'Your glasses work by slowing light down': a vertical Short (1080x1920, 30 fps, ~58 s).

    python lens_short.py            -> out/lens_short.mp4
    python lens_short.py still 20   -> PNG of the frame at 20 s

Scenes: hook · speed race through lens materials · car-and-puddle refraction (differential-wheel
simulation) and the same for a light beam · ray-traced plus and minus lenses · a career of lenses
stacked next to the Calgary Tower · end card. All speeds are c/n; the stack assumes 100,000 lenses
at ~2 mm average centre thickness and n ~ 1.6.
"""
import os, sys, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
W, H, FPS = 1080, 1920, 30
C = 299_792.458
FD = "/usr/share/fonts/opentype/inter"

BG_TOP = np.array([6, 10, 20], np.float32) / 255
BG_BOT = np.array([3, 5, 10], np.float32) / 255
GOLD = np.array([1.0, 0.70, 0.28], np.float32)
WHITE = np.array([1.0, 0.97, 0.90], np.float32)
CYAN = np.array([0.20, 0.70, 1.00], np.float32)
GLASS = np.array([0.10, 0.38, 0.62], np.float32)
TXT = (238, 243, 250)
SUB = (150, 176, 205)
ACC = (255, 186, 80)

MATERIALS = [("Air", 1.000293), ("CR-39 plastic", 1.498), ("Trivex", 1.532), ("Polycarbonate", 1.586),
             ("High-index 1.67", 1.67), ("High-index 1.74", 1.74)]

SCENES = [("hook", 3.5), ("race", 11.0), ("car", 14.0), ("lens", 13.0), ("stack", 13.5), ("end", 3.5)]

_fonts = {}


def font(name, size):
    k = (name, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(os.path.join(FD, name), size)
    return _fonts[k]


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, ramp=0.35):
    """1 inside [t0, t1], ramping in/out."""
    return float(np.clip(min((t - t0) / ramp, (t1 - t) / ramp), 0, 1))


class Frame:
    def __init__(self):
        g = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        self.img = (BG_TOP * (1 - g) + BG_BOT * g).repeat(W, 1)
        self.E = np.zeros((H, W, 3), np.float32)
        self.texts = []

    # --- AA drawing via 8-bit masks
    def mask(self):
        return np.zeros((H, W), np.uint8)

    def add(self, m, color, gain=1.0, emissive=True):
        tgt = self.E if emissive else self.img
        ys, xs = np.nonzero(m)
        if len(ys):
            tgt[ys, xs] += (m[ys, xs, None].astype(np.float32) / 255) * color * gain

    def blend(self, m, color, alpha):
        ys, xs = np.nonzero(m)
        if len(ys):
            a = m[ys, xs, None].astype(np.float32) / 255 * alpha
            self.img[ys, xs] = self.img[ys, xs] * (1 - a) + color * a

    def line(self, pts, color, gain=1.0, thick=2, emissive=True, closed=False):
        m = self.mask()
        cv2.polylines(m, [np.round(np.asarray(pts) * 16).astype(np.int32)], closed, 255, thick, cv2.LINE_AA, shift=4)
        self.add(m, color, gain, emissive)

    def dashed(self, p0, p1, color, gain=0.5, dash=14, gap=10, thick=2):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        if L < 1:
            return
        d = (p1 - p0) / L
        m = self.mask()
        s = 0.0
        while s < L:
            a, b = p0 + d * s, p0 + d * min(s + dash, L)
            cv2.line(m, tuple(np.round(a * 16).astype(int)), tuple(np.round(b * 16).astype(int)), 255, thick, cv2.LINE_AA, shift=4)
            s += dash + gap
        self.add(m, color, gain)

    def poly_fill(self, pts, color, alpha, edge=None, edge_gain=0.8):
        m = self.mask()
        cv2.fillPoly(m, [np.round(np.asarray(pts) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        self.blend(m, color, alpha)
        if edge is not None:
            self.line(pts, edge, edge_gain, 2, closed=True)

    def photon(self, x, y, gain=1.0, size=1.0):
        n = 81; h = n // 2
        k = np.arange(n, dtype=np.float32) - h
        r2 = k[None] ** 2 + k[:, None] ** 2
        spr = (np.exp(-r2 / (2 * (1.6 * size) ** 2))[..., None] * WHITE * 2.4 +
               np.exp(-r2 / (2 * (5 * size) ** 2))[..., None] * GOLD * 1.0 +
               np.exp(-r2 / (2 * (16 * size) ** 2))[..., None] * GOLD * 0.30) * gain
        xi, yi = int(round(x)), int(round(y))
        a0, a1, b0, b1 = max(yi - h, 0), min(yi + h + 1, H), max(xi - h, 0), min(xi + h + 1, W)
        if a1 > a0 and b1 > b0:
            self.E[a0:a1, b0:b1] += spr[a0 - (yi - h):a1 - (yi - h), b0 - (xi - h):b1 - (xi - h)]

    def trail(self, p0, p1, gain=1.0, thick=3):
        """gradient streak from p0 (dim) to p1 (bright)"""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        for i in range(8):
            a = p0 + (p1 - p0) * i / 8; b = p0 + (p1 - p0) * (i + 1) / 8
            self.line([a, b], GOLD, gain * (0.15 + 0.85 * (i + 1) / 8), thick)

    def text(self, s, xy, size, color=TXT, alpha=1.0, bold=False, anchor="ma"):
        if alpha > 0.01:
            self.texts.append((s, xy, size, color, alpha, bold, anchor))

    def render(self):
        E = self.E
        small = cv2.resize(E, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3) * 0.6 + cv2.GaussianBlur(small, (0, 0), 12) * 0.5, (W, H))
        img = self.img + E + bloom
        img = 1 - np.exp(-img * 1.1)
        im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for s, xy, size, color, alpha, bold, anchor in self.texts:
            f = font("Inter-Bold.otf" if bold else "Inter-Medium.otf", size)
            d.text(xy, s, font=f, fill=tuple(color) + (int(255 * alpha),), anchor=anchor)
        im.paste(ov, (0, 0), ov)
        return np.asarray(im)


# ------------------------------------------------------------------------------------- scenes
def scene_hook(F, t):
    x = -100 + 1300 * ease(t / 0.9)
    if t < 1.2:
        F.trail((x - 420, 1000), (x, 1000), 1.0, 4)
        F.photon(x, 1000, 1.2)
    a = fade(t, 0.9, 99)
    F.text("Your glasses work", (W // 2, 760), 82, TXT, a, True)
    F.text("by slowing", (W // 2, 865), 82, TXT, a, True)
    F.text("light down.", (W // 2, 970), 82, ACC, fade(t, 1.4, 99), True)
    F.text("Here's how.", (W // 2, 1110), 44, SUB, fade(t, 2.2, 99))


def scene_race(F, t):
    F.text("Light slows down in lenses", (W // 2, 170), 64, TXT, 1, True)
    F.text("Same light, different materials", (W // 2, 258), 38, SUB, fade(t, 0.3, 99))
    x0, x1 = 90, 990
    for i, (name, n) in enumerate(MATERIALS):
        y = 520 + i * 150
        a = fade(t, 0.1 + i * 0.08, 99)
        F.text(name, (x0, y - 66), 34, TXT, a, False, "la")
        F.text(f"index {n:.2f}", (x0, y + 28), 24, SUB, a, False, "la")
        F.text(f"{C / n:,.0f} km/s", (x1, y - 66), 34, ACC if i else TXT, a, True, "ra")
        bar = [(x0, y - 17), (x1, y - 17), (x1, y + 17), (x0, y + 17)]
        if i:
            F.poly_fill(bar, GLASS, 0.45 * a, CYAN, 0.35 * a)
        else:
            F.line(bar, CYAN, 0.12 * a, 1, closed=True)
        T = 3.2 * n
        u = np.clip((t - 0.8) / T, 0, 1)
        px = x0 + (x1 - x0) * u
        if t > 0.8:
            F.trail((max(x0, px - 160), y), (px, y), 0.9, 3)
            F.photon(px, y, 0.9)
        if u >= 1:
            F.text(f"{n:.2f}× slower" if i else "fastest", (x1, y + 28), 24, SUB, fade(t, 0.8 + T, 99), False, "ra")
    F.text("Higher index = slower light", (W // 2, 1470), 46, TXT, fade(t, 7.0, 99), True)
    F.text("= thinner lenses for the same prescription", (W // 2, 1540), 34, SUB, fade(t, 7.6, 99))


# car-and-puddle: differential-wheel simulation, precomputed
CAR = {}


def car_sim():
    if CAR:
        return CAR
    ang = np.radians(-28)
    bdir = np.array([np.cos(ang), np.sin(ang)])            # boundary direction (rises to the right)
    b0 = np.array([540.0, 1000.0])
    nrm = np.array([bdir[1], -bdir[0]])                     # points into the puddle (up)
    if nrm[1] > 0:
        nrm = -nrm
    inside = lambda p: np.dot(p - b0, nrm) > 0
    vf, vs, track = 165.0, 82.0, 84.0
    x, y, th = 470.0, 1640.0, 0.0
    dt = 1 / 240
    out = []
    for k in range(int(9.6 / dt) + 1):
        h = np.array([np.sin(th), -np.cos(th)]); l = np.array([-np.cos(th), -np.sin(th)])
        c = np.array([x, y]) + h * 40                         # front axle
        vl = vs if inside(c + l * track / 2) else vf
        vr = vs if inside(c - l * track / 2) else vf
        out.append((x, y, th, vl < vf, vr < vf))
        om = (vl - vr) / track
        th += om * dt
        v = (vl + vr) / 2
        x += np.sin(th) * v * dt; y += -np.cos(th) * v * dt
    CAR.update(traj=np.array(out), b0=b0, bdir=bdir, nrm=nrm, dt=dt)
    return CAR


def draw_boundary_regions(F, b0, bdir, top_y, bot_y, color, alpha, edge):
    # polygon above the boundary line between top_y and the line, full width
    def y_on(x):
        return b0[1] + (x - b0[0]) * bdir[1] / bdir[0]
    pts = [(0, top_y), (W, top_y), (W, y_on(W)), (0, y_on(0))]
    F.poly_fill(pts, color, alpha)
    F.line([(0, y_on(0)), (W, y_on(W))], edge, 0.7, 2)


def scene_car(F, t):
    S = car_sim()
    b0, bdir, nrm = S["b0"], S["bdir"], S["nrm"]
    tr = S["traj"]
    car_phase = t < 9.8
    if car_phase:
        a_sc = fade(t, -1, 9.8, 0.4)
        F.img[420:1760] = F.img[420:1760] * (1 - 0.6 * a_sc) + np.array([0.075, 0.082, 0.095], np.float32) * 0.6 * a_sc
        draw_boundary_regions(F, b0, bdir, 420, 1760, np.array([0.06, 0.20, 0.36], np.float32), 0.85 * a_sc, CYAN)
        F.text("PAVEMENT", (90, 1700), 26, SUB, 0.8 * a_sc, False, "la")
        F.text("PUDDLE", (W - 90, 470), 26, (120, 190, 240), 0.9 * a_sc, False, "ra")
        i = min(int(t / S["dt"]), len(tr) - 1)
        x, y, th, slow_l, slow_r = tr[i]
        # ghost: where the car would have gone
        cross = np.nonzero((tr[:, 3] > 0) | (tr[:, 4] > 0))[0]
        if len(cross) and i > cross[0]:
            cx, cy = tr[cross[0], 0], tr[cross[0], 1]
            F.dashed((cx, cy), (cx, 430), WHITE, 0.35)
        F.line(tr[: i + 1: 8, :2], GOLD, 0.55, 3)
        # car body
        h = np.array([np.sin(th), -np.cos(th)]); l = np.array([-np.cos(th), -np.sin(th)])
        c = np.array([x, y])
        body = [c + h * 70 + l * 42, c + h * 70 - l * 42, c - h * 70 - l * 42, c - h * 70 + l * 42]
        F.poly_fill(body, np.array([0.85, 0.88, 0.92], np.float32), 0.95)
        F.poly_fill([c + h * 30 + l * 32, c + h * 30 - l * 32, c - h * 10 - l * 32, c - h * 10 + l * 32],
                    np.array([0.25, 0.32, 0.42], np.float32), 0.95)
        for side, slow in ((1, slow_l), (-1, slow_r)):
            for ax in (40, -45):
                wc = c + h * ax + l * side * 46
                wheel = [wc + h * 18 + l * 8, wc + h * 18 - l * 8, wc - h * 18 - l * 8, wc - h * 18 + l * 8]
                F.poly_fill(wheel, np.array([0.05, 0.05, 0.06], np.float32), 1.0)
                if slow and ax == 40:
                    m = F.mask()
                    rr = 16 + 10 * ((t * 3) % 1)
                    cv2.circle(m, tuple(np.round(wc * 16).astype(int)), int(rr * 16), 255, 2, cv2.LINE_AA, shift=4)
                    F.add(m, CYAN, 0.8 * (1 - (t * 3) % 1))
        F.text("A car drives into a puddle at an angle", (W // 2, 170), 46, TXT, fade(t, 0.1, 3.3), True)
        F.text("The left wheels hit the water first", (W // 2, 170), 46, TXT, fade(t, 3.3, 6.1), True)
        F.text("and slow down.", (W // 2, 240), 46, ACC, fade(t, 3.6, 6.1), True)
        F.text("The right wheels are still fast,", (W // 2, 170), 46, TXT, fade(t, 6.1, 9.8), True)
        F.text("so the car turns.", (W // 2, 240), 46, ACC, fade(t, 6.5, 9.8), True)
    else:
        u = t - 9.8
        a = fade(u, 0, 99, 0.4)
        n = 1.6
        draw_boundary_regions(F, b0, bdir, 420, 1760, GLASS, 0.55 * a, CYAN)
        F.text("GLASS", (W - 90, 470), 26, (120, 190, 240), 0.9 * a, False, "ra")
        F.text("AIR", (90, 1700), 26, SUB, 0.8 * a, False, "la")
        # beam from below, straight up, hits boundary at P
        sx = 470.0
        P = np.array([sx, b0[1] + (sx - b0[0]) * bdir[1] / bdir[0]])
        d1 = np.array([0.0, -1.0])
        nn = nrm.copy()                                      # normal into the glass
        cosi = np.dot(d1, nn)
        eta = 1 / n
        k = 1 - eta ** 2 * (1 - cosi ** 2)
        d2 = eta * d1 + (np.sqrt(k) - eta * cosi) * nn
        d2 /= np.linalg.norm(d2)
        F.dashed(P - nn * 260, P + nn * 380, WHITE, 0.3)
        F.text("normal", tuple((P + nn * 400 + np.array([0, -10])).astype(int)), 24, SUB, 0.7 * a)
        v_air, L1 = 420.0, P[1] - 0.0
        S0 = np.array([sx, 1760.0])
        t_hit = (S0[1] - P[1]) / v_air
        tt = max(u - 0.3, 0)
        if tt <= t_hit:
            pos = S0 + d1 * v_air * tt
            F.trail(S0, pos, 1.0, 4); F.photon(*pos, 1.0)
        else:
            pos = P + d2 * (v_air / n) * (tt - t_hit)
            F.trail(S0, P, 1.0, 4); F.trail(P, pos, 1.0, 4); F.photon(*pos, 1.0)
            F.dashed(P, P + d1 * 420, WHITE, 0.25)
        F.text("Light does exactly the same thing.", (W // 2, 170), 46, TXT, fade(u, 0.2, 99), True)
        F.text("It slows down in glass, so it bends.", (W // 2, 240), 40, SUB, fade(u, 1.2, 99))
        F.text("That's refraction.", (W // 2, 310), 46, ACC, fade(u, 2.2, 99), True)


def trace_lens(kind, cx=540.0, y0=780.0, a=330.0, n=1.6, span=0.45):
    """Ray-trace vertical rays through a spherical lens. Returns rays [(points, segment speeds)] and lens outline."""
    if kind == "plus":
        R = 660.0                                   # f ~ R / (2(n-1)) ~ 550 px; no ray exceeds the critical angle
        t_c = 16.0 + 2 * (R - np.sqrt(R * R - a * a))
        top_c = np.array([cx, y0 - t_c / 2 + R]); bot_c = np.array([cx, y0 + t_c / 2 - R])
        top_y = lambda x: top_c[1] - np.sqrt(R * R - (x - cx) ** 2)
        bot_y = lambda x: bot_c[1] + np.sqrt(R * R - (x - cx) ** 2)
    else:
        R, t_c = 600.0, 26.0
        top_c = np.array([cx, y0 - t_c / 2 - R]); bot_c = np.array([cx, y0 + t_c / 2 + R])
        top_y = lambda x: top_c[1] + np.sqrt(R * R - (x - cx) ** 2)
        bot_y = lambda x: bot_c[1] - np.sqrt(R * R - (x - cx) ** 2)
    xs = np.linspace(cx - a, cx + a, 120)
    outline = [(x, top_y(x)) for x in xs] + [(x, bot_y(x)) for x in xs[::-1]]

    def hit(p, d, ctr):
        oc = p - ctr
        b = np.dot(oc, d); cc = np.dot(oc, oc) - R * R
        disc = b * b - cc
        r = np.sqrt(max(disc, 0))
        ss = [s for s in (-b - r, -b + r) if s > 1e-6]
        return p + d * min(ss)

    def refract(d, nrm, n1, n2):
        nrm = nrm / np.linalg.norm(nrm)
        if np.dot(nrm, d) > 0:
            nrm = -nrm
        cosi = -np.dot(nrm, d)
        eta = n1 / n2
        k = 1 - eta * eta * (1 - cosi * cosi)
        out = eta * d + (eta * cosi - np.sqrt(k)) * nrm
        return out / np.linalg.norm(out)

    rays = []
    for x in np.linspace(cx - a * span, cx + a * span, 9):
        p0 = np.array([x, 380.0]); d = np.array([0.0, 1.0])
        p1 = hit(p0, d, top_c); d1 = refract(d, p1 - top_c, 1.0, n)
        p2 = hit(p1, d1, bot_c); d2 = refract(d1, p2 - bot_c, n, 1.0)
        p3 = p2 + d2 * ((1500 - p2[1]) / max(d2[1], 1e-3))
        rays.append((np.array([p0, p1, p2, p3]), np.array([1.0, 1.0 / n, 1.0]), d2))
    return rays, outline


def ray_pos(pts, speeds, dist):
    """position after travelling 'dist' units of time*speed-1 along a polyline with per-segment speeds."""
    tleft = dist
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        L = np.linalg.norm(seg)
        need = L / speeds[i]
        if tleft <= need:
            return pts[i] + seg * (tleft * speeds[i] / L), i
        tleft -= need
    return pts[-1], len(pts) - 2


def scene_lens(F, t):
    kind = "plus" if t < 6.8 else "minus"
    u = t if kind == "plus" else t - 6.8
    rays, outline = trace_lens(kind)
    a = fade(u, 0, 99, 0.3)
    F.poly_fill(outline, GLASS, 0.55 * a, CYAN, 0.7 * a)
    v = 330.0
    tt = max(u - 0.4, 0) * v
    front = []
    for pts, sp, d2 in rays:
        pos, seg = ray_pos(pts, sp, tt)
        path = list(pts[: seg + 1]) + [pos]
        F.line(path, GOLD, 0.35, 2)
        F.photon(*pos, 0.75, 0.8)
        front.append(pos)
        if kind == "minus" and seg >= 2:
            p2 = pts[2]
            F.dashed(p2, p2 - d2 * 380, CYAN, 0.25, 8, 10, 1)
    front = np.array(front)
    F.line(front, WHITE, 0.45, 2)
    if kind == "plus":
        F.text("Now curve the edge of the puddle.", (W // 2, 170), 46, TXT, fade(u, 0.1, 99), True)
        F.text("The middle of the light slows longest,", (W // 2, 240), 38, SUB, fade(u, 1.4, 99))
        F.text("so every ray turns toward one point.", (W // 2, 296), 38, SUB, fade(u, 2.0, 99))
        F.text("That's focusing.", (W // 2, 1560), 52, ACC, fade(u, 4.3, 99), True)
    else:
        F.text("Nearsighted lenses do the opposite:", (W // 2, 170), 46, TXT, fade(u, 0.1, 99), True)
        F.text("thicker edges spread the light out", (W // 2, 240), 38, SUB, fade(u, 0.8, 99))
        F.text("so it focuses further back, on your retina.", (W // 2, 296), 38, SUB, fade(u, 1.6, 99))


def tower(F, cx, base, ppm, alpha):
    """Calgary Tower silhouette, 191 m."""
    col = np.array([0.30, 0.38, 0.50], np.float32)
    y = lambda m: base - m * ppm
    shaft = [(cx - 34, y(0)), (cx + 34, y(0)), (cx + 20, y(150)), (cx - 20, y(150))]
    F.poly_fill(shaft, col, 0.9 * alpha)
    pod = [(cx - 62, y(150)), (cx + 62, y(150)), (cx + 72, y(160)), (cx + 66, y(172)), (cx - 66, y(172)), (cx - 72, y(160))]
    F.poly_fill(pod, col * 1.15, 0.95 * alpha)
    F.poly_fill([(cx - 40, y(172)), (cx + 40, y(172)), (cx + 28, y(177)), (cx - 28, y(177))], col, 0.9 * alpha)
    F.poly_fill([(cx - 4, y(177)), (cx + 4, y(177)), (cx + 2, y(191)), (cx - 2, y(191))], col, 0.9 * alpha)


def scene_stack(F, t):
    base, ppm = 1600.0, 4.8
    sx, ax, tx = 300.0, 520.0, 800.0
    grow = ease(t / 4.0)
    n_l = int(100_000 * grow)
    hgt = 200.0 * grow
    tower(F, tx, base, ppm, fade(t, 0, 99, 0.4))
    F.text("Calgary Tower", (tx, base + 24), 30, SUB, fade(t, 0.2, 99))
    F.text("191 m", (tx, base + 62), 30, SUB, fade(t, 0.2, 99))
    top = base - hgt * ppm
    if hgt > 0.5:
        F.poly_fill([(sx - 70, base), (sx + 70, base), (sx + 70, top), (sx - 70, top)], GLASS, 0.6, CYAN, 0.6)
        m = F.mask()
        for yy in np.arange(base, top, -9):
            cv2.line(m, (int((sx - 66) * 16), int(yy * 16)), (int((sx + 66) * 16), int(yy * 16)), 255, 1, cv2.LINE_AA, shift=4)
        F.add(m, CYAN, 0.12)
    F.text(f"{n_l:,} lenses", (sx, base + 24), 30, ACC, 1, True)
    F.text(f"{hgt:,.0f} m", (sx, base + 62), 30, SUB, 1)
    r_done = 5.0 + 4.6
    F.text("Every lens I've prescribed,", (W // 2, 160), 50, TXT, fade(t, 0.1, r_done), True)
    F.text("stacked up", (W // 2, 230), 50, TXT, fade(t, 0.1, r_done), True)
    F.text("~100,000 lenses × ~2 mm each", (W // 2, 300), 34, SUB, fade(t, 0.5, r_done))
    # race from the bottom: 1 microsecond of light = 4 s of video
    r0 = 5.0
    if t > r0 - 0.6:
        F.text("Now send light up both:", (W // 2, 380), 40, ACC, fade(t, r0 - 0.6, r0 + 4.6), True)
    if t > r0:
        tau = (t - r0) / 4.0 * 1000            # nanoseconds
        Lm = 200.0
        air_m = min(tau * 1e-9 * C * 1000, Lm)
        st_m = min(tau * 1e-9 * C * 1000 / 1.6, Lm)
        for x, mm, label, total in ((ax, air_m, "air", Lm / (C * 1000) * 1e9), (sx, st_m, "lenses", Lm * 1.6 / (C * 1000) * 1e9)):
            y = base - mm * ppm
            F.trail((x, base), (x, y), 1.0, 4)
            F.photon(x, y, 1.0)
            ns = min(tau, total)
            F.text(f"{ns:,.0f} ns", (x, y - 70) if mm < Lm else (x, top - 70), 32, TXT, 1, True)
        late = tau > Lm * 1.6 / (C * 1000) * 1e9
        F.text("A whole career of lenses:", (W // 2, 160), 50, TXT, fade(t, r0 + 4.8, 99), True)
        F.text("light arrives 0.4 millionths", (W // 2, 232), 50, ACC, fade(t, r0 + 5.1, 99), True)
        F.text("of a second late.", (W // 2, 302), 50, ACC, fade(t, r0 + 5.1, 99), True)
        F.text("(Light in air gets 120 m ahead.)", (W // 2, 380), 34, SUB, fade(t, r0 + 5.8, 99))


def scene_end(F, t):
    F.text("Light in air:", (W // 2, 700), 44, SUB, fade(t, 0, 99))
    F.text("299,792 km/s", (W // 2, 760), 84, TXT, fade(t, 0, 99), True)
    F.text("Light in your glasses:", (W // 2, 930), 44, SUB, fade(t, 0.5, 99))
    F.text("~190,000 km/s", (W // 2, 990), 84, ACC, fade(t, 0.5, 99), True)
    F.text("Slowing it down is how you see clearly.", (W // 2, 1180), 40, TXT, fade(t, 1.2, 99))
    F.photon(W // 2, 1300, 0.8 * fade(t, 1.2, 99))


FUNCS = {"hook": scene_hook, "race": scene_race, "car": scene_car, "lens": scene_lens, "stack": scene_stack, "end": scene_end}


def frame_at(T):
    acc = 0.0
    for name, dur in SCENES:
        if T < acc + dur or name == SCENES[-1][0]:
            t = T - acc
            F = Frame()
            FUNCS[name](F, t)
            img = F.render().astype(np.float32)
            # short dip between scenes
            edge = min(t, dur - t)
            if edge < 0.25 and not (name == "hook" and t < 0.25):
                img *= 0.35 + 0.65 * edge / 0.25
            return img.astype(np.uint8)
        acc += dur


def main():
    os.makedirs(OUT, exist_ok=True)
    total = sum(d for _, d in SCENES)
    n = int(round(total * FPS))
    path = os.path.join(OUT, "lens_short.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for f in range(n):
        p.stdin.write(frame_at(f / FPS).tobytes())
    p.stdin.close(); p.wait()
    print("wrote", path, f"{n / FPS:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        os.makedirs(OUT, exist_ok=True)
        for s in sys.argv[2:]:
            cv2.imwrite(os.path.join(OUT, f"still_{s}.png"), frame_at(float(s))[..., ::-1])
            print("still", s)
    else:
        main()
