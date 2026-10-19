"""The look of the vernier film: a desk lit by a lamp, cream paper, a printed white card, a glass hand
magnifier, red-pencil notes and serif type. Rendered at 4K; geometry is in 1080p units (1 unit = 2 px).

The test line is drawn with exact area coverage in linear light, so a step of a quarter of a 1080p pixel
really moves the lower half's light a quarter of a pixel. Nothing textured (paper, grain) is drawn on the
card the line is printed on.
"""
import os, sys, math, time, subprocess, functools
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONTS = os.path.join(HERE, "fonts")
W, H, FPS = 3840, 2160, 30
S = 2                                                   # 4K pixels per 1080p unit

PAPER = (236, 228, 212)
CARD = (247, 246, 242)
INK = (30, 27, 25)
INK_SOFT = (84, 77, 70)
RED = (172, 50, 38)
CREAM = (244, 238, 226)

# ------------------------------------------------------------------ colour
_LUT = None


def lin_lut():
    global _LUT
    if _LUT is None:
        c = np.arange(256) / 255.0
        _LUT = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return _LUT


def to_lin(a):
    return lin_lut()[np.asarray(a, np.uint8)]


_SLUT = None


def to_srgb(x):
    """linear light (0..1) to sRGB uint8, through a 65,536-step table"""
    global _SLUT
    if _SLUT is None:
        c = np.arange(65536) / 65535.0
        _SLUT = np.round(np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0).astype(np.uint8)
    return _SLUT[(np.clip(x, 0, 1) * 65535 + 0.5).astype(np.uint16)]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease(x):
    """slow in, slow out, gentler than smoothstep at the ends"""
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def fade(t, t0, t1, fin=0.6, fout=0.6):
    if t < t0 or t > t1:
        return 0.0
    return float(min(1.0, (t - t0) / fin if fin else 1.0, (t1 - t) / fout if fout else 1.0))


# ------------------------------------------------------------------ type
@functools.lru_cache(None)
def font(kind, size, weight=None):
    """kind: serif, italic, hand. size in 1080p units."""
    path = {"serif": "SourceSerif4.ttf", "italic": "SourceSerif4-Italic.ttf", "hand": "Caveat.ttf"}[kind]
    f = ImageFont.truetype(os.path.join(FONTS, path), int(round(size * S)), layout_engine=ImageFont.Layout.RAQM)
    if kind == "hand":
        f.set_variation_by_axes([weight or 520])
    else:
        f.set_variation_by_axes([weight or 400, max(8, min(60, size * 0.6))])
    return f


def text(d, xy, s, size, col, a=1.0, kind="serif", weight=None, anchor="la"):
    if a <= 0 or not s:
        return
    f = font(kind, size, weight)
    X, Y = xy[0] * S, xy[1] * S
    l, t, r, b = d.textbbox((X, Y), s, font=f, anchor=anchor)
    l, t, r, b = int(l) - 4, int(t) - 4, int(r) + 6, int(b) + 6
    mask = Image.new("L", (r - l, b - t), 0)
    ImageDraw.Draw(mask).text((X - l, Y - t), s, font=f, fill=255, anchor=anchor)
    if a < 1:
        mask = mask.point(lambda v: int(v * a))
    d._image.paste(tuple(col), (l, t, r, b), mask)


def caption(im, lines, a, x=150, y=930, col=INK, size=40, rise=0.0, shadow=False):
    """documentary subtitles: left-aligned serif lines, bottom-left, no box. lines: str or [str]."""
    if a <= 0:
        return
    if isinstance(lines, str):
        lines = [lines]
    d = ImageDraw.Draw(im)
    yy = y - (len(lines) - 1) * size * 1.32 + (1 - a) * 8 + rise
    for ln in lines:
        if shadow:
            text(d, (x + 1.2, yy + 1.6), ln, size, (0, 0, 0), a * 0.55)
        text(d, (x, yy), ln, size, col, a)
        yy += size * 1.32


def note(im, xy, s, a, size=40, col=RED, anchor="la"):
    """a red-pencil note in handwriting"""
    text(ImageDraw.Draw(im), xy, s, size, col, a, kind="hand", anchor=anchor)


# ------------------------------------------------------------------ the camera
class Cam:
    """world (1080p units) to screen (4K px): centre (cx, cy) at the middle of the frame, zoom z"""

    def __init__(self, cx=960.0, cy=540.0, z=1.0):
        self.cx, self.cy, self.z = cx, cy, z

    def X(self, x):
        return (np.asarray(x, np.float64) - self.cx) * self.z * S + W / 2

    def Y(self, y):
        return (np.asarray(y, np.float64) - self.cy) * self.z * S + H / 2

    def wx(self, X):
        return (X - W / 2) / (self.z * S) + self.cx

    def wy(self, Y):
        return (Y - H / 2) / (self.z * S) + self.cy

    @staticmethod
    def lerp(a, b, u):
        """move between cameras: centre linearly in screen terms, zoom geometrically"""
        z = math.exp(math.log(a.z) + (math.log(b.z) - math.log(a.z)) * u)
        # keep the motion smooth on screen: interpolate the centre weighted by zoom
        wa, wb = (1 - u) * a.z, u * b.z
        return Cam((a.cx * wa + b.cx * wb) / (wa + wb), (a.cy * wa + b.cy * wb) / (wa + wb), z)


# ------------------------------------------------------------------ paper
WORLD = (-600.0, -400.0, 2520.0, 3900.0)                 # the desk's extent in world units (cards stack downward)
TEX_S = 2.0                                              # texture px per world unit


@functools.lru_cache(None)
def paper_tex():
    path = os.path.join(OUT, "paper_tex.png")
    if os.path.exists(path):
        return cv2.imread(path)[..., ::-1].copy()
    os.makedirs(OUT, exist_ok=True)
    x0, y0, x1, y1 = WORLD
    tw, th = int((x1 - x0) * TEX_S), int((y1 - y0) * TEX_S)
    rng = np.random.default_rng(17)
    lum = np.zeros((th, tw), np.float32)
    for sig, amp in ((420, 0.012), (140, 0.010), (40, 0.008), (6, 0.008), (1.2, 0.014)):
        n = rng.normal(0, 1, (th, tw)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sig)
        lum += n / (n.std() + 1e-9) * amp
    # fibres: short curved strands, a shade lighter or darker
    fib = np.zeros((th, tw), np.float32)
    for _ in range(9000):
        x, y = rng.uniform(0, tw), rng.uniform(0, th)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(15, 90)
        bend = rng.normal(0, 0.4)
        pts = []
        for k in range(8):
            u = k / 7
            a_ = ang + bend * (u - 0.5)
            pts.append((x + math.cos(a_) * L * (u - 0.5), y + math.sin(a_) * L * (u - 0.5)))
        cv2.polylines(fib, [np.round(np.array(pts) * 4).astype(np.int32)], False, float(rng.choice([-1, 1]) * rng.uniform(0.3, 1)),
                      1, cv2.LINE_AA, shift=2)
    lum += cv2.GaussianBlur(fib, (0, 0), 0.6) * 0.045
    base = np.array(PAPER, np.float32)
    tex = np.clip(base[None, None, :] * (1 + lum[..., None]) + np.array([2, 0, -3], np.float32) * lum[..., None] * 20, 0, 255)
    tex = tex.astype(np.uint8)
    cv2.imwrite(path, tex[..., ::-1])
    return tex


def paper(cam, box=None):
    """the desk under the camera, uint8 sRGB, for the screen box (X0, Y0, X1, Y1) or the whole frame"""
    X0, Y0, X1, Y1 = box or (0, 0, W, H)
    tex = paper_tex()
    k = cam.z * S / TEX_S
    # screen = (world - c) * z * S + centre; texture = (world - WORLD0) * TEX_S
    M = np.array([[k, 0, (WORLD[0] - cam.cx) * cam.z * S + W / 2 - X0],
                  [0, k, (WORLD[1] - cam.cy) * cam.z * S + H / 2 - Y0]], np.float32)
    return cv2.warpAffine(tex, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR if k > 0.7 else cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REFLECT)


@functools.lru_cache(None)
def lamp():
    """the lamp: brighter top left, a soft vignette"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = x / W, y / H
    g = 1.03 - 0.10 * np.hypot(u - 0.28, v - 0.18) ** 1.4
    vig = 1 - 0.22 * (np.hypot((u - 0.5) * 1.15, v - 0.5) / 0.75) ** 2.6
    return np.clip(g * vig, 0.6, 1.08).astype(np.float32)


@functools.lru_cache(None)
def lamp_u8():
    return cv2.merge([np.round(lamp() * 200).astype(np.uint8)] * 3)


@functools.lru_cache(None)
def grains():
    """six frames of film grain (about 1.6 levels), as positive and negative uint8 parts"""
    rng = np.random.default_rng(3)
    out = []
    for _ in range(6):
        n = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 0.7), (W, H), interpolation=cv2.INTER_LINEAR)
        n = np.round(n / n.std() * 1.6)
        pos = cv2.merge([np.clip(n, 0, 255).astype(np.uint8)] * 3)
        neg = cv2.merge([np.clip(-n, 0, 255).astype(np.uint8)] * 3)
        out.append((pos, neg))
    return out


def finish(fr, t, keep=None, light=True, grain=True):
    """lamp light and film grain over the whole frame; inside `keep` (screen box) only the light, so the
    test line's surroundings stay exactly as drawn"""
    if light:
        fr = cv2.multiply(fr, lamp_u8(), scale=1 / 200)
    if grain:
        clean = None
        if keep is not None:
            X0, Y0, X1, Y1 = [int(v) for v in keep]
            X0, Y0, X1, Y1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
            if X1 > X0 and Y1 > Y0:
                clean = (X0, Y0, X1, Y1, fr[Y0:Y1, X0:X1].copy())
        pos, neg = grains()[int(round(t * FPS)) % 6]
        fr = cv2.subtract(cv2.add(fr, pos), neg)
        if clean is not None:
            X0, Y0, X1, Y1, c = clean
            fr[Y0:Y1, X0:X1] = c
    return fr


# ------------------------------------------------------------------ exact drawing
def rect_cov(box, X0, Y0, X1, Y1):
    """exact area coverage of a screen rectangle (4K px, floats) over the pixels of box"""
    bx0, by0, bx1, by1 = box
    xs = np.arange(bx0, bx1, dtype=np.float64)
    ys = np.arange(by0, by1, dtype=np.float64)
    cx = np.clip(np.minimum(X1, xs + 1) - np.maximum(X0, xs), 0, 1)
    cy = np.clip(np.minimum(Y1, ys + 1) - np.maximum(Y0, ys), 0, 1)
    return np.outer(cy, cx).astype(np.float32)


def blend_lin(region, cov, col):
    """composite colour `col` (sRGB) with coverage cov over a uint8 region, in linear light"""
    lin = to_lin(region)
    c = cov[..., None]
    out = lin * (1 - c) + c * to_lin(np.array(col, np.uint8))[None, None, :]
    return to_srgb(out)


# ------------------------------------------------------------------ the card and the test line
class Card:
    """a printed card on the desk with a horizontal line on it, its right half `step` units lower"""

    def __init__(self, x0=330, y0=330, x1=1590, y1=750, line_y=540, line_x0=420, line_x1=1500, thick=2.0, step=0.5,
                 xm=960.0, label=""):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.line_y, self.lx0, self.lx1, self.th, self.step, self.xm = line_y, line_x0, line_x1, thick, step, xm
        self.label = label

    def draw(self, fr, cam, box=(0, 0, W, H), alpha=1.0, line=1.0):
        """draw onto fr (uint8, the screen box region)"""
        bx0, by0, bx1, by1 = box
        X0, X1 = cam.X(self.x0), cam.X(self.x1)
        Y0, Y1 = cam.Y(self.y0), cam.Y(self.y1)
        # soft shadow (worked at a quarter of the resolution), then the card
        sx0, sy0 = int(max(bx0, X0 - 60)), int(max(by0, Y0 - 60))
        sx1, sy1 = int(min(bx1, X1 + 140)), int(min(by1, Y1 + 160))
        if sx1 > sx0 + 8 and sy1 > sy0 + 8:
            q = 4
            m = np.zeros(((sy1 - sy0) // q + 1, (sx1 - sx0) // q + 1), np.float32)
            o = 10 * cam.z
            cv2.rectangle(m, (int((X0 + o - sx0) / q), int((Y0 + 1.6 * o - sy0) / q)), (int((X1 + o - sx0) / q), int((Y1 + 1.6 * o - sy0) / q)), 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), max(0.5, 9 * cam.z / q))
            m = cv2.resize(m, (m.shape[1] * q, m.shape[0] * q), interpolation=cv2.INTER_LINEAR)[:sy1 - sy0, :sx1 - sx0]
            mul = np.round((1 - 0.28 * alpha * m) * 256).astype(np.uint16)
            reg = fr[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0]
            fr[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0] = ((reg.astype(np.uint16) * mul[..., None]) >> 8).astype(np.uint8)
        cx0, cy0, cx1, cy1 = int(max(bx0, math.floor(X0))), int(max(by0, math.floor(Y0))), int(min(bx1, math.ceil(X1))), int(min(by1, math.ceil(Y1)))
        if cx1 <= cx0 or cy1 <= cy0:
            return
        if alpha >= 1:
            # the whole pixels inside, then the four edges with their exact coverage
            ix0, iy0, ix1, iy1 = int(max(bx0, math.ceil(X0))), int(max(by0, math.ceil(Y0))), int(min(bx1, math.floor(X1))), int(min(by1, math.floor(Y1)))
            if ix1 > ix0 and iy1 > iy0:
                fr[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0] = CARD
            for sub in ((cx0, cy0, cx1, min(cy1, iy0 + 1)), (cx0, max(cy0, iy1 - 1), cx1, cy1),
                        (cx0, cy0, min(cx1, ix0 + 1), cy1), (max(cx0, ix1 - 1), cy0, cx1, cy1)):
                if sub[2] > sub[0] and sub[3] > sub[1]:
                    cov = rect_cov(sub, X0, Y0, X1, Y1)
                    reg = fr[sub[1] - by0:sub[3] - by0, sub[0] - bx0:sub[2] - bx0]
                    fr[sub[1] - by0:sub[3] - by0, sub[0] - bx0:sub[2] - bx0] = blend_lin(reg, cov, CARD)
        else:
            sub = (cx0, cy0, cx1, cy1)
            cov = rect_cov(sub, X0, Y0, X1, Y1) * alpha
            reg = fr[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
            fr[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0] = blend_lin(reg, cov, CARD)
        if line > 0:
            self.draw_line(fr, cam, box, line * alpha)

    def plate(self, im, cam, a=1.0):
        """the card's plate number, printed small in its lower left corner"""
        if self.label and a > 0:
            d = ImageDraw.Draw(im)
            text(d, ((cam.X(self.x0 + 36) / S), (cam.Y(self.y1 - 44) / S)), self.label, 22 * cam.z, INK_SOFT, a, kind="italic")

    def line_box(self, cam, pad=30):
        return (int(cam.X(self.x0)), int(cam.Y(self.y0)), int(cam.X(self.x1)), int(cam.Y(self.y1)))

    def draw_line(self, fr, cam, box, a):
        bx0, by0, bx1, by1 = box
        segs = [(self.lx0, self.xm, self.line_y), (self.xm, self.lx1, self.line_y + self.step)]
        ylo, yhi = cam.Y(self.line_y - 4), cam.Y(self.line_y + self.step + self.th + 4)
        rx0, rx1 = int(max(bx0, math.floor(cam.X(self.lx0)) - 2)), int(min(bx1, math.ceil(cam.X(self.lx1)) + 2))
        ry0, ry1 = int(max(by0, math.floor(ylo))), int(min(by1, math.ceil(yhi)))
        if rx1 <= rx0 or ry1 <= ry0:
            return
        sub = (rx0, ry0, rx1, ry1)
        cov = np.zeros((ry1 - ry0, rx1 - rx0), np.float32)
        for xa, xb, y in segs:
            cov += rect_cov(sub, cam.X(xa), cam.Y(y), cam.X(xb), cam.Y(y + self.th))
        cov = np.clip(cov, 0, 1) * a
        reg = fr[ry0 - by0:ry1 - by0, rx0 - bx0:rx1 - bx0]
        fr[ry0 - by0:ry1 - by0, rx0 - bx0:rx1 - bx0] = blend_lin(reg, cov, INK)


# ------------------------------------------------------------------ pencil
def wobble(pts, amp, seed, freq=3.0):
    """a hand-drawn version of a polyline (screen px): low-frequency sideways drift"""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    L = max(s[-1], 1e-6)
    n = max(2, int(L / 6))
    u = np.linspace(0, L, n)
    P = np.c_[np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])]
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 3)
    d = np.gradient(P, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    off = amp * (np.sin(2 * np.pi * freq * u / L * 0.5 + ph[0]) * 0.6 + np.sin(2 * np.pi * freq * u / L * 1.7 + ph[1]) * 0.4)
    return P + nrm * off[:, None]


@functools.lru_cache(None)
def pencil_grain():
    rng = np.random.default_rng(9)
    n = rng.uniform(0.55, 1.0, (H, W)).astype(np.float32)
    return cv2.GaussianBlur(n, (0, 0), 0.8)


def pencil(fr, strokes, col=RED, a=1.0, width=3.0):
    """strokes: list of (screen points, drawn fraction 0..1, seed). Graphite-textured, anti-aliased."""
    if a <= 0:
        return fr
    m = np.zeros((H, W), np.float32)
    for pts, f, seed in strokes:
        if f <= 0:
            continue
        P = wobble(pts, 2.2 * S / 2, seed)
        k = max(2, int(len(P) * min(f, 1.0)))
        P = P[:k]
        cv2.polylines(m, [np.round(P * 16).astype(np.int32)], False, 1.0, max(1, int(width * S / 2 * 2)), cv2.LINE_AA, shift=4)
    if not m.any():
        return fr
    ys, xs = np.nonzero(m > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    mm = (m[y0:y1, x0:x1] * pencil_grain()[y0:y1, x0:x1] * a * 0.92)[..., None]
    reg = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = np.clip(reg * (1 - mm) + np.array(col, np.float32) * mm, 0, 255).astype(np.uint8)
    return fr


# ------------------------------------------------------------------ the magnifier
def loupe(fr, inner, cx, cy, R, a=1.0, t=0.0, handle=True):
    """a glass hand magnifier at screen (cx, cy) (4K px), radius R (4K px). `inner` is the magnified view
    rendered for the square box (cx-R, cy-R, cx+R, cy+R), uint8. Barrel distortion and colour fringes
    near the rim, a glass sheen, a metal rim, a handle and a shadow on the desk."""
    if a <= 0:
        return fr
    R = int(R)
    X0, Y0 = int(cx) - R, int(cy) - R
    n = 2 * R
    # shadow on the desk (down-right of the lens)
    pad = int(R * 0.6)
    sx0, sy0, sx1, sy1 = max(0, X0 - pad), max(0, Y0 - pad), min(W, X0 + n + 2 * pad), min(H, Y0 + n + 2 * pad)
    sm = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
    cv2.circle(sm, (int(cx + R * 0.10 - sx0), int(cy + R * 0.16 - sy0)), int(R * 1.04), 1.0, -1, cv2.LINE_AA)
    if handle:
        hx, hy = cx + R * 0.71 * 1.05, cy + R * 0.71 * 1.05
        cv2.line(sm, (int(hx + R * 0.1 - sx0), int(hy + R * 0.16 - sy0)), (int(hx + R * 1.1 - sx0), int(hy + R * 1.16 - sy0)),
                 1.0, int(R * 0.22), cv2.LINE_AA)
    sm = cv2.GaussianBlur(sm, (0, 0), R * 0.08)
    reg = fr[sy0:sy1, sx0:sx1].astype(np.float32)
    fr[sy0:sy1, sx0:sx1] = np.clip(reg * (1 - 0.30 * a * sm[..., None]), 0, 255).astype(np.uint8)
    # the lens: radial barrel distortion and lateral colour near the rim. The working box holds the rim
    # (to 1.1 R); `inner` covers the lens itself (radius R).
    Rr = int(math.ceil(R * 1.12))
    n2 = 2 * Rr
    X0, Y0 = int(cx) - Rr, int(cy) - Rr
    yy, xx = np.mgrid[0:n2, 0:n2].astype(np.float32)
    dx, dy = (xx - Rr + 0.5) / R, (yy - Rr + 0.5) / R
    r2 = dx * dx + dy * dy
    out = np.empty((n2, n2, 3), np.uint8)
    for ch, k in ((0, 1.006), (1, 1.0), (2, 0.994)):
        sc = (1 - 0.07 * np.minimum(r2, 1)) * (1 + (k - 1) * np.minimum(r2, 1))
        mx = (dx * sc * R + R - 0.5).astype(np.float32)
        my = (dy * sc * R + R - 0.5).astype(np.float32)
        out[..., ch] = cv2.remap(np.ascontiguousarray(inner[..., ch]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g = out.astype(np.float32)
    rr = np.sqrt(r2)
    g *= (1.02 - 0.12 * np.clip(rr - 0.78, 0, 1) / 0.22)[..., None]                    # darker toward the rim
    up = np.clip((-dy - 0.05) / 0.5, 0, 1)
    sheen = (np.exp(-((dx + 0.40) ** 2 + (dy + 0.50) ** 2 * 2.2) / 0.035) * 0.035
             + np.exp(-((rr - 0.90) ** 2) / 0.0012) * 0.05 * up * up)
    g *= (1 - 0.05 * np.clip(rr - 0.6, 0, 1))[..., None] * np.array([0.995, 1.0, 0.99], np.float32)
    g += (sheen * 255)[..., None]
    # rim: dark gunmetal ring with a highlight
    rim_t = np.clip((rr - 1.0) / 0.10, 0, 1)
    rim_light = 0.5 + 0.5 * np.cos(np.arctan2(dy, dx) + 2.3)
    rim_col = (34 + 46 * rim_light * (1 - np.abs(rim_t - 0.45) * 1.6)).astype(np.float32)
    a_in = np.clip((1.0 - rr) * R + 0.5, 0, 1) * a
    a_rim = np.clip(np.minimum(rr - 1.0 + 0.5 / R, 1.10 - rr) * R + 0.5, 0, 1) * a
    bx0, by0, bx1, by1 = max(0, X0), max(0, Y0), min(W, X0 + n2), min(H, Y0 + n2)
    ox0, oy0 = bx0 - X0, by0 - Y0
    sl = (slice(oy0, oy0 + by1 - by0), slice(ox0, ox0 + bx1 - bx0))
    base = fr[by0:by1, bx0:bx1].astype(np.float32)
    lens = np.clip(g[sl], 0, 255)
    rc = np.repeat(rim_col[sl][..., None], 3, axis=2) * np.array([1.0, 0.98, 0.95], np.float32)
    ai, ar = a_in[sl][..., None], a_rim[sl][..., None]
    base = base * (1 - ai) + lens * ai
    base = base * (1 - ar) + rc * ar
    fr[by0:by1, bx0:bx1] = np.clip(base, 0, 255).astype(np.uint8)
    if handle:
        hx0, hy0 = cx + R * 1.06 * 0.7071, cy + R * 1.06 * 0.7071
        hx1, hy1 = cx + R * 2.05 * 0.7071 * 1.35, cy + R * 2.05 * 0.7071 * 1.35
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        wdt = int(R * 0.20)
        d.line([(hx0, hy0), (hx1, hy1)], fill=(52, 34, 26, int(255 * a)), width=wdt)
        d.line([(hx0 - wdt * 0.18, hy0 + wdt * 0.18), (hx1 - wdt * 0.18, hy1 + wdt * 0.18)], fill=(96, 66, 48, int(140 * a)),
               width=max(2, int(wdt * 0.22)))
        d.line([(hx0, hy0), (hx0 + R * 0.12, hy0 + R * 0.12)], fill=(60, 60, 62, int(255 * a)), width=int(wdt * 1.15))
        fr[:] = np.asarray(im)
    return fr


# ------------------------------------------------------------------ sections
class Section:
    name, dur = "section", 1.0

    def frame(self, t):
        raise NotImplementedError

    def patch(self, t0, t1):
        """re-render only the frames in [t0, t1) of an already rendered section, keeping the rest"""
        path = os.path.join(OUT, f"{self.name}.mp4")
        tmp = path + ".patch.mp4"
        n = int(round(self.dur * FPS))
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                                "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
        size = W * H * 3
        for i in range(n):
            buf = dec.stdout.read(size)
            if len(buf) < size:
                raise RuntimeError(f"{self.name}: only {i} frames in {path}")
            if t0 <= i / FPS < t1:
                buf = np.ascontiguousarray(self.frame(i / FPS)).tobytes()
            enc.stdin.write(buf)
        dec.stdout.close(); dec.wait()
        enc.stdin.close(); enc.wait()
        os.replace(tmp, path)
        print(f"patched {self.name} {t0}-{t1} s", flush=True)

    def render(self, still=None):
        os.makedirs(OUT, exist_ok=True)
        if still is not None:
            Image.fromarray(self.frame(still)).save(os.path.join(OUT, f"{self.name}_{still:.1f}.png"))
            return
        CHUNK = 120
        cdir = os.path.join(OUT, "chunks")
        os.makedirs(cdir, exist_ok=True)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        parts = []
        for c0 in range(0, n, CHUNK):
            c1 = min(n, c0 + CHUNK)
            part = os.path.join(cdir, f"{self.name}_{c0:05d}.mp4")
            parts.append(part)
            if os.path.exists(part + ".ok"):
                continue
            tmp = part + ".tmp.mp4"
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                  "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                                  "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
            for i in range(c0, c1):
                p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            p.stdin.close(); p.wait()
            os.replace(tmp, part)
            open(part + ".ok", "w").close()
            print(f"{self.name}: frames {c0}-{c1}/{n} {time.time() - t0:.0f}s", flush=True)
        lst = os.path.join(cdir, f"{self.name}.txt")
        open(lst, "w").write("".join(f"file '{q}'\n" for q in parts))
        path = os.path.join(OUT, f"{self.name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path], check=True)
        for q in parts:
            os.remove(q); os.remove(q + ".ok")
        print("wrote", path, flush=True)
