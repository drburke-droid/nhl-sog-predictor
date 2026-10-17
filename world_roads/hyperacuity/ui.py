"""Shared look and exact drawing for the hyperacuity video (rendered at 4K, laid out in 1080p units).

Every stimulus offset in this video is in pixels of a 1080p frame ("1 px" = 2x2 pixels of the 4K upload),
so a viewer watching at 1080p sees exactly the stated offset. Stimuli are drawn with exact area coverage
in linear light and only then encoded to sRGB, so a line placed a quarter pixel over really has its
light centred a quarter pixel over on a correctly calibrated screen.
"""
import os, sys, math, time, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                   # Inter fonts, and Pillow's translucent-text fix

OUT = os.path.join(HERE, "out")
W, H, FPS = 3840, 2160, 30
S = 2                                                # 4K pixels per 1080p pixel
BG = (8, 10, 16)
INK, INK2, MUTED = (240, 244, 250), (170, 186, 206), (120, 136, 158)
GOLD, RED, GREEN, CYAN = (255, 186, 80), (255, 110, 90), (110, 220, 140), (90, 200, 255)
PANEL = (6, 10, 18)
FG_LUM = 0.80                                        # stimulus white, linear light (sRGB ~231)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


fade = ST.fade


def font(size, weight="SemiBold"):
    """size in 1080p units"""
    return ST.font(int(round(size * S)), weight)


# ---------------------------------------------------------------- linear light
def to_lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


BG_LIN = to_lin(BG)


class Patch:
    """A rectangle of the frame (4K pixel box) holding coverage of a white stimulus, composited in linear light.
    Coordinates passed to the drawing methods are in 1080p units, measured from the frame's top-left."""

    def __init__(self, x0, y0, x1, y1, s=S):
        self.s = s                                                         # output pixels per 1080p pixel
        self.X0, self.Y0 = int(math.floor(x0 * s)), int(math.floor(y0 * s))
        self.X1, self.Y1 = int(math.ceil(x1 * s)), int(math.ceil(y1 * s))
        self.cov = np.zeros((self.Y1 - self.Y0, self.X1 - self.X0), np.float32)
        self.xs = np.arange(self.X0, self.X1, dtype=np.float64)          # 4K pixel left edges
        self.ys = np.arange(self.Y0, self.Y1, dtype=np.float64)

    def rect(self, x0, y0, x1, y1):
        """Exact area coverage of an axis-aligned rectangle (1080p units)."""
        s = self.s
        cx = np.clip(np.minimum(x1 * s, self.xs + 1) - np.maximum(x0 * s, self.xs), 0, 1)
        cy = np.clip(np.minimum(y1 * s, self.ys + 1) - np.maximum(y0 * s, self.ys), 0, 1)
        self.cov = np.minimum(1.0, self.cov + np.outer(cy, cx).astype(np.float32))

    def sdf(self, fn):
        """Coverage from a signed distance (1080p units, negative inside), at 4K pixel centres."""
        X, Y = np.meshgrid((self.xs + 0.5) / self.s, (self.ys + 0.5) / self.s)
        d = fn(X, Y) * self.s                                              # in output pixels
        self.cov = np.maximum(self.cov, np.clip(0.5 - d, 0, 1).astype(np.float32))

    def srgb(self, lum=FG_LUM):
        """The patch on its own, over BG, as uint8 sRGB (for magnifiers)."""
        lin = BG_LIN * (1 - self.cov[..., None]) + self.cov[..., None] * lum
        return np.round(to_srgb(lin)).astype(np.uint8)

    def paste(self, fr, lum=FG_LUM, tint=(1.0, 1.0, 1.0), alpha=1.0):
        """Composite onto the uint8 sRGB frame (in place). The frame under the patch is assumed to be BG."""
        if alpha <= 0:
            return
        under = to_lin(fr[self.Y0:self.Y1, self.X0:self.X1])
        c = self.cov[..., None] * alpha
        lin = under * (1 - c) + c * lum * np.asarray(tint, np.float32)
        fr[self.Y0:self.Y1, self.X0:self.X1] = np.round(to_srgb(lin)).astype(np.uint8)


def blank():
    fr = np.empty((H, W, 3), np.uint8)
    fr[:] = BG
    return fr


# ---------------------------------------------------------------- captions and text
def caption(d, lines, a, anchor="bottom", y=None, x=None):
    """Rounded panel of centred lines [(text, size, colour, weight)], sizes in 1080p units."""
    if a <= 0:
        return
    pad, gap = 30 * S, 14 * S
    fs = [font(sz, wt) for _, sz, _, wt in lines]
    ws = [d.textlength(tx, font=f) for (tx, *_), f in zip(lines, fs)]
    hs = [int(sz * 1.18 * S) for _, sz, _, _ in lines]
    bw = int(max(ws) + 2 * pad)
    bh = int(sum(hs) + gap * (len(lines) - 1) + 2 * pad)
    if x is None:
        x = (W - bw) // 2
    if y is None:
        y = H - bh - 70 * S if anchor == "bottom" else 70 * S
    else:
        y = int(y * S)
    d.rounded_rectangle([x, y, x + bw, y + bh], 22 * S, fill=PANEL + (int(205 * a),))
    yy = y + pad
    for (tx, sz, col, wt), f, w, h in zip(lines, fs, ws, hs):
        d.text((x + bw / 2, yy), tx, font=f, fill=tuple(col) + (int(255 * a),), anchor="ma")
        yy += h + gap


def captions(fr, t, caps):
    """caps: [(t0, t1, lines, kwargs)]"""
    act = [(c, fade(t, c[0], c[1])) for c in caps]
    act = [(c, a) for c, a in act if a > 0]
    if not act:
        return fr
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    for c, a in act:
        caption(d, c[2], a, **(c[3] if len(c) > 3 else {}))
    return np.asarray(im).copy()


def text(d, xy, s, size, col, a=1.0, weight="SemiBold", anchor="mm"):
    if a > 0:
        d.text((xy[0] * S, xy[1] * S), s, font=font(size, weight), fill=tuple(col) + (int(255 * a),), anchor=anchor)


def frac(x):
    """1, 1/2, 1/4 ... as a pretty string"""
    return {0.5: "½", 0.25: "¼", 0.125: "⅛", 0.0625: "1/16"}.get(x, f"{x:g}")


# ---------------------------------------------------------------- rendering
class Section:
    name, dur = "section", 1.0

    def frame(self, t):
        raise NotImplementedError

    def render(self, still=None, parts=1):
        os.makedirs(OUT, exist_ok=True)
        if still is not None:
            Image.fromarray(self.frame(still)).save(os.path.join(OUT, f"{self.name}_{still:.1f}.png"))
            return
        path = os.path.join(OUT, f"{self.name}.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                              "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        for i in range(n):
            p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            if i % 150 == 0:
                print(f"{self.name}: frame {i}/{n} {time.time() - t0:.0f}s", flush=True)
        p.stdin.close(); p.wait()
        print("wrote", path, flush=True)


def magnify(fr, draw, src, dst, radius, mag, a, lines=(), grid=True):
    """A round magnifier: draw(patch) paints the stimulus (1080p units) into a 1080p-pixel patch around
    src; it is shown mag times larger, centred on dst, with the pixel grid. lines: [(text, size, colour, weight)]."""
    if a <= 0:
        return fr
    half = radius / mag + 2
    q = Patch(src[0] - half, src[1] - half, src[0] + half, src[1] + half, s=1)
    draw(q)
    img = q.srgb()
    k = mag * S
    R = int(radius * S)
    # sample the magnified image directly (nearest neighbour), so src lands exactly on dst
    yy, xx = np.mgrid[-R:R, -R:R].astype(np.float64) + 0.5
    sx = src[0] - q.X0 + xx / k
    sy = src[1] - q.Y0 + yy / k
    ix = np.clip(np.floor(sx).astype(int), 0, img.shape[1] - 1)
    iy = np.clip(np.floor(sy).astype(int), 0, img.shape[0] - 1)
    crop = img[iy, ix].astype(np.int16)
    if grid:
        edge = (np.abs(sx - np.round(sx)) * k < S) | (np.abs(sy - np.round(sy)) * k < S)
        crop[edge] = (crop[edge] * 0.55 + 18).astype(np.int16)
    crop = crop.astype(np.uint8)
    im = Image.fromarray(fr)
    mask = Image.new("L", (2 * R, 2 * R), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 2 * R - 1, 2 * R - 1], fill=int(255 * a))
    X, Y = int(dst[0] * S) - R, int(dst[1] * S) - R
    im.paste(Image.fromarray(crop), (X, Y), mask)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.ellipse([X, Y, X + 2 * R, Y + 2 * R], outline=GOLD + (int(255 * a),), width=5 * S)
    ty = Y + 2 * R + 22 * S
    for tx, sz, col, wt in lines:
        f = font(sz, wt)
        w = dr.textlength(tx, font=f)
        cx = min(max(X + R, w / 2 + 30 * S), W - w / 2 - 30 * S)
        dr.text((cx, ty), tx, font=f, fill=tuple(col) + (int(255 * a),), anchor="ma")
        ty += int(sz * 1.25 * S)
    return np.asarray(im).copy()


def countdown(d, left, total, x=1710, y=120):
    cxr, cyr, rr = x * S, y * S, 52 * S
    d.ellipse([cxr - rr, cyr - rr, cxr + rr, cyr + rr], outline=(60, 70, 90, 255), width=8 * S)
    d.arc([cxr - rr, cyr - rr, cxr + rr, cyr + rr], -90, -90 + 360 * left / total, fill=GOLD + (255,), width=8 * S)
    text(d, (x, y + 2), f"{math.ceil(left)}", 50, INK, 1.0, "Bold")


class Quiz(Section):
    """'Which one is different?': items appear, a countdown, then the answer is ringed and explained.

    draw(patch, i, odd): paint item i (odd: is it the different one) into the patch (1080p units).
    reveal(fr, u, a) -> fr: extra overlay after the answer (magnifier, exaggeration...)."""
    T_IN, T_LOOK, T_SHOW = 1.2, 5.0, 4.0

    def __init__(self, name, title, sub, box, labels, label_xy, ring, answer, draw, reveal=None, caps=(), look=None, show=None):
        self.name, self.title, self.sub = name, title, sub
        self.box, self.labels, self.label_xy, self.ring = box, labels, label_xy, ring
        self.answer, self.draw, self.reveal, self.caps = answer, draw, reveal, list(caps)
        self.T_LOOK = look or self.T_LOOK
        self.T_SHOW = show or self.T_SHOW
        self.t_rev = self.T_IN + self.T_LOOK
        self.dur = self.t_rev + self.T_SHOW + 0.4
        self.sfx = [(self.T_IN + k, "tick") for k in range(int(self.T_LOOK))] + [(self.t_rev, "ding")]

    def frame(self, t):
        fr = blank()
        a = fade(t, 0.25, self.dur - 0.05, 0.4, 0.35)
        p = Patch(*self.box)
        for i in range(len(self.labels)):
            self.draw(p, i, i == self.answer)
        p.paste(fr, alpha=a)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        text(d, (960, 120), self.title, 54, INK, a, "Bold")
        if self.sub:
            text(d, (960, 182), self.sub, 32, INK2, a, "Regular")
        for i, (lab, (lx, ly)) in enumerate(zip(self.labels, self.label_xy)):
            hit = t >= self.t_rev and i == self.answer
            text(d, (lx, ly), lab, 44, GOLD if hit else INK2, a, "Bold")
        if self.T_IN <= t < self.t_rev:
            countdown(d, self.t_rev - t, self.T_LOOK)
        if t >= self.t_rev:
            g = smooth((t - self.t_rev) / 0.35) * a
            x0, y0, x1, y1 = self.ring
            e = 0.12 * (1 - smooth((t - self.t_rev) / 0.35))
            d.rounded_rectangle([(x0 - e * (x1 - x0)) * S, (y0 - e * (y1 - y0)) * S, (x1 + e * (x1 - x0)) * S, (y1 + e * (y1 - y0)) * S],
                                24 * S, outline=GOLD + (int(255 * g),), width=5 * S)
        fr = np.asarray(im).copy()
        if t >= self.t_rev and self.reveal:
            fr = self.reveal(fr, t - self.t_rev, a)
        return captions(fr, t, self.caps)
