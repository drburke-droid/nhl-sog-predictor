"""The look of the fantasy race: a deep navy field with a soft light from above, film grain and a vignette; condensed
display type (Barlow Condensed, SIL OFL, in fonts/) over Inter; bars with a vertical sheen, a bright top edge and a
glow for the season being played; gold for records.
"""
import os, sys, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "signal"))
import style as ST                                    # Inter, and translucent text  # noqa: E402

W, H = 1920, 1080
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
GOLD = (255, 196, 92)
PANEL = (9, 14, 28)
GRID = (44, 58, 86)
POSC = {"QB": (213, 81, 129), "RB": (57, 135, 229), "WR": (201, 133, 0), "TE": (0, 131, 0)}
inter = ST.font


@functools.lru_cache(None)
def bc(size, weight="Bold"):
    """Barlow Condensed"""
    return ImageFont.truetype(os.path.join(HERE, "fonts", f"BarlowCondensed-{weight}.ttf"), size,
                              layout_engine=ImageFont.Layout.RAQM)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


_BG = {}


def background(grain=True):
    """deep navy, lit softly from above the board, a faint diagonal beam, vignette and fine grain"""
    if grain not in _BG:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W * 0.46) / (W * 0.75)) ** 2 + ((yy - H * 0.30) / (H * 0.95)) ** 2)
        glow = np.clip(1 - r, 0, 1) ** 1.6
        top, base = np.array([16, 26, 50], np.float32), np.array([3, 5, 11], np.float32)
        img = base + (top - base) * glow[..., None]
        beam = np.exp(-((xx * 0.42 + yy * 0.91 - 520) / 210) ** 2) * np.clip(1 - yy / H, 0, 1)
        img += beam[..., None] * np.array([7, 10, 18], np.float32)
        vig = 1 - 0.45 * np.clip(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.70)) ** 2 - 0.25, 0, 1)
        img *= vig[..., None]
        if grain:
            img += np.random.default_rng(3).normal(0, 1.6, (H, W, 1)).astype(np.float32)
        _BG[grain] = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(_BG[grain].copy())


@functools.lru_cache(None)
def _tile(col, h):
    """a bar's fill: lighter at the top, darker at the base, brighter toward the tip"""
    c = np.array(col, np.float32)
    v = np.linspace(1.22, 0.78, h)[:, None, None]
    u = np.linspace(0.86, 1.06, 1400)[None, :, None]
    img = np.clip(c[None, None, :] * v * u + (v - 1).clip(0) * 60, 0, 255).astype(np.uint8)
    img[:2] = np.clip(img[:2].astype(np.int16) + 46, 0, 255)          # a bright top edge
    return Image.fromarray(img)


@functools.lru_cache(None)
def _mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    return m


def glow(im, box, col, alpha, spread=16):
    """a soft halo around a box"""
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    p = spread * 2
    m = Image.new("L", (x1 - x0 + 2 * p, y1 - y0 + 2 * p), 0)
    ImageDraw.Draw(m).rounded_rectangle([p, p, p + x1 - x0, p + y1 - y0], (y1 - y0) // 3, fill=int(255 * min(1, alpha)))
    m = m.filter(ImageFilter.GaussianBlur(spread * 0.6))
    im.paste(tuple(int(c) for c in col), (x0 - p, y0 - p), m)


def bar(im, x0, y, x1, h, col, alpha=1.0):
    """a rounded bar with a vertical sheen"""
    w = int(round(x1 - x0))
    if w < 3 or alpha <= 0.01:
        return
    x0, y = int(round(x0)), int(round(y))
    t = _tile(tuple(col), h).crop((1400 - w, 0, 1400, h))
    m = _mask(w, h, max(3, h // 5))
    if alpha < 0.999:
        m = m.point(lambda v: int(v * alpha))
    im.paste(t, (x0, y), m)


def pill(d, x, y, text, fill, ink=(10, 12, 20), size=20, weight="Bold", anchor="l", alpha=1.0, pad=10):
    """a rounded label; anchor l: x is its left edge, r: its right edge, m: its centre; y is its middle"""
    f = bc(size, weight)
    tw = d.textlength(text, font=f)
    w = tw + 2 * pad
    x0 = x if anchor == "l" else x - w if anchor == "r" else x - w / 2
    hh = size * 0.62 + 6
    d.rounded_rectangle([x0, y - hh, x0 + w, y + hh], hh, fill=tuple(fill) + (int(255 * alpha),))
    d.text((x0 + w / 2, y + 1), text, font=f, fill=tuple(ink) + (int(255 * alpha),), anchor="mm")
    return w
