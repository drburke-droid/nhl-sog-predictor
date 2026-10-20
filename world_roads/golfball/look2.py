"""The second look for the golf-ball film: golf broadcast meets vintage golf poster.

* Course scenes are flat, layered shapes (a printed poster's way of drawing trees, fairways and sky) under
  film grain: no photoreal rendering, nothing that could pass for a generated photo.
* The graphics borrow golf's own vocabulary: a shot tracer and a broadcast lower third, a caddie's yardage
  book, a tumbling-E eye chart, a leaderboard, and a telestrator's marker.
* Type: Barlow Condensed for numbers and labels (broadcast), Source Serif 4 for captions (documentary).
* One accent colour (tracer orange) on a restrained palette.

All geometry is in 1080p units; K is screen pixels per unit (1 for 1920×1080 stills, 2 for 4K).
"""
import os, math, functools
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
VFONTS = os.path.join(os.path.dirname(HERE), "vernier", "fonts")

K = 1
W, H = 1920 * K, 1080 * K

# ------------------------------------------------------------------ palette
INK = (16, 23, 18)              # night / ink
PANEL = (14, 20, 16)
TREE_BACK = (52, 78, 60)
TREE_MID = (35, 56, 42)
TREE_FRONT = (22, 36, 27)
TREE_LIT = (58, 88, 56)
FAIR_A = (98, 140, 66)
FAIR_B = (84, 124, 56)
ROUGH = (60, 94, 44)
SKY_TOP = (118, 160, 196)
SKY_LOW = (214, 228, 233)
OVERCAST = (216, 219, 218)
PAPER = (241, 234, 219)
PAPER_INK = (40, 46, 40)
SAND = (226, 210, 168)
TRACER = (236, 78, 40)
GOLD = (244, 197, 66)
WHITE = (255, 255, 255)
CREAM = (246, 240, 228)
MUTED = (150, 158, 150)


# ------------------------------------------------------------------ type
@functools.lru_cache(None)
def font(kind, size):
    """kind: cond-<weight> (Barlow Condensed), sans-<weight> (Barlow), serif, serif-italic. size in 1080p units"""
    px = int(round(size * K))
    if kind.startswith("cond-"):
        return ImageFont.truetype(os.path.join(FONTS, f"BarlowCondensed-{kind[5:]}.ttf"), px, layout_engine=ImageFont.Layout.RAQM)
    if kind.startswith("sans-"):
        return ImageFont.truetype(os.path.join(FONTS, f"Barlow-{kind[5:]}.ttf"), px, layout_engine=ImageFont.Layout.RAQM)
    path = "SourceSerif4-Italic.ttf" if kind == "serif-italic" else "SourceSerif4.ttf"
    f = ImageFont.truetype(os.path.join(VFONTS, path), px, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([420 if kind == "serif" else 400, max(8, min(60, size * 0.6))])
    return f


def text(im, xy, s, size, col, kind="cond-Bold", anchor="la", a=1.0, track=0.0, shadow=None):
    """draw text at 1080p point xy. track: extra letter spacing (units). shadow: (dx, dy, alpha)"""
    if a <= 0 or not s:
        return
    d = ImageDraw.Draw(im, "RGBA")
    f = font(kind, size)
    x, y = xy[0] * K, xy[1] * K
    if track:
        # letter-spaced: lay out glyph by glyph, every glyph on the string's own baseline
        widths = [f.getlength(c) for c in s]
        total = sum(widths) + track * K * (len(s) - 1)
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        if anchor[1] != "s":
            y += f.getbbox(s, anchor="l" + anchor[1])[1] - f.getbbox(s, anchor="ls")[1]
        anchor = "ls"
        for c, w_ in zip(s, widths):
            if shadow:
                d.text((x + shadow[0] * K, y + shadow[1] * K), c, font=f, fill=(0, 0, 0, int(255 * shadow[2] * a)), anchor=anchor)
            d.text((x, y), c, font=f, fill=tuple(col) + (int(255 * a),), anchor=anchor)
            x += w_ + track * K
        return
    if shadow:
        d.text((x + shadow[0] * K, y + shadow[1] * K), s, font=f, fill=(0, 0, 0, int(255 * shadow[2] * a)), anchor=anchor)
    d.text((x, y), s, font=f, fill=tuple(col) + (int(255 * a),), anchor=anchor)


def text_width(s, size, kind="cond-Bold"):
    return font(kind, size).getlength(s) / K


def caption(im, lines, a=1.0, x=120, y=968, size=40, col=CREAM, shadow=True):
    """documentary captions, serif, bottom left"""
    if isinstance(lines, str):
        lines = [lines]
    yy = y - (len(lines) - 1) * size * 1.3
    for ln in lines:
        text(im, (x, yy), ln, size, col, kind="serif", anchor="ls", a=a, shadow=(0, 2, 0.55) if shadow else None)
        yy += size * 1.3


def headline(im, lines, x, y, size=96, col=WHITE, a=1.0, accent=None, kind="cond-ExtraBold", anchor="ls", lead=0.98,
             shadow=True):
    """big condensed caps, first baseline at y. accent: index of a line to colour in TRACER. Returns the
    last line's baseline."""
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        c = TRACER if accent == i else col
        text(im, (x, y + i * size * lead), ln.upper(), size, c, kind=kind, anchor=anchor, a=a, track=size * 0.01,
             shadow=(0, 3, 0.35) if shadow else None)
    return y + (len(lines) - 1) * size * lead


def block(im, head, cap=None, x=120, y=300, size=90, accent=None, cap_size=42, col=WHITE, cap_col=CREAM, shadow=True):
    """a headline with a caption under it, spaced so they never touch"""
    last = headline(im, head, x, y, size=size, accent=accent, col=col, shadow=shadow)
    if cap:
        cap = [cap] if isinstance(cap, str) else cap
        caption(im, cap, x=x + 4, y=last + cap_size * 1.6 + (len(cap) - 1) * cap_size * 1.3, size=cap_size, col=cap_col,
                shadow=shadow)


# ------------------------------------------------------------------ finishing
@functools.lru_cache(None)
def grain_field(seed=0):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (H, W)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6 * K)
    return n / n.std()


@functools.lru_cache(None)
def vignette(strength=0.22):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((x / W - 0.5) * 1.1, (y / H - 0.5)) / 0.72
    return (1 - strength * np.clip(r, 0, 1.4) ** 2.2).astype(np.float32)


def finish(fr, grain=2.2, vig=0.22, seed=0):
    """film grain (stronger in the shadows, like film) and a soft vignette"""
    f = fr.astype(np.float32)
    lum = f.mean(-1, keepdims=True) / 255
    g = grain_field(seed % 6)[..., None] * grain * (1.25 - 0.6 * lum)
    f = f * vignette(vig)[..., None] + g
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ drawing
def poly_mask(shape, polys, ss=4):
    """anti-aliased fill of polygons (lists of (x, y) in screen px)"""
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255


def over(fr, mask, col, a=1.0):
    """paint colour col over fr where mask (0..1)"""
    m = (mask * a)[..., None]
    return (fr.astype(np.float32) * (1 - m) + np.array(col, np.float32) * m).astype(np.uint8)


def marker(fr, pts, col=TRACER, width=7.0, a=1.0, seed=0, wobble=1.2, frac=1.0):
    """a telestrator stroke: thick, round-capped, slightly uneven, as a TV analyst draws it (pts in 1080p units)"""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2 or frac <= 0:
        return fr
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(s[-1] / 2))
    u = np.linspace(0, s[-1] * min(frac, 1.0), n)
    P = np.c_[np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])]
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, 2)
    d = np.gradient(P, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    P = P + nrm * (wobble * np.sin(u / 37 + ph[0]) + 0.5 * wobble * np.sin(u / 11 + ph[1]))[:, None]
    m = np.zeros(fr.shape[:2], np.float32)
    wv = width * (0.85 + 0.15 * np.sin(u / 23 + ph[1]))
    for i in range(len(P) - 1):
        cv2.line(m, tuple(np.round(P[i] * K * 16).astype(int)), tuple(np.round(P[i + 1] * K * 16).astype(int)), 1.0,
                 max(1, int(round(wv[i] * K))), cv2.LINE_AA, shift=4)
    m = cv2.GaussianBlur(m, (0, 0), 0.5 * K)
    return over(fr, np.clip(m, 0, 1), col, 0.92 * a)


def ring(cx, cy, r, n=80, a0=-0.6, extra=0.35):
    th = np.linspace(a0, a0 + 2 * math.pi + extra, n)
    return np.c_[cx + r * np.cos(th), cy + r * np.sin(th) * 0.96]


def lower_third(im, label, value, x=120, y=860, a=1.0, sub=None):
    """a broadcast lower third: a dark plate with a label and a big number"""
    d = ImageDraw.Draw(im, "RGBA")
    wv = text_width(value, 84, "cond-ExtraBold")
    wl = text_width(label, 26, "cond-SemiBold") + 4 * len(label)
    ws = text_width(sub, 26, "cond-SemiBold") + len(sub) if sub else 0
    w_ = max(wv, wl, ws) + 60
    hh = 150 if not sub else 186
    d.rectangle([x * K, (y - hh) * K, (x + w_) * K, y * K], fill=PANEL + (int(225 * a),))
    d.rectangle([x * K, (y - hh) * K, (x + 8) * K, y * K], fill=TRACER + (int(255 * a),))
    text(im, (x + 30, y - hh + 22), label.upper(), 26, MUTED, "cond-SemiBold", "lt", a, track=4)
    text(im, (x + 28, y - hh + 54), value, 84, WHITE, "cond-ExtraBold", "lt", a)
    if sub:
        text(im, (x + 30, y - 22), sub, 26, GOLD, "cond-SemiBold", "ls", a, track=1)


def ball(fr, cx, cy, r, col=WHITE, a=1.0, glow=0.0):
    """the ball: a disk with exact coverage (cx, cy, r in 1080p units)"""
    R = r * K
    X, Y = cx * K, cy * K
    hh, ww = fr.shape[:2]
    x0, x1 = int(max(0, X - R - 3)), int(min(ww, X + R + 4))
    y0, y1 = int(max(0, Y - R - 3)), int(min(hh, Y + R + 4))
    if x1 <= x0 or y1 <= y0:
        return fr
    n = 8
    xs = np.arange(x0, x1)[None, :, None, None] + (np.arange(n)[None, None, None, :] + 0.5) / n
    ys = np.arange(y0, y1)[:, None, None, None] + (np.arange(n)[None, None, :, None] + 0.5) / n
    cov = ((xs - X) ** 2 + (ys - Y) ** 2 <= R * R).mean(axis=(2, 3)).astype(np.float32) * a
    reg = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = (reg * (1 - cov[..., None]) + np.array(col, np.float32) * cov[..., None]).astype(np.uint8)
    if glow > 0:
        g = np.zeros(fr.shape[:2], np.float32)
        cv2.circle(g, (int(X), int(Y)), int(max(2, R * 2.5)), 1.0, -1, cv2.LINE_AA)
        g = cv2.GaussianBlur(g, (0, 0), max(1.0, R * 2))
        fr = np.clip(fr.astype(np.float32) + g[..., None] * 90 * glow, 0, 255).astype(np.uint8)
    return fr
