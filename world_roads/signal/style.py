"""Shared look for the long signal-fire video (1920x1080): fonts, colours, captions, title cards."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FD = "/usr/share/fonts/opentype/inter"
INK, INK2, MUTED = (240, 244, 250), (170, 186, 206), (120, 136, 158)
GOLD, RED = (255, 186, 80), (255, 110, 90)
PANEL = (6, 10, 18)
_F = {}

# Pillow 12 draws text at full opacity whatever the fill's alpha (shapes blend correctly), so every text fade
# would pop on and off. Text with a translucent fill is drawn into a mask scaled by its alpha instead.
_TEXT = ImageDraw.ImageDraw.text


def _text(self, xy, text, fill=None, font=None, anchor=None, *args, **kw):
    if args or not (isinstance(fill, tuple) and len(fill) == 4 and fill[3] < 255) or kw.get("stroke_width"):
        return _TEXT(self, xy, text, fill, font, anchor, *args, **kw)
    a = fill[3]
    if a <= 0 or not text:
        return
    l, t, r, b = self.textbbox(xy, text, font=font, anchor=anchor, **kw)
    l, t, r, b = int(l) - 2, int(t) - 2, int(r) + 3, int(b) + 3
    mask = Image.new("L", (r - l, b - t), 0)
    _TEXT(ImageDraw.Draw(mask), (xy[0] - l, xy[1] - t), text, 255, font, anchor, **kw)
    mask = mask.point(lambda v: v * a // 255)
    self._image.paste(tuple(fill[:3]), (l, t, r, b), mask)


ImageDraw.ImageDraw.text = _text


def font(size, weight="SemiBold"):
    k = (size, weight)
    if k not in _F:
        _F[k] = ImageFont.truetype(os.path.join(FD, f"Inter-{weight}.otf"), size, layout_engine=ImageFont.Layout.RAQM)
    return _F[k]


def fade(t, t0, t1, fin=0.35, fout=0.35):
    """0..1 envelope for something shown from t0 to t1."""
    if t < t0 or t > t1:
        return 0.0
    return float(min(1.0, (t - t0) / fin if fin else 1.0, (t1 - t) / fout if fout else 1.0))


def caption(d, lines, a, y=None, x=None, W=1920, H=1080, anchor="bottom", width=None, cx=None):
    """A rounded panel of text lines. lines: [(text, size, colour, weight)]; a: opacity 0..1; cx: centre x."""
    if a <= 0:
        return
    pad, gap = 30, 14
    fs = [font(sz, wt) for _, sz, _, wt in lines]
    ws = [d.textlength(tx, font=f) for (tx, *_), f in zip(lines, fs)]
    hs = [int(sz * 1.18) for _, sz, _, _ in lines]
    bw = int(max(ws) + 2 * pad) if width is None else width
    bh = int(sum(hs) + gap * (len(lines) - 1) + 2 * pad)
    if x is None:
        x = (W - bw) // 2 if cx is None else int(cx - bw / 2)
    if y is None:
        y = H - bh - 70 if anchor == "bottom" else 70
    d.rounded_rectangle([x, y, x + bw, y + bh], 22, fill=PANEL + (int(205 * a),))
    yy = y + pad
    for (tx, sz, col, wt), f, w, h in zip(lines, fs, ws, hs):
        d.text((x + bw / 2, yy), tx, font=f, fill=tuple(col) + (int(255 * a),), anchor="ma")
        yy += h + gap


def draw_captions(frame, t, caps, W=1920, H=1080):
    """frame: uint8 HxWx3. caps: [(t0, t1, lines, kwargs)]. Returns the composited frame."""
    act = [(c, fade(t, c[0], c[1])) for c in caps]
    act = [(c, a) for c, a in act if a > 0]
    if not act:
        return frame
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    for c, a in act:
        caption(d, c[2], a, W=W, H=H, **(c[3] if len(c) > 3 else {}))
    return np.asarray(im)


def ring_label(frame, x, y, text, a, r=26, side=1):
    """A thin ring around a point of interest with a short label."""
    if a <= 0:
        return frame
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    col = GOLD + (int(230 * a),)
    d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)
    f = font(30)
    lx = x + side * (r + 18)
    d.line([(x + side * r * 0.72, y - r * 0.72), (lx, y - r - 22), (lx + side * 16, y - r - 22)], fill=col, width=3)
    tw = d.textlength(text, font=f)
    tx = lx + side * 24 if side > 0 else lx - 24 - tw
    d.rounded_rectangle([tx - 12, y - r - 46, tx + tw + 12, y - r + 2], 12, fill=PANEL + (int(190 * a),))
    d.text((tx, y - r - 22), text, font=f, fill=INK + (int(255 * a),), anchor="lm")
    return np.asarray(im)
