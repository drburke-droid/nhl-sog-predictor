"""Thumbnails (1280x720) for the A/B/C test.

    python thumbs.py   -> out/rivers_thumb_A.png, _B.png, _C.png
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video as V
import style as ST

OUT = V.OUT


def big(d, xy, s, size, col, anchor="la", weight="Black", stroke=0):
    d.text(xy, s, font=ST.font(size, weight if os.path.exists(os.path.join(ST.FD, f"Inter-{weight}.otf")) else "Bold"),
           fill=col, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))


def globe_rivers(lon, lat, rpx, cx, gold=(), dim=0.9):
    v = V.GlobeView(lon, lat, rpx, cx=cx)
    img = v.base()
    gl = V.Glow()
    meta, _ = V.rivers()
    for k in range(len(meta)):
        xy, z = v.project(V.river_xy(k))
        for run in V.visible_runs(xy, z):
            gl.poly(gl.C if k in gold else gl.A, run, 1.8 if k in gold else dim, 3 if k in gold else 2)
    return gl.compose(img, exposure=1.35)


def thumb_a():
    """the night globe, every river lit, the Brahmaputra in gold: IS EVERY RIVER pi?"""
    meta, _ = V.rivers()
    k = next(i for i, m in enumerate(meta) if m["display"] == "Brahmaputra")
    im = Image.fromarray(globe_rivers(84.0, 24.0, 560, 1300, gold=(k,)))
    d = ImageDraw.Draw(im, "RGBA")
    big(d, (90, 300), "DO RIVERS", 120, V.INK)
    big(d, (90, 440), "WIND BY", 120, V.INK)
    big(d, (90, 560), "π ?", 200, V.GOLD)
    return im


def thumb_b():
    """a meandering river measured: LENGTH ÷ STRAIGHT LINE = pi?"""
    o = V.Open()
    im = Image.fromarray(o.frame(8.0))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 740, V.W, V.H], fill=V.BG + (255,))
    big(d, (V.W / 2, 860), "LENGTH ÷ STRAIGHT LINE = π ?", 92, V.INK, anchor="mm")
    big(d, (V.W / 2, 980), "1,000 RIVERS MEASURED", 60, V.GOLD, anchor="mm", weight="Bold")
    return im


def thumb_c():
    """the Brahmaputra close up: 3.16, of the 1,000 the river nearest pi"""
    meta, Z = V.rivers()
    k = next(i for i, m in enumerate(meta) if m["display"] == "Brahmaputra")
    lon, lat, rpx = V.fit_view(k, frac=0.7, cx=V.W / 2 + 260)
    v = V.GlobeView(lon, lat, rpx, cx=V.W / 2 + 260)
    img = v.base()
    gl = V.Glow()
    xy, z = v.project(V.river_xy(k, full=True))
    for run in V.visible_runs(xy, z):
        gl.poly(gl.C, run, 2.2, 5)
    cxy, cz = v.project(Z["c_xy"][k])
    gl.poly(gl.B, cxy, 1.4, 3)
    im = Image.fromarray(gl.compose(img, exposure=1.3))
    d = ImageDraw.Draw(im, "RGBA")
    big(d, (80, 120), "3.16", 260, V.GOLD)
    big(d, (90, 420), "THE RIVER", 64, V.INK, weight="Bold")
    big(d, (90, 500), "NEAREST TO π", 64, V.INK, weight="Bold")
    big(d, (90, 590), "OF 1,000", 64, V.INK2, weight="Bold")
    return im


if __name__ == "__main__":
    for nm, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c)):
        im = f().resize((1280, 720), Image.LANCZOS)
        p = os.path.join(OUT, f"rivers_thumb_{nm}.png")
        im.save(p)
        print("wrote", p)
