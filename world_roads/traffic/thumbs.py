"""Three thumbnails for A/B/C testing (1280x720):  python thumbs.py -> out/thumb_A.png, thumb_B.png, thumb_C.png"""
import os
import numpy as np
from PIL import Image, ImageDraw

import look as L
import video as V
from look import INK, INK2, GOLD, text
import style as ST

N = V.N
OUT = L.OUT


def map_frame(view, width=7, glow_scale=1.0, dim=1.0):
    sc = V.Scene(view)
    sc.glow.clear()
    for i in range(3):
        sc.route(i, width, 1.0 * glow_scale)
    return sc, sc.compose(dim=dim)


def big_pill(d, x, y, i, s, size=64):
    L.pill(d, x, y, i, s, None, size=size, anchor="lm")


def thumb_a():
    """the map and three identical-looking ETAs: 3 ROUTES. 1 ETA."""
    view = L.MapView(1420, 520, 26.0, name="thumbA")
    sc, fr = map_frame(view, 8)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    sc.badges(d)
    text(d, (70, 90), "3 ROUTES.", 150, INK, 1.0, "Black" if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else "ExtraBold")
    text(d, (70, 260), "SAME ETA.", 150, GOLD, 1.0, "Black" if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else "ExtraBold")
    for i in range(3):
        big_pill(d, 80, 560 + i * 140, i, f"{N['eta_final'][i]} min", size=66)
    return im


def thumb_b():
    """not a coincidence: the three times meeting on the chart"""
    view = L.MapView(1500, 540, 24.0, name="thumbB")
    sc, fr = map_frame(view, 7, dim=0.7)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    ax = L.Axes((120, 380, 1000, 900), (0, N["d_rush"]), (30, 50))
    dg = np.linspace(0, N["d_rush"], 301)
    Tg = np.array([V.M.parallel_ue(V.T0, V.CAP, x)["times"] for x in dg])
    for i in range(3):
        j = N["joins"][i]
        ax.series(d, dg[dg <= j], Tg[dg <= j, i], i=i, width=5, a=0.5, dash=(6, 10))
        ax.series(d, dg[dg >= j], Tg[dg >= j, i] + (i - 1) * 0.15, i=i, width=10, dash=None)
        L.badge(d, float(ax.X(j)) - (0 if i == 0 else 0), float(ax.Y(N["t0"][i])) - 44, i, r=28)
    x, y = float(ax.X(N["d_rush"])), float(ax.Y(Tg[-1, 0]))
    lo, hi = min(N["eta_final"]), max(N["eta_final"])
    text(d, (x + 20, y + 4), f"all {lo}–{hi} min", 58, GOLD, 1.0, "ExtraBold", "lm")
    text(d, (100, 70), "NOT A", 130, INK, 1.0, "ExtraBold")
    text(d, (100, 210), "COINCIDENCE", 130, GOLD, 1.0, "ExtraBold")
    return im


def thumb_c():
    """3 a.m. vs rush hour"""
    view = L.MapView(1460, 540, 25.0, name="thumbC")
    sc, fr = map_frame(view, 8)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    sc.badges(d)
    text(d, (80, 80), "3 A.M.", 86, INK2, 1.0, "ExtraBold")
    for i in range(3):
        L.pill(d, 90, 230 + i * 100, i, f"{N['t0'][i]:.0f} min", None, size=50, anchor="lm")
    text(d, (80, 560), "RUSH HOUR", 86, GOLD, 1.0, "ExtraBold")
    for i in range(3):
        L.pill(d, 90, 710 + i * 100, i, f"{N['eta_final'][i]} min", None, size=50, anchor="lm")
    return im


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c)):
        f().convert("RGB").resize((1280, 720), Image.LANCZOS).save(os.path.join(OUT, f"thumb_{k}.png"), optimize=True)
        print("wrote", f"thumb_{k}.png")
