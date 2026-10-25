"""Thumbnails for A/B/C testing (1280x720):  python thumbs.py [A B C D] -> out/thumb_<k>.png

D is built around a real maps-app screenshot: put it at media/traffic/maps_screenshot.png (a phone screenshot of
two or more routes showing the same ETA). maps_screenshot.json next to it can crop it, ring the ETAs and set the
words, in the screenshot's own pixels: {"crop": [x0, y0, x1, y1], "rings": [[x, y, rx, ry], ...], "source": "...",
"trip": "...", "words": [...]}. The screenshot is only scaled, never edited. Without a screenshot, D is a
layout preview with an empty placeholder phone (out/thumb_D_layout.png).
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

import look as L
import video as V
from look import INK, INK2, GOLD, text
import style as ST

N = V.N
OUT = L.OUT


USED = V.USED                     # the routes in use at rush hour: the ones with the same ETA


def map_frame(view, width=7, glow_scale=1.0, dim=1.0, which=(0, 1, 2)):
    sc = V.Scene(view)
    sc.glow.clear()
    for i in which:
        sc.route(i, width, 1.0 * glow_scale)
    return sc, sc.compose(dim=dim)


def big_pill(d, x, y, i, s, size=64):
    L.pill(d, x, y, i, s, None, size=size, anchor="lm")


def thumb_a():
    """the map and two identical ETAs home: 2 ROUTES HOME. SAME ETA."""
    view = L.MapView.fit((900, 60, 1890, 1020), "thumbA", which=tuple(USED))
    sc, fr = map_frame(view, 8, which=USED)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    sc.badges(d, which=USED)
    heavy = "Black" if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else "ExtraBold"
    text(d, (70, 70), "2 ROUTES", 150, INK, 1.0, heavy)
    text(d, (70, 230), "HOME.", 150, INK, 1.0, heavy)
    text(d, (70, 390), "SAME ETA.", 150, GOLD, 1.0, heavy)
    for k, i in enumerate(USED):
        big_pill(d, 80, 690 + k * 140, i, f"{N['eta_final'][i]} min", size=66)
    return im


def thumb_b():
    """not a coincidence: the two times meeting on the chart"""
    view = L.MapView.fit((1040, 60, 1890, 1020), "thumbB", which=tuple(USED))
    sc, fr = map_frame(view, 7, dim=0.7, which=USED)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    lo_t = 5 * np.floor((min(N["t0"]) - 2) / 5)
    ax = L.Axes((120, 380, 1000, 900), (0, N["d_rush"]), (lo_t, 5 * np.ceil((N["ue"]["level"] + 4) / 5)))
    dg = np.linspace(0, N["d_rush"], 301)
    Tg = np.array([V.M.parallel_ue(V.T0, V.CAP, x)["times"] for x in dg])
    for i in USED:
        j = N["joins"][i]
        ax.series(d, dg[dg <= j], Tg[dg <= j, i], i=i, width=5, a=0.5, dash=(6, 10))
        ax.series(d, dg[dg >= j], Tg[dg >= j, i] + (i - 1) * 0.15, i=i, width=10, dash=None)
        L.badge(d, float(ax.X(j)) - (0 if i == 0 else 0), float(ax.Y(N["t0"][i])) - 44, i, r=28)
    x, y = float(ax.X(N["d_rush"])), float(ax.Y(Tg[-1, 0]))
    eta = [N["eta_final"][i] for i in USED]
    lo, hi = min(eta), max(eta)
    both = "both" if len(USED) == 2 else "all"
    text(d, (x, y + 150), f"{both}: {lo} min" if lo == hi else f"{both} {lo}–{hi} min", 64, GOLD, 1.0, "ExtraBold", "rm")
    text(d, (100, 70), "NOT A", 130, INK, 1.0, "ExtraBold")
    text(d, (100, 210), "COINCIDENCE", 130, GOLD, 1.0, "ExtraBold")
    return im


def thumb_c():
    """3 a.m. vs the evening rush, for the two routes home"""
    view = L.MapView.fit((820, 60, 1890, 1020), "thumbC", which=tuple(USED))
    sc, fr = map_frame(view, 8, which=USED)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    sc.badges(d, which=USED)
    text(d, (80, 90), "3 A.M.", 96, INK2, 1.0, "ExtraBold")
    for k, i in enumerate(USED):
        L.pill(d, 90, 260 + k * 110, i, f"{N['t0'][i]:.0f} min", None, size=56, anchor="lm")
    text(d, (80, 560), "RUSH HOUR", 96, GOLD, 1.0, "ExtraBold")
    for k, i in enumerate(USED):
        L.pill(d, 90, 730 + k * 110, i, f"{N['eta_final'][i]} min", None, size=56, anchor="lm")
    return im


SHOT = L.SHOT                     # MAPS_SHOT=... to try another screenshot


def placeholder(h=930):
    """an empty phone for the layout preview: no imitation of anyone's app"""
    w = int(h * 0.462)
    im = Image.new("RGB", (w, h), (28, 32, 42))
    d = ImageDraw.Draw(im, "RGBA")
    for k, s in enumerate(["YOUR MAPS APP", "SCREENSHOT HERE", "", "two routes,", "the same ETA", "", "maps_screenshot.png"]):
        text(d, (w / 2, h * 0.36 + k * 46), s, 34 if k < 2 else 28, INK if k < 2 else INK2, 1.0, "Bold" if k < 2 else "Medium", "mm")
    return im


def thumb_d():
    """a real maps-app screenshot with near-equal ETAs, beside the hook"""
    view = L.MapView.fit((60, 60, 1100, 1040), "thumbD", which=tuple(USED))
    sc, fr = map_frame(view, 7, dim=0.55, which=USED)
    im = Image.fromarray(fr).convert("RGBA")
    have = os.path.exists(SHOT)
    shot = Image.open(SHOT) if have else placeholder()
    marks = L.shot_marks(SHOT) if have else {}
    tall = (marks.get("crop", (0, 0, shot.width, shot.height))[3] - marks.get("crop", (0, 0))[1]) >= \
           (marks.get("crop", (0, 0, shot.width, shot.height))[2] - marks.get("crop", (0, 0))[0])
    card, _, _ = L.shot_card(shot, (720, 930) if tall else (1000, 760), marks, tilt=-3.0, rad=52)
    cx = 1460 if tall else 1370
    im.alpha_composite(card, (int(cx - card.width / 2), int(520 - card.height / 2)))
    d = ImageDraw.Draw(im, "RGBA")
    if marks.get("source"):                          # say whose screenshot it is
        text(d, (cx, 520 + card.height / 2 - 34), f"{marks['source']} · {marks.get('trip', '')}", 28, INK, 1.0, "SemiBold", "mm")
    heavy = "Black" if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else "ExtraBold"
    words = marks.get("words", ["2 ROUTES.", "SAME ETA.", "WHY?"])
    size = 160 if tall else 140
    for k, s in enumerate(words):
        col = GOLD if k == 1 else INK
        x, y = 70, 120 + k * int(size * 1.25)
        for dx, dy in ((0, 6), (0, 3)):                # a drop shadow so the words read over the map
            text(d, (x + dx, y + dy), s, size, (0, 0, 0), 0.55, heavy)
        text(d, (x, y), s, size, col, 1.0, heavy)
    if not have:
        text(d, (70, 1010), "LAYOUT PREVIEW: add media/traffic/maps_screenshot.png, then run thumbs.py D", 30, (255, 120, 110), 1.0, "Bold")
    return im.convert("RGB"), have

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:] or ["A", "B", "C", "D"]
    for k, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c)):
        if k in want:
            f().convert("RGB").resize((1280, 720), Image.LANCZOS).save(os.path.join(OUT, f"thumb_{k}.png"), optimize=True)
            print("wrote", f"thumb_{k}.png")
    if "D" in want:
        im, have = thumb_d()
        name = "thumb_D.png" if have else "thumb_D_layout.png"
        im.resize((1280, 720), Image.LANCZOS).save(os.path.join(OUT, name), optimize=True)
        print("wrote", name, "" if have else "(no screenshot yet: a layout preview)")
