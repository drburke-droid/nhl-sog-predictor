"""Thumbnails for A/B/C testing (1280x720):  python thumbs.py [A B C D] -> out/thumb_<k>.png

D is built around a real Google Maps screenshot: put it at media/traffic/gmaps_screenshot.png (a phone screenshot
of three routes showing the same ETA). An optional gmaps_screenshot.json next to it can crop it and ring the ETAs,
in the screenshot's own pixels: {"crop": [x0, y0, x1, y1], "circles": [[x, y, r], ...]}. The screenshot is placed
as it is (never edited), so keep Google's logo and map-data attribution in the crop. Without a screenshot, D is a
layout preview with an empty placeholder phone (out/thumb_D_layout.png).
"""
import os, sys, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

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
    view = L.MapView.fit((900, 40, 1890, 1040), "thumbA")
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
    view = L.MapView.fit((1040, 60, 1890, 1020), "thumbB")
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
    text(d, (x, y + 80), f"all three: {lo} min" if lo == hi else f"all {lo}–{hi} min", 58, GOLD, 1.0, "ExtraBold", "rm")
    text(d, (100, 70), "NOT A", 130, INK, 1.0, "ExtraBold")
    text(d, (100, 210), "COINCIDENCE", 130, GOLD, 1.0, "ExtraBold")
    return im


def thumb_c():
    """3 a.m. vs rush hour"""
    view = L.MapView.fit((800, 40, 1890, 1040), "thumbC")
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


SHOT = os.environ.get("GMAPS_SHOT", os.path.join(L.MEDIA, "gmaps_screenshot.png"))      # GMAPS_SHOT=... to try another


def phone(shot, box=(720, 930), marks=None, tilt=-3.0):
    """the screenshot, fitted into box, in a phone-like frame (rounded, dark bezel, soft shadow), rings on the
    ETAs, a slight tilt"""
    marks = marks or {}
    if "crop" in marks:
        shot = shot.crop(tuple(marks["crop"]))
    s = min(box[0] / shot.width, box[1] / shot.height)
    w, h = int(round(shot.width * s)), int(round(shot.height * s))
    scr = shot.convert("RGB").resize((w, h), Image.LANCZOS)
    dd = ImageDraw.Draw(scr, "RGBA")
    ox, oy = marks.get("crop", (0, 0))[:2]
    for x, y, r in marks.get("circles", []):
        cx, cy, rr = (x - ox) * s, (y - oy) * s, r * s
        for wdt, al in ((16, 70), (9, 255)):
            dd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=GOLD + (al,), width=wdt)
    bz, rad = 16, 52
    W, H = w + 2 * bz, h + 2 * bz
    body = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    db = ImageDraw.Draw(body)
    db.rounded_rectangle([0, 0, W - 1, H - 1], rad + bz, fill=(18, 20, 26, 255), outline=(120, 128, 142, 255), width=3)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], rad, fill=255)
    body.paste(scr, (bz, bz), mask)
    pad = 60
    lay = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("L", lay.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle([pad + 10, pad + 24, pad + W + 10, pad + H + 24], rad + bz, fill=200)
    sh = sh.filter(ImageFilter.GaussianBlur(24))
    lay.paste(Image.new("RGBA", lay.size, (0, 0, 0, 255)), (0, 0), sh)
    lay.alpha_composite(body, (pad, pad))
    return lay.rotate(tilt, resample=Image.BICUBIC, expand=True)


def placeholder(h=930):
    """an empty phone for the layout preview: no imitation of anyone's app"""
    w = int(h * 0.462)
    im = Image.new("RGB", (w, h), (28, 32, 42))
    d = ImageDraw.Draw(im, "RGBA")
    for k, s in enumerate(["YOUR GOOGLE MAPS", "SCREENSHOT HERE", "", "three routes,", "the same ETA", "", "gmaps_screenshot.png"]):
        text(d, (w / 2, h * 0.36 + k * 46), s, 34 if k < 2 else 28, INK if k < 2 else INK2, 1.0, "Bold" if k < 2 else "Medium", "mm")
    return im


def thumb_d():
    """a real Google Maps screenshot with three equal ETAs, beside the hook"""
    view = L.MapView.fit((60, 60, 1100, 1040), "thumbD")
    sc, fr = map_frame(view, 7, dim=0.55)
    im = Image.fromarray(fr).convert("RGBA")
    have = os.path.exists(SHOT)
    shot = Image.open(SHOT) if have else placeholder()
    mp = os.path.splitext(SHOT)[0] + ".json"
    marks = json.load(open(mp)) if have and os.path.exists(mp) else None
    ph = phone(shot, (720, 930) if shot.height >= shot.width else (860, 800), marks=marks)
    im.alpha_composite(ph, (int(1460 - ph.width / 2), int(540 - ph.height / 2)))
    d = ImageDraw.Draw(im, "RGBA")
    heavy = "Black" if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else "ExtraBold"
    for k, (s, col) in enumerate((("3 ROUTES.", INK), ("SAME ETA.", GOLD), ("WHY?", INK))):
        x, y = 70, 120 + k * 200
        for dx, dy in ((0, 6), (0, 3)):                # a drop shadow so the words read over the map
            text(d, (x + dx, y + dy), s, 160, (0, 0, 0), 0.55, heavy)
        text(d, (x, y), s, 160, col, 1.0, heavy)
    if not have:
        text(d, (70, 1010), "LAYOUT PREVIEW: add media/traffic/gmaps_screenshot.png, then run thumbs.py D", 30, (255, 120, 110), 1.0, "Bold")
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
