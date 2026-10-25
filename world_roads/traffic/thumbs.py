"""Thumbnails for A/B/C testing (1280x720):  python thumbs.py [A B C D] -> out/thumb_<k>.png

D is built around a real maps-app screenshot: put it at media/traffic/maps_screenshot.png (a phone screenshot of
two or more routes showing the same ETA). maps_screenshot.json next to it can crop it, ring the ETAs and set the
words, in the screenshot's own pixels: {"crop": [x0, y0, x1, y1], "rings": [[x, y, rx, ry], ...], "source": "...",
"trip": "...", "words": [...]}. The screenshot is only scaled, never edited. Without a screenshot, D is a
layout preview with an empty placeholder phone (out/thumb_D_layout.png).
"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

import look as L
import video as V
from look import INK, INK2, GOLD, text, lerp
from video import GOLDISH
import style as ST

N = V.N
OUT = L.OUT


USED = V.USED                     # the routes in use at rush hour: the ones with the same ETA
W_, H_ = 1920, 1080


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

# ------------------------------------------------------------------ the bottleneck, the tunnel, the real times
RED = (255, 60, 48)


def bloom(img, em, small=10, big=34, k=(1.0, 0.8, 0.55)):
    """add a glowing layer: the layer itself, a tight blur and a wide blur"""
    e = np.asarray(em, np.float32)
    out = np.asarray(img, np.float32) + k[0] * e + k[1] * cv2.GaussianBlur(e, (0, 0), small) + k[2] * cv2.GaussianBlur(e, (0, 0), big)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def heavy_text(d, xy, s, size, col, anchor="la"):
    w = "Black" if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else "ExtraBold"
    for dy in (7, 3):                                   # a drop shadow so the words read over anything
        text(d, (xy[0], xy[1] + dy), s, size, (0, 0, 0), 0.6, w, anchor)
    text(d, xy, s, size, col, 1.0, w, anchor)


def thumb_e():
    """the bottleneck: four lanes squeeze into two and a long red queue backs up behind it"""
    class Long(V.Neck):                                 # the video's toy queue, on a longer road, left to fill up
        X0, XM, XT, X1 = -300, 520, 610, 960
    nk = Long(6.0, 31.0, 7.0)
    cars = nk.frames[-1]
    s, cy = 2000 / (Long.X1 - Long.X0), 700
    X = lambda x: (x - Long.X0) * s - 40
    Y = lambda y: cy + (y - Long.CY) * s
    view = L.MapView.fit((200, 100, 1800, 980), "thumbE", which=tuple(USED))
    base = V.Scene(view).base * 0.32
    img = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    xs = [Long.X0 - 60, Long.XM, Long.XT, Long.X1 + 60]
    top = [(X(x), Y(Long.CY - nk.half(x))) for x in xs]
    bot = [(X(x), Y(Long.CY + nk.half(x))) for x in xs[::-1]]
    d.polygon(top + bot, fill=(30, 36, 50, 255))
    em = Image.new("RGB", img.size, (0, 0, 0))
    de = ImageDraw.Draw(em)
    for P in (top, [(X(x), Y(Long.CY + nk.half(x))) for x in xs]):
        de.line(P, fill=(150, 200, 255), width=5)
    for off in (-1, 1):                                  # the lanes that end
        L.dashed(de, [(X(xs[0]), Y(Long.CY + off * Long.LW)), (X(Long.XM), Y(Long.CY + off * Long.LW))], (110, 140, 180), 4, (26, 20), 1.0)
    L.dashed(de, [(X(xs[0]), Y(Long.CY)), (X(xs[-1]), Y(Long.CY))], (110, 140, 180), 4, (26, 20), 1.0)
    cw, ch = Long.CW * s, Long.CH * s
    for x, y, stop, queued in cars:
        jam = min(1.0, stop * 4.0) if queued else 0.0      # anything crawling in the queue shows its brake lights
        col = tuple(int(lerp(c0, c1, jam)) for c0, c1 in zip((235, 242, 255), RED))
        de.rounded_rectangle([X(x) - cw / 2, Y(y) - ch / 2, X(x) + cw / 2, Y(y) + ch / 2], 6, fill=col)
    img = bloom(img, em, k=(1.0, 0.7, 0.45))
    d = ImageDraw.Draw(img, "RGBA")
    bx = (X(Long.XM) + X(Long.XT)) / 2
    yb = Y(Long.CY + 2 * Long.LW) + 26
    d.polygon([(bx, yb), (bx - 22, yb + 36), (bx + 22, yb + 36)], fill=GOLD + (255,))
    heavy_text(d, (bx, yb + 52), "BOTTLENECK", 54, GOLD, "ma")
    text(d, (X(-60), Y(Long.CY - 2 * Long.LW) - 30), "4 LANES", 44, INK, 1.0, "Bold", "ms")
    text(d, ((X(Long.XT) + X(Long.X1)) / 2, Y(Long.CY - Long.LW) - 30), "2 LANES", 44, INK, 1.0, "Bold", "ms")
    heavy_text(d, (70, 70), "SAME ETA?", 170, INK)
    heavy_text(d, (78, 270), "THIS IS WHY.", 120, GOLD)
    return img


def thumb_f():
    """Braess on the map: the mythical tunnel that makes everyone slower"""
    keep = V.RIGHT
    V.RIGHT = L.MapView.fit((860, 70, 1880, 1010), "thumbF", which=tuple(USED))
    try:
        m = V.Myth()
    finally:
        V.RIGHT = keep
    sc, g = m.sc, m.sc.glow
    g.clear()
    sc.route(m.X, 9, 1.0)
    sc.route(m.Y, 9, 1.0)
    for P in m.bn:
        g.line(P, (255, 246, 230), 18, 1.0)
    for seg in m.dash:
        g.line(seg, GOLDISH, 14, 1.0)
    img = Image.fromarray(sc.compose(dim=0.75))
    d = ImageDraw.Draw(img, "RGBA")
    sc.badges(d, which=(m.X, m.Y))
    mx, my = m.bore[len(m.bore) // 2]
    lab = "MYTHICAL TUNNEL"
    tw = d.textlength(lab, font=ST.font(34, "Bold"))
    d.rounded_rectangle([mx - tw / 2 - 18, my - 30, mx + tw / 2 + 18, my + 30], 18, fill=ST.PANEL + (235,), outline=GOLDISH + (255,), width=3)
    text(d, (mx, my), lab, 34, GOLDISH, 1.0, "Bold", "mm")
    heavy_text(d, (70, 70), "NEW ROAD.", 150, INK)
    heavy_text(d, (70, 250), "EVERYONE", 150, INK)
    heavy_text(d, (70, 430), "SLOWER.", 150, RED)
    gain = round(m.m["opened"]["avg"] - m.m["closed"]["avg"])
    d.rounded_rectangle([80, 680, 520, 860], 40, fill=(120, 20, 24, 235), outline=RED + (255,), width=5)
    text(d, (300, 770), f"+{gain} MIN", 104, INK, 1.0, "Black" if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else "ExtraBold", "mm")
    return img


def thumb_g():
    """the real screenshot's three times, as big as they go: 59, 60, 61"""
    marks = L.shot_marks()
    shot = Image.open(SHOT).convert("RGB").crop(tuple(marks["crop"]))
    s = max(W_ / shot.width, H_ / shot.height)
    bg = shot.resize((int(shot.width * s), int(shot.height * s)), Image.LANCZOS)
    bg = bg.crop(((bg.width - W_) // 2, (bg.height - H_) // 2, (bg.width - W_) // 2 + W_, (bg.height - H_) // 2 + H_))
    a = cv2.GaussianBlur(np.asarray(bg, np.float32), (0, 0), 7) * 0.3
    yy, xx = np.mgrid[0:H_, 0:W_].astype(np.float32)
    a *= np.clip(1.25 - 0.9 * (((xx - W_ / 2) / W_) ** 2 + ((yy - H_ / 2) / H_) ** 2) * 2.2, 0.35, 1)[..., None]
    img = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    heavy_text(d, (W_ / 2, 60), "3 WAYS HOME", 120, INK, "ma")
    nums = ["59", "60", "61"]
    for k, n in enumerate(nums):
        cx = W_ / 2 + (k - 1) * 560
        heavy_text(d, (cx, 560), n, 330, GOLD, "mm")
        text(d, (cx, 760), "MIN", 64, INK, 1.0, "Bold", "mm")
    heavy_text(d, (W_ / 2, 870), "COINCIDENCE?", 120, INK, "ma")
    text(d, (W_ - 40, H_ - 30), f"{marks.get('source', '')} · {marks.get('trip', '')}", 26, INK2, 1.0, "SemiBold", "rs")
    return img


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:] or ["A", "B", "C", "D", "E", "F", "G"]
    for k, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c), ("E", thumb_e), ("F", thumb_f), ("G", thumb_g)):
        if k in want:
            f().convert("RGB").resize((1280, 720), Image.LANCZOS).save(os.path.join(OUT, f"thumb_{k}.png"), optimize=True)
            print("wrote", f"thumb_{k}.png")
    if "D" in want:
        im, have = thumb_d()
        name = "thumb_D.png" if have else "thumb_D_layout.png"
        im.resize((1280, 720), Image.LANCZOS).save(os.path.join(OUT, name), optimize=True)
        print("wrote", name, "" if have else "(no screenshot yet: a layout preview)")
