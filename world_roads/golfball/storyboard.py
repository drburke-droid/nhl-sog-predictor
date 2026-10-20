"""Storyboard stills for the restyled golf-ball film (1920×1080), one per beat.

    python storyboard.py [names...]      -> out/storyboard/NN_name.png
"""
import os, sys, math
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import look2 as S2
import course as C
import physics as P
from look2 import W, H, text, caption, headline, marker, ring, lower_third, TRACER, GOLD, WHITE, CREAM, MUTED, INK, PANEL

OUT = os.path.join(HERE, "out", "storyboard")
SHOT = C.Shot(cx=1240.0)                                # the fairway's axis right of centre: room for words


def yds_at(t, carry=250.0):
    x, y, z = C.flight(t, carry)
    return float(z) / P.YD


def t_for_yds(yd, carry=250.0):
    ts = np.linspace(0, 1, 4001)
    return float(np.interp(yd * P.YD, C.flight(ts, carry)[2], ts))


def ball_line(yd):
    return f"BALL: {float(P.ball_arcmin(yd)):.2f} ARCMINUTE WIDE".replace("1.00 ARCMINUTE", "1 ARCMINUTE")


def drive(t1, seed=1, ball=True):
    """the course with the tracer drawn to flight fraction t1, and the ball at its head"""
    fr = C.course(SHOT, seed=seed)
    fr, end = C.tracer(fr, SHOT, 0.0, t1)
    if ball and end is not None:
        fr = S2.ball(fr, end[0], end[1], 5, glow=0.6)
    return fr, end


def bug(im, s, a=1.0):
    """the small broadcast label, top left"""
    d = ImageDraw.Draw(im, "RGBA")
    w_ = S2.text_width(s, 24, "cond-SemiBold") + 3 * len(s) + 36
    d.rectangle([60, 56, 60 + w_, 96], fill=PANEL + (200,))
    d.rectangle([60, 56, 66, 96], fill=TRACER + (255,))
    text(im, (80, 76), s, 24, CREAM, "cond-SemiBold", "lm", a, track=3)


# ================================================================== the cold open
def f01_drive():
    t = t_for_yds(96)
    fr, end = drive(t)
    im = Image.fromarray(S2.finish(fr))
    bug(im, "TRACER VIEW · BALL ENLARGED")
    lower_third(im, "Distance", "96 YDS", y=980, sub=ball_line(96))
    return im


def f02_160():
    t = t_for_yds(160)
    fr, end = drive(t)
    fr = marker(fr, ring(end[0], end[1], 34), TRACER, width=6, seed=3)
    im = Image.fromarray(S2.finish(fr))
    bug(im, "TRACER VIEW · BALL ENLARGED")
    lower_third(im, "Distance", "160 YDS", y=980, sub=ball_line(160))
    S2.block(im, ["160 yards:", "one arcminute"], ["Exactly as wide as the finest detail", "on a 20/20 eye chart."], y=260, size=88,
             accent=1)
    return im


def f03_land():
    t = t_for_yds(262)
    fr, end = drive(t)
    im = Image.fromarray(S2.finish(fr))
    bug(im, "TRACER VIEW · BALL ENLARGED")
    lower_third(im, "Distance", "262 YDS", y=980, sub=ball_line(262))
    S2.block(im, ["You shouldn’t be", "able to see this."], "Yet you watch it land.", y=260, size=92, accent=1)
    return im


def f04_title():
    fr, end = drive(0.999, ball=False)
    x, y = end
    fr = S2.ball(fr, x, y, 5, glow=0.6)
    f = fr.astype(np.float32) * 0.42
    fr = f.astype(np.uint8)
    im = Image.fromarray(S2.finish(fr, vig=0.35))
    text(im, (140, 330), "A VISION SCIENCE SHORT", 30, GOLD, "cond-SemiBold", "ls", track=6)
    headline(im, ["The 160-yard", "problem"], 136, 470, size=150, accent=1)
    caption(im, ["Why you can still see a golf ball", "your eyes can’t resolve."], x=142, y=740, size=46)
    return im


# ================================================================== the numbers: a yardage book page
def paper(col=S2.PAPER, seed=4, amt=0.02):
    fr = np.empty((H, W, 3), np.float32)
    fr[:] = col
    n = C.noise(seed, 1.2) * 0.7 + C.noise(seed + 1, 30) * 0.25
    fr *= (1 + amt * n)[..., None]
    return np.clip(fr, 0, 255).astype(np.uint8)


def ink_poly(fr, pts, col=S2.PAPER_INK, width=2.2, closed=False):
    m = np.zeros(fr.shape[:2], np.uint8)
    cv2.polylines(m, [np.round(np.asarray(pts) * 16).astype(np.int32)], closed, 255, max(1, int(round(width))), cv2.LINE_AA, shift=4)
    return S2.over(fr, m.astype(np.float32) / 255, col)


def fill_poly(fr, pts, col):
    return S2.over(fr, S2.poly_mask(fr.shape[:2], [np.asarray(pts)]), col)


def blob_pts(cx, cy, rx, ry, seed, n=60, rough=0.08):
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    k = 1 + rough * (np.sin(th * 3 + rng.uniform(0, 6)) + 0.6 * np.sin(th * 5 + rng.uniform(0, 6)))
    return np.c_[cx + rx * k * np.cos(th), cy + ry * k * np.sin(th)]


def f05_yardage():
    fr = paper()
    Yd = lambda yd: 940 - yd * 1.95                      # the hole runs up the page: 1.95 units per yard
    # rough, fairway, bunkers, green, tee
    rough = [(330, 980), (300, 700), (310, 380), (360, 120), (560, 70), (700, 130), (720, 420), (700, 760), (660, 980)]
    from scipy.interpolate import splprep, splev
    def smooth_closed(pts, n=200):
        p = np.array(pts + [pts[0]], float).T
        tck, _ = splprep(p, s=0, per=True)
        return np.column_stack(splev(np.linspace(0, 1, n), tck))
    fr = fill_poly(fr, smooth_closed(rough), (196, 208, 160))
    fair = [(450, 820), (430, 640), (420, 470), (450, 300), (500, 200), (560, 190), (600, 290), (610, 470), (590, 650), (560, 820)]
    fair_p = smooth_closed(fair)
    fr = fill_poly(fr, fair_p, (150, 182, 112))
    fr = ink_poly(fr, fair_p, width=2.0, closed=True)
    for cx, cy, rx, ry, sd in ((610, 410, 34, 22, 1), (410, 300, 26, 34, 2), (560, 150, 30, 16, 3)):
        b = blob_pts(cx, cy, rx, ry, sd)
        fr = fill_poly(fr, b, S2.SAND)
        fr = ink_poly(fr, b, width=1.8, closed=True)
    g = blob_pts(530, 125, 62, 48, 7, rough=0.05)
    fr = fill_poly(fr, g, (170, 204, 128))
    fr = ink_poly(fr, g, width=2.0, closed=True)
    fr = ink_poly(fr, [(528, 128), (528, 88)], width=2)
    fr = fill_poly(fr, [(528, 88), (552, 96), (528, 104)], S2.TRACER)
    fr = fill_poly(fr, [(486, 930), (534, 930), (534, 952), (486, 952)], (170, 204, 128))
    fr = ink_poly(fr, [(486, 930), (534, 930), (534, 952), (486, 952)], width=1.8, closed=True)
    # trees: stippled crowns along both sides
    rng = np.random.default_rng(5)
    for side in (-1, 1):
        for k in range(26):
            yy = rng.uniform(140, 960)
            xx = (330 + rng.uniform(-60, 10)) if side < 0 else (690 + rng.uniform(-10, 60))
            r = rng.uniform(13, 24)
            b = blob_pts(xx, yy, r, r, 100 + k * 2 + (side > 0), n=40, rough=0.12)
            fr = fill_poly(fr, b, (96, 128, 84))
            fr = ink_poly(fr, b, width=1.4, closed=True)
    # distance lines from the tee
    im = Image.fromarray(fr)
    for yd, col in ((100, S2.PAPER_INK), (160, S2.TRACER), (320, S2.TRACER)):
        y = Yd(yd)
        for x0 in range(300, 730, 16):
            ImageDraw.Draw(im).line([(x0, y), (x0 + 9, y)], fill=col, width=3)
        text(im, (790, y), f"{yd}", 34, col, "cond-Bold", "lm")
    fr = np.asarray(im).copy()
    fr = S2.finish(fr, grain=1.6, vig=0.18)
    im = Image.fromarray(fr)
    text(im, (300, 60), "HOLE 7  ·  PAR 4  ·  412 YDS", 30, S2.PAPER_INK, "cond-Bold", "ls", track=3)
    # the notes column
    x0 = 900
    text(im, (x0, 170), "THE BALL’S WIDTH, SEEN FROM THE TEE", 30, (110, 112, 100), "cond-SemiBold", "ls", track=3)
    text(im, (x0, 222), "1.68 in · 42.67 mm  ·  1′ = 1/60 of a degree", 30, S2.PAPER_INK, "serif", "ls")
    rows = [("100 YDS", "1.6′", "", S2.PAPER_INK), ("160 YDS", "1′", "the 20/20 limit", S2.TRACER),
            ("320 YDS", "0.5′", "one cone", S2.TRACER), ("640 YDS", "0.25′", "a quarter of a cone", S2.PAPER_INK)]
    y = 330
    d = ImageDraw.Draw(im)
    for lab, ang, note, col in rows:
        text(im, (x0, y), lab, 64, col, "cond-ExtraBold", "ls")
        text(im, (x0 + 250, y), ang, 64, col, "cond-ExtraBold", "ls")
        if note:
            text(im, (x0 + 420, y), note.upper(), 30, col, "cond-SemiBold", "ls", track=2)
        d.line([(x0, y + 22), (x0 + 860, y + 22)], fill=(200, 192, 176), width=2)
        y += 112
    caption(im, ["Twice as far, half the angle."], x=x0, y=860, size=40, col=S2.PAPER_INK, shadow=False)
    return im


# ================================================================== the eye chart
def e_glyph(fr, cx, cy, size, rot=0, col=(18, 18, 18)):
    """a tumbling E, size units tall (5×5 grid), opening right/down/left/up for rot 0..3"""
    s = size / 5
    rects = [(0, 0, 1, 5)] + [(0, k, 5, k + 1) for k in (0, 2, 4)]
    out = []
    for x0, y0, x1, y1 in rects:
        pts = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float) - 2.5
        for _ in range(rot):
            pts = np.c_[-pts[:, 1], pts[:, 0]]
        out.append(pts * s + [cx, cy])
    return S2.over(fr, S2.poly_mask(fr.shape[:2], out), col)


def f06_chart():
    fr = np.empty((H, W, 3), np.uint8)
    fr[:] = (176, 168, 152)
    fr = (fr.astype(np.float32) * (1 + 0.012 * C.noise(9, 40) + 0.02 * C.noise(10, 1.2))[..., None]).astype(np.uint8)
    # the chart card, with a soft shadow
    x0, y0, x1, y1 = 170, 70, 830, 1010
    sh = np.zeros((H, W), np.float32)
    cv2.rectangle(sh, (x0 + 14, y0 + 20), (x1 + 14, y1 + 20), 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 18)
    fr = (fr.astype(np.float32) * (1 - 0.35 * sh[..., None])).astype(np.uint8)
    cv2.rectangle(fr, (x0, y0), (x1, y1), (250, 249, 245), -1)
    rows = [("20/200", 1, 150), ("20/100", 2, 75), ("20/70", 3, 52), ("20/50", 4, 37.5), ("20/40", 5, 30),
            ("20/30", 6, 22.5), ("20/25", 7, 18.75), ("20/20", 8, 15), ("20/15", 8, 11.25)]
    rng = np.random.default_rng(2)
    y = 110
    pos20 = None
    im_rows = []
    for lab, n, size in rows:
        gap = size * 0.9
        wrow = n * size + (n - 1) * gap
        cy = y + size / 2
        for k in range(n):
            fr = e_glyph(fr, (x0 + x1) / 2 - wrow / 2 + size / 2 + k * (size + gap), cy, size, int(rng.integers(0, 4)))
        im_rows.append((lab, cy))
        if lab == "20/20":
            pos20 = (cy, wrow)
        y += size + max(26, size * 0.55)
    fr = marker(fr, ring((x0 + x1) / 2, pos20[0], pos20[1] / 2 + 34, a0=0.2) * [1, 0.22] + [0, pos20[0] * 0.78], TRACER, width=6, seed=5)
    # the magnifier: a 20/20 E and a golf ball at 160 yards, at the same scale
    mx, my, mr = 1330, 470, 330
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (mx, my), mr, 1.0, -1, cv2.LINE_AA)
    fr = S2.over(fr, m, (250, 249, 245))
    unit = 52.0                                                   # one arcminute in the magnifier
    fr = e_glyph(fr, mx - 70, my, 5 * unit, 0)
    fr = S2.ball(fr, mx + 230, my, unit / 2, col=(18, 18, 18))
    ring_m = np.zeros((H, W), np.uint8)
    cv2.circle(ring_m, (mx, my), mr, 255, 10, cv2.LINE_AA)
    fr = S2.over(fr, ring_m.astype(np.float32) / 255, S2.PANEL)
    # dimension marks: one stroke, and the ball
    fr = marker(fr, [(mx + 92, my - 130), (mx + 120, my - 130)], TRACER, width=4, wobble=0)
    fr = marker(fr, [(mx + 106, my - 130), (mx + 106, my - 78)], TRACER, width=4, wobble=0)
    fr = marker(fr, [(mx + 92, my - 78), (mx + 120, my - 78)], TRACER, width=4, wobble=0)
    fr = marker(fr, [(mx + 204, my + 54), (mx + 256, my + 54)], TRACER, width=4, wobble=0)
    fr = S2.finish(fr, grain=1.8)
    im = Image.fromarray(fr)
    for lab, cy in im_rows:
        text(im, (x0 + 26, cy), lab, 22, (120, 120, 120), "cond-SemiBold", "lm")
    text(im, (mx + 132, my - 104), "1′", 40, TRACER, "cond-Bold", "lm")
    text(im, (mx + 230, my + 66), "1′", 40, TRACER, "cond-Bold", "mt")
    text(im, (mx - 70, my + mr + 56), "20/20 E", 34, S2.PANEL, "cond-Bold", "mt", track=2)
    text(im, (mx + 230, my + mr + 56), "BALL AT 160 YDS", 34, S2.PANEL, "cond-Bold", "mt", track=2)
    S2.block(im, ["Same width"], None, x=1010, y=980, size=72, col=S2.PANEL, shadow=False)
    return im


# ================================================================== into the eye
def f07_eye():
    fr = np.empty((H, W, 3), np.uint8)
    fr[:] = S2.INK
    cx, cy, R = 1230, 520, 330
    def ell(fr, c, rx, ry, col, a0=0, a1=360, thick=-1):
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(rx), int(ry)), 0, a0, a1, 1.0, thick, cv2.LINE_AA)
        return S2.over(fr, m, col)
    sclera = (232, 226, 214)
    fr = ell(fr, (cx, cy), R + 12, R + 12, sclera)
    fr = ell(fr, (cx, cy), R, R, (38, 33, 38))                                    # the vitreous
    corn_c = (cx - R + 70, cy)
    fr = ell(fr, corn_c, 150, 150, (26, 30, 38), 140, 220)                        # the cornea's bulge
    fr = ell(fr, corn_c, 150, 150, sclera, 140, 220, 10)
    fr = ell(fr, (cx, cy), R - 6, R - 6, (176, 64, 48), -72, 72, 16)              # the retina
    for sg in (-1, 1):                                                           # the iris
        fr = S2.over(fr, S2.poly_mask((H, W), [np.array([(cx - R + 34, cy + sg * 44), (cx - R + 48, cy + sg * 44),
                                                         (cx - R + 48, cy + sg * 128), (cx - R + 34, cy + sg * 128)])]), (92, 120, 128))
    fr = ell(fr, (cx - R + 100, cy), 42, 96, (196, 214, 222))                     # the lens
    fovea = (cx + R - 12, cy)
    for sg in (-1, 1):
        fr = marker(fr, [(130, cy), (cx - R - 40, cy + sg * 46), (cx - R + 100, cy + sg * 40), fovea], TRACER, width=3.5,
                    wobble=0, seed=10 + sg)
    fr = S2.ball(fr, 130, cy, 9, glow=0.5)
    fr = marker(fr, ring(fovea[0], fovea[1], 40), GOLD, width=4, seed=7)
    fr = S2.finish(fr)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im)
    def callout(x, y, tx, ty, s, anchor):
        d.line([(x, y), (tx, ty)], fill=MUTED, width=2)
        text(im, (tx + (10 if anchor[0] == "l" else -10), ty), s, 30, CREAM, "cond-SemiBold", anchor, track=3)
    callout(cx - R - 72, cy + 110, cx - R - 150, cy + 300, "CORNEA", "rm")
    callout(cx - R + 100, cy + 90, cx - R + 60, cy + 380, "LENS", "rm")
    callout(cx + R - 70, cy - 250, cx + R + 40, cy - 390, "RETINA", "lm")
    callout(fovea[0] - 6, fovea[1] + 42, fovea[0] - 60, cy + 410, "FOVEA: WHERE YOU LOOK", "rm")
    S2.block(im, ["Into the eye"], ["The ball’s image lands on the fovea,", "where the cones are packed tightest."], x=120, y=200, size=84)
    return im


# ================================================================== on the retina
def retina_mod():
    import imaging as I, film as F, retina as R
    return I, F, R


def to_hd(fr4k):
    return cv2.resize(fr4k, (W, H), interpolation=cv2.INTER_AREA)


def tag(im, x, y, s, col=CREAM, size=30, anchor="mt"):
    text(im, (x, y), s, size, col, "cond-Bold", anchor, track=2, shadow=(0, 2, 0.5))


def f08_e_cones():
    I, F, R = retina_mod()
    sc = R.Scope(0.0, 0.0, 120.0)
    cx4, cy4 = 3840 * 0.6, 2160 * 0.5
    fr = I.mosaic(sc, lambda u, v: I.e_sharp_at(u, v), overlay=lambda u, v: I.e_sharp_at(u, v), band=0.3, curve="print",
                  centre=(cx4, cy4), radius=2160 * 0.44)
    fr = to_hd(fr)
    k = 60.0                                                     # 1080p px per cone spacing
    cx, cy = cx4 / 2, cy4 / 2
    fl, frt = cx - 2160 * 0.44 / 2, cx + 2160 * 0.44 / 2                 # the field's left and right edges
    fr = marker(fr, [(cx + 5.6 * k, cy - 5 * k), (cx + 5.6 * k, cy - 3 * k)], TRACER, width=5, wobble=0)
    fr = marker(fr, [(cx + 5.6 * k, cy - 4 * k), (frt + 30, cy - 4 * k)], TRACER, width=2.5, wobble=0)
    fr = marker(fr, [(cx - 5.6 * k, cy - 5 * k), (cx - 5.6 * k, cy + 5 * k)], TRACER, width=5, wobble=0)
    fr = marker(fr, [(cx - 5.6 * k, cy + 2.2 * k), (fl - 30, cy + 2.2 * k)], TRACER, width=2.5, wobble=0)
    rb = k * float(P.ball_arcmin(160)) / 2 / 0.5
    fr = marker(fr, ring(cx + 2.6 * k, cy, rb), GOLD, width=5, seed=4)
    fr = marker(fr, [(cx + 2.6 * k + rb * 0.7, cy + rb * 0.7), (cx + 4.4 * k, cy + 2.6 * k), (frt + 30, cy + 2.6 * k)], GOLD,
                width=2.5, wobble=0)
    fr = S2.finish(fr)
    im = Image.fromarray(fr)
    tag(im, frt + 40, cy - 4 * k, "1′ = 2 CONES", TRACER, 34, "lm")
    tag(im, fl - 40, cy + 2.2 * k, "5′ = 10 CONES", TRACER, 34, "rm")
    tag(im, frt + 40, cy + 2.6 * k, "BALL AT 160 YDS", GOLD, 34, "lm")
    S2.block(im, ["The 20/20 E", "on your cones"], ["If each cone worked like a pixel:", "dark if the E covers its centre."],
             x=110, y=260, size=84, accent=1)
    return im


def triptych_fields(light_for, labels, sub, k4=150.0, curve="linear", exposure=None):
    I, F, R = retina_mod()
    out = np.full((2160, 3840, 3), R.OUTSIDE.astype(np.uint8), np.uint8)
    cxs = [3840 * f for f in (0.2, 0.5, 0.8)]
    cy, rad = 2160 * 0.47, 2160 * 0.26
    for cx, lf in zip(cxs, light_for):
        sc = R.Scope(F.BALL_C[0], F.BALL_C[1], k4)
        f = I.mosaic(sc, lf, exposure=exposure or F.EXPO, centre=(cx, cy), radius=rad, curve=curve)
        x0, x1 = int(cx - rad - 6), int(cx + rad + 6)
        out[:, x0:x1] = f[:, x0:x1]
    fr = to_hd(out)
    return fr, [c / 2 for c in cxs], cy / 2, rad / 2


def f09_pixels():
    I, F, R = retina_mod()
    yds = (160, 320, 640)
    fr, cxs, cy, rad = triptych_fields([F.ball_light(y, sharp=1.0) for y in yds], None, None)
    k = 75.0
    for cx, y in zip(cxs, yds):
        fr = marker(fr, ring(cx, cy, k * float(P.ball_arcmin(y)) / 2 / 0.5 + 1, n=60), GOLD, width=3, wobble=0.3, seed=y)
    fr = S2.finish(fr)
    im = Image.fromarray(fr)
    for cx, y, n in zip(cxs, yds, ("4 CONES LIT", "1 CONE LIT", "0 CONES LIT")):
        tag(im, cx, cy + rad + 26, f"{y} YDS", CREAM, 40)
        tag(im, cx, cy + rad + 74, n, TRACER, 30)
    S2.headline(im, ["If cones were pixels, it would vanish."], 960, 150, size=72, anchor="ms")
    caption(im, "Gold ring: the ball’s real size on the retina.", x=110, y=1010, size=36)
    return im


def f10_fainter():
    I, F, R = retina_mod()
    yds = (160, 320, 640)
    fr, cxs, cy, rad = triptych_fields([F.ball_light(y, sharp=0.0) for y in yds], None, None, exposure=4.0)
    k = 75.0
    for cx in cxs:
        fr = marker(fr, ring(cx, cy, k * 1.6, n=90), CREAM, width=2.5, wobble=0.3, a=0.5, seed=3)
    fr = S2.finish(fr)
    im = Image.fromarray(fr)
    for cx, y, n in zip(cxs, yds, ("ALL THE LIGHT", "¼ OF THE LIGHT", "1/16 OF THE LIGHT")):
        tag(im, cx, cy + rad + 26, f"{y} YDS", CREAM, 40)
        tag(im, cx, cy + rad + 74, n, GOLD, 30)
    S2.headline(im, ["It doesn’t get smaller.", "It gets fainter."], 960, 130, size=72, anchor="ms", accent=1)
    caption(im, "Your eye blurs every point into a spot about 1′ wide. Past 160 yards, the ball’s image is that spot.",
            x=110, y=1010, size=34)
    return im


# ================================================================== broadcast data graphics
def chart_bg():
    fr = np.empty((H, W, 3), np.uint8)
    fr[:] = S2.INK
    return fr


def grid(fr, x0, y0, x1, y1, nx=6, ny=4, col=(40, 52, 44)):
    for i in range(nx + 1):
        x = int(x0 + (x1 - x0) * i / nx)
        cv2.line(fr, (x, int(y0)), (x, int(y1)), col, 1, cv2.LINE_AA)
    for j in range(ny + 1):
        y = int(y0 + (y1 - y0) * j / ny)
        cv2.line(fr, (int(x0), y), (int(x1), y), col, 1, cv2.LINE_AA)
    return fr


def line(fr, pts, col, width=4.0, a=1.0):
    m = np.zeros(fr.shape[:2], np.uint8)
    cv2.polylines(m, [np.round(np.asarray(pts) * 16).astype(np.int32)], False, 255, max(1, int(round(width))), cv2.LINE_AA, shift=4)
    return S2.over(fr, m.astype(np.float32) / 255, col, a)


def f11_profile():
    fr = chart_bg()
    x0, x1, y0, y1 = 720, 1760, 200, 760
    fr = grid(fr, x0, y0, x1, y1)
    xs = np.linspace(-3, 3, 241)
    peak = float(P.disk_image(0, float(P.ball_arcmin(160)), aperture=0.5)[0])
    cols = [WHITE, TRACER, GOLD, (150, 190, 150)]
    for j, yd in enumerate((160, 320, 640, 1280)):
        pr = P.disk_image(np.abs(xs), float(P.ball_arcmin(yd)), 1.0, 0.5)
        pts = np.c_[x0 + (xs + 3) / 6 * (x1 - x0), y1 - pr / peak * (y1 - y0) * 0.92]
        fr = line(fr, pts, cols[j], 5 if j == 0 else 4)
    # the spot's width at half height for the far ball: about 1′
    hx0, hx1 = x0 + (-0.48 + 3) / 6 * (x1 - x0), x0 + (0.48 + 3) / 6 * (x1 - x0)
    fr = line(fr, [(hx0, y1 + 40), (hx1, y1 + 40)], CREAM, 3)
    fr = line(fr, [(hx0, y1 + 30), (hx0, y1 + 50)], CREAM, 3)
    fr = line(fr, [(hx1, y1 + 30), (hx1, y1 + 50)], CREAM, 3)
    fr = S2.finish(fr)
    im = Image.fromarray(fr)
    labs = [("160 YDS", "ALL THE LIGHT"), ("320 YDS", "¼"), ("640 YDS", "1/16"), ("1,280 YDS", "1/64")]
    for j, (a, b) in enumerate(labs):
        y = 300 + j * 62
        text(im, (x1 - 230, y), a, 34, cols[j], "cond-Bold", "ls", track=1)
        text(im, (x1 - 90, y), b, 34, cols[j], "cond-SemiBold", "ls")
    text(im, ((hx0 + hx1) / 2, y1 + 66), "ABOUT 1′ WIDE AT EVERY DISTANCE", 28, CREAM, "cond-SemiBold", "mt", track=2)
    text(im, (x0, y0 - 24), "EXTRA LIGHT ON THE CONES, ACROSS THE SPOT", 28, MUTED, "cond-SemiBold", "ls", track=3)
    S2.block(im, ["Same size.", "Less light."], ["Twice as far: a quarter of the light,", "spread over the same spot."],
             x=110, y=330, size=96, accent=1)
    return im


def f12_recipe():
    fr = chart_bg()
    x0, x1, y0, y1 = 860, 1780, 190, 760
    rx = lambda f: x0 + np.log10(f) / 2 * (x1 - x0)
    ry = lambda lg: y0 + np.clip(-lg, 0, 4) / 4 * (y1 - y0)
    cut = P.csf_cutoff()
    win = np.zeros((H, W), np.float32)
    cv2.rectangle(win, (int(rx(1)), int(y0)), (int(rx(cut)), int(y1)), 1.0, -1)
    fr = S2.over(fr, win, (34, 52, 40))
    fr = grid(fr, x0, y0, x1, y1, nx=4, ny=4)
    fr = line(fr, [(rx(cut), y0), (rx(cut), y1)], GOLD, 3)
    f = np.logspace(0, 2, 600)
    a40 = float(P.ball_arcmin(40)) ** 2
    cols = {40: (120, 140, 125), 80: (150, 170, 150), 160: WHITE, 320: TRACER, 640: GOLD}
    for yd in (40, 80, 160, 320, 640):
        th = float(P.ball_arcmin(yd))
        amp = th ** 2 / a40 * np.abs(P.jinc(th * f / 60))
        lg = np.log10(np.maximum(amp, 1e-9))
        fr = line(fr, np.c_[rx(f), ry(lg)], cols[yd], 4 if yd >= 160 else 3)
    fr = S2.finish(fr)
    im = Image.fromarray(fr)
    for yd in (40, 80, 160, 320, 640):
        th = float(P.ball_arcmin(yd))
        lg = math.log10(th ** 2 / a40)
        text(im, (rx(1.1), ry(lg) - 10), f"{yd} YDS", 28, cols[yd], "cond-Bold", "lb", track=1)
    for fq in (1, 10, 100):
        text(im, (rx(fq), y1 + 18), str(fq), 28, MUTED, "cond-SemiBold", "mt")
    text(im, ((x0 + x1) / 2, y1 + 60), "FINER RIPPLES  (CYCLES PER DEGREE)", 28, MUTED, "cond-SemiBold", "mt", track=3)
    text(im, ((rx(1) + rx(cut)) / 2, y0 - 24), "WHAT YOUR EYE CAN SEE", 30, GOLD, "cond-Bold", "ms", track=3)
    S2.block(im, ["Same recipe.", "Only weaker."], ["Every image is a recipe", "of ripples of light. Past 320 yards,",
                                                     "the ball’s recipe is flat across", "everything your eye can see."],
             x=110, y=330, size=96, accent=1)
    return im


# ================================================================== why a dark background wins
OVERCAST_SKY = ((150, 156, 160), (196, 200, 202), (214, 216, 214))


def badge(im, x, y, s, col, a=1.0, size=44):
    d = ImageDraw.Draw(im, "RGBA")
    w_ = S2.text_width(s, size, "cond-ExtraBold") + 36
    d.rounded_rectangle([x - w_ / 2, y - size * 0.72, x + w_ / 2, y + size * 0.72], radius=8, fill=S2.PANEL + (230,))
    text(im, (x, y), s, size, col, "cond-ExtraBold", "mm", a)


def f13_backgrounds():
    out = np.empty((H, W, 3), np.uint8)
    out[:] = S2.INK
    shot = C.Shot(cx=960.0)
    scenes = [("DARK TREES", "+2,900%", C.course(shot, seed=1), (960, 600), (255, 255, 255)),
              ("BLUE SKY", "+150%", C.course(shot, seed=1), (960, 330), (255, 255, 255)),
              ("OVERCAST SKY", "−45%", C.course(shot, seed=1, sky=OVERCAST_SKY), (960, 330), (120, 122, 122))]
    pw, ph, gap = 560, 640, 40
    x0 = (W - 3 * pw - 2 * gap) / 2
    for j, (name, pct, fr, (bx, by), bcol) in enumerate(scenes):
        crop = fr[int(by - ph / 2):int(by + ph / 2), int(bx - pw / 2):int(bx + pw / 2)].copy()
        crop = S2.ball(np.ascontiguousarray(crop), pw / 2, ph / 2, 7, col=bcol, glow=0.25 if j < 2 else 0.0)
        X = int(x0 + j * (pw + gap))
        out[200:200 + ph, X:X + pw] = crop
    out = S2.finish(out)
    im = Image.fromarray(out)
    for j, (name, pct, *_r) in enumerate(scenes):
        X = x0 + j * (pw + gap)
        text(im, (X + 24, 236), name, 34, WHITE, "cond-Bold", "lt", track=3, shadow=(0, 2, 0.6))
        badge(im, X + pw / 2, 200 + ph - 70, pct, GOLD if j == 0 else (CREAM if j == 1 else S2.TRACER))
    S2.headline(im, ["Same ball. Different background."], 960, 130, size=72, anchor="ms")
    caption(im, ["Your eye judges a spot by how much brighter or darker it is than what’s around it.",
                 "A dark speck can never be more than 100% darker. A bright one can be thousands of percent brighter."],
            x=110, y=1010, size=32)
    return im


def f14_weber():
    I, F, R = retina_mod()
    out = np.full((2160, 3840, 3), R.OUTSIDE.astype(np.uint8), np.uint8)
    cxs = [3840 * f for f in (0.2, 0.5, 0.8)]
    cy, rad = 2160 * 0.47, 2160 * 0.26
    for cx, C_ in zip(cxs, (29.0, 1.5, -0.45)):
        sc = R.Scope(F.BALL_C[0], F.BALL_C[1], 150.0)
        light = lambda u, v, C_=C_: 0.16 * (1 + C_ * I.ball_at(u, v, 400, True, *F.BALL_C))
        f = I.mosaic(sc, light, exposure=1.0, centre=(cx, cy), radius=rad)
        x0, x1 = int(cx - rad - 6), int(cx + rad + 6)
        out[:, x0:x1] = f[:, x0:x1]
    fr = S2.finish(to_hd(out))
    im = Image.fromarray(fr)
    for cx, n, pct, col in zip(cxs, ("DARK TREES", "BLUE SKY", "OVERCAST SKY"), ("+146%", "+7.6%", "−2.3%"), (GOLD, CREAM, TRACER)):
        tag(im, cx / 2, cy / 2 + rad / 2 + 26, n, CREAM, 38)
        tag(im, cx / 2, cy / 2 + rad / 2 + 74, f"BRIGHTEST CONE {pct}", col, 30)
    S2.headline(im, ["Your eye sees change, not light."], 960, 150, size=72, anchor="ms")
    caption(im, "The ball at 400 yards, each field at the same exposure: what’s left is the change.", x=110, y=1010, size=34)
    return im


# ================================================================== the leaderboard
def f15_board():
    shot = C.Shot(cx=960.0)
    bg = C.course(shot, seed=1)
    bg = cv2.GaussianBlur(bg, (0, 0), 14)
    bg = (bg.astype(np.float32) * 0.55).astype(np.uint8)
    fr = bg
    bx0, by0, bx1, by1 = 300, 120, 1620, 930
    sh = np.zeros((H, W), np.float32)
    cv2.rectangle(sh, (bx0 + 16, by0 + 24), (bx1 + 16, by1 + 24), 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 20)
    fr = (fr.astype(np.float32) * (1 - 0.5 * sh[..., None])).astype(np.uint8)
    cv2.rectangle(fr, (bx0 - 14, by0 - 14), (bx1 + 14, by1 + 14), (28, 64, 40), -1)          # the frame
    cv2.rectangle(fr, (bx0, by0), (bx1, by1), (244, 242, 234), -1)
    cv2.rectangle(fr, (bx0, by0), (bx1, by0 + 120), (28, 70, 42), -1)                       # the header
    rows = [("1", "DARK TREES", "+2,900%", "980"), ("2", "THE FAIRWAY", "+500%", "620"), ("3", "BLUE SKY", "+150%", "380"),
            ("4", "BLACK BALL, ANY SKY", "−100%", "310"), ("5", "OVERCAST SKY", "−45%", "200")]
    cols_x = [bx0 + 70, bx0 + 170, bx0 + 860, bx1 - 70]
    rh = 112
    for i in range(len(rows) + 1):
        yy = by0 + 176 + i * rh
        cv2.line(fr, (bx0 + 20, yy), (bx1 - 20, yy), (200, 198, 188), 2)
    for x in (bx0 + 130, bx0 + 820, bx1 - 250):
        cv2.line(fr, (x, by0 + 176), (x, by0 + 176 + 5 * rh), (200, 198, 188), 2)
    fr = S2.finish(fr, grain=1.8)
    im = Image.fromarray(fr)
    text(im, ((bx0 + bx1) / 2, by0 + 62), "HOW FAR CAN YOU SEE IT?", 76, WHITE, "cond-ExtraBold", "mm", track=3)
    heads = ["POS", "BACKGROUND", "CONTRAST", "YARDS"]
    for x, h_ in zip(cols_x, heads):
        text(im, (x if x != cols_x[2] else x + 40, by0 + 160), h_, 24, (120, 120, 112), "cond-Bold", "ms" if x != cols_x[1] else "ls", track=3)
    for i, (pos, name, pct, yd) in enumerate(rows):
        yy = by0 + 176 + i * rh + rh / 2
        red = pct.startswith("+")
        text(im, (cols_x[0], yy), pos, 64, S2.PAPER_INK, "cond-ExtraBold", "mm")
        text(im, (cols_x[1], yy), name, 64, S2.PAPER_INK, "cond-ExtraBold", "lm")
        text(im, (cols_x[2] + 40, yy), pct, 64, (196, 40, 36) if red else S2.PAPER_INK, "cond-ExtraBold", "mm")
        text(im, (cols_x[3] - 60, yy), yd, 76, (196, 40, 36) if red else S2.PAPER_INK, "cond-ExtraBold", "mm")
    text(im, ((bx0 + bx1) / 2, by1 + 60), "A STILL BALL · CLEAR DAY · YOU KNOW WHERE TO LOOK", 30, CREAM, "cond-SemiBold", "mm", track=4)
    text(im, ((bx0 + bx1) / 2, 60), "THE RULE: CONTRAST × AREA ≥ ABOUT ONE CONE", 34, GOLD, "cond-Bold", "mm", track=4)
    return im


# ================================================================== stars
def f16_stars():
    shot = C.Shot(cx=960.0)
    night = ((6, 10, 22), (14, 24, 42), (34, 44, 58))
    fr = C.course(shot, seed=1, sky=night, light=0.0)
    f = fr.astype(np.float32)
    yy = np.arange(H)[:, None]
    ground = yy > shot.ground_y(310)
    f[ground[:, 0]] *= 0.18
    tm, _ = C.treeline(1, shot.ground_y(310), 16 * shot.f / 320, 28 * shot.f / 320)
    f = f * (1 - tm[..., None]) + np.array([4, 7, 6], np.float32) * tm[..., None]
    fr = np.clip(f, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(8)
    for _ in range(1400):
        x, y = rng.uniform(0, W), rng.uniform(0, shot.ground_y(310) - 120)
        b = rng.pareto(1.8) * 0.35
        r = 0.6 + min(b, 2.5) * 0.5
        fr = S2.ball(fr, x, y, r, col=(235, 238, 255), a=min(1.0, 0.25 + b), glow=0.3 if b > 1.5 else 0.0)
    fr = S2.finish(fr)
    im = Image.fromarray(fr)
    S2.block(im, ["A star is thousands of times", "narrower than a cone."], ["You see it anyway, because the sky behind it is almost black."],
             x=110, y=760, size=72, accent=1)
    return im


# ================================================================== try it
def f17_tryit():
    fr = np.empty((H, W, 3), np.uint8)
    fr[:] = S2.INK
    scenes = [("DARK TREES", (14, 30, 18), (255, 255, 255), [1, 1, 1, 1]), ("THE FAIRWAY", (78, 116, 52), (255, 255, 255), [1, 1, 1, 0]),
              ("BLUE SKY", (124, 160, 214), (255, 255, 255), [1, 1, 0, 0]), ("OVERCAST SKY", (230, 230, 232), (176, 176, 176), [1, 0, 0, 0])]
    yds = (100, 250, 500, 800)
    px_arcmin = 41.0 / 60
    pw, ph = 820, 300
    pos = [(110, 200), (990, 200), (110, 600), (990, 600)]
    marks = []
    for (name, bg, bc, seen), (x, y) in zip(scenes, pos):
        cv2.rectangle(fr, (x, y), (x + pw, y + ph), bg, -1)
        for k, yd in enumerate(yds):
            r = float(P.ball_arcmin(yd)) / px_arcmin / 2                # drawn to scale for a phone at reading distance
            fr = S2.ball(fr, x + 150 + k * 175, y + ph / 2, r, col=bc)
            marks.append((x + 150 + k * 175, y + ph + 50, seen[k]))
    for mx, my, ok in marks:
        if ok:
            fr = marker(fr, [(mx - 14, my), (mx - 3, my + 12), (mx + 18, my - 16)], TRACER, width=5, wobble=0.3)
        else:
            fr = marker(fr, [(mx - 12, my - 12), (mx + 12, my + 12)], MUTED, width=4, wobble=0.2)
            fr = marker(fr, [(mx + 12, my - 12), (mx - 12, my + 12)], MUTED, width=4, wobble=0.2)
    fr = S2.finish(fr, grain=1.2)
    im = Image.fromarray(fr)
    for (name, bg, bc, seen), (x, y) in zip(scenes, pos):
        col = WHITE if sum(bg) < 400 else S2.PAPER_INK
        text(im, (x + 20, y + 20), name, 30, col, "cond-Bold", "lt", track=3)
        for k, yd in enumerate(yds):
            text(im, (x + 150 + k * 175, y + ph - 26), f"{yd}", 26, col, "cond-SemiBold", "mb")
    S2.headline(im, ["Try it: drawn to scale"], 110, 140, size=64)
    text(im, (1810, 140), "SIZES FOR A PHONE AT READING DISTANCE · RED TICK: SEEN ON A SUNNY COURSE", 24, MUTED, "cond-SemiBold", "rs", track=2)
    return im


# ================================================================== the close
def f18_outro():
    fr, end = drive(0.999, ball=False)
    fr = S2.ball(fr, end[0], end[1] - 3, 5, glow=0.6)
    fr = marker(fr, ring(end[0], end[1] - 3, 30), TRACER, width=6, seed=9)
    im = Image.fromarray(S2.finish(fr))
    bug(im, "TRACER VIEW · BALL ENLARGED")
    lower_third(im, "Carry", "273 YDS", y=980, sub=ball_line(273))
    S2.block(im, ["You don’t need to", "resolve it."], ["You only need one spot on your retina", "a little brighter than the rest."],
             x=120, y=260, size=88, accent=1)
    return im


# ================================================================== thumbnails (1280×720)
def thumb(im):
    return im.resize((1280, 720), Image.LANCZOS)


def t1_160():
    fr, end = drive(t_for_yds(262))
    fr = marker(fr, ring(end[0], end[1], 44), TRACER, width=9, seed=11)
    im = Image.fromarray(S2.finish(fr))
    text(im, (84, 400), "160", 340, WHITE, "cond-Black", "ls", shadow=(0, 6, 0.5))
    text(im, (96, 520), "YARDS", 120, TRACER, "cond-Black", "ls", track=8, shadow=(0, 4, 0.5))
    text(im, (100, 610), "YOUR EYES SHOULDN’T", 62, WHITE, "cond-ExtraBold", "ls", track=1, shadow=(0, 3, 0.6))
    text(im, (100, 680), "SEE THIS BALL", 62, WHITE, "cond-ExtraBold", "ls", track=1, shadow=(0, 3, 0.6))
    return thumb(im)


def t2_fainter():
    im = f10_fainter()
    fr = np.asarray(im).copy()
    cv2.rectangle(fr, (0, 0), (W, 240), S2.INK, -1)
    cv2.rectangle(fr, (0, 940), (W, H), S2.INK, -1)
    im = Image.fromarray(fr)
    S2.headline(im, ["It doesn’t get smaller.", "It gets fainter."], 960, 120, size=96, anchor="ms", accent=1)
    return thumb(im)


def t3_board():
    im = f15_board()
    fr = np.asarray(im).copy()
    fr = marker(fr, ring(1620 - 70 - 60, 120 + 176 + 56, 72), TRACER, width=9, seed=12)
    return thumb(Image.fromarray(fr))


FRAMES = {"01_drive": f01_drive, "02_160": f02_160, "03_land": f03_land, "04_title": f04_title, "05_yardage": f05_yardage,
          "06_chart": f06_chart, "07_eye": f07_eye, "08_e_cones": f08_e_cones, "09_pixels": f09_pixels, "10_fainter": f10_fainter,
          "11_profile": f11_profile, "12_recipe": f12_recipe, "13_backgrounds": f13_backgrounds, "14_weber": f14_weber,
          "15_board": f15_board, "16_stars": f16_stars, "17_tryit": f17_tryit, "18_outro": f18_outro,
          "thumb_A": t1_160, "thumb_B": t2_fainter, "thumb_C": t3_board}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(FRAMES)
    for n in names:
        im = FRAMES[n]()
        im.save(os.path.join(OUT, f"{n}.png"))
        print("wrote", n, flush=True)
