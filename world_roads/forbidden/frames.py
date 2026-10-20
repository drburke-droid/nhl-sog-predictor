"""Storyboard style frames for "Reddish green" (stills only, no video).

    python frames.py            -> out/frame_01.png ... out/frame_09.png
"""
import os, sys, math
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lab as L
from lab import W, H, GEL

# ------------------------------------------------------------------ colour science
import warnings
warnings.filterwarnings("ignore")
import colour

_C31 = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
_SS = colour.MSDS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"]


def _interp(sd, lam):
    w = sd.wavelengths
    return np.stack([np.interp(lam, w, sd.values[:, i], left=0, right=0) for i in range(3)], -1)


def cmf(lam):
    """CIE 1931 2-degree colour matching functions (the tabulated standard)"""
    return _interp(_C31, np.asarray(lam, np.float64))


def cones(lam):
    """L, M, S cone fundamentals (Stockman & Sharpe 2000, 2 degree), each peaking at 1"""
    return _interp(_SS, np.asarray(lam, np.float64))


# the 3x3 matrix taking CIE 1931 XYZ to the cone fundamentals, by least squares over the spectrum
_lam = np.arange(390, 781, 1.0)
XYZ2LMS = np.linalg.lstsq(cmf(_lam), cones(_lam), rcond=None)[0].T

XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def gamut(rgb):
    """bring linear sRGB into the screen's range without changing its hue: desaturate toward the grey of the same
    luminance just far enough that no channel is negative, then scale so the largest channel is 1"""
    rgb = np.asarray(rgb, np.float64)
    Y = (rgb @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    lo = rgb.min(-1, keepdims=True)
    k = np.where(lo < 0, Y / np.maximum(Y - lo, 1e-9), 1.0)
    rgb = Y + (rgb - Y) * k
    return rgb / np.maximum(rgb.max(-1, keepdims=True), 1e-9)


def spectral_rgb(lam):
    """display colour of a monochromatic light (linear sRGB, hue kept, saturation reduced to fit the screen)"""
    return gamut(cmf(lam) @ XYZ2RGB.T)


# ------------------------------------------------------------------ frames
def stage_two_spots(sep, cap, title=None):
    img = L.room().copy()
    y = 450
    xs = (960 - sep / 2, 960 + sep / 2)
    for x, c, src, seed in ((xs[0], "red", (260, -260), 1), (xs[1], "green", (1660, -260), 2)):
        L.beam(img, src, (x, y), 250, GEL[c], gain=0.09, seed=seed)
        L.dust(img, src, (x, y), 250, GEL[c], seed=seed)
        L.pool(img, x, y, 250, GEL[c], gain=0.9)
    if title:
        L.projected_text(img, (960, 860), title, 64, gain=0.0)
    L.floor(img)
    im = Image.fromarray(L.finish(img, seed=3))
    if title:
        d = ImageDraw.Draw(im)
        d.text((960, 830), title, font=L.font("display", 66, 700), fill=(238, 238, 234), anchor="ms")
    L.caption(im, cap)
    return im


def f_open():
    return stage_two_spots(300, ["Red light plus green light makes yellow.",
                                 "But yellow isn’t a red that’s also green. Can you see one?"])


def hering_hue(a):
    """Hering's circle: the four unique hues 90 degrees apart, opposites across the middle (display approximations)"""
    keys = [(0, (1.00, 0.80, 0.00)),        # yellow, top
            (90, (0.95, 0.03, 0.10)),       # red, right
            (180, (0.02, 0.12, 0.95)),      # blue, bottom
            (270, (0.00, 0.62, 0.30)),      # green, left
            (360, (1.00, 0.80, 0.00))]
    a = a % 360
    for (a0, c0), (a1, c1) in zip(keys[:-1], keys[1:]):
        if a0 <= a <= a1:
            u = (a - a0) / (a1 - a0)
            return np.array(c0) * (1 - u) + np.array(c1) * u


def f_ring():
    img = L.room().copy()
    cx, cy, R0, R1 = 960, 455, 250, 320
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xx - cx, yy - cy)
    ang = (np.degrees(np.arctan2(xx - cx, -(yy - cy))) + 360) % 360
    band = np.clip(np.minimum(r - R0, R1 - r) / 2 + 0.5, 0, 1)
    lut = np.array([hering_hue(a) for a in range(361)], np.float32)
    col = lut[np.round(ang).astype(int)]
    img += (band[..., None] * col) * 0.85
    img += cv2.GaussianBlur(band[..., None] * col, (0, 0), 18) * 0.25
    # opposite pairs: lines through the middle that end in a question
    for a in (0, 90):
        t = math.radians(a)
        p0 = (cx + math.sin(t) * (R0 - 20), cy - math.cos(t) * (R0 - 20))
        p1 = (cx - math.sin(t) * (R0 - 20), cy + math.cos(t) * (R0 - 20))
        segs = []
        n = 18
        for k in range(n):
            if k % 2 == 0:
                u0, u1 = k / n, (k + 1) / n
                segs.append([(p0[0] + (p1[0] - p0[0]) * u0, p0[1] + (p1[1] - p0[1]) * u0),
                             (p0[0] + (p1[0] - p0[0]) * u1, p0[1] + (p1[1] - p0[1]) * u1)])
        L.light_lines(img, segs, (0.8, 0.8, 0.8), width=2.0, gain=0.5)
    L.floor(img, 0.10)
    im = Image.fromarray(L.finish(img, seed=4))
    d = ImageDraw.Draw(im)
    f = L.font("text", 30)
    for a, s in ((45, "orange: a reddish yellow"), (135, "purple: a bluish red"),
                 (225, "teal: a greenish blue"), (315, "lime: a yellowish green")):
        t = math.radians(a)
        x, y = cx + math.sin(t) * (R1 + 40), cy - math.cos(t) * (R1 + 40)
        d.text((x, y), s, font=f, fill=(214, 214, 210), anchor="lm" if 0 < a < 180 else "rm")
    d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=(150, 150, 150))
    d.text((cx + 20, cy - 26), "?", font=L.font("display", 60, 600), fill=(240, 240, 236), anchor="lm")
    L.caption(im, ["Neighbours blend. Opposites don’t:",
                   "there is no reddish green, and no yellowish blue."])
    return im


def f_cancel():
    img = L.room().copy()
    # the patch mid-sweep: green-yellow with some red light added
    col = L.to_lin((214, 205, 18)) * 1.0
    yy, xx = np.ogrid[0:H, 0:W]
    d = np.hypot(xx - 960, yy - 430)
    disc = np.clip((270 - d) / 1.5 + 0.5, 0, 1).astype(np.float32)
    img = img * (1 - disc[..., None]) + disc[..., None] * col[None, None, :]
    # the red light, as a meter of light below the patch
    m = np.zeros((H, W), np.float32)
    for k in range(12):
        x0 = 720 + k * 40
        cv2.rectangle(m, (x0, 780), (x0 + 28, 800), 1.0 if k < 5 else 0.12, -1)
    L.light_fill(img, m, GEL["red"], 0.8)
    L.floor(img, 0.08)
    im = Image.fromarray(L.finish(img, seed=5, grain=0.0))
    L.label(im, (720, 820), "red light added", 24)
    L.test_tag(im, 1, 18, 0.35)
    L.caption(im, ["Watch the patch as red light is added.",
                   "Say “now” when it stops looking green, before it starts looking red."])
    return im


def plot_xy(lam):
    return 260 + (lam - 400) / 300 * 1400


def f_cones():
    img = L.room().copy()
    lam = np.linspace(400, 700, 601)
    # the spectrum, projected as a strip
    strip = np.zeros((H, W), np.float32)
    x0, x1 = int(plot_xy(400)), int(plot_xy(700))
    strip[760:790, x0:x1] = 1
    rgb = spectral_rgb(np.interp(np.arange(W), plot_xy(lam), lam))
    img += strip[..., None] * rgb * 0.55
    img += cv2.GaussianBlur(strip[..., None] * rgb, (0, 0), 10) * 0.2
    lms = cones(lam)
    base, height = 720, 470
    for i, c in ((2, (0.35, 0.30, 1.0)), (1, (0.15, 1.0, 0.25)), (0, (1.0, 0.25, 0.10))):
        pts = np.c_[plot_xy(lam), base - lms[:, i] * height]
        L.light_lines(img, [pts], c, width=3.2, gain=0.95)
    # the probe: 580 nm
    xp = plot_xy(575)
    L.light_lines(img, [[(xp, 230), (xp, 795)]], (1.0, 0.85, 0.1), width=2.0, gain=0.6)
    L.floor(img, 0.08)
    im = Image.fromarray(L.finish(img, seed=6))
    d = ImageDraw.Draw(im)
    l575 = cones(np.array([575.0]))[0]
    for i, name, colr in ((0, "L", (255, 120, 90)), (1, "M", (120, 255, 140)), (2, "S", (150, 150, 255))):
        pk = lam[np.argmax(lms[:, i])]
        d.text((plot_xy(pk), base - height - 26), name, font=L.font("display", 48, 700), fill=colr, anchor="ms")
    for i in (0, 1):
        y = base - l575[i] * height
        d.ellipse((xp - 8, y - 8, xp + 8, y + 8), fill=(255, 240, 200))
    for nm in (400, 450, 500, 550, 600, 650, 700):
        d.text((plot_xy(nm), 818), f"{nm}", font=L.font("mono", 22), fill=(170, 170, 166), anchor="mt")
    d.text((plot_xy(700) + 14, 818), "nm", font=L.font("mono", 22), fill=(170, 170, 166), anchor="lt")
    d.text((xp + 14, 240), "575 nm", font=L.font("mono2", 24), fill=(255, 225, 120), anchor="lt")
    L.caption(im, ["Three kinds of cone, each tuned to its own band of the spectrum.",
                   "Yellow light, around 575 nm, drives both L and M cones strongly."])
    return im


def seesaw(img, cx, cy, tilt, cols, length=420):
    """a balance beam of light: pivot, beam tilted by `tilt` degrees (+: right end down), coloured ends"""
    t = math.radians(tilt)
    dx, dy = math.cos(t) * length / 2, math.sin(t) * length / 2
    L.light_lines(img, [[(cx - dx, cy - dy), (cx + dx, cy + dy)]], (0.9, 0.9, 0.9), width=3.0, gain=0.7)
    L.light_lines(img, [[(cx, cy + 4), (cx - 22, cy + 40), (cx + 22, cy + 40), (cx, cy + 4)]], (0.6, 0.6, 0.6), width=2.4, gain=0.6)
    for sgn, c in ((-1, cols[0]), (1, cols[1])):
        m = np.zeros((H, W), np.float32)
        cv2.circle(m, (int(cx + sgn * dx), int(cy + sgn * dy)), 22, 1.0, -1, cv2.LINE_AA)
        L.light_fill(img, m, c, 0.9)
        L.light_fill(img, cv2.GaussianBlur(m, (0, 0), 14), c, 0.5)
    return img


def f_seesaws():
    img = L.room().copy()
    rows = [(300, 0, (GEL["red"], GEL["green"])), (500, -14, (GEL["yellow"], GEL["blue"])), (700, -10, (GEL["white"] * 0.9, np.array([0.08, 0.08, 0.08])))]
    for y, tilt, cols in rows:
        seesaw(img, 1020, y, tilt, cols)
    # the probe colour, as a projected swatch
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (1580, 490), 120, 1.0, -1, cv2.LINE_AA)
    sw = spectral_rgb(np.array([575.0]))[0]
    L.light_fill(img, m, sw, 0.85)
    # wires from the cones
    for y in (300, 500, 700):
        L.light_lines(img, [[(470, y), (760, y)]], (0.5, 0.5, 0.5), width=1.6, gain=0.4)
    L.floor(img, 0.08)
    im = Image.fromarray(L.finish(img, seed=7))
    d = ImageDraw.Draw(im)
    f = L.font("mono2", 26)
    for y, s in ((300, "L − M"), (500, "S − (L + M)"), (700, "L + M")):
        d.text((450, y), s, font=f, fill=(210, 210, 206), anchor="rm")
    fl = L.font("text", 26)
    for y, tilt, a, b in ((300, 0, "red", "green"), (500, -14, "yellow", "blue"), (700, -10, "light", "dark")):
        t = math.radians(tilt)
        d.text((1020 - math.cos(t) * 210, y - math.sin(t) * (-210) * -1 + 44), a, font=fl, fill=(200, 200, 196), anchor="mt")
        d.text((1020 + math.cos(t) * 210, y + math.sin(t) * 210 + 44), b, font=fl, fill=(200, 200, 196), anchor="mt")
    d.text((1580, 640), "575 nm", font=L.font("mono2", 24), fill=(230, 210, 140), anchor="mt")
    L.caption(im, ["Your brain reads colour on three seesaws. Each can tip one way or the other.",
                   "At 575 nm the red–green one balances: no red, no green. You see yellow."])
    return im


def f_stare():
    img = np.zeros((H, W, 3), np.float32) + 0.003
    yy, xx = np.ogrid[0:H, 0:W]
    d = np.hypot(xx - 960, yy - 440)
    disc = np.clip((230 - d) / 1.5 + 0.5, 0, 1).astype(np.float32)
    img += disc[..., None] * L.to_lin((255, 214, 0))[None, None, :]
    im = Image.fromarray(L.finish(img, seed=8, grain=0.0))
    dr = ImageDraw.Draw(im)
    dr.line((948, 440, 972, 440), fill=(10, 10, 10), width=4)
    dr.line((960, 428, 960, 452), fill=(10, 10, 10), width=4)
    L.test_tag(im, 2, 30, 0.1)
    L.caption(im, ["Stare at the cross for 30 seconds. Don’t look away.",
                   "When the yellow disappears, keep looking at the same spot."])
    return im


def f_stabilized():
    img = L.room().copy()
    # left: an eye, front view, with the tracker's reticle locked on the pupil
    t = np.linspace(-1, 1, 80)
    upper = np.c_[520 + 230 * t, 450 - 120 * (1 - t ** 2) ** 0.9]
    lower = np.c_[520 + 230 * t, 450 + 105 * (1 - t ** 2) ** 0.9]
    a = np.linspace(0, 2 * np.pi, 120)
    iris = np.c_[540 + 88 * np.cos(a), 445 + 88 * np.sin(a)]
    pupil = np.c_[540 + 34 * np.cos(a), 445 + 34 * np.sin(a)]
    L.light_lines(img, [upper, lower], (0.8, 0.8, 0.8), width=2.6, gain=0.6)
    L.light_lines(img, [iris], (0.45, 0.65, 0.8), width=2.4, gain=0.6)
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (540, 445), 34, 1.0, -1, cv2.LINE_AA)
    img *= (1 - 0.9 * m)[..., None]
    L.light_lines(img, [pupil], (0.6, 0.6, 0.6), width=1.6, gain=0.4)
    r = 62
    ret = [[(540 - r, 445 - r + 18), (540 - r, 445 - r), (540 - r + 18, 445 - r)], [(540 + r - 18, 445 - r), (540 + r, 445 - r), (540 + r, 445 - r + 18)],
           [(540 - r, 445 + r - 18), (540 - r, 445 + r), (540 - r + 18, 445 + r)], [(540 + r - 18, 445 + r), (540 + r, 445 + r), (540 + r, 445 + r - 18)]]
    L.light_lines(img, ret, (1.0, 0.35, 0.25), width=2.4, gain=0.8)
    # the lock: a dashed line from the eye to the stripes
    segs = [[(640 + k * 30, 445), (655 + k * 30, 445)] for k in range(13)]
    L.light_lines(img, segs, (1.0, 0.35, 0.25), width=2.0, gain=0.5)
    # right: red and green stripes, fading into each other where they meet
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, R = 1300, 450, 260
    inside = np.clip((R - np.hypot(xx - cx, yy - cy)) / 1.5 + 0.5, 0, 1)
    ph = (xx - cx) / 90.0
    s = 0.5 + 0.5 * np.tanh(np.sin(ph * math.pi) * 2.2)
    red, green = L.to_lin((205, 40, 30)), L.to_lin((30, 150, 50))
    stripes = s[..., None] * red + (1 - s[..., None]) * green
    img = img * (1 - inside[..., None]) + inside[..., None] * stripes * 0.9
    L.floor(img, 0.08)
    im = Image.fromarray(L.finish(img, seed=9))
    d = ImageDraw.Draw(im)
    d.text((540, 610), "eye tracker: the stripes move with every eye movement", font=L.font("mono", 22), fill=(190, 190, 186), anchor="mt")
    d.text((1300, 760), "Crane & Piantanida, 1983", font=L.font("mono", 22), fill=(170, 170, 166), anchor="mt")
    L.caption(im, ["1983: hold red and green stripes perfectly still on the retina,",
                   "and the borders fade. Some observers reported a reddish green."])
    return im


def f_olo():
    """the cone triangle: each point a mix of L, M and S responses. Real light stays inside the spectral curve,
    because the cones' sensitivities overlap; the corners are out of reach. Olo sits by the M corner."""
    img = L.room().copy()
    V = {"L": np.array([1370.0, 760.0]), "M": np.array([960.0, 120.0]), "S": np.array([550.0, 760.0])}
    def P(l, m, s_):
        t = l + m + s_
        return (V["L"] * l + V["M"] * m + V["S"] * s_) / t
    lam = np.arange(400, 701, 1.0)
    lms = cones(lam)
    curve = np.array([P(*v) for v in lms])
    curve = np.vstack([curve, [P(*cones(np.array([400.0]))[0])]])
    # fill the region real light can reach with its colours
    mask = np.zeros((H, W), np.float32)
    cv2.fillPoly(mask, [np.round(curve * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A = np.array([[V["L"][0] - V["S"][0], V["M"][0] - V["S"][0]], [V["L"][1] - V["S"][1], V["M"][1] - V["S"][1]]])
    inv = np.linalg.inv(A)
    rel = np.stack([xx - V["S"][0], yy - V["S"][1]], -1) @ inv.T
    l_, m_ = rel[..., 0], rel[..., 1]
    s_ = 1 - l_ - m_
    with np.errstate(all="ignore"):
        rgb = gamut((np.stack([l_, m_, s_], -1) @ np.linalg.inv(XYZ2LMS).T) @ XYZ2RGB.T)
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)
    img += mask[..., None] * rgb.astype(np.float32) * 0.42
    L.light_lines(img, [curve], (0.9, 0.9, 0.9), width=2.2, gain=0.6)
    L.light_lines(img, [[V["L"], V["M"], V["S"], V["L"]]], (0.45, 0.45, 0.45), width=1.6, gain=0.45)
    # the screen's primaries
    prim = np.array([[0.64, 0.33, 0.03], [0.30, 0.60, 0.10], [0.15, 0.06, 0.79]])      # x, y, z of sRGB primaries
    tri = [P(*(XYZ2LMS @ (p / p[1]))) for p in prim]
    L.light_lines(img, [tri + [tri[0]]], (1.0, 1.0, 1.0), width=2.0, gain=0.8)
    m = np.zeros((H, W), np.float32)
    o = P(0.02, 0.96, 0.02)
    cv2.circle(m, (int(o[0]), int(o[1])), 13, 1.0, -1, cv2.LINE_AA)
    L.light_fill(img, m, (0.2, 1.0, 0.75), 1.0)
    L.light_fill(img, cv2.GaussianBlur(m, (0, 0), 16), (0.2, 1.0, 0.75), 0.6)
    L.floor(img, 0.06)
    im = Image.fromarray(L.finish(img, seed=10))
    d = ImageDraw.Draw(im)
    fm = L.font("mono", 22)
    d.text((V["L"][0] + 18, V["L"][1] + 6), "only L", font=fm, fill=(200, 200, 196), anchor="lt")
    d.text((V["S"][0] - 18, V["S"][1] + 6), "only S", font=fm, fill=(200, 200, 196), anchor="rt")
    for nm in (460, 490, 510, 530, 560, 600):
        q = curve[int(nm - 400)]
        c = curve.mean(0)
        v = (q - c) / np.linalg.norm(q - c)
        d.text(tuple(q + v * 30), f"{nm}", font=L.font("mono", 20), fill=(170, 170, 166), anchor="mm")
    cen = np.mean(tri, 0)
    d.text((cen[0], cen[1] + 40), "your screen", font=L.font("text", 26), fill=(235, 235, 230), anchor="mm")
    d.text((o[0] + 30, o[1] + 4), "olo: only M", font=L.font("display", 44, 700), fill=(150, 255, 210), anchor="lm")
    d.text((690, 420), "every colour", font=L.font("text", 26), fill=(210, 210, 206), anchor="rm")
    d.text((690, 454), "real light can make", font=L.font("text", 26), fill=(210, 210, 206), anchor="rm")
    L.caption(im, ["2025: a laser stimulated only M cones. People saw “olo”, a blue-green",
                   "outside the range of any light in nature, and of any screen."])
    return im


def f_close():
    return stage_two_spots(40, ["Three kinds of cone. Three seesaws.", "And some colours that never fit."])


FRAMES = [f_open, f_ring, f_cancel, f_cones, f_seesaws, f_stare, f_stabilized, f_olo, f_close]

if __name__ == "__main__":
    os.makedirs(L.OUT, exist_ok=True)
    which = [int(a) for a in sys.argv[1:]] or range(1, len(FRAMES) + 1)
    for k in which:
        r = FRAMES[k - 1]()
        im = r[0] if isinstance(r, tuple) else r
        if isinstance(r, tuple):
            print("olo xy", r[1])
        p = os.path.join(L.OUT, f"frame_{k:02d}.png")
        im.save(p)
        print("wrote", p, flush=True)
