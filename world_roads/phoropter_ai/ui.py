"""Shared look and drawing for the phoropter video (1920x1080, 30 fps): the patient's view, the phoropter, panels."""
import os, sys, math
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                   # Inter fonts, Pillow translucent-text fix, captions
from optics import psf, effective_residual

OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
BG = (8, 10, 16)
INK, INK2, MUTED = (240, 244, 250), (170, 186, 206), (120, 136, 158)
GOLD, RED, GREEN, CYAN = (255, 186, 80), (255, 110, 90), (110, 220, 140), (90, 200, 255)
PANEL = (6, 10, 18)
CARD = (18, 24, 36)
LINE = (44, 54, 72)
font = ST.font
fade = ST.fade


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def to_lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return (np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ the eye chart
PX = 0.22                                            # arcmin of visual angle per chart pixel
CHART_LINES = [(0.70, "20/100", "HV"), (0.54, "20/70", "KDN"), (0.40, "20/50", "ZRSC"), (0.30, "20/40", "NOVHK"),
               (0.18, "20/30", "DSZRC"), (0.10, "20/25", "VKNHOZ"), (0.00, "20/20", "RCSDVN"),
               (-0.12, "20/15", "OHZKCRS")]
CHART_W, CHART_H = 500, 772
WHITE_LIN, BLACK_LIN = 0.80, 0.012


@lru_cache(None)
def chart():
    """Ink coverage (1 = black letter) at PX arcmin per pixel, plus the y centre of each line."""
    ss = 4
    im = Image.new("L", (CHART_W * ss, CHART_H * ss), 0)
    d = ImageDraw.Draw(im)
    ref = ImageFont.truetype(os.path.join(ST.FD, "Inter-Bold.otf"), 1000)
    cap = ref.getbbox("H")[3] - ref.getbbox("H")[1]
    y = 44.0
    ys = []
    for lm, name, letters in CHART_LINES:
        h = 5 * 10 ** lm / PX                         # letter height in chart pixels (5 x MAR)
        f = ImageFont.truetype(os.path.join(ST.FD, "Inter-Bold.otf"), int(round(1000 * h * ss / cap)))
        n = len(letters)
        total = (2 * n - 1) * h                      # letters one letter-width apart
        x = (CHART_W - total) / 2
        for k, ch in enumerate(letters):
            cx = x + (2 * k + 0.5) * h
            d.text((cx * ss, (y + h / 2) * ss), ch, fill=255, font=f, anchor="mm")
        ys.append(y + h / 2)
        y += h + max(0.75 * h, 14)
    ink = np.asarray(im, np.float32).reshape(CHART_H, ss, CHART_W, ss).mean(axis=(1, 3)) / 255.0
    return ink, ys


@lru_cache(64)
def proj_chart(top):
    """A projector screen showing chart.LINES[top] and the smaller lines below it (as many as fit), each with
    chart.n_letters letters one letter-width apart. Returns (ink, [(line index, y centre, letter height)])."""
    import chart as CH
    ss = 4
    im = Image.new("L", (CHART_W * ss, CHART_H * ss), 0)
    d = ImageDraw.Draw(im)
    ref = ImageFont.truetype(os.path.join(ST.FD, "Inter-Bold.otf"), 1000)
    cap = ref.getbbox("H")[3] - ref.getbbox("H")[1]
    sloan = "CDHKNORSVZ"
    y, rows = 0.0, []
    for li in range(top, len(CH.LINES)):                   # lay out first, then centre vertically
        h = 5 * 10 ** CH.LINES[li] / PX
        if y + h > CHART_H - 60 and rows:
            break
        rows.append((li, y + h / 2, h))
        y += h + max(0.75 * h, 14)
    used = rows[-1][1] + rows[-1][2] / 2
    dy = (CHART_H - used) / 2
    rows = [(li, yc + dy, h) for li, yc, h in rows]
    for li, yc, h in rows:
        f = ImageFont.truetype(os.path.join(ST.FD, "Inter-Bold.otf"), int(round(1000 * h * ss / cap)))
        n = int(CH.N_PER_LINE[li])
        x = (CHART_W - (2 * n - 1) * h) / 2
        letters = np.random.default_rng(100 + li).choice(10, n, replace=False)    # fixed per line, no repeats
        for k in range(n):
            d.text(((x + (2 * k + 0.5) * h) * ss, yc * ss), sloan[letters[k]], fill=255, font=f, anchor="mm")
    ink = np.asarray(im, np.float32).reshape(CHART_H, ss, CHART_W, ss).mean(axis=(1, 3)) / 255.0
    return ink, rows


@lru_cache(4096)
def _view(M, J0, J45, pupil, which=None):
    ink = chart()[0] if which is None else proj_chart(which)[0]
    if abs(M) < 1e-4 and abs(J0) < 1e-4 and abs(J45) < 1e-4:
        h = psf(0.0, 0.0, 0.0, pupil, PX)
    else:
        h = psf(M, J0, J45, pupil, PX)
    pad = h.shape[0] // 2 + 2
    blurred = fftconvolve(np.pad(ink, pad), h, mode="same")[pad:-pad, pad:-pad]
    lin = WHITE_LIN + (BLACK_LIN - WHITE_LIN) * np.clip(blurred, 0, 1)
    return to_srgb(lin)


def patient_view(true_pv, lens_pv, pupil, amp, which=None):
    """What the patient sees through lens_pv (they accommodate if it helps). which: None for the wall chart, or
    the index of the top line on the projector screen. Returns (uint8 image, accommodation)."""
    Mr = true_pv[0] - lens_pv[0]
    Me, acc = effective_residual(Mr, amp)
    key = (round(float(Me), 3), round(float(true_pv[1] - lens_pv[1]), 3), round(float(true_pv[2] - lens_pv[2]), 3),
           round(float(pupil), 2), which)
    return _view(*key), float(acc)


# ------------------------------------------------------------------ drawing helpers
def rr(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def text(d, xy, s, size, col=INK, weight="SemiBold", anchor="la", a=1.0):
    if a <= 0:
        return
    d.text(xy, s, font=font(size, weight), fill=tuple(col) + (int(255 * a),), anchor=anchor)


def fmt_d(x, plus=True):
    x = float(x)
    if abs(x) < 0.005:
        return "Plano" if plus else "0.00"
    return f"{'+' if x > 0 else '−'}{abs(x):.2f}"


def fmt_rx(S, C, ax):
    if abs(C) < 0.005:
        return f"{fmt_d(S)} DS"
    return f"{fmt_d(S)} {fmt_d(C)} × {int(round(ax)):03d}"


def phoropter(d, cx, cy, s, ax, cyl, jcc=None, a=1.0):
    """A stylised phoropter face (examiner's view). The patient's right eye is on the left.
    ax: cylinder axis (TABO, counter-clockwise from the examiner's right); jcc: minus-axis angle of a JCC in front."""
    A = int(255 * a)
    body = (34, 42, 58, A)
    rr(d, [cx - 330 * s, cy - 120 * s, cx + 330 * s, cy + 120 * s], int(110 * s), fill=body, outline=(70, 84, 108, A),
       width=max(1, int(3 * s)))
    rr(d, [cx - 40 * s, cy - 150 * s, cx + 40 * s, cy - 100 * s], int(14 * s), fill=body)
    for k, ex in enumerate((-170, 170)):
        x, y, r = cx + ex * s, cy, 88 * s
        d.ellipse([x - r - 14 * s, y - r - 14 * s, x + r + 14 * s, y + r + 14 * s], fill=(22, 28, 40, A),
                  outline=(90, 104, 128, A), width=max(1, int(3 * s)))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(14, 30, 44, A), outline=(60, 120, 150, A), width=max(1, int(2 * s)))
        # degree ticks
        for deg in range(0, 180, 15):
            t = math.radians(deg)
            for sgn in (1, -1):
                r0, r1 = r + 3 * s, r + (12 if deg % 45 == 0 else 7) * s
                d.line([(x + sgn * r0 * math.cos(t), y - sgn * r0 * math.sin(t)),
                        (x + sgn * r1 * math.cos(t), y - sgn * r1 * math.sin(t))], fill=(120, 136, 158, A),
                       width=max(1, int(2 * s)))
        if k == 0:
            if cyl < 0:
                t = math.radians(ax)
                dx, dy = math.cos(t) * r * 0.92, -math.sin(t) * r * 0.92
                d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=GOLD + (A,), width=max(2, int(5 * s)))
            if jcc is not None:
                t = math.radians(jcc)
                for sgn in (1, -1):
                    for f in (0.35, 0.62):
                        px, py = x + sgn * math.cos(t) * r * f, y - sgn * math.sin(t) * r * f
                        d.ellipse([px - 9 * s, py - 9 * s, px + 9 * s, py + 9 * s], fill=(235, 70, 60, A))
                        qx, qy = x + sgn * math.cos(t + math.pi / 2) * r * f, y - sgn * math.sin(t + math.pi / 2) * r * f
                        d.ellipse([qx - 9 * s, qy - 9 * s, qx + 9 * s, qy + 9 * s], fill=(240, 240, 240, A))
        else:
            d.ellipse([x - r * 0.9, y - r * 0.9, x + r * 0.9, y + r * 0.9], fill=(10, 14, 20, A))
            text(d, (x, y), "OCCLUDED", int(20 * s), MUTED, "Medium", "mm", a)
    text(d, (cx - 170 * s, cy + 128 * s), "PATIENT'S RIGHT EYE", int(18 * s), MUTED, "Medium", "ma", a)


def snellen_labels(d, x, y0, a=1.0, size=19):
    _, ys = chart()
    for (lm, name, _), yy in zip(CHART_LINES, ys):
        text(d, (x, y0 + yy), name, size, MUTED, "Medium", "lm", a)


def badge(d, cx, cy, s, col, a=1.0, r=34):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(col) + (int(235 * a),))
    text(d, (cx, cy + 1), s, int(r * 1.15), BG, "Bold", "mm", a)


def new_frame(col=BG):
    im = Image.new("RGB", (W, H), col)
    return im, ImageDraw.Draw(im, "RGBA")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    from optics import to_pv
    E = to_pv(-2.25, -1.0, 170)
    tiles = []
    for L in [(-2.25, -1.0, 170), (-1.25, -1.0, 170), (-2.25, 0.0, 170), (-3.25, -1.0, 170)]:
        v, acc = patient_view(E, to_pv(*L), 4.6, 9.0)
        tiles.append(v)
        print(L, "accommodating", acc)
    Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(OUT, "test_views.png"))
    im, d = new_frame()
    phoropter(d, 960, 400, 1.2, 170, -1.0, jcc=215)
    im.save(os.path.join(OUT, "test_phoropter.png"))
