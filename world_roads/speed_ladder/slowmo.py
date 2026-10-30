"""The slow-motion sign, drawn on the finished 1920x1080 frame: in the black band above the picture, a stopwatch, how
many times slower than real life the picture is running (or REAL SPEED), the frame rate a camera would need, and a
clock that runs at real time, so you can watch it crawl; and when the picture slows, a big "48x SLOWER" across it
for a moment."""
import math
import numpy as np
from PIL import Image, ImageDraw
import comp as C

ACCENT = (255, 190, 40)
WHITE, DIM, RED = (255, 255, 255), (200, 206, 214), (255, 70, 60)
BAND = C.TOP                                # the black band above the picture


def _stopwatch(d, cx, cy, r, col, frac, alpha):
    """a stopwatch r px across the face, its hand at frac of a turn"""
    a = int(255 * alpha)
    w = max(3, r // 7)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col + (a,), width=w)
    d.rectangle((cx - r * 0.22, cy - r * 1.42, cx + r * 0.22, cy - r * 1.12), fill=col + (a,))
    d.line((cx + r * 0.78, cy - r * 0.78, cx + r * 1.0, cy - r * 1.0), fill=col + (a,), width=w)
    for i in range(12):
        ang = 2 * math.pi * i / 12
        r0 = r * (0.72 if i % 3 == 0 else 0.8)
        d.line((cx + r0 * math.sin(ang), cy - r0 * math.cos(ang), cx + r * 0.9 * math.sin(ang),
                cy - r * 0.9 * math.cos(ang)), fill=col + (a,), width=max(2, w // 2))
    ang = 2 * math.pi * frac
    d.line((cx, cy, cx + r * 0.7 * math.sin(ang), cy - r * 0.7 * math.cos(ang)), fill=col + (a,), width=w)
    d.ellipse((cx - w, cy - w, cx + w, cy + w), fill=col + (a,))


def fmt_clock(t):
    return f"{t:.2f} s" if t >= 10 else f"{t:.3f} s" if t >= 1 else f"{t * 1000:.1f} ms"


def badge(frame8, slow, t_real, alpha=1.0, fps_note=True):
    """slow: this frame's step against real time (1 real speed, 1/48 48x slower); t_real: real seconds since the
    clock started"""
    im = Image.fromarray(frame8).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a = int(255 * alpha)
    factor = 1 / slow
    is_slow = factor > 1.05
    col = ACCENT if is_slow else WHITE
    cy = BAND // 2 + 4
    # left: the stopwatch and the clock at real time (a turn of the hand every second of real time)
    _stopwatch(d, 70, cy, 30, col, t_real % 1.0, alpha)
    f1, f2 = C.font(44, "Bold"), C.font(22, "Medium")
    d.text((122, cy - 34), fmt_clock(t_real), font=f1, fill=WHITE + (a,))
    d.text((124, cy + 16), "real time gone by", font=f2, fill=DIM + (a,))
    # right: the rate
    fb, fs, fn = C.font(92, "Black"), C.font(34, "Bold"), C.font(22, "Medium")
    xr = C.W - 56
    if is_slow:
        big = f"{factor:.0f}×" if factor >= 9.95 else f"{factor:.1f}×"     # the ramps: 1.3x, not "1x slower"
        wb = d.textlength(big, font=fb)
        ws = max(d.textlength("SLOWER", font=fs), d.textlength("THAN REAL", font=fn))
        x0 = xr - ws - 18 - wb
        d.text((x0, cy - 62), big, font=fb, fill=ACCENT + (a,))
        d.text((x0 + wb + 18, cy - 40), "SLOWER", font=fs, fill=ACCENT + (a,))
        d.text((x0 + wb + 20, cy + 2), "THAN REAL", font=fn, fill=DIM + (a,))
        if fps_note:
            note = f"a camera at {25 * factor:,.0f} frames a second"
            wn = d.textlength(note, font=fn)
            d.ellipse((xr - wn - 30, cy + 40, xr - wn - 16, cy + 54), fill=RED + (a,))
            d.text((xr - wn, cy + 35), note, font=fn, fill=DIM + (a,))
        # how slow, on a scale from real speed to 48x
        frac = math.log(factor) / math.log(48.0)
        bx0, bx1, by = x0 - 250, x0 - 40, cy + 2
        d.rounded_rectangle((bx0, by - 7, bx1, by + 7), 7, outline=DIM + (int(a * 0.6),), width=2)
        d.rounded_rectangle((bx0, by - 7, bx0 + max(14, (bx1 - bx0) * min(1.0, frac)), by + 7), 7, fill=ACCENT + (a,))
        d.text((bx0, by + 12), "real", font=C.font(18, "Medium"), fill=DIM + (a,))
        w48 = d.textlength("48×", font=C.font(18, "Medium"))
        d.text((bx1 - w48, by + 12), "48×", font=C.font(18, "Medium"), fill=DIM + (a,))
    else:
        txt = "REAL SPEED"
        w = d.textlength(txt, font=C.font(56, "Black"))
        d.text((xr - w, cy - 34), txt, font=C.font(56, "Black"), fill=WHITE + (a,))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def slam(frame8, factor, k, n_in=6, n_hold=22, n_out=12):
    """the big sign across the picture as it slows: k frames since it began"""
    if k < 0 or k >= n_in + n_hold + n_out: return frame8
    if k < n_in:
        s, a = 1.6 - 0.6 * (k + 1) / n_in, (k + 1) / n_in
    elif k < n_in + n_hold:
        s, a = 1.0, 1.0
    else:
        s, a = 1.0, 1 - (k - n_in - n_hold + 1) / n_out
    im = Image.fromarray(frame8).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    fb, fs = C.font(int(150 * s), "Black"), C.font(int(56 * s), "Bold")
    big, sub = f"{factor:.0f}× SLOWER", "slow motion"
    wb, ws = d.textlength(big, font=fb), d.textlength(sub, font=fs)
    cy = C.TOP + C.PH // 2 - 40
    for dx, dy in ((4, 5), (0, 4), (4, 0)):
        d.text(((C.W - wb) / 2 + dx, cy - 110 * s + dy), big, font=fb, fill=(0, 0, 0, int(120 * a)))
    d.text(((C.W - wb) / 2, cy - 110 * s), big, font=fb, fill=ACCENT + (int(255 * a),))
    d.text(((C.W - ws) / 2 + 3, cy + 70 * s + 3), sub, font=fs, fill=(0, 0, 0, int(110 * a)))
    d.text(((C.W - ws) / 2, cy + 70 * s), sub, font=fs, fill=WHITE + (int(255 * a),))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def tint(frame8, amount):
    """the picture cooled a touch while slowed: a cue that holds even when the signs are not read"""
    if amount <= 0: return frame8
    f = frame8.astype(np.float32)
    pic = f[C.TOP:C.TOP + C.PH]
    g = pic.mean(-1, keepdims=True)
    pic = pic * (1 - 0.18 * amount) + g * 0.18 * amount
    pic *= np.array([1 - 0.03 * amount, 1.0, 1 + 0.04 * amount], np.float32)
    f[C.TOP:C.TOP + C.PH] = pic
    return np.clip(f, 0, 255).astype(np.uint8)
