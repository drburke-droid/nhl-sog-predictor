"""Part 2: the Artal et al. (2004) experiment, in the Light Speed style.

Scene lengths follow the narration (part2/vo/durations.json from tts.py). Writes part2/part2_video.mp4
and part2/timeline.json (for audio.py).

    python3 part2.py stills      # one still per scene -> part2/stills/
    python3 part2.py video
"""
import os
import sys
import json
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo1_v2 as D                                       # noqa: E402  (shared style: palette, fonts, bloom)
import optics as O                                          # noqa: E402

BASE = D.BASE.rsplit("/", 1)[0]
P2 = f"{BASE}/part2"
W, H, FPS = D.W, D.H, D.FPS
F = D.F
INK, MUTED, DIM, GOLD8 = D.INK, D.MUTED, D.DIM, D.GOLD8
GOLDT = (255, 190, 90)
XFADE = 0.5


def ease(t, a, b):
    if b <= a:
        return 1.0 if t >= b else 0.0
    x = min(max((t - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def to_tile(rgb, size):
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255


def paste(img, tile, x, y, a=1.0):
    h, w = tile.shape[:2]
    if a <= 0:
        return
    reg = img[y:y + h, x:x + w]
    img[y:y + h, x:x + w] = reg * (1 - a) + tile[..., :3] * a


def padd(img, tile, x, y, a=1.0):
    """additive paste for glowing PSF tiles (black = transparent)"""
    h, w = tile.shape[:2]
    if a > 0:
        img[y:y + h, x:x + w] += tile[..., :3] * a


def framed(em, x, y, w, h, col, b=3, k=1.0):
    D.frame_rect(em, x, y, w, h, b, col, k)


def base_bg(card=False, lit=0.0):
    if card:
        return D.card_base(lit)
    return (D.BG + D.RING_DEMO * np.array([0.25, 0.45, 0.75], np.float32)).astype(np.float32)


def header(d, title, sub=None):
    d.text((58, 38), title, font=F["title"], fill=INK)
    if sub:
        d.text((60, 90), sub, font=F["sub"], fill=MUTED)


def label(d, xy, s, font="sub", fill=MUTED, anchor="mm"):
    d.text(xy, s, font=F[font], fill=fill, anchor=anchor)


def text_layer(fn):
    return D.overlay(fn)


def comp(img, ov, a=1.0):
    rgb, al = ov
    al = al * a
    return img * (1 - al) + rgb * al


# ------------------------------------------------------------------ precomputed optics
class Assets:
    def __init__(self):
        e = O.EYES
        self.WA = O.wavefront(e["A"])
        self.psfA = O.psf(self.WA, 16)
        self.tiles = {}
        self.psf_rgb = {k: O.heat(O.psf(O.wavefront(e[k]), 16)) for k in "ABC"}
        self.uncorr = O.heat(O.psf(O.wavefront({**e["A"], **O.LOW}), 30))
        self.withg = O.heat(O.psf(self.WA, 30))
        self.rot = {a: O.heat(O.psf(O.wavefront(e["A"], a), 16)) for a in range(0, 360, 45)}
        self.diff = O.heat(O.psf(np.zeros_like(self.WA), 16))
        stim = O.binary_noise(lo=0.012, hi=0.035, seed=3)
        self.stim = stim
        self.Fgrid = np.round(np.arange(0.60, 1.21, 0.02), 2)
        self.own = O.blur_with(stim, self.WA, 0.1)
        self.rotF = {f: O.blur_with(stim, O.wavefront(e["A"], 90, f), 0.1) for f in self.Fgrid}
        self.rot180 = O.blur_with(stim, O.wavefront(e["A"], 180), 0.1)
        # twinkling star: low-order turbulence frames, looped
        rng = np.random.default_rng(4)
        modes = [(2, -2), (2, 0), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (1, -1), (1, 1)]
        c = rng.normal(0, 0.25, (len(modes),))
        self.twinkle = []
        for i in range(48):
            c = 0.86 * c + rng.normal(0, 0.12, c.shape)
            Wt = O.wavefront({m: float(v) for m, v in zip(modes, c)})
            self.twinkle.append(O.heat(O.psf(Wt, 16), gain=1.6))
        self.spots = O.lenslet_spots(self.WA, gain=900.0)
        self.acts = O.actuators()
        self.act_w = O.sample(self.WA, [p[0] for p in self.acts], [p[1] for p in self.acts])
        self.corr_rgb = [O.heat(O.psf(self.WA * (1 - p), 16)) for p in np.linspace(0, 1, 25)]
        self.wmap = O.diverge(self.WA)

    def tile(self, key, rgb, size):
        k = (key, size)
        if k not in self.tiles:
            self.tiles[k] = to_tile(rgb, size)
        return self.tiles[k]


def gray_tile(x, size):
    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    g = np.asarray(im, np.float32)[..., None] / 255
    return np.repeat(g, 3, 2)


# ------------------------------------------------------------------ scenes
# each scene: fn(t, dur, A) -> float image. Static text layers are cached by key.
_cache = {}


def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def hud(img, label_s, frac):
    def fn():
        return None
    ov = D.overlay(lambda d: (
        d.text((W - 60, 1018), label_s, font=F["small"], fill=(130, 150, 175), anchor="ra"),
        d.rectangle([W - 360, 1040, W - 60, 1042], fill=(40, 55, 75)),
        d.rectangle([W - 360, 1040, W - 360 + int(300 * frac), 1042], fill=GOLD8)))
    return comp(img, ov)


def s01(t, dur, A):
    img = cached("s01bg", lambda: base_bg(True, 1.0)).copy()
    em = cached("s01em", lambda: D.bloom(D.glow_text(lambda d: d.text((110, 430), "THE EXPERIMENT", font=F["h1"],
                                                                         fill=(120, 70, 15)))) * 1.2)
    img = img + em * ease(t, 0.4, 1.6)
    ov = cached("s01t", lambda: text_layer(lambda d: (
        d.text((114, 360), "PART 2", font=F["step"], fill=GOLD8),
        d.text((110, 430), "THE EXPERIMENT", font=F["h1"], fill=GOLDT),
        d.text((114, 590), "Your own blur vs. the exact same blur, rotated", font=F["body"], fill=INK),
        d.text((114, 650), "Artal, Chen, Fernández, Singer, Manzanera & Williams  ·  Journal of Vision, 2004",
               font=F["sub"], fill=MUTED))))
    return comp(img, ov, ease(t, 0.2, 1.4))


def s02(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    if u < 0.62:
        # a point of light becomes a smudge
        m = ease(u, 0.18, 0.40)
        big = A.tile("A560", A.psf_rgb["A"], 560)
        dot = np.zeros_like(img)
        D.add_dot(dot, True, 1.0, W // 2, 560, 1.4)
        img = img + dot * (1 - m) * ease(u, 0.0, 0.06)
        paste(img, big, W // 2 - 280, 280, m)
        ov = cached("s02a", lambda: text_layer(lambda d: (
            header(d, "Every eye blurs light in its own way", "A single point of light, through one (simulated) eye"),
            label(d, (W // 2, 900), "a point of light…", "step", INK))))
        img = comp(img, ov, 1 - ease(u, 0.36, 0.42))
        ov2 = cached("s02b", lambda: text_layer(lambda d: (
            header(d, "Every eye blurs light in its own way", "A single point of light, through one (simulated) eye"),
            label(d, (W // 2, 900), "…lands as a small, uneven smudge", "step", GOLDT))))
        img = comp(img, ov2, ease(u, 0.38, 0.46))
    else:
        k = ease(u, 0.62, 0.75)
        for i, key in enumerate("ABC"):
            x = 160 + i * 560
            a = 1.0 if key == "A" else ease(u, 0.66 + 0.06 * i, 0.74 + 0.06 * i)
            padd(img, A.tile(key + "460", A.psf_rgb[key], 460), x, 300, a * (k if key != "A" else 1))
        ov = cached("s02c", lambda: text_layer(lambda d: (
            header(d, "A blur fingerprint", "Three simulated eyes, each with typical higher-order aberrations"),
            *[label(d, (160 + i * 560 + 230, 810), f"Eye {k}", "step", GOLDT if k == "A" else INK) for i, k in enumerate("ABC")],
            label(d, (W // 2, 900), "Point-spread functions, 16 arcmin across, 6 mm pupil", "sub", DIM))))
        img = comp(img, ov, k)
    return img


def s03(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    padd(img, A.tile("unc", A.uncorr, 560), 300, 250, ease(u, 0.0, 0.1))
    padd(img, A.tile("wg", A.withg, 560), 1060, 250, ease(u, 0.38, 0.5))
    ov = cached("s03a", lambda: text_layer(lambda d: (
        header(d, "Glasses fix the big errors, not the fine ones", "Same simulated eye, 30 arcmin view"),
        label(d, (580, 860), "Uncorrected", "step", INK),
        label(d, (580, 910), "nearsightedness + astigmatism + the rest", "sub", MUTED))))
    img = comp(img, ov)
    ov2 = cached("s03b", lambda: text_layer(lambda d: (
        label(d, (1340, 860), "With glasses", "step", GOLDT),
        label(d, (1340, 910), "higher-order aberrations remain", "sub", MUTED))))
    return comp(img, ov2, ease(u, 0.42, 0.54))


def starfield():
    rng = np.random.default_rng(9)
    img = base_bg().copy() * 0.7
    for _ in range(420):
        x, y = rng.integers(0, W), rng.integers(140, H - 80)
        b = rng.uniform(0.15, 0.7) ** 2
        img[y, x] += b
        if b > 0.3:
            img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] += b * 0.25
    return img


def s04(t, dur, A):
    img = cached("stars", starfield).copy()
    u = t / dur
    on = ease(u, 0.55, 0.68)
    fr = A.twinkle[int(t * 12) % len(A.twinkle)]
    tw = to_tile(fr, 560)
    sharp = A.tile("diff560", A.diff, 560)
    tile = tw * (1 - on) + sharp * on
    x, y = W // 2 - 280, 250
    reg = img[y:y + 560, x:x + 560]
    img[y:y + 560, x:x + 560] = np.maximum(reg, tile)
    ov = cached("s04a", lambda: text_layer(lambda d: (
        header(d, "Adaptive optics: borrowed from astronomy", "Telescopes use it to undo the shimmer of the atmosphere"),
        label(d, (W // 2, 880), "atmosphere: the star shimmers", "step", INK))))
    img = comp(img, ov, 1 - on)
    ov2 = cached("s04b", lambda: text_layer(lambda d: (
        header(d, "Adaptive optics: borrowed from astronomy", "Telescopes use it to undo the shimmer of the atmosphere"),
        label(d, (W // 2, 880), "ADAPTIVE OPTICS ON", "step", GOLDT))))
    return comp(img, ov2, on)


def s05(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    p = ease(u, 0.42, 0.80)
    S = 440
    xs = [150, 740, 1330]
    y0 = 270
    # 1: wavefront sensor (lenslet spots shift with the local slope; shrink as the mirror corrects)
    sens = np.zeros((S, S, 3), np.float32)
    sens += to_tile(A.wmap, S) * 0.35 * (1 - p)
    im = Image.fromarray((np.clip(sens, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    jit = 0.06 * math.sin(t * 2 * math.pi * 3)
    for (x, y, dx, dy) in A.spots:
        cx, cy = (x + 1) / 2 * S, (y + 1) / 2 * S
        d.line((cx - 4, cy, cx + 4, cy), fill=(60, 80, 105))
        d.line((cx, cy - 4, cx, cy + 4), fill=(60, 80, 105))
        k = (1 - p) * (1 + jit)
        sx, sy = cx + dx * k, cy + dy * k
        d.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(120, 200, 255))
    d.ellipse((2, 2, S - 3, S - 3), outline=(70, 110, 150), width=2)
    paste(img, np.asarray(im, np.float32) / 255, xs[0], y0)
    # 2: deformable mirror (actuators push to the opposite shape)
    mir = Image.new("RGB", (S, S), (6, 10, 18))
    d = ImageDraw.Draw(mir)
    lim = np.abs(A.act_w).max()
    for (x, y), w in zip(A.acts, A.act_w):
        cx, cy = (x * 0.92 + 1) / 2 * S, (y * 0.92 + 1) / 2 * S
        v = -w / lim * p
        col = (np.array([1.0, 0.62, 0.18]) if v > 0 else np.array([0.16, 0.62, 0.98])) * abs(v) + 0.10 * (1 - abs(v))
        r = 13
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=tuple(int(255 * c) for c in np.clip(col, 0, 1)),
                  outline=(80, 100, 125))
    d.ellipse((2, 2, S - 3, S - 3), outline=(70, 110, 150), width=2)
    paste(img, np.asarray(mir, np.float32) / 255, xs[1], y0)
    # 3: what the eye sees
    idx = int(round(p * (len(A.corr_rgb) - 1)))
    padd(img, to_tile(A.corr_rgb[idx], S), xs[2], y0)
    ov = cached("s05a", lambda: text_layer(lambda d: (
        header(d, "The machine: measure the blur, then cancel it", "University of Rochester adaptive-optics system (schematic)"),
        label(d, (xs[0] + S // 2, y0 + S + 50), "Wavefront sensor", "step", INK),
        label(d, (xs[0] + S // 2, y0 + S + 100), "measures the blur 30× a second", "sub", MUTED),
        label(d, (xs[1] + S // 2, y0 + S + 50), "Deformable mirror", "step", INK),
        label(d, (xs[1] + S // 2, y0 + S + 100), "97 actuators bend to the opposite shape", "sub", MUTED),
        label(d, (xs[2] + S // 2, y0 + S + 50), "What the eye sees", "step", GOLDT),
        label(d, (xs[2] + S // 2, y0 + S + 100), "a point of light, almost perfect", "sub", MUTED),
        *[d.text((xs[i] + S + 30, y0 + S // 2), "→", font=F["h2"], fill=DIM, anchor="mm") for i in (0, 1)])))
    img = comp(img, ov)
    ov2 = D.overlay(lambda d: d.text((W - 60, 210), f"correction {int(round(p * 100))}%", font=F["bodyb"],
                                     fill=GOLDT if p > 0.99 else MUTED, anchor="ra"))
    return comp(img, ov2, ease(u, 0.38, 0.45))


def s06(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    cx, cy, R, S0, S1 = W // 2, 575, 365, 270, 190
    em = cached("s06em", lambda: _s06_em(cx, cy, S0))
    img = img + em * ease(u, 0.0, 0.1)
    padd(img, A.tile("rot0", A.rot[0], S0), cx - S0 // 2, cy - S0 // 2, ease(u, 0.0, 0.1))
    angs = list(range(45, 360, 45))
    for i, a in enumerate(angs):
        th = math.radians(-90 + a)
        x, y = int(cx + R * math.cos(th)) - S1 // 2, int(cy + R * math.sin(th)) - S1 // 2
        al = ease(u, 0.42 + 0.06 * i, 0.48 + 0.06 * i)
        padd(img, A.tile(f"rot{a}", A.rot[a], S1), x, y, al)
    ov = cached("s06a", lambda: text_layer(lambda d: (
        header(d, "Then: put the blur back — or rotate it", "The mirror recreated each person's own blur, or a rotated copy"),
        label(d, (cx, cy + S0 // 2 + 26), "YOUR OWN BLUR", "small", GOLDT))))
    img = comp(img, ov)
    for i, a in enumerate(angs):
        th = math.radians(-90 + a)
        lx, ly = int(cx + (R + S1 * 0.5 + 40) * math.cos(th)), int(cy + (R + S1 * 0.5 + 20) * math.sin(th))
        al = ease(u, 0.44 + 0.06 * i, 0.5 + 0.06 * i)
        if al > 0:
            ov = cached(f"s06l{a}", lambda a=a, lx=lx, ly=ly: text_layer(
                lambda d: d.text((lx, ly), f"{a}°", font=F["bodyb"], fill=INK, anchor="mm")))
            img = comp(img, ov, al)
    return img


def _s06_em(cx, cy, S0):
    em = np.zeros((H, W, 3), np.float32)
    framed(em, cx - S0 // 2, cy - S0 // 2, S0, S0, D.GOLD, 4)
    return em + D.bloom(em)


def s07(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S = 470
    padd(img, A.tile("r0_470", A.rot[0], S), 330, 230)
    padd(img, A.tile("r90_470", A.rot[90], S), 1120, 230, ease(u, 0.05, 0.15))
    fill = ease(u, 0.3, 0.6)
    ov = cached("s07a", lambda: text_layer(lambda d: (
        header(d, "Rotated blur is just as strong", "Measured image quality (Strehl ratio) was the same at every angle"),
        label(d, (330 + S // 2, 740), "Own blur (0°)", "step", GOLDT),
        label(d, (1120 + S // 2, 740), "Rotated (90°)", "step", INK),
        label(d, (W // 2, 960), "Only the orientation changed", "bodyb", INK))))
    img = comp(img, ov)
    ov2 = D.overlay(lambda d: [
        (d.rectangle((x, 800, x + S, 820), fill=(30, 42, 60)),
         d.rectangle((x, 800, x + int(S * fill), 820), fill=GOLD8 if i == 0 else (60, 150, 240)),
         d.text((x, 845), "amount of blur", font=F["sub"], fill=MUTED, anchor="lm"),
         d.text((x + S, 845), "identical" if fill > 0.95 else "", font=F["bodyb"], fill=GOLDT, anchor="rm"))
        for i, x in enumerate((330, 1120))])
    return comp(img, ov2)


def dial_value(u):
    """the subject's search: up a little, then down to a match"""
    if u < 0.30:
        return 1.0
    if u < 0.45:
        return 1.0 + 0.12 * ease(u, 0.30, 0.45)
    if u < 0.85:
        return 1.12 - 0.40 * ease(u, 0.45, 0.85)
    return 0.72


def s08(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S = 520
    xl, xr, y0 = 300, 1100, 220
    Fv = dial_value(u)
    fk = A.Fgrid[np.argmin(np.abs(A.Fgrid - Fv))]
    flash_l = 1.0 if int(t / 0.5) % 2 == 0 else 0.8          # alternate 500 ms presentations
    paste(img, gray_tile(A.own, S) * flash_l, xl, y0)
    paste(img, gray_tile(A.rotF[fk], S) * (1.8 - flash_l), xr, y0)
    ov = cached("s08a", lambda: text_layer(lambda d: (
        header(d, "The task: match the blur", "5 people · binary-noise pattern · 500 ms flashes · 6 mm pupil"),
        label(d, (xl + S // 2, y0 + S + 45), "Own blur", "step", GOLDT),
        label(d, (xr + S // 2, y0 + S + 45), "Rotated blur  × dial", "step", INK),
        label(d, (xl + S // 2, y0 + S + 92), "fixed at 100%", "sub", MUTED))))
    img = comp(img, ov)
    # the dial
    ov2 = D.overlay(lambda d: _dial(d, xr + S // 2, y0 + S + 150, Fv))
    return comp(img, ov2)


def _dial(d, cx, cy, Fv):
    w = 420
    x0 = cx - w // 2
    lo, hi = 0.5, 1.25
    d.rectangle((x0, cy, x0 + w, cy + 6), fill=(30, 42, 60))
    x100 = x0 + int((1.0 - lo) / (hi - lo) * w)
    d.line((x100, cy - 14, x100, cy + 20), fill=(110, 130, 155), width=2)
    xv = x0 + int((Fv - lo) / (hi - lo) * w)
    d.rectangle((x0, cy, xv, cy + 6), fill=GOLD8)
    d.ellipse((xv - 11, cy - 8, xv + 11, cy + 14), fill=(255, 220, 160))
    d.text((cx, cy + 50), f"rotated blur strength: {int(round(Fv * 100))}%", font=F["bodyb"], fill=INK, anchor="mm")


def s09(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    x0, x1, y = 260, 1660, 600
    lo, hi = 0.0, 1.2

    def X(v):
        return int(x0 + (v - lo) / (hi - lo) * (x1 - x0))

    em = np.zeros((H, W, 3), np.float32)
    band = ease(u, 0.40, 0.62)
    if band > 0:
        em[y - 40:y + 40, X(0.6):X(0.6) + int((X(0.8) - X(0.6)) * band)] += D.GOLD * 0.8
    img = img + em + D.bloom(em)
    ov = cached("s09a", lambda: text_layer(lambda d: (
        header(d, "The result", "Where the rotated blur had to be set to look as blurry as each person's own"),
        d.rectangle((x0, y - 3, x1, y + 3), fill=(50, 66, 90)),
        *[(d.line((X(v), y - 12, X(v), y + 12), fill=(110, 130, 155), width=2),
           d.text((X(v), y + 70), f"{int(v * 100)}%", font=F["sub"], fill=MUTED, anchor="mm")) for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)],
        d.line((X(1.0), y - 120, X(1.0), y + 40), fill=(150, 176, 205), width=3),
        d.text((X(1.0), y - 150), "if all blur were treated the same", font=F["sub"], fill=MUTED, anchor="mm"))))
    img = comp(img, ov, ease(u, 0.0, 0.15))
    ov2 = cached("s09b", lambda: text_layer(lambda d: (
        d.text(((X(0.6) + X(0.8)) // 2, y - 90), "all 5 people: 60–80%", font=F["step"], fill=GOLDT, anchor="mm"),
        d.text((W // 2, 850), "The rotated blur had to be turned DOWN by 20–40%", font=F["h2"], fill=INK, anchor="mm"),
        d.text((W // 2, 930), "to look as blurry as their own. 5 out of 5.", font=F["body"], fill=MUTED, anchor="mm"))))
    return comp(img, ov2, ease(u, 0.55, 0.68))


def s10(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S = 560
    paste(img, gray_tile(A.own, S), 290, 230)
    paste(img, gray_tile(A.rotF[1.0], S), 1070, 230, ease(u, 0.0, 0.1))
    ov = cached("s10a", lambda: text_layer(lambda d: (
        header(d, "Same blur, just turned — and it looked worse", "Equal physical blur on both sides (simulated)"),
        label(d, (290 + S // 2, 840), "Own blur", "step", GOLDT),
        label(d, (290 + S // 2, 890), "looked sharper", "bodyb", GOLDT),
        label(d, (1070 + S // 2, 840), "Rotated 90°", "step", INK),
        label(d, (1070 + S // 2, 890), "looked blurrier", "bodyb", MUTED))))
    return comp(img, ov, ease(u, 0.05, 0.2))


def s11(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S = 470
    padd(img, A.tile("r0_470", A.rot[0], S), 330, 220)
    padd(img, A.tile("r180_470", A.rot[180], S), 1120, 220, ease(u, 0.05, 0.15))
    ov = cached("s11a", lambda: text_layer(lambda d: (
        header(d, "Even a half-turn looked worse", "A 180° rotation removes exactly the same contrast at every detail size"),
        label(d, (330 + S // 2, 730), "Own blur (0°)", "step", GOLDT),
        label(d, (1120 + S // 2, 730), "Half-turn (180°)", "step", INK))))
    img = comp(img, ov)
    ov2 = cached("s11b", lambda: text_layer(lambda d: (
        d.text((W // 2, 850), "Not just HOW MUCH your eye blurs.", font=F["h2"], fill=INK, anchor="mm"),
        d.text((W // 2, 935), "The brain is tuned to the SHAPE of your blur.", font=F["h2"], fill=GOLDT, anchor="mm"))))
    return comp(img, ov2, ease(u, 0.45, 0.58))


def s12(t, dur, A):
    img = base_bg(True, 0.35).copy()
    u = t / dur
    S = 420
    padd(img, A.tile("A420", A.psf_rgb["A"], S), 260, 280)
    crisp = O.heat(O.psf(A.WA * 0.45, 16))
    padd(img, cached("crisp", lambda: to_tile(crisp, S)), 1240, 280, ease(u, 0.25, 0.4))
    ov = cached("s12a", lambda: text_layer(lambda d: (
        header(d, "Your brain quietly compensates", "Illustration"),
        label(d, (260 + S // 2, 760), "what your optics deliver", "step", INK),
        d.text((W // 2, 490), "→", font=F["h1"], fill=DIM, anchor="mm"),
        d.text((W // 2, 590), "your brain", font=F["sub"], fill=MUTED, anchor="mm"))))
    img = comp(img, ov)
    ov2 = cached("s12b", lambda: text_layer(lambda d: (
        label(d, (1240 + S // 2, 760), "what you perceive: sharper", "step", GOLDT),
        d.text((W // 2, 900), "You see more clearly than your optics alone should allow.", font=F["h2"], fill=INK,
               anchor="mm"))))
    return comp(img, ov2, ease(u, 0.3, 0.45))


def s13(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S = 340
    xs = [180, 790, 1400]
    padd(img, A.tile("A340", A.psf_rgb["A"], S), xs[0], 300)
    padd(img, A.tile("d340", A.diff, S), xs[1], 300, ease(u, 0.15, 0.28))
    ov = cached("s13a", lambda: text_layer(lambda d: (
        header(d, "Why new corrections can take time", "e.g. custom laser surgery or new lenses that remove old aberrations"),
        label(d, (xs[0] + S // 2, 700), "your old blur", "step", INK),
        label(d, (xs[0] + S // 2, 745), "the brain is tuned to it", "sub", MUTED))))
    img = comp(img, ov)
    ov2 = cached("s13b", lambda: text_layer(lambda d: (
        label(d, (xs[1] + S // 2, 700), "new, near-perfect optics", "step", GOLDT),
        label(d, (xs[1] + S // 2, 745), "physically sharper", "sub", MUTED))))
    img = comp(img, ov2, ease(u, 0.18, 0.3))
    k = ease(u, 0.45, 0.6)
    ov3 = D.overlay(lambda d: (
        d.text((xs[2] + S // 2, 470), "?", font=F["big"], fill=GOLDT, anchor="mm"),
        d.text((xs[2] + S // 2, 700), "brain still compensating", "step", INK, anchor="mm") if False else
        d.text((xs[2] + S // 2, 700), "brain still compensating", font=F["step"], fill=INK, anchor="mm"),
        d.text((xs[2] + S // 2, 745), "for the old blur", font=F["sub"], fill=MUTED, anchor="mm"),
        d.text((W // 2, 900), "“Some benefits … might be undone”  — Artal et al., 2004", font=F["body"], fill=MUTED,
               anchor="mm")))
    return comp(img, ov3, k)


def s14(t, dur, A):
    img = base_bg(True, 0.2).copy()
    u = t / dur
    th = cached("p1thumb", lambda: _p1_thumb())
    paste(img, th, (W - th.shape[1]) // 2, 200, ease(u, 0.0, 0.12))
    ov = cached("s14a", lambda: text_layer(lambda d: (
        header(d, "You just felt the fast version", "Part 1, one minute of blur"),
        d.text((W // 2, 880), "One minute of blur, and your sense of “sharp” shifted.", font=F["h2"], fill=INK, anchor="mm"),
        d.text((W // 2, 965), "Your brain is re-tuning all the time.", font=F["h2"], fill=GOLDT, anchor="mm"))))
    return comp(img, ov, ease(u, 0.15, 0.3))


def _p1_thumb():
    p = f"{D.STILLS}/4_adapt.png"
    im = Image.open(p).convert("RGB").resize((1080, 608), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255


def s15(t, dur, A):
    img = cached("s15bg", lambda: base_bg(True, 1.0)).copy()
    ov = cached("s15a", lambda: text_layer(lambda d: (
        d.text((110, 330), "“Sharp” is learned.", font=F["h2"], fill=GOLDT),
        d.text((112, 450), "Artal P, Chen L, Fernández EJ, Singer B, Manzanera S, Williams DR (2004).", font=F["body"], fill=INK),
        d.text((112, 500), "Neural compensation for the eye’s optical aberrations. Journal of Vision 4(4): 281–287.",
               font=F["body"], fill=INK),
        d.text((112, 580), "Webster MA, Georgeson MA, Webster SM (2002). Neural adjustments to image blur.",
               font=F["sub"], fill=MUTED),
        d.text((112, 615), "Nature Neuroscience 5: 839–840.", font=F["sub"], fill=MUTED),
        d.text((112, 700), "Eye blur shown here is simulated (illustrative aberrations, computed with Zernike optics).",
               font=F["sub"], fill=DIM),
        d.text((112, 735), ("Photos in Part 1: NASA." if D.DEMO_SET != "fire" else "Photos in Part 1: public domain (U.S. government, Library of Congress).") + " Voice: AI. Music: original synthesized score.", font=F["sub"], fill=DIM))))
    return comp(img, ov, ease(t, 0.2, 1.2))


SCENES = [("s01", s01, "THE EXPERIMENT"), ("s02", s02, "BLUR FINGERPRINTS"), ("s03", s03, "WHAT GLASSES FIX"),
          ("s04", s04, "ADAPTIVE OPTICS"), ("s05", s05, "THE MACHINE"), ("s06", s06, "ROTATED BLUR"),
          ("s07", s07, "SAME STRENGTH"), ("s08", s08, "THE TASK"), ("s09", s09, "THE RESULT"),
          ("s10", s10, "THE RESULT"), ("s11", s11, "THE HALF-TURN"), ("s12", s12, "WHAT IT MEANS"),
          ("s13", s13, "WHAT IT MEANS"), ("s14", s14, "BACK TO PART 1"), ("s15", s15, "")]


def timeline():
    durs = json.load(open(f"{P2}/vo/durations.json"))
    t = 0.0
    out = []
    for sid, fn, lab in SCENES:
        if sid in durs:
            d = durs[sid]["seconds"] + 1.3
            out.append(dict(id=sid, start=round(t, 3), dur=round(d, 3), vo=True, vo_start=round(t + 0.5, 3), label=lab))
        else:
            d = 8.0
            out.append(dict(id=sid, start=round(t, 3), dur=d, vo=False, label=lab))
        t += d
    total = round(t, 3)
    S = {s["id"]: s for s in out}

    def span(a, b):
        return S[a]["start"], S[b]["start"] + S[b]["dur"]

    music = []
    for (a, b), chords, dens in (((("s01", "s01")), ["F", "Bb"], 0.12),
                                 (("s02", "s03"), ["Dm9", "Bb", "F/A", "Gm"], 0.20),
                                 (("s04", "s05"), ["Am", "Bb", "Dm9", "Csus"], 0.24),
                                 (("s06", "s07"), ["F", "C", "Dm9", "Bb"], 0.22),
                                 (("s08", "s08"), ["Gm", "Csus"], 0.10),
                                 (("s09", "s11"), ["Fadd9", "Am", "Bb", "Dm9", "Gm", "Csus"], 0.24),
                                 (("s12", "s14"), ["Bb", "F/A", "Gm", "Dm9", "Bb", "Csus"], 0.22)):
        t0, t1 = span(a, b)
        music.append(dict(t0=t0, t1=t1, chords=chords, dens=dens))
    tl = dict(total=total, scenes=out, music=music, chimes=[S["s09"]["start"] + 0.55 * S["s09"]["dur"]],
              final=S["s15"]["start"])
    json.dump(tl, open(f"{P2}/timeline.json", "w"), indent=1)
    return tl


def frames(tl, A):
    fns = {sid: fn for sid, fn, _ in SCENES}
    total = tl["total"]
    prev_last = None
    for sc in tl["scenes"]:
        n = int(round(sc["dur"] * FPS))
        nx = int(XFADE * FPS)
        last = None
        for i in range(n):
            t = i / FPS
            img = fns[sc["id"]](t, sc["dur"], A)
            if sc["label"]:
                img = hud(img, f"PART 2  ·  {sc['label']}", (sc["start"] + t) / total)
            if prev_last is not None and i < nx:
                a = (i + 1) / (nx + 1)
                img = prev_last * (1 - a) + img * a
            last = img
            yield img
        prev_last = last


def main(mode):
    tl = timeline()
    A = Assets()
    if mode == "stills":
        os.makedirs(f"{P2}/stills", exist_ok=True)
        fns = {sid: fn for sid, fn, _ in SCENES}
        for sc in tl["scenes"]:
            for frac in (0.3, 0.85):
                img = fns[sc["id"]](frac * sc["dur"], sc["dur"], A)
                if sc["label"]:
                    img = hud(img, f"PART 2  ·  {sc['label']}", 0.5)
                Image.fromarray(D.to8(img)).save(f"{P2}/stills/{sc['id']}_{int(frac * 100)}.png")
        print("total", tl["total"])
        return
    out = f"{P2}/part2_video.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img in frames(tl, A):
        p.stdin.write(D.to8(img).tobytes())
    p.stdin.close()
    p.wait()
    print(out, tl["total"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "video")
