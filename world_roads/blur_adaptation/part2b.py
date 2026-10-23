"""Part 2 (v2): the layman cut. Every blur is shown beside a PERFECT EYE point; the experiment's
prediction comes first, then the result, then the knob that measured it.

Scene lengths follow v2/vo_p2/durations.json (tts2.py) + each line's pause. Writes v2/part2_video.mp4 and
v2/timeline.json.

    python3 part2b.py stills | video
"""
import os
import sys
import json
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part2 as P                                           # noqa: E402  (style helpers + reusable scenes)
import optics as O                                          # noqa: E402

D = P.D
F = P.F
W, H, FPS = P.W, P.H, P.FPS
INK, MUTED, DIM, GOLD8, GOLDT = P.INK, P.MUTED, P.DIM, P.GOLD8, P.GOLDT
BLUE8 = (90, 170, 245)
V2 = os.environ.get("PART2_DIR", f"{P.BASE}/v2")
CAVEAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vernier", "fonts", "Caveat.ttf")
ease, paste, padd, comp, cached = P.ease, P.paste, P.padd, P.comp, P.cached
header, label, text_layer, base_bg = P.header, P.label, P.text_layer, P.base_bg


# ------------------------------------------------------------------ extra assets
class Assets2(P.Assets):
    def __init__(self):
        super().__init__()
        self.diff30 = O.heat(O.psf(np.zeros_like(self.WA), 30))
        self.word = self._word()
        self.word_own = O.blur_with(self.word, self.WA, 0.09)
        self.perceived = O.heat(O.psf(self.WA * 0.45, 16))

    @staticmethod
    def _word(n=1600):
        im = Image.new("L", (n, n), 0)
        d = ImageDraw.Draw(im)
        d.text((n // 2, n // 2), "SHARP", font=ImageFont.truetype(f"{D.FONT_DIR}/InterDisplay-SemiBold.otf", 300),
               fill=255, anchor="mm")
        return np.asarray(im, np.float32) / 255


def word_tile(x, w=1100, h=300):
    n = x.shape[0]
    crop = x[n // 2 - 200:n // 2 + 200, n // 2 - 733:n // 2 + 733]
    im = Image.fromarray((np.clip(crop, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
    g = np.asarray(im, np.float32)[..., None] / 255
    return np.repeat(g, 3, 2) * np.array([1.0, 0.95, 0.86], np.float32)


def frames_em(key, rects):
    """cached glow frames: rects = [(x, y, w, h, colour, border)]"""
    def mk():
        em = np.zeros((H, W, 3), np.float32)
        for x, y, w, h, col, b in rects:
            P.framed(em, x, y, w, h, col, b)
        return em + D.bloom(em) * 0.8
    return cached(key, mk)


WHITE = np.array([0.80, 0.86, 0.95], np.float32)
BLUE = np.array([0.16, 0.58, 0.95], np.float32)


def perfect_panel(img, A, x, y, S, a=1.0, crop=16):
    padd(img, A.tile(f"diff{crop}_{S}", A.diff if crop == 16 else A.diff30, S), x, y, a)


# ------------------------------------------------------------------ chapter A: your blur
def title_card(t, small, big, sub, key):
    img = cached(key + "bg", lambda: base_bg(True, 1.0)).copy()
    em = cached(key + "em", lambda: D.bloom(D.glow_text(lambda d: [d.text((110, 430 + 120 * i), s, font=F["h1"],
                                                                          fill=(120, 70, 15)) for i, s in enumerate(big)])) * 1.2)
    img = img + em * ease(t, 0.4, 1.6)
    ov = cached(key + "t", lambda: text_layer(lambda d: (
        d.text((114, 360), small, font=F["step"], fill=GOLD8),
        *[d.text((110, 430 + 120 * i), s, font=F["h1"], fill=GOLDT) for i, s in enumerate(big)],
        *[d.text((114, 470 + 120 * len(big) + 56 * i), s, font=F["body"] if i == 0 else F["sub"],
                 fill=INK if i == 0 else MUTED) for i, s in enumerate(sub)])))
    return comp(img, ov, ease(t, 0.2, 1.4))


def A1(t, dur, A):
    return title_card(t, "PART 2", ["WHAT YOUR BLUR", "LOOKS LIKE"], ["The blur inside your own eye, and why you don't notice it"], "A1")


def A2(t, dur, A):
    img = base_bg().copy()
    S, x, y = 560, W // 2 - 280, 230
    img += frames_em("A2f", [(x, y, S, S, WHITE, 2)])
    perfect_panel(img, A, x, y, S, ease(t, 0.8, 2.0))
    ov = cached("A2t", lambda: text_layer(lambda d: (
        header(d, "A perfect eye", "One star → one tiny, crisp point of light on the retina"),
        label(d, (W // 2, y + S + 50), "PERFECT EYE", "step", INK),
        label(d, (W // 2, y + S + 98), "what the eye's optics would deliver if they were flawless", "sub", MUTED))))
    return comp(img, ov)


def two_up(img, A, u, slide_from=0.0, right="psf"):
    """PERFECT EYE (left) and YOUR EYE (right), right panel slides in"""
    S, xl, xr, y = 560, 300, 1060, 230
    k = ease(u, slide_from, slide_from + 0.12) if slide_from > 0 else 1.0
    off = int((1 - k) * 120)
    img += frames_em("two_f_l", [(xl, y, S, S, WHITE, 2)])
    perfect_panel(img, A, xl, y, S)
    if k > 0:
        em = frames_em("two_f_r", [(xr, y, S, S, D.GOLD, 3)])
        img += em * k
        padd(img, A.tile("A560", A.psf_rgb["A"], S), xr + off, y, k)
    return S, xl, xr, y, k


def A3(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S, xl, xr, y, k = two_up(img, A, u, slide_from=0.12)
    ov = cached("A3t", lambda: text_layer(lambda d: (
        header(d, "A real eye", "The same starlight lands as a small, uneven smudge (simulated eye)"),
        label(d, (xl + S // 2, y + S + 50), "PERFECT EYE", "step", INK),
        label(d, (xl + S // 2, y + S + 98), "one crisp point", "sub", MUTED))))
    img = comp(img, ov)
    ov2 = cached("A3t2", lambda: text_layer(lambda d: (
        label(d, (xr + S // 2, y + S + 50), "YOUR EYE", "step", GOLDT),
        label(d, (xr + S // 2, y + S + 98), "the same star: a smudge", "sub", MUTED))))
    return comp(img, ov2, k)


def A4(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    w, h, x = 1100, 300, (W - 1100) // 2 + 120
    y1, y2 = 200, 600
    img += frames_em("A4f", [(x, y1, w, h, WHITE, 2), (x, y2, w, h, D.GOLD, 3)])
    paste(img, cached("A4w0", lambda: word_tile(A.word)), x, y1)
    paste(img, cached("A4w1", lambda: word_tile(A.word_own)), x, y2, ease(u, 0.25, 0.4))
    ov = cached("A4t", lambda: text_layer(lambda d: (
        header(d, "Your blur is on everything you see", "Every point of light gets the same smudge (simulated eye)"),
        d.text((x - 40, y1 + h // 2 - 18), "PERFECT", font=F["step"], fill=INK, anchor="rm"),
        d.text((x - 40, y1 + h // 2 + 26), "EYE", font=F["step"], fill=INK, anchor="rm"),
        d.text((x - 40, y2 + h // 2 - 18), "YOUR", font=F["step"], fill=GOLDT, anchor="rm"),
        d.text((x - 40, y2 + h // 2 + 26), "EYE", font=F["step"], fill=GOLDT, anchor="rm"))))
    return comp(img, ov)


def A5(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    Sp, S = 300, 400
    xp, yp = 110, 330
    xs = [500, 960, 1420]
    y = 280
    img += frames_em("A5f", [(xp, yp, Sp, Sp, WHITE, 2)] + [(x, y, S, S, D.GOLD, 2) for x in xs])
    perfect_panel(img, A, xp, yp, Sp)
    for i, k in enumerate("ABC"):
        padd(img, A.tile(k + "400", A.psf_rgb[k], S), xs[i], y, ease(u, 0.30 + 0.08 * i, 0.38 + 0.08 * i))
    ov = cached("A5t", lambda: text_layer(lambda d: (
        header(d, "Everyone's smudge is different", "Three simulated eyes, each with its own blur fingerprint"),
        label(d, (xp + Sp // 2, yp + Sp + 45), "PERFECT EYE", "step", INK),
        label(d, (xp + Sp // 2, yp + Sp + 90), "the same for everyone", "sub", MUTED),
        *[label(d, (xs[i] + S // 2, y + S + 45), f"EYE {k}", "step", GOLDT) for i, k in enumerate("ABC")])))
    return comp(img, ov)


def A6(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S, y = 440, 270
    xs = [150, 740, 1330]
    img += frames_em("A6f", [(xs[0], y, S, S, D.GOLD, 2), (xs[1], y, S, S, D.GOLD, 3), (xs[2], y, S, S, WHITE, 2)])
    padd(img, A.tile("unc440", A.uncorr, S), xs[0], y)
    padd(img, A.tile("wg440", A.withg, S), xs[1], y, ease(u, 0.15, 0.3))
    perfect_panel(img, A, xs[2], y, S, ease(u, 0.45, 0.6), crop=30)
    ov = cached("A6t", lambda: text_layer(lambda d: (
        header(d, "Glasses fix most of it, not all of it", "Same simulated eye, 30 arcmin view"),
        label(d, (xs[0] + S // 2, y + S + 50), "WITHOUT GLASSES", "step", INK),
        label(d, (xs[0] + S // 2, y + S + 98), "nearsightedness + astigmatism + the rest", "sub", MUTED),
        label(d, (xs[1] + S // 2, y + S + 50), "WITH GLASSES", "step", GOLDT),
        label(d, (xs[1] + S // 2, y + S + 98), "a small, uneven smudge is left", "sub", MUTED),
        label(d, (xs[2] + S // 2, y + S + 50), "PERFECT EYE", "step", INK),
        label(d, (xs[2] + S // 2, y + S + 98), "what glasses can't quite reach", "sub", MUTED),
        *[d.text((xs[i] + S + 75, y + S // 2), "→", font=F["h2"], fill=DIM, anchor="mm") for i in (0, 1)])))
    return comp(img, ov)


def A7(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S, x, y = 460, W // 2 - 230, 200
    img += frames_em("A7f", [(x, y, S, S, D.GOLD, 3)])
    padd(img, A.tile("A460", A.psf_rgb["A"], S), x, y)
    q = 1 - ease(u, 0.55, 0.65)
    ov = D.overlay(lambda d: d.text((x + S + 120, y + S // 2), "?", font=F["big"], fill=GOLDT, anchor="mm"))
    img = comp(img, ov, ease(u, 0.05, 0.15) * q)
    ov1 = cached("A7t", lambda: text_layer(lambda d: (
        header(d, "So why doesn't the world look smudged?"),
        d.text((W // 2, 790), "Why doesn't the world look like this?", font=F["h2"], fill=INK, anchor="mm"))))
    img = comp(img, ov1, q)
    ov2 = cached("A7t2", lambda: text_layer(lambda d: (
        header(d, "So why doesn't the world look smudged?"),
        d.text((W // 2, 790), "Maybe your brain has learned it.", font=F["h2"], fill=GOLDT, anchor="mm"),
        d.text((W // 2, 870), "…and quietly cleans it up", font=F["body"], fill=MUTED, anchor="mm"))))
    return comp(img, ov2, ease(u, 0.6, 0.72))


# ------------------------------------------------------------------ chapter B: the question
def B1(t, dur, A):
    return title_card(t, "ROCHESTER  ·  2004", ["THE EXPERIMENT"],
                      ["Testing whether your brain has learned your blur",
                       "Artal, Chen, Fernández, Singer, Manzanera & Williams  ·  Journal of Vision, 2004"], "B1")


def B2(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S, xl, xr, y = 520, 320, 1080, 220
    img += frames_em("B2f", [(xl, y, S, S, D.GOLD, 4)])
    padd(img, A.tile("r0_520", A.rot[0], S), xl, y)
    k = ease(u, 0.45, 0.58)
    img += frames_em("B2f2", [(xr, y, S, S, BLUE, 4)]) * k
    padd(img, A.tile("r90_520", A.rot[90], S), xr, y, k)
    ov = cached("B2t", lambda: text_layer(lambda d: (
        header(d, "The prediction", "If your brain has learned your smudge, a turned copy should look worse"),
        label(d, (xl + S // 2, y + S + 50), "YOURS", "step", GOLDT),
        label(d, (xl + S // 2, y + S + 98), "learned over a lifetime", "sub", MUTED))))
    img = comp(img, ov)
    ov2 = cached("B2t2", lambda: text_layer(lambda d: (
        label(d, (xr + S // 2, y + S + 50), "TURNED 90°", "step", BLUE8),
        label(d, (xr + S // 2, y + S + 98), "unfamiliar", "sub", MUTED),
        d.text((W // 2, 960), "same amount of blur  ·  should look worse?", font=F["bodyb"], fill=INK, anchor="mm"))))
    return comp(img, ov2, k)


def _hand_card(rot=0, w=700, h=330):
    im = Image.new("RGB", (w, h), (236, 230, 214))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(CAVEAT, 64)
    for i, s in enumerate(["Your own handwriting", "is easy for you", "to read."]):
        d.text((50, 40 + 85 * i), s, font=f, fill=(40, 45, 70))
    if rot:
        im = im.rotate(rot, expand=True, fillcolor=(0, 0, 0))
    return np.asarray(im, np.float32) / 255


def B3(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    c0 = cached("hand0", _hand_card)
    c1 = cached("hand90", lambda: _hand_card(90))
    paste(img, c0, 170, 340)
    k = ease(u, 0.45, 0.6)
    h1, w1 = c1.shape[:2]
    x1, y1 = 1260, 180
    reg = img[y1:y1 + h1, x1:x1 + w1]
    m = (c1.sum(2, keepdims=True) > 0).astype(np.float32) * k
    img[y1:y1 + h1, x1:x1 + w1] = reg * (1 - m) + c1 * m
    ov = cached("B3t", lambda: text_layer(lambda d: (
        header(d, "Like handwriting", "Analogy"),
        label(d, (520, 740), "upright: easy", "step", GOLDT))))
    img = comp(img, ov)
    ov2 = cached("B3t2", lambda: text_layer(lambda d: (
        label(d, (x1 + w1 // 2, 940), "same writing, turned: more effort", "step", BLUE8),)))
    return comp(img, ov2, k)


# ------------------------------------------------------------------ chapter C: the machine (reuses part2 scenes)
def C1(t, dur, A):
    return P.s04(t, dur, A)


def C2(t, dur, A):
    xs, y0, S = [150, 740, 1330], 270, 440
    P._cache["s05a"] = cached("C2t", lambda: text_layer(lambda d: (
        header(d, "Cancel the blur", "Measure the smudge, then bend a mirror into the opposite shape (schematic)"),
        label(d, (xs[0] + S // 2, y0 + S + 50), "Sensor", "step", INK),
        label(d, (xs[0] + S // 2, y0 + S + 100), "measures your smudge 30× a second", "sub", MUTED),
        label(d, (xs[1] + S // 2, y0 + S + 50), "Flexible mirror", "step", INK),
        label(d, (xs[1] + S // 2, y0 + S + 100), "97 actuators push into the opposite shape", "sub", MUTED),
        label(d, (xs[2] + S // 2, y0 + S + 50), "YOUR EYE  →  ≈ PERFECT EYE", "step", GOLDT),
        label(d, (xs[2] + S // 2, y0 + S + 100), "the smudge shrinks to a point", "sub", MUTED),
        *[d.text((xs[i] + S + 30, y0 + S // 2), "→", font=F["h2"], fill=DIM, anchor="mm") for i in (0, 1)])))
    return P.s05(t, dur, A)


def C3(t, dur, A):
    P._cache["s06a"] = cached("C3t", lambda: text_layer(lambda d: (
        header(d, "Then: put the blur back in", "The person's own smudge, or the very same smudge turned to a new angle"),
        label(d, (W // 2, 575 + 135 + 26), "YOURS", "small", GOLDT),
        label(d, (190, 980), "PERFECT EYE", "small", INK))))
    img = P.s06(t, dur, A)
    img += frames_em("C3f", [(115, 790, 150, 150, WHITE, 2)])
    perfect_panel(img, A, 115, 790, 150)
    return img


def C4(t, dur, A):
    P._cache["s07a"] = cached("C4t", lambda: text_layer(lambda d: (
        header(d, "Turned, but just as blurry", "The scientists measured it: the same amount of blur at every angle"),
        label(d, (330 + 235, 740), "YOURS (0°)", "step", GOLDT),
        label(d, (1120 + 235, 740), "TURNED (90°)", "step", BLUE8),
        label(d, (W // 2, 960), "Only the direction changed", "bodyb", INK))))
    return P.s07(t, dur, A)


# ------------------------------------------------------------------ chapter D: the test
def D1(t, dur, A):
    img = base_bg().copy()
    S, x, y = 600, W // 2 - 300, 200
    own = int(t / 0.5) % 2 == 0
    img += frames_em("D1g", [(x, y, S, S, D.GOLD, 4)]) if own else frames_em("D1b", [(x, y, S, S, BLUE, 4)])
    paste(img, cached("D1own", lambda: P.gray_tile(A.own, S)) if own else
          cached("D1rot", lambda: P.gray_tile(A.rotF[1.0], S)), x, y)
    ov = cached("D1t", lambda: text_layer(lambda d: header(
        d, "The test", "A speck pattern, flicking every half second between their own smudge and a turned one")))
    img = comp(img, ov)
    ov2 = cached("D1own_t" if own else "D1rot_t", lambda: text_layer(lambda d: label(
        d, (W // 2, y + S + 60), "THROUGH YOUR OWN SMUDGE" if own else "THROUGH THE TURNED SMUDGE", "step",
        GOLDT if own else BLUE8)))
    return comp(img, ov2)


def _pair(img, A, key):
    S, xl, xr, y = 540, 300, 1080, 210
    img += frames_em(key + "f", [(xl, y, S, S, D.GOLD, 4), (xr, y, S, S, BLUE, 4)])
    paste(img, cached("pair_own", lambda: P.gray_tile(A.own, S)), xl, y)
    paste(img, cached("pair_rot", lambda: P.gray_tile(A.rotF[1.0], S)), xr, y)
    return S, xl, xr, y


def D2(t, dur, A):
    img = base_bg().copy()
    S, xl, xr, y = _pair(img, A, "D2")
    ov = cached("D2t", lambda: text_layer(lambda d: (
        header(d, "Same amount of blur…", "If the brain doesn't care about the shape, these should look equally blurry"),
        d.text((W // 2, y + S // 2), "=", font=F["big"], fill=INK, anchor="mm"),
        label(d, (xl + S // 2, y + S + 50), "YOURS", "step", GOLDT),
        label(d, (xr + S // 2, y + S + 50), "TURNED", "step", BLUE8),
        d.text((W // 2, 930), "same amount of blur", font=F["bodyb"], fill=INK, anchor="mm"))))
    return comp(img, ov, ease(t, 0.0, 0.5))


def D3(t, dur, A):
    img = base_bg().copy()
    u = t / dur
    S, xl, xr, y = _pair(img, A, "D3")
    k = ease(u, 0.12, 0.22)
    ov = cached("D3a", lambda: text_layer(lambda d: (
        header(d, "…but not the same look", "What every one of the five people saw"),
        d.text((W // 2, y + S // 2), "=", font=F["big"], fill=INK, anchor="mm"))))
    img = comp(img, ov, 1 - k)
    ov2 = cached("D3b", lambda: text_layer(lambda d: (
        header(d, "…but not the same look", "What every one of the five people saw"),
        d.text((W // 2, y + S // 2), "≠", font=F["big"], fill=GOLDT, anchor="mm"),
        label(d, (xl + S // 2, y + S + 50), "YOURS: looked sharper", "step", GOLDT),
        label(d, (xr + S // 2, y + S + 50), "TURNED: looked blurrier", "step", BLUE8))))
    img = comp(img, ov2, k)
    ov3 = cached("D3c", lambda: text_layer(lambda d: d.text((W // 2, 930), "5 out of 5 people", font=F["h2"],
                                                            fill=INK, anchor="mm")))
    return comp(img, ov3, ease(u, 0.45, 0.55))


def D4(t, dur, A):
    u = t / dur
    if u < 0.55:
        img = base_bg().copy()
        S, xl, xr, y = 520, 320, 1080, 200
        uu = u / 0.55
        Fv = 1.0 - 0.30 * ease(uu, 0.25, 0.9)
        fk = A.Fgrid[np.argmin(np.abs(A.Fgrid - Fv))]
        img += frames_em("D4f", [(xl, y, S, S, D.GOLD, 4), (xr, y, S, S, BLUE, 4)])
        paste(img, cached("D4own", lambda: P.gray_tile(A.own, S)), xl, y)
        paste(img, P.gray_tile(A.rotF[fk], S), xr, y)
        ov = cached("D4t", lambda: text_layer(lambda d: (
            header(d, "How much blurrier?", "Each person turned the turned blur down, like a focus knob, until both matched"),
            label(d, (xl + S // 2, y + S + 50), "YOURS", "step", GOLDT),
            label(d, (xl + S // 2, y + S + 96), "left alone", "sub", MUTED),
            label(d, (xr + S // 2, y + S + 50), "TURNED", "step", BLUE8))))
        img = comp(img, ov)
        return comp(img, D.overlay(lambda d: P._dial(d, xr + S // 2, y + S + 110, Fv)))
    # second half: where all five ended up
    _s09_text()
    return P.s09((u - 0.55) / 0.45 * dur, dur, A)


def _s09_text():
    x0, x1, y = 260, 1660, 600

    def X(v):
        return int(x0 + v / 1.2 * (x1 - x0))
    P._cache["s09a"] = cached("D4a", lambda: text_layer(lambda d: (
        header(d, "Where all five people ended up", "How far each person had to turn the turned blur down to match their own"),
        d.rectangle((x0, y - 3, x1, y + 3), fill=(50, 66, 90)),
        *[(d.line((X(v), y - 12, X(v), y + 12), fill=(110, 130, 155), width=2),
           d.text((X(v), y + 70), f"{int(v * 100)}%", font=F["sub"], fill=MUTED, anchor="mm")) for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)],
        d.line((X(1.0), y - 120, X(1.0), y + 40), fill=(150, 176, 205), width=3),
        d.text((X(1.0), y - 150), "no change needed", font=F["sub"], fill=MUTED, anchor="mm"))))
    P._cache["s09b"] = cached("D4b", lambda: text_layer(lambda d: (
        d.text(((X(0.6) + X(0.8)) // 2, y - 90), "all 5 people: 60–80%", font=F["step"], fill=GOLDT, anchor="mm"),
        d.text((W // 2, 850), "They had to remove 20–40% of the turned blur", font=F["h2"], fill=INK, anchor="mm"),
        d.text((W // 2, 930), "before it looked as clear as their own.", font=F["body"], fill=MUTED, anchor="mm"))))


def D5(t, dur, A):
    P._cache["s11a"] = cached("D5t", lambda: text_layer(lambda d: (
        header(d, "Even upside down, it looked worse", "Flipped 180°: the amount of blur is identical in every way"),
        label(d, (330 + 235, 730), "YOURS", "step", GOLDT),
        label(d, (1120 + 235, 730), "UPSIDE DOWN (180°)", "step", BLUE8))))
    P._cache["s11b"] = cached("D5t2", lambda: text_layer(lambda d: (
        d.text((W // 2, 850), "Not just how much your eye blurs.", font=F["h2"], fill=INK, anchor="mm"),
        d.text((W // 2, 935), "The exact SHAPE of your blur.", font=F["h2"], fill=GOLDT, anchor="mm"))))
    return P.s11(t, dur, A)


# ------------------------------------------------------------------ chapter E: what it means
def E1(t, dur, A):
    img = base_bg(True, 0.3).copy()
    u = t / dur
    S, y = 420, 250
    xs = [180, 750, 1320]
    img += frames_em("E1f", [(xs[0], y, S, S, WHITE, 2), (xs[1], y, S, S, D.GOLD, 3), (xs[2], y, S, S, D.GOLD, 2)])
    perfect_panel(img, A, xs[0], y, S)
    padd(img, A.tile("A420", A.psf_rgb["A"], S), xs[1], y, ease(u, 0.12, 0.25))
    padd(img, cached("E1p", lambda: P.to_tile(A.perceived, S)), xs[2], y, ease(u, 0.42, 0.56))
    ov = cached("E1t", lambda: text_layer(lambda d: (
        header(d, "What your brain does", "Illustration"),
        label(d, (xs[0] + S // 2, y + S + 50), "PERFECT EYE", "step", INK),
        label(d, (xs[1] + S // 2, y + S + 50), "YOUR EYE DELIVERS", "step", GOLDT),
        label(d, (xs[2] + S // 2, y + S + 50), "YOU PERCEIVE", "step", GOLDT),
        label(d, (xs[2] + S // 2, y + S + 96), "closer to perfect than your optics", "sub", MUTED))))
    img = comp(img, ov)
    ov2 = cached("E1t2", lambda: text_layer(lambda d: d.text(
        (W // 2, 960), "You see more clearly than your eyes alone should allow.", font=F["h2"], fill=INK, anchor="mm")))
    return comp(img, ov2, ease(u, 0.7, 0.8))


def E2(t, dur, A):
    return P.s13(t, dur, A)


def E3(t, dur, A):
    return P.s14(t, dur, A)


def E4(t, dur, A):
    return P.s15(t, dur, A)


CHAPTER = {"A": "YOUR BLUR", "B": "THE QUESTION", "C": "THE MACHINE", "D": "THE TEST", "E": "WHAT IT MEANS"}
FNS = {k: v for k, v in globals().items() if len(k) == 2 and k[0] in "ABCDE" and k[1].isdigit()}


def timeline():
    script = json.load(open(f"{V2}/part2_script.json"))
    durs = json.load(open(f"{V2}/vo_p2/durations.json"))
    t, out = 0.0, []
    for ln in script:
        d = 0.4 + durs[ln["id"]]["seconds"] + ln["pause"]
        if ln["id"] == "E4":                                 # end card: at least 8 s, voice starts after it settles
            d = max(8.0, 1.2 + durs["E4"]["seconds"] + ln["pause"])
        out.append(dict(id=ln["id"], start=round(t, 3), dur=round(d, 3), vo=True,
                        vo_start=round(t + (1.2 if ln["id"] == "E4" else 0.4), 3),
                        label=CHAPTER[ln["id"][0]] if ln["id"] not in ("A1", "B1", "E4") else ""))
        t += d
    if "E4" not in [ln["id"] for ln in script]:
        out.append(dict(id="E4", start=round(t, 3), dur=8.0, vo=False, label=""))
        t += 8.0
    S = {s["id"]: s for s in out}

    def span(a, b):
        return S[a]["start"], S[b]["start"] + S[b]["dur"]

    music = []
    for (a, b), chords, dens in ((("A1", "A1"), ["F", "Bb"], 0.10), (("A2", "A4"), ["Dm9", "Bb", "F/A", "Gm"], 0.16),
                                 (("A5", "A7"), ["Am", "Bb", "Dm9", "Csus"], 0.18), (("B1", "B3"), ["F", "C", "Dm9", "Bb"], 0.18),
                                 (("C1", "C4"), ["Am", "Bb", "Dm9", "Gm", "F/A", "Csus"], 0.22), (("D1", "D2"), ["Gm", "Csus"], 0.08),
                                 (("D3", "D5"), ["Fadd9", "Am", "Bb", "Dm9", "Gm", "Csus"], 0.20),
                                 (("E1", "E3"), ["Bb", "F/A", "Gm", "Dm9", "Bb", "Csus"], 0.18)):
        t0, t1 = span(a, b)
        music.append(dict(t0=t0, t1=t1, chords=chords, dens=dens))
    tl = dict(total=round(t, 3), scenes=out, music=music, chimes=[],
              final=S["E4"]["start"])
    json.dump(tl, open(f"{V2}/timeline.json", "w"), indent=1)
    return tl


def render(tl, A):
    total = tl["total"]
    prev_last = None
    for sc in tl["scenes"]:
        n = int(round(sc["dur"] * FPS))
        nx = int(P.XFADE * FPS)
        last = None
        for i in range(n):
            t = i / FPS
            img = FNS[sc["id"]](t, sc["dur"], A)
            if sc["label"]:
                img = P.hud(img, f"PART 2  ·  {sc['label']}", (sc["start"] + t) / total)
            if prev_last is not None and i < nx:
                a = (i + 1) / (nx + 1)
                img = prev_last * (1 - a) + img * a
            last = img
            yield img
        prev_last = last


def tail(tl, A, first):
    """re-render scenes from `first` on; frames before it are copied from the existing part2_video.mp4"""
    k = [s["id"] for s in tl["scenes"]].index(first)
    head_frames = int(round(tl["scenes"][k]["start"] * FPS))
    prev = tl["scenes"][k - 1]
    prev_last = FNS[prev["id"]]((int(round(prev["dur"] * FPS)) - 1) / FPS, prev["dur"], A)
    if prev["label"]:
        prev_last = P.hud(prev_last, f"PART 2  ·  {prev['label']}", (prev["start"] + prev["dur"]) / tl["total"])
    old = f"{V2}/part2_video.mp4"
    head = f"{V2}/p2_head.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", old, "-vf", f"trim=end_frame={head_frames},setpts=PTS-STARTPTS",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-r", "30", head], check=True)
    tailf = f"{V2}/p2_tail.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", tailf]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sub = dict(tl, scenes=tl["scenes"][k:])
    first_scene = True
    for sc in sub["scenes"]:
        n = int(round(sc["dur"] * FPS))
        nx = int(P.XFADE * FPS)
        last = None
        for i in range(n):
            t = i / FPS
            img = FNS[sc["id"]](t, sc["dur"], A)
            if sc["label"]:
                img = P.hud(img, f"PART 2  ·  {sc['label']}", (sc["start"] + t) / tl["total"])
            if prev_last is not None and i < nx:
                a = (i + 1) / (nx + 1)
                img = prev_last * (1 - a) + img * a
            last = img
            p.stdin.write(D.to8(img).tobytes())
        prev_last = last
    p.stdin.close()
    p.wait()
    lst = f"{V2}/p2_concat.txt"
    open(lst, "w").write(f"file '{head}'\nfile '{tailf}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", f"{V2}/part2_video_new.mp4"], check=True)
    os.replace(f"{V2}/part2_video_new.mp4", old)
    print("tail re-rendered from", first, "head frames", head_frames, "total", tl["total"])


def main(mode):
    tl = timeline()
    A = Assets2()
    if mode == "stills":
        os.makedirs(f"{V2}/stills", exist_ok=True)
        for sc in tl["scenes"]:
            img = FNS[sc["id"]](0.85 * sc["dur"], sc["dur"], A)
            if sc["label"]:
                img = P.hud(img, f"PART 2  ·  {sc['label']}", 0.5)
            Image.fromarray(D.to8(img)).save(f"{V2}/stills/{sc['id']}.png")
        print("total", tl["total"])
        return
    if mode == "tail":
        return tail(tl, A, sys.argv[2])
    out = f"{V2}/part2_video.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img in render(tl, A):
        p.stdin.write(D.to8(img).tobytes())
    p.stdin.close()
    p.wait()
    print(out, tl["total"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "video")
