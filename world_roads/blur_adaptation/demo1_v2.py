"""Demo 1 v2: split-screen blur adaptation, styled after the Light Speed films.

Left panel adapts to blurred photos, right panel to over-sharpened ones, while the viewer stares at a
glowing dot between them. Test pairs (identical images on both sides) are marked by GOLD frames, a
gold dot and gold L/R letters; viewers decide which side looks sharper. Expected answer: LEFT.

Usage: python3 demo1_v2.py [video|stills]
"""
from paths import WORK  # noqa: E402
import os
import sys
import random
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom

BASE = WORK + "/blur"
# DEMO_SET=fire swaps in the "We Didn't Start the Fire" photo set (fire/sets.json); default = the NASA set
DEMO_SET = os.environ.get("DEMO_SET", "nasa")
if DEMO_SET == "fire":
    _FIRE = BASE.rsplit("/", 1)[0] + "/fire"
    SRC = f"{_FIRE}/src_images"
    OUT = f"{_FIRE}/demo1_fire.mp4"
    STILLS = f"{_FIRE}/stills"
    PHOTO_CREDIT = "Photos: public domain (U.S. government, Library of Congress)."
else:
    SRC = f"{BASE}/src_images"
    OUT = f"{BASE}/demo1_v2.mp4"
    STILLS = f"{BASE}/v2_stills"
    PHOTO_CREDIT = "Photos: NASA (public domain)."
FONT_DIR = "/usr/share/fonts/opentype/inter"
W, H, FPS = 1920, 1080, 30
P, GAP = 800, 72
LX, RX = (W - GAP) // 2 - P, (W + GAP) // 2
TOP = 130
CX, CY = W // 2, TOP + P // 2

# Light Speed palette (display-space 0..1)
GOLD = np.array([1.00, 0.56, 0.13], np.float32) * 0.92
HOT = np.array([1.00, 0.93, 0.78], np.float32)
CYAN = np.array([0.16, 0.58, 0.95], np.float32)
INK = (236, 242, 250)
MUTED = (150, 176, 205)
DIM = (120, 140, 165)
GOLD8 = (255, 176, 66)

ADAPT = ["GSFC_20171208_Archive_e000236", "GSFC_20171208_Archive_e000507", "GSFC_20171208_Archive_e002000",
         "PIA16717", "SL2-04-018", "iss074e0208838", "KSC-20210226-PH-JBS01_0081",
         "KSC-20210226-PH-JBS01_0090", "S71-38189", "GSFC_20171208_Archive_e000994", "iss058e005276",
         "iss072e519889", "iss074e0150190", "PIA17068", "as15-97-13160"]
TESTS = ["61c-14-008", "9250827", "S84-27026"]
if DEMO_SET == "fire":
    import json as _json
    _sets = _json.load(open(f"{_FIRE}/sets.json"))
    ADAPT, TESTS = _sets["adapt"], _sets["tests"]
    CROPS = _sets.get("crops", {})
else:
    CROPS = {}

ADAPT_S, TOPUP_S, STEP_S, TEST_S, ANSWER_S = 60.0, 10.5, 1.5, 3.0, 2.5


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


# ------------------------------------------------------------------ image processing
def load(name, flip=False):
    path = f"{SRC}/{name}" if "." in os.path.basename(name) else f"{SRC}/{name}.jpg"
    im = Image.open(path).convert("RGB")
    cx, cy, frac = CROPS.get(os.path.basename(path), (0.5, 0.5, 1.0))
    s = int(min(im.size) * frac)
    l = int(min(max(cx * im.width - s / 2, 0), im.width - s))
    t = int(min(max(cy * im.height - s / 2, 0), im.height - s))
    im = im.crop((l, t, l + s, t + s)).resize((P, P), Image.LANCZOS)
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return (np.asarray(im, np.float32) / 255.0) ** 2.2


def blur(lin, s):
    return gaussian_filter(lin, sigma=(s, s, 0), mode="reflect")


def sharpen(lin, amount=1.6, radius=2.0):
    return lin + amount * (lin - blur(lin, radius))


def disp(lin):
    return np.clip(lin, 0, 1) ** (1 / 2.2)


# ------------------------------------------------------------------ background + glow
def sprite(sig, n):
    k = np.arange(n, dtype=np.float32) - n // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig))


def bloom(em):
    """Two-scale bloom at half resolution, like render.py."""
    h = em[::2, ::2]
    b = 0.55 * gaussian_filter(h, (5, 5, 0)) + 0.65 * gaussian_filter(h, (16, 16, 0))
    return zoom(b, (2, 2, 1), order=1)[:H, :W]


def cone_mosaic(strength, gold_band=None):
    """Faint hexagonal cone mosaic (the hyperacuity motif); optional gold-lit diagonal band."""
    im = Image.new("L", (W, H), 0)
    g = Image.new("L", (W, H), 0)
    d, dg = ImageDraw.Draw(im), ImageDraw.Draw(g)
    r, sx = 24, 56
    sy = sx * 0.866
    for j in range(int(H / sy) + 2):
        for i in range(int(W / sx) + 2):
            x = i * sx + (sx / 2 if j % 2 else 0)
            y = j * sy
            d.ellipse((x - r, y - r, x + r, y + r), outline=255, width=2)
            if gold_band is not None:
                x0, slope, width = gold_band
                dist = abs(x - (x0 + slope * (y - H / 2)))
                if dist < width:
                    dg.ellipse((x - r + 2, y - r + 2, x + r - 2, y + r - 2), fill=int(255 * (1 - dist / width) ** 1.5))
    ring = np.asarray(im, np.float32)[..., None] / 255 * strength
    lit = np.asarray(g, np.float32)[..., None] / 255
    return ring, lit


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGN = np.clip(1 - 0.45 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.52) / (H * 0.75)) ** 2), 0.35, 1)[..., None]
NAVY = np.array([0.020, 0.034, 0.064], np.float32)
BG = NAVY * VIGN
RING_DEMO, _ = cone_mosaic(0.035)
RING_CARD, LIT_CARD = cone_mosaic(0.06, gold_band=(1560, -0.28, 170))
GRAIN = np.random.default_rng(1234).uniform(-0.004, 0.004, (H, W, 1)).astype(np.float32)

DOT_N = 81
DOT_CORE = sprite(3.0, DOT_N)[..., None]
DOT_GLOW = sprite(6.5, DOT_N)[..., None]
DOT_HALO = sprite(13.0, DOT_N)[..., None]


def add_dot(img, test, pulse=1.0, cx=CX, cy=CY, scale=1.0):
    tint = GOLD if test else CYAN
    spr = DOT_CORE * HOT * 2.6 + DOT_GLOW * (tint * 0.7 + 0.3 * HOT) * 1.3 * pulse + DOT_HALO * tint * 0.5 * pulse
    if scale != 1.0:
        spr = zoom(spr, (scale, scale, 1), order=1)
    n = spr.shape[0]
    y0, x0 = cy - n // 2, cx - n // 2
    img[y0:y0 + n, x0:x0 + n] += spr


# ------------------------------------------------------------------ text overlays
def overlay(draw_fn):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(im))
    a = np.asarray(im, np.float32) / 255
    return a[..., :3], a[..., 3:]


def comp(img, ov):
    rgb, a = ov
    return img * (1 - a) + rgb * a


def glow_text(draw_fn):
    """Text drawn into an emission layer (it glows)."""
    im = Image.new("RGB", (W, H), (0, 0, 0))
    draw_fn(ImageDraw.Draw(im))
    return np.asarray(im, np.float32) / 255


def frame_rect(em, x, y, w, h, b, col, k=1.0):
    em[y - b:y, x - b:x + w + b] += col * k
    em[y + h:y + h + b, x - b:x + w + b] += col * k
    em[y:y + h, x - b:x] += col * k
    em[y:y + h, x + w:x + w + b] += col * k


F = {
    "title": font("Inter-SemiBold.otf", 40), "sub": font("Inter-Regular.otf", 22),
    "small": font("Inter-Medium.otf", 15), "clock": font("InterDisplay-Medium.otf", 58),
    "lr": font("InterDisplay-SemiBold.otf", 46), "foot": font("Inter-Regular.otf", 14),
    "h1": font("InterDisplay-Black.otf", 104), "h2": font("InterDisplay-Bold.otf", 64),
    "body": font("Inter-Regular.otf", 34), "bodyb": font("Inter-SemiBold.otf", 34),
    "step": font("InterDisplay-Bold.otf", 40), "num": font("InterDisplay-Black.otf", 60),
    "big": font("InterDisplay-Black.otf", 150),
}


def hud_static(d):
    d.text((58, 38), "Re-tune your sense of sharpness", font=F["title"], fill=INK)
    d.text((60, 90), "Blur adaptation  ·  keep your eyes on the glowing dot", font=F["sub"], fill=MUTED)
    d.text((58, H - 40), f"{PHOTO_CREDIT}  After Webster, Georgeson & Webster, Nature Neuroscience (2002).",
           font=F["foot"], fill=DIM, anchor="ls")


def hud_clock(d, label, secs, frac, gold=False):
    right = W - 60
    d.text((right, 968), label, font=F["small"], fill=(GOLD8 if gold else (130, 150, 175)), anchor="ra")
    s = secs if isinstance(secs, str) else f"{int(secs // 60)}:{int(secs % 60):02d}"
    d.text((right, 986), s, font=F["clock"], fill=(246, 236, 214), anchor="ra")
    bw = 300
    d.rectangle([right - bw, 1052, right, 1054], fill=(40, 55, 75))
    d.rectangle([right - bw, 1052, right - bw + int(bw * frac), 1054], fill=GOLD8)


def lr_letters(d, fill):
    d.text((LX + P // 2, TOP + P + 46), "L", font=F["lr"], fill=fill, anchor="mm")
    d.text((RX + P // 2, TOP + P + 46), "R", font=F["lr"], fill=fill, anchor="mm")


# ------------------------------------------------------------------ demo frames
class DemoLayers:
    """Precomputed static layers for the adapt and test states."""

    def __init__(self):
        self.base = {}
        for test in (False, True):
            em = np.zeros((H, W, 3), np.float32)
            if test:
                frame_rect(em, LX, TOP, P, P, 6, GOLD, 1.0)
                frame_rect(em, RX, TOP, P, P, 6, GOLD, 1.0)
                em += glow_text(lambda d: lr_letters(d, GOLD8))
                tri = glow_text(lambda d: [d.polygon(pts, fill=GOLD8) for pts in (
                    [(CX - 34, CY), (CX - 24, CY - 9), (CX - 24, CY + 9)],
                    [(CX + 34, CY), (CX + 24, CY - 9), (CX + 24, CY + 9)])])
                em += tri
            else:
                frame_rect(em, LX, TOP, P, P, 2, CYAN, 0.55)
                frame_rect(em, RX, TOP, P, P, 2, CYAN, 0.55)
            img = BG + RING_DEMO * np.array([0.25, 0.45, 0.75], np.float32) + em + bloom(em)
            if not test:
                img = comp(img, overlay(lambda d: lr_letters(d, (70, 88, 112))))
            img = comp(img, overlay(hud_static))
            self.base[test] = img.astype(np.float32)

    def render(self, left, right, test, label, secs, frac, pulse):
        img = self.base[test].copy()
        img[TOP:TOP + P, LX:LX + P] = disp(left)
        img[TOP:TOP + P, RX:RX + P] = disp(right)
        add_dot(img, test, pulse)
        img = comp(img, overlay(lambda d: hud_clock(d, label, secs, frac, gold=test)))
        return img


# ------------------------------------------------------------------ cards
def card_base(lit=0.0):
    em = LIT_CARD * GOLD * 0.55 * lit
    return BG + RING_CARD * np.array([0.25, 0.45, 0.75], np.float32) + em + bloom(em)


def mini_pair(em, d_ov, cx, cy, test, s=150, gap=14):
    lx, rx, y = cx - gap // 2 - s, cx + gap // 2, cy - s // 2
    col = GOLD if test else CYAN
    frame_rect(em, lx, y, s, s, 4 if test else 2, col, 1.0 if test else 0.6)
    frame_rect(em, rx, y, s, s, 4 if test else 2, col, 1.0 if test else 0.6)
    d_ov.append((lx, y, s))
    d_ov.append((rx, y, s))
    return (cx, cy)


def card_title():
    img = card_base(lit=1.0)

    def t(d):
        d.text((110, 300), "YOUR BRAIN IS", font=F["h1"], fill=(255, 255, 255))
        d.text((110, 420), "ABOUT TO", font=F["h1"], fill=(255, 255, 255))
        d.text((110, 540), "RE-TUNE", font=F["h1"], fill=(255, 190, 90))
        d.text((110, 660), "YOUR EYES", font=F["h1"], fill=(255, 255, 255))
        d.text((114, 830), "A 2-minute experiment  ·  watch full screen", font=F["body"], fill=MUTED)
    em = glow_text(lambda d: d.text((110, 540), "RE-TUNE", font=F["h1"], fill=(120, 70, 15)))
    img = img + bloom(em) * 1.2
    return comp(img, overlay(t))


def card_rules():
    img = card_base(lit=0.25)
    em = np.zeros((H, W, 3), np.float32)
    rows = [(250, False, "1", "STARE at the glowing dot.", "The whole time. Let the pictures play in the corner of your eye."),
            (520, True, "2", "GOLD frames = TEST.", "Keep staring at the dot. Don't look at either picture."),
            (790, None, "3", "Which side looks SHARPER?", "Decide: L or R. There are 3 tests, so remember your answers.")]
    panels, dots = [], []
    for y, test, *_ in rows:
        if test is not None:
            dots.append((mini_pair(em, panels, 420, y, test), test))
    em += glow_text(lambda d: d.text((420, 790), "L  ?  R", font=F["num"], fill=GOLD8, anchor="mm"))
    img = img + em + bloom(em)
    for x, y, s in panels:
        img[y:y + s, x:x + s] = np.array([0.30, 0.33, 0.38], np.float32)
    for (cx, cy), test in dots:
        add_dot(img, test, 1.0, cx, cy)

    def t(d):
        d.text((110, 90), "How the test works", font=F["h2"], fill=INK)
        for y, test, num, head, body in rows:
            d.text((700, y - 44), num, font=F["num"], fill=GOLD8, anchor="mm")
            d.text((760, y - 44), head, font=F["step"], fill=(255, 190, 90) if test or test is None else INK, anchor="lm")
            d.text((760, y + 14), body, font=F["body"], fill=MUTED, anchor="lm")
    return comp(img, overlay(t))


def card_ready(n):
    img = BG + RING_DEMO * np.array([0.25, 0.45, 0.75], np.float32)
    add_dot(img, False, 1.0)

    def t(d):
        d.text((CX, CY - 150), str(n), font=F["big"], fill=INK, anchor="mm")
        d.text((CX, CY + 120), "Eyes on the dot", font=F["step"], fill=MUTED, anchor="mm")
    return comp(img, overlay(t))


def card_answer(k):
    img = card_base(lit=0.15)
    em = glow_text(lambda d: (d.text((CX - 190, 560), "L", font=F["big"], fill=GOLD8, anchor="mm"),
                              d.text((CX + 190, 560), "R", font=F["big"], fill=GOLD8, anchor="mm")))
    img = img + em * 0.9 + bloom(em)

    def t(d):
        d.text((CX, 300), f"TEST {k} OF 3", font=F["bodyb"], fill=GOLD8, anchor="mm")
        d.text((CX, 380), "Which side looked sharper?", font=F["h2"], fill=INK, anchor="mm")
        d.text((CX, 560), "or", font=F["body"], fill=MUTED, anchor="mm")
        d.text((CX, 760), "Remember it, then eyes straight back on the dot", font=F["body"], fill=MUTED, anchor="mm")
    return comp(img, overlay(t))


def card_reveal():
    img = card_base(lit=0.6)
    em = glow_text(lambda d: d.text((110, 330), "IDENTICAL.", font=F["h1"], fill=(120, 70, 15)))
    img = img + bloom(em) * 1.2

    def t(d):
        d.text((110, 220), "ALL THREE TEST PAIRS WERE", font=F["h2"], fill=INK)
        d.text((110, 330), "IDENTICAL.", font=F["h1"], fill=(255, 190, 90))
        lines = [("Did LEFT look sharper? That's the effect.", True),
                 ("The left side spent a minute on blurry photos, so your brain turned", False),
                 ("its sharpness up. The right side saw over-sharpened photos, so your", False),
                 ("brain turned it down. Same image, two different settings for “sharp”.", False)]
        y = 520
        for s, b in lines:
            d.text((112, y), s, font=F["bodyb"] if b else F["body"], fill=INK if b else MUTED)
            y += 64 if b else 52
    return comp(img, overlay(t))


def card_end():
    img = card_base(lit=1.0)

    def t(d):
        d.text((110, 360), "“Sharp” isn’t a fact about the image.", font=F["h2"], fill=INK)
        d.text((110, 450), "It’s what your brain has learned to expect.", font=F["h2"], fill=(255, 190, 90))
        d.text((112, 600), "Webster, Georgeson & Webster (2002). Neural adjustments to image blur.",
               font=F["body"], fill=MUTED)
        d.text((112, 650), "Nature Neuroscience 5, 839–840.", font=F["body"], fill=MUTED)
    return comp(img, overlay(t))


# ------------------------------------------------------------------ timeline
def timeline():
    """Yields (kind, payload, seconds) segments."""
    rng = random.Random(7)
    order = []
    while len(order) < 120:
        batch = ADAPT[:]
        rng.shuffle(batch)
        order += batch
    it = iter(order)
    flips = random.Random(11)
    segs = [("card", card_title(), 6.0), ("card", card_rules(), 13.0)]
    segs += [("card", card_ready(n), 1.0) for n in (3, 2, 1)]

    def adapt(seconds):
        out = []
        for i in range(int(round(seconds / STEP_S))):
            out.append(("adapt", (next(it), flips.random() < 0.5, seconds - i * STEP_S), STEP_S))
        return out

    segs += adapt(ADAPT_S)
    for k, t in enumerate(TESTS, 1):
        segs.append(("test", (t, k), TEST_S))
        segs.append(("card", card_answer(k), ANSWER_S))
        if k < len(TESTS):
            segs.append(("card", card_ready("•"), 0.7))
            segs += adapt(TOPUP_S)
    segs.append(("card", card_reveal(), 11.0))
    segs.append(("control", TESTS[0], 5.0))
    segs.append(("card", card_end(), 6.0))
    return segs


def frames(segs):
    layers = DemoLayers()
    total = sum(s for _, _, s in segs)
    demo_total = sum(s for k, _, s in segs if k in ("adapt", "test"))
    demo_t = 0.0
    t = 0.0
    prev_card = None
    cache = {}
    for kind, payload, secs in segs:
        n = int(round(secs * FPS))
        if kind == "card":
            img = payload
            nf = min(9, n)
            for i in range(n):
                if prev_card is not None and i < nf:          # short crossfade between cards
                    a = (i + 1) / (nf + 1)
                    yield prev_card * (1 - a) + img * a
                else:
                    yield img
            prev_card = img
        elif kind in ("adapt", "test", "control"):
            prev_card = None
            if kind == "adapt":
                name, flip, remain = payload
                key = (name, flip)
                if key not in cache:
                    lin = load(name, flip)
                    cache[key] = (blur(lin, 7.0), sharpen(lin))
                L, R = cache[key]
            else:
                name = payload if kind == "control" else payload[0]
                L = R = load(name)
            for i in range(n):
                ft = i / FPS
                pulse = 1.0 + 0.18 * np.sin(2 * np.pi * (t + ft) / 2.4)
                if kind == "adapt":
                    rem = remain - ft
                    yield layers.render(L, R, False, "ADAPTING  ·  KEEP STARING", max(rem, 0) + 0.999,
                                        (demo_t + ft) / demo_total, pulse)
                elif kind == "test":
                    yield layers.render(L, R, True, f"WHICH SIDE IS SHARPER?", f"TEST {payload[1]}/3",
                                        (demo_t + ft) / demo_total, 1.25)
                else:
                    img = layers.render(L, R, True, "EYES RESET", "CONTROL", 1.0, 1.25)
                    img = comp(img, overlay(lambda d: d.rectangle((0, 0, W, 120), fill=(5, 9, 16))))
                    img = comp(img, overlay(lambda d: (
                        d.text((58, 38), "Same test pair again, now that your eyes have reset", font=F["title"], fill=INK),
                        d.text((60, 90), "They should look the same", font=F["sub"], fill=(255, 190, 90)))))
                    yield img
            if kind != "control":
                demo_t += secs
        t += secs


def to8(img):
    return (np.clip(img + GRAIN, 0, 1) * 255 + 0.5).astype(np.uint8)


def main(mode):
    segs = timeline()
    if mode == "stills":
        d = STILLS
        os.makedirs(d, exist_ok=True)
        layers = DemoLayers()
        lin = load(ADAPT[3])
        shots = {"1_title": card_title(), "2_rules": card_rules(), "3_ready": card_ready(3),
                 "4_adapt": layers.render(blur(lin, 7.0), sharpen(lin), False, "ADAPTING  ·  KEEP STARING",
                                          42.5, 0.3, 1.0),
                 "5_test": layers.render(load(TESTS[0]), load(TESTS[0]), True,
                                         "WHICH SIDE IS SHARPER?", "TEST 1/3", 0.66, 1.25),
                 "6_answer": card_answer(1), "7_reveal": card_reveal(), "8_end": card_end()}
        for k, img in shots.items():
            Image.fromarray(to8(img)).save(f"{d}/{k}.png")
        return
    total = sum(s for _, _, s in segs)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img in frames(segs):
        p.stdin.write(to8(img).tobytes())
    p.stdin.close()
    p.wait()
    print(OUT, f"{total:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "video")
