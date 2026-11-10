"""Compose three thumbnail designs (1280x720) from Cycles close-ups of the film's scene (thumb_render.py).

    python3 compose.py ../media/thumbnails/render_embers_1944.png ../media/thumbnails/render_macro_1944.png \
        612 548 ../media/thumbnails
GRAIN_X, GRAIN_Y: where (in the 1280x720 macro) the circled grain is.

A  Embers       the photograph, with an editorial serif (Instrument Serif) in warm off-white
B  The page     a bone-paper panel, black serif numerals and one red line, beside the photograph
C  Annotated    a macro of the grains, one circled by hand: "your ancestor" (Caveat)
The fonts (SIL Open Font License) are fetched from the Google Fonts repository into data/fonts/ on first use.
"""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fonts")
FONTS = {"InstrumentSerif-Regular.ttf": "ofl/instrumentserif/InstrumentSerif-Regular.ttf",
         "InstrumentSerif-Italic.ttf": "ofl/instrumentserif/InstrumentSerif-Italic.ttf",
         "Caveat[wght].ttf": "ofl/caveat/Caveat%5Bwght%5D.ttf"}
os.makedirs(FD, exist_ok=True)
for _name, _path in FONTS.items():
    if not os.path.exists(os.path.join(FD, _name)):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/google/fonts/main/" + _path, os.path.join(FD, _name))
INTER = "/usr/share/fonts/opentype/inter"
W, H = 1280, 720
r1 = r3 = out = None
gx = gy = 0.0
rng = np.random.default_rng(7)


def serif(size, italic=False):
    return ImageFont.truetype(os.path.join(FD, "InstrumentSerif-Italic.ttf" if italic else "InstrumentSerif-Regular.ttf"), size)


def hand(size, weight=600):
    f = ImageFont.truetype(os.path.join(FD, "Caveat[wght].ttf"), size)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


def inter(size, name="Inter-Medium.otf"):
    return ImageFont.truetype(os.path.join(INTER, name), size)


def photo(path, crop=None):
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
    return im.resize((W, H), Image.LANCZOS) if crop is None else im


def grain(im, amount=4.0):
    """A little film grain, so it reads as a photograph (monochrome, fine)."""
    a = np.asarray(im).astype(np.float32)
    n = rng.normal(0, amount, a.shape[:2])[..., None]
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


def soft_shadow(base, draw_fn, radius=14, opacity=150):
    """Draw text twice: a blurred dark copy first (legibility without an outline), then the text."""
    sh = Image.new("L", base.size, 0)
    draw_fn(ImageDraw.Draw(sh), fill=255)
    sh = sh.filter(ImageFilter.GaussianBlur(radius))
    dark = Image.new("RGB", base.size, (8, 7, 6))
    base.paste(dark, (0, 0), sh.point(lambda v: v * opacity // 255))


INK_LIGHT = (243, 237, 226)       # warm off-white
RED = (205, 52, 44)               # the film's war red, a touch deeper


# ------------------------------------------------------------------ A: Embers (photograph + editorial serif)
def design_a():
    im = Image.open(r1).convert("RGB").resize((W, H), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    xx = np.linspace(0, 1, W)[None, :, None]
    a *= 0.42 + 0.58 * np.clip(xx / 0.62, 0, 1) ** 1.4
    im = grain(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), 3.5)
    f1, f2 = serif(90), serif(90, italic=True)
    x, y = 62, 84
    lines = [("Every one of your", f1), ("ancestors", f1), ("survived this.", f2)]

    def draw(d, fill=None):
        yy = y
        for t, f in lines:
            d.text((x, yy), t, font=f, fill=fill if fill is not None else INK_LIGHT)
            yy += 86
    soft_shadow(im, draw, radius=18, opacity=170)
    draw(ImageDraw.Draw(im))
    d = ImageDraw.Draw(im)
    d.text((x + 3, 44), "2,000 YEARS  ·  640 MILLION LIVES", font=inter(21, "Inter-SemiBold.otf"), fill=(236, 128, 112))
    return im


# ------------------------------------------------------------------ B: The page (light, paper and photograph)
def design_b():
    page = Image.new("RGB", (W, H), (237, 231, 219))
    a = np.asarray(page).astype(np.float32)
    a += rng.normal(0, 2.2, a.shape[:2])[..., None]                         # paper tooth
    yy, xx = np.mgrid[0:H, 0:W]
    a *= (1 - 0.06 * ((xx / W - 0.25) ** 2 + (yy / H - 0.5) ** 2))[..., None]
    page = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    pw = 548                                                                # the photograph: right side
    src = Image.open(r1).convert("RGB")
    sw, sh = src.size
    cw = int(sh * (W - pw) / H)                                             # crop the source to the panel's shape
    cx = int(sw * 0.62)                                                     # centred on the column
    x0 = max(0, min(sw - cw, cx - cw // 2))
    ph = grain(src.crop((x0, 0, x0 + cw, sh)).resize((W - pw, H), Image.LANCZOS), 3.5)
    page.paste(ph, (pw, 0))
    d = ImageDraw.Draw(page)
    d.line([(pw, 0), (pw, H)], fill=(20, 18, 16), width=2)
    ink = (22, 20, 18)
    x = 56
    size = 120
    while serif(size).getlength("640,000,000") > pw - x - 34:
        size -= 2
    d.text((x, 92 + (120 - size) * 0.5), "640,000,000", font=serif(size), fill=ink)
    d.text((x + 2, 222), "lifted away.", font=serif(78, italic=True), fill=ink)
    d.line([(x + 4, 352), (x + 84, 352)], fill=RED, width=5)
    d.text((x + 2, 380), "You're still here.", font=serif(78, italic=True), fill=RED)
    d.text((x + 4, H - 74), "2,000 YEARS OF CATASTROPHE, IN SAND", font=inter(19, "Inter-SemiBold.otf"),
           fill=(110, 102, 92))
    return page


# ------------------------------------------------------------------ C: Annotated (macro + a hand-drawn circle)
def marker_circle(d, cx, cy, rx, ry, width, col, rot=-0.25):
    """A circle drawn in one quick stroke: slightly wobbly, not closed, the end overshooting the start."""
    pts = []
    n = 120
    a0 = 1.9
    for i in range(n + 1):
        t = a0 + (2 * math.pi + 0.55) * i / n
        wob = 1 + 0.035 * math.sin(3 * t + 0.7) + 0.02 * math.sin(7 * t)
        r_x, r_y = rx * wob, ry * wob * (1 + 0.06 * i / n)                   # the stroke drifts as it closes
        x, y = r_x * math.cos(t), r_y * math.sin(t)
        pts.append((cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot)))
    for k in range(len(pts) - 1):                                           # taper at both ends
        u = k / (len(pts) - 1)
        w = max(1, int(round(width * min(1.0, 6 * u, 6 * (1 - u)) ** 0.5)))
        d.line([pts[k], pts[k + 1]], fill=col, width=w)
        d.ellipse([pts[k][0] - w / 2, pts[k][1] - w / 2, pts[k][0] + w / 2, pts[k][1] + w / 2], fill=col)


def marker_arrow(d, p0, p1, width, col):
    """A quick curved stroke from p0 to p1 with two short barbs."""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2 + (y1 - y0) * 0.18, (y0 + y1) / 2 - (x1 - x0) * 0.18
    pts = [((1 - t) ** 2 * x0 + 2 * (1 - t) * t * mx + t * t * x1, (1 - t) ** 2 * y0 + 2 * (1 - t) * t * my + t * t * y1)
           for t in np.linspace(0, 1, 40)]
    d.line(pts, fill=col, width=width, joint="curve")
    ang = math.atan2(y1 - pts[-4][1], x1 - pts[-4][0])
    for s in (-1, 1):
        a = ang + math.pi + s * 0.5
        d.line([(x1, y1), (x1 + 26 * math.cos(a), y1 + 26 * math.sin(a))], fill=col, width=width)


def design_c():
    im = grain(Image.open(r3).convert("RGB").resize((W, H), Image.LANCZOS), 3.0)
    ss = 3                                                                  # draw the marks at 3x, then shrink
    big = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    white = (250, 250, 247, 255)
    marker_circle(d, gx * ss, gy * ss, 66 * ss, 60 * ss, 7 * ss, white)
    # the note, up and to the left of the grain, and an arrow down to the circle
    nx, ny = gx - 10, gy - 270
    note = Image.new("RGBA", (900 * ss, 260 * ss), (0, 0, 0, 0))
    ImageDraw.Draw(note).text((10 * ss, 10 * ss), "your ancestor", font=hand(124 * ss, 700), fill=white)
    note = note.rotate(4, resample=Image.BICUBIC, expand=False)
    big.alpha_composite(note, (int(nx * ss), int(ny * ss)))
    marker_arrow(d, ((nx + 70) * ss, (ny + 152) * ss), ((gx + 22) * ss, (gy - 72) * ss), 7 * ss, white)
    marks = big.resize((W, H), Image.LANCZOS)
    sh = marks.split()[3].filter(ImageFilter.GaussianBlur(6)).point(lambda v: v * 120 // 255)
    im.paste(Image.new("RGB", (W, H), (10, 8, 6)), (0, 0), sh)
    im = im.convert("RGBA")
    im.alpha_composite(marks)
    im = im.convert("RGB")
    a = np.asarray(im).astype(np.float32)                                 # a little shade at the top for the line
    yy = np.linspace(0, 1, H)[:, None, None]
    a *= 0.5 + 0.5 * np.clip(yy / 0.28, 0, 1) ** 1.2
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.text((48, 34), "640 million were lifted away.", font=serif(58, italic=True), fill=INK_LIGHT)
    return im


if __name__ == "__main__":
    r1, r3, gx, gy, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    for name, fn in (("A_embers", design_a), ("B_page", design_b), ("C_annotated", design_c)):
        im = fn()
        im.save(os.path.join(out, f"thumb_{name}.png"))
        im.convert("RGB").save(os.path.join(out, f"thumb_{name}.jpg"), quality=92)
        print("wrote", name)
