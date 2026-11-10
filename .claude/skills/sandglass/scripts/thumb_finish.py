"""Thumbnail finishing in the Sandglass house style: make a render read as a photograph of a real model,
and set at most one small caption like a museum label. No headline type, no outlines, no glow.

    import thumb_finish as F
    im = F.photo(F.load("exit.png"))                             # 1280x720, photographic finish
    F.label(im, (64, 662), "c. 55,000 years ago", F.mono(40))     # a date or a scale, never a slogan
    im.save("thumbnail.png")

The Latin subsets of the fonts have no "≈": write "about".
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts") + "/"
SERIF = FD + "newsreader-latin-{w}-{s}.woff"          # Newsreader (OFL): 400 normal/italic, 500 normal
MONO = FD + "ibm-plex-mono-latin-{w}-normal.woff"       # IBM Plex Mono (OFL): 400, 500
W, H = 1280, 720
PAPER = (241, 236, 225)


def serif(size, w=400, italic=False):
    return ImageFont.truetype(SERIF.format(w=w, s="italic" if italic else "normal"), size)


def mono(size, w=400):
    return ImageFont.truetype(MONO.format(w=w), size)


def load(path, crop=None):
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
    return im.resize((W, H), Image.LANCZOS)


def photo(im, sat=0.96, warm=0.0, vignette=0.28, grain=0.018, halation=0.10, seed=1, lift=0.012, exposure=1.0):
    """Make a render read as a photograph: slightly lower saturation, a soft glow round the brightest
    highlights (lens halation), a natural vignette, lifted blacks and fine film grain."""
    a = np.asarray(im).astype(np.float32) / 255
    g = a.mean(2, keepdims=True)
    a = ((a - g) * sat + g) * exposure
    if warm:
        a = a * np.array([1 + warm, 1 + warm * 0.35, 1 - warm * 0.6], np.float32)
    if halation:
        hi = np.clip((a.max(2) - 0.55) / 0.45, 0, 1)
        glow = np.asarray(Image.fromarray((hi * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(14))).astype(np.float32) / 255
        a = a + halation * glow[..., None] * np.array([1.0, 0.72, 0.45], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / np.sqrt(2)
    a = a * (1 - vignette * np.clip(r, 0, 1) ** 2.2)[..., None]
    a = lift + a * (1 - lift)
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (H, W)).astype(np.float32)
    n = np.asarray(Image.fromarray(((n * 0.25 + 0.5).clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))).astype(np.float32) / 255 - 0.5
    lum = a.mean(2, keepdims=True)
    a = a + grain * 4 * n[..., None] * (0.35 + 0.65 * np.sqrt(np.clip(lum, 0, 1)))
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def label(im, xy, s, font, fill=PAPER, anchor="la", shadow=0.55, alpha=1.0, spacing=0):
    """Text with only a soft, tight shadow (like a caption set over a photo), no outline."""
    sh = Image.new("L", im.size, 0)
    ImageDraw.Draw(sh).text((xy[0], xy[1] + 1), s, font=font, fill=255, anchor=anchor, spacing=spacing)
    sh = sh.filter(ImageFilter.GaussianBlur(max(font.size * 0.08, 1.5)))
    im.paste(Image.new("RGB", im.size, (0, 0, 0)), (0, 0), sh.point(lambda v: int(v * shadow)))
    if alpha >= 1:
        ImageDraw.Draw(im).text(xy, s, font=font, fill=fill, anchor=anchor, spacing=spacing)
    else:
        m = Image.new("L", im.size, 0)
        ImageDraw.Draw(m).text(xy, s, font=font, fill=int(255 * alpha), anchor=anchor, spacing=spacing)
        im.paste(Image.new("RGB", im.size, fill), (0, 0), m)
    return ImageDraw.Draw(im).textbbox(xy, s, font=font, anchor=anchor)


def leader(im, pts, fill=PAPER, width=2, alpha=0.9, dot=4):
    """A thin hand-placed leader line ending in a small dot."""
    m = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(m)
    d.line([(x * 2, y * 2) for x, y in pts], fill=255, width=width * 2)
    x, y = pts[-1]
    d.ellipse((x * 2 - dot * 2, y * 2 - dot * 2, x * 2 + dot * 2, y * 2 + dot * 2), fill=255)
    m = m.resize((W, H), Image.LANCZOS).point(lambda v: int(v * alpha))
    sh = m.filter(ImageFilter.GaussianBlur(2)).point(lambda v: int(v * 0.6))
    im.paste(Image.new("RGB", im.size, (0, 0, 0)), (0, 0), sh)
    im.paste(Image.new("RGB", im.size, fill), (0, 0), m)


def shade(im, bottom=None, left=None, top=None, strength=0.6):
    a = np.asarray(im).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = np.ones((H, W), np.float32)
    if bottom:
        m *= 1 - strength * np.clip((yy - (H - bottom)) / bottom, 0, 1) ** 1.6
    if top:
        m *= 1 - strength * np.clip(1 - yy / top, 0, 1) ** 1.6
    if left:
        m *= 1 - strength * np.clip(1 - xx / left, 0, 1) ** 1.6
    return Image.fromarray((a * m[..., None]).astype(np.uint8))
