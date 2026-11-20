"""YouTube channel art for Calgary Vision Centre, "actually interesting waiting room videos".

The CVC mark (cvc_mark_source.png, 414x243) is rebuilt as a clean high-resolution shape: upsampled 10x,
re-thresholded, lightly smoothed, and made exactly symmetric (the original is 99.6% symmetric). It is set
in the light-speed films' gold on their night-map navy.

    python make_branding.py <4K world frame>   e.g. ../../world_roads/coast/out/clean_0123.60.png
                                               (python film.py clean 123.6 with RES=2: the whole world gold)
    -> profile_800.png        channel picture (YouTube shows it as a circle, 98 px on screen)
       banner_2560x1440.jpg   channel banner; the name and tagline sit inside the 1546x423 area every device shows
       banner_preview.jpg     the banner with the TV / desktop / phone crops outlined (not for upload)
       watermark_150.png      video watermark (white mark on transparent)
"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "world_roads", "thumb_fonts")
INTER = "/usr/share/fonts/opentype/inter"
GOLD_TOP = np.array([255, 214, 130], np.float32) / 255
GOLD_BOT = np.array([244, 158, 48], np.float32) / 255
NAVY_IN = np.array([13, 24, 42], np.float32) / 255
NAVY_OUT = np.array([3, 7, 14], np.float32) / 255


def mark_mask():
    """the mark as a smooth, symmetric, high-resolution coverage mask (1 = ink), cropped to its box"""
    a = np.asarray(Image.open(os.path.join(HERE, "cvc_mark_source.png")).convert("RGBA")).astype(np.float32) / 255
    ink = (1 - a[..., :3].mean(2)) * a[..., 3]
    k = 10
    big = cv2.resize(ink, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    ys, xs = np.nonzero(big > 0.5)
    cx = (xs.min() + xs.max()) / 2
    # mirror about the mark's centre line and average: exactly symmetric
    xx = np.arange(big.shape[1])
    src = np.clip(np.rint(2 * cx - xx).astype(int), 0, big.shape[1] - 1)
    big = 0.5 * (big + big[:, src])
    m = (big > 0.5).astype(np.float32)
    m = (cv2.GaussianBlur(m, (0, 0), 3.0) > 0.5).astype(np.float32)
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def mark_alpha(M, width):
    h = int(round(M.shape[0] * width / M.shape[1]))
    return cv2.resize(M, (width, h), interpolation=cv2.INTER_AREA)


def gold_mark(alpha):
    """RGBA float: the mark filled with a top-to-bottom gold gradient"""
    h, w = alpha.shape
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    rgb = GOLD_TOP * (1 - t) + GOLD_BOT * t
    return np.concatenate([np.broadcast_to(rgb, (h, w, 3)), alpha[..., None]], -1)


def paste(canvas, rgba, x, y, glow=0.0, glow_sigma=18):
    """alpha-composite a float RGBA patch onto a float RGB canvas at (x, y), with an optional gold glow"""
    h, w = rgba.shape[:2]
    if glow > 0:
        pad = int(glow_sigma * 4)
        a = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        a[pad:pad + h, pad:pad + w] = rgba[..., 3]
        g = cv2.GaussianBlur(a, (0, 0), glow_sigma) * glow
        y0, x0 = y - pad, x - pad
        sl = canvas[max(0, y0):y0 + a.shape[0], max(0, x0):x0 + a.shape[1]]
        gg = g[max(0, -y0):max(0, -y0) + sl.shape[0], max(0, -x0):max(0, -x0) + sl.shape[1]]
        sl += gg[..., None] * (GOLD_BOT * 0.9)
    reg = canvas[y:y + h, x:x + w]
    a = rgba[..., 3:4]
    canvas[y:y + h, x:x + w] = reg * (1 - a) + rgba[..., :3] * a


def radial_bg(w, h, inner=NAVY_IN, outer=NAVY_OUT, cx=0.5, cy=0.45, r=0.75):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.clip(np.hypot((xx / w - cx), (yy / h - cy)) / r, 0, 1) ** 1.3
    return inner * (1 - d[..., None]) + outer * d[..., None]


def to8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ profile picture
def profile(M):
    S = 800
    c = radial_bg(S, S, cy=0.5, r=0.72)
    w = 560
    al = mark_alpha(M, w)
    h = al.shape[0]
    paste(c, gold_mark(al), (S - w) // 2, (S - h) // 2 + 6, glow=0.55, glow_sigma=22)
    Image.fromarray(to8(c)).save(os.path.join(HERE, "profile_800.png"), optimize=True)
    # how it looks: the circle crop at YouTube's sizes
    im = Image.fromarray(to8(c)).convert("RGBA")
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, S - 1, S - 1], fill=255)
    im.putalpha(mask)
    prev = Image.new("RGBA", (S + 360, S), (255, 255, 255, 255))
    prev.alpha_composite(im, (0, 0))
    for i, sz in enumerate((176, 98, 48)):
        prev.alpha_composite(im.resize((sz, sz), Image.LANCZOS), (S + 60, 40 + sum((176, 98, 48)[:i]) + 40 * i))
    prev.convert("RGB").save(os.path.join(HERE, "profile_preview.png"))


# ------------------------------------------------------------------ watermark
def watermark(M):
    S = 150
    al = mark_alpha(M, 140)
    h = al.shape[0]
    out = np.zeros((S, S, 4), np.float32)
    y0 = (S - h) // 2
    out[y0:y0 + h, 5:5 + al.shape[1], :3] = 1.0
    out[y0:y0 + h, 5:5 + al.shape[1], 3] = al
    Image.fromarray(to8(out), "RGBA").save(os.path.join(HERE, "watermark_150.png"), optimize=True)


# ------------------------------------------------------------------ banner
def banner(M, frame_path):
    W, H = 2560, 1440
    fr = cv2.imread(frame_path)[..., ::-1].astype(np.float32) / 255
    fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
    # the gold world, dimmed; darker still behind the words, in the band every device shows
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    band = np.exp(-((yy - H / 2) / 260) ** 2) * np.exp(-((xx - W / 2) / 900) ** 2)
    c = fr * (0.62 - 0.40 * band[..., None])
    c += radial_bg(W, H, cy=0.5, r=0.9) * 0.35
    # the mark and the words, centred in the 1546x423 safe area
    mw = 360
    al = mark_alpha(M, mw)
    mh = al.shape[0]
    f_name = ImageFont.truetype(os.path.join(FONTS, "Anton-Regular.ttf"), 112)
    f_tag = ImageFont.truetype(os.path.join(INTER, "Inter-MediumItalic.otf"), 46)
    f_sub = ImageFont.truetype(os.path.join(INTER, "Inter-Regular.otf"), 30)
    name, tag, sub = "CALGARY VISION CENTRE", "Actually interesting waiting room videos", \
        "Eye science  ·  Light-speed maps of the whole world  ·  Calgary, Alberta"
    d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    tw = max(d0.textlength(name, font=f_name), d0.textlength(tag, font=f_tag), d0.textlength(sub, font=f_sub))
    gap = 64
    total = mw + gap + tw
    x0 = int((W - total) / 2)
    paste(c, gold_mark(al), x0, int(H / 2 - mh / 2), glow=0.45, glow_sigma=20)
    im = Image.fromarray(to8(c)).convert("RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    tx = x0 + mw + gap
    d.text((tx, H / 2 - 128), name, font=f_name, fill=(255, 255, 255, 255))
    d.text((tx + 4, H / 2 + 22), tag, font=f_tag, fill=(255, 205, 120, 255))
    d.text((tx + 4, H / 2 + 92), sub, font=f_sub, fill=(196, 206, 222, 255))
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh.putalpha(lay.getchannel("A").filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.8)))
    im.alpha_composite(sh)
    im.alpha_composite(lay)
    im = im.convert("RGB")
    im.save(os.path.join(HERE, "banner_2560x1440.jpg"), quality=92, optimize=True)
    # preview: the crops YouTube makes
    pv = im.copy()
    d = ImageDraw.Draw(pv)
    f = ImageFont.truetype(os.path.join(INTER, "Inter-SemiBold.otf"), 34)
    boxes = [((0, 0, W - 1, H - 1), "TV: the whole image", (120, 200, 255)),
             ((0, (H - 423) // 2, W - 1, (H + 423) // 2), "desktop: 2560 x 423", (255, 255, 255)),
             (((W - 1546) // 2, (H - 423) // 2, (W + 1546) // 2, (H + 423) // 2), "every device (phone): 1546 x 423", (255, 120, 120))]
    for (bx0, by0, bx1, by1), label, col in boxes:
        d.rectangle([bx0, by0, bx1, by1], outline=col, width=5)
        d.text((bx0 + 16, by0 + 10), label, font=f, fill=col)
    pv.resize((1280, 720), Image.LANCZOS).save(os.path.join(HERE, "banner_preview.jpg"), quality=90)


if __name__ == "__main__":
    M = mark_mask()
    profile(M)
    watermark(M)
    banner(M, sys.argv[1])
    for f in ("profile_800.png", "watermark_150.png", "banner_2560x1440.jpg"):
        p = os.path.join(HERE, f)
        print(f, Image.open(p).size, f"{os.path.getsize(p) / 1e6:.2f} MB")
