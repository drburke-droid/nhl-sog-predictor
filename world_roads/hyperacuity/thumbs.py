"""Three thumbnails for A/B/C testing (1280x720).

    python thumbs.py   -> out/thumb_A.png (1 pixel, magnified), thumb_B.png (the 1/8-pixel challenge),
                          thumb_C.png (the cone mosaic: smaller than your own pixels)
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw
import ui
from explain import mosaic, lsf

TW, TH = 1280, 720
OUT = ui.OUT
BG = ui.BG


def f(n, w="Bold"):
    return ui.ST.font(n, w)


def headline(d, xy, s, size, col, anchor="la"):
    d.text(xy, s, font=f(size, "Bold"), fill=col, stroke_width=max(5, size // 16), stroke_fill=(0, 0, 0), anchor=anchor)


def thumb_A():
    """A long line with a real 1-pixel step, and a loupe that proves it."""
    img = np.zeros((TH, TW, 3), np.uint8); img[:] = BG
    y0 = 300
    img[y0:y0 + 8, 60:640] = 245
    img[y0 + 1:y0 + 9, 640:1220] = 245
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    cx, cy, r, z = 880, 505, 180, 12
    src = np.asarray(im).astype(np.int16)
    yy, xx = np.mgrid[-r:r, -r:r] + 0.5
    sx = np.clip((640 + xx / z).astype(int), 0, TW - 1); sy = np.clip((y0 + 4.5 + yy / z).astype(int), 0, TH - 1)
    crop = src[sy, sx]
    grid = (np.abs((640 + xx / z) - np.round(640 + xx / z)) * z < 1) | (np.abs((y0 + 4.5 + yy / z) - np.round(y0 + 4.5 + yy / z)) * z < 1)
    crop[grid] = (crop[grid] * 0.5 + 25).astype(np.int16)
    mask = Image.new("L", (2 * r, 2 * r), 0); ImageDraw.Draw(mask).ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
    d.line([(643, y0 + 12), (cx - r * 0.72, cy - r * 0.72)], fill=ui.GOLD, width=5)
    d.ellipse([634, y0 - 4, 650, y0 + 14], outline=ui.GOLD, width=3)
    im.paste(Image.fromarray(crop.astype(np.uint8)), (cx - r, cy - r), mask)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ui.GOLD, width=8)
    headline(d, (cx + r - 30, cy + r - 60), "1 px", 58, ui.GOLD)
    headline(d, (60, 50), "1 PIXEL", 160, ui.GOLD)
    headline(d, (64, 390), "YOU CAN", 88, (255, 255, 255))
    headline(d, (64, 485), "SEE IT.", 88, (255, 255, 255))
    im.save(os.path.join(OUT, "thumb_A.png"))


def thumb_B():
    """The challenge: four vernier pairs, one shifted by 1/8 pixel (really)."""
    big = ui.Patch(0, 0, TW, TH, s=1)
    xs, yc, bw, bh, gap = [700, 840, 980, 1120], 400, 6, 150, 6
    for i, x in enumerate(xs):
        d_ = 0.125 if i == 2 else 0.0
        big.rect(x - bw / 2, yc - gap / 2 - bh, x + bw / 2, yc - gap / 2)
        big.rect(x - bw / 2 + d_, yc + gap / 2, x + bw / 2 + d_, yc + gap / 2 + bh)
    im = Image.fromarray(big.srgb())
    d = ImageDraw.Draw(im)
    for i, x in enumerate(xs):
        d.text((x, yc + bh + 50), "ABCD"[i], font=f(54, "Bold"), fill=(170, 186, 206), anchor="mm")
    headline(d, (50, 90), "⅛", 300, ui.GOLD)
    headline(d, (50, 420), "PIXEL", 110, ui.GOLD)
    headline(d, (690, 50), "WHICH ONE", 84, (255, 255, 255))
    headline(d, (690, 135), "IS OFF?", 84, (255, 255, 255))
    im.save(os.path.join(OUT, "thumb_B.png"))


def thumb_C():
    """The cone mosaic, lit by a blurred line: you see shifts smaller than your own 'pixels'."""
    sp = 92.0
    pts = mosaic(-sp, -sp, TW + sp, TH + sp, sp, seed=7)
    xc = 840.0
    e = lsf((pts[:, 0] - xc - np.where(pts[:, 1] > 360, 9.0, 0.0)) / (2 * sp))
    im = Image.new("RGB", (TW, TH), BG)
    # the light falling on the cones
    X = (np.arange(TW) - xc) / (2 * sp)
    Y = np.arange(TH)
    prof = np.where((Y > 360)[:, None], lsf(X - 9.0 / (2 * sp))[None, :], lsf(X)[None, :])
    glow = (prof[..., None] * np.array([255, 170, 70]) * 0.35).astype(np.uint8)
    im = Image.fromarray(np.clip(np.asarray(im).astype(np.int16) + glow, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    rr = sp * 0.44
    for (x, y, _), v in zip(pts, e ** 1.3):
        col = tuple(int(c) for c in (np.array([22, 26, 36]) * (1 - v) + np.array([255, 205, 120]) * v))
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col, outline=(70, 82, 104), width=2)
    a = np.clip(1 - np.arange(TW) / 760.0, 0, 1) ** 1.2 * 0.78           # darken the left, smoothly, for the text
    arr = np.asarray(im).astype(np.float32)
    arr = arr * (1 - a[None, :, None]) + np.array(BG, np.float32) * a[None, :, None]
    im = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    headline(d, (50, 60), "YOUR EYES", 92, (255, 255, 255))
    headline(d, (50, 160), "SEE SMALLER", 92, (255, 255, 255))
    headline(d, (50, 260), "THAN THEIR", 92, (255, 255, 255))
    headline(d, (50, 360), "OWN PIXELS", 92, ui.GOLD)
    d.text((54, 520), "these are cones, magnified", font=f(34, "SemiBold"), fill=(170, 186, 206))
    im.save(os.path.join(OUT, "thumb_C.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    thumb_A(); thumb_B(); thumb_C()
    print("wrote thumbnails")
