"""
The lettering on the mixer's trim band and hub cap, as a generic brand.

The mixer model's own texture carries a real maker's name, model line and
conformity marks.  This writes a replacement with the same layout (so the
model's UV mapping still fits): "Kitchen Inc" where the name was, the speed
scale kept, everything else left blank.

    python make_lettering.py        # -> models/kitchenaid/kitchen_inc_lettering.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "models", "kitchenaid", "kitchenaid_writing1.png")
DST = os.path.join(HERE, "models", "kitchenaid", "kitchen_inc_lettering.png")
BRAND = "Kitchen Inc"
SPEED_SCALE = (1200, 1700)                    # columns of the original that hold the 1-10 speed scale
NAME_BOXES = [(2894, 35, 4078, 134),          # the large name: centre of the band, and the hub cap
              (133, 52, 434, 100)]            # the small name at the back of the band


def font(size):
    for path in ("/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "C:/Windows/Fonts/arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise SystemExit("no bold font found")


def main():
    src = np.asarray(Image.open(SRC).convert("RGBA"))
    h, w = src.shape[:2]
    alpha = Image.new("L", (w, h), 0)
    alpha.paste(Image.fromarray(src[:, SPEED_SCALE[0]:SPEED_SCALE[1], 3]), (SPEED_SCALE[0], 0))
    d = ImageDraw.Draw(alpha)
    for x0, y0, x1, y1 in NAME_BOXES:
        size = 10
        while True:                               # largest size whose ink fits the box
            l, t, r, b = d.textbbox((0, 0), BRAND, font=font(size + 2))
            if r - l > x1 - x0 or b - t > y1 - y0:
                break
            size += 2
        l, t, r, b = d.textbbox((0, 0), BRAND, font=font(size))
        d.text(((x0 + x1) / 2 - (l + r) / 2, (y0 + y1) / 2 - (t + b) / 2), BRAND, font=font(size), fill=255)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = 232
    out[..., 3] = np.asarray(alpha)
    Image.fromarray(out, "RGBA").save(DST, optimize=True)
    print("wrote", DST)


if __name__ == "__main__":
    main()
