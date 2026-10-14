"""YouTube thumbnail for the draw/erase cut: half the world's routes already erased, text over the
dark Americas.  python thumb_frames.py 300 && python thumbnail_v2.py"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FD = "/usr/share/fonts/opentype/inter"


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


base = Image.open(os.path.join(OUT, "thumbbase_300.png")).convert("RGB")
# punch up the network a little (thumbnails are viewed small)
a = np.asarray(base).astype(np.float32) / 255
a = np.clip(a * 1.25, 0, 1) ** 0.92
base = Image.fromarray((a * 255).astype(np.uint8)).resize((1280, 720), Image.LANCZOS)
yy, xx = np.mgrid[0:720, 0:1280]
shade = np.clip(1 - xx / 760, 0, 1) ** 1.5 * 0.55                     # darker on the left, behind the type
arr = np.asarray(base).astype(np.float32) * (1 - shade[..., None])
base = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")

VARIANTS = {
    "A": [("EVERY FLIGHT", font("Inter-Black.otf", 84), (255, 255, 255)),
          ("ON EARTH", font("Inter-Black.otf", 84), (255, 255, 255)),
          ("AT LIGHT SPEED", font("Inter-Black.otf", 60), (255, 186, 80))],
    "B": [("ONE PLANE.", font("Inter-Black.otf", 84), (255, 255, 255)),
          ("EVERY FLIGHT.", font("Inter-Black.otf", 84), (255, 255, 255)),
          ("LIGHT SPEED.", font("Inter-Black.otf", 84), (255, 186, 80))],
}
for k, lines in VARIANTS.items():
    img = base.copy()
    ys, y = [], 262 if k == "A" else 196                             # keep A's first line under the burst
    for s, f, c in lines:
        ys.append(y)
        y += int(f.size * 1.12)
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
    for (s, f, c), y in zip(lines, ys):
        sd.text((44, y + 5), s, font=f, fill=(0, 0, 0, 235))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(7)))
    d = ImageDraw.Draw(img)
    for (s, f, c), y in zip(lines, ys):
        d.text((40, y), s, font=f, fill=c)
    p = os.path.join(OUT, f"every_flight_thumbnail_v2{k}.png")
    img.convert("RGB").save(p)
    print(p)
