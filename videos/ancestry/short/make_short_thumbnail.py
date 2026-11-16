"""The Short's thumbnail (1080x1920): a vertical still rendered from the film .blend, finished like a photograph,
with the Short's headline on a darkened band at the top.

    blender -b every_grain_of_us.blend -P render_vertical_still.py -- OUT_DIR 5200 96 1080 1920   # ~3 min on CPU
    python3 make_short_thumbnail.py OUT_DIR/v_5200.png short_thumbnail.png
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

src, out = sys.argv[1:3]
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "..", "..", "..", ".claude", "skills", "sandglass", "assets", "fonts")
im = Image.open(src).convert("RGB")
W, H = im.size
a = np.asarray(im).astype(np.float32) / 255
g = a.mean(2, keepdims=True)
a = ((a - g) * 0.97 + g) * 1.05                                     # photographic finish, as the film thumbnails
hi = np.clip((a.max(2) - 0.55) / 0.45, 0, 1)
glow = np.asarray(Image.fromarray((hi * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(20))).astype(np.float32) / 255
a += 0.10 * glow[..., None] * np.array([1.0, 0.72, 0.45], np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / np.sqrt(2)
a *= (1 - 0.30 * np.clip(r, 0, 1) ** 2.2)[..., None]
a *= (1 - 0.72 * np.clip(1 - yy / 560, 0, 1) ** 1.5)[..., None]       # darker band for the headline
rng = np.random.default_rng(3)
n = rng.normal(0, 1, (H, W)).astype(np.float32) * 0.012
a = np.clip(0.012 + a * 0.988 + n[..., None], 0, 1)
img = Image.fromarray((a * 255).astype(np.uint8)).convert("RGBA")

f = ImageFont.truetype(os.path.join(FONTS, "barlow-condensed-latin-800-normal.woff"), 150)
lines = [("200,000 YEARS OF", (250, 248, 242, 255)), ("HUMAN MIGRATION", (132, 236, 216, 255))]
lay = Image.new("RGBA", img.size, (0, 0, 0, 0)); sh = Image.new("L", img.size, 0)
d, ds = ImageDraw.Draw(lay), ImageDraw.Draw(sh)
y = 150
for s, col in lines:
    size = 150
    while d.textlength(s, font=f) > W - 80:
        size -= 2; f = ImageFont.truetype(f.path, size)
    d.text((W / 2, y), s, font=f, fill=col, anchor="mt"); ds.text((W / 2, y + 4), s, font=f, fill=255, anchor="mt")
    y += f.getbbox(s, anchor="mt")[3] + 8
sh = sh.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.75))
base = Image.new("RGBA", img.size, (0, 0, 0, 0)); base.paste((0, 0, 0, 255), (0, 0), sh)
img.alpha_composite(base); img.alpha_composite(lay)
img.convert("RGB").save(out, optimize=True)
print("wrote", out)
