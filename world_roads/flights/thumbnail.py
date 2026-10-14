"""YouTube thumbnail (1280x720) for "Every flight on Earth": a mid-run frame of the light-speed plane."""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flights_render as FR  # noqa: E402

FD = "/usr/share/fonts/opentype/inter"
T = float(sys.argv[1]) if len(sys.argv) > 1 else 334.0


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


R = FR.FlightRenderer()
R.reset_to(T - 3.0)
for tt in np.arange(T - 3.0, T + 1e-9, 1 / 30):
    E, _ = R.frame_light(float(tt), 0.88)
img = Image.fromarray(R.finish(E, [])).resize((1280, 720), Image.LANCZOS)
# darken the top-left so the type reads
grad = np.zeros((720, 1280), np.float32)
yy, xx = np.mgrid[0:720, 0:1280]
grad = np.clip(1 - (xx / 1280 * 0.9 + yy / 720 * 1.3), 0, 1) * 0.75
arr = np.asarray(img).astype(np.float32) * (1 - grad[..., None])
img = Image.fromarray(arr.astype(np.uint8))
lines = [("EVERY FLIGHT", font("Inter-Black.otf", 104), (255, 255, 255)),
         ("ON EARTH", font("Inter-Black.otf", 104), (255, 255, 255)),
         ("AT LIGHT SPEED", font("Inter-Black.otf", 74), (255, 186, 80))]
ys = [34, 140, 252]
shadow = Image.new("RGBA", img.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(shadow)
for (s, f, c), y in zip(lines, ys):
    sd.text((48, y + 4), s, font=f, fill=(0, 0, 0, 230))
shadow = shadow.filter(ImageFilter.GaussianBlur(6))
img = Image.alpha_composite(img.convert("RGBA"), shadow)
d = ImageDraw.Draw(img)
for (s, f, c), y in zip(lines, ys):
    d.text((44, y), s, font=f, fill=c)
out = os.path.join(FR.OUT, "every_flight_thumbnail.png")
img.convert("RGB").save(out)
print(out)
