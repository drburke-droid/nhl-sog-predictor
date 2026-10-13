"""Static graphics for the blog post (Squarespace has no table block).

    python blog_graphics.py   -> out/blog/checkpoints_map.png, out/blog/light_ladder.png
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import render
from proj import W, H

OUT = os.path.join(render.OUT, "blog")
FD = "/usr/share/fonts/opentype/inter"
INK = (236, 242, 250)
INK2 = (160, 178, 200)
MUTED = (110, 128, 150)
GOLD = (255, 176, 66)
CYAN = (60, 170, 240)


def font(name, size):
    return ImageFont.truetype(os.path.join(FD, name), size)


def checkpoints_map():
    R = render.Renderer()
    R.static = []
    R._dynamic_text = lambda t, labels: (None, None)
    R.L["j_t"] = np.zeros(0)                       # no jump rings
    render.DURATION, keep = 1e9, render.DURATION   # no finale pulse
    t_end = keep
    R.reset_to(t_end - 1e-9)
    img = R.frame(t_end)
    render.DURATION = keep
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    stops = [(0, "Northern Alaska"), (40, "Southern Ontario"), (80, "Tierra del Fuego"), (120, "European Russia"),
             (160, "Mozambique"), (200, "Korea"), (240, "Eastern Himalaya"), (280, "New Zealand")]
    # label offsets (dx, dy, anchor) chosen so labels sit on open ocean / clear of each other
    offs = {0: (22, 18, "la"), 40: (26, -66, "la"), 80: (-26, -40, "ra"), 120: (24, -70, "la"),
            160: (28, -10, "la"), 200: (28, -60, "la"), 240: (-28, 30, "ra"), 280: (-26, -66, "ra")}
    for t, place in stops:
        x, y, _ = R.dot_pos(min(t, t_end - 1e-6))
        d.ellipse([x - 13, y - 13, x + 13, y + 13], outline=(255, 255, 255, 230), width=3)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255, 255))
        dx, dy, anc = offs[t]
        m, s = divmod(t, 60)
        tx, ty = x + dx, y + dy
        label = f"{m}:{s:02d}"
        f1, f2 = font("Inter-Bold.otf", 40), font("Inter-Medium.otf", 24)
        w1 = d.textlength(label, font=f1); w2 = d.textlength(place, font=f2)
        bw = max(w1, w2) + 28
        bx0 = tx - 14 if anc == "la" else tx - bw + 14
        d.rounded_rectangle([bx0, ty - 8, bx0 + bw, ty + 82], 10, fill=(4, 8, 16, 190))
        d.text((tx, ty), label, font=f1, fill=INK, anchor=anc)
        d.text((tx, ty + 48), place, font=f2, fill=INK2, anchor=anc)
    d.text((58, 40), "Where the light is along the way", font=font("Inter-SemiBold.otf", 46), fill=INK)
    d.text((60, 100), "Every road on Earth at the speed of light: 4 minutes 40 seconds", font=font("Inter-Regular.otf", 25), fill=INK2)
    d.text((60, 1030), "Road data © OpenStreetMap contributors via Overture Maps · Thought experiment; all figures are estimates.",
           font=font("Inter-Regular.otf", 16), fill=MUTED)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "checkpoints_map.png")
    im.save(path)
    return path


def light_ladder():
    im = Image.new("RGB", (W, H), (6, 10, 18))
    d = ImageDraw.Draw(im, "RGBA")
    d.text((58, 40), "How long light takes", font=font("Inter-SemiBold.otf", 54), fill=INK)
    d.text((60, 112), "at 299,792 km per second", font=font("Inter-Regular.otf", 28), fill=INK2)
    rows = [("Every street in Calgary", "~11,100 km, with backtracking", 0.037, "0.037 s"),
            ("Once around the equator", "40,075 km", 0.1337, "0.13 s"),
            ("Earth to the Moon", "384,400 km", 1.28, "1.3 s"),
            ("Every road on Earth", "~84 million km, with backtracking", 280.0, "4 min 40 s"),
            ("The Sun to your eye", "~150 million km", 499.0, "8 min 20 s")]
    x0, x1 = 760, 1680                       # plot area: 0.01 s .. 1,000 s, log scale
    lo, hi = -2.0, 3.0
    X = lambda v: x0 + (np.log10(v) - lo) / (hi - lo) * (x1 - x0)
    top, step = 250, 140
    # recessive grid + ticks
    for p, lab in zip(range(-2, 4), ["0.01 s", "0.1 s", "1 s", "10 s", "100 s", "1,000 s"]):
        gx = X(10.0 ** p)
        d.line([(gx, top - 40), (gx, top + step * 5 - 40)], fill=(255, 255, 255, 22), width=1)
        d.text((gx, top + step * 5 - 20), lab, font=font("Inter-Regular.otf", 22), fill=MUTED, anchor="ma")
    for i, (name, sub, v, txt) in enumerate(rows):
        y = top + i * step
        hl = name == "Every road on Earth"
        d.text((120, y - 22), name, font=font("Inter-SemiBold.otf" if hl else "Inter-Medium.otf", 36), fill=INK)
        d.text((120, y + 24), sub, font=font("Inter-Regular.otf", 24), fill=INK2)
        bx = X(v)
        col = GOLD if hl else CYAN
        d.rounded_rectangle([x0, y - 14, bx, y + 14], 4, fill=col)
        d.text((bx + 18, y), txt, font=font("Inter-Bold.otf" if hl else "Inter-SemiBold.otf", 34), fill=INK, anchor="lm")
    d.text((x0, top + step * 5 + 30), "Logarithmic scale: each gridline is 10× longer than the one before.",
           font=font("Inter-Regular.otf", 22), fill=MUTED)
    d.text((60, 1030), "Road distances include ~30% backtracking (Calgary: 45%). All figures are estimates.",
           font=font("Inter-Regular.otf", 16), fill=MUTED)
    path = os.path.join(OUT, "light_ladder.png")
    os.makedirs(OUT, exist_ok=True)
    im.save(path)
    return path


if __name__ == "__main__":
    print(checkpoints_map())
    print(light_ladder())
