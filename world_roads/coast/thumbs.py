"""Thumbnails (1280x720) for the coast film, from its own text-free frames (python film.py clean 11.9 77.0 123.6),
with the roads film's thumbnail fonts (../thumb_fonts: Anton, Permanent Marker).

    python thumbs.py   -> out/coast_thumb_A.png  half the world gold, the fish in Africa: "light needs 1:18"
                          out/coast_thumb_B.png  the rule, close up on the Bow and the Elbow in Calgary
                          out/coast_thumb_C.png  the whole world gold
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONTS = os.path.join(os.path.dirname(HERE), "thumb_fonts")
TW, TH, S = 1280, 720, 2
GOLD = (255, 196, 84)
RED = (255, 44, 44)


def base(name, crop=None):
    im = Image.open(os.path.join(OUT, name)).convert("RGB")
    if crop:
        im = im.crop(crop)
    return im.resize((TW * S, TH * S), Image.LANCZOS).convert("RGBA")


def anton(size):
    return ImageFont.truetype(os.path.join(FONTS, "Anton-Regular.ttf"), size * S)


def marker(size):
    return ImageFont.truetype(os.path.join(FONTS, "PermanentMarker-Regular.ttf"), size * S)


def shadow_text(im, xy, text, font, fill, anchor="la", blur=10, dark=210):
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((xy[0] + 3 * S, xy[1] + 4 * S), text, font=font, fill=(0, 0, 0, dark), anchor=anchor)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(im).text(xy, text, font=font, fill=fill, anchor=anchor)


def darken_top(im, h=0.42, a=170):
    g = Image.new("L", (1, 256))
    for y in range(256):
        g.putpixel((0, y), int(a * max(0.0, 1 - y / 255 / h) ** 1.4) if y / 255 < h else 0)
    m = g.resize(im.size)
    black = Image.new("RGBA", im.size, (0, 0, 0, 255))
    black.putalpha(m)
    im.alpha_composite(black)


def ring(d, cx, cy, rx, ry, width, color, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(-0.25, 2 * np.pi + 0.35, 260)
    wob = 1 + 0.035 * np.sin(3 * t + rng.random() * 6) + 0.02 * np.sin(7 * t + rng.random() * 6)
    pts = [(S * (cx + rx * wob[i] * np.cos(t[i])), S * (cy + ry * wob[i] * np.sin(t[i]))) for i in range(len(t))]
    for i in range(len(pts) - 1):
        w = width * S * (0.55 + 0.45 * np.sin(np.pi * i / (len(pts) - 1)))
        d.line([pts[i], pts[i + 1]], fill=color, width=max(1, int(w)))


def save(im, name):
    im = im.convert("RGB").resize((TW, TH), Image.LANCZOS)
    p = os.path.join(OUT, name)
    im.save(p, optimize=True)
    print(p)


def thumb_a():
    im = base("clean_0077.00.png", crop=(150, 90, 1830, 1035))
    darken_top(im)
    shadow_text(im, (40 * S, 22 * S), "EVERY SHORE", anton(118), (255, 255, 255))
    shadow_text(im, (40 * S, 150 * S), "ON EARTH", anton(118), GOLD)
    d = ImageDraw.Draw(im)
    # the fish is at the Red Sea in this frame (python -c "film.Film().world.fish_at(...)": 739, 241 here): circle it
    ring(d, 739, 241, 62, 52, 9, RED, seed=3)
    shadow_text(im, (812 * S, 268 * S), "light needs", marker(58), RED, blur=6, dark=170)
    shadow_text(im, (850 * S, 338 * S), "1:18", marker(104), (255, 255, 255), blur=7, dark=210)
    save(im, "coast_thumb_A.png")


def thumb_b():
    im = base("clean_0011.90.png", crop=(500, 220, 1500, 783))
    darken_top(im, 0.5, 190)
    shadow_text(im, (40 * S, 22 * S), "TURN RIGHT AT", anton(104), (255, 255, 255))
    shadow_text(im, (40 * S, 140 * S), "EVERY FORK", anton(104), GOLD)
    shadow_text(im, (60 * S, 600 * S), "...every river on Earth, at light speed", marker(40), RED, blur=6, dark=170)
    save(im, "coast_thumb_B.png")


def thumb_c():
    im = base("clean_0123.60.png", crop=(150, 90, 1830, 1035))
    darken_top(im, 0.4, 150)
    shadow_text(im, (TW * S / 2, 20 * S), "EVERY COAST, LAKE & RIVERBANK", anton(84), (255, 255, 255), anchor="ma")
    shadow_text(im, (TW * S / 2, 600 * S), "78 seconds at light speed", marker(52), RED, anchor="ma", blur=6, dark=180)
    save(im, "coast_thumb_C.png")


if __name__ == "__main__":
    thumb_a(); thumb_b(); thumb_c()
