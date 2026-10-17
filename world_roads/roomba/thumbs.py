"""Three thumbnails for A/B/C testing (1280x720), drawn from the video's own maps (no captions), with a big Roomba.

    python thumbs.py     -> out/thumb_A.png (America, half driven), thumb_B.png (New York, half driven),
                            thumb_C.png (the last block, on a Florida barrier island)
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw
import video as V
import maps

TW, TH = 1280, 720
OUT = V.OUT
ST = V.ST


def f(n, w="Bold"):
    return ST.font(n, w)


def headline(d, xy, s, size, col, anchor="la"):
    d.text(xy, s, font=f(size), fill=col, stroke_width=max(5, size // 16), stroke_fill=(0, 0, 0), anchor=anchor)


def glow_roomba(im, x, y, ang, diam):
    """a big Roomba with a light-speed streak behind it and a gold halo"""
    a = np.zeros((im.height, im.width), np.float32)
    import cv2
    dx, dy = math.cos(ang), math.sin(ang)
    for k in range(40):
        u = k / 40
        cv2.circle(a, (int(x - dx * diam * 2.6 * u), int(y - dy * diam * 2.6 * u)), int(diam * 0.42 * (1 - 0.6 * u)), 1.0 - u, -1)
    a = cv2.GaussianBlur(a, (0, 0), diam * 0.12)
    g = np.zeros_like(a)
    cv2.circle(g, (int(x), int(y)), int(diam * 0.55), 1.0, -1)
    g = cv2.GaussianBlur(g, (0, 0), diam * 0.25)
    arr = np.asarray(im).astype(np.float32)
    arr = arr + (a * 0.85)[..., None] * np.array([255, 235, 190]) + (g * 0.9)[..., None] * np.array(V.GOLD)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    maps.draw_roomba(im, x, y, ang, diam=diam)
    return im


def thumb_A():
    """America about 2 minutes in: gold where the Roomba has been, blue where it hasn't."""
    U = maps.USA(seed=V.USA_SEED, scale=V.USA_SCALE, ox=V.USA_OX, oy=V.USA_OY)
    U.map.advance(U.block_t[len(U.block_t) // 2])                  # half the blocks driven
    fr = maps.tonemap(U.map.image(gamma=0.42, dim=1.5, fgamma=4.0))
    im = Image.fromarray(fr).crop((150, 70, 1830, 1015)).resize((TW, TH), Image.LANCZOS)
    im = glow_roomba(im, 820, 330, math.radians(-25), 190)
    d = ImageDraw.Draw(im)
    headline(d, (40, 470), "EVERY ROAD", 124, V.GOLD)
    headline(d, (44, 600), "IN AMERICA", 84, (255, 255, 255))
    im.save(os.path.join(OUT, "thumb_A.png"))


def thumb_B():
    """New York, part driven, Roomba mid-run."""
    c = V.NYC
    C = maps.City(c["prefix"], c["seed"], 15.0, c["cx"], c["cy"] + 1.0, 640, 380, w=TW, h=TH)
    D = C.total_km * 0.012
    C.map.advance(D)
    im = Image.fromarray(maps.tonemap(C.map.image()))
    x, y, ang = C.head(D)
    im = glow_roomba(im, 900, 250, math.radians(-30), 170)
    d = ImageDraw.Draw(im)
    headline(d, (40, 40), "EVERY STREET", 110, V.GOLD)
    headline(d, (44, 160), "IN NEW YORK", 72, (255, 255, 255))
    st = V.city_stats("nyc")
    headline(d, (40, TH - 40), f"{st['median_s']:.1f} s", 130, V.GOLD, "ls")
    im.save(os.path.join(OUT, "thumb_B.png"))


def thumb_C():
    """The Roomba about to drive the last block in America."""
    L = V.Last()
    fr, f_, D, dark, head, s, lc = L.map_frame(14.6)
    im = Image.fromarray(fr).crop((240, 60, 1680, 870)).resize((TW, TH), Image.LANCZOS)
    k = TW / 1440
    cx, cy = (lc[0] - 240) * k, (lc[1] - 60) * k
    d = ImageDraw.Draw(im)
    d.ellipse([cx - 64, cy - 64, cx + 64, cy + 64], outline=V.GOLD, width=8)
    if head is not None:
        im = glow_roomba(im, (head[0] - 240) * k, (head[1] - 60) * k, head[2], 110)
    d = ImageDraw.Draw(im)
    headline(d, (40, 40), "THE LAST ROAD", 104, V.GOLD)
    headline(d, (44, 170), "IN AMERICA", 64, (255, 255, 255))
    im.save(os.path.join(OUT, "thumb_C.png"))


if __name__ == "__main__":
    import sys
    for k in (sys.argv[1:] or "ABC"):
        globals()[f"thumb_{k}"]()
    print("wrote", [os.path.join(OUT, f"thumb_{k}.png") for k in "ABC"])
