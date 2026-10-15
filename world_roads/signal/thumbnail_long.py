"""Thumbnails for the long signal-fire video (1280x720).

    python thumbnail_long.py   -> out/long/thumb_A.png (the question), out/long/thumb_B.png (the countries)
"""
import os, json, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import style as ST
import globe16 as G

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out", "long")
TW, TH = 1280, 720


def globe_ring(lat0, lon0, rpx, cx, cy, routes):
    g = G.Globe16()
    img = g.render(lat0, lon0, rpx, cx=cx, cy=cy)
    Ly = G.Layer()
    for la, lo, col, amp in routes:
        X, Y, Z = g.project(la, lo, lat0, lon0, rpx, cx=cx, cy=cy)
        for i in range(1, len(la)):
            if Z[i] > 0.02 and Z[i - 1] > 0.02:
                Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), amp * 1.4, 2)
        for i in range(len(la)):
            if Z[i] > 0.02:
                Ly.fire(X[i], Y[i], amp * 1.3, 1.0, np.array(col, np.float32))
    return Ly.compose(img, exposure=1.35)


def text_block(im, lines, x, y):
    """Big outlined headline text: [(text, size, colour)]"""
    d = ImageDraw.Draw(im)
    for s, size, col in lines:
        f = ST.font(size, "Black") if os.path.exists(os.path.join(ST.FD, "Inter-Black.otf")) else ST.font(size, "Bold")
        d.text((x, y), s, font=f, fill=col, stroke_width=max(4, size // 14), stroke_fill=(0, 0, 0))
        y += int(size * 1.02)
    return y


def main():
    L = json.load(open(os.path.join(DATA, "loop.json")))
    st = L["stations"]
    la1 = np.array([s["lat"] for s in st]); lo1 = np.array([s["lon"] for s in st])
    n_relays = len(st) - 1
    T = json.load(open(os.path.join(DATA, "tour.json")))
    la2, lo2 = np.array(T["lat"]), np.array(T["lon"])
    cj = json.load(open(os.path.join(DATA, "countries.json")))
    n_c = len({c for c in T["counts_for"] if c >= 0 and cj["counts"][c]})
    # A: the beacon (left) + the ring around the world (right)
    fire = cv2.imread(os.path.join(OUT, "thumb_fire.png"))[..., ::-1]
    fire = cv2.resize(fire[110:110 + 760, 690:690 + 590], (560, 720), interpolation=cv2.INTER_AREA).astype(np.float32)
    ring = globe_ring(70.0, 30.0, 470, 960, 540, [(la1, lo1, (1.0, 0.55, 0.15), 1.0)])
    ring = cv2.resize(ring[60:1020, 480:1440], (720, 720), interpolation=cv2.INTER_AREA).astype(np.float32)
    canvas = np.zeros((TH, TW, 3), np.float32)
    canvas[:, 560:] = ring
    w = np.clip((np.arange(560) - 470) / 90.0, 0, 1)[None, :, None]
    canvas[:, :560] = fire * (1 - w) + canvas[:, :560] * w
    im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    text_block(im, [("HOW MANY", 86, (255, 255, 255)), ("FIRES TO", 86, (255, 255, 255))], 40, 40)
    text_block(im, [("CIRCLE", 86, (255, 186, 80)), ("EARTH?", 86, (255, 186, 80))], 40, 470)
    im.save(os.path.join(OUT, "thumb_A.png"))
    # B: the countries
    ring = globe_ring(22.0, 30.0, 700, 1920 * 0.62, 1080 / 2 + 40, [(la2, lo2, (1.0, 0.78, 0.35), 0.8)])
    ring = cv2.resize(ring, (TW, TH), interpolation=cv2.INTER_AREA)
    im = Image.fromarray(ring)
    text_block(im, [(f"{n_c}", 190, (255, 186, 80)), ("COUNTRIES", 84, (255, 255, 255)), ("ONE CHAIN", 84, (255, 255, 255)),
                    ("OF FIRE", 84, (255, 255, 255))], 60, 110)
    im.save(os.path.join(OUT, "thumb_B.png"))
    print("wrote thumbnails", n_relays, n_c)


if __name__ == "__main__":
    main()
