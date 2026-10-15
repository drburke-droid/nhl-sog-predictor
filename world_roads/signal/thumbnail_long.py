"""Thumbnails for the two long signal-fire videos (1280x720).

    python thumbnail_long.py   -> Video 1: out/long/thumb_A.png (the question), thumb_B.png (the number)
                                  Video 2: out/long/thumb_ai_A.png (AI took the internet), thumb_ai_B.png (the hours)
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
    # B: the number
    ring = globe_ring(48.0, 20.0, 640, 1920 * 0.64, 1080 / 2 + 20, [(la1, lo1, (1.0, 0.55, 0.15), 1.0)])
    ring = cv2.resize(ring, (TW, TH), interpolation=cv2.INTER_AREA)
    im = Image.fromarray(ring)
    text_block(im, [(f"{n_relays}", 200, (255, 186, 80)), ("FIRES", 92, (255, 255, 255)), ("AROUND THE", 64, (255, 255, 255)),
                    ("WORLD", 92, (255, 255, 255))], 50, 70)
    im.save(os.path.join(OUT, "thumb_B.png"))
    ai_thumbs()
    print("wrote thumbnails", n_relays)


def globe_tree(lat0, lon0, rpx, cx, cy, la, lo, par, red=()):
    g = G.Globe16()
    img = g.render(lat0, lon0, rpx, cx=cx, cy=cy)
    Ly = G.Layer()
    X, Y, Z = g.project(la, lo, lat0, lon0, rpx, cx=cx, cy=cy)
    for i, p in enumerate(par):
        if p >= 0 and Z[i] > 0.02 and Z[p] > 0.02:
            Ly.line((X[p], Y[p]), (X[i], Y[i]), 1.1, 2)
    for i in range(len(la)):
        if Z[i] > 0.02:
            Ly.fire(X[i], Y[i], 0.9, 1.0, np.array((1.0, 0.7, 0.3), np.float32))
    out = Ly.compose(img, exposure=1.35)
    if red:
        im = Image.fromarray(out)
        d = ImageDraw.Draw(im)
        rx, ry, rz = g.project(np.array([c[0] for c in red]), np.array([c[1] for c in red]), lat0, lon0, rpx, cx=cx, cy=cy)
        for x, y, z in zip(rx, ry, rz):
            if z > 0.05:
                d.line([(x - 13, y - 13), (x + 13, y + 13)], fill=ST.RED, width=7)
                d.line([(x - 13, y + 13), (x + 13, y - 13)], fill=ST.RED, width=7)
        out = np.asarray(im)
    return out


def static_panel(w, h, seed=5):
    """TV static with torn bands, for the NO SIGNAL look."""
    rng = np.random.default_rng(seed)
    st = cv2.resize((rng.random((h // 4, w // 4)) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    img = np.repeat(st[..., None] * 0.5, 3, -1)
    for _ in range(9):
        y0 = int(rng.integers(0, h - 30)); hh = int(rng.integers(5, 40))
        img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], int(rng.integers(-80, 80)), axis=1)
    img[..., 0] = np.roll(img[..., 0], 5, axis=1); img[..., 2] = np.roll(img[..., 2], -5, axis=1)
    img[::3] *= 0.7
    return img


def ai_thumbs():
    T = json.load(open(os.path.join(DATA, "capitals_tree.json")))
    F = T["fires"]
    la, lo = np.array([f["lat"] for f in F]), np.array([f["lon"] for f in F])
    par = np.array([f["parent"] for f in F])
    red = [(c["lat"], c["lon"]) for c in T["unreachable"]]
    hours = T["capitals"][-1]["hours"]
    # A: AI took the internet. We used fire.
    tree = globe_tree(30.0, 25.0, 760, 1920 * 0.66, 1080 / 2 + 60, la, lo, par)
    tree = cv2.resize(tree, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32)
    w = np.clip((np.arange(TW) - 330) / 260.0, 0, 1)[None, :, None]
    canvas = static_panel(TW, TH) * (1 - w) * 0.75 + tree * w + tree * (1 - w) * 0.25
    im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([36, 300, 446, 400], fill=(0, 0, 0, 225))
    d.text((241, 350), "NO SIGNAL", font=ST.font(62, "Bold"), fill=(240, 240, 240), anchor="mm")
    text_block(im, [("AI TOOK THE", 80, (255, 255, 255)), ("INTERNET", 80, (255, 255, 255))], 40, 40)
    text_block(im, [("WE USED", 80, (255, 186, 80)), ("FIRE", 112, (255, 186, 80))], 40, 470)
    im.save(os.path.join(OUT, "thumb_ai_A.png"))
    # B: the hours
    tree = globe_tree(25.0, 30.0, 700, 1920 * 0.64, 1080 / 2 + 30, la, lo, par, red)
    im = Image.fromarray(cv2.resize(tree, (TW, TH), interpolation=cv2.INTER_AREA))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([50, 52, 370, 112], fill=(200, 40, 30, 235))
    d.text((210, 82), "NO INTERNET", font=ST.font(40, "Bold"), fill=(255, 255, 255), anchor="mm")
    text_block(im, [(f"{int(round(hours))} HOURS", 132, (255, 186, 80)), ("TO WARN THE", 64, (255, 255, 255)),
                    ("WORLD", 64, (255, 255, 255)), ("BY FIRE", 64, (255, 255, 255))], 50, 140)
    im.save(os.path.join(OUT, "thumb_ai_B.png"))
    print(f"wrote AI thumbnails ({int(round(hours))} h)")


if __name__ == "__main__":
    main()
