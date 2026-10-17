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


def hero_roomba(im, x, y, width, heading=-40.0):
    """a three-quarter-view Roomba centred at (x, y), driving toward `heading` (screen degrees, counter-clockwise
    from right), with a light-speed streak behind it and a warm glow on the ground"""
    import cv2
    import roomba3d
    arr = np.asarray(im.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    hd = math.radians(heading)
    dx, dy = math.cos(hd), -math.sin(hd)                        # image coordinates, y down
    px, py = -dy, dx
    gy = y + width * 0.10                                       # ground under the body
    L, r0 = width * 2.8, width * 0.34
    tail = (x - dx * L, gy - dy * L)
    streak = np.zeros((h, w), np.float32)
    poly = np.array([(x + px * r0, gy + py * r0 * 0.55), (x - px * r0, gy - py * r0 * 0.55),
                     (tail[0], tail[1])], np.int32)
    cv2.fillPoly(streak, [poly], 1.0, cv2.LINE_AA)
    # brighter near the robot, fading toward the tail
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    along = ((xx - x) * -dx + (yy - gy) * -dy) / L
    streak *= np.clip(1 - along, 0, 1) ** 1.6
    streak = cv2.GaussianBlur(streak, (0, 0), width * 0.05)
    glow = np.zeros((h, w), np.float32)
    cv2.ellipse(glow, (int(x), int(gy + width * 0.05)), (int(width * 0.50), int(width * 0.24)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), width * 0.09)
    arr = arr + streak[..., None] * np.array([255, 236, 196]) * 0.9 + glow[..., None] * np.array(V.GOLD) * 0.6
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    sp = roomba3d.sprite(int(width), heading)
    out.alpha_composite(sp, (int(x - sp.width / 2), int(y - sp.height / 2)))
    return out.convert("RGB")


def thumb_A():
    """America about 2 minutes in: gold where the Roomba has been, blue where it hasn't."""
    U = maps.USA(seed=V.USA_SEED, scale=V.USA_SCALE, ox=V.USA_OX, oy=V.USA_OY)
    U.map.advance(U.block_t[len(U.block_t) // 2])                  # half the blocks driven
    fr = maps.tonemap(U.map.image(gamma=0.42, dim=1.5, fgamma=4.0))
    im = Image.fromarray(fr).crop((150, 70, 1830, 1015)).resize((TW, TH), Image.LANCZOS)
    im = hero_roomba(im, 840, 320, 460, -35)
    d = ImageDraw.Draw(im)
    headline(d, (40, 470), "NO MAP.", 124, (255, 255, 255))
    headline(d, (44, 600), "LIGHT SPEED.", 92, V.GOLD)
    im.save(os.path.join(OUT, "thumb_A.png"))


def thumb_B():
    """New York, part driven, Roomba mid-run."""
    c = V.NYC
    C = maps.City(c["prefix"], c["seed"], 15.0, c["cx"], c["cy"] + 1.0, 640, 380, w=TW, h=TH)
    D = C.total_km * 0.012
    C.map.advance(D)
    im = Image.fromarray(maps.tonemap(C.map.image()))
    x, y, ang = C.head(D)
    im = hero_roomba(im, 930, 330, 380, -35)
    d = ImageDraw.Draw(im)
    headline(d, (40, 40), "EVERY STREET", 110, V.GOLD)
    headline(d, (44, 160), "IN NEW YORK", 72, (255, 255, 255))
    st = V.city_stats("nyc")
    headline(d, (40, TH - 40), f"{st['median_s']:.1f} s", 130, V.GOLD, "ls")
    im.save(os.path.join(OUT, "thumb_B.png"))


def thumb_C():
    """The Roomba about to drive the last block in America."""
    L = V.Last()
    fr, f_, D, dark, head, s, lc = L.map_frame(14.2)       # six blocks to go, the Roomba heading for the last
    im = Image.fromarray(fr).crop((240, 60, 1680, 870)).resize((TW, TH), Image.LANCZOS)
    k = TW / 1440
    cx, cy = (lc[0] - 240) * k, (lc[1] - 60) * k
    d = ImageDraw.Draw(im)
    d.ellipse([cx - 64, cy - 64, cx + 64, cy + 64], outline=V.GOLD, width=8)
    im = hero_roomba(im, cx - 300, cy + 20, 320, -12)                    # heading for the last block (the ring)
    d = ImageDraw.Draw(im)
    headline(d, (40, 40), "THE LAST ROAD", 104, V.GOLD)
    headline(d, (44, 170), "IN AMERICA", 64, (255, 255, 255))
    im.save(os.path.join(OUT, "thumb_C.png"))


if __name__ == "__main__":
    import sys
    for k in (sys.argv[1:] or "ABC"):
        globals()[f"thumb_{k}"]()
    print("wrote", [os.path.join(OUT, f"thumb_{k}.png") for k in "ABC"])
