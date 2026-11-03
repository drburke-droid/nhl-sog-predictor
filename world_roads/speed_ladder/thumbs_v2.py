"""Thumbnails, second set (1280x720): one big subject each, two or three huge words, the speeds as small tags.
  python thumbs_v2.py <outdir> <frames dir>   (frames: the picture, 1920x804, of race_light_clean.mp4 frames 429
  (race_6.png), 578 (cand_10.png) and 129 (cand_03.png))"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumbs as T                                       # reads the same two arguments

OUT = sys.argv[1]
D = sys.argv[2]


def rung_through(img, x_from, x_to, y_guess, half=10):
    """the ladder line nearest y_guess, followed column by column and fitted (it leans a little with the view)"""
    hp = (img - cv2.GaussianBlur(img, (1, 0), sigmaX=0.1, sigmaY=4.0)).mean(-1)
    cols = np.arange(x_from, x_to)
    rows = np.array([y_guess - half + np.argmax(hp[y_guess - half:y_guess + half + 1, c]) for c in cols], np.float32)
    return np.polyfit(cols, rows, 1)


def pulse(img, px, line, length=900, core=1.8, halo=0.6):
    """light's pulse at px on the given ladder line, and the line it draws behind it, glowing"""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    py = np.polyval(line, px)
    ly = np.polyval(line, xx)
    r2 = (xx - px) ** 2 + (yy - py) ** 2
    glow = core * np.exp(-r2 / (2 * 10 ** 2)) + halo * np.exp(-r2 / (2 * 46 ** 2)) + 0.12 * np.exp(-r2 / (2 * 140 ** 2))
    trail = (np.exp(-((yy - ly) ** 2) / (2 * 2.6 ** 2)) + 0.25 * np.exp(-((yy - ly) ** 2) / (2 * 9 ** 2))) * \
        np.clip((xx - (px - length)) / length, 0, 1) ** 1.6 * (xx <= px)
    out = img * (1 - 0.35 * np.exp(-r2 / (2 * 160 ** 2)))[..., None]       # the sky round it a touch darker
    flare = 0.8 * np.exp(-((yy - py) ** 2) / (2 * 1.8 ** 2) - (xx - px) ** 2 / (2 * 260 ** 2))
    cyan = np.array([0.45, 0.82, 1.0], np.float32)
    core_w = np.exp(-r2 / (2 * 5 ** 2))[..., None]
    out = out + (glow + flare + 1.1 * trail)[..., None] * cyan + 1.5 * core_w
    return out, py


def zoom169(img, x0, w):
    """a 16:9 window w px wide from the top of the picture, scaled to 1280x720"""
    h = int(round(w * 9 / 16))
    return cv2.resize(img[:h, x0:x0 + w], (1280, 720), interpolation=cv2.INTER_CUBIC), 1280 / w


def thumb_a():
    """the F-35 in the race, light already past it"""
    img = T.lift_lines(T.load("race_6.png"), 1.8)
    line = rung_through(img, 200, 480, 330, half=8)
    px = 1430                                          # just ahead of the jet's nose, in clear sky
    img, py = pulse(img, px, line, length=1300, core=2.4, halo=0.35)
    x0 = 400
    img, s = zoom169(img, x0, 1150)
    img = T.shade(T.grade(img), 400, 720, top=False, strength=0.78)
    im = T.to_im(img)
    T.words(im, [("TOO SLOW.", 214, T.AMBER)], 36, 478)
    T.tag(im, "F-35 · 1,228 MPH", (900 - x0) * s, 92, T.WHITE, 36)
    T.tag(im, "LIGHT", (px - x0) * s, py * s + 50, T.CYAN, 38)
    return im


def thumb_b():
    """the fastest machine ever built, frozen while light passes it"""
    img = T.load("cand_10.png")
    lum = cv2.GaussianBlur(img.mean(-1), (0, 0), 3)
    sub = lum[:250, 900:]
    py, px = np.unravel_index(np.argmax(sub), sub.shape)
    px += 900
    img = T.lift_lines(img, 2.0)
    line = rung_through(img, max(0, px - 700), px - 12, int(py))
    img, _ = pulse(img, px, line, length=640)
    img, x0, s = T.crop169(img, 0)
    img = T.shade(T.grade(img), 380, 720, top=False, strength=0.75)
    im = T.to_im(img)
    T.words(im, [("FROZEN.", 230, T.AMBER)], 40, 455)
    T.tag(im, "FASTEST MACHINE EVER BUILT", 300, 255, T.WHITE, 30)
    T.tag(im, "LIGHT", (px - x0) * s, np.polyval(line, px) * s + 44, T.CYAN, 36)
    return im


def thumb_c():
    """the Wright Flyer, and the laps light makes while it crosses"""
    img = T.lift_lines(T.load("cand_03.png"), 3.0)
    img, x0, s = T.crop169(img, 491)
    img = T.grade(img)
    lw = np.clip((700 - np.arange(1280)) / 500, 0, 1) ** 1.3          # darker on the left, under the words
    img = img * (1 - 0.55 * lw)[None, :, None]
    im = T.to_im(img)
    T.words(im, [("17×", 300, T.AMBER)], 40, 60)
    T.words(im, [("AROUND", 104, T.WHITE), ("EARTH", 104, T.WHITE)], 48, 372)
    return im


if __name__ == "__main__":
    for name, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c)):
        f().convert("RGB").save(f"{OUT}/speed_ladder_v2_thumb_{name}.png")
        print(name)
