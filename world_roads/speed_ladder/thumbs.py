"""Thumbnails A/B/C (1280x720) for the light film, in the style of the series' others: a frame from the film, two
short lines of heavy capitals, white then amber.
  python thumbs.py <outdir> <frames dir>   (race_light_clean.mp4 frames 119, 463 and 578, cropped to the picture)"""
import sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

D = sys.argv[2]            # the candidate frames: the picture (1920x804) of clean frames, as cand_02/06/10.png
OUT = sys.argv[1]
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-{}.otf"
WHITE, AMBER, CYAN = (255, 255, 255), (255, 186, 60), (140, 210, 255)


def font(size, weight="Black"):
    return ImageFont.truetype(FONT.format(weight), size)


def load(name):
    return cv2.imread(f"{D}/{name}")[..., ::-1].astype(np.float32) / 255


def lift_lines(img, gain=2.5):
    """the ladder's lines, faint in the film, drawn out: the image's own thin horizontal detail in the sky amplified
    (so planes in front still hide them)"""
    blur = cv2.GaussianBlur(img, (1, 0), sigmaX=0.1, sigmaY=4.0)
    hp = np.clip(img - blur, 0, 1)
    sky = np.zeros(img.shape[:2], np.float32)
    sky[:int(img.shape[0] * 0.47)] = 1
    sky = cv2.GaussianBlur(sky, (0, 0), 12)[..., None]
    tint = np.array([0.75, 0.9, 1.0], np.float32)
    return np.clip(img + gain * hp.mean(-1, keepdims=True) * tint * sky, 0, 1)


def crop169(img, x0):
    h = img.shape[0]
    w = int(round(h * 16 / 9))
    x0 = int(max(0, min(img.shape[1] - w, x0)))
    return cv2.resize(img[:, x0:x0 + w], (1280, 720), interpolation=cv2.INTER_AREA), x0, 1280 / w


def grade(img, k=1.12):
    """a little more contrast and colour, for a small picture"""
    g = img.mean(-1, keepdims=True)
    img = g + (img - g) * 1.15
    return np.clip((img - 0.5) * k + 0.5, 0, 1)


def shade(img, y0, y1, top=True, strength=0.65):
    """darken toward an edge, where the words go"""
    h = img.shape[0]
    ramp = np.zeros(h, np.float32)
    ys = np.arange(h)
    if top: ramp = np.clip((y1 - ys) / (y1 - y0), 0, 1)
    else: ramp = np.clip((ys - y0) / (y1 - y0), 0, 1)
    return img * (1 - strength * ramp ** 1.4)[:, None, None]


def words(im, lines, x, y, anchor="la"):
    """[(text, size, colour)] stacked from (x, y), with a soft dark halo so they read on anything"""
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    halo = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d, dh = ImageDraw.Draw(lay), ImageDraw.Draw(halo)
    yy = y
    for text, size, col in lines:
        f = font(size)
        dh.text((x, yy), text, font=f, fill=(0, 0, 0, 200), anchor=anchor, stroke_width=10, stroke_fill=(0, 0, 0, 200))
        d.text((x, yy), text, font=f, fill=col + (255,), anchor=anchor)
        yy += int(size * 1.02)
    im.alpha_composite(halo.filter(ImageFilter.GaussianBlur(12)))
    im.alpha_composite(lay)


def tag(im, text, x, y, col=WHITE, size=30, anchor="mm"):
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    f = font(size, "Bold")
    d.text((x, y), text, font=f, fill=col + (255,), anchor=anchor, stroke_width=5, stroke_fill=(0, 0, 0, 170))
    im.alpha_composite(lay)


def to_im(img):
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).convert("RGBA")


def thumb_a():
    """light racing past the frozen Parker"""
    img = load("cand_10.png")
    # find the pulse: the brightest small spot in the sky right of the probe's column
    lum = cv2.GaussianBlur(img.mean(-1), (0, 0), 3)
    sub = lum[:250, 900:]
    py, px = np.unravel_index(np.argmax(sub), sub.shape)
    px += 900
    img = lift_lines(img, 2.0)
    # the pulse made to read at a glance: its glow, and the line it is drawing, brighter
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    glow = 1.6 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 9 ** 2)) + \
        0.5 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 40 ** 2))
    # the line it is drawing (its own rung, leaning a little with the view): found column by column, smoothed
    hp = (img - cv2.GaussianBlur(img, (1, 0), sigmaX=0.1, sigmaY=4.0)).mean(-1)
    cols = np.arange(max(0, px - 700), px - 12)
    rows = np.array([py - 10 + np.argmax(hp[py - 10:py + 11, c]) for c in cols], np.float32)
    fit = np.polyval(np.polyfit(cols, rows, 1), np.arange(img.shape[1]))
    trail = np.exp(-((yy - fit[None, :]) ** 2) / (2 * 2.2 ** 2)) * np.clip((xx - (px - 620)) / 620, 0, 1) ** 2 * (xx <= px)
    img = img + (glow + 0.9 * trail)[..., None] * np.array([0.85, 0.95, 1.0], np.float32)
    img, x0, s = crop169(img, 0)
    img = shade(grade(img), 380, 720, top=False, strength=0.7)
    im = to_im(img)
    words(im, [("FASTEST MACHINE EVER BUILT", 70, WHITE), ("VS. LIGHT SPEED.", 128, AMBER)], 44, 500)
    tag(im, "LIGHT", (px - x0) * s, py * s + 42, CYAN, 34)
    tag(im, "PARKER SOLAR PROBE", 250, 255, WHITE, 28)
    return im


def thumb_b():
    """the jets on the road, the ladder of light behind them"""
    img = lift_lines(load("cand_06.png"), 3.0)
    img, x0, s = crop169(img, 330)
    img = shade(grade(img), 300, 720, top=False, strength=0.72)
    im = to_im(img)
    words(im, [("120 YEARS OF SPEED.", 92, WHITE), ("VS. LIGHT SPEED.", 128, AMBER)], 44, 465)
    return im


def thumb_c():
    """the Wright Flyer, and light's laps of the Earth behind it"""
    img = lift_lines(load("cand_02.png"), 3.2)
    img, x0, s = crop169(img, 245)
    img = shade(grade(img), 0, 250, top=True, strength=0.55)
    img = shade(img, 430, 720, top=False, strength=0.72)
    im = to_im(img)
    words(im, [("17 LAPS OF THE EARTH", 104, AMBER)], 640, 40, anchor="ma")
    words(im, [("BEFORE THIS PLANE", 84, WHITE), ("CROSSED THE ROAD.", 84, WHITE)], 640, 520, anchor="ma")
    return im


if __name__ == "__main__":
    for name, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c)):
        f().convert("RGB").save(f"{OUT}/speed_ladder_thumb_{name}.png")
        print(name)
