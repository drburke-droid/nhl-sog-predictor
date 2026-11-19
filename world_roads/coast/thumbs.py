"""Thumbnails (1280x720) for the coast film, built from its own text-free 4K frames:

    RES=2 python film.py clean 12.6 77.0 123.6     # the frames (out/clean_*.png)
    python thumbs.py                               # -> out/coast_thumb_A.png  the night globe, half gold
                                                   #    out/coast_thumb_B.png  the first right turn, in Calgary
                                                   #    out/coast_thumb_C.png  the whole world gold

Titles in Anton (../thumb_fonts), the rest in Inter; the time sits in a solid gold chip.
A wraps the 4K world frame onto an orthographic night globe, as the roads film's thumbnail did
(../thumb_globe.py): every line on the globe is the film's real layer at that moment.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONTS = os.path.join(os.path.dirname(HERE), "thumb_fonts")
INTER = "/usr/share/fonts/opentype/inter"
TW, TH, S = 1280, 720, 2
GOLD = (255, 196, 84)
INK = (14, 18, 28)
TIME = "1:18"

# the film's projection at 4K (cproj with RES=2), for sampling frames onto the globe
os.environ["RES"] = "2"
sys.path.insert(0, HERE)
import cproj  # noqa: E402


def anton(size):
    return ImageFont.truetype(os.path.join(FONTS, "Anton-Regular.ttf"), int(size * S))


def inter(size, w="SemiBold"):
    return ImageFont.truetype(os.path.join(INTER, f"Inter-{w}.otf"), int(size * S))


def load(name):
    return np.asarray(Image.open(os.path.join(OUT, name)).convert("RGB")).astype(np.float32) / 255.0


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).convert("RGBA")


def crop_resize(frame, box):
    x0, y0, x1, y1 = box
    return to_img(frame[y0:y1, x0:x1]).resize((TW * S, TH * S), Image.LANCZOS)


def text_layer(size_wh, items):
    """items: (xy, text, font, fill, anchor); drawn with a soft dark shadow so they hold over the map"""
    lay = Image.new("RGBA", size_wh, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for xy, txt, f, fill, anchor in items:
        d.text(xy, txt, font=f, fill=fill + (255,), anchor=anchor)
    sh = Image.new("RGBA", size_wh, (0, 0, 0, 0))
    sh.putalpha(lay.getchannel("A").filter(ImageFilter.GaussianBlur(12 * S)).point(lambda v: int(v * 0.9)))
    out = Image.new("RGBA", size_wh, (0, 0, 0, 0))
    out.alpha_composite(sh)
    out.alpha_composite(lay)
    return out


def chip(im, x, y, big, small, anchor="left"):
    """the time in a solid gold chip: big digits, a small caption under them"""
    d = ImageDraw.Draw(im)
    fb, fs = anton(big[1]), inter(small[1], "Bold")
    bw = d.textlength(big[0], font=fb)
    sw = d.textlength(small[0], font=fs)
    w = max(bw, sw) + 44 * S
    h = (big[1] * 1.12 + small[1] * 1.45 + 30) * S
    x0 = x if anchor == "left" else (x - w if anchor == "right" else x - w / 2)
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([x0 + 4 * S, y + 8 * S, x0 + w + 4 * S, y + h + 8 * S], radius=18 * S, fill=(0, 0, 0, 170))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10 * S)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([x0, y, x0 + w, y + h], radius=18 * S, fill=GOLD + (255,))
    d.text((x0 + w / 2, y + 10 * S), big[0], font=fb, fill=INK + (255,), anchor="ma")
    d.text((x0 + w / 2, y + h - 16 * S), small[0], font=fs, fill=INK + (255,), anchor="md")


def vignette(im, strength=0.55, top=0.0):
    w, h = im.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (w * 0.6), (yy - h / 2) / (h * 0.65))
    a = np.clip((r - 0.55) / 0.6, 0, 1) ** 1.5 * strength
    if top:
        a = np.maximum(a, np.clip(1 - yy / (h * top), 0, 1) ** 1.6 * 0.75)
    m = Image.fromarray((a * 255).astype(np.uint8))
    black = Image.new("RGBA", im.size, (0, 0, 0, 255))
    black.putalpha(m)
    im.alpha_composite(black)


def save(im, name):
    im = im.convert("RGB").resize((TW, TH), Image.LANCZOS)
    p = os.path.join(OUT, name)
    im.save(p, optimize=True)
    print(p)


# ---------------------------------------------------------------- A: the night globe
GLOBE = dict(cx=430, cy=405, r=430, lon0=55.0, lat0=18.0)


def ortho_inverse(px, py, g):
    x = (px - g["cx"] * S) / (g["r"] * S)
    y = (g["cy"] * S - py) / (g["r"] * S)
    rho = np.hypot(x, y)
    inside = rho <= 1.0
    c = np.arcsin(np.clip(rho, 0, 1))
    p0, l0 = math.radians(g["lat0"]), math.radians(g["lon0"])
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.clip(np.cos(c) * math.sin(p0) + np.where(rho > 0, y * np.sin(c) * math.cos(p0) / rho, 0), -1, 1))
        lon = l0 + np.arctan2(x * np.sin(c), rho * np.cos(c) * math.cos(p0) - y * np.sin(c) * math.sin(p0))
    return np.degrees(lon), np.degrees(lat), np.cos(c), rho, inside


def gauss_spot(shape, x, y, sx, sy=None):
    sy = sx if sy is None else sy
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-(((xx - x) / sx) ** 2 + ((yy - y) / sy) ** 2) / 2)


def globe(frame, g=GLOBE):
    h, w = TH * S, TW * S
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    lon, lat, mu, rho, inside = ortho_inverse(px + 0.5, py + 0.5, g)
    lonw = ((lon + 180) % 360) - 180
    # night sky with faint stars
    r_bg = np.hypot(px / w - 0.33, py / h - 0.55)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([0.012, 0.020, 0.040]) * np.clip(1.2 - r_bg, 0, 1)[..., None]
    rng = np.random.default_rng(7)
    for _ in range(240):
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        img += rng.uniform(0.15, 0.5) ** 2 * gauss_spot((h, w), sx, sy, rng.uniform(0.7, 1.3) * S)[..., None] * np.array([0.8, 0.85, 1.0])
    img *= (~inside)[..., None]
    # the film's frame (land, coasts, rivers) through its own projection, shaded toward the limb
    fx, fy = cproj.to_px(lonw, lat)
    surf = np.stack([map_coordinates(frame[..., c], [fy - 0.5, fx - 0.5], order=1, mode="nearest") for c in range(3)], -1)
    shade = np.clip(mu, 0, 1) ** 0.5
    surf = surf * (0.35 + 0.65 * shade[..., None]) * 1.15
    img = np.where(inside[..., None], surf, img)
    # glow off the bright lines
    lum = img.max(axis=2)
    bright = img * (lum > 0.3)[..., None]
    for s_, k in ((2.5 * S, 0.22), (12 * S, 0.14)):
        img += k * np.stack([gaussian_filter(bright[..., c], s_) for c in range(3)], -1)
    # atmosphere
    R = g["r"] * S
    dist = np.hypot(px - g["cx"] * S, py - g["cy"] * S)
    rim = np.where(inside, np.clip(rho, 0, 1) ** 14, 0)
    halo = np.where(~inside, np.exp(-(dist - R) / (16 * S)), 0) + 0.35 * np.where(~inside, np.exp(-(dist - R) / (70 * S)), 0)
    img += (0.55 * rim + 0.45 * halo)[..., None] * np.array([0.30, 0.55, 1.0]) * 0.6
    return img


def fish_lonlat(frame):
    """the fish is the whitest spot of the map; read it back through the projection"""
    white = frame.min(axis=2).copy()
    white[:300] = 0
    white[-200:] = 0
    white = gaussian_filter(white, 3.0)
    yy, xx = np.unravel_index(np.argmax(white), white.shape)
    lons, lats = np.meshgrid(np.arange(-20, 80, 0.02), np.arange(-40, 50, 0.02))
    px, py = cproj.to_px(lons, lats)
    k = np.argmin((px - xx) ** 2 + (py - yy) ** 2)
    return float(lons.flat[k]), float(lats.flat[k])


def ortho_forward(lon, lat, g):
    p0, l0 = math.radians(g["lat0"]), math.radians(g["lon0"])
    p, l = math.radians(lat), math.radians(lon)
    x = math.cos(p) * math.sin(l - l0)
    y = math.cos(p0) * math.sin(p) - math.sin(p0) * math.cos(p) * math.cos(l - l0)
    return g["cx"] + g["r"] * x, g["cy"] - g["r"] * y


def thumb_a():
    frame = load("clean_0077.00.png")
    img = globe(frame)
    # the fish, bright at the frontier
    flon, flat = fish_lonlat(frame)
    x0, y0 = ortho_forward(flon, flat, GLOBE)
    x0, y0 = x0 * S, y0 * S
    shp = img.shape[:2]
    warm = np.array(GOLD) / 255.0
    img += gauss_spot(shp, x0, y0, 3.5 * S)[..., None] * 5.0 + gauss_spot(shp, x0, y0, 16 * S)[..., None] * (1.1 * warm + 0.7) \
        + gauss_spot(shp, x0, y0, 70 * S)[..., None] * 0.5 * warm \
        + (gauss_spot(shp, x0, y0, 95 * S, 2.0 * S) * 0.7)[..., None] * (0.7 * warm + 0.5)
    lum = img.max(axis=2, keepdims=True)
    tm = 1 - np.exp(-lum * 1.5)
    img = img * (tm / np.maximum(lum, 1e-6))
    im = to_img(img)
    right = 1236 * S
    im.alpha_composite(text_layer(im.size, [((right, 30 * S), "EVERY SHORE", anton(120), (255, 255, 255), "ra"),
                                            ((right, 176 * S), "ON EARTH", anton(120), GOLD, "ra"),
                                            ((right, 338 * S), "every coast, lake & riverbank", inter(30, "SemiBold"), (226, 232, 242), "ra")]))
    chip(im, right, 420 * S, (TIME, 132), ("AT LIGHT SPEED", 25), anchor="right")
    save(im, "coast_thumb_A.png")
    print("fish at", flon, flat)


# ---------------------------------------------------------------- B: the rule, close up
def thumb_b():
    frame = load("clean_0012.60.png")
    # lift the thin 4K lines and give them a glow, so the banks read at thumbnail size
    # (but not the fish, which is bright enough: the lift fades out around the brightest spot)
    white = gaussian_filter(frame.min(axis=2), 4)
    fy, fx = np.unravel_index(np.argmax(white), white.shape)
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    keep = 1 - np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * 110.0 ** 2))
    lines = np.clip(frame - np.array([0.03, 0.05, 0.08]), 0, None) * keep[..., None]
    glow = np.stack([gaussian_filter(lines[..., c], 4) for c in range(3)], -1)
    frame = frame + lines * 0.45 + glow * 0.9
    im = crop_resize(frame, (1180, 520, 2860, 1465))     # the Bow-Elbow fork; the fish just past it, up the Elbow
    vignette(im, 0.5, top=0.42)
    im.alpha_composite(text_layer(im.size, [((44 * S, 26 * S), "TURN RIGHT AT", anton(106), (255, 255, 255), "la"),
                                            ((44 * S, 150 * S), "EVERY FORK", anton(106), GOLD, "la")]))
    chip(im, 1236 * S, 486 * S, (TIME, 104), ("EVERY SHORE ON EARTH", 21), anchor="right")
    save(im, "coast_thumb_B.png")


# ---------------------------------------------------------------- C: the whole world gold
def thumb_c():
    frame = load("clean_0123.60.png")
    im = crop_resize(frame, (240, 200, 3600, 2090))
    vignette(im, 0.45, top=0.30)
    im.alpha_composite(text_layer(im.size, [((TW * S / 2, 16 * S), "EVERY COAST, LAKE & RIVERBANK", anton(86), (255, 255, 255), "ma")]))
    chip(im, 36 * S, 400 * S, (TIME, 150), ("AT THE SPEED OF LIGHT", 25), anchor="left")
    save(im, "coast_thumb_C.png")


if __name__ == "__main__":
    thumb_a(); thumb_b(); thumb_c()
