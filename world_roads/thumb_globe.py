"""Thumbnail for "All world roads at light speed": the film's own frame wrapped onto a night globe, with a
red-marker note circling YouTube's duration badge (the film runs 4:40, the estimated time for light).

The globe samples a frame of the finished film (media/world_roads_light_speed.mp4), so every road on it is the
film's real road layer at that moment: gold where the light has been, blue where it hasn't.

    python thumb_globe.py thumb_fonts <ne_50m_land.geojson> <out dir>
      (ne_50m_land.geojson: raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/)
      -> thumb_globe_A.jpg   for YouTube: the circle waits for YouTube's own "4:40" badge
         thumb_globe_B.jpg   for anywhere without a badge: "4:40" written in the circle
         thumb_globe_clean.png  globe and title only, no marker
"""
import os, sys, json, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from proj import natural_earth, to_px  # noqa: E402

FILM = os.path.join(HERE, "media", "world_roads_light_speed.mp4")
T = 150.0                       # 2:30: Europe, the Americas and West Africa gold; the light in the Congo basin
W, H, SS = 1280, 720, 2         # YouTube thumbnail, drawn at 2x
GLOBE = dict(cx=440, cy=430, r=470, lon0=2.0, lat0=14.0)
GOLD = (255, 196, 84)
RED = (255, 44, 44)
BADGE = (1186, 666)             # where YouTube puts the duration badge (bottom right, ~2% in)


def film_frame(t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", FILM, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).astype(np.float32) / 255.0


def land_raster(path, res=0.05):
    w, h = int(360 / res), int(180 / res)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for f in json.load(open(path))["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            for k, ring in enumerate(p):
                d.polygon([((x + 180) / res, (90 - y) / res) for x, y in ring], fill=0 if k else 255)
    return np.asarray(im, np.float32) / 255.0, res


def ortho_inverse(px, py, g):
    x = (px - g["cx"] * SS) / (g["r"] * SS)
    y = (g["cy"] * SS - py) / (g["r"] * SS)
    rho = np.hypot(x, y)
    inside = rho <= 1.0
    c = np.arcsin(np.clip(rho, 0, 1))
    p0, l0 = math.radians(g["lat0"]), math.radians(g["lon0"])
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.clip(np.cos(c) * math.sin(p0) + np.where(rho > 0, y * np.sin(c) * math.cos(p0) / rho, 0), -1, 1))
        lon = l0 + np.arctan2(x * np.sin(c), rho * np.cos(c) * math.cos(p0) - y * np.sin(c) * math.sin(p0))
    return np.degrees(lon), np.degrees(lat), np.cos(c), rho, inside


def ortho_forward(lon, lat, g):
    p0, l0 = math.radians(g["lat0"]), math.radians(g["lon0"])
    p, l = math.radians(lat), math.radians(lon)
    x = math.cos(p) * math.sin(l - l0)
    y = math.cos(p0) * math.sin(p) - math.sin(p0) * math.cos(p) * math.cos(l - l0)
    return g["cx"] + g["r"] * x, g["cy"] - g["r"] * y


def photon_lonlat(frame):
    """The light is the whitest spot on the map; read its position back through the film's projection."""
    white = frame.min(axis=2)
    white[:120] = 0
    white[960:] = 0                                   # title and clock rows
    white = gaussian_filter(white, 2.0)
    yy, xx = np.unravel_index(np.argmax(white), white.shape)
    lons, lats = np.meshgrid(np.arange(-30, 60, 0.02), np.arange(-40, 40, 0.02))
    px, py = to_px(lons, lats)
    k = np.argmin((px - xx) ** 2 + (py - yy) ** 2)
    return float(lons.flat[k]), float(lats.flat[k])


def gauss_spot(shape, x, y, sx, sy=None):
    sy = sx if sy is None else sy
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-(((xx - x) / sx) ** 2 + ((yy - y) / sy) ** 2) / 2)


def render_globe(frame, land, g=GLOBE):
    land_img, res = land
    h, w = H * SS, W * SS
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    lon, lat, mu, rho, inside = ortho_inverse(px + 0.5, py + 0.5, g)
    lonw = ((lon + 180) % 360) - 180

    # background: deep night, faint stars
    r_bg = np.hypot(px / w - 0.38, py / h - 0.55)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([0.012, 0.020, 0.040]) * np.clip(1.2 - r_bg, 0, 1)[..., None]
    rng = np.random.default_rng(7)
    for _ in range(260):
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        b = rng.uniform(0.15, 0.55) ** 2
        img += b * gauss_spot((h, w), sx, sy, rng.uniform(0.7, 1.4) * SS)[..., None] * np.array([0.8, 0.85, 1.0])
    img *= (~inside)[..., None]

    # land and ocean, shaded toward the limb
    li = map_coordinates(land_img, [np.clip((90 - lat) / res, 0, land_img.shape[0] - 1),
                                    np.clip((lonw + 180) / res, 0, land_img.shape[1] - 1)], order=1)
    shade = np.clip(mu, 0, 1) ** 0.55
    ocean = np.array([0.010, 0.020, 0.045])
    ground = np.array([0.030, 0.045, 0.075])
    base = (ocean * (1 - li[..., None]) + ground * li[..., None]) * (0.45 + 0.55 * shade[..., None])

    # the film's roads, sampled through its own Natural Earth projection
    fx, fy = to_px(lonw, lat)
    roads = np.stack([map_coordinates(frame[..., c], [fy - 0.5, fx - 0.5], order=1, mode="constant") for c in range(3)], -1)
    roads = np.clip(roads - np.array([0.010, 0.018, 0.034]), 0, 1) * 1.1
    roads *= (0.30 + 0.70 * shade[..., None])
    glob = base + roads
    img = np.where(inside[..., None], glob, img)

    # glow off the roads
    lum = img.max(axis=2)
    bright = img * (lum > 0.25)[..., None]
    for s, k in ((3 * SS, 0.25), (14 * SS, 0.16)):
        img += k * np.stack([gaussian_filter(bright[..., c], s) for c in range(3)], -1)

    # atmosphere: a thin rim inside the limb and a haze outside it
    R = g["r"] * SS
    dist = np.hypot(px - g["cx"] * SS, py - g["cy"] * SS)
    rim_in = np.where(inside, np.clip(rho, 0, 1) ** 14, 0)
    halo = np.where(~inside, np.exp(-(dist - R) / (16 * SS)), 0) + 0.35 * np.where(~inside, np.exp(-(dist - R) / (70 * SS)), 0)
    sky = np.array([0.30, 0.55, 1.0])
    img += (0.55 * rim_in + 0.45 * halo)[..., None] * sky * 0.6

    # the light itself, at the frontier
    plon, plat = photon_lonlat(frame)
    x0, y0 = ortho_forward(plon, plat, g)
    x0, y0 = x0 * SS, y0 * SS
    core = gauss_spot((h, w), x0, y0, 4.0 * SS)
    halo2 = gauss_spot((h, w), x0, y0, 18 * SS)
    bloom = gauss_spot((h, w), x0, y0, 80 * SS)
    streak = gauss_spot((h, w), x0, y0, 300 * SS, 2.6 * SS) + 0.45 * gauss_spot((h, w), x0, y0, 2.4 * SS, 90 * SS)
    warm = np.array(GOLD) / 255.0
    img += core[..., None] * 6.0 + halo2[..., None] * (1.2 * warm + 0.9) + bloom[..., None] * 0.55 * warm \
        + streak[..., None] * (0.7 * warm + 0.6)

    # filmic shoulder on brightness only, so gold stays gold and the hot spot goes white without clipping flat
    lum = img.max(axis=2, keepdims=True)
    tm = 1 - np.exp(-lum * 1.5)
    img = img * (tm / np.maximum(lum, 1e-6))
    white = np.clip((lum - 1.2) / 3.0, 0, 1)            # only the very hottest light desaturates to white
    img = img * (1 - white) + tm * white
    return img, (x0 / SS, y0 / SS)


# ---------------------------------------------------------------- marker work

def wobble_path(pts, seed, amp=1.2):
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    noise = gaussian_filter(rng.normal(0, 1, (n, 2)), max(1, n / 12)) * amp * 6
    return pts + noise


def brush(draw, pts, width, color, taper=(0.55, 0.35)):
    """A marker stroke: overlapping round dabs, thinner where the pen lands and lifts."""
    pts = np.asarray(pts, np.float64)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    L = seg[-1]
    for s in np.arange(0, L, 0.6 * SS):
        x, y = np.interp(s, seg, pts[:, 0]), np.interp(s, seg, pts[:, 1])
        u = s / L
        k = 1.0
        if u < 0.08:
            k = taper[0] + (1 - taper[0]) * u / 0.08
        elif u > 0.85:
            k = 1 - (1 - taper[1]) * (u - 0.85) / 0.15
        r = width * k / 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def marker_layer(fonts, write_time=False):
    lay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    S = SS
    # circle around the badge: one loop that overshoots where it closes, as a hand draws it
    cx, cy = BADGE
    a = np.linspace(math.radians(200), math.radians(200 + 385), 400)
    rx, ry = 104, 58
    ell = np.c_[cx + rx * np.cos(a) * (1 + 0.04 * np.sin(3 * a)), cy + ry * np.sin(a) * (1 + 0.05 * np.cos(2 * a))]
    rot = math.radians(-7)
    ell = np.c_[cx + (ell[:, 0] - cx) * math.cos(rot) - (ell[:, 1] - cy) * math.sin(rot),
                cy + (ell[:, 0] - cx) * math.sin(rot) + (ell[:, 1] - cy) * math.cos(rot)]
    brush(d, wobble_path(ell, 3, 0.25) * S, 14 * S, RED, taper=(0.7, 0.5))
    # the arrow: from under the note, sweeping down into the circle
    p = bezier(np.array([992, 582.]), np.array([990, 640.]), np.array([1030, 676.]), np.array([1072, 668.]))
    p = wobble_path(p, 5, 0.18)
    brush(d, p * S, 13 * S, RED, taper=(0.8, 0.9))
    tip = p[-1]
    ang = math.atan2(*(p[-1] - p[-12])[::-1])
    for da, ln in ((2.55, 34), (-2.45, 30)):
        e = tip + ln * np.array([math.cos(ang + da), math.sin(ang + da)])
        brush(d, np.linspace(tip, e, 40) * S, 12 * S, RED, taper=(0.9, 0.5))
    # the note
    f = ImageFont.truetype(os.path.join(fonts, "PermanentMarker-Regular.ttf"), 60 * S)
    note = Image.new("RGBA", (620 * S, 260 * S), (0, 0, 0, 0))
    nd = ImageDraw.Draw(note)
    nd.text((10 * S, 0), "light needs", font=f, fill=RED)
    nd.text((60 * S, 74 * S), "this long", font=f, fill=RED)
    note = note.rotate(6, resample=Image.BICUBIC, expand=True)
    lay.alpha_composite(note, (int(846 * S), int(372 * S)))
    if write_time:
        ft = ImageFont.truetype(os.path.join(fonts, "PermanentMarker-Regular.ttf"), 50 * S)
        tw = d.textlength("4:40", font=ft)
        d.text((cx * S - tw / 2, (cy - 36) * S), "4:40", font=ft, fill=(255, 255, 255))
    return lay


def title_layer(fonts):
    lay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    f = ImageFont.truetype(os.path.join(fonts, "Anton-Regular.ttf"), 128 * SS)
    d = ImageDraw.Draw(lay)
    right = 1240 * SS
    for text, y, col in (("EVERY ROAD", 34, (255, 255, 255)), ("ON EARTH", 186, GOLD)):
        tw = d.textlength(text, font=f)
        d.text((right - tw, y * SS), text, font=f, fill=col + (255,))
    # a soft dark shadow under the letters so they hold over the globe's limb
    sh = Image.new("RGBA", lay.size, (0, 0, 0, 0))
    sh.putalpha(lay.getchannel("A").filter(ImageFilter.GaussianBlur(14 * SS)).point(lambda v: int(v * 0.85)))
    out = Image.new("RGBA", lay.size, (0, 0, 0, 0))
    out.alpha_composite(sh)
    out.alpha_composite(lay)
    return out


def main():
    fonts, land_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    frame = film_frame(T)
    img, photon = render_globe(frame, land_raster(land_path))
    base = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    base.alpha_composite(title_layer(fonts))
    base.resize((W, H), Image.LANCZOS).convert("RGB").save(os.path.join(out, "thumb_globe_clean.png"))
    for name, wt in (("A", False), ("B", True)):
        im = base.copy()
        im.alpha_composite(marker_layer(fonts, write_time=wt))
        im.resize((W, H), Image.LANCZOS).convert("RGB").save(os.path.join(out, f"thumb_globe_{name}.jpg"), quality=93)
    print("photon at", photon)


if __name__ == "__main__":
    main()
