"""The look for "Reddish green": a dark light-lab. Everything on screen is light: stage spotlights with gels
on a matte black wall, beams through haze, diagrams projected as lines of light, a faint reflection on the
floor. Worked in linear light, tone-mapped, then sRGB.

Storyboard stills are 1920x1080; the film would render the same scenes at 4K.
"""
import os, math, functools
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
OUT = os.path.join(HERE, "out")
W, H = 1920, 1080
FLOOR = 0.86                                   # the floor line (fraction of the height)


# ------------------------------------------------------------------ colour
def to_lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb8(x):
    x = np.clip(x, 0, 1)
    return (np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


def tonemap(x, exposure=1.0):
    """a gentle filmic shoulder: linear through the mid-tones, so test colours stay true, rolling off highlights"""
    x = x * exposure
    knee = 0.8
    return np.where(x < knee, x, knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee))))


# gels, as linear RGB at full intensity
GEL = {"red": np.array([1.0, 0.015, 0.02], np.float32),
       "green": np.array([0.02, 1.0, 0.06], np.float32),
       "blue": np.array([0.02, 0.10, 1.0], np.float32),
       "yellow": np.array([1.0, 0.80, 0.02], np.float32),
       "white": np.array([1.0, 0.95, 0.88], np.float32)}


# ------------------------------------------------------------------ the room
@functools.lru_cache(None)
def wall_texture():
    """matte black plaster: a little large-scale unevenness and fine grit"""
    rng = np.random.default_rng(7)
    t = np.zeros((H, W), np.float32)
    for sig, amp in ((200, 0.10), (40, 0.06), (6, 0.05), (1.0, 0.06)):
        n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sig)
        t += n / (n.std() + 1e-9) * amp
    return 1 + t


@functools.lru_cache(None)
def room():
    """the dark room: wall albedo, ambient light, and where the floor begins"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    ambient = 0.006 * (1 - 0.6 * ((u - 0.5) ** 2 + (v - 0.4) ** 2))
    base = np.clip(ambient * wall_texture(), 0, None)
    return np.repeat(base[..., None], 3, -1).astype(np.float32)


def pool(img, cx, cy, r, col, gain=1.0, soft=0.05, squash=0.92):
    """a spotlight's pool of light on the wall, added in linear light"""
    yy, xx = np.ogrid[0:H, 0:W]
    d = np.sqrt(((xx - cx) / r) ** 2 + ((yy - cy) / (r * squash)) ** 2).astype(np.float32)
    edge = np.clip((1 - d) / soft, 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    hot = 1 + 0.25 * np.clip(1 - d, 0, 1) ** 2
    lum = edge * hot * gain * (0.85 + 0.15 * wall_texture())
    img += lum[..., None] * col[None, None, :]
    return img


def beam(img, src, dst, r_end, col, gain=0.05, seed=0):
    """the beam's cone through haze, from the fixture (off frame) to its pool"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = src
    dx, dy = dst[0] - sx, dst[1] - sy
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    along = (xx - sx) * ux + (yy - sy) * uy
    across = np.abs(-(xx - sx) * uy + (yy - sy) * ux)
    t = np.clip(along / L, 0, 1)
    rad = 6 + (r_end - 6) * t
    inside = np.clip((rad - across) / (0.35 * rad + 1), 0, 1) * (along > 0) * (along < L)
    rng = np.random.default_rng(seed)
    haze = cv2.GaussianBlur(rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32), (0, 0), 6)
    haze = cv2.resize(haze / (haze.std() + 1e-9), (W, H))
    dens = np.clip(0.75 + 0.35 * haze, 0.2, None)
    core = np.exp(-(across / (rad + 1)) ** 2 * 2.0)
    lum = inside * core * dens * gain * (0.4 + 0.6 * t)
    img += lum[..., None] * col[None, None, :]
    return img


def dust(img, src, dst, r_end, col, n=220, seed=1):
    """motes caught in a beam"""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.float32)
    for _ in range(n):
        t = rng.uniform(0.15, 0.95)
        off = rng.normal(0, 0.45) * (6 + (r_end - 6) * t)
        x = src[0] + (dst[0] - src[0]) * t
        y = src[1] + (dst[1] - src[1]) * t
        L = math.hypot(dst[0] - src[0], dst[1] - src[1])
        nx, ny = -(dst[1] - src[1]) / L, (dst[0] - src[0]) / L
        px, py = x + nx * off, y + ny * off
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(m, (int(px * 4), int(py * 4)), int(rng.uniform(2, 6)), float(rng.uniform(0.2, 1.0)), -1, cv2.LINE_AA, shift=2)
    m = cv2.GaussianBlur(m, (0, 0), 0.8) * 0.08
    img += m[..., None] * col[None, None, :]
    return img


def floor(img, strength=0.16):
    """a dark, slightly glossy floor below the wall: a soft, stretched reflection"""
    fy = int(H * FLOOR)
    wall = img[:fy].copy()
    refl = wall[::-1][: H - fy]
    refl = cv2.GaussianBlur(refl, (0, 0), sigmaX=6, sigmaY=22)
    fade = np.linspace(1, 0, H - fy, dtype=np.float32) ** 1.6
    img[fy:] = img[fy:] * 0.35 + refl * strength * fade[:, None, None]
    # the skirting where wall meets floor
    img[fy - 1:fy + 1] *= 0.4
    return img


def bloom(img, amount=0.05):
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(small, (0, 0), 6) + 0.5 * cv2.GaussianBlur(small, (0, 0), 18)
    return img + cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * amount


def finish(img, exposure=1.0, grain=0.006, seed=0):
    out = tonemap(bloom(img), exposure)
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0, grain, out.shape[:2]).astype(np.float32)[..., None] * (0.3 + out)
    return to_srgb8(out)


# ------------------------------------------------------------------ projected drawing: lines of light
def light_lines(img, polylines, col, width=3.0, gain=1.0, glow=0.6):
    """polylines (lists of (x, y)) drawn as projected light: a bright core and a soft halo"""
    m = np.zeros((H, W), np.float32)
    for pts in polylines:
        P = np.round(np.asarray(pts, np.float32) * 16).astype(np.int32)
        cv2.polylines(m, [P], False, 1.0, max(1, int(round(width))), cv2.LINE_AA, shift=4)
    halo = cv2.GaussianBlur(m, (0, 0), width * 2.5) * glow
    lum = (m + halo) * gain
    img += lum[..., None] * np.asarray(col, np.float32)[None, None, :]
    return img


def light_fill(img, mask, col, gain=1.0):
    img += (mask * gain)[..., None] * np.asarray(col, np.float32)[None, None, :]
    return img


# ------------------------------------------------------------------ type
def font(kind, size, weight=None):
    path = {"display": "BigShouldersDisplay[wght].ttf", "text": "AtkinsonHyperlegible-Regular.ttf",
            "bold": "AtkinsonHyperlegible-Bold.ttf", "mono": "IBMPlexMono-Regular.ttf", "mono2": "IBMPlexMono-Medium.ttf"}[kind]
    f = ImageFont.truetype(os.path.join(FONTS, path), int(round(size)), layout_engine=ImageFont.Layout.RAQM)
    if kind == "display":
        f.set_variation_by_axes([weight or 800])
    return f


def text_mask(xy, s, size, kind="text", anchor="la", weight=None, spacing=0):
    """a text shape as a float mask, for drawing as light"""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    f = font(kind, size, weight)
    if spacing:
        x, y = xy
        for ch in s:
            d.text((x, y), ch, font=f, fill=255, anchor=anchor)
            x += d.textlength(ch, font=f) + spacing
    else:
        d.text(xy, s, font=f, fill=255, anchor=anchor)
    return np.asarray(im, np.float32) / 255.0


def projected_text(img, xy, s, size, col=(1, 1, 1), gain=0.9, kind="display", anchor="la", weight=None, spacing=0):
    m = text_mask(xy, s, size, kind, anchor, weight, spacing)
    m = cv2.GaussianBlur(m, (0, 0), 0.7)
    halo = cv2.GaussianBlur(m, (0, 0), size * 0.12) * 0.35
    return light_fill(img, m + halo, col, gain)


def caption(im, lines, x=120, y=1000, size=40, col=(236, 236, 232)):
    """the subtitles: left-aligned, bottom-left, over the dark floor (drawn after tone mapping)"""
    d = ImageDraw.Draw(im)
    f = font("text", size)
    yy = y - (len(lines) - 1) * size * 1.3
    for ln in lines:
        d.text((x + 2, yy + 2), ln, font=f, fill=(0, 0, 0), anchor="ls")
        d.text((x, yy), ln, font=f, fill=col, anchor="ls")
        yy += size * 1.3


def label(im, xy, s, size=24, col=(200, 200, 196), kind="mono", anchor="la"):
    ImageDraw.Draw(im).text(xy, s, font=font(kind, size), fill=col, anchor=anchor)


def test_tag(im, n, secs, frac, x=1700, y=110):
    """the viewer-test marker: a tag and a countdown ring (drawn after tone mapping)"""
    d = ImageDraw.Draw(im)
    d.text((x - 70, y), f"TEST {n}", font=font("mono2", 26), fill=(240, 240, 236), anchor="rm")
    r = 34
    d.ellipse((x - r, y - r, x + r, y + r), outline=(90, 90, 90), width=4)
    d.arc((x - r, y - r, x + r, y + r), -90, -90 + 360 * frac, fill=(240, 240, 236), width=4)
    d.text((x, y), str(secs), font=font("mono2", 24), fill=(240, 240, 236), anchor="mm")
