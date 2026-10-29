"""Composite rendered CG (straight-alpha RGBA PNGs) over the plate, in linear light, then add what a real camera
would: bloom around anything glowing, the flash of a hypersonic pass on the scene, the shake when the shock reaches
the camera, the plate's softness and grain, and the speed label. Output 1920x1080 frames, the 2.39:1 picture
letterboxed."""
import cv2, numpy as np, os, math
from PIL import Image, ImageDraw, ImageFont
PLATE = "/home/user/lot/plate/p{:03d}.png"
NPLATE = 149
W, H, PH = 1920, 1080, 804
TOP = (H - PH) // 2
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-{}.otf"


def lin(x): return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
def srgb(x): x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


_cache = {}


def plate_loop(i):
    L, X = NPLATE, 25                     # clip length, cross-fade length
    period = L - X
    j = i % period
    if i >= period and j < X:             # blend the clip's tail (frames period..L-1) into its head
        a = (j + 1) / (X + 1)
        return plate_raw(period + j) * (1 - a) + plate_raw(j) * a
    return plate_raw(j)


def plate_raw(k):
    if k not in _cache:
        if len(_cache) > 80: _cache.clear()
        _cache[k] = lin(cv2.imread(PLATE.format(k))[..., ::-1].astype(np.float32) / 255)
    return _cache[k]


def cg(path):
    if not path or not os.path.exists(path): return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
    if im.shape[1] != W:
        im = cv2.resize(im, (W, PH), interpolation=cv2.INTER_LINEAR)
    a = im[..., 3:4]
    c = lin(im[..., 2::-1])
    return c, a


def over(bg, layer, soften=0.6):
    if layer is None: return bg
    c, a = layer
    if soften:
        c = cv2.GaussianBlur(c * a, (0, 0), soften); a = cv2.GaussianBlur(a, (0, 0), soften)[..., None] if a.ndim == 3 else a
        if a.ndim == 2: a = a[..., None]
        return c + bg * (1 - a)
    return c * a + bg * (1 - a)


def bloom(img, thresh=0.85, sigma=(6, 24, 70), gain=(0.5, 0.35, 0.25)):
    hi = np.clip(img - thresh, 0, None)
    out = img.copy()
    for s_, g_ in zip(sigma, gain):
        out += g_ * cv2.GaussianBlur(hi, (0, 0), s_)
    return out


def flash(img, strength, y_line=None):
    """the whole scene lit by a brief blue-white light from the path"""
    if strength <= 0: return img
    h = img.shape[0]
    yy = np.arange(h, dtype=np.float32)[:, None, None]
    yl = y_line if y_line is not None else 0.62 * h
    fall = 1.0 / (1.0 + ((yy - yl) / (0.18 * h)) ** 2)
    tint = np.array([0.85, 0.9, 1.0], np.float32)
    return img + strength * fall * tint * (0.35 + img)


def shake(img, t, amp=10.0, decay=5.0, freq=11.0, seed=0):
    """damped camera shake t seconds after the shock reaches the camera"""
    if t < 0: return img
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 3)
    e = amp * math.exp(-decay * t)
    dx = e * math.sin(2 * math.pi * freq * t + ph[0])
    dy = e * 0.8 * math.sin(2 * math.pi * freq * 1.3 * t + ph[1])
    rot = e * 0.02 * math.sin(2 * math.pi * freq * 0.9 * t + ph[2])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0 + 0.004 * min(1.0, e / amp) + 0.012)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def grain(img, seed, s=0.006):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, s, img.shape[:2]).astype(np.float32)
    return img + n[..., None] * np.sqrt(np.clip(img, 0, 1) + 0.02)


def font(size, weight="SemiBold"):
    return ImageFont.truetype(FONT.format(weight), size)


def label(frame8, lines, alpha=1.0, corner="tl"):
    """lines: [(text, size, weight, colour)]; drawn in the picture's top-left with a soft shadow"""
    im = Image.fromarray(frame8)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x, y = 64, TOP + 48
    for text, size, weight, colour in lines:
        f = font(size, weight)
        for dx, dy in ((2, 2), (0, 2), (2, 0)):
            d.text((x + dx, y + dy), text, font=f, fill=(0, 0, 0, int(110 * alpha)))
        d.text((x, y), text, font=f, fill=colour + (int(255 * alpha),))
        y += int(size * 1.18)
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    return np.asarray(im)


def dust(path_d, path_g):
    """the dust renderer's output: premultiplied light, transmittance, ground shadow (sun let through)"""
    if not os.path.exists(path_d): return None
    d = cv2.imread(path_d, cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 32767.5
    g = cv2.imread(path_g, cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535
    if d.shape[1] != W:
        d = cv2.resize(d, (W, PH), interpolation=cv2.INTER_LINEAR); g = cv2.resize(g, (W, PH), interpolation=cv2.INTER_LINEAR)
    return d, g[..., 0:1], g[..., 1:2]


SUN_FRACTION = 0.45                       # of the light on open ground that comes straight from the sun


def frame(bg_lin, layer=None, fl=0.0, shake_t=-1.0, seed=0, lines=None, label_alpha=1.0, dust_layer=None, shake_amp=10.0):
    if dust_layer is not None:
        bg_lin = bg_lin * (1 - SUN_FRACTION * (1 - dust_layer[2]))
    img = over(bg_lin, layer)
    if dust_layer is not None:
        img = img * dust_layer[1] + dust_layer[0]
    img = flash(img, fl)
    img = bloom(img)
    img = shake(img, shake_t, amp=shake_amp, seed=seed)
    img = grain(img, seed)
    pic = (srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((H, W, 3), np.uint8)
    out[TOP:TOP + PH] = pic
    if lines: out = label(out, lines, label_alpha)
    return out


def props(path):
    """a props render (render_props.py): premultiplied linear rgb, alpha, z-depth, and relighting passes if present"""
    if not os.path.exists(path): return None
    d = np.load(path)
    rgba = d["rgba"].astype(np.float32)
    out = dict(rgb=rgba[..., :3], a=rgba[..., 3:4], depth=d["depth"].astype(np.float32))
    for k in ("normal", "position", "diffcol"):
        if k in d: out[k] = d[k].astype(np.float32)
    return out


def over_props(bg, p):
    return bg if p is None else p["rgb"] + bg * (1 - p["a"])


def relight(p, lights):
    """the light a set of point lights [((x, y, z), rgb W/sr)] adds to the props' diffuse surfaces (no shadows)"""
    if p is None or "position" not in p or not lights: return 0.0
    P, N, A = p["position"], p["normal"], p["diffcol"]
    add = np.zeros_like(A)
    for (lx, ly, lz), c in lights:
        L = np.array([lx, ly, lz], np.float32) - P
        d2 = (L ** 2).sum(-1, keepdims=True) + 0.01
        cos = np.clip((N * L).sum(-1, keepdims=True) / np.sqrt(d2), 0, 1)
        add += A / math.pi * cos / d2 * np.array(c, np.float32)
    return add * p["a"]


def scuff(img, uv, rx, ry, t_age, seed=4):
    """a dark scuff on the road where the ball hit: an ellipse of crushed asphalt, ragged"""
    if t_age <= 0: return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((xx - uv[0]) / rx) ** 2 + ((yy - uv[1]) / ry) ** 2
    rng = np.random.default_rng(seed)
    n = cv2.resize(rng.random((h // 6, w // 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    m = np.clip(1.3 - r - 0.6 * n, 0, 1) * 0.45
    return img * (1 - m[..., None])
