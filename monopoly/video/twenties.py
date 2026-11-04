"""Roaring Twenties pieces: public-domain period photos, sepia-toned, as full-frame newsreel shots or as deckle-edged
snapshots dropped on the desk, and Art Deco title cards in black and gold.

Photos live in ../art/source/twenties/ (provenance in its SOURCES.md).
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from engine import H, W, clamp, ease_out, sf, smooth, txt
from kit import F, El, Obj, paste, shadowed, with_alpha

PHOTOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'art', 'source', 'twenties')
GOLD = (206, 170, 86)
GOLD_DARK = (150, 116, 52)
NIGHT = (17, 14, 11)

_CACHE = {}


def trim(im, cap=0.12, margin=0.02):
    """Cut away a scan's border (the black film edge and the white card round a glass-plate print)."""
    sw = 400
    g = np.asarray(im.convert('L').resize((sw, max(8, int(sw * im.size[1] / im.size[0])))), np.float32)

    def depth(lines):
        n = 0
        for k, l in enumerate(lines[:int(len(lines) * cap)]):
            if (l < 40).mean() > 0.5 or (l > 232).mean() > 0.75:
                n = k + 1
        return n / len(lines)
    t, b = depth(g), depth(g[::-1])
    l, r = depth(g.T), depth(g.T[::-1])
    W0, H0 = im.size
    box = (int(W0 * (l + margin)), int(H0 * (t + margin)), int(W0 * (1 - r - margin)), int(H0 * (1 - b - margin)))
    return im.crop(box)


def photo(name, tone=True):
    """A period photo, border trimmed and sepia-toned: (RGB image). Cached."""
    key = (name, tone)
    if key not in _CACHE:
        im = trim(Image.open(os.path.join(PHOTOS, name)).convert('L'))
        im = ImageOps.autocontrast(im, cutoff=1)
        if tone:
            a = np.asarray(im, np.float32) / 255
            a = a ** 0.95
            stops = np.array([0.0, 0.5, 1.0])
            dark, mid, light = (30, 22, 15), (146, 114, 80), (240, 226, 196)
            rgb = np.stack([np.interp(a, stops, [dark[c], mid[c], light[c]]) for c in range(3)], -1)
            im = Image.fromarray(rgb.astype(np.uint8))
        else:
            im = im.convert('RGB')
        _CACHE[key] = im
    return _CACHE[key]


def cover(im, w, h, focus=(0.5, 0.5)):
    """Crop and scale to fill w x h."""
    s = max(w / im.size[0], h / im.size[1])
    nw, nh = int(math.ceil(im.size[0] * s)), int(math.ceil(im.size[1] * s))
    big = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = int((nw - w) * focus[0]), int((nh - h) * focus[1])
    return big.crop((x0, y0, x0 + w, y0 + h))


def vignette(w=W, h=H, strength=0.55):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    a = np.clip((d - 0.55) / 0.75, 0, 1) ** 1.6 * 255 * strength
    lay = Image.new('RGBA', (w, h), NIGHT + (0,))
    lay.putalpha(Image.fromarray(a.astype(np.uint8)))
    return lay


class KenBurns(El):
    """A full-frame photo with a slow push (or drift) across the page's life, toned and vignetted."""
    fade_in = 0

    def __init__(self, t0, t1, name, z0=1.0, z1=1.12, focus0=(0.5, 0.5), focus1=(0.5, 0.45), crop_focus=(0.5, 0.5)):
        super().__init__(t0)
        self.t_end, self.z0, self.z1, self.f0, self.f1 = t1, z0, z1, focus0, focus1
        base = cover(photo(name), int(W * 1.25), int(H * 1.25), crop_focus)
        self.base = base
        self.vig = vignette()

    def draw(self, img, t):
        x = smooth(clamp((t - self.t0) / max(0.1, self.t_end - self.t0)))
        z = self.z0 + (self.z1 - self.z0) * x
        fx = self.f0[0] + (self.f1[0] - self.f0[0]) * x
        fy = self.f0[1] + (self.f1[1] - self.f0[1]) * x
        BW, BH = self.base.size  # 1.25 x the frame: zoom 1 shows all of it, 1.25 is pixel for pixel
        w, h = BW / z, BH / z
        cx, cy = fx * BW, fy * BH
        cx, cy = clamp(cx, w / 2, BW - w / 2), clamp(cy, h / 2, BH - h / 2)
        frame = self.base.resize((W, H), Image.BILINEAR, box=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)).convert('RGBA')
        frame.alpha_composite(self.vig)
        img.paste(frame, (0, 0))


def deckle_print(name, w=760, caption=None, angle=0, crop_focus=(0.5, 0.5), aspect=None, seed=0):
    """A period snapshot: sepia print in a white deckle-edged border, with a typed caption on the border."""
    im = photo(name)
    if aspect:
        h0 = int(w / aspect)
        im = cover(im, w, h0, crop_focus)
    else:
        im = im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS)
    pad, bottom = 26, (70 if caption else 26)
    pw, ph = im.size[0] + 2 * pad, im.size[1] + pad + bottom
    card = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
    paper = sf.paper(pw, ph, base=(242, 236, 222), seed=40 + seed).convert('RGBA')
    # deckled edge: a scalloped mask
    m = Image.new('L', (pw, ph), 0)
    d = ImageDraw.Draw(m)
    d.rectangle([6, 6, pw - 7, ph - 7], fill=255)
    r = np.random.default_rng(seed)
    step = 14
    for x in range(0, pw, step):
        for y0 in (0, ph - 12):
            d.ellipse([x - 2, y0 + (2 if y0 else 0) + r.integers(-1, 2), x + step + 2, y0 + 10 + r.integers(-1, 2)], fill=255)
    for y in range(0, ph, step):
        for x0 in (0, pw - 12):
            d.ellipse([x0 + (2 if x0 else 0), y - 2, x0 + 10, y + step + 2], fill=255)
    paper.putalpha(m)
    card.alpha_composite(paper)
    card.paste(im.convert('RGB'), (pad, pad))
    ImageDraw.Draw(card).rectangle([pad, pad, pad + im.size[0] - 1, pad + im.size[1] - 1], outline=(60, 45, 30, 120), width=1)
    if caption:
        txt(card, (pw / 2, ph - bottom / 2 - 2), caption, F('type', 26), fill=(70, 52, 34), anchor='mm')
    if angle:
        card = card.rotate(angle, expand=True, resample=Image.BICUBIC)
    return card


def deco_frame(img, box, color=GOLD, width=3):
    """Art Deco double frame with stepped (ziggurat) corners."""
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    for inset, wdt in ((0, width), (12, 1)):
        a, b, c_, e = x0 + inset, y0 + inset, x1 - inset, y1 - inset
        s = 26 - inset // 2
        pts = [(a + s, b), (c_ - s, b), (c_ - s, b + s * 0.4), (c_ - s * 0.4, b + s * 0.4), (c_ - s * 0.4, b + s), (c_, b + s),
               (c_, e - s), (c_ - s * 0.4, e - s), (c_ - s * 0.4, e - s * 0.4), (c_ - s, e - s * 0.4), (c_ - s, e), (a + s, e),
               (a + s, e - s * 0.4), (a + s * 0.4, e - s * 0.4), (a + s * 0.4, e - s), (a, e - s), (a, b + s), (a + s * 0.4, b + s),
               (a + s * 0.4, b + s * 0.4), (a + s, b + s * 0.4), (a + s, b)]
        d.line(pts, fill=color + (255,), width=wdt, joint='curve')


def sunburst(img, cx, cy, r0, r1, n=21, spread=150, color=GOLD, alpha=150, width=2):
    """Fan of rays rising from (cx, cy), the classic Deco sunrise."""
    d = ImageDraw.Draw(img)
    for k in range(n):
        a = math.radians(-90 - spread / 2 + spread * k / (n - 1))
        d.line([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))],
               fill=color + (alpha,), width=width)


def deco_title(kicker, title, sub=None, w=1240, h=470):
    """A black-and-gold Deco plaque for a section title, to sit over a photo."""
    lay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    plate = Image.new('RGBA', (w, h), NIGHT + (226,))
    lay.alpha_composite(plate)
    ty = h / 2 + (12 if sub else 30)
    fk, ft = F('scb', 38), F('deco', 104)
    rays = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    sunburst(rays, w / 2, h - 30, 60, h * 0.95, n=25, spread=160, alpha=46, width=3)
    # keep the rays out from behind the words: a soft window round the kicker and the title
    m = Image.new('L', (w, h), 255)
    md = ImageDraw.Draw(m)
    for box in (md.multiline_textbbox((w / 2, ty), title, font=ft, anchor='mm', align='center', spacing=10),
                md.textbbox((w / 2, 92), kicker, font=fk, anchor='mm')):
        md.rounded_rectangle([box[0] - 30, box[1] - 18, box[2] + 30, box[3] + 18], 20, fill=0)
    m = m.filter(ImageFilter.GaussianBlur(14))
    rays.putalpha(Image.fromarray((np.asarray(rays.getchannel('A'), np.float32) * np.asarray(m, np.float32) / 255).astype(np.uint8)))
    lay.alpha_composite(rays)
    deco_frame(lay, (14, 14, w - 15, h - 15))
    d = ImageDraw.Draw(lay)
    half = d.textlength(kicker, font=fk) / 2 + 28
    for x in (w / 2 - half - 110, w / 2 + half):  # rules either side of the kicker
        d.line([(x, 92), (x + 110, 92)], fill=GOLD + (255,), width=2)
    txt(lay, (w / 2, 92), kicker, fk, fill=(226, 192, 110), anchor='mm')
    txt(lay, (w / 2, ty), title, ft, fill=(244, 232, 200), anchor='mm', align='center', spacing=10)
    if sub:
        txt(lay, (w / 2, h - 70), sub, F('sc', 30), fill=GOLD, anchor='mm')
    return lay


class Plaque(El):
    """A Deco plaque that rises into place."""
    fade_in = 0.5

    def __init__(self, t0, layer, center, t1=None, rise=40):
        super().__init__(t0, t1)
        self.layer, self.center, self.rise = layer, center, rise

    def draw(self, img, t):
        e = ease_out((t - self.t0) / 0.6)
        x = self.center[0] - self.layer.size[0] / 2
        y = self.center[1] - self.layer.size[1] / 2 + self.rise * (1 - e)
        paste(img, with_alpha(self.layer, self.alpha(t)), (x, y))


def caption_bar(text, w=None):
    """A newsreel-style caption strip: dark band, gold rule, typed place and date."""
    f = F('type', 30)
    tw = int(ImageDraw.Draw(Image.new('RGBA', (4, 4))).textlength(text, font=f))
    w = w or tw + 80
    lay = Image.new('RGBA', (w, 64), NIGHT + (200,))
    ImageDraw.Draw(lay).line([(0, 63), (w, 63)], fill=GOLD + (255,), width=2)
    txt(lay, (w / 2, 32), text, f, fill=(238, 226, 196), anchor='mm')
    return lay


def photo_bg(name, crop_focus=(0.5, 0.5), darken=0.0, scale=1.0):
    """A full-frame page background: the toned photo, cropped to the frame, vignetted (and darkened for text)."""
    bg = cover(photo(name), int(W * scale), int(H * scale), crop_focus).resize((W, H), Image.LANCZOS).convert('RGBA')
    bg.alpha_composite(vignette())
    if darken:
        bg.alpha_composite(Image.new('RGBA', (W, H), NIGHT + (int(255 * darken),)))
    return bg


def photo_page(scene, name, t0, caption=None, crop_focus=(0.5, 0.5), zoom=0.07, focus=(0.5, 0.45), enter='slide', darken=0.0,
               cap_xy=(60, H - 140)):
    """A newsreel cutaway: a full-frame period photo with a slow push-in and, if given, a place-and-date strip."""
    p = scene.page(photo_bg(name, crop_focus, darken), t0, enter=enter, zoom=zoom, focus=focus)
    if caption:
        bar = caption_bar(caption)
        p.add(Obj(t0 + 0.5, bar, cap_xy, how='left', shadow=None))
    return p


def deco_opener(scene, n, title, name, caption=None, crop_focus=(0.5, 0.5), focus=(0.5, 0.45)):
    """A decision opens on a period photo with a black-and-gold Deco plaque rising over it."""
    p = photo_page(scene, name, 0, caption, crop_focus, zoom=0.06, focus=focus)
    words = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']
    lines = title.count('\n') + 1
    plate = deco_title('DECISION  %s' % words[n - 1], title, w=1180, h=330 + 110 * lines)
    p.add(Plaque(0.15, plate, (W / 2, H / 2 - 30)))
    return p
