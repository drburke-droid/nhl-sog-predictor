"""Team colours, portraits and sport-specific charts for the countdown cards.

Each card is drawn in the colours of the team (or country) its athlete is most associated with. The chart is the
model's own comparison (everyone else's best, each season, in this season's units; the curve; one tick per step),
drawn with sport-specific marks: pucks, baseballs, bats, strikeout K's, game tiles, basketballs, javelins in the
sector, landing marks in the pit. Portraits are cut out of their backgrounds (rembg, photos/cut/) and toned in the
team's colours (a duotone), so old black-and-white photos and new colour photos sit in the same scene.
"""
import math, os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 1080
BG = (15, 18, 21)


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c[:3])


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


# team (or country) most associated with each season; bg = the dominant colour, accent = the highlight that reads
# on the dark tinted background, light = the colour of the portrait's highlights
PALETTE = {
    10: dict(team="Seattle Mariners", bg="#0C2C56", accent="#2DB7A9", light="#C4CED4"),
    9: dict(team="New York Yankees", bg="#0C2340", accent="#F1F3F6", light="#C4CED3", stripes=True),
    8: dict(team="Detroit Pistons", bg="#1D42BA", accent="#EF3B4E", light="#D9DDE3"),
    7: dict(team="Golden State Warriors", bg="#1D428A", accent="#FFC72C", light="#FFF1C9"),
    6: dict(team="Brooklyn Robins", bg="#005A9C", accent="#8CC6F2", light="#E6F1FA"),
    5: dict(team="Czech Republic", bg="#11457E", accent="#EE3340", light="#F2F2F2"),
    4: dict(team="Edmonton Oilers", bg="#00387A", accent="#FF6A13", light="#FFE3D1"),
    3: dict(team="New York Yankees", bg="#0C2340", accent="#F1F3F6", light="#C4CED3", stripes=True),
    2: dict(team="Philadelphia Warriors", bg="#1D428A", accent="#FDB927", light="#FFF1C9"),
    1: dict(team="United States", bg="#0A3161", accent="#E8343F", light="#F2F2F2", stars=True),
}

# how each cut-out is framed: box height in px, centre x, and the bottom edge (y); Beamon floats mid-jump
FRAMING = {"ichiro": (1010, 1585, 1110), "ruth": (1040, 1560, 1120), "rodman": (900, 1565, 1010),
           "curry": (980, 1565, 1100), "vance": (700, 1560, 1100), "zelezny": (780, 1570, 1095),
           "gretzky": (800, 1560, 1095), "dimaggio": (880, 1560, 1100), "chamberlain": (960, 1560, 1100),
           "beamon": (640, 1560, 660), "ledecky": (980, 1570, 1110)}
CUT_TRIM = {"beamon": (0.0, 0.0, 1.0, 0.745)}
CUT_ERASE = {}
OVAL = {"rodman"}   # l, t, r, b fractions kept


# ------------------------------------------------------------------ backgrounds and portraits
@lru_cache(maxsize=12)
def background(rank):
    """the card's base image: dark surface, a glow of the team colour behind the portrait, a team texture"""
    p = PALETTE[rank]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - 1560) / 1150) ** 2 + ((yy - 380) / 900) ** 2)
    glow = np.clip(1.0 - d, 0, 1) ** 1.6 * 0.62
    left = np.clip((xx - 200) / 1400, 0, 1) * 0.16                       # a little colour under the chart too
    k = np.maximum(glow, left)[..., None]
    base = np.array(BG, np.float32)
    col = np.array(rgb(p["bg"]), np.float32)
    img = base * (1 - k) + col * k
    if p.get("stripes"):                                                  # Yankee pinstripes
        stripe = ((xx.astype(int) % 30) < 2) & (xx > 1150)
        img[stripe] = img[stripe] * 0.86 + 255 * 0.14 * k[stripe][:, 0:1].clip(0.15, 1) * 0.9
    if p.get("stars"):                                                    # a few faint stars, top right
        rs = np.random.default_rng(1968)
        for _ in range(26):
            x, y = int(rs.uniform(1250, 1900)), int(rs.uniform(30, 330))
            img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] * 0.6 + 255 * 0.4 * 0.35
    vig = np.clip(1.15 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2), 0.75, 1.0)[..., None]
    return np.clip(img * vig, 0, 255).astype(np.uint8)


def duotone(im, dark, mid, light):
    """map luminance onto dark -> mid -> light"""
    a = np.asarray(im.convert("RGBA"), np.float32)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255.0
    lo, hi = np.percentile(lum[a[..., 3] > 128], [2, 98]) if (a[..., 3] > 128).any() else (0, 1)
    lum = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)
    d, m, l = (np.array(c, np.float32) for c in (dark, mid, light))
    t = lum[..., None]
    out = np.where(t < 0.5, d + (m - d) * (t / 0.5), m + (l - m) * ((t - 0.5) / 0.5))
    return np.concatenate([out, a[..., 3:4]], axis=2)


@lru_cache(maxsize=12)
def portrait_layer(key, rank):
    """full-frame RGBA: the cut-out, toned in the team's colours, fading into the text panel below it"""
    p = PALETTE[rank]
    if key in OVAL:                                    # halftone newsprint: a soft oval instead of a hard matte
        im = Image.open(os.path.join(HERE, "photos", key + ".png")).convert("RGBA")
        a_ = np.asarray(im, np.float32).copy()
        yy, xx = np.mgrid[0:im.height, 0:im.width].astype(np.float32)
        e = ((xx - im.width * 0.52) / (im.width * 0.50)) ** 2 + ((yy - im.height * 0.52) / (im.height * 0.56)) ** 2
        a_[..., 3] = np.clip((1.0 - e) / 0.45, 0, 1) * 255
        im = Image.fromarray(a_.astype(np.uint8))
    else:
        im = Image.open(os.path.join(HERE, "photos", "cut", key + ".png")).convert("RGBA")
    l, t, r, b = CUT_TRIM.get(key, (0, 0, 1, 1))
    im = im.crop((int(l * im.width), int(t * im.height), int(r * im.width), int(b * im.height)))
    for (e0, e1, e2, e3) in CUT_ERASE.get(key, ()):                       # leftover background the matte kept
        ImageDraw.Draw(im).rectangle((int(e0 * im.width), int(e1 * im.height), int(e2 * im.width), int(e3 * im.height)),
                                     fill=(0, 0, 0, 0))
    bb = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    im = im.crop(bb)
    hgt, cx, bottom = FRAMING[key]
    s = hgt / im.height
    im = im.resize((max(1, int(im.width * s)), hgt), Image.LANCZOS)
    col = rgb(p["bg"])
    tone = duotone(im, mix(BG, col, 0.35), mix(col, (255, 255, 255), 0.22), mix(rgb(p["light"]), (255, 255, 255), 0.35))
    layer = np.zeros((H, W, 4), np.float32)
    x0, y0 = int(cx - im.width / 2), int(bottom - hgt)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + im.width), min(H, y0 + hgt)
    if xb > xa and yb > ya:
        layer[ya:yb, xa:xb] = tone[ya - y0:yb - y0, xa - x0:xb - x0]
    yy = np.arange(H, dtype=np.float32)[:, None]
    fade = np.clip(1.0 - (yy - 470) / 420.0, 0.22, 1.0)                 # fades where the score and lines sit
    layer[..., 3] *= fade * 0.95
    # a soft rim of the accent colour around the figure
    a = Image.fromarray((layer[..., 3]).astype(np.uint8))
    rim = np.asarray(a.filter(ImageFilter.GaussianBlur(14)), np.float32) / 255.0
    rim = np.clip(rim - layer[..., 3] / 255.0, 0, 1) * 0.55
    acc = np.array(rgb(p["accent"]), np.float32)
    out = np.zeros_like(layer)
    alpha = layer[..., 3:4] / 255.0
    out[..., :3] = layer[..., :3] * alpha + acc * rim[..., None] * (1 - alpha)
    out[..., 3] = np.clip(layer[..., 3] / 255.0 + rim, 0, 1) * 255
    with np.errstate(invalid="ignore", divide="ignore"):
        out[..., :3] = np.where(out[..., 3:4] > 0, out[..., :3] / np.maximum(out[..., 3:4] / 255.0, 1e-6), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def scrim():
    """dark gradient under the right-hand text panel"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.clip((yy - 470) / 230.0, 0, 1) * np.clip((xx - 1150) / 90.0, 0, 1) * 0.78
    out = np.zeros((H, W, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = BG
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ sprites (drawn 4x, downsampled)
SS = 4


def _canvas(w, h):
    return Image.new("RGBA", (int(w * SS), int(h * SS)), (0, 0, 0, 0))


def _down(im, w, h):
    return im.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)


@lru_cache(maxsize=64)
def baseball(d, tone="full"):
    im = _canvas(d, d)
    g = ImageDraw.Draw(im)
    D = d * SS
    face, edge, stitch = ((246, 243, 236, 255), (170, 165, 155, 255), (200, 30, 45, 255))
    if tone == "dim":
        face, edge, stitch = ((150, 156, 165, 150), (110, 116, 124, 150), (120, 80, 90, 150))
    g.ellipse((SS, SS, D - SS, D - SS), fill=face, outline=edge, width=max(1, SS))
    w = max(1, int(SS * d / 14))
    g.arc((-D * 0.42, D * 0.08, D * 0.36, D * 0.92), -50, 50, fill=stitch, width=w)
    g.arc((D * 0.64, D * 0.08, D * 1.42, D * 0.92), 130, 230, fill=stitch, width=w)
    return _down(im, d, d)


@lru_cache(maxsize=64)
def basketball(d, tone="full"):
    im = _canvas(d, d)
    g = ImageDraw.Draw(im)
    D = d * SS
    face, seam = ((227, 110, 38, 255), (40, 24, 16, 255))
    if tone == "dim":
        face, seam = ((150, 120, 100, 150), (70, 60, 55, 150))
    g.ellipse((SS, SS, D - SS, D - SS), fill=face)
    w = max(1, int(SS * d / 16))
    g.line((D / 2, SS, D / 2, D - SS), fill=seam, width=w)
    g.line((SS, D / 2, D - SS, D / 2), fill=seam, width=w)
    g.arc((-D * 0.35, SS, D * 0.42, D - SS), -62, 62, fill=seam, width=w)
    g.arc((D * 0.58, SS, D * 1.35, D - SS), 118, 242, fill=seam, width=w)
    g.ellipse((SS, SS, D - SS, D - SS), outline=seam, width=w)
    return _down(im, d, d)


@lru_cache(maxsize=64)
def puck(w, accent=None, tone="full"):
    h = w * 0.5
    im = _canvas(w, h)
    g = ImageDraw.Draw(im)
    Wd, Hd = w * SS, h * SS
    body, top, rim = ((22, 22, 24, 255), (48, 48, 52, 255), (95, 95, 100, 255))
    if tone == "dim":
        body, top, rim = ((60, 64, 70, 150), (85, 90, 96, 150), (115, 120, 128, 150))
    g.rounded_rectangle((SS, Hd * 0.30, Wd - SS, Hd - SS), radius=Hd * 0.32, fill=body)
    g.ellipse((SS, SS, Wd - SS, Hd * 0.62), fill=top, outline=(rgb(accent) + (255,)) if accent else rim, width=SS * 2)
    return _down(im, w, h)


@lru_cache(maxsize=4)
def _font(size_px, bold=True):
    path = font_manager.findfont(FontProperties(family="Inter Display", weight="bold" if bold else "regular"))
    return ImageFont.truetype(path, int(size_px))


@lru_cache(maxsize=64)
def kmark(h, color, alpha=255):
    im = _canvas(h * 0.9, h)
    g = ImageDraw.Draw(im)
    f = _font(h * SS * 1.05)
    g.text((h * SS * 0.45, h * SS * 0.5), "K", font=f, fill=rgb(color) + (alpha,), anchor="mm")
    return _down(im, h * 0.9, h)


@lru_cache(maxsize=64)
def bat(length, thick, tone="full"):
    im = _canvas(length, thick)
    g = ImageDraw.Draw(im)
    L, T = length * SS, thick * SS
    wood, grain = ((205, 160, 105, 255), (160, 115, 70, 255))
    if tone == "dim":
        wood, grain = ((120, 112, 104, 140), (95, 90, 84, 140))
    poly = [(SS, T * 0.40), (L * 0.45, T * 0.36), (L * 0.70, T * 0.12), (L - SS, T * 0.10),
            (L - SS, T * 0.90), (L * 0.70, T * 0.88), (L * 0.45, T * 0.64), (SS, T * 0.60)]
    g.polygon(poly, fill=wood)
    g.ellipse((L - T * 0.5, T * 0.10, L - SS, T * 0.90), fill=wood)
    g.ellipse((0, T * 0.28, T * 0.35, T * 0.72), fill=grain)                            # the knob
    g.line((L * 0.55, T * 0.5, L * 0.92, T * 0.5), fill=grain, width=max(1, SS))
    return _down(im, length, thick)


@lru_cache(maxsize=64)
def tile(d, color, alpha=255):
    im = _canvas(d, d)
    g = ImageDraw.Draw(im)
    D = d * SS
    c = rgb(color) + (alpha,)
    g.rounded_rectangle((SS, SS, D - SS, D - SS), radius=D * 0.18, outline=c, width=max(1, int(SS * 1.6)))
    g.line((D * 0.28, D * 0.72, D * 0.72, D * 0.28), fill=c, width=max(1, int(SS * 1.8)))        # a scorebook "hit"
    return _down(im, d, d)


ICONS = os.path.join(HERE, "icons")


@lru_cache(maxsize=96)
def model_sprite(name, w, tone="full"):
    """a rendered model (icons/<name>.png) at width w; 'dim' greys it for everyone else's stacks"""
    im = Image.open(os.path.join(ICONS, name + ".png")).convert("RGBA")
    h = max(1, int(round(im.height * w / im.width)))
    im = im.resize((max(1, int(round(w))), h), Image.LANCZOS)
    if tone == "dim":
        a = np.asarray(im, np.float32).copy()
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        grey = 0.5 * lum + 0.5 * 150.0
        for i in range(3):
            a[..., i] = grey
        a[..., 3] *= 0.62
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im


def paste_h(canvas, sprite, cx, cy, frac=1.0):
    """a horizontal sprite centred at (cx, cy); frac < 1 keeps only its left part"""
    sp = sprite
    if frac < 0.999:
        keep = max(1, int(round(sp.width * frac)))
        sp = sp.crop((0, 0, keep, sp.height))
        cx = cx - (sprite.width - keep) / 2
    canvas.alpha_composite(sp, (int(round(cx - sp.width / 2)), int(round(cy - sp.height / 2))))


def paste(canvas, sprite, cx, cy, alpha=1.0, frac=1.0):
    """alpha-composite a sprite centred at (cx, cy); frac < 1 keeps only its bottom part (a partial icon)"""
    if alpha <= 0.01 or frac <= 0.01:
        return
    sp = sprite
    if frac < 0.999:
        keep = max(1, int(round(sp.height * frac)))
        sp = sp.crop((0, sp.height - keep, sp.width, sp.height))
        cy = cy + (sprite.height - keep) / 2
    if alpha < 0.999:
        a = np.asarray(sp, np.float32).copy()
        a[..., 3] *= alpha
        sp = Image.fromarray(a.astype(np.uint8))
    canvas.alpha_composite(sp, (int(round(cx - sp.width / 2)), int(round(cy - sp.height / 2))))


def glow(canvas, box, color, strength=0.45, blur=16):
    """a soft glow of the accent colour behind a region (x0, y0, x1, y1)"""
    g = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).rounded_rectangle(box, radius=10, fill=rgb(color) + (int(255 * strength),))
    canvas.alpha_composite(g.filter(ImageFilter.GaussianBlur(blur)))


# ------------------------------------------------------------------ the charts
import look                                                           # noqa: E402
from look import ax_px, fs, px, text, outlined, ease, ease_out, clamp, fade, MUTED, INK, INK2  # noqa: E402
from matplotlib.ticker import MaxNLocator                             # noqa: E402

CURVE = "#ece9e1"
BOX = (130, 300, 1150, 968)
SCENE = {   # rank: how the comparison is drawn
    10: dict(kind="stack", icon="baseball", per=10, word="hits", key="one ball = 10 hits", bins=True, court="diamond"),
    9: dict(kind="hstack", icon="bat", per=10, word="home runs", key="one bat = 10 home runs"),
    8: dict(kind="stack", icon="basketball", per=1, word="rebounds a game", key="one ball = 1 rebound a game", court="key"),
    7: dict(kind="stack", icon="basketball", per=20, word="three-pointers", key="one ball = 20 three-pointers", court="arc"),
    6: dict(kind="stack", icon="k", per=10, word="strikeouts", key="one K = 10 strikeouts"),
    5: dict(kind="javelin"),
    4: dict(kind="stack", icon="puck", per=5, word="assists", key="one puck = 5 assists", court="rink"),
    3: dict(kind="stack", icon="tile", per=2, word="games in a row", key="one square = 2 games with a hit"),
    2: dict(kind="stack", icon="basketball", per=2, word="points a game", key="one ball = 2 points a game", court="circle"),
    1: dict(kind="pit"),
}


def _poly(ax, xs, ys, color, lw, alpha, z=6, ls="-", outline=True):
    kw = dict(path_effects=outlined(5)) if outline else {}
    ax.plot(xs, ys, color=color, lw=px(lw), alpha=alpha, zorder=z, ls=ls, solid_capstyle="round", **kw)


class Stack:
    """each season's best by anyone else as a stack of sport icons; the athlete's stack keeps climbing past the
    curve, one tick per step"""

    def __init__(self, c, pal, spec, explain=False):
        r = c["ruler"]
        self.c, self.p, self.s, self.explain = c, pal, spec, explain
        self.year = c["year"]
        pts = {int(s): float(v) for s, v in r["window"]}
        self.cols = list(range(min(min(pts), self.year), max(max(pts), self.year) + 1))
        self.vals = {s: v for s, v in pts.items() if s != self.year}
        self.x, self.base, self.step = r["x"], r["base"], r["step"]
        self.trend = np.array(r["trend"])
        x0, y0, x1, y1 = BOX
        self.w, self.h = x1 - x0, y1 - y0
        self.pl, self.pr, self.pt, self.pb = 64, 18, 124, 40
        self.cw = (self.w - self.pl - self.pr) / len(self.cols)
        self.vmax = max(max(self.vals.values()), self.x) * 1.04
        self.sy = (self.h - self.pt - self.pb) / self.vmax
        self.ipx = spec["per"] * self.sy                              # pixels per icon
        self.acc = pal["accent"]

    def X(self, s):
        return self.pl + (s - self.cols[0] + 0.5) * self.cw

    def Y(self, v):
        return self.h - self.pb - v * self.sy

    def sprite(self, mine):
        kind, cw, ip = self.s["icon"], self.cw, self.ipx
        tone = "full" if mine else "dim"
        if kind == "baseball":
            return model_sprite("baseball", int(min(cw * 0.8, ip * 1.04)), tone)
        if kind == "basketball":
            return model_sprite("basketball", int(min(cw * 0.86, ip * 1.04)), tone)
        if kind == "puck":
            return model_sprite("puck", int(min(cw * 0.96, ip / 0.6)), tone)
        if kind == "k":
            return kmark(int(min(cw * 0.95, ip * 1.0)), self.acc if mine else "#7f858e", 255 if mine else 165)
        if kind == "bat":
            return bat(int(cw * 0.92), max(4, int(ip * 1.05)), tone)
        return tile(int(min(cw * 0.82, ip * 0.92)), self.acc if mine else "#8b919a", 255 if mine else 170)

    def column(self, canvas, s, v, mine):
        """icons from 0 up to v (the top one partial)"""
        sp = self.sprite(mine)
        n = v / self.s["per"]
        k = 0
        while k < n:
            frac = min(1.0, n - k)
            cy = self.Y((k + 0.5) * self.s["per"])
            if frac < 1:
                cy = self.Y(k * self.s["per"]) - sp.height / 2
            paste(canvas, sp, self.X(s), cy, frac=frac)
            k += 1

    def court(self, g):
        c = self.s.get("court")
        line = (255, 255, 255, 22)
        w, h = self.w * 2, self.h * 2
        if c == "rink":
            g.rectangle((w * 0.36, 0, w * 0.37, h), fill=(80, 140, 255, 30))
            for cx_ in (0.2, 0.62, 0.86):
                g.ellipse((w * cx_ - 90, h * 0.42 - 90, w * cx_ + 90, h * 0.42 + 90), outline=(255, 80, 80, 26), width=4)
        elif c == "key":
            g.rectangle((w * 0.38, h * 0.12, w * 0.62, h * 1.1), outline=line, width=4)
            g.ellipse((w * 0.38, h * 0.12 - w * 0.12, w * 0.62, h * 0.12 + w * 0.12), outline=line, width=4)
        elif c == "arc":
            g.arc((w * 0.08, -h * 0.75, w * 0.92, h * 0.95), 0, 180, fill=line, width=5)
        elif c == "circle":
            g.ellipse((w * 0.5 - 170, h * 0.45 - 170, w * 0.5 + 170, h * 0.45 + 170), outline=line, width=4)
            g.line((w * 0.5, 0, w * 0.5, h), fill=line, width=4)
        elif c == "diamond":
            g.polygon([(w * 0.5, h * 0.18), (w * 0.82, h * 0.55), (w * 0.5, h * 0.92), (w * 0.18, h * 0.55)], outline=line, width=4)

    def prog(self, t):
        return ease(clamp((t - 14.2) / 4.0))

    def shown(self, t, i, v):
        ts = 1.2 + 8.6 * i / len(self.cols)
        return v * ease(clamp((t - ts) / 1.3))

    def draw(self, fig, t, al):
        x0, y0, x1, y1 = BOX
        ax = ax_px(fig, x0, y0, x1, y1)
        ax.set_axis_off()
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        cv = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        bgl = Image.new("RGBA", (self.w * 2, self.h * 2), (0, 0, 0, 0))
        self.court(ImageDraw.Draw(bgl))
        cv.alpha_composite(bgl.resize((self.w, self.h), Image.LANCZOS))
        if self.s.get("bins"):
            g = ImageDraw.Draw(cv)
            for s in self.cols:
                xc = self.X(s)
                g.rounded_rectangle((xc - self.cw * 0.46, self.Y(self.vmax * 0.985), xc + self.cw * 0.46, self.Y(0) + 2),
                                    radius=6, outline=(255, 255, 255, 34), width=2)
        prog = self.prog(t)
        mine_v = self.shown(t, self.cols.index(self.year), min(self.x, self.base))
        if t >= 14.2:
            mine_v = self.base + (self.x - self.base) * prog
            xc = self.X(self.year)
            glow(cv, (xc - self.cw * 0.6, self.Y(mine_v) - 10, xc + self.cw * 0.6, self.Y(0)), self.acc, 0.38 * min(1, (t - 14.2) * 2))
        for i, s in enumerate(self.cols):
            if s in self.vals:
                self.column(cv, s, self.shown(t, i, self.vals[s]), False)
        self.column(cv, self.year, mine_v, True)
        ax.imshow(np.asarray(cv), extent=(0, self.w, self.h, 0), alpha=al, zorder=2, interpolation="antialiased")
        # axes furniture
        ax.plot([self.pl - 8, self.w - self.pr], [self.Y(0), self.Y(0)], color="#5a5f66", lw=px(1.5), alpha=al, zorder=3)
        for v in MaxNLocator(4).tick_values(0, self.vmax)[1:]:
            if v < self.vmax:
                ax.plot([self.pl - 8, self.w - self.pr], [self.Y(v)] * 2, color="#ffffff", alpha=0.06 * al, lw=px(1), zorder=1)
                ax.text(self.pl - 14, self.Y(v), f"{v:g}", ha="right", va="center", fontsize=fs(17), color=MUTED, alpha=al)
        for s in self.cols:
            if s % 5 == 0:
                ax.text(self.X(s), self.Y(0) + 24, str(s), ha="center", va="center", fontsize=fs(17), color=MUTED, alpha=al)
        c = self.c
        conv = c["ruler"]["converted"]
        unit = self.s["word"]
        if conv:
            games = f" and {c['games_per_team']:.0f} games" if c["games_per_team"] else ""
            top = f"{unit} · other seasons converted to {c['season']}'s league rate{games}"
        else:
            top = unit
        ax.text(0, 8, top, ha="left", va="top", fontsize=fs(19), color=INK2, alpha=al)
        ax.text(0, 36, f"{self.s['key']} · each stack: the best by anyone else that season", ha="left", va="top",
                fontsize=fs(17), color=MUTED, alpha=al)
        # the curve
        tc = clamp((t - 11.0) / 3.0)
        if tc > 0:
            tr = self.trend
            k = max(2, int(len(tr) * tc))
            xs = [self.X(s) - 0.0 for s in tr[:k, 0]]
            _poly(ax, xs, [self.Y(v) for v in tr[:k, 1]], CURVE, 3.2, al * 0.95, z=6)
            if tc >= 1:
                ax.text(self.X(tr[-1, 0]) - 4, self.Y(tr[-1, 1]) - 16, "the curve", ha="right", va="bottom",
                        fontsize=fs(18), color=CURVE, alpha=al, path_effects=outlined(5))
        # the ruler on the athlete's stack
        if t >= 14.2:
            xc = self.X(self.year)
            right = self.year < self.cols[-3]
            top_v = self.base + (self.x - self.base) * prog
            for kk in range(1, int((top_v - self.base) / self.step + 1e-9) + 1):
                yk = self.Y(self.base + kk * self.step)
                ax.plot([xc - self.cw * 0.62, xc + self.cw * 0.62], [yk, yk], color=self.acc, lw=px(2.4), alpha=al,
                        zorder=8, path_effects=outlined(4))
                ax.text(xc + (self.cw * 0.8 if right else -self.cw * 0.8), yk, f"{kk}", ha="left" if right else "right",
                        va="center", fontsize=fs(17), color=self.acc, weight="semibold", alpha=al, zorder=8,
                        path_effects=outlined(5))
            yb = self.Y(self.base)
            ax.plot([xc - self.cw * 0.62, xc + self.cw * 0.62], [yb, yb], color=CURVE, lw=px(2.4), alpha=al, zorder=8,
                    path_effects=outlined(4))
            if prog > 0.97:
                lab = f"{look_value(c)}"
                ax.text(xc, self.Y(self.x) - 22, lab, ha="center", va="bottom", fontsize=fs(30), color=self.acc,
                        weight="bold", family="Inter Display", alpha=al, zorder=9, path_effects=outlined(6))
        if self.explain:
            self.callouts(ax, t, al)

    def callouts(self, ax, t, al):
        """card #10 only: how to read the chart"""
        items = [(4.0, 10.6, self.cols[3], self.vals.get(self.cols[3], self.base) * 0.62, "each bin: the most hits by anyone\nelse that season"),
                 (11.5, 15.8, self.cols[-6], None, "the curve: where they usually land"),
                 (15.6, 21.5, self.year, None, "each tick: one step,\nthe usual give-or-take")]
        for t0, t1, s, v, msg in items:
            a = al * fade(t, t0, t1, 0.5, 0.6)
            if a <= 0:
                continue
            if v is None and s == self.year:
                tx, ty = self.X(s), self.Y(self.base + 2.2 * self.step)
                xt, yt = tx - 230, ty - 40
            elif v is None:
                tr = self.trend
                vv = float(np.interp(s, tr[:, 0], tr[:, 1]))
                tx, ty = self.X(s), self.Y(vv)
                xt, yt = tx - 60, ty - 150
            else:
                tx, ty = self.X(s), self.Y(v)
                xt, yt = tx + 60, ty - 170
            ax.annotate(msg, (tx, ty), xytext=(xt, yt), textcoords="data", fontsize=fs(21), color=INK, weight="semibold",
                        ha="center", va="bottom", alpha=a, zorder=12, path_effects=outlined(6),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=px(2), alpha=a, shrinkA=4, shrinkB=6))


class HStack(Stack):
    """rows instead of columns: each season's best by anyone else as bats laid end to end (a bat is long and thin);
    the curve runs down the rows and the ticks stand on the athlete's row"""

    def __init__(self, c, pal, spec, explain=False):
        super().__init__(c, pal, spec, explain)
        self.pl, self.pr, self.pt, self.pb = 64, 60, 84, 46
        self.rh = (self.h - self.pt - self.pb) / len(self.cols)
        self.sx = (self.w - self.pl - self.pr) / self.vmax
        self.blen = spec["per"] * self.sx

    def X(self, v):
        return self.pl + v * self.sx

    def Yr(self, s):
        return self.pt + (s - self.cols[0] + 0.5) * self.rh

    def row(self, canvas, s, v, mine):
        sp = model_sprite("bat", int(round(self.blen * 0.97)), "full" if mine else "dim")
        n = v / self.s["per"]
        k = 0
        while k < n:
            frac = min(1.0, n - k)
            paste_h(canvas, sp, self.X((k + 0.5) * self.s["per"]), self.Yr(s), frac=frac)
            k += 1

    def draw(self, fig, t, al):
        x0, y0, x1, y1 = BOX
        ax = ax_px(fig, x0, y0, x1, y1)
        ax.set_axis_off()
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        cv = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        prog = self.prog(t)
        mine_v = self.shown(t, self.cols.index(self.year), min(self.x, self.base))
        if t >= 14.2:
            mine_v = self.base + (self.x - self.base) * prog
            yc = self.Yr(self.year)
            glow(cv, (self.X(0), yc - self.rh * 0.9, self.X(mine_v) + 10, yc + self.rh * 0.9), self.acc,
                 0.38 * min(1, (t - 14.2) * 2))
        for i, s in enumerate(self.cols):
            if s in self.vals:
                self.row(cv, s, self.shown(t, i, self.vals[s]), False)
        self.row(cv, self.year, mine_v, True)
        ax.imshow(np.asarray(cv), extent=(0, self.w, self.h, 0), alpha=al, zorder=2, interpolation="antialiased")
        ax.plot([self.X(0)] * 2, [self.pt - 6, self.h - self.pb + 4], color="#5a5f66", lw=px(1.5), alpha=al, zorder=3)
        for v in MaxNLocator(5).tick_values(0, self.vmax)[1:]:
            if v < self.vmax:
                ax.plot([self.X(v)] * 2, [self.pt - 6, self.h - self.pb + 4], color="#ffffff", alpha=0.06 * al, lw=px(1), zorder=1)
                ax.text(self.X(v), self.h - self.pb + 22, f"{v:g}", ha="center", va="center", fontsize=fs(17), color=MUTED, alpha=al)
        for s in self.cols:
            if s % 5 == 0:
                ax.text(self.pl - 12, self.Yr(s), str(s), ha="right", va="center", fontsize=fs(16), color=MUTED, alpha=al)
        c = self.c
        games = f" and {c['games_per_team']:.0f} games" if c["games_per_team"] else ""
        ax.text(0, 8, f"{self.s['word']} · other seasons converted to {c['season']}'s league rate{games}", ha="left", va="top",
                fontsize=fs(19), color=INK2, alpha=al)
        ax.text(0, 36, f"{self.s['key']} · each row: the best by anyone else that season", ha="left", va="top",
                fontsize=fs(17), color=MUTED, alpha=al)
        tc = clamp((t - 11.0) / 3.0)
        if tc > 0:
            tr = self.trend
            k = max(2, int(len(tr) * tc))
            _poly(ax, [self.X(v) for v in tr[:k, 1]], [self.Yr(s) for s in tr[:k, 0]], CURVE, 3.2, al * 0.95, z=6)
            if tc >= 1:
                ax.text(self.X(tr[-1, 1]) + 10, self.Yr(tr[-1, 0]), "the curve", ha="left", va="center", fontsize=fs(18),
                        color=CURVE, alpha=al, path_effects=outlined(5))
        if t >= 14.2:
            yc = self.Yr(self.year)
            top_v = self.base + (self.x - self.base) * prog
            xb = self.X(self.base)
            ax.plot([xb, xb], [yc - self.rh * 1.1, yc + self.rh * 1.1], color=CURVE, lw=px(2.4), alpha=al, zorder=8,
                    path_effects=outlined(4))
            for kk in range(1, int((top_v - self.base) / self.step + 1e-9) + 1):
                xk = self.X(self.base + kk * self.step)
                ax.plot([xk, xk], [yc - self.rh * 1.1, yc + self.rh * 1.1], color=self.acc, lw=px(2.4), alpha=al, zorder=8,
                        path_effects=outlined(4))
                ax.text(xk, yc - self.rh * 1.3, f"{kk}", ha="center", va="bottom", fontsize=fs(17), color=self.acc,
                        weight="semibold", alpha=al, zorder=8, path_effects=outlined(5))
            if prog > 0.97:
                ax.text(self.X(self.x) + 14, yc, look_value(c), ha="left", va="center", fontsize=fs(30), color=self.acc,
                        weight="bold", family="Inter Display", alpha=al, zorder=9, path_effects=outlined(6))


def look_value(c):
    v = c["ruler"]["x"]
    return f"{v:.1f}" if c["category"] in ("nba_ppg", "nba_rpg") else f"{v:.0f}"


class Javelin:
    """a bird's-eye view of the landing sector: each season's longest throw by anyone else is a javelin stuck in the
    grass; rings mark the curve and each step beyond it"""

    def __init__(self, c, pal):
        r = c["ruler"]
        self.c, self.p, self.acc = c, pal, pal["accent"]
        self.year = c["year"]
        self.pts = [(int(s), float(v)) for s, v in r["window"] if int(s) != self.year]
        self.x, self.base, self.step = r["x"], r["base"], r["step"]
        x0, y0, x1, y1 = BOX
        self.w, self.h = x1 - x0, y1 - y0
        self.dmin = math.floor(min(v for _, v in self.pts)) - 1.0
        self.dmax = math.ceil(self.x) + 1.0
        self.ppm = (self.h - 80) / (self.dmax - self.dmin)
        self.cx = self.w * 0.5
        self.apex = self.h - 20 + self.dmin * self.ppm
        rs = np.random.default_rng(1996)
        self.ang = {s: float(rs.uniform(-0.15, 0.15)) for s, _ in self.pts}

    def P(self, d, a):
        r = d * self.ppm
        return self.cx + r * math.sin(a), self.apex - r * math.cos(a)

    def at_x(self, d, x):
        """the point on the arc of distance d at horizontal position x"""
        r = d * self.ppm
        a = math.asin(max(-1.0, min(1.0, (x - self.cx) / r)))
        return self.P(d, a)

    def arc(self, d, a0=-0.27, a1=0.27, n=60):
        aa = np.linspace(a0, a1, n)
        return zip(*[self.P(d, a) for a in aa])

    def field(self):
        im = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        r = np.sqrt((xx - self.cx) ** 2 + (self.apex - yy) ** 2) / self.ppm
        ang = np.arctan2(xx - self.cx, self.apex - yy)
        inside = np.abs(ang) <= math.radians(14.48)
        band = (np.floor(r / 2.0) % 2 == 0)
        g = np.zeros((self.h, self.w, 4), np.float32)
        g[..., 0], g[..., 1], g[..., 2] = 34, 62, 44
        g[band, 1] += 8
        g[..., 3] = np.where(inside, 205, 70)
        return Image.fromarray(g.astype(np.uint8))

    def draw(self, fig, t, al):
        x0, y0, x1, y1 = BOX
        ax = ax_px(fig, x0, y0, x1, y1)
        ax.set_axis_off()
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        if not hasattr(self, "_field"):
            self._field = np.asarray(self.field())
        ax.imshow(self._field, extent=(0, self.w, self.h, 0), alpha=al, zorder=1)
        for d in np.arange(math.ceil(self.dmin), self.dmax + 0.01, 2.0):
            xs, ys = self.arc(d)
            ax.plot(list(xs), list(ys), color="#ffffff", lw=px(1.2), alpha=0.22 * al, zorder=2)
            lx, ly = self.at_x(d, 12)
            ax.text(lx, ly - 10, f"{d:.0f} m", ha="left", va="bottom", fontsize=fs(16), color=INK2, alpha=0.8 * al,
                    path_effects=outlined(4))
        for a in (-math.radians(14.48), math.radians(14.48)):
            p0, p1 = self.P(self.dmin - 2, a), self.P(self.dmax + 2, a)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#ffffff", lw=px(2.2), alpha=0.55 * al, zorder=2)
        # everyone else's javelins land, season by season
        n = len(self.pts)
        for i, (s, d) in enumerate(sorted(self.pts)):
            ts = 1.2 + 8.6 * i / n
            pop = ease_out(clamp((t - ts) / 0.5))
            if pop <= 0:
                continue
            self.javelin(ax, d, self.ang[s], "#c9ccd2", 0.85 * al * pop, 30 * pop, 2.2)
        ax.text(0, 8, "metres · seen from above: the landing sector", ha="left", va="top", fontsize=fs(19), color=INK2, alpha=al)
        ax.text(0, 36, "each grey javelin: the longest throw by anyone else that season, 1986–2011", ha="left", va="top",
                fontsize=fs(17), color=MUTED, alpha=al)
        tc = clamp((t - 11.0) / 3.0)
        if tc > 0:
            xs, ys = self.arc(self.base, -0.25, -0.25 + 0.5 * tc)
            _poly(ax, list(xs), list(ys), CURVE, 3.0, al, z=6, ls=(0, (4, 3)))
            if tc >= 1:
                lx, ly = self.at_x(self.base, self.w * 0.30)
                ax.text(lx, ly + 12, f"the curve: {self.base:.1f} m", ha="center", va="top", fontsize=fs(17), color=CURVE,
                        alpha=al, path_effects=outlined(5), zorder=7)
        if t >= 14.2:
            prog = ease(clamp((t - 14.2) / 4.0))
            d = self.dmin + (self.x - self.dmin) * prog
            a_my = 0.05
            for kk in range(1, int((d - self.base) / self.step + 1e-9) + 1):
                dk = self.base + kk * self.step
                xs, ys = self.arc(dk, -0.25, 0.25)
                ax.plot(list(xs), list(ys), color=self.acc, lw=px(1.8), alpha=0.85 * al, zorder=5)
                lx, ly = self.at_x(dk, self.w - 22)
                ax.text(lx, ly - 4, f"{kk}", ha="center", va="bottom", fontsize=fs(17), color=self.acc, weight="semibold",
                        alpha=al, path_effects=outlined(5), zorder=7)
            self.javelin(ax, d, a_my, self.acc, al, 40, 3.6)
            if prog > 0.97:
                px_, py_ = self.P(self.x, a_my)
                ax.text(px_ + 20, py_ - 8, f"{self.x:.2f} m", ha="left", va="bottom", fontsize=fs(30), color=self.acc,
                        weight="bold", family="Inter Display", alpha=al, path_effects=outlined(6), zorder=9)

    def javelin(self, ax, d, a, color, alpha, length, lw):
        tx, ty = self.P(d, a)
        bx, by = tx - length * math.sin(a) * 0.95 - length * 0.18, ty + length * math.cos(a) * 0.95
        ax.plot([bx + 4, tx + 4], [by + 3, ty + 3], color="#000000", lw=px(lw), alpha=0.35 * alpha, zorder=3,
                solid_capstyle="round")
        ax.plot([bx, tx], [by, ty], color=color, lw=px(lw), alpha=alpha, zorder=4, solid_capstyle="round")
        ax.plot([tx], [ty], marker="o", ms=px(3.2), color="#3a2c1e", alpha=alpha, zorder=4)


class Pit:
    """the long-jump pit from above: each season's longest jump by anyone else is a landing mark in the sand; lines
    mark the curve, the old world record and each step beyond the curve"""

    def __init__(self, c, pal):
        r = c["ruler"]
        self.c, self.p, self.acc = c, pal, pal["accent"]
        self.year = c["year"]
        self.pts = [(int(s), float(v)) for s, v in r["window"] if int(s) != self.year]
        self.x, self.base, self.step = r["x"], r["base"], r["step"]
        x0, y0, x1, y1 = BOX
        self.w, self.h = x1 - x0, y1 - y0
        self.dmin = math.floor((min(v for _, v in self.pts) - 0.06) * 10) / 10
        self.dmax = self.x + 0.10
        self.ppm = (self.w - 40) / (self.dmax - self.dmin)
        self.top, self.bot = 132, self.h - 52
        rs = np.random.default_rng(1968)
        self.yy = {s: float(rs.uniform(self.top + 40, self.bot - 30)) for s, _ in self.pts}
        self.my_y = (self.top + self.bot) / 2 + 20
        pr = c.get("previous_record") or {}
        self.wr = float(pr["value"]) if pr else None

    def X(self, d):
        return 20 + (d - self.dmin) * self.ppm

    def sand(self):
        rs = np.random.default_rng(5)
        a = np.zeros((self.h, self.w, 4), np.float32)
        a[..., 0], a[..., 1], a[..., 2] = 150, 128, 92
        a[..., :3] += rs.normal(0, 7, (self.h, self.w, 1))
        a[..., 3] = 0
        a[self.top:self.bot, :, 3] = 150
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    def mark(self, ax, d, y, color, alpha, size, z=4):
        xr = self.X(d)
        ax.add_patch(__import__("matplotlib").patches.Ellipse((xr - size * 0.7, y), size * 1.9, size * 1.3, color="#3b2f22",
                                                              alpha=0.55 * alpha, zorder=z, lw=0))
        ax.add_patch(__import__("matplotlib").patches.Ellipse((xr - size * 0.55, y), size * 1.2, size * 0.8, color=color,
                                                              alpha=alpha, zorder=z + 0.1, lw=0))

    def draw(self, fig, t, al):
        x0, y0, x1, y1 = BOX
        ax = ax_px(fig, x0, y0, x1, y1)
        ax.set_axis_off()
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        if not hasattr(self, "_sand"):
            self._sand = np.asarray(self.sand())
        ax.imshow(self._sand, extent=(0, self.w, self.h, 0), alpha=al, zorder=1)
        # a measuring tape along the top edge
        d = math.ceil(self.dmin * 20) / 20
        while d <= self.dmax + 1e-9:
            big = abs(d * 10 - round(d * 10)) < 1e-6
            ax.plot([self.X(d)] * 2, [self.top - (16 if big else 8), self.top], color="#d8d4ca", lw=px(1.4), alpha=al, zorder=3)
            if big:
                ax.text(self.X(d), self.top - 22, f"{d:.1f}", ha="center", va="bottom", fontsize=fs(16), color=MUTED, alpha=al)
            d += 0.05
        ax.text(0, 8, "metres from the take-off board · the pit seen from above", ha="left", va="top", fontsize=fs(19),
                color=INK2, alpha=al)
        ax.text(0, 36, "each grey mark: the longest jump by anyone else that season, 1960–1983", ha="left", va="top",
                fontsize=fs(17), color=MUTED, alpha=al)
        ax.text(6, self.bot + 30, "← take-off board, 8 m back", ha="left", va="center", fontsize=fs(16), color=MUTED, alpha=al)
        n = len(self.pts)
        for i, (s, dd) in enumerate(sorted(self.pts)):
            pop = ease_out(clamp((t - (1.2 + 8.6 * i / n)) / 0.5))
            if pop > 0:
                self.mark(ax, dd, self.yy[s], "#d9d6cf", 0.9 * al * pop, 13 * (0.6 + 0.4 * pop))
        if self.wr:
            a = al * fade(t, 6.0, 99, 0.8, 0)
            ax.plot([self.X(self.wr)] * 2, [self.top, self.bot], color="#ffffff", lw=px(1.6), alpha=0.6 * a, ls=(0, (2, 3)), zorder=5)
            ax.text(self.X(self.wr) + 6, self.bot - 10, f"world record\nbefore: {self.wr:.2f} m", ha="left", va="bottom",
                    fontsize=fs(16), color=INK2, alpha=a, path_effects=outlined(4), zorder=6)
        tc = clamp((t - 11.0) / 3.0)
        if tc > 0:
            yb = self.top + (self.bot - self.top) * tc
            _poly(ax, [self.X(self.base)] * 2, [self.top, yb], CURVE, 3.0, al, z=6, ls=(0, (4, 3)))
            if tc >= 1:
                ax.text(self.X(self.base) - 8, self.bot - 10, f"the curve\n{self.base:.2f} m", ha="right", va="bottom",
                        fontsize=fs(17), color=CURVE, alpha=al, path_effects=outlined(5), zorder=6)
        if t >= 14.2:
            prog = ease(clamp((t - 14.2) / 4.0))
            dx = self.dmin + (self.x - self.dmin) * prog
            for kk in range(1, int((dx - self.base) / self.step + 1e-9) + 1):
                xk = self.X(self.base + kk * self.step)
                ax.plot([xk, xk], [self.top, self.bot], color=self.acc, lw=px(1.8), alpha=0.85 * al, zorder=5)
                ax.text(xk, self.top + 18, f"{kk}", ha="center", va="top", fontsize=fs(17), color=self.acc, weight="semibold",
                        alpha=al, zorder=7, path_effects=outlined(5))
            if prog < 0.999:                                      # in flight: a shadow crossing the sand
                hgt = math.sin(math.pi * prog)
                ax.plot([self.X(dx)], [self.my_y + 18 * hgt], marker="o", ms=px(10 - 4 * hgt), color="#000000",
                        alpha=0.35 * al, zorder=7)
                ax.plot([self.X(dx)], [self.my_y - 60 * hgt], marker="o", ms=px(14), color=self.acc, alpha=al, zorder=8,
                        markeredgecolor="#111111")
            else:
                splash = clamp((t - 18.2) / 0.8)
                if splash < 1:
                    ax.plot([self.X(self.x) - 8], [self.my_y], marker="o", ms=px(18 + 50 * splash), color="none",
                            markeredgecolor="#efe6d2", markeredgewidth=px(2), alpha=(1 - splash) * al, zorder=7)
                self.mark(ax, self.x, self.my_y, self.acc, al, 18, z=8)
                ax.text(self.X(self.x) - 4, self.my_y - 34, f"{self.x:.2f} m", ha="right", va="bottom", fontsize=fs(30),
                        color=self.acc, weight="bold", family="Inter Display", alpha=al, path_effects=outlined(6), zorder=9)


def make_scene(c, explain=False):
    pal = PALETTE[c["rank"]]
    s = SCENE[c["rank"]]
    if s["kind"] == "javelin":
        return Javelin(c, pal)
    if s["kind"] == "pit":
        return Pit(c, pal)
    if s["kind"] == "hstack":
        return HStack(c, pal, s)
    return Stack(c, pal, s, explain=explain)


@lru_cache(maxsize=16)
def face_disc(key, rank, d=150):
    """a round, team-toned head-and-shoulders thumbnail (from the 4:5 portrait crop)"""
    p = PALETTE[rank]
    im = Image.open(os.path.join(HERE, "photos", key + ".png")).convert("RGB")
    im = im.crop((0, 0, im.width, im.width)).resize((d * SS, d * SS), Image.LANCZOS)
    col = rgb(p["bg"])
    tone = duotone(im, mix(BG, col, 0.5), mix(col, (255, 255, 255), 0.25), mix(rgb(p["light"]), (255, 255, 255), 0.4))
    out = Image.fromarray(np.clip(tone, 0, 255).astype(np.uint8))
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).ellipse((SS * 2, SS * 2, d * SS - SS * 2, d * SS - SS * 2), fill=255)
    out.putalpha(mask)
    ring = Image.new("RGBA", out.size, (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((SS * 2, SS * 2, d * SS - SS * 2, d * SS - SS * 2), outline=rgb(p["accent"]) + (255,),
                                 width=SS * 4)
    out.alpha_composite(ring)
    return np.asarray(out.resize((d, d), Image.LANCZOS))
