"""Building blocks for scenes 03-11.

A PagedScene is a run of paper pages that slide in across the desk. Each page holds elements that appear on cue:
typed text, ledger bars, rubber stamps, telegrams, drawings dropped onto the desk, lines that ink themselves.
Every element is placed by phrase (scene.cue("...")), so the whole scene follows the voice-over.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine import FPS, H, W, Scene, clamp, ease_out, lerp, ramp, sf, shake, smooth, stamp_in, txt

TR = 0.7  # page transition, seconds
F = sf.font


# ---------------------------------------------------------------- elements
class El:
    """Something drawn on a page from t0 (fading in) until t1 (fading out), if given."""
    fade_in, fade_out = 0.35, 0.3

    def __init__(self, t0, t1=None):
        self.t0, self.t1 = t0, t1

    def alpha(self, t):
        a = ramp(t, self.t0, self.fade_in) if self.fade_in else 1.0
        if self.t1 is not None:
            a *= 1 - ramp(t, self.t1, self.fade_out)
        return a

    def live(self, t):
        return t >= self.t0 and (self.t1 is None or t < self.t1 + self.fade_out)

    def draw(self, img, t):
        raise NotImplementedError


class Txt(El):
    def __init__(self, t0, xy, s, fnt, t1=None, **kw):
        super().__init__(t0, t1)
        self.xy, self.s, self.fnt, self.kw = xy, s, fnt, kw

    def draw(self, img, t):
        txt(img, self.xy, self.s, self.fnt, alpha=int(255 * self.alpha(t)), **self.kw)


class Typed(El):
    """Typewriter text, one keystroke at a time."""
    fade_in = 0

    def __init__(self, t0, xy, s, fnt, cps=32, t1=None, **kw):
        super().__init__(t0, t1)
        self.xy, self.s, self.fnt, self.cps, self.kw = xy, s, fnt, cps, kw

    def draw(self, img, t):
        n = int(clamp((t - self.t0) * self.cps, 0, len(self.s)))
        if n:
            txt(img, self.xy, self.s[:n], self.fnt, alpha=int(255 * self.alpha(t)), **self.kw)


class Stamp(El):
    fade_in = 0

    def __init__(self, t0, center, lines, angle=-8, size=64, seed=5, t1=None):
        super().__init__(t0, t1)
        self.center, self.lines, self.angle, self.size, self.seed = center, lines, angle, size, seed

    def draw(self, img, t):
        stamp_in(img, t, self.t0, self.center, self.lines, angle=self.angle, size=self.size, seed=self.seed)


def shadowed(obj, offset=(10, 14), blur=12, alpha=90):
    """obj with its drop shadow baked in; returns (layer, (dx, dy)) where obj's corner sits at -dx, -dy."""
    pad = blur * 3 + max(abs(offset[0]), abs(offset[1]))
    big = sf.ink_layer((obj.size[0] + 2 * pad, obj.size[1] + 2 * pad))
    a = obj.split()[-1]
    sh = Image.new('RGBA', obj.size, (40, 28, 15, 0)); sh.putalpha(a.point(lambda v: v * alpha // 255))
    big.alpha_composite(sh, (pad + offset[0], pad + offset[1]))
    big = big.filter(ImageFilter.GaussianBlur(blur))
    big.alpha_composite(obj, (pad, pad))
    return big, (pad, pad)


def with_alpha(layer, a):
    if a >= 0.999:
        return layer
    out = layer.copy()
    out.putalpha(Image.fromarray((np.asarray(layer.split()[-1], np.float32) * a).astype(np.uint8)))
    return out


def paste(img, layer, xy):
    x, y = int(round(xy[0])), int(round(xy[1]))
    if x >= img.size[0] or y >= img.size[1] or x + layer.size[0] <= 0 or y + layer.size[1] <= 0:
        return
    img.alpha_composite(layer, (max(0, x), max(0, y)), (max(0, -x), max(0, -y)))


class Obj(El):
    """A paper object put down on the desk: drops in, or slides in from an edge."""

    def __init__(self, t0, obj, xy, how='drop', dur=0.45, t1=None, angle=0, shadow=(10, 14, 12, 90)):
        super().__init__(t0, t1)
        self.fade_in = dur if how in ('drop', 'fade') else 0.05
        if angle:
            obj = obj.rotate(angle, expand=True, resample=Image.BICUBIC)
        self.layer, (self.px, self.py) = shadowed(obj, shadow[:2], shadow[2], shadow[3]) if shadow else (obj, (0, 0))
        self.xy, self.how, self.dur, self.size = xy, how, dur, obj.size

    def draw(self, img, t):
        e = ease_out((t - self.t0) / self.dur)
        x, y = self.xy[0] - self.px, self.xy[1] - self.py
        lay = self.layer
        if e < 1:
            if self.how == 'drop':
                s = 1.07 - 0.07 * e
                w, h = lay.size
                lay = lay.resize((int(w * s), int(h * s)), Image.BILINEAR)
                x -= (lay.size[0] - w) / 2; y -= (lay.size[1] - h) / 2 + 14 * (1 - e)
            elif self.how == 'left':
                x -= (self.xy[0] + self.size[0] + 60) * (1 - e)
            elif self.how == 'right':
                x += (W - self.xy[0] + 60) * (1 - e)
            elif self.how == 'up':
                y += (H - self.xy[1] + 60) * (1 - e)
            elif self.how == 'down':
                y -= (self.xy[1] + self.size[1] + 60) * (1 - e)
        paste(img, with_alpha(lay, self.alpha(t)), (x, y))


class Bar(El):
    """A hatched ledger bar that grows to its value, then its figure is typed at the end."""
    fade_in = 0

    def __init__(self, t0, xy, length, h=38, label=None, fnt=None, color=sf.INK, dur=0.6, wash=None, label_fill=None, t1=None):
        super().__init__(t0, t1)
        self.xy, self.length, self.dur, self.label = xy, max(4, int(length)), dur, label
        self.fnt = fnt or F('type', 40)
        self.label_fill = label_fill or color
        bar = sf.ink_layer((self.length + 4, h + 4))
        if wash:
            sf.wash(bar, (2, 2, self.length + 2, h + 2), wash, alpha=170, seed=int(length))
        sf.hatch(bar, (2, 2, self.length + 2, h + 2), spacing=6, angle=60, color=color, alpha=200 if not wash else 90)
        ImageDraw.Draw(bar).rectangle([2, 2, self.length + 1, h + 1], outline=color + (230,), width=2)
        self.bar, self.h = bar, h

    def draw(self, img, t):
        e = ease_out((t - self.t0) / self.dur)
        w = int(self.bar.size[0] * e)
        if w > 0:
            paste(img, with_alpha(self.bar.crop((0, 0, w, self.bar.size[1])), self.alpha(t) if self.t1 else 1), self.xy)
        if self.label and t >= self.t0 + self.dur * 0.8:
            n = int(clamp((t - self.t0 - self.dur * 0.8) * 30, 0, len(self.label)))
            txt(img, (self.xy[0] + self.length + 18, self.xy[1] + self.h / 2 + 2), self.label[:n], self.fnt, fill=self.label_fill,
                anchor='lm', alpha=int(255 * (self.alpha(t) if self.t1 else 1)))


def polyline_part(pts, frac):
    """The first `frac` of a polyline, by length."""
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    total, goal, out = sum(seg), sum(seg) * clamp(frac), [pts[0]]
    for (a, b), L in zip(zip(pts, pts[1:]), seg):
        if goal >= L:
            out.append(b); goal -= L
        else:
            k = goal / L if L else 0
            out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)); break
    return out


def wobble_pts(pts, amp=1.0, seed=0, step=10):
    r = np.random.default_rng(seed)
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
        walk = np.cumsum(r.normal(0, amp, n)); walk -= np.linspace(walk[0], walk[-1], n)
        nx, ny = -(y1 - y0), (x1 - x0); L = math.hypot(nx, ny) or 1
        for i in range(n - (1 if (x1, y1) != pts[-1] else 0)):
            k = i / (n - 1)
            out.append((x0 + (x1 - x0) * k + nx / L * walk[i], y0 + (y1 - y0) * k + ny / L * walk[i]))
    return out


class Line(El):
    """A line that inks itself along its path: underlines, strike-throughs, arrows, ringed answers."""
    fade_in = 0

    def __init__(self, t0, pts, fill=sf.STAMP, width=5, dur=0.4, amp=1.0, seed=0, t1=None):
        super().__init__(t0, t1)
        self.pts, self.fill, self.width, self.dur = wobble_pts(pts, amp, seed) if amp else list(pts), fill, width, dur

    def draw(self, img, t):
        part = polyline_part(self.pts, ease_out((t - self.t0) / self.dur))
        if len(part) > 1:
            a = int(230 * (self.alpha(t) if self.t1 else 1))
            ImageDraw.Draw(img).line(part, fill=tuple(self.fill) + (a,), width=self.width, joint='curve')


def ring(cx, cy, rx, ry, turns=1.12, start=-2.2, n=90):
    """Points of a hand-drawn ring, which overshoots its start a little."""
    return [(cx + rx * math.cos(start + 2 * math.pi * turns * k / n) * (1 + 0.04 * k / n),
             cy + ry * math.sin(start + 2 * math.pi * turns * k / n) * (1 + 0.04 * k / n)) for k in range(n + 1)]


class Counter(El):
    def __init__(self, t0, t_end, xy, v0, v1, fmt, fnt, t1=None, **kw):
        super().__init__(t0, t1)
        self.t_end, self.xy, self.v0, self.v1, self.fmt, self.fnt, self.kw = t_end, xy, v0, v1, fmt, fnt, kw

    def draw(self, img, t):
        x = clamp((t - self.t0) / max(0.01, self.t_end - self.t0))
        v = lerp(self.v0, self.v1, 1 - (1 - x) ** 2)
        txt(img, self.xy, self.fmt(v), self.fnt, alpha=int(255 * self.alpha(t)), **self.kw)


class Fn(El):
    """Anything else: fn(img, t, alpha)."""
    fade_in = 0

    def __init__(self, t0, fn, t1=None):
        super().__init__(t0, t1)
        self.fn = fn

    def draw(self, img, t):
        self.fn(img, t, self.alpha(t) if self.t1 else 1.0)


# ---------------------------------------------------------------- pages and the scene
class Page:
    def __init__(self, bg, t0, enter='slide', zoom=0.025, focus=(0.5, 0.5)):
        self.bg, self.t0, self.enter, self.zoom, self.focus = bg.convert('RGBA'), t0, enter, zoom, focus
        self.els = []

    def add(self, *els):
        self.els.extend(els)
        return self


_EDGE = None


def edge_shadow():
    global _EDGE
    if _EDGE is None:
        a = np.linspace(0, 1, 70) ** 2 * 120
        g = np.tile(a.astype(np.uint8), (H, 1))
        _EDGE = Image.new('RGBA', (70, H), (30, 20, 10, 0)); _EDGE.putalpha(Image.fromarray(g))
    return _EDGE


class PagedScene(Scene):
    def __init__(self, name):
        super().__init__(name)
        self.pages = []

    def page(self, bg, t0=0.0, **kw):
        p = Page(bg, t0, **kw)
        self.pages.append(p)
        return p

    def draw_page(self, i, t):
        p = self.pages[i]
        img = p.bg.copy()
        for e in p.els:
            if e.live(t):
                e.draw(img, t)
        end = self.pages[i + 1].t0 + TR if i + 1 < len(self.pages) else self.dur
        if p.zoom:
            z = 1 + p.zoom * smooth(clamp((t - p.t0) / max(0.1, end - p.t0)))
            w, h = W / z, H / z
            fx, fy = p.focus
            x0, y0 = (W - w) * fx, (H - h) * fy
            img = img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + w, y0 + h))
        for e in p.els:  # a stamp landing jolts the page
            if isinstance(e, Stamp) and e.t0 + 0.12 <= t < e.t0 + 0.4:
                img = shake(img, t, e.t0 + 0.12)
        return img

    def frame(self, t):
        i = max(k for k, p in enumerate(self.pages) if p.t0 <= t or k == 0)
        img = self.draw_page(i, t)
        p = self.pages[i]
        if i > 0 and t < p.t0 + TR and p.enter != 'cut':
            prev = self.draw_page(i - 1, t)
            x = smooth((t - p.t0) / TR)
            if p.enter == 'fade':
                return Image.blend(prev, img, x)
            off = int(W * (1 - x))
            prev = prev.transform(prev.size, Image.AFFINE, (1, 0, (W - off) * 0.25, 0, 1, 0), resample=Image.BILINEAR)
            prev.alpha_composite(edge_shadow(), (off - 70, 0)) if off > 0 else None
            prev.paste(img.crop((0, 0, W - off, H)), (off, 0))
            return prev
        return img


# ---------------------------------------------------------------- paper goods
def ledger_paper(seed=3, rules=(), margin=152, top=160, cols=()):
    img = sf.paper(base=(240, 233, 212), seed=seed)
    layer = sf.ink_layer(img.size); d = ImageDraw.Draw(layer)
    for y in rules:
        d.line([(80, y), (1840, y)], fill=(120, 150, 170, 110), width=1)
    for x in cols:
        d.line([(x, top), (x, 1040)], fill=(60, 120, 90, 120), width=2)
    if margin:
        d.line([(margin, top), (margin, 1040)], fill=(170, 60, 50, 140), width=2)
    img.alpha_composite(layer)
    return img


def fig_page(num, title, seed=40, sub=None):
    """A patent-sheet page: "Fig. 2." and a caption, as each decision opens."""
    img = sf.paper(seed=seed)
    txt(img, (W / 2, 120), 'Fig. %s.' % num, F('serifb', 64), anchor='mm')
    txt(img, (W / 2, 190), title, F('sc', 36), fill=sf.SEPIA, anchor='mm')
    if sub:
        txt(img, (W / 2, 240), sub, F('serif', 30), anchor='mm')
    d = ImageDraw.Draw(img)
    sf.wobble_line(d, [(W / 2 - 260, 222 if not sub else 272), (W / 2 + 260, 222 if not sub else 272)], width=2, seed=seed)
    return img


def decision_card(n, title, seed=50):
    """Section opener: DECISION No. n, in the manner of a 1930s certificate."""
    img = sf.paper(base=(236, 229, 210), seed=seed)
    sf.guilloche_frame(img, inset=30, band=56)
    txt(img, (W / 2, 330), 'DECISION No. %d' % n if isinstance(n, int) else n, F('scb', 54), fill=sf.NOTE_GREEN, anchor='mm')
    txt(img, (W / 2, 500), title, F('deco', 120), anchor='mm', align='center', spacing=16)
    return img


def card(w, h, base=(244, 238, 222), seed=0, border=True):
    c = sf.paper(w, h, base=base, seed=seed)
    if border:
        ImageDraw.Draw(c).rectangle([0, 0, w - 1, h - 1], outline=sf.INK + (70,), width=1)
    return c


def note_card(lines, fnt=None, w=760, seed=0, pad=26, gap=12, fill=sf.INK, base=(244, 238, 222)):
    """An index card with lines of text on it."""
    fnt = fnt or F('type', 32)
    d = ImageDraw.Draw(sf.ink_layer((10, 10)))
    hs = [d.textbbox((0, 0), l or 'x', font=fnt)[3] for l in lines]
    h = pad * 2 + sum(hs) + gap * (len(lines) - 1)
    c = card(w, h, base=base, seed=seed)
    y = pad
    for l, lh in zip(lines, hs):
        txt(c, (pad, y), l, fnt, fill=fill); y += lh + gap
    return c


class Telegram(El):
    """A telegram dropped on the desk; its strips type themselves word by word."""
    fade_in = 0.3

    def __init__(self, t0, xy, lines, w=880, angle=-2, seed=9, head='TELEGRAM', place='ATLANTIC CITY N.J.', wps=5.0,
                 t1=None, how='drop', strip_font=None, reply=None):
        super().__init__(t0, t1)
        self.lines, self.angle, self.wps, self.how = lines, angle, wps, how
        self.sf_ = strip_font or F('type', 32)
        self.h = 150 + 58 * len(lines) + 30
        self.w = w
        base = sf.paper(w, self.h, base=(236, 226, 196), seed=seed)
        d = ImageDraw.Draw(base)
        txt(base, (36, 26), head, F('fat', 50))
        txt(base, (w - 36, 46), place, F('type', 22), fill=sf.SEPIA, anchor='ra')
        d.line([(28, 104), (w - 28, 104)], fill=sf.INK + (200,), width=2)
        d.line([(28, 110), (w - 28, 110)], fill=sf.INK + (120,), width=1)
        self.base = base
        self.xy = xy
        self.words = [l.split() for l in lines]
        rot = base.rotate(angle, expand=True, resample=Image.BICUBIC)
        self.shadow, (self.px, self.py) = shadowed(rot)
        self.shadow = self.shadow.copy()

    def n_words(self, t):
        return int(max(0.0, t - self.t0 - 0.35) * self.wps)

    def draw(self, img, t):
        c = self.base.copy()
        budget = self.n_words(t)
        for k, ws in enumerate(self.words):
            if budget <= 0:
                break
            shown = ' '.join(ws[:budget]); budget -= len(ws)
            sw = int(ImageDraw.Draw(c).textlength(shown, font=self.sf_)) + 26
            strip = Image.new('RGBA', (sw, 48), (250, 246, 232, 255))
            c.alpha_composite(strip, (30, 128 + k * 58))
            txt(c, (42, 134 + k * 58), shown, self.sf_)
        rot = c.rotate(self.angle, expand=True, resample=Image.BICUBIC)
        lay = self.shadow.copy(); lay.alpha_composite(rot, (self.px, self.py))
        e = ease_out((t - self.t0) / 0.45)
        x, y = self.xy[0] - self.px, self.xy[1] - self.py
        if self.how == 'right':
            x += (W - self.xy[0] + 60) * (1 - e)
        elif self.how == 'left':
            x -= (self.xy[0] + self.w + 60) * (1 - e)
        else:
            y -= 16 * (1 - e)
        paste(img, with_alpha(lay, self.alpha(t)), (x, y))

    def done_at(self):
        return self.t0 + 0.35 + sum(len(w) for w in self.words) / self.wps


def grid100(n, size=34, gap=8, color=sf.INK, wash=None, cols=10, seed=0):
    """100 small boxes, the first n filled: 'n games in a hundred'."""
    w = cols * size + (cols - 1) * gap
    rows = math.ceil(100 / cols)
    img = sf.ink_layer((w + 6, rows * size + (rows - 1) * gap + 6))
    d = ImageDraw.Draw(img)
    for k in range(100):
        x, y = 3 + (k % cols) * (size + gap), 3 + (k // cols) * (size + gap)
        if k < n:
            if wash:
                sf.wash(img, (x, y, x + size, y + size), wash, alpha=200, seed=seed + k)
            else:
                sf.hatch(img, (x, y, x + size, y + size), spacing=5, angle=60, color=color, alpha=220)
        d.rectangle([x, y, x + size, y + size], outline=color + (200,), width=2)
    return img


class Grid100(El):
    """The hundred boxes fill one by one."""
    fade_in = 0

    def __init__(self, t0, xy, n, dur=1.0, t1=None, **kw):
        super().__init__(t0, t1)
        self.xy, self.n, self.dur = xy, n, dur
        self.frames = {}
        self.kw = kw
        self.empty = grid100(0, **kw)
        self.full = grid100(n, **kw)

    def draw(self, img, t):
        k = int(round(self.n * clamp((t - self.t0) / self.dur)))
        if k not in self.frames:
            self.frames = {k: grid100(k, **self.kw)}
        a = self.alpha(t) if self.t1 else min(1.0, (t - self.t0) / 0.2 + 0.0001)
        paste(img, with_alpha(self.frames[k], a), self.xy)


def house_elevation(h=160, seed=0, n_floors=1):
    """An architect's elevation of a small clapboard house, in ink."""
    w = int(h * 0.95)
    img = sf.ink_layer((w + 10, h + 10))
    d = ImageDraw.Draw(img)
    base, eave, ridge = h, h * 0.45, h * 0.05
    sf.hatch(img, (5 + w * 0.08, eave, 5 + w * 0.92, base), spacing=9, angle=0, alpha=70)  # clapboard
    walls = [(5 + w * 0.08, base), (5 + w * 0.08, eave), (5 + w * 0.92, eave), (5 + w * 0.92, base), (5 + w * 0.08, base)]
    roof = [(5, eave + 4), (5 + w / 2, ridge), (5 + w, eave + 4), (5, eave + 4)]
    d.polygon(roof, fill=(239, 228, 204, 255))
    sf.hatch(img, (5 + w * 0.2, ridge + h * 0.1, 5 + w * 0.8, eave), spacing=6, angle=45, alpha=40)
    sf.wobble_line(d, roof, width=3, seed=seed)
    sf.wobble_line(d, walls, width=3, seed=seed + 1)
    dx = 5 + w * 0.42
    d.rectangle([dx, base - h * 0.32, dx + w * 0.16, base], fill=(239, 228, 204, 255), outline=sf.INK + (230,), width=2)
    for wx in (5 + w * 0.16, 5 + w * 0.66):
        d.rectangle([wx, eave + h * 0.12, wx + w * 0.18, eave + h * 0.3], fill=(239, 228, 204, 255), outline=sf.INK + (230,), width=2)
        d.line([(wx + w * 0.09, eave + h * 0.12), (wx + w * 0.09, eave + h * 0.3)], fill=sf.INK + (200,), width=1)
    d.rectangle([5 + w * 0.68, ridge + h * 0.12, 5 + w * 0.78, eave - h * 0.1], fill=(239, 228, 204, 255), outline=sf.INK + (230,), width=2)  # chimney
    return img


def hotel_elevation(h=330, seed=0):
    """A grand 1930s seafront hotel facade, as an elevation drawing."""
    w = int(h * 1.05)
    img = sf.ink_layer((w + 10, h + 10))
    d = ImageDraw.Draw(img)
    x0, x1, top, base = 5 + w * 0.06, 5 + w * 0.94, h * 0.2, h
    paper = (239, 228, 204, 255)
    d.rectangle([x0, top, x1, base], fill=paper)
    sf.hatch(img, (x0, top, x1, base), spacing=11, angle=0, alpha=45)
    # towers and the central block
    d.rectangle([5 + w * 0.36, h * 0.07, 5 + w * 0.64, top], fill=paper)
    d.polygon([(5 + w * 0.36, h * 0.07), (5 + w * 0.5, 0), (5 + w * 0.64, h * 0.07)], fill=paper)
    sf.wobble_line(d, [(x0, base), (x0, top), (5 + w * 0.36, top), (5 + w * 0.36, h * 0.07), (5 + w * 0.5, 2),
                       (5 + w * 0.64, h * 0.07), (5 + w * 0.64, top), (x1, top), (x1, base), (x0, base)], width=3, seed=seed)
    # flag
    d.line([(5 + w * 0.5, 2), (5 + w * 0.5, -h * 0.0 + 2)], fill=sf.INK + (230,), width=2)
    # windows: rows of arched openings
    rows, cols = 6, 9
    for r in range(rows):
        for c in range(cols):
            wx = x0 + (x1 - x0) * (c + 0.25) / cols
            wy = top + (base - top) * (r + 0.18) / (rows + 0.6)
            ww, wh = (x1 - x0) / cols * 0.5, (base - top) / (rows + 0.6) * 0.55
            if r == rows - 1 and 3 <= c <= 5:
                continue
            d.rectangle([wx, wy, wx + ww, wy + wh], outline=sf.INK + (210,), width=2)
            d.line([(wx, wy + wh * 0.5), (wx + ww, wy + wh * 0.5)], fill=sf.INK + (120,), width=1)
    # entrance and awning
    ex0, ex1 = x0 + (x1 - x0) * 0.38, x0 + (x1 - x0) * 0.62
    d.rectangle([ex0, base - h * 0.15, ex1, base], fill=paper, outline=sf.INK + (230,), width=3)
    d.polygon([(ex0 - 14, base - h * 0.15), (ex1 + 14, base - h * 0.15), (ex1 + 4, base - h * 0.2), (ex0 - 4, base - h * 0.2)],
              fill=sf.STAMP + (200,))
    txt(img, ((x0 + x1) / 2, top + 18), 'HOTEL', F('sc', int(h * 0.06)), anchor='mt')
    return img


def deed(name, price, group, seed=0, w=640, h=400):
    """A period deed of sale (our own layout, not a game card)."""
    c = sf.paper(w, h, base=(238, 230, 206), seed=seed)
    d = ImageDraw.Draw(c)
    d.rectangle([12, 12, w - 12, h - 12], outline=sf.INK + (200,), width=2)
    d.rectangle([20, 20, w - 20, h - 20], outline=sf.INK + (120,), width=1)
    sf.wash(c, (20, 20, 44, h - 20), sf.GROUP[group], alpha=190, seed=seed)
    txt(c, (w / 2 + 12, 58), 'Deed of Sale', F('script', 54), anchor='mm')
    txt(c, (w / 2 + 12, 112), 'ATLANTIC CITY, NEW JERSEY', F('sc', 20), fill=sf.SEPIA, anchor='mm')
    txt(c, (w / 2 + 12, 182), name, F('fat', 44), anchor='mm')
    txt(c, (w / 2 + 12, 248), 'for the consideration of', F('serif', 24), fill=sf.SEPIA, anchor='mm')
    txt(c, (w / 2 + 12, 296), '$%d' % price, F('type', 48), anchor='mm')
    d.line([(90, h - 56), (w / 2 - 20, h - 56)], fill=sf.INK + (160,), width=1)
    d.line([(w / 2 + 50, h - 56), (w - 60, h - 56)], fill=sf.INK + (160,), width=1)
    txt(c, (90, h - 50), 'buyer', F('serif', 18), fill=sf.SEPIA)
    txt(c, (w / 2 + 50, h - 50), 'the bank', F('serif', 18), fill=sf.SEPIA)
    return c


# ---------------------------------------------------------------- shared pieces for the decision scenes
PATENT_SRC = (1805, 2809)  # the scanned patent sheet, pixels


def patent_world():
    """The patent sheet on the desk, as in the cold open; returns (world, sheet-pixel -> world-pixel)."""
    WW, WH = 3840, 2160
    world = sf.paper(WW, WH, seed=21)
    sheet_h = 2040
    k = sheet_h / PATENT_SRC[1]
    pd = sf.patent_drawing(sheet_h)
    ox, oy = (WW - pd.size[0]) // 2, 60
    sheet = sf.paper(pd.size[0] + 60, pd.size[1] + 40, base=(242, 234, 214), seed=22)
    sheet.alpha_composite(pd, (30, 10))
    sf.paste_shadowed(world, sheet, (ox - 30, oy - 10), offset=(14, 18), blur=18, alpha=110)
    return world, (lambda x, y: (ox + x * k, oy + y * k))


def tick(page, t0, x, y, size=30, seed=0):
    """A red check mark, inked."""
    page.add(Line(t0, [(x, y + size * 0.5), (x + size * 0.38, y + size), (x + size, y)], width=6, dur=0.25, amp=0.6, seed=seed))


def ledger_row(page, t0, y, label, value, scale, x_label=180, x_bar=820, fnt=None, fmt='%.1f', bar_t=None, wash=None,
               fill=sf.INK, h=40, extra=None):
    """A typed label, then a hatched bar to `value`, with its figure."""
    fnt = fnt or F('type', 36)
    page.add(Typed(t0, (x_label, y), label, fnt, fill=fill))
    bt = t0 + 0.25 if bar_t is None else bar_t
    page.add(Bar(bt, (x_bar, y - 2), value * scale, h=h, label=fmt % value if isinstance(fmt, str) else fmt(value),
                 fnt=F('type', 38), wash=wash, color=fill if fill != sf.INK else sf.INK))
    return bt


def padlock(size=220, seed=0):
    img = sf.ink_layer((size, int(size * 1.25)))
    d = ImageDraw.Draw(img)
    w = size
    d.arc([w * 0.2, 4, w * 0.8, w * 0.62], 180, 360, fill=sf.INK + (235,), width=int(w * 0.07))
    d.line([(w * 0.235, w * 0.33), (w * 0.235, w * 0.5)], fill=sf.INK + (235,), width=int(w * 0.07))
    d.line([(w * 0.765, w * 0.33), (w * 0.765, w * 0.5)], fill=sf.INK + (235,), width=int(w * 0.07))
    d.rounded_rectangle([4, w * 0.48, w - 4, w * 1.2], radius=int(w * 0.06), fill=(196, 160, 92, 255), outline=sf.INK + (235,), width=4)
    sf.hatch(img, (8, w * 0.5, w - 8, w * 1.18), spacing=7, angle=60, alpha=60)
    d.ellipse([w * 0.44, w * 0.7, w * 0.56, w * 0.82], fill=sf.INK + (235,))
    d.polygon([(w * 0.47, w * 0.78), (w * 0.53, w * 0.78), (w * 0.55, w * 1.0), (w * 0.45, w * 1.0)], fill=sf.INK + (235,))
    return img


def pawn_layer(h=150, color=sf.BRASS):
    img = sf.ink_layer((int(h * 1.2), int(h * 1.25)))
    sf.pawn(img, img.size[0] / 2, h * 1.08, h=h, color=color)
    return img


def board_layer(S=900, highlight=None, X=10, Y=10):
    """Our patent-style board on a transparent layer, for pasting anywhere."""
    lay = sf.ink_layer((S + 2 * X, S + 2 * Y))
    sf.draw_board(lay, X, Y, S, highlight=highlight)
    return lay


def sheet(scene, path, times=None, cols=4, scale=0.25):
    """A contact sheet of frames: page starts plus a beat into each page, for checking layout."""
    if times is None:
        times = []
        ps = getattr(scene, 'pages', [])
        for k, p in enumerate(ps):
            end = ps[k + 1].t0 if k + 1 < len(ps) else scene.dur
            times += [p.t0 + TR + 0.2, (p.t0 + end) / 2, end - 0.1]
    fw, fh = int(W * scale), int(H * scale)
    rows = math.ceil(len(times) / cols)
    out = Image.new('RGB', (cols * fw, rows * (fh + 22)), (30, 30, 30))
    d = ImageDraw.Draw(out)
    for k, t in enumerate(times):
        img = scene.frame(min(t, scene.dur - 0.05)).convert('RGB').resize((fw, fh), Image.BILINEAR)
        x, y = (k % cols) * fw, (k // cols) * (fh + 22)
        out.paste(img, (x, y + 22))
        d.text((x + 6, y + 4), '%.1fs' % t, fill=(230, 230, 230))
    out.save(path)
    return path


def main(scenes):
    """python3 <module>.py <scene> [--captions] [--sheet] [--frame t ...]"""
    import os
    import sys
    from engine import OUT
    name = sys.argv[1]
    sc = scenes[name]()
    os.makedirs(OUT, exist_ok=True)
    if '--sheet' in sys.argv:
        print(sheet(sc, os.path.join(OUT, name + '_sheet.png')))
    elif '--frame' in sys.argv:
        for t in sys.argv[sys.argv.index('--frame') + 1:]:
            p = os.path.join(OUT, '%s_%05.1f.png' % (name, float(t)))
            sc.frame(float(t)).convert('RGB').save(p); print(p)
    else:
        print(sc.render(captions='--captions' in sys.argv))
