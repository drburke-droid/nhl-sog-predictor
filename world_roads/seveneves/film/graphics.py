"""Explainer graphics for the impact-ladder film (drawn with PIL, numbers from the simulation).

  fate      where one impact's ejecta goes (crater -> escapes the Moon -> Earth / Sun orbit / Moon)
  chely     what a "Chelyabinsk" is (15 Feb 2013)
  tung      what a "Tunguska" is (30 Jun 1908)
  heat      how falling Moon rock heats the Earth, and how much, by impactor size
"""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W, H, FPS = 1920, 1080, 24
BG = (7, 8, 11)
INK = (232, 232, 228)
INK2 = (170, 172, 176)
INK3 = (110, 112, 118)
AMBER = (255, 190, 110)
RED = (235, 110, 80)
BLUE = (110, 160, 235)
GREY = (120, 120, 120)
F = {}
FONT = '/usr/share/fonts/opentype/inter/Inter-Regular.otf'
FONT_M = '/usr/share/fonts/opentype/inter/Inter-Medium.otf'
FONT_B = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'


def font(path, size):
    if (path, size) not in F:
        F[(path, size)] = ImageFont.truetype(path, size)
    return F[(path, size)]


def fade(c, a):
    return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))


def txt(d, xy, s, size=32, col=INK, path=FONT, a=1.0, anchor='la'):
    d.text(xy, s, font=font(path, size), fill=fade(col, a), anchor=anchor)


def appear(t, t0, dur=0.6):
    return max(0.0, min(1.0, (t - t0) / dur))


def frames(draw_fn, secs):
    n = int(secs * FPS)
    for f in range(n):
        t = f / FPS
        a_all = min(1.0, t / 0.5, (secs - t) / 0.5)
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        draw_fn(d, t, a_all)
        yield im


def numbers():
    li = json.load(open(os.path.join(ROOT, 'results', 'lunar_impact.json')))
    c = li['cases']['20kms_rock_average']
    lad = {r['D_m']: r for r in json.load(open(os.path.join(ROOT, 'results', 'impact_ladder.json')))['rows']
           if r['material'] == 'rock'}
    return c, lad


def sci(x):
    e = int(math.floor(math.log10(x)))
    sup = str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{x / 10 ** e:.0f} × 10{sup}'


# ---------------------------------------------------------------------------------------- fate
def fate(secs=11.0):
    c, lad = numbers()
    r = lad[1000.0]
    excavated = 5.6e14
    esc = r['escaping_kg']
    p_earth = c['frac_to_earth_30y'] + c['frac_tail']
    p_moon = c['frac_back_to_moon']
    p_sun = 1 - p_earth - p_moon
    X0, X1 = 160, 1760

    def bar(d, y, parts, a, label, g=1.0, al=None):
        """A stacked bar that grows out from the left (g: 0..1); its part labels (alpha al) come after."""
        al = a if al is None else al
        txt(d, (X0, y - 46), label, 28, INK2, a=a)
        x = X0
        for frac, col, _ in parts:
            w = max((X1 - X0) * frac, 6)
            x0, x1 = x, min(x + w - 3, X0 + (X1 - X0) * g)
            if x1 > x0:
                d.rectangle([x0, y, x1, y + 54], fill=fade(col, a))
            x += w
        a = al
        x = X0
        for frac, col, lab in parts:
            w = max((X1 - X0) * frac, 6)
            if lab:
                lx = x + 14 if w > 400 else min(x + w + 14, X1 - 10)
                anchor = 'la' if w > 400 or x + w + 14 < X1 - 600 else 'ra'
                if anchor == 'ra':
                    lx = X1
                txt(d, (lx, y + 70 if w < 400 else y + 12), lab, 26, INK if w > 400 else col, FONT_M, a, anchor)
            x += w

    def draw(d, t, A):
        txt(d, (X0, 70), 'Where does the debris go?', 54, INK, FONT_B, A)
        txt(d, (X0, 145), 'A 1 km asteroid hits the Moon at 20 km/s', 30, INK2, a=A)
        a1 = appear(t, 0.8) * A
        bar(d, 290, [(1 - esc / excavated, GREY, f'{100 * (1 - esc / excavated):.1f}% falls back onto the Moon'),
                     (esc / excavated, AMBER, f'{100 * esc / excavated:.1f}% is faster than 2.4 km/s and escapes the Moon')],
            a1, f'Everything the impact digs out: ~{sci(excavated)} kg', appear(t, 1.0, 0.9), appear(t, 2.0) * A)
        a2 = appear(t, 3.4) * A
        bar(d, 530, [(p_sun, BLUE, f'{100 * p_sun:.0f}% leaves into orbit around the Sun'),
                     (p_earth, RED, f'{100 * p_earth:.0f}% hits the Earth'),
                     (p_moon, GREY, '')],
            a2, f'What escapes: ~{sci(esc)} kg (twice the asteroid\'s own mass)', appear(t, 3.6, 0.9), appear(t, 4.6) * A)
        a3 = appear(t, 6.2) * A
        txt(d, (X0, 730), 'When the Earth-bound part arrives', 28, INK2, a=a3)
        y = 800
        d.line([X0, y, X1, y], fill=fade(INK3, a3), width=3)
        marks = [(9, '10% by day 9'), (400, 'half by month 13'), (2286, '90% by year 6')]
        lo, hi = math.log10(1), math.log10(4000)
        for k, (day, lab) in enumerate(marks):
            x = X0 + (math.log10(day) - lo) / (hi - lo) * (X1 - X0)
            am = appear(t, 6.6 + 0.4 * k) * A                  # each dot, then its label
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=fade(RED, am))
            txt(d, (x, y + 26), lab, 26, INK, FONT_M, appear(t, 7.0 + 0.4 * k) * A, 'ma')
        for day, lab in [(1, '1 day'), (30, '1 month'), (365, '1 year'), (3650, '10 years')]:
            x = X0 + (math.log10(day) - lo) / (hi - lo) * (X1 - X0)
            txt(d, (x, y - 40), lab, 22, INK3, a=a3, anchor='ma')
        a4 = appear(t, 8.2) * A
        txt(d, (X0, 960), 'Over the next million years, some of the Sun-orbiting debris will find the Earth too.', 26, INK2, a=a4)

    return frames(draw, secs)


# ---------------------------------------------------------------------------------------- Chelyabinsk
def person(d, x, ground, h_px, col):
    head = h_px * 0.13
    d.ellipse([x - head / 2, ground - h_px, x + head / 2, ground - h_px + head], fill=col)
    d.line([x, ground - h_px + head, x, ground - h_px * 0.45], fill=col, width=max(2, int(h_px * 0.05)))
    d.line([x, ground - h_px * 0.45, x - h_px * 0.12, ground], fill=col, width=max(2, int(h_px * 0.05)))
    d.line([x, ground - h_px * 0.45, x + h_px * 0.12, ground], fill=col, width=max(2, int(h_px * 0.05)))


def chely(secs=10.0):
    def draw(d, t, A):
        txt(d, (160, 70), 'First: what is a "Chelyabinsk"?', 54, INK, FONT_B, A)
        txt(d, (160, 145), 'Chelyabinsk, Russia, 15 February 2013', 30, INK2, a=A)
        rows = [(0.8, 'A stony asteroid about 20 m across: ~12,000 tonnes'),
                (2.0, 'Hit the air at 19 km/s and exploded about 30 km up'),
                (3.2, 'Energy ~500 kilotons of TNT: about 30 Hiroshima bombs'),
                (4.4, 'Its flash was brighter than the Sun; the shock wave blew out'),
                (4.4, 'windows across a city of a million. ~1,500 people were hurt,'),
                (4.4, 'mostly by flying glass. Nobody died.')]
        y = 270
        for t0, s in rows:
            txt(d, (160, y), s, 32, INK, a=appear(t, t0) * A)
            y += 58
        # scale drawing: the rock next to a six-storey building and a person (1 m = 14 px)
        a = appear(t, 0.8) * A
        ground, s = 900, 14.0
        d.line([1180, ground, 1760, ground], fill=fade(INK3, a), width=2)
        d.ellipse([1240, ground - 20 * s, 1240 + 20 * s, ground], fill=fade((140, 130, 120), a))
        al = appear(t, 1.4) * A                                # labels after the drawing
        txt(d, (1240 + 10 * s, ground + 18), '20 m rock', 24, INK2, a=al, anchor='ma')
        bx = 1560
        d.rectangle([bx, ground - 20 * s, bx + 120, ground], outline=fade(INK2, a), width=3)
        for k in range(1, 6):
            d.line([bx, ground - k * 20 * s / 6, bx + 120, ground - k * 20 * s / 6], fill=fade(INK3, a), width=1)
        txt(d, (bx + 60, ground + 18), '6-storey building', 24, INK2, a=al, anchor='ma')
        person(d, 1720, ground, 1.8 * s, fade(INK, a))
        a2 = appear(t, 3.2) * A
        txt(d, (160, 690), 'Hiroshima bombs', 24, INK3, a=appear(t, 3.8) * A)
        for k in range(30):
            x = 160 + (k % 15) * 46
            yy = 730 + (k // 15) * 46
            d.rectangle([x, yy, x + 36, yy + 36], fill=fade(AMBER, a2 * min(1, (t - 3.2) * 6 - k * 0.15) if t > 3.2 else 0))
        a3 = appear(t, 6.5) * A
        txt(d, (160, 990), 'Rocks this size come off the Moon only in impacts of about a kilometre or more.', 28, INK2, a=a3)

    return frames(draw, secs)


# ---------------------------------------------------------------------------------------- Tunguska
def tung(secs=10.0):
    def draw(d, t, A):
        txt(d, (160, 70), 'And a "Tunguska"?', 54, INK, FONT_B, A)
        txt(d, (160, 145), 'Tunguska, Siberia, 30 June 1908', 30, INK2, a=A)
        rows = [(0.8, 'An object about 50–60 m across exploded 5–10 km above a forest'),
                (2.0, 'Energy 10–15 megatons: roughly 1,000 Hiroshima bombs'),
                (3.2, 'It flattened about 80 million trees over 2,150 km²,'),
                (3.2, 'an area bigger than Greater London. It left no crater.')]
        y = 270
        for t0, s in rows:
            txt(d, (160, y), s, 32, INK, a=appear(t, t0) * A)
            y += 58
        # areas to scale: 1 km^2 = 0.165 px^2 ... circles of equal area
        a = A
        k = 7.2                                          # px per km
        r_t = math.sqrt(2150 / math.pi) * k
        r_l = math.sqrt(1572 / math.pi) * k
        cx, cy = 1340, 640
        g1, g2 = appear(t, 3.4, 0.9), appear(t, 4.6, 0.9)    # each circle grows, then gets its label
        if g1 > 0:
            r_ = r_t * g1
            d.ellipse([cx - r_, cy - r_, cx + r_, cy + r_], fill=fade((60, 42, 24), a * 0.9), outline=fade(AMBER, a), width=4)
        lx = cx + r_t + 60 + r_l
        if g2 > 0:
            r_ = r_l * g2
            d.ellipse([lx - r_, cy - r_, lx + r_, cy + r_], fill=fade((50, 58, 75), a * 0.8), outline=fade(INK2, a), width=3)
        a1, a2 = appear(t, 4.3) * A, appear(t, 5.5) * A
        txt(d, (cx, cy - 10), 'Forest flattened', 26, AMBER, FONT_M, a1, 'ma')
        txt(d, (cx, cy + 24), '2,150 km²', 26, AMBER, a=a1, anchor='ma')
        txt(d, (lx, cy - 10), 'Greater London', 24, INK2, a=a2, anchor='ma')
        txt(d, (lx, cy + 22), '1,572 km²', 24, INK2, a=a2, anchor='ma')
        d.line([cx - r_t, cy + r_t + 30, cx - r_t + 20 * k, cy + r_t + 30], fill=fade(INK2, a1), width=3)
        txt(d, (cx - r_t, cy + r_t + 40), '20 km', 22, INK3, a=a1)
        a3 = appear(t, 6.0) * A
        txt(d, (160, 990), 'A 10 km impact on the Moon could send thousands of these to the Earth over a few years.', 28, INK2, a=a3)

    return frames(draw, secs)


# ---------------------------------------------------------------------------------------- heat
def heat(secs=13.0):
    _, lad = numbers()
    vals = [(1000.0, '1 km'), (1e4, '10 km'), (3e4, '30 km'), (1e5, '100 km')]

    def draw(d, t, A):
        txt(d, (160, 70), 'How does falling Moon rock heat the Earth?', 54, INK, FONT_B, A)
        rows = [(0.6, 'It falls in at 11 km/s or more, the Earth\'s escape speed.'),
                (2.0, 'So every kilogram arrives with 62 MJ: the energy of 15 kg of TNT.'),
                (3.6, 'Dust and pebbles burn up 80–100 km high and heat the air there. That hot air radiates'),
                (3.6, 'about half the heat out to space and half down to the ground. Big rocks explode lower.')]
        y = 170
        for t0, s in rows:
            txt(d, (160, y), s, 30, INK, a=appear(t, t0) * A)
            y += 52
        a = appear(t, 5.8) * A
        txt(d, (160, 430), 'Extra heating in the worst month, averaged over the whole Earth (W per m², log scale)', 26, INK2, a=a)
        X0, X1 = 420, 1720
        lo, hi = -3, 3

        def X(v):
            return X0 + (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo) * (X1 - X0)
        y0 = 500
        for i, (D, lab) in enumerate(vals):
            r = lad[D]
            v = max(r['peak_month_W'], 1e-3)
            yy = y0 + i * 92
            ai = appear(t, 6.2 + 0.7 * i) * A
            g = appear(t, 6.4 + 0.7 * i, 0.6)                  # the bar grows out, then its value appears
            txt(d, (160, yy + 8), f'{lab} impactor', 30, INK, FONT_M, ai)
            if g > 0:
                d.rectangle([X0, yy, X0 + max((X(v) - X0) * g, 4), yy + 52], fill=fade(RED if v > 42 else AMBER, A))
            s = f'{v:.2f} W/m²' if v < 1 else f'{v:,.0f} W/m²'
            if r['peak_month_W'] < 1e-3:
                s = 'less than 0.001 W/m²'
            txt(d, (X(v) + 16, yy + 10), s, 28, INK, a=appear(t, 7.0 + 0.7 * i) * A)
        a5 = appear(t, 9.2) * A
        xs = X(240)
        d.line([xs, y0 - 12, xs, y0 + 4 * 92 + 30], fill=fade(BLUE, a5), width=3)
        txt(d, (xs - 12, y0 + 4 * 92 + 2), 'all the sunlight the Earth absorbs: 240 W/m²', 24, BLUE, FONT_M,
            appear(t, 9.7) * A, 'ra')
        a6 = appear(t, 10.6) * A
        txt(d, (160, 950), 'A wet planet can only shed ~40 W/m² of extra heat. Keep adding more for years and the oceans start to boil.', 26, INK2, a=a6)

    return frames(draw, secs)


GRAPHICS = dict(fate=fate, chely=chely, tung=tung, heat=heat)
