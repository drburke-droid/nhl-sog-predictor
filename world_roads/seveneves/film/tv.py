"""The waiting-room cut: a silent, self-explaining loop. Everything must read from across a room.

  * one idea per screen, at most three short lines, big type (titles ~90 px, body ~56 px at 1080p)
  * every screen stays up long enough to read (1.8 s + words / 2.6 per second, never less than 3.5 s)
  * the same five beats every round: ROUND n of 5 + the rock to scale -> the hit -> the crater ->
    what the Earth gets -> a verdict stamp (HARMLESS / TROUBLE / DANGER / CATASTROPHE) and how often
  * a ladder strip across the top shows which of the five sizes we are on, and how the earlier ones went

python3 film/tv.py [quality] [out.mp4]      quality: final (falls back to animatic per shot) | animatic
Also writes film/tv_script.md (every caption with its time: a narration script) and the score.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import shots_meta

W, H, FPS = 1920, 1080, 24
RENDER = '/opt/seveneves_render'
FONT = '/usr/share/fonts/opentype/inter/Inter-Regular.otf'
FONT_M = '/usr/share/fonts/opentype/inter/Inter-Medium.otf'
FONT_B = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
FONT_X = '/usr/share/fonts/opentype/inter/Inter-Bold.otf'
INK = (240, 240, 236)
INK2 = (190, 192, 196)
INK3 = (120, 122, 128)
AMBER = (255, 190, 110)
BG = (7, 8, 11)
VERDICT = {   # colour, symbol
    'HARMLESS': ((95, 200, 120), '✓'),
    'TROUBLE': ((250, 200, 80), '!'),
    'DANGER': ((245, 130, 60), '!!'),
    'CATASTROPHE': ((235, 70, 60), '×'),
}
SIZES = ['1 m', '60 m', '1 km', '10 km', '100 km']
_F = {}


def font(path, size):
    if (path, size) not in _F:
        _F[(path, size)] = ImageFont.truetype(path, size)
    return _F[(path, size)]


class _AlphaDraw(ImageDraw.ImageDraw):
    """ImageDraw whose text() honours the alpha of an RGBA fill on an RGB image. (Pillow 12 draws such text fully
    opaque, which made every caption and chart label appear at once instead of fading in after its graphic.)"""

    def text(self, xy, text, fill=None, font=None, anchor=None, *args, stroke_width=0, stroke_fill=None, **kw):
        if self._image.mode != 'RGB' or not (isinstance(fill, tuple) and len(fill) == 4):
            return super().text(xy, text, fill, font, anchor, *args, stroke_width=stroke_width, stroke_fill=stroke_fill, **kw)
        a = fill[3]
        sa = stroke_fill[3] if isinstance(stroke_fill, tuple) and len(stroke_fill) == 4 else a
        if a <= 0 and (not stroke_width or sa <= 0):
            return
        x0, y0, x1, y1 = self.textbbox(xy, text, font=font, anchor=anchor, stroke_width=stroke_width)
        x0, y0, x1, y1 = int(x0) - 2, int(y0) - 2, int(x1) + 2, int(y1) + 2
        if x1 <= x0 or y1 <= y0:
            return
        loc = (xy[0] - x0, xy[1] - y0)
        if stroke_width and stroke_fill is not None and sa > 0:
            m = Image.new('L', (x1 - x0, y1 - y0), 0)
            ImageDraw.Draw(m).text(loc, text, fill=255, font=font, anchor=anchor, stroke_width=stroke_width, stroke_fill=255)
            self._image.paste(tuple(stroke_fill[:3]), (x0, y0, x1, y1), m.point(lambda v: v * sa // 255))
        if a > 0:
            m = Image.new('L', (x1 - x0, y1 - y0), 0)
            ImageDraw.Draw(m).text(loc, text, fill=255, font=font, anchor=anchor)
            self._image.paste(tuple(fill[:3]), (x0, y0, x1, y1), m.point(lambda v: v * a // 255))


def Draw(im, mode=None):
    return _AlphaDraw(im, mode)


def happens(y):
    e = every(y)
    return 'Today: never. No rocks this big are left near the Earth.' if e.startswith('never') else 'Happens ' + e


def read_time(lines, minimum=3.5):
    words = sum(len(s.split()) for s in lines)
    return max(minimum, 1.8 + words / 2.6)


def ladder():
    rows = json.load(open(os.path.join(ROOT, 'results', 'impact_ladder.json')))['rows']
    return {r['D_m']: r for r in rows if r['material'] == 'rock'}


def every(y):
    if y < 1 / 12:
        return f'about every {y * 365.25:.0f} days'
    if y < 1:
        return f'about every {y * 12:.0f} months'
    if y < 1e3:
        return f'about every {y:,.0f} years'
    if y < 1e6:
        return f'about once every {round(y, -3):,.0f} years'
    if y < 1e9:
        return f'about once every {y / 1e6:,.0f} million years'
    if y < 13.8e9:
        return f'about once every {y / 1e9:,.0f} billion years'
    return 'never: no rocks this big are left near the Earth'


# ------------------------------------------------------------------------------------ drawing helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def alpha_in_out(t, dur, fin=0.45, fout=0.45):
    return ease(min(t / fin, 1.0)) * ease(min((dur - t) / fout, 1.0))


def panel_text(im, lines, a, where='lower', title_size=78, body_size=54, accent=None, maxw=1500):
    """Lines in a soft dark panel. lines[0] is the headline."""
    if a <= 0 or not lines:
        return
    d = Draw(im, 'RGBA')
    sizes = [(FONT_B, title_size)] + [(FONT_M, body_size)] * (len(lines) - 1)
    gaps = [int(s * 1.28) for _, s in sizes]
    widths = [d.textlength(s, font=font(p, z)) for s, (p, z) in zip(lines, sizes)]
    bw = min(max(widths) + 100, W - 120)
    bh = sum(gaps) + 50
    x0 = 90
    y0 = {'lower': H - bh - 70, 'upper': 150, 'center': (H - bh) // 2}[where]
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=22, fill=(5, 6, 9, int(185 * a)))
    if accent:
        d.rounded_rectangle([x0, y0, x0 + 10, y0 + bh], radius=5, fill=accent + (int(255 * a),))
    y = y0 + 26
    for s, (p, z), g in zip(lines, sizes, gaps):
        col = INK if p == FONT_B else INK2
        d.text((x0 + 50, y), s, font=font(p, z), fill=col + (int(255 * a),))
        y += g


def center_text(im, lines, a, sizes=None, y_mid=None, cols=None):
    d = Draw(im, 'RGBA')
    sizes = sizes or [96] + [58] * (len(lines) - 1)
    cols = cols or [INK] + [INK2] * (len(lines) - 1)
    hts = [int(s * 1.35) for s in sizes]
    y = (y_mid or H / 2) - sum(hts) / 2
    for s, z, c, h in zip(lines, sizes, cols, hts):
        d.text((W / 2, y + h / 2), s, font=font(FONT_B if z >= 80 else FONT_M, z), fill=c + (int(255 * a),),
               anchor='mm')
        y += h


def hud(im, rnd, results, a=1.0):
    """'ROUND n OF 5' and the ladder of sizes, with past rounds' verdict colours."""
    if rnd is None:
        return
    d = Draw(im, 'RGBA')
    d.rectangle([0, 0, W, 96], fill=(4, 5, 8, int(170 * a)))
    d.text((60, 48), f'ROUND {rnd} OF 5', font=font(FONT_X, 40), fill=AMBER + (int(255 * a),), anchor='lm')
    x = 760
    for i, s in enumerate(SIZES):
        on = i + 1 == rnd
        w = d.textlength(s, font=font(FONT_B, 36)) + 44
        if on:
            d.rounded_rectangle([x, 24, x + w, 72], radius=24, fill=AMBER + (int(255 * a),))
            d.text((x + w / 2, 48), s, font=font(FONT_B, 36), fill=(20, 20, 20, int(255 * a)), anchor='mm')
        else:
            col = INK3
            if i + 1 < rnd and results.get(i + 1):
                col = VERDICT[results[i + 1]][0]
                d.ellipse([x + w / 2 - 7, 78, x + w / 2 + 7, 92], fill=col + (int(255 * a),))
            d.text((x + w / 2, 48), s, font=font(FONT_M, 36), fill=col + (int(255 * a),), anchor='mm')
        x += w + 26
        if i < 4:
            d.text((x - 13, 48), '›', font=font(FONT_M, 36), fill=INK3 + (int(255 * a),), anchor='mm')


def stamp(im, verdict, note, a):
    col, sym = VERDICT[verdict]
    d = Draw(im, 'RGBA')
    s = 0.85 + 0.15 * ease(a)
    cx, cy = W / 2, H / 2 - 40
    f = font(FONT_X, int(120 * s))
    tw = d.textlength(f'{sym}  {verdict}', font=f)
    pad = 50
    d.rounded_rectangle([cx - tw / 2 - pad, cy - 100 * s, cx + tw / 2 + pad, cy + 100 * s], radius=26,
                        fill=(6, 7, 10, int(200 * a)), outline=col + (int(255 * a),), width=8)
    d.text((cx, cy), f'{sym}  {verdict}', font=f, fill=col + (int(255 * a),), anchor='mm')
    if note:
        d.text((cx, cy + 170), note, font=font(FONT_M, 50), fill=INK + (int(255 * a),), anchor='mm')


# ------------------------------------------------------------------------------------ backgrounds
class ShotFrames:
    def __init__(self, shot, quality):
        self.shot = shot
        self.n = shots_meta.FRAMES[shot]
        self.dirs = []
        for q in ([quality, 'animatic'] if quality != 'animatic' else ['animatic']):
            d = os.path.join(RENDER, q, shot)
            if os.path.isdir(d):
                self.dirs.append(d)
        self.cache = {}

    def get(self, f):
        f = int(min(max(f, 0), self.n - 1))
        for d in self.dirs:
            for g in range(f, -1, -1):              # nearest rendered frame at or before f
                p = os.path.join(d, f'{g + 1:04d}.png')
                if os.path.exists(p):
                    if p not in self.cache:
                        im = Image.open(p).convert('RGB')
                        if im.size != (W, H):
                            im = im.resize((W, H), Image.LANCZOS)
                        self.cache = {p: im}
                    return self.cache[p].copy()
                if g < f - 3:
                    break
        return Image.new('RGB', (W, H), BG)


_FLASH = {}


def flash_xy(sf):
    """Screen position of the impact flash: the brightest pixel a few frames after the hit."""
    if sf.shot not in _FLASH:
        a = np.asarray(sf.get(34)).astype(float).sum(2)
        y, x = np.unravel_index(np.argmax(a), a.shape)
        _FLASH[sf.shot] = (int(x), int(y))
    return _FLASH[sf.shot]


def ring(im, xy, a, label_s='impact'):
    d = Draw(im, 'RGBA')
    x, y = xy
    for r_, w_ in ((70, 6),):
        d.ellipse([x - r_, y - r_, x + r_, y + r_], outline=AMBER + (int(255 * a),), width=w_)
    d.text((x, y - 100), label_s, font=font(FONT_B, 44), fill=AMBER + (int(255 * a),), anchor='mm')


def push(im, z):
    if z <= 1.0001:
        return im
    w, h = int(W / z), int(H / z)
    x, y = (W - w) // 2, (H - h) // 2
    return im.crop((x, y, x + w, y + h)).resize((W, H), Image.BILINEAR)


# ------------------------------------------------------------------------------------ rendered scale cards
_LABELS = {}


def shot_labels(shot, quality):
    """Per-frame screen positions (fractions) of a rendered shot's labels, written by render.py."""
    if shot not in _LABELS:
        _LABELS[shot] = {}
        for q in ([quality, 'animatic'] if quality != 'animatic' else ['animatic']):
            f = os.path.join(RENDER, q, shot, 'labels.json')
            if os.path.exists(f):
                _LABELS[shot] = json.load(open(f))
                break
    return _LABELS[shot]


def outlined(d, xy, s, f, fill, a, anchor='mm', stroke=4):
    d.text(xy, s, font=f, fill=fill + (int(255 * a),), anchor=anchor, stroke_width=stroke,
           stroke_fill=(0, 0, 0, int(200 * a)))


def scale_overlay(im, t, dur, p, fr, quality):
    """A rendered scene with the title and comparison in a band at the bottom, and labels that follow the rock
    and the familiar thing (from the per-frame label positions render.py saved)."""
    a = alpha_in_out(t, dur)
    scrim = Image.new('L', (1, 380))
    scrim.putdata([int(200 * a * (y / 380) ** 1.3) for y in range(380)])
    im.paste((0, 0, 0), (0, H - 380, W, H), scrim.resize((W, 380)))
    d = Draw(im, 'RGBA')
    outlined(d, (W / 2, H - 190), p['title'], font(FONT_B, 84), INK, a, stroke=3)
    outlined(d, (W / 2, H - 92), p['compare'], font(FONT_M, 52), INK2, a * ease((t - 0.8) / 0.6), stroke=3)
    lab = shot_labels(p['shot'], quality).get(str(int(min(fr, shots_meta.FRAMES[p['shot']] - 1))), {})
    b = a * ease((t - 1.2) / 0.6)
    if b <= 0:
        return
    f = font(FONT_B, 46)
    for key, col in (('rock', AMBER), ('other', INK)):
        if key not in lab or key not in p['labels']:
            continue
        x, y = lab[key][0] * W, lab[key][1] * H
        lift = p.get('lift', {}).get(key, 72)
        tw = d.textlength(p['labels'][key], font=f)
        if y - lift >= 150:                                   # label above the object
            ty = y - lift
            tx = min(max(x, tw / 2 + 40), W - tw / 2 - 40)
            d.line([(x, y - 8), (x, ty + 28)], fill=col + (int(230 * b),), width=4)
        else:                                                 # too near the top: label beside it
            ty = max(y, 150)
            side = -1 if x - tw - 120 > 40 else 1
            tx = x + side * (tw / 2 + 90)
            d.line([(x, ty), (x + side * 70, ty)], fill=col + (int(230 * b),), width=4)
        d.ellipse([x - 7, y - 15, x + 7, y - 1], fill=col + (int(255 * b),))
        outlined(d, (tx, ty), p['labels'][key], f, col, b, stroke=4)


class LineupFrames:
    """The line-up of five turning rocks (RGBA)."""
    def __init__(self, quality):
        self.n = shots_meta.FRAMES['P_lineup']
        self.dirs = [os.path.join(RENDER, q, 'P_lineup') for q in dict.fromkeys([quality, 'animatic'])]
        self.cache = {}

    def get(self, f):
        f = int(min(max(f, 0), self.n - 1))
        for dd in self.dirs:
            for g in range(f, max(-1, f - 6), -1):
                pth = os.path.join(dd, f'{g + 1:04d}.png')
                if os.path.exists(pth):
                    if pth not in self.cache:
                        im = Image.open(pth).convert('RGBA')
                        if im.size != (W, H):
                            im = im.resize((W, H), Image.LANCZOS)
                        self.cache = {pth: im}
                    return self.cache[pth]
        return None


_LINEUP = {}


# ------------------------------------------------------------------------------------ the story
def casualties():
    rows = json.load(open(os.path.join(ROOT, 'results', 'casualties.json')))['rows']
    return {r['D_m']: r for r in rows}


def per_place(per_day_world, radius_km=500.0):
    """How often a fireball seen worldwide `per_day_world` times a day shows up in one place's sky (visible
    to ~500 km)."""
    d = per_day_world * math.pi * radius_km ** 2 / 5.1e8
    if d < 0.7:
        return f'about once every {round(1 / d):.0f} days'
    if d < 2:
        return 'about once a day'
    if d < 48:
        return f'about {d:.0f} times a day'
    mins = 1440 / d
    return 'every couple of minutes' if mins < 4 else f'about every {mins:.0f} minutes'


def seveneves():
    """The default Seveneves run (pieces drift apart at 150 m/s): when the rain starts, how much ocean boils."""
    r = json.load(open(os.path.join(ROOT, 'results', 'results.json')))['scenarios']['150']['summary']
    frac = r['climate']['max_ocean_boiled']
    words = {0.25: 'a quarter', 0.5: 'half', 0.75: 'three-quarters', 1.0: 'all'}
    ocean = words[min(words, key=lambda k: abs(k - frac))]
    return dict(ocean=ocean, days=max(1, round(r['rain']['first_debris_day'])))


def millions(x):
    if x < 1e3:
        return f'{x:,.0f}'
    if x < 1e6:
        return f'{round(x, -3):,.0f}'
    if x < 1e9:
        return f'{x / 1e6:,.0f} million'
    return f'{x / 1e9:,.1f} billion'


def build():
    L = ladder()
    C = casualties()
    seg = []          # (kind, duration, payload); a background (shot, None) continues from where that shot was

    def say(bg, lines, where='lower', dur=None, rnd=None, accent=None, **kw):
        seg.append(('say', dur or read_time(lines), dict(bg=bg, lines=lines, where=where, rnd=rnd, accent=accent, **kw)))

    def card(lines, dur=None, sizes=None, rnd=None, cols=None):
        seg.append(('card', dur or read_time(lines, 4.0), dict(lines=lines, sizes=sizes, rnd=rnd, cols=cols)))

    def draw(fn, dur, rnd=None, script=''):
        seg.append(('draw', dur, dict(fn=fn, rnd=rnd, script=script)))

    def scale(shot, title, compare, labels, dur, rnd, script, **kw):
        seg.append(('scale', dur, dict(bg=(shot, 0), play=1.0, shot=shot, title=title, compare=compare, labels=labels,
                                       rnd=rnd, script=script, **kw)))

    def verdict(bg, v, note, rnd):
        seg.append(('verdict', 5.5, dict(bg=bg, v=v, note=note, rnd=rnd)))

    def bullets(title, items, rnd, per=None):
        dur = 1.5 + sum(read_time([i], 2.6) for i in items) + 1.0
        seg.append(('list', dur, dict(title=title, items=items, rnd=rnd)))

    C3, C4, C5 = C[1000.0], C[1e4], C[1e5]
    c4, lo4, hi4 = C4['central'], C4['low'], C4['high']
    # ---------------------------------------------------------------- intro (the Moon turns slowly, never resets)
    say(('H_ladder', 0), [], where='center', dur=4.5, big=['WHAT IF SOMETHING', 'HIT THE MOON?'])
    # the novel that asks the question
    say(('B_breakup', 0), ['In Neal Stephenson\'s novel Seveneves,', 'something shatters the Moon', 'into seven pieces.'],
        dur=7.0)
    say(('E_rain', 60), ['The pieces grind each other to rubble.', 'Two years later it rains down on Earth:',
                         'the "Hard Rain" sets the sky on fire', 'and wipes out almost all life.'], dur=8.0)
    say(('H_ladder', None), ['Could a hit on the Moon really hurt us?', 'The Moon gets hit by space rocks all the time.',
                             'Most are tiny. A few have been huge.'])
    card(['Let\'s find out.', 'We\'ll throw bigger and bigger rocks at the Moon,', 'and see what happens to Earth each time.'],
         dur=7.0, sizes=[96, 54, 54])
    seg.append(('plan', 6.0, {}))
    # ---------------------------------------------------------------- round 1: 1 m
    r = L[1.0]
    scale('S1_1m', 'A rock 1 metre across', 'about the size of an armchair',
          dict(rock='1 m rock', other='a person (1.75 m)'), 7.0, 1,
          'Round 1: a rock 1 metre across, about the size of an armchair.')
    say(('I1_1m', 0), ['It hits the Moon at 20 km per second', '(about 60 times faster than a bullet).'], rnd=1, dur=6.0)
    say(('I1_1m', None), ['It leaves a crater about 20 m wide.'], rnd=1, dur=4.0)
    say(('I1_1m', None), ['What does Earth get?', 'Nothing at all.'], rnd=1)
    verdict(('I1_1m', None), 'HARMLESS', happens(r['moon_interval_yr']), 1)
    # ---------------------------------------------------------------- round 2: 60 m
    r = L[60.0]
    scale('S2_60m', 'A rock 60 metres across', 'as tall as a 20-storey building',
          dict(rock='60 m rock', other='20-storey building (60 m)'), 7.0, 2,
          'Round 2: a rock 60 metres across, as tall as a 20-storey building.')
    say(('I2_60m', 0), ['Same speed: 20 km per second.', 'About as big as asteroid 2024 YR4.'], rnd=2)
    say(('I2_60m', None), ['A flash you could see without a telescope.', 'It leaves a crater about 1.2 km wide.'], rnd=2)
    say(('I2_60m', None), ['What does Earth get?', 'A few extra shooting stars for a week.'], rnd=2)
    verdict(('I2_60m', None), 'HARMLESS', happens(r['moon_interval_yr']), 2)
    # ---------------------------------------------------------------- round 3: 1 km
    r = L[1000.0]
    scale('S3_1km', 'A rock 1 kilometre across', 'taller than the world\'s tallest building',
          dict(rock='1 km rock', other='world\'s tallest building (828 m)'), 7.0, 3,
          'Round 3: a rock 1 kilometre across, taller than the world\'s tallest building.')
    say(('I3_1km', 0), ['It hits the Moon...'], rnd=3, dur=3.5)
    say(('I3_1km', None), ['...and blasts rock out of the crater', 'faster than the Moon\'s gravity can hold it.'], rnd=3)
    say(('I3_1km', None), ['The crater: 18 km wide.'], rnd=3, dur=4.0)
    say(('J_ejecta', 0), ['Where does the blasted rock go?'], rnd=3, dur=4.0)
    seg.append(('fate', 9.0, dict(rnd=3)))
    say(('J_ejecta', None), ['Most of it drifts off around the Sun.',
                             'The orange streaks are pieces that will hit Earth.'], rnd=3, dur=8.0, play=2.0)
    say(('E3_1km', 0), ['What does Earth get?', 'A storm of shooting stars:', '10,000 times more than normal, for a week.'],
        rnd=3)
    say(('N3_1km', 0), ['From the ground, at night:', 'about 30 shooting stars every second,', 'all week long.'],
        rnd=3, dur=7.0, tag='REAL SPEED')
    bullets('What would that mean for us?',
            [f'Satellites take ~{round(C3["satellite_years_in_a_week"], -1):.0f} years\' worth of meteoroid hits in one week.',
             'Some would fail: GPS, satellite TV and weather forecasts could glitch.',
             'On the ground: a spectacular meteor storm. Most likely nobody is hurt.'], 3)
    verdict(('E3_1km', None), 'TROUBLE', happens(r['moon_interval_yr']), 3)
    # ---------------------------------------------------------------- round 4: 10 km
    r = L[1e4]
    scale('S4_10km', 'A rock 10 kilometres across', 'it would dwarf Mount Everest: the size of the dinosaur-killer',
          dict(rock='10 km rock', other='Mount Everest (8,849 m)'), 7.0, 4,
          'Round 4: a rock 10 kilometres across, the size of the one that wiped out the dinosaurs.')
    say(('I4_10km', 0), ['It hits the Moon.'], rnd=4, dur=3.5)
    say(('I4_10km', None), ['A crater 135 km wide.', 'Now some of the pieces flying at Earth', 'are as big as office towers.'], rnd=4)
    card(['Pieces that big explode in our sky.', 'Two real explosions show what that means:'], dur=5.5, sizes=[72, 54], rnd=4)
    seg.append(('chely', 9.5, dict(rnd=4)))
    seg.append(('tung', 9.5, dict(rnd=4)))
    say(('E4_10km', 0), ['What does Earth get, over about six years?',
                         f'~{round(c4["chely"], -3):,.0f} blasts as strong as Chelyabinsk,',
                         f'~{round(c4["tung"], -2):,.0f} as strong as Tunguska.'], rnd=4)
    say(('N4_10km', 0), ['From the ground, at night:', 'about 30,000 shooting stars every second,',
                         'as much light as a full Moon, and',
                         f'{per_place(C4["chely_per_day_first_week"])} a fireball as bright as the Sun.'],
        rnd=4, dur=7.5, tag='REAL SPEED')
    bullets('What would daily life be like?',
            [f'At first, ~{round(C4["chely_per_day_first_week"], -2):,.0f} big blasts a day, anywhere on Earth.',
             f'Any city: about a 1 in {round(1 / C4["city_chely_first_year"]):.0f} chance of one nearby in the first year.',
             'No warning: the pieces are too small and too many to track.',
             'Every satellite gone within days: no GPS, satellite TV or weather satellites.'], 4)
    seg.append(('toll', 10.0, dict(rnd=4, deaths=c4['deaths'], injured=c4['injuries'], hi=hi4['deaths'])))
    say(('E4_10km', None), ['And the climate?', 'It barely changes (+0.01 °C): the heat is spread over years.',
                            'The danger is the blasts, not the heat.'], rnd=4)
    verdict(('E4_10km', None), 'DANGER', happens(r['moon_interval_yr']), 4)
    # ---------------------------------------------------------------- round 5: 100 km
    r = L[1e5]
    scale('S5_100km', 'A rock 100 kilometres across', 'standing on the ground, it would reach the edge of space',
          dict(rock='100 km rock', other='10 km (dinosaur-killer)'), 7.0, 5,
          'Round 5: a rock 100 kilometres across, ten times wider than the dinosaur-killer. Standing on the ground, '
          'it would reach the edge of space.')
    say(('I5_100km', 0), ['It hits the Moon.'], rnd=5, dur=3.5)
    say(('I5_100km', None), ['A crater 1,000 km wide,', 'like the Moon\'s biggest dark patches.'], rnd=5)
    say(('K_entry', 40), ['Every piece of Moon rock hits our air', 'at more than 11 km per second.'], rnd=5)
    say(('K_entry', None), ['That carries 15 times the energy', 'of the same weight of dynamite.'], rnd=5)
    seg.append(('heat', 9.0, dict(rnd=5)))
    say(('E5_100km', 0), ['What does Earth get?', 'Blasts that flatten almost every city within a year,',
                          'and for months, more heat from the sky than from the Sun.'], rnd=5)
    say(('N5_100km', 0), ['From the ground, at night:', 'night turns to day, lit by countless shooting stars,',
                          f'and {per_place(C5["chely_per_day_first_week"])} a fireball as bright as the Sun.'],
        rnd=5, dur=7.0, tag='REAL SPEED (YOUR EYES ADJUST TO THE GLARE)')
    say(('E5_100km', None), ['Earth warms by about 11 °C.', 'Most life would die out.'], rnd=5)
    verdict(('E5_100km', None), 'CATASTROPHE', happens(r['moon_interval_yr']), 5)
    # ---------------------------------------------------------------- the end
    seg.append(('score', 12.0, dict(deaths4=c4['deaths'])))
    card(['Satellites get hurt at about 1 km.', 'People are in danger at about 10 km.',
          'That hits the Moon about once every 2 billion years.'], dur=8.0, sizes=[72, 72, 54],
         cols=[INK, INK, INK2])
    # back to the novel
    sv = seveneves()
    say(('B_breakup', 0), ['And Seveneves?', 'Shattering the Moon would take a rock about 3,000 km across,',
                            'nearly as big as the Moon itself.', 'Nothing that big has hit it since it formed.'], dur=9.0)
    say(('F_steam', 0), ['If the Moon did shatter...', 'our simulation agrees with the book: the falling rock',
                         f'would boil away {sv["ocean"]} of the oceans.',
                         f'Only it would start within {sv["days"]} days, not two years.'], dur=9.0, play=1.0)
    card(['So relax.', 'The Moon is not going to rain down on us.'], dur=6.0, sizes=[96, 58])
    card(['Real physics: crater scaling, 6,000 simulated debris paths, impact rates from', 'NASA-funded surveys,',
          'casualties scaled from Chelyabinsk and Tunguska.',
          'Moon images: NASA LRO.  Earth: Solar System Scope (CC BY 4.0).',
          'Everest: Copernicus DEM GLO-30 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH',
          '2014-2018, provided under COPERNICUS by the European Union and ESA).',
          'People: Blender Studio Human Base Meshes (CC0).  Trees and skies: Poly Haven (CC0).',
          'Building and ground textures: ambientCG (CC0).'], dur=8.0, sizes=[32] * 8, cols=[INK3] * 8)
    return seg


# ------------------------------------------------------------------------------------ explainer screens
def fate_screen(im, t, dur):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 190), 'For every 1,000 tonnes of rock the impact blasts out...', font=font(FONT_B, 58),
           fill=INK + (int(255 * a),), anchor='mm')
    rows = [(1.2, '996 tonnes', 'fall back onto the Moon', (140, 140, 140)),
            (3.4, '4 tonnes', 'escape into space', AMBER),
            (5.6, '160 kg', 'of those hit Earth, over months to years', (235, 110, 80))]
    for i, (t0, big, small, col) in enumerate(rows):
        b = ease((t - t0) / 0.6) * a
        y = 390 + i * 190
        d.text((620, y), big, font=font(FONT_X, 92), fill=col + (int(255 * b),), anchor='rm')
        d.text((680, y), small, font=font(FONT_M, 56), fill=INK + (int(255 * b),), anchor='lm')


def chely_screen(im, t, dur):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 170), 'What is a "Chelyabinsk"?', font=font(FONT_B, 84), fill=INK + (int(255 * a),), anchor='mm')
    lines = [(0.8, 'In 2013 a 20-metre rock exploded in the sky over Chelyabinsk, Russia.'),
             (2.6, 'The blast broke windows across the whole city.'),
             (4.4, 'About 1,500 people were hurt, mostly by flying glass.')]
    for i, (t0, s) in enumerate(lines):
        b = ease((t - t0) / 0.6) * a
        d.text((W / 2, 340 + i * 100), s, font=font(FONT_M, 52), fill=INK2 + (int(255 * b),), anchor='mm')
    # a sketch: the rock's path, the flash 30 km up, a city below
    b = ease((t - 1.2) / 0.8) * a
    g = 950
    d.line([300, g, 1620, g], fill=INK3 + (int(255 * b),), width=3)
    for i in range(16):
        x = 820 + i * 22
        hgt = 14 + (i * 37 % 40)
        d.rectangle([x, g - hgt, x + 16, g], fill=(110, 115, 125, int(255 * b)))
    fx, fy = 960, 700
    d.line([520, 610, fx, fy], fill=(255, 220, 170, int(200 * b)), width=6)
    flash = ease((t - 2.0) / 0.4) * a
    for r_, al in ((70, 60), (40, 140), (18, 255)):
        d.ellipse([fx - r_, fy - r_, fx + r_, fy + r_], fill=(255, 230, 180, int(al * flash)))
    d.text((fx + 90, fy), '~30 km up', font=font(FONT_M, 36), fill=INK3 + (int(255 * ease((t - 2.5) / 0.5) * a),),
           anchor='lm')


def tung_screen(im, t, dur):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 170), 'What is a "Tunguska"?', font=font(FONT_B, 84), fill=INK + (int(255 * a),), anchor='mm')
    lines = [(0.8, 'In 1908 a rock about 50–60 metres across'),
             (0.8, 'exploded over a forest in Siberia.'),
             (2.8, 'It flattened 80 million trees: an area'),
             (2.8, 'bigger than the whole of Greater London.')]
    for i, (t0, s) in enumerate(lines):
        b = ease((t - t0) / 0.6) * a
        d.text((W / 2, 320 + i * 76), s, font=font(FONT_M, 52), fill=INK2 + (int(255 * b),), anchor='mm')
    # each circle grows first; its label appears only once the circle is drawn
    k = 4.6
    rt, rl = math.sqrt(2150 / math.pi) * k, math.sqrt(1572 / math.pi) * k
    cy = 740
    g1 = ease((t - 3.4) / 0.9)
    if g1 > 0:
        r_ = rt * g1
        d.ellipse([760 - r_, cy - r_, 760 + r_, cy + r_], fill=(70, 50, 28, int(255 * a)), outline=AMBER + (int(255 * a),),
                  width=4)
    b1 = ease((t - 4.3) / 0.5) * a
    d.text((760, cy + rt + 40), 'flattened forest', font=font(FONT_M, 38), fill=AMBER + (int(255 * b1),), anchor='mm')
    g2 = ease((t - 4.7) / 0.9)
    if g2 > 0:
        r_ = rl * g2
        d.ellipse([1200 - r_, cy - r_, 1200 + r_, cy + r_], fill=(60, 70, 90, int(200 * a)), outline=INK2 + (int(255 * a),),
                  width=4)
    b2 = ease((t - 5.6) / 0.5) * a
    d.text((1200, cy + rt + 40), 'Greater London', font=font(FONT_M, 38), fill=INK2 + (int(255 * b2),), anchor='mm')


def heat_screen(im, t, dur, L):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 160), 'Heat from falling rock vs. heat from the Sun', font=font(FONT_B, 70),
           fill=INK + (int(255 * a),), anchor='mm')
    d.text((W / 2, 240), 'watts per square metre, averaged over the whole Earth, in the worst month', font=font(FONT_M, 38),
           fill=INK3 + (int(255 * a),), anchor='mm')
    rows = [('1 km rock', max(L[1000.0]['peak_month_W'], 0.001)), ('10 km rock', L[1e4]['peak_month_W']),
            ('100 km rock', L[1e5]['peak_month_W']), ('The Sun', 240.0)]
    X0, X1 = 600, 1700
    lo, hi = -3, 3
    for i, (lab, v) in enumerate(rows):
        t0 = 0.8 + 1.2 * i
        b = ease((t - t0) / 0.5) * a                        # the row's name
        grow = ease((t - t0 - 0.3) / 0.7)                   # then the bar grows out to its value
        bv = ease((t - t0 - 1.0) / 0.4) * a                 # and only then the number
        y = 330 + i * 150
        sun = lab == 'The Sun'
        col = (120, 170, 240) if sun else ((235, 90, 70) if v > 240 else AMBER)
        x = max(X0 + (math.log10(v) - lo) / (hi - lo) * (X1 - X0), X0 + 8)
        d.text((X0 - 30, y + 35), lab, font=font(FONT_B, 54), fill=INK + (int(255 * b),), anchor='rm')
        if grow > 0:
            d.rounded_rectangle([X0, y, X0 + max((x - X0) * grow, 8), y + 70], radius=10, fill=col + (int(255 * a),))
        s_ = 'almost none' if v < 0.01 else (f'{v:.1f}' if v < 10 else f'{v:,.0f}')
        d.text((x + 24, y + 35), s_, font=font(FONT_M, 50), fill=INK + (int(255 * bv),), anchor='lm')
    b = ease((t - 6.0) / 0.6) * a
    d.text((W / 2, 960), 'Only the 100 km rock out-heats the Sun.', font=font(FONT_M, 54), fill=INK + (int(255 * b),),
           anchor='mm')


def plan_screen(im, t, dur, quality='animatic'):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 230), 'Five rounds. Each rock is bigger.', font=font(FONT_B, 80), fill=INK + (int(255 * a),), anchor='mm')
    d.text((W / 2, 320), '(not to scale: each one is 10 to 60 times bigger than the one before)', font=font(FONT_M, 38),
           fill=INK3 + (int(255 * a),), anchor='mm')
    xs = [300, 620, 960, 1300, 1640]
    rad = [16, 30, 52, 92, 150]
    if quality not in _LINEUP:
        _LINEUP[quality] = LineupFrames(quality)
    lu = _LINEUP[quality].get(t * FPS)
    for i, (x, r_, s) in enumerate(zip(xs, rad, SIZES)):
        b = ease((t - 0.6 - 0.5 * i) / 0.5) * a
        if lu is not None and b > 0:
            m = int(r_ * 1.7) + 6
            box = (x - m, 600 - m, x + m, 600 + m)
            crop = lu.crop(box)
            al = crop.getchannel('A').point(lambda v, b=b: int(v * b))
            crop.putalpha(al)
            im.paste(crop, box[:2], crop)
        elif lu is None:
            d.ellipse([x - r_, 600 - r_, x + r_, 600 + r_], fill=(150, 140, 128, int(255 * b)))
        bl = ease((t - 0.9 - 0.5 * i) / 0.5) * a             # each size label just after its rock
        d.text((x, 830), s, font=font(FONT_B, 58), fill=INK + (int(255 * bl),), anchor='mm')
        d.text((x, 900), f'round {i + 1}', font=font(FONT_M, 38), fill=INK3 + (int(255 * bl),), anchor='mm')


def score_screen(im, t, dur, L):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 120), 'The results', font=font(FONT_B, 86), fill=INK + (int(255 * a),), anchor='mm')
    rows = [('1 m', 'HARMLESS', L[1.0], 'a flash'), ('60 m', 'HARMLESS', L[60.0], 'shooting stars'),
            ('1 km', 'TROUBLE', L[1000.0], 'satellites hit'), ('10 km', 'DANGER', L[1e4], '~40 million die'),
            ('100 km', 'CATASTROPHE', L[1e5], 'most life dies')]
    d.text((120, 230), 'Rock', font=font(FONT_M, 38), fill=INK3 + (int(255 * a),), anchor='lm')
    d.text((380, 230), 'For Earth', font=font(FONT_M, 38), fill=INK3 + (int(255 * a),), anchor='lm')
    d.text((1300, 230), 'Hits the Moon', font=font(FONT_M, 38), fill=INK3 + (int(255 * a),), anchor='lm')
    for i, (s, v, r, what) in enumerate(rows):
        b = ease((t - 0.6 - 0.6 * i) / 0.5) * a
        y = 330 + i * 140
        col, sym = VERDICT[v]
        d.text((120, y), s, font=font(FONT_B, 62), fill=INK + (int(255 * b),), anchor='lm')
        d.rounded_rectangle([370, y - 42, 800, y + 42], radius=14, outline=col + (int(255 * b),), width=5)
        d.text((585, y), f'{sym} {v}', font=font(FONT_X, 42), fill=col + (int(255 * b),), anchor='mm')
        d.text((840, y), what, font=font(FONT_M, 46), fill=INK2 + (int(255 * b),), anchor='lm')
        e = every(r['moon_interval_yr']).replace('about once every ', 'every ').replace('about every ', 'every ')
        e = e.replace(': no rocks this big are left near the Earth', '')
        d.text((1300, y), e, font=font(FONT_M, 46), fill=INK + (int(255 * b),), anchor='lm')


def wrap(d, s, f, maxw):
    lines, cur = [], ''
    for w_ in s.split():
        trial = (cur + ' ' + w_).strip()
        if d.textlength(trial, font=f) <= maxw or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w_
    return lines + ([cur] if cur else [])


def list_screen(im, t, dur, title, items):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 200), title, font=font(FONT_B, 80), fill=INK + (int(255 * a),), anchor='mm')
    f = font(FONT_M, 52)
    t0 = 1.0
    y = 340
    for it in items:
        b = ease((t - t0) / 0.5) * a
        d.ellipse([170, y + 20, 194, y + 44], fill=AMBER + (int(255 * b),))
        for ln in wrap(d, it, f, 1500):
            d.text((230, y), ln, font=f, fill=INK + (int(255 * b),), anchor='la')
            y += 68
        y += 44
        t0 += read_time([it], 2.6)


def toll_screen(im, t, dur, deaths, injured, hi):
    a = alpha_in_out(t, dur)
    d = Draw(im, 'RGBA')
    d.text((W / 2, 190), 'The toll, over about six years', font=font(FONT_B, 76), fill=INK + (int(255 * a),), anchor='mm')
    b = ease((t - 0.8) / 0.5) * a
    d.text((W / 2, 390), f'~{millions(round(deaths, -7))} deaths', font=font(FONT_X, 120), fill=(245, 130, 60, int(255 * b)),
           anchor='mm')
    d.text((W / 2, 520), 'about as many people as live in Canada', font=font(FONT_M, 50), fill=INK2 + (int(255 * b),),
           anchor='mm')
    c = ease((t - 2.6) / 0.5) * a
    d.text((W / 2, 680), f'~{millions(round(injured, -7))} injured', font=font(FONT_B, 84), fill=INK + (int(255 * c),),
           anchor='mm')
    e = ease((t - 4.5) / 0.5) * a
    d.text((W / 2, 850), f'Could be far fewer, or up to {millions(round(hi, -7))}: it depends on how big', font=font(FONT_M, 44),
           fill=INK3 + (int(255 * e),), anchor='mm')
    d.text((W / 2, 910), 'the largest pieces are, and where they happen to fall.', font=font(FONT_M, 44),
           fill=INK3 + (int(255 * e),), anchor='mm')


# ------------------------------------------------------------------------------------ render
def render(quality='final', out=None):
    L = ladder()
    seg = build()
    out = out or os.path.join(RENDER, f'waiting_room_{quality}.mp4')
    cache = {}
    results = {}
    total = sum(s[1] for s in seg)
    script = []
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                           '-r', str(FPS), '-i', '-'] +
                          (['-i', os.path.join(ROOT, 'out', 'film_data', 'score_tv.wav'), '-c:a', 'aac', '-b:a', '160k',
                            '-shortest'] if os.path.exists(os.path.join(ROOT, 'out', 'film_data', 'score_tv.wav')) else []) +
                          ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium',
                           '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    clock = 0.0
    playhead = {}
    for kind, dur, p in seg:
        n = int(round(dur * FPS))
        rnd = p.get('rnd')
        text_for_script = None
        bg = p.get('bg')
        if bg is not None:
            shot, start = bg
            if start is None:
                start = playhead.get(shot, 0.0)
            play_ = p.get('play', 1.0 if kind == 'say' else 0.0)
            playhead[shot] = start + n * play_
            bg = (shot, start)
        if kind == 'say':
            text_for_script = ' '.join(p.get('big', [])) or ' '.join(p['lines'])
        elif kind == 'card':
            text_for_script = ' '.join(p['lines'])
        elif kind in ('draw', 'scale'):
            text_for_script = p['script']
        elif kind == 'verdict':
            text_for_script = f'Verdict: {p["v"].lower()}. {p["note"]}.'
            results[rnd] = p['v']
        elif kind == 'fate':
            text_for_script = ('For every 1,000 tonnes of rock blasted out, 996 fall back onto the Moon, 4 escape into '
                               'space, and about 160 kilograms of those hit Earth over the following months and years.')
        elif kind == 'chely':
            text_for_script = ('What is a Chelyabinsk? In 2013 a 20-metre rock exploded over Chelyabinsk in Russia. '
                               'It broke windows across the city and hurt about 1,500 people, mostly with flying glass.')
        elif kind == 'tung':
            text_for_script = ('What is a Tunguska? In 1908 a rock about 50 to 60 metres across exploded over a forest '
                               'in Siberia and flattened 80 million trees, an area bigger than Greater London.')
        elif kind == 'heat':
            text_for_script = ('Compare the heat from the falling rock with the heat from the Sun: only the 100 km rock '
                               'out-heats the Sun.')
        elif kind == 'plan':
            text_for_script = 'Five rounds: 1 metre, 60 metres, 1 kilometre, 10 kilometres, 100 kilometres.'
        elif kind == 'score':
            text_for_script = ('The results: 1 metre and 60 metres, harmless. 1 kilometre, trouble for satellites. '
                               '10 kilometres, danger: tens of millions of deaths. 100 kilometres, catastrophe.')
        elif kind == 'list':
            text_for_script = p['title'] + ' ' + ' '.join(p['items'])
        elif kind == 'toll':
            text_for_script = (f'The toll, over about six years: about {millions(round(p["deaths"], -7))} deaths, about as many '
                               f'people as live in Canada, and {millions(round(p["injured"], -7))} injured. It could be far '
                               f'fewer, or up to {millions(round(p["hi"], -7))}, depending on how big the largest pieces are '
                               'and where they fall.')
        script.append((clock, dur, text_for_script))
        hold_results = dict(results) if kind != 'verdict' else {k: v for k, v in results.items() if k != rnd}
        for f in range(n):
            t = f / FPS
            if bg is not None:
                shot, start = bg
                if shot not in cache:
                    cache = {shot: ShotFrames(shot, quality)}
                sf = cache[shot]
                play = p.get('play', 1.0 if kind == 'say' else 0.0)
                fr = start + t * FPS * play
                z = 1.0
                if fr >= sf.n - 1:
                    over = (fr - (sf.n - 1)) / FPS
                    z = 1.0 + 0.012 * over                        # slow push-in while holding the last frame
                im = push(sf.get(fr), z)
                if shot in ('I1_1m', 'I2_60m') and fr >= 30:
                    ring(im, flash_xy(sf), min(1.0, (fr - 30) / 6.0) * (1.0 if z < 1.0001 else 1.0))
            else:
                im = Image.new('RGB', (W, H), BG)
            if kind == 'say':
                a = alpha_in_out(t, dur)
                if p.get('tag'):
                    dt = Draw(im, 'RGBA')
                    ft = font(FONT_X, 30)
                    tw = dt.textlength(p['tag'], font=ft)
                    dt.rounded_rectangle([W - tw - 100, 120, W - 50, 172], radius=12, fill=(5, 6, 9, int(170 * a)))
                    dt.text((W - 75 - tw / 2, 146), p['tag'], font=ft, fill=AMBER + (int(255 * a),), anchor='mm')
                if p.get('big'):
                    dd = Draw(im, 'RGBA')
                    dd.rectangle([0, 0, W, H], fill=(0, 0, 0, int(150 * a)))
                    center_text(im, p['big'], a, sizes=[110] * len(p['big']))
                else:
                    panel_text(im, p['lines'], a, p['where'])
            elif kind == 'card':
                center_text(im, p['lines'], alpha_in_out(t, dur), sizes=p.get('sizes'), cols=p.get('cols'))
            elif kind == 'draw':
                p['fn'](im, t, dur)
            elif kind == 'scale':
                scale_overlay(im, t, dur, p, fr, quality)
            elif kind == 'verdict':
                d = Draw(im, 'RGBA')
                d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(120 * alpha_in_out(t, dur))))
                stamp(im, p['v'], p['note'], alpha_in_out(t, dur, 0.3, 0.45))
            elif kind == 'fate':
                fate_screen(im, t, dur)
            elif kind == 'chely':
                chely_screen(im, t, dur)
            elif kind == 'tung':
                tung_screen(im, t, dur)
            elif kind == 'heat':
                heat_screen(im, t, dur, L)
            elif kind == 'plan':
                plan_screen(im, t, dur, quality)
            elif kind == 'score':
                score_screen(im, t, dur, L)
            elif kind == 'list':
                list_screen(im, t, dur, p['title'], p['items'])
            elif kind == 'toll':
                toll_screen(im, t, dur, p['deaths'], p['injured'], p['hi'])
            hud(im, rnd, hold_results)
            ff.stdin.write(im.tobytes())
        clock += n / FPS
    ff.stdin.close()
    ff.wait()
    with open(os.path.join(HERE, 'tv_script.md'), 'w') as f:
        f.write('# Waiting-room cut: on-screen text and a narration script\n\n'
                'Every screen, with its start time and how long it stays up. The narration is the on-screen text, '
                'so the film reads the same with the sound off.\n\n| Time | Seconds | On screen / narration |\n|---|---:|---|\n')
        for t0, d_, s in script:
            f.write(f'| {int(t0 // 60)}:{t0 % 60:04.1f} | {d_:.1f} | {s} |\n')
    print('wrote', out, f'{total:.0f} s')
    return seg




def score():
    """Drone; a hit at each impact (deeper for bigger rocks); a chord at each verdict; a soft pulse under the
    Earth shots. Silent TVs lose nothing: the film never depends on the sound."""
    import music as M
    vm, SR = M.vm, M.SR
    seg = build()
    total = sum(round(s[1] * FPS) / FPS for s in seg)
    n = int((total + 3) * SR)
    dry = np.zeros(n)
    for note, g in (('D2', 0.26), ('A2', 0.16)):
        M.add(dry, vm.pad_tone(vm.hz(note), total + 1, fade_in=3.0, fade_out=3.0, seed=5), 0.0, g)
    gain = {'I1_1m': 0.12, 'I2_60m': 0.25, 'I3_1km': 0.5, 'I4_10km': 0.75, 'I5_100km': 1.0}
    chords = {'HARMLESS': ['D4', 'F#4', 'A4'], 'TROUBLE': ['D4', 'G4', 'A4'], 'DANGER': ['D4', 'F4', 'A4'],
              'CATASTROPHE': ['D3', 'Eb3', 'A3']}
    t = 0.0
    for kind, dur, p in seg:
        bg = p.get('bg')
        if kind == 'say' and bg and bg[0] in gain and bg[1] == 0:
            at = t + (30 / 24 if bg[0] in ('I1_1m', 'I2_60m') else 0.1)
            M.add(dry, M.boom(5.0), at, gain[bg[0]])
        if kind == 'verdict':
            for i, note in enumerate(chords[p['v']]):
                M.add(dry, vm.piano(vm.hz(note), 2.0, 0.35, seed=40 + i), t + 0.15 + 0.06 * i, 0.5)
        if kind == 'say' and bg and bg[0].startswith('E') and bg[1] == 0:
            k = {'E3_1km': 0.3, 'E4_10km': 0.5, 'E5_100km': 0.7}[bg[0]]
            M.add(dry, M.pulse(dur + 8, 60, lambda x: k), t, 0.7)
        t += round(dur * FPS) / FPS
    from scipy.signal import fftconvolve
    ir = vm.reverb_ir(rt60=3.0, length=4.0)
    wet = np.stack([fftconvolve(dry, ir[:, c])[:n] for c in range(2)], 1)
    mix = np.stack([dry, dry], 1) * 0.6 + wet * 0.5
    k = int(2.5 * SR)
    mix[-k:] *= np.linspace(1, 0, k)[:, None]
    mix /= max(np.abs(mix).max() / 0.8, 1e-9)
    vm.write(mix.astype(np.float32), os.path.join(ROOT, 'out', 'film_data', 'score_tv.wav'))
    print('score', f'{total:.1f} s')


if __name__ == '__main__':
    q = sys.argv[1] if len(sys.argv) > 1 else 'final'
    score()
    render(q, sys.argv[2] if len(sys.argv) > 2 else None)
