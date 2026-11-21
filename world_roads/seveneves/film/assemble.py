"""Cut the rendered shots together with captions, live counters and title cards.

python3 film/assemble.py <quality> [out.mp4]
Frames come from /opt/seveneves_render/<quality>/<shot>/; missing frames (e.g. an animatic rendered at
every other frame) repeat the previous one, so timing is always the final film's.
Counters are read from the simulation: out/film_data/rain.npz (rain, temperature, oceans) and each
shot's times.json (simulated time of every frame).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W, H, FPS = 1920, 1080, 24
FONT = '/usr/share/fonts/opentype/inter/Inter-Regular.otf'
FONT_M = '/usr/share/fonts/opentype/inter/Inter-Medium.otf'
FONT_B = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
DAY, YEAR = 86400.0, 365.25 * 86400.0
A_EARTH = 5.1e14

F = {}


def font(path, size):
    k = (path, size)
    if k not in F:
        F[k] = ImageFont.truetype(path, size)
    return F[k]


def text(d, xy, s, size=34, fill=(235, 235, 235), path=FONT, anchor='la', alpha=1.0, shadow=True):
    f = font(path, size)
    c = tuple(int(v * alpha) for v in fill)
    if shadow:
        d.text((xy[0] + 2, xy[1] + 2), s, font=f, fill=(0, 0, 0), anchor=anchor)
    d.text(xy, s, font=f, fill=c, anchor=anchor)


def fmt_time(t):
    if t < 60:
        return f'T + {t:.0f} s'
    if t < 3600:
        return f'T + {t / 60:.0f} min'
    if t < 2 * DAY:
        return f'T + {t / 3600:.1f} h'
    if t < YEAR:
        return f'Day {t / DAY:.0f}'
    if t < 20 * YEAR:
        return f'Year {t / YEAR:.1f}'
    return f'Year {t / YEAR:,.0f}'


def sci(x):
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    sup = str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{m:.1f} × 10{sup}'


class Rain:
    def __init__(self):
        r = np.load(os.path.join(ROOT, 'out', 'film_data', 'rain.npz'))
        self.t, self.P, self.T, self.ocean = r['series_t'], r['series_P'], r['series_T'], r['series_ocean']
        dt = np.diff(self.t)
        self.Mcum = np.r_[0, np.cumsum(self.P[:-1] * dt * A_EARTH / (0.5 * 11.1e3 ** 2))]   # kg landed

    def at(self, t):
        i = lambda a: float(np.interp(t, self.t, a))
        # 30-day running mean of the heating, as the climate model sees it
        lo, hi = max(t - 15 * DAY, 0), t + 15 * DAY
        E = np.r_[0, np.cumsum(self.P[:-1] * np.diff(self.t))]
        Pm = (np.interp(hi, self.t, E) - np.interp(lo, self.t, E)) / (hi - lo)
        return dict(P=Pm, T=i(self.T) - 273.15, ocean=i(self.ocean), M=i(self.Mcum))


# ------------------------------------------------------------------ the cut
# (shot, caption lines, counters)
CUT = [
    ('card', 3.5, ['In Neal Stephenson\'s Seveneves, the Moon breaks into seven pieces.',
                   'Two years later, the pieces start falling on Earth.', 'We simulated it.']),
    ('A_moon', ['The Moon', '7.3 × 10²² kg, 384,000 km away'], None),
    ('B_breakup', ['Seven pieces', 'Just enough energy to escape each other\'s gravity',
                   'Fresh interior rock: 1,000–1,600 K'], 'time'),
    ('C_merge', ['They don\'t shatter. They merge.',
                 'Two pieces fall back together at 1.25 km/s and stick'], 'time'),
    ('D_wide', ['The broken Moon and its dust', 'Day 2: the first debris reaches the Earth'], 'time'),
    ('E_rain', ['The rain', 'Thousands of dinosaur-killers\' worth of rock',
                'The upper atmosphere glows red hot'], 'rain'),
    ('F_steam', ['The oceans boil', 'A wet planet can\'t radiate more than ~282 W/m²,',
                 'so the heat stays for thousands of years'], 'steam'),
    ('card', 4.0, ['Could something smaller', 'have done this?']),
    ('H_ladder', ['How big a rock, hitting the Moon at 20 km/s, hurts the Earth?'], 'ladder'),
    ('card', 6.0, ['Simulation: REBOUND N-body (Sun, Earth, fragments), 4,300 debris parcels,',
                   'collision physics for planet-sized bodies, and a steam-table climate model.',
                   'Moon: NASA LRO / LOLA (SVS CGI Moon Kit).  Earth: Solar System Scope, CC BY 4.0.']),
]

LADDER_CUT = [
    ('card', 4.5, ['Could something hitting the Moon', 'cause problems on Earth?', 'How big would it have to be?']),
    ('H_ladder', ['The Moon is hit all the time', 'Here is what happens as the impactor gets bigger'], None),
    ('I1_1m', ['1 metre', 'A flash visible from Earth through a telescope', 'Crater: ~20 m.  Earth: nothing'], ('freq', 1.0)),
    ('I2_60m', ['60 metres (like asteroid 2024 YR4)', 'Crater: 1.2 km.  Bright enough to see by eye',
                'Earth: meteors 2–4× normal for a week'], ('freq', 60.0)),
    ('I3_1km', ['1 kilometre', 'Crater: 18 km.  Rock is thrown out at up to several km/s'], ('crater', 1000.0)),
    ('J_ejecta', ['The debris, over the next four months', 'Most of it flies off into orbit around the Sun',
                  'The orange streaks are the pieces that hit the Earth'], ('time', 1000.0)),
    ('graphic', 11.0, 'fate'),
    ('graphic', 10.0, 'chely'),
    ('E3_1km', ['The Earth, the next four months', 'A meteoroid storm 10,000× normal: satellites at risk',
                'Perhaps up to ~100 Chelyabinsk-size fireballs'], ('freq', 1000.0)),
    ('I4_10km', ['10 kilometres (the dinosaur-killer\'s size)', 'Crater: 135 km'], ('crater', 10000.0)),
    ('graphic', 10.0, 'tung'),
    ('E4_10km', ['The Earth, the next year', '~150,000 Chelyabinsk-size and up to ~10,000 Tunguska-size airbursts',
                 'Yet the climate barely notices (+0.01 °C). Why?'], ('freq', 10000.0)),
    ('K_entry', ['A piece of the Moon hits the air', '11 km/s, 100 km up'], None),
    ('graphic', 13.0, 'heat'),
    ('I5_100km', ['100 kilometres', 'A basin the size of Mare Imbrium'], ('crater', 100000.0)),
    ('E5_100km', ['The Earth, the next three years', 'More heat than the Sun\'s for months: ~11 °C of warming',
                  'A mass extinction'], ('freq', 100000.0)),
    ('summary', 9.0, None),
    ('card', 6.0, ['Ejecta followed from 6,000 launch sites on the Moon with the Sun and the Earth (N-body).',
                   'Craters: Collins et al. 2005.  Ejecta: Housen & Holsapple 2011.  Impact rates: Brown et al. 2002,',
                   'Harris & Chodas 2021.  Moon: NASA LRO/LOLA.  Earth: Solar System Scope, CC BY 4.0.']),
]


def ladder_row(D):
    rows = json.load(open(os.path.join(ROOT, 'results', 'impact_ladder.json')))['rows']
    return [r for r in rows if r['material'] == 'rock' and abs(r['D_m'] - D) / D < 1e-6][0]


def fmt_interval(y):
    if y < 1 / 12:
        return f'every {y * 365.25:.0f} days'
    if y < 1:
        return f'every {y * 12:.0f} months'
    if y < 1e3:
        return f'every {y:,.0f} years'
    if y < 1e6:
        return f'every {y / 1e3:,.0f},000 years'
    if y < 1e9:
        return f'every {y / 1e6:,.0f} million years'
    if y < 13.8e9:
        return f'every {y / 1e9:,.0f} billion years'
    return 'never, in today\'s Solar System'


def ladder_overlay(d, kind, f, n, t, a):
    tag, D = kind
    r = ladder_row(D)
    if tag in ('crater', 'time'):
        text(d, (W - 80, 70), fmt_time(t), 44, path=FONT_M, anchor='ra', alpha=a)
    if tag == 'freq':
        y = H - 150
        text(d, (80, y), 'Hits the Moon', 28, fill=(190, 190, 190), alpha=a)
        text(d, (330, y), fmt_interval(r['moon_interval_yr']), 30, path=FONT_M, fill=(255, 196, 120), alpha=a)
        text(d, (80, y + 50), 'Hits the Earth', 28, fill=(190, 190, 190), alpha=a)
        text(d, (330, y + 50), fmt_interval(r['earth_interval_yr']), 30, path=FONT_M, fill=(255, 196, 120), alpha=a)
        if D >= 1000:
            text(d, (W - 80, 70), fmt_time(t), 44, path=FONT_M, anchor='ra', alpha=a)


def summary_frames(secs):
    rows = [(1.0, '1 m', 'a flash'), (60.0, '60 m', 'meteors 2–4× normal'),
            (1000.0, '1 km', 'satellite-killing meteoroid storm'),
            (10000.0, '10 km', '~150,000 city-shaking fireballs'),
            (100000.0, '100 km', 'mass extinction (+11 °C)')]
    n = int(secs * FPS)
    for f in range(n):
        a = min(1.0, f / 12.0, (n - f) / 12.0)
        im = Image.new('RGB', (W, H), (6, 7, 10))
        d = ImageDraw.Draw(im)
        text(d, (160, 120), 'Hitting the Moon', 54, path=FONT_B, alpha=a, shadow=False)
        text(d, (160, 200), 'Impactor', 26, fill=(150, 150, 150), alpha=a, shadow=False)
        text(d, (420, 200), 'What the Earth gets', 26, fill=(150, 150, 150), alpha=a, shadow=False)
        text(d, (1260, 200), 'How often it hits the Moon', 26, fill=(150, 150, 150), alpha=a, shadow=False)
        for i, (D, ds, eff) in enumerate(rows):
            r = ladder_row(D)
            y = 280 + 120 * i
            hot = D >= 1000
            col = (255, 196, 120) if hot else (225, 225, 225)
            text(d, (160, y), ds, 40, path=FONT_M, fill=col, alpha=a, shadow=False)
            text(d, (420, y), eff, 34, fill=col, alpha=a, shadow=False)
            text(d, (1260, y), fmt_interval(r['moon_interval_yr']), 34, path=FONT_M, fill=col, alpha=a, shadow=False)
        text(d, (160, H - 120), 'Problems start around 1 km. An object that big hits the Moon about once every 14 million years.',
             28, fill=(200, 200, 200), alpha=a, shadow=False)
        yield im


LADDER = [   # diameter km, label (20 km/s, rock-to-regolith range from results/lunar_impact.json)
    (4, '2–5 km', 'A Tunguska blast\'s energy every day for a year'),
    (100, '80–200 km', 'Mass extinction: +10 °C worldwide'),
    (280, '170–400 km', 'The oceans start to boil'),
    (900, '800–1,400 km', 'Every ocean boils away'),
    (3000, '~3,000 km', 'The Moon itself shatters'),
]


def shot_frames(quality, shot):
    d = os.path.join('/opt/seveneves_render', quality, shot)
    times = json.load(open(os.path.join(d, 'times.json'))) if os.path.exists(os.path.join(d, 'times.json')) else {}
    import shots_meta
    n = shots_meta.FRAMES[shot]
    mp4 = os.path.join(ROOT, 'media', 'shots', f'{shot}_{quality}.mp4')
    if not os.path.exists(os.path.join(d, '0001.png')) and os.path.exists(mp4):
        # rebuild from the committed shot video
        times = json.load(open(mp4.replace('.mp4', '_times.json')))
        p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', mp4, '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                              '-s', f'{W}x{H}', '-'], stdout=subprocess.PIPE)
        for f in range(n):
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            yield f, n, Image.frombytes('RGB', (W, H), buf), times.get(str(f), 0.0)
        p.wait()
        return
    last_img, last_t = None, 0.0
    for f in range(n):
        p = os.path.join(d, f'{f + 1:04d}.png')
        if os.path.exists(p):
            last_img = Image.open(p).convert('RGB')
            if last_img.size != (W, H):
                last_img = last_img.resize((W, H), Image.LANCZOS)
            last_t = times.get(str(f), last_t)
        yield f, n, (last_img.copy() if last_img is not None else Image.new('RGB', (W, H))), last_t


def caption(d, lines, a):
    y = 70
    for i, s in enumerate(lines):
        if i == 0:
            text(d, (80, y), s, 54, path=FONT_B, alpha=a)
            y += 78
        else:
            text(d, (80, y), s, 32, fill=(210, 210, 210), alpha=a)
            y += 46


def counters(d, kind, t, rain, a):
    if kind is None:
        return
    text(d, (W - 80, 70), fmt_time(t), 44, path=FONT_M, anchor='ra', alpha=a)
    if kind in ('rain', 'steam'):
        r = rain.at(t)
        rows = []
        if kind == 'rain':
            rows = [('Moon that has landed', f'{sci(max(r["M"], 1e15))} kg'),
                    ('Heat from the sky', f'{r["P"]:,.0f} W/m²  (sunlight: 240)'),
                    ('Oceans boiled', f'{100 * r["ocean"]:.1f}%')]
        else:
            rows = [('Oceans boiled', f'{100 * r["ocean"]:.0f}%'),
                    ('Average surface', f'{r["T"]:.0f} °C')]
        y = H - 80 - 52 * len(rows)
        for k, v in rows:
            text(d, (80, y), k, 28, fill=(190, 190, 190), alpha=a)
            text(d, (520, y), v, 30, path=FONT_M, fill=(255, 196, 120), alpha=a)
            y += 52


def ladder(d, f, n, a):
    # the Moon in H_ladder: 55 mm lens, 7.4 units away -> ~0.52 px per km near the disc centre
    px_km = 0.519
    cx, cy = W // 2, H // 2
    shown = min(len(LADDER), 1 + int(f / (n / len(LADDER))))
    for i, (dkm, rng, what) in enumerate(LADDER[:shown]):
        r = max(dkm * px_km / 2, 2.5)
        fade = min(1.0, (f - i * n / len(LADDER)) / 12.0) * a
        col = (255, int(200 - 30 * i), int(120 - 20 * i))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=tuple(int(c * fade) for c in col), width=4)
        y = H - 80 - 50 * (len(LADDER) - i)
        text(d, (80, y), rng, 30, path=FONT_M, fill=col, alpha=fade)
        text(d, (330, y), what, 30, alpha=fade)


def card(lines, secs):
    n = int(secs * FPS)
    for f in range(n):
        a = min(1.0, f / 12.0, (n - f) / 12.0)
        im = Image.new('RGB', (W, H))
        d = ImageDraw.Draw(im)
        y0 = H / 2 - 30 * len(lines)
        for i, s in enumerate(lines):
            text(d, (W / 2, y0 + 62 * i), s, 40 if len(lines) <= 3 else 32, anchor='mm', alpha=a, shadow=False)
        yield im


def main(quality, out, cut=None):
    sys.path.insert(0, HERE)
    cut = CUT if cut is None else cut
    rain = Rain() if cut is CUT else None
    score = os.path.join(ROOT, 'out', 'film_data', 'score.wav' if cut is CUT else 'score_ladder.wav')
    audio = ['-i', score, '-c:a', 'aac', '-b:a', '192k', '-shortest'] if os.path.exists(score) else []
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-'] + audio +
                          ['-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                           '-crf', '17', '-preset', 'medium', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for item in cut:
        if item[0] == 'card':
            for im in card(item[2], item[1]):
                ff.stdin.write(im.tobytes())
            continue
        if item[0] == 'summary':
            for im in summary_frames(item[1]):
                ff.stdin.write(im.tobytes())
            continue
        if item[0] == 'graphic':
            import graphics
            for im in graphics.GRAPHICS[item[2]](item[1]):
                ff.stdin.write(im.tobytes())
            continue
        shot, lines, kind = item
        for f, n, im, t in shot_frames(quality, shot):
            a = min(1.0, f / 10.0, (n - f) / 10.0)
            d = ImageDraw.Draw(im)
            if isinstance(kind, tuple):
                ca = min(1.0, f / 10.0, (n - f) / 10.0)
                caption(d, lines, ca)
                ladder_overlay(d, kind, f, n, t, a)
            elif kind == 'ladder':
                caption(d, lines, a)
                ladder(d, f, n, a)
            else:
                ca = min(1.0, f / 10.0) * (1.0 if f < n * 0.8 else max(0.0, (n - f) / (n * 0.2)))
                caption(d, lines, ca)
                counters(d, kind, t, rain, a)
            ff.stdin.write(im.tobytes())
    ff.stdin.close()
    ff.wait()
    print('wrote', out)


if __name__ == '__main__':
    q = sys.argv[1] if len(sys.argv) > 1 else 'animatic'
    which = sys.argv[2] if len(sys.argv) > 2 else 'ladder'
    out = sys.argv[3] if len(sys.argv) > 3 else f'/opt/seveneves_render/{which}_{q}.mp4'
    main(q, out, LADDER_CUT if which == 'ladder' else CUT)
