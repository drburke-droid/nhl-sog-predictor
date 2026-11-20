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


def main(quality, out):
    sys.path.insert(0, HERE)
    rain = Rain()
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                           '-crf', '17', '-preset', 'medium', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for item in CUT:
        if item[0] == 'card':
            for im in card(item[2], item[1]):
                ff.stdin.write(im.tobytes())
            continue
        shot, lines, kind = item
        for f, n, im, t in shot_frames(quality, shot):
            a = min(1.0, f / 10.0, (n - f) / 10.0)
            d = ImageDraw.Draw(im)
            if kind == 'ladder':
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
    out = sys.argv[2] if len(sys.argv) > 2 else f'/opt/seveneves_render/seveneves_{q}.mp4'
    main(q, out)
