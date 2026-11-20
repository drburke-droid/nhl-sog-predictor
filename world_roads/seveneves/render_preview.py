"""Preview animation: the Earth-Moon system seen from above the ecliptic, first few years.

python3 render_preview.py <scenario> [years=3] [fps=30] [step_days=2]
Writes out/<scenario>/preview.mp4. Fragment and Earth sizes are exaggerated (labelled on screen).
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from constants import DAY, YEAR, R_EARTH, A_MOON, M_MOON, radius_of

W, H = 1920, 1080
FONT = '/usr/share/fonts/opentype/inter/Inter-Regular.otf'
FONT_B = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
SIZE_X = 8.0      # exaggeration of body sizes


def main(name, years=3.0, fps=30, step_days=2.0):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', name)
    bod = np.load(os.path.join(d, 'bodies.npz'))
    fr = np.load(os.path.join(d, 'frames_gen0.npz'))
    rain = np.load(os.path.join(d, 'rain.npz'))
    g0 = rain['gen'] == 0
    fate, tf, m = rain['fate'][g0], rain['t_f'][g0], rain['m'][g0]
    t_frames = fr['t']
    P = fr['P']
    bt, B = bod['t'], bod['B']
    scale = 430 / A_MOON          # px per metre
    cx, cy = W // 2 + 260, H // 2
    f_big = ImageFont.truetype(FONT_B, 44)
    f_med = ImageFont.truetype(FONT, 30)
    f_sm = ImageFont.truetype(FONT, 22)
    times = np.arange(0, years * YEAR, step_days * DAY)
    out = os.path.join(d, 'preview.mp4')
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                           '-crf', '18', '-preset', 'medium', out], stdin=subprocess.PIPE)
    acc = np.zeros((H, W), np.float32)                 # persistent faint trail of debris
    rng = np.random.default_rng(0)
    stars = (rng.random((900, 2)) * [W, H]).astype(int)
    hit_mass = 0.0
    hit_order = np.argsort(tf[fate == 1])
    hit_t = np.sort(tf[fate == 1])
    hit_m = m[fate == 1][hit_order]
    for k, t in enumerate(times):
        img = np.zeros((H, W, 3), np.float32)
        img[stars[:, 1], stars[:, 0]] = 0.25
        # debris
        j = min(np.searchsorted(t_frames, t), len(t_frames) - 1)
        p = P[j]
        ok = ~np.isnan(p[:, 0])
        x = (cx + p[ok, 0] * scale).astype(int)
        y = (cy - p[ok, 1] * scale).astype(int)
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        acc *= 0.85
        np.add.at(acc, (y[inb], x[inb]), 0.6)
        glow = np.minimum(acc + 0.8 * gaussian_filter(acc, 1.2), 1.5)
        img[..., 0] += glow * 0.85
        img[..., 1] += glow * 0.88
        img[..., 2] += glow * 0.95
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        # original lunar orbit
        r0 = A_MOON * scale
        dr.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], outline=(55, 70, 95), width=1)
        # Earth
        re = R_EARTH * scale * SIZE_X
        dr.ellipse([cx - re, cy - re, cx + re, cy + re], fill=(40, 110, 220))
        # flash for recent Earth hits
        n_new = np.searchsorted(hit_t, t) - np.searchsorted(hit_t, t - step_days * DAY)
        hit_mass = hit_m[:np.searchsorted(hit_t, t)].sum()
        if n_new:
            rr = re + 6 + min(n_new, 40)
            dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 160, 60), width=3)
        # fragments
        i = min(np.searchsorted(bt, t), len(bt) - 1)
        b = B[i]
        b = b[~np.isnan(b[:, 3])]
        for row in b:
            r = max(radius_of(row[3]) * scale * SIZE_X, 1.5)
            px, py = cx + row[0] * scale, cy - row[1] * scale
            dr.ellipse([px - r, py - r, px + r, py + r], fill=(205, 200, 190))
        # captions
        dr.text((70, 60), 'Seveneves simulator', font=f_big, fill=(235, 235, 235))
        days = t / DAY
        dr.text((70, 130), f'Day {days:,.0f}' + (f'   (year {t / YEAR:.1f})' if days > 365 else ''),
                font=f_med, fill=(210, 210, 210))
        dr.text((70, 190), f'Fragments in orbit: {len(b)}', font=f_med, fill=(200, 200, 200))
        dr.text((70, 235), f'Largest: {b[:, 3].max() / M_MOON:.0%} of the Moon' if len(b) else '',
                font=f_med, fill=(200, 200, 200))
        dr.text((70, 300), f'Debris that has hit Earth:', font=f_med, fill=(255, 180, 90))
        dr.text((70, 345), f'{hit_mass:.2e} kg', font=f_big, fill=(255, 180, 90))
        dr.text((70, 405), f'= {hit_mass / 1e15:,.0f} dinosaur-killer asteroids by mass', font=f_sm, fill=(230, 160, 90))
        dr.text((70, H - 70), f'View from above the ecliptic. Earth and fragment sizes x{SIZE_X:.0f}. '
                              f'Circle: the Moon\'s old orbit.', font=f_sm, fill=(140, 140, 150))
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close()
    ff.wait()
    print('wrote', out)


if __name__ == '__main__':
    kw = dict(a.split('=') for a in sys.argv[2:])
    main(sys.argv[1], **{k: float(v) for k, v in kw.items()})
