"""Trajectories of one impact's escaping ejecta, for the film's "where does the debris go" shot.

One site on the Moon, one moment; 4,000 test particles with launch speeds drawn from the escaping part of
the Housen & Holsapple (2011) rock distribution for a 20 km/s impact, in a 45-degree cone. Same field
(Sun, Earth, intact Moon) and integrator as lunar_impact.py. Positions saved every 2 hours for 120 days.

python3 ejecta_viz.py   -> out/film_data/ejecta_paths.npz
"""
import math
import os

import numpy as np

import lunar_impact as L
import tracers
from constants import R_MOON, V_ESC_MOON, DAY, YEAR

HERE = os.path.dirname(os.path.abspath(__file__))


def main(n=4000, days=120.0, lat=10.0, lon=90.0, seed=4):   # lon 90 = trailing side
    rng = np.random.default_rng(seed)
    f = np.load(os.path.join(L.OUT, 'field.npz'))
    T, F, SUN = f['T'], f['F'], f['SUN']
    pm, vm = F[0, 0, :3], F[0, 0, 3:6]
    # texture-style frame: x towards the Earth, z = orbit normal
    x = -pm / np.linalg.norm(pm)
    h = np.cross(pm, vm)
    z = h / np.linalg.norm(h)
    y = np.cross(z, x)
    la, lo = math.radians(lat), math.radians(lon)
    site = math.cos(la) * math.cos(lo) * x + math.cos(la) * math.sin(lo) * y + math.sin(la) * z
    # speeds: M(>v) ~ (v / C1 U)^(-3 mu) - n1^3 above escape speed
    hh = L.MATERIALS['rock']
    Uv = 20e3 * math.sin(math.radians(45))
    x_lo = (V_ESC_MOON * 1.001 / (hh['C1'] * Uv)) ** (-3 * hh['mu'])
    xx = hh['n1'] ** 3 + rng.random(n) * (x_lo - hh['n1'] ** 3)
    v = hh['C1'] * Uv * xx ** (-1 / (3 * hh['mu']))
    a = np.cross(site, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(site, a)
    ph = rng.random(n) * 2 * math.pi
    el = np.radians(45 + rng.normal(0, 8, n))
    d = np.cos(el)[:, None] * (np.cos(ph)[:, None] * a + np.sin(ph)[:, None] * b) + np.sin(el)[:, None] * site
    y0 = np.hstack([np.repeat((pm + site * R_MOON * 1.002)[None], n, 0), vm + v[:, None] * d])
    frame_t = np.arange(0, days * DAY + 1, 2 * 3600.0)
    frames = np.full((len(frame_t), n, 3), np.nan, np.float32)
    out = tracers.integrate(np.zeros(n), y0, days * DAY, T, F, SUN, 1e-9, frame_t, frames, 0.0, 0.0, 9)
    fate, t_f = out[0], out[1]
    moon = np.stack([np.interp(frame_t, T, F[:, 0, c]) for c in range(3)], 1)
    sun = np.stack([np.interp(frame_t, T, SUN[:, c]) for c in range(3)], 1)
    np.savez_compressed(os.path.join(HERE, 'out', 'film_data', 'ejecta_paths.npz'), t=frame_t,
                        P=(frames * 1e-6).astype(np.float32), fate=fate, t_f=t_f, v=v,
                        moon=moon * 1e-6, sun=sun * 1e-6, site=site)
    print({k: f'{np.mean(fate == i):.1%}' for i, k in enumerate(['orbiting', 'Earth', 'Moon', 'left'])},
          'within', days, 'days')


if __name__ == '__main__':
    main()
