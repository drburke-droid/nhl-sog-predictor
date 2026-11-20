"""Hit the (intact) Moon with ever bigger objects; what happens to the Earth, and how often does it happen?

For each impactor diameter D (stony, 2,600 kg/m3, 20 km/s, 45 degrees):
  * crater on the Moon       Collins, Melosh & Marcus (2005) transient-crater scaling, lunar gravity,
                             simple/complex transition at 18 km; formation time ~ sqrt(D_tc / g)
  * mass leaving the Moon    Housen & Holsapple (2011) ejecta scaling, rock and regolith (lunar_impact.py)
  * reaching the Earth, when the 6,000 simulated ejecta trajectories (out/lunar_impact/ejecta.npz)
  * on the Earth             meteoroid storm (mass arriving per day vs the normal ~110 t/day, Love &
                             Brownlee 1993: the satellite hazard); fireball energy; largest fragments
                             (spall scaling: ~1% of the impactor, range 0.3-3%); counts of Chelyabinsk-
                             (>= 20 m) and Tunguska-class (>= 50 m) fragments; global warming and
                             ocean boiling from the climate model
  * how often                average interval between impacts of this size on the Earth (Brown et al.
                             2002 bolide flux below 100 m; Harris & Chodas 2021 near-Earth asteroid
                             population above), and on the Moon (about 1/20 of the Earth's rate)

python3 impact_ladder.py  -> results/impact_ladder.json and a printed table
"""
import json
import math
import os

import numpy as np

import lunar_impact as L
from constants import DAY, YEAR, M_MOON

HERE = os.path.dirname(os.path.abspath(__file__))
U = 20e3
ANGLE = 45.0
RHO_I, RHO_T = 2600.0, 2800.0
G_MOON = 1.62
D_SC = 18e3                      # simple-to-complex crater transition on the Moon (m)
BACKGROUND_KG_DAY = 110e3        # Love & Brownlee (1993): ~40,000 t/yr of meteoroids reach the Earth
MOON_RATE = 1 / 20.0             # the Moon's share of the Earth's impact rate (Le Feuvre & Wieczorek 2011)
KT = 4.184e12
SIZES = [1, 10, 60, 140, 1e3, 1e4, 3e4, 6e4, 1e5, 2e5, 4e5, 1e6]   # m


def crater(D):
    """Final rim-to-rim crater diameter and depth on the Moon (m), and its formation time (s)."""
    dtc = 1.161 * (RHO_I / RHO_T) ** (1 / 3) * D ** 0.78 * U ** 0.44 * G_MOON ** -0.22 * \
        math.sin(math.radians(ANGLE)) ** (1 / 3)
    if 1.25 * dtc < D_SC:
        # small impacts are limited by the strength of the regolith/rock, not gravity: craters ~20x the
        # impactor (LRO's new craters from ~1 m impactors are 10-20 m across)
        dfr = min(1.25 * dtc, 20.0 * D)
        dtc = dfr / 1.25
        depth = 0.2 * dfr
    else:
        dfr = 1.17 * dtc ** 1.13 / D_SC ** 0.13
        depth = 1.04e3 * (dfr / 1e3) ** 0.30            # lunar complex craters (Pike 1977)
    return dfr, depth, math.sqrt(dtc / G_MOON)


def earth_interval_years(D):
    """Average years between impacts of objects >= D on the Earth."""
    if D <= 100:
        m = math.pi / 6 * RHO_I * D ** 3
        e_kt = 0.5 * m * U ** 2 / KT
        return 1.0 / (3.7 * e_kt ** -0.90)              # Brown et al. (2002), N(>E) per year
    anchors = [(100, earth_interval_years(100)), (1e3, 7e5), (1e4, 1e8)]   # Harris & Chodas (2021)
    xs = np.log10([a[0] for a in anchors])
    ys = np.log10([a[1] for a in anchors])
    x = math.log10(D)
    if x <= xs[-1]:
        return 10 ** float(np.interp(x, xs, ys))
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return 10 ** (ys[-1] + slope * (x - xs[-1]))


def fragments(M_earth, D, frac, d_min=1e-5):
    """Fragments >= 20 m and >= 50 m expected to reach the Earth. The arriving mass is spread evenly over
    log(size) from 10 microns (dust) up to the largest escaping fragment, frac x D (spall scaling,
    Melosh 1984: ~1% of the impactor, range 0.3-3%). Order of magnitude only."""
    out = {}
    d_max = frac * D
    lam = math.log(d_max / d_min) if d_max > d_min else 1.0
    k = 6 / (math.pi * RHO_T)
    for d in (20, 50):
        out[d] = M_earth / lam * k * (d ** -3 - d_max ** -3) / 3 if d < d_max else 0.0
    return out


def main():
    rows = []
    for mat in ('rock', 'regolith'):
        dl = L.delivery(U, mat)
        t, w = dl['t'], dl['w']
        order = np.argsort(t)
        for D in SIZES:
            m = L.impactor_mass(D)
            M_esc, qrat = L.escaping_mass(D, U, mat)
            M_e = M_esc * (dl['frac_in_run'] + dl['tail_mass'])
            # mass arriving per day: densest 7-day window
            arr = t[order] / DAY
            cum = np.cumsum(w[order]) * M_esc
            best = 0.0
            for i in range(len(arr)):
                j = np.searchsorted(arr, arr[i] + 7.0, side='right') - 1
                best = max(best, cum[j] - (cum[i - 1] if i else 0.0))
            storm = best / 7.0 / BACKGROUND_KG_DAY
            eff = L.earth_effects(M_esc, dl)
            dc, depth, tf = crater(D)
            frag = {f: fragments(M_e, D, f) for f in (0.003, 0.01, 0.03)}
            j1 = 0.0
            for i in range(len(arr)):
                j = np.searchsorted(arr, arr[i] + 1.0, side='right') - 1
                j1 = max(j1, cum[j] - (cum[i - 1] if i else 0.0))
            rows.append(dict(material=mat, D_m=D, impactor_kg=m, impactor_J=0.5 * m * U ** 2,
                             impactor_Mt=0.5 * m * U ** 2 / (1e3 * KT), crater_km=dc / 1e3, crater_depth_km=depth / 1e3,
                             crater_time_s=tf, escaping_kg=M_esc, to_earth_kg=M_e, first_week_kg=best,
                             storm_factor=storm, storm_day_factor=j1 / BACKGROUND_KG_DAY, energy_J=eff['E_total'], first_year_J=eff['E_first_year'],
                             peak_month_W=eff['peak_month_W'], warming_C=eff['warming_C'], ocean=eff['ocean'],
                             moon_q=qrat, largest_fragment_m=0.01 * D,
                             chelyabinsks=frag[0.01][20], tunguskas=frag[0.01][50],
                             chelyabinsks_range=(frag[0.003][20], frag[0.03][20]),
                             tunguskas_range=(frag[0.003][50], frag[0.03][50]),
                             earth_interval_yr=earth_interval_years(D),
                             moon_interval_yr=earth_interval_years(D) / MOON_RATE))
    out = dict(speed=U, angle=ANGLE, background_kg_day=BACKGROUND_KG_DAY, moon_rate=MOON_RATE, rows=rows)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'impact_ladder.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    def yrs(x):
        if x < 1:
            return f'{x * 365.25:.0f} days'
        if x < 1e3:
            return f'{x:.0f} yr'
        if x < 1e6:
            return f'{x / 1e3:.0f} thousand yr'
        if x < 1e9:
            return f'{x / 1e6:.0f} million yr'
        return f'{x / 1e9:.0f} billion yr'
    print(f'{"D":>8} {"crater":>9} {"to Earth":>10} {"week x":>9} {"Chelyab.":>9} {"Tungus.":>9} {"warming":>8} '
          f'{"oceans":>7}  {"Earth hit every":>18} {"Moon hit every":>18}')
    for r in rows:
        if r['material'] != 'rock':
            continue
        D = r['D_m']
        ds = f'{D:.0f} m' if D < 1e3 else f'{D / 1e3:.0f} km'
        print(f'{ds:>8} {r["crater_km"]:8.2f}k {r["to_earth_kg"]:10.2e} {r["storm_factor"]:9.2g} {r["chelyabinsks"]:9.2g} '
              f'{r["tunguskas"]:9.2g} {r["warming_C"]:7.2f}C {100 * r["ocean"]:6.1f}%  {yrs(r["earth_interval_yr"]):>18} '
              f'{yrs(r["moon_interval_yr"]):>18}')


if __name__ == '__main__':
    main()
