"""Collect the small numbers behind the findings page into results/results.json (committed)."""
import json
import os

import numpy as np

import analyze
import collisions
from constants import M_MOON, YEAR, DAY, G, GM_EARTH, A_MOON, radius_of

HERE = os.path.dirname(os.path.abspath(__file__))
SPEEDS = [50, 150, 300, 600, 1200]
DIAL = [0.0, 0.001, 0.003, 0.01, 0.02, 0.05]


def downsample(series):
    t = series['t'] / YEAR
    pts = np.unique(np.r_[np.geomspace(1 / 365.25, t[-1], 260), t[-1]])
    idx = np.clip(np.searchsorted(t, pts), 0, len(t) - 1)
    # rain power: mean over a window of +-0.1 in log10(time) around each point (a running mean that
    # keeps the energy honest and doesn't drop to zero between individual parcels)
    P = series['P']
    ts = series['t']
    E = np.r_[0, np.cumsum(P[:-1] * np.diff(ts))]
    tt = t[idx][1:] * YEAR
    lo, hi = tt * 10 ** -0.1, np.minimum(tt * 10 ** 0.1, ts[-1])
    Pm = (np.interp(hi, ts, E) - np.interp(lo, ts, E)) / (hi - lo)
    return dict(t=[round(float(x), 5) for x in t[idx][1:]],
                P=[round(float(x), 2) for x in Pm],
                T=[round(float(x) - 273.15, 1) for x in series['Ts'][idx][1:]],
                ocean=[round(float(x), 4) for x in series['ocean'][idx][1:]])


def main():
    out = dict()
    ens = json.load(open(os.path.join(HERE, 'out', 'ensemble.json')))
    out['ensemble'] = []
    for v in SPEEDS:
        r = [x for x in ens if x['v'] == v]
        out['ensemble'].append(dict(
            v=v, runs=len(r), ke_over_binding=float(np.mean([x['ke_over_binding'] for x in r])),
            orbiting=float(np.mean([x['orbiting'] for x in r])), largest=float(np.mean([x['largest'] for x in r])),
            escaped=float(np.mean([x['escaped'] for x in r])), earth=float(np.mean([x['earth'] for x in r])),
            runs_with_fragment_impact=sum(1 for x in r if x['earth_impacts']),
            collisions=float(np.mean([x['n_collisions'] for x in r])),
            regimes={k: sum(x['regimes'][k] for x in r) for k in r[0]['regimes']},
            impacts=[dict(seed=x['seed'], year=round(a, 2), moon_frac=b, energy=c) for x in r for a, b, c in x['earth_impacts']]))
    out['scenarios'] = {}
    for v in SPEEDS:
        d = os.path.join(HERE, 'out', f'v{v}')
        if not os.path.exists(os.path.join(d, 'rain.npz')):
            continue
        s = analyze.analyse(d, save=True)
        ser = s.pop('_series')
        rain = np.load(os.path.join(d, 'rain.npz'))
        g = rain['gen']
        hit_frag = float(rain['m'][(g == 0) & (rain['fate'] == 2)].sum())
        ejecta = float(rain['m'][g == 1].sum())
        dial = []
        for d0 in DIAL:
            x = analyze.analyse(d, debris0=d0, save=False)
            dial.append(dict(debris0=d0, rain_moon_frac=x['rain']['mass_frac_of_moon'],
                             boil=x['rain']['ocean_boil_energies'], peak_C=x['climate']['peak_T_C'],
                             ocean=x['climate']['max_ocean_boiled'], years50=x['climate']['years_above_50C'],
                             years100=x['climate']['years_above_100C'],
                             chicxulubs=x['rain']['chicxulub_size_impacts']))
        x0 = analyze.analyse(d, debris0=0.0, save=False)
        out['scenarios'][v] = dict(summary=s, series=downsample(ser), series_no_breakup_debris=downsample(x0['_series']),
                                   cascade=dict(hit_fragments_kg=hit_frag, escaping_ejecta_kg=ejecta,
                                                multiplier=ejecta / hit_frag if hit_frag else 0.0),
                                   dial=dial)
    # the most violent collision two equal pieces can have while both orbit the Earth at lunar distance
    m = M_MOON / 7
    R = radius_of(m)
    vesc = (2 * G * 2 * m / (2 * R)) ** 0.5
    vmax = 2 * (2 * GM_EARTH / A_MOON) ** 0.5
    rows = []
    for vr in [150, 600, 1000, vmax]:
        vi = (vr ** 2 + vesc ** 2) ** 0.5
        qr = 0.5 * (m / 2) * vi ** 2 / (2 * m)
        row = dict(v_rel=vr, v_imp=vi)
        for cs, mb, key in [(5.0, 0.37, 'planet'), (1.9, 0.36, 'rock')]:
            qs = collisions.qstar_rd(2 * m, 1.0, 1.0, cs, mb)
            row[key] = collisions.largest_remnant(2 * m, qr, qs) / (2 * m)
        rows.append(row)
    qs = collisions.qstar_rd(2 * m, 1.0, 1.0, 5.0, 0.37)
    shatter = (8 * qs) ** 0.5          # head-on impact speed where Q_R = Q*_RD (largest remnant = half)
    out['max_collision'] = dict(piece_kg=m, piece_radius_km=R / 1e3, mutual_vesc=vesc, max_encounter=vmax, rows=rows,
                                shatter_speed=shatter)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'results.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('wrote results/results.json:', list(out['scenarios']))


if __name__ == '__main__':
    main()
