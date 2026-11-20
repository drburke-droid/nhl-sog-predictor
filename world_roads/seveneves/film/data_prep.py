"""Turn simulation output into compact per-shot arrays for Blender (out/film_data/*.npz).

Units out: Blender units (1 = 1,000 km), seconds of simulated time.
Sources:
  out/v150_fine   first 12 days at 10-minute cadence (fragments + every debris parcel)
  out/v150        the full 50-year run (Earth impacts: when and where each parcel arrives)
  out/v150/series.npz  rain power, temperature and ocean boiled (climate model)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'out', 'film_data')
U = 1e-6          # metres -> Blender units


def main():
    os.makedirs(OUT, exist_ok=True)
    fine = os.path.join(ROOT, 'out', os.environ.get('FILM_RUN', 'v150_film'))
    b = np.load(os.path.join(fine, 'bodies.npz'))
    t_b, B = b['t'], b['B']                       # B[frame, slot, (x, y, z, m, id)]
    ev = json.load(open(os.path.join(fine, 'events.json')))
    f = np.load(os.path.join(fine, 'field.npz'))
    T, SUN = f['T'], f['SUN']
    sun_at = np.stack([np.interp(t_b, T, SUN[:, c]) for c in range(3)], 1)
    ids = sorted({int(i) for i in B[..., 4][~np.isnan(B[..., 4])]})
    pos = np.full((len(t_b), max(ids) + 1, 3), np.nan, np.float64)
    mass = np.full((len(t_b), max(ids) + 1), np.nan)
    for k in range(len(t_b)):
        row = B[k]
        ok = ~np.isnan(row[:, 4])
        for r in row[ok]:
            pos[k, int(r[4])] = r[:3] * U
            mass[k, int(r[4])] = r[3]
    merges = [dict(t=e['t'], ids=e['ids'], new=e['new']) for e in ev['events'] if e['kind'] == 'collision']
    # debris parcels: positions every 10 minutes (frames_gen0), with each parcel's birth time and mass
    fr = np.load(os.path.join(fine, 'frames_gen0.npz'))
    tr0 = np.load(os.path.join(fine, 'tracers0.npz'))
    rain_f = np.load(os.path.join(fine, 'rain.npz'))
    n0 = int((rain_f['gen'] == 0).sum())
    P = (fr['P'][:, :n0] * U).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, 'fine.npz'), t=t_b, pos=pos, mass=mass, sun=sun_at * U,
                        debris=P, debris_t=fr['t'], debris_born=tr0['t'][:n0], debris_src=tr0['src'][:n0],
                        merges=json.dumps(merges))
    # Earth impacts from the full run (time, unit position at 100 km altitude, mass)
    full = os.path.join(ROOT, 'out', 'v150')
    rain = np.load(os.path.join(full, 'rain.npz'))
    hit = (rain['fate'] == 1) & (rain['m'] > 0)
    nrm = rain['nrm'][hit]
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    s = np.load(os.path.join(full, 'series.npz'))
    np.savez_compressed(os.path.join(OUT, 'rain.npz'), t=rain['t_f'][hit], dir=nrm, m=rain['m'][hit],
                        v=rain['vimp'][hit], series_t=s['t'], series_P=s['P'], series_T=s['Ts'],
                        series_ocean=s['ocean'])
    # the bodies over 50 years (daily) for the late shots
    bl = np.load(os.path.join(full, 'bodies.npz'))
    np.savez_compressed(os.path.join(OUT, 'bodies_long.npz'), t=bl['t'], B=bl['B'])
    print('film data:', P.shape, 'debris frames;', int(hit.sum()), 'Earth arrivals;', len(merges), 'merges')


if __name__ == '__main__':
    main()
