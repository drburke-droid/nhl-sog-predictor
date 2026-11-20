"""Fragment-only ensemble: the same breakup with different random piece shapes (seeds), for each
dispersal speed. Debris is not followed here (that is `tracers.py`); this tells us how robust the
fate of the big pieces is.

python3 ensemble.py [years=50] [seeds=1,2,3,4,5,6] [speeds=50,150,300,600,1200]
Writes out/ensemble.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

import nbody
from constants import M_MOON, YEAR

HERE = os.path.dirname(os.path.abspath(__file__))


def one(args):
    v, seed, years = args
    name = f'ens_v{v}_s{seed}'
    out = os.path.join(HERE, 'out', 'ensemble', name)
    cfg = dict(nbody.DEFAULTS, v_disperse=float(v), seed=seed, years=years, debris0=0.0)
    fs = nbody.FragmentSim(cfg, out)
    fs.build()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        fs.run()
    ev = fs.events
    left = [p.m for p in fs.sim.particles[2:]]
    coll = [e for e in ev if e['kind'] == 'collision']
    res = dict(v=v, seed=seed, pieces=fs.breakup['pieces_frac_of_moon'],
               ke_over_binding=fs.breakup['ke_over_moon_binding'],
               orbiting=sum(left) / M_MOON, largest=max(left) / M_MOON if left else 0.0, n_left=len(left),
               escaped=sum(e['m'] for e in ev if e['kind'] == 'escaped') / M_MOON,
               earth=sum(e['m'] for e in ev if e['kind'] == 'earth_impact') / M_MOON,
               earth_impacts=[(e['t'] / YEAR, e['m'] / M_MOON, e['energy']) for e in ev if e['kind'] == 'earth_impact'],
               n_collisions=len(coll), collision_debris=sum(e['debris'] for e in coll) / M_MOON,
               regimes={r: sum(e['regime'] == r for e in coll) for r in ('merge', 'erode', 'supercatastrophic', 'hit-and-run')})
    for f in ('field.npz',):
        p = os.path.join(out, f)
        if os.path.exists(p):
            os.remove(p)
    return res


if __name__ == '__main__':
    kw = dict(a.split('=') for a in sys.argv[1:])
    years = float(kw.get('years', 50))
    seeds = [int(s) for s in kw.get('seeds', '1,2,3,4,5,6').split(',')]
    speeds = [int(s) for s in kw.get('speeds', '50,150,300,600,1200').split(',')]
    jobs = [(v, s, years) for v in speeds for s in seeds]
    with Pool(int(kw.get('procs', 2))) as p:
        res = p.map(one, jobs, chunksize=1)
    with open(os.path.join(HERE, 'out', 'ensemble.json'), 'w') as f:
        json.dump(res, f, indent=1)
    for v in speeds:
        r = [x for x in res if x['v'] == v]
        print(f'{v:5d} m/s: orbiting {np.mean([x["orbiting"] for x in r]):.0%} (largest {np.mean([x["largest"] for x in r]):.0%}), '
              f'escaped {np.mean([x["escaped"] for x in r]):.0%}, hit Earth {np.mean([x["earth"] for x in r]):.1%} '
              f'(runs with a fragment impact: {sum(1 for x in r if x["earth"] > 0)}/{len(r)}), '
              f'collisions {np.mean([x["n_collisions"] for x in r]):.1f}, collision debris {np.mean([x["collision_debris"] for x in r]):.2%}')
