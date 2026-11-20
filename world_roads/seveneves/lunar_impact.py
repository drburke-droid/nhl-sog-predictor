"""How small can the thing that hits the Moon be, and still hurt the Earth?

An asteroid of diameter D hits the intact Moon at speed U (default 20 km/s, at 45 degrees). Ejecta
faster than the Moon's escape speed leave it (Housen & Holsapple 2011 crater-ejecta scaling, for
competent rock and for regolith/sand). For impactors big enough to erode the Moon as a whole, the
Leinhardt & Stewart (2012) law takes over, and above Q*_RD the Moon shatters (the Seveneves case).

Where the ejecta go is computed, not assumed: thousands of test particles are launched from random
sites on the real Moon (with the Sun and the Earth) and followed for `years`. Their launch speeds are
spread evenly in log(v) and re-weighted afterwards to each scenario's ejecta speed distribution, so one
run serves every impactor size, speed and material.

python3 lunar_impact.py field        # REBOUND: Sun, Earth, intact Moon (~1 min)
python3 lunar_impact.py tracers      # ejecta test particles (~15 min on 4 cores)
python3 lunar_impact.py ladder       # thresholds -> results/lunar_impact.json
"""
import json
import math
import os
import sys

import numpy as np
import rebound

from constants import (G, M_SUN, M_EARTH, R_ENTRY, M_MOON, R_MOON, AU, DAY, YEAR, A_MOON, E_MOON, I_MOON,
                       E_EARTH, V_ESC_MOON, A_EARTH)
import collisions
import climate
import tracers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'lunar_impact')
RHO_IMP = 2600.0                 # stony asteroid
CHICXULUB_J = 4.2e23
TUNGUSKA_J = 15 * 4.184e15          # 15 megatons of TNT
MATERIALS = {   # Housen & Holsapple 2011, Table 3
    'rock': dict(mu=0.55, C1=1.5, k=0.2, n1=1.2),
    'regolith': dict(mu=0.41, C1=0.55, k=0.3, n1=1.2),
}
V_LO, V_HI = V_ESC_MOON * 1.0005, 16e3


def make_field(years=30.0, dt=0.1 * DAY):
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = 'ias15'
    sim.add(m=M_SUN)
    sim.add(m=M_EARTH, a=AU, e=E_EARTH, primary=sim.particles[0])
    sim.add(m=M_MOON, a=A_MOON, e=E_MOON, inc=I_MOON, Omega=0.3, omega=1.1, f=2.0, primary=sim.particles[1])
    sim.move_to_com()
    K = int(years * YEAR / dt) + 1
    T = np.zeros(K)
    F = np.zeros((K, 1, 8))
    S = np.zeros((K, 6))
    for k in range(K):
        if k:
            sim.integrate(k * dt, exact_finish_time=1)
        s, e, m = sim.particles[0], sim.particles[1], sim.particles[2]
        ex, ev = np.array(e.xyz), np.array(e.vxyz)
        T[k] = sim.t
        F[k, 0, :3] = np.array(m.xyz) - ex
        F[k, 0, 3:6] = np.array(m.vxyz) - ev
        F[k, 0, 6] = G * M_MOON
        F[k, 0, 7] = R_MOON
        S[k, :3] = np.array(s.xyz) - ex
        S[k, 3:] = np.array(s.vxyz) - ev
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 'field.npz'), T=T, F=F, SUN=S)
    print('field:', K, 'samples')


def launch(n=6000, seed=11, t_spread_days=27.3):
    """Ejecta from random sites, random launch times over one lunar month (so the Moon's orbital
    phase relative to the Sun is sampled too). Speeds log-uniform in [V_LO, V_HI]."""
    rng = np.random.default_rng(seed)
    f = np.load(os.path.join(OUT, 'field.npz'))
    T, F = f['T'], f['F']
    t0 = rng.random(n) * t_spread_days * DAY
    k = np.searchsorted(T, t0)
    pm, vm = F[k, 0, :3], F[k, 0, 3:6]
    site = collisions.random_dirs(n, rng)
    # 45 degree ejecta cone around the local normal
    a = np.cross(site, np.where(np.abs(site[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]]))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(site, a)
    ph = rng.random(n) * 2 * math.pi
    d = math.cos(math.radians(45)) * site + math.sin(math.radians(45)) * (np.cos(ph)[:, None] * a + np.sin(ph)[:, None] * b)
    v = np.exp(rng.uniform(math.log(V_LO), math.log(V_HI), n))
    x0 = pm + R_MOON * 1.002 * site
    # trailing-ness: how much the site faces away from the Moon's geocentric motion
    trail = -(site * (vm / np.linalg.norm(vm, axis=1, keepdims=True))).sum(1)
    t0 = T[k]
    return t0, np.hstack([x0, vm + v[:, None] * d]), v, trail


def run_tracers(n=6000, years=30.0, rtol=1e-9):
    f = np.load(os.path.join(OUT, 'field.npz'))
    T, F, SUN = f['T'], f['F'], f['SUN']
    t0, y0, v, trail = launch(n)
    t_end = min(years * YEAR, T[-1])
    frames = np.zeros((0, n, 3), np.float32)
    out = tracers.integrate(t0, y0, t_end, T, F, SUN, rtol, np.zeros(0), frames, 0.0, 0.0, 3)
    fate, t_f, vimp = out[0], out[1], out[2]
    np.savez(os.path.join(OUT, 'ejecta.npz'), t0=t0, v=v, trail=trail, fate=fate, t_f=t_f, vimp=vimp, t_end=t_end)
    for name, sel in [('all sites', np.ones(n, bool)), ('trailing side', trail > 0.7)]:
        print(f'{name}: Earth {np.mean(fate[sel] == 1):.1%}, Moon {np.mean(fate[sel] == 2):.1%}, '
              f'escaped {np.mean(fate[sel] == 3):.1%}, orbiting {np.mean(fate[sel] == 0):.1%}')


# ------------------------------------------------------------------ physics of the impact
def impactor_mass(D):
    return math.pi / 6 * RHO_IMP * D ** 3


def escaping_mass(D, U, mat, angle=45.0):
    """Mass leaving the Moon (kg): crater ejecta (HH 2011) or whole-Moon erosion (L&S 2012), whichever is
    larger. Also returns the L&S ratio Q_R/Q*_RD (>= 1 means the Moon is shattered)."""
    m = impactor_mass(D)
    h = MATERIALS[mat]
    Uv = U * math.sin(math.radians(angle))
    x = (V_ESC_MOON / (h['C1'] * Uv)) ** (-3 * h['mu'])
    crater = m * max(3 * h['k'] / (4 * math.pi) * (x - h['n1'] ** 3), 0.0)
    M = M_MOON + m
    gamma = m / M_MOON
    mu = m * M_MOON / M
    q_r = 0.5 * mu * U ** 2 / M
    R_imp = D / 2
    alpha = collisions.interacting_fraction(R_MOON, R_imp, math.sin(math.radians(angle)))
    mu_a = alpha * m * M_MOON / (alpha * m + M_MOON)
    qs = collisions.qstar_rd(M, gamma, mu / mu_a, 5.0, 0.37)
    m_lr = collisions.largest_remnant(M, q_r, qs)
    return min(max(crater, M - m_lr), M_MOON), q_r / qs


def speed_weights(v, U, mat, angle=45.0):
    """Mass per log-speed of escaping ejecta at launch speed v (relative weights)."""
    h = MATERIALS[mat]
    Uv = U * math.sin(math.radians(angle))
    vmax = h['C1'] * Uv * h['n1'] ** (-1 / h['mu'])
    return np.where(v < vmax, (v / (h['C1'] * Uv)) ** (-3 * h['mu']), 0.0)


def delivery(U, mat, sel=None):
    """Fraction of the escaping mass that reaches the Earth, as arrival-time samples with weights,
    plus a tail estimate for what is still orbiting at the end."""
    e = np.load(os.path.join(OUT, 'ejecta.npz'))
    s = np.ones(len(e['v']), bool) if sel is None else sel
    w = speed_weights(e['v'][s], U, mat)
    w = w / w.sum()
    fate, tf, t0, vimp = e['fate'][s], e['t_f'][s], e['t0'][s], e['vimp'][s]
    t_end = float(e['t_end'])
    hit = fate == 1
    # tail: like the last fifth of the run
    alive = w[fate == 0].sum()
    late = (fate > 0) & (tf > 0.8 * t_end)
    r_all = w[late].sum()
    r_e = w[late & hit].sum()
    tau = alive / r_all * 0.2 * t_end if r_all > 0 else np.inf
    return dict(t=tf[hit] - t0[hit], w=w[hit], v=vimp[hit], frac_in_run=float(w[hit].sum()),
                tail_mass=float(alive * (r_e / r_all if r_all > 0 else 0.0)), tail_tau=float(tau),
                moon=float(w[fate == 2].sum()), escaped=float(w[fate == 3].sum()), orbiting=float(alive))


def earth_effects(M_esc, dl, f_space=0.5, horizon=3000.0):
    """Arrival history -> energy, peak heating, climate."""
    grid = np.unique(np.r_[np.arange(0, 2 * YEAR, DAY), np.arange(2 * YEAR, 40 * YEAR, 10 * DAY),
                          np.arange(40 * YEAR, horizon * YEAR, 0.5 * YEAR), horizon * YEAR])
    m = M_esc * dl['w']
    E = np.zeros(len(grid) - 1)
    k = np.clip(np.searchsorted(grid, dl['t'], side='right') - 1, 0, len(E) - 1)
    np.add.at(E, k, 0.5 * m * dl['v'] ** 2)
    # smooth over 15 days: each "parcel" is a stream of pieces, not one rock
    from analyze import smooth_deposit
    E = smooth_deposit(grid, dl['t'], 0.5 * m * dl['v'] ** 2, 15 * DAY)
    t_end = dl['t'].max() if len(dl['t']) else 30 * YEAR
    if np.isfinite(dl['tail_tau']) and dl['tail_mass'] > 0:
        a, b = np.maximum(grid[:-1], 30 * YEAR), np.maximum(grid[1:], 30 * YEAR)
        tm = M_esc * dl['tail_mass'] * (np.exp(-(a - 30 * YEAR) / dl['tail_tau']) - np.exp(-(b - 30 * YEAR) / dl['tail_tau']))
        E += 0.5 * tm * 11.1e3 ** 2
    dt = np.diff(grid)
    P = E / dt / A_EARTH
    Ts, Td, vap, olr = climate.run_ebm(grid, np.r_[P, 0.0], f_space, climate.TT, climate.HH, climate.MV, climate.COL_OCEAN)
    first_year = E[grid[:-1] < YEAR].sum()
    # monthly-mean peak heating
    cum = np.r_[0, np.cumsum(E)]
    tt = grid
    mon = 30 * DAY
    ends = np.interp(tt[:-1] + mon, tt, cum) - cum[:-1]
    return dict(E_total=float(E.sum()), E_first_year=float(first_year), peak_month_W=float((ends / mon / A_EARTH).max()),
                peak_T_C=float(Ts.max() - 273.15), ocean=float(vap.max() / climate.COL_OCEAN),
                warming_C=float(Ts.max() - 288.0))


RUNGS = [
    ('tunguska_daily', 'A Tunguska-sized blast\'s worth of energy every day for a year', lambda r: r['E_first_year'] >= 365 * TUNGUSKA_J),
    ('chicxulub_year', 'A dinosaur-killer\'s worth of energy in the first year', lambda r: r['E_first_year'] >= CHICXULUB_J),
    ('warming_10', 'Global average warming of 10 °C (the end-Permian extinction was ~10 °C)', lambda r: r['warming_C'] >= 10),
    ('sun_month', 'A month of heat from the sky as strong as all the sunlight', lambda r: r['peak_month_W'] >= 240),
    ('oceans_boil', 'The oceans start to boil (global average above 100 °C)', lambda r: r['peak_T_C'] >= 100),
    ('oceans_gone', 'Every ocean boils away', lambda r: r['ocean'] >= 0.999),
    ('moon_shatters', 'The Moon itself shatters (Seveneves)', lambda r: r['q_ratio'] >= 1.0),
]


def ladder(speeds=(12e3, 20e3, 50e3)):
    e = np.load(os.path.join(OUT, 'ejecta.npz'))
    trail = e['trail']
    Ds = np.geomspace(0.1e3, 5000e3, 130)
    out = dict(materials=MATERIALS, speeds=list(speeds), diameters_km=list(Ds / 1e3), cases={})
    for U in speeds:
        for mat in MATERIALS:
            for site, sel in [('average', None), ('trailing', trail > 0.7)]:
                dl = delivery(U, mat, sel)
                rows = []
                for D in Ds:
                    M_esc, qrat = escaping_mass(D, U, mat)
                    r = earth_effects(M_esc, dl)
                    r.update(D_km=D / 1e3, impactor_kg=impactor_mass(D), impactor_J=0.5 * impactor_mass(D) * U ** 2,
                             M_esc=M_esc, q_ratio=qrat, M_earth=M_esc * (dl['frac_in_run'] + dl['tail_mass']))
                    rows.append(r)
                thr = {}
                for key, label, test in RUNGS:
                    hit = [r['D_km'] for r in rows if test(r)]
                    thr[key] = hit[0] if hit else None
                name = f'{int(U / 1e3)}kms_{mat}_{site}'
                out['cases'][name] = dict(U=U, material=mat, site=site, thresholds=thr,
                                          frac_to_earth_30y=dl['frac_in_run'], frac_tail=dl['tail_mass'],
                                          frac_back_to_moon=dl['moon'], frac_escaped=dl['escaped'],
                                          rows=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
                print(f'{name:28s} to Earth {dl["frac_in_run"]:.1%} (+{dl["tail_mass"]:.1%} later): ' +
                      ', '.join(f'{k} {v:.0f} km' if v and v >= 1 else f'{k} {v * 1e3:.0f} m' if v else f'{k} -' for k, v in thr.items()))
    out['rungs'] = [(k, label) for k, label, _ in RUNGS]
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'lunar_impact.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'field':
        make_field()
    elif cmd == 'tracers':
        run_tracers(n=int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
    elif cmd == 'ladder':
        ladder()
