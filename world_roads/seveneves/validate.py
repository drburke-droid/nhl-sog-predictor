"""Check the tracer integrator against REBOUND (IAS15) on the same problem.

Sun + Earth + one massive fragment (a 1e22 kg body on the Moon's orbit), and 40 test particles on
lunar-distance orbits (some passing close to the fragment, some Earth-grazing). REBOUND integrates
everything together; `tracers.integrate` integrates the test particles in the sampled field.
Prints the position differences after 1 year, and compares impact times for particles that hit the
Earth.
"""
import math

import numpy as np
import rebound

from constants import G, M_SUN, M_EARTH, AU, E_EARTH, A_MOON, DAY, YEAR, R_ENTRY, radius_of
import tracers


def main(n=40, years=1.0, dt_sample=0.1 * DAY, seed=3):
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = 'ias15'
    sim.add(m=M_SUN)
    sim.add(m=M_EARTH, a=AU, e=E_EARTH, primary=sim.particles[0])
    mf = 1e22
    sim.add(m=mf, r=radius_of(mf), a=A_MOON, e=0.05, inc=0.09, primary=sim.particles[1])
    frag = sim.particles[2]
    fx, fv = np.array(frag.xyz), np.array(frag.vxyz)
    vg = fv - np.array(sim.particles[1].vxyz)                 # fragment's geocentric velocity
    for i in range(n):
        if i % 2:                                              # odd ones: kill most of the orbital speed
            x = fx + rng.normal(size=3) * 5e6
            v = fv - vg * (0.80 + 0.12 * rng.random()) + rng.normal(size=3) * 30
            sim.add(m=0, x=x[0], y=x[1], z=x[2], vx=v[0], vy=v[1], vz=v[2])
            continue
        # near the fragment, with velocity kicks of 0-1.2 km/s
        dx = rng.normal(size=3)
        dx *= (20e6 + 60e6 * rng.random()) / np.linalg.norm(dx)
        dv = rng.normal(size=3)
        dv *= 1200 * rng.random() / np.linalg.norm(dv)
        x, v = fx + dx, fv + dv
        sim.add(m=0, x=x[0], y=x[1], z=x[2], vx=v[0], vy=v[1], vz=v[2])
    sim.N_active = 3
    sim.move_to_com()
    e = sim.particles[1]
    y0 = np.array([np.r_[np.array(p.xyz) - np.array(e.xyz), np.array(p.vxyz) - np.array(e.vxyz)]
                   for p in sim.particles[3:]])
    K = int(years * YEAR / dt_sample) + 1
    T = np.zeros(K)
    F = np.zeros((K, 1, 8))
    SUN = np.zeros((K, 6))
    hit_t = np.full(n, np.nan)
    for k in range(K):
        t = k * dt_sample
        if k:
            sim.integrate(t, exact_finish_time=1)
        e, s, f = sim.particles[1], sim.particles[0], sim.particles[2]
        ex, ev = np.array(e.xyz), np.array(e.vxyz)
        T[k] = t
        F[k, 0, :3] = np.array(f.xyz) - ex
        F[k, 0, 3:6] = np.array(f.vxyz) - ev
        F[k, 0, 6] = G * mf
        F[k, 0, 7] = radius_of(mf)
        SUN[k, :3] = np.array(s.xyz) - ex
        SUN[k, 3:] = np.array(s.vxyz) - ev
        for i, p in enumerate(sim.particles[3:]):
            if np.isnan(hit_t[i]) and np.linalg.norm(np.array(p.xyz) - ex) < R_ENTRY:
                hit_t[i] = t
    yr = np.array([np.r_[np.array(p.xyz) - ex, np.array(p.vxyz) - ev] for p in sim.particles[3:]])
    frames = np.zeros((0, n, 3), np.float32)
    out = tracers.integrate(np.zeros(n), y0, T[-1], T, F, SUN, 1e-9, np.zeros(0), frames, 0.0, 0.0, 1)
    fate, t_f, y_end = out[0], out[1], out[7]
    print(f'{n} test particles, {years} yr; fates (0 orbit, 1 Earth, 2 fragment, 3 escaped): {np.bincount(fate, minlength=4)}')
    ok = fate == 0
    d = np.linalg.norm(y_end[ok, :3] - yr[ok, :3], axis=1)
    print(f'still orbiting: position difference vs REBOUND median {np.median(d) / 1e3:.1f} km, '
          f'max {d.max() / 1e3:.1f} km (orbits ~{A_MOON / 1e3:,.0f} km)')
    both = (fate == 1) & ~np.isnan(hit_t)
    if both.any():
        print(f'Earth impacts in both: {both.sum()} of {np.sum(fate == 1)} (REBOUND: {np.sum(~np.isnan(hit_t))}); '
              f'impact-time difference max {np.max(np.abs(t_f[both] - hit_t[both])) / 3600:.1f} h '
              f'(REBOUND checked every {dt_sample / 3600:.1f} h)')
    return d


if __name__ == '__main__':
    main()
