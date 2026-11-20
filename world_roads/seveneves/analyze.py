"""Stage 4: where did the Moon go, and what did the Earth feel?

Reads out/<scenario>/{events.json, rain.npz, bodies.npz}; writes out/<scenario>/summary.json and
series.npz (rain power, temperature, ocean boiled off, over 10,000 years).
"""
import json
import math
import os
import sys

import numpy as np

from constants import M_MOON, A_EARTH, YEAR, DAY, M_OCEAN
import climate

CHICXULUB_J = 4.2e23          # ~100 million megatons of TNT
SUN_W = climate.S_ABS


def smooth_deposit(t_grid, times, values, sigma):
    """Deposit `values` at `times` onto bins of t_grid, each spread as a Gaussian of width sigma."""
    out = np.zeros(len(t_grid) - 1)
    if len(times) == 0:
        return out
    from scipy.special import erf
    edges = t_grid
    for t, v in zip(times, values):
        if sigma <= 0:
            k = np.searchsorted(edges, t, side='right') - 1
            if 0 <= k < len(out):
                out[k] += v
            continue
        lo = np.searchsorted(edges, t - 5 * sigma) - 1
        hi = np.searchsorted(edges, t + 5 * sigma) + 1
        lo, hi = max(lo, 0), min(hi, len(edges) - 1)
        c = 0.5 * (1 + erf((edges[lo:hi + 1] - t) / (sigma * math.sqrt(2))))
        out[lo:hi] += v * np.diff(c)
    return out


def analyse(outdir, f_space=0.5, horizon_years=10_000, swarm_sigma_days=15.0, debris0=None, save=True):
    """debris0: re-weight the breakup debris to this fraction of the Moon (tracers are test particles,
    so their mass only scales the result; fragment masses are left as simulated)."""
    ev = json.load(open(os.path.join(outdir, 'events.json')))
    rain = dict(np.load(os.path.join(outdir, 'rain.npz')))
    src0 = np.load(os.path.join(outdir, 'tracers0.npz'))['src']
    n0 = int((rain['gen'] == 0).sum())
    src = np.full(len(rain['m']), 2, np.int8)
    src[:n0] = src0[::len(src0) // n0][:n0]
    d0_sim = ev['config']['debris0']
    if debris0 is not None:
        sc = debris0 / d0_sim if d0_sim > 0 else 0.0
        rain['m'] = np.where(src == 0, rain['m'] * sc, rain['m'])
        ev['config'] = dict(ev['config'], debris0=debris0)
    bodies = np.load(os.path.join(outdir, 'bodies.npz'))
    t_end = float(np.load(os.path.join(outdir, 'field.npz'))['T'][-1])
    t_end = min(t_end, float(np.max(rain['t_f'])))          # the tracer run may stop earlier
    events = ev['events']
    m = rain['m']
    fate = rain['fate']
    gen = rain['gen']
    v = rain['vimp']

    # ---- where the Moon's mass went ----
    last = bodies['B'][-1]
    last = last[~np.isnan(last[:, 3])]
    frag_orbit = float(last[:, 3].sum())
    frag_earth = sum(e['m'] for e in events if e['kind'] == 'earth_impact')
    frag_esc = sum(e['m'] for e in events if e['kind'] == 'escaped')
    g0 = gen == 0
    deb = {name: float(m[g0 & (fate == f)].sum()) for f, name in
           [(0, 'orbiting'), (1, 'earth'), (2, 'reaccreted'), (3, 'escaped')]}
    # ejecta generations: mass that hit the Earth etc. (counted separately; it came out of fragments)
    ej = {name: float(m[~g0 & (fate == f)].sum()) for f, name in
          [(0, 'orbiting'), (1, 'earth'), (2, 'reaccreted'), (3, 'escaped')]}
    budget = dict(fragments_in_orbit=frag_orbit, fragments_hit_earth=frag_earth, fragments_escaped=frag_esc,
                  debris=deb, ejecta=ej)
    biggest = float(last[:, 3].max()) if len(last) else 0.0

    # ---- the rain at the top of the atmosphere ----
    hit = (fate == 1) & (m > 0)
    th, mh, vh = rain['t_f'][hit], m[hit], v[hit]
    eh = 0.5 * mh * vh ** 2
    big = [(e['t'], e['m'], e['energy']) for e in events if e['kind'] == 'earth_impact']

    # tail: debris still in orbit at the end decays like the last fifth of the run
    alive = fate == 0
    m_alive = float(m[alive].sum())
    win = t_end * 0.8
    rem = (fate > 0) & (rain['t_f'] > win)
    r_all = m[rem].sum() / (t_end - win)
    r_earth = m[rem & (fate == 1)].sum() / (t_end - win)
    tau = m_alive / r_all if r_all > 0 else np.inf
    f_e = r_earth / r_all if r_all > 0 else 0.0
    late = (th > win) & (mh > 0)
    v_late = float(np.average(vh[late], weights=mh[late])) if late.any() else 11.1e3

    # time grid: 1 day out to the end of the N-body run, then 30 days, then 1 year
    g1 = np.arange(0, t_end, DAY)
    g2 = np.arange(t_end, min(200 * YEAR, horizon_years * YEAR), 30 * DAY)
    g3 = np.arange(max(200 * YEAR, t_end), horizon_years * YEAR + 1, YEAR)
    grid = np.unique(np.r_[g1, g2, g3, horizon_years * YEAR])
    dtg = np.diff(grid)
    E = smooth_deposit(grid, th, eh, swarm_sigma_days * DAY)
    M = smooth_deposit(grid, th, mh, swarm_sigma_days * DAY)
    if big:
        bt, bm, be = map(np.array, zip(*big))
        E += smooth_deposit(grid, bt, be, 0)
        M += smooth_deposit(grid, bt, bm, 0)
    tail_m = np.zeros_like(E)
    if np.isfinite(tau) and m_alive > 0:
        a, b = grid[:-1], grid[1:]
        aa, bb = np.maximum(a, t_end), np.maximum(b, t_end)
        tail_m = f_e * m_alive * (np.exp(-(aa - t_end) / tau) - np.exp(-(bb - t_end) / tau))
        E += 0.5 * tail_m * v_late ** 2
        M += tail_m
    P = E / dtg / A_EARTH                      # W/m^2, global mean, interval means
    Ts, Td, vap, olr = climate.run_ebm(grid, np.r_[P, 0.0], f_space, climate.TT, climate.HH, climate.MV,
                                       climate.COL_OCEAN)
    ocean_frac = vap / climate.COL_OCEAN

    yrs = grid / YEAR
    E_tot = float(E.sum())
    # chunk tracers are single bodies; a swarm parcel of mass M holds ~ (M (1-b)/b / 1e15 kg)^b
    # Chicxulub-size (>= 1e15 kg) pieces under the rank-ordered power law
    b = ev['config']['size_slope']
    m_q = ev['config']['m_quantum']
    n_chicx = 0.0
    for mm in mh:
        mmax = mm * (1 - b) / b if mm <= 1.01 * m_q else mm
        n_chicx += max((1e15 / mmax) ** (-b), 1.0) if mmax >= 1e15 else 0.0
    n_chicx += sum(1 for e in big)
    n_chicx += sum(tail_m) / m_q * ((1e15 / (m_q * (1 - b) / b)) ** (-b))

    def years_above(x, thr):
        return float(dtg[(x[:-1] > thr)].sum() / YEAR)

    first = float(th.min() / DAY) if len(th) else None
    summary = dict(
        scenario=os.path.basename(outdir), config=ev['config'], breakup=ev['breakup'],
        sim_years=t_end / YEAR,
        budget_frac_of_moon=dict(
            fragments_in_orbit=frag_orbit / M_MOON, largest_body=biggest / M_MOON,
            fragments_hit_earth=frag_earth / M_MOON, fragments_escaped=frag_esc / M_MOON,
            debris={k: x / M_MOON for k, x in deb.items()}, ejecta={k: x / M_MOON for k, x in ej.items()}),
        n_collisions=sum(e['kind'] == 'collision' for e in events),
        collision_regimes={r: sum(1 for e in events if e.get('regime') == r) for r in
                           ('merge', 'erode', 'supercatastrophic', 'hit-and-run')},
        debris_from_breakup_hit_earth=float(m[(src == 0) & (fate == 1)].sum()),
        debris_from_collisions_hit_earth=float(m[(src == 1) & (fate == 1)].sum()),
        rain=dict(mass_kg=float(M.sum()), mass_frac_of_moon=float(M.sum()) / M_MOON,
                  energy_J=E_tot, chicxulubs_energy=E_tot / CHICXULUB_J,
                  ocean_boil_energies=E_tot * (1 - f_space) / (climate.HH[np.searchsorted(climate.TT, 700)] * A_EARTH),
                  chicxulub_size_impacts=float(n_chicx), first_debris_day=first,
                  in_sim_frac=float(mh.sum() + sum(x[1] for x in big)) / max(float(M.sum()), 1),
                  tail=dict(m_alive=m_alive, tau_years=tau / YEAR, frac_to_earth=f_e),
                  peak_W_m2_30d=float(np.convolve(P[:len(g1)], np.ones(30) / 30, 'same').max()) if len(g1) > 30 else float(P.max()),
                  mean_W_m2_first_year=float(E[grid[:-1] < YEAR].sum() / YEAR / A_EARTH)),
        climate=dict(f_space=f_space, peak_T_K=float(Ts.max()), peak_T_C=float(Ts.max() - 273.15),
                     year_of_peak=float(yrs[Ts.argmax()]), max_ocean_boiled=float(ocean_frac.max()),
                     years_above_50C=years_above(Ts, 323.15), years_above_100C=years_above(Ts, 373.15),
                     T_at_10000y_C=float(Ts[-1] - 273.15), T_at_100y_C=float(np.interp(100, yrs, Ts) - 273.15)),
    )
    series = dict(t=grid, P=np.r_[P, 0.0], Ts=Ts, Td=Td, ocean=ocean_frac, olr=olr, M=np.r_[M, 0.0])
    if save:
        np.savez_compressed(os.path.join(outdir, 'series.npz'), **series)
        with open(os.path.join(outdir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1, default=float)
    summary['_series'] = series
    return summary


def describe(s):
    b = s['budget_frac_of_moon']
    r = s['rain']
    c = s['climate']
    br = s['breakup']
    lines = [
        f"== {s['scenario']}: pieces drift apart at {s['config']['v_disperse']:.0f} m/s "
        f"(breakup energy {br['ke_over_moon_binding']:.2f} x Moon's binding energy); {s['sim_years']:.0f} years simulated",
        f"   collisions: {s['n_collisions']} {s['collision_regimes']}",
        f"   fragments: {b['fragments_in_orbit']:.1%} of the Moon still orbiting (largest {b['largest_body']:.1%}), "
        f"{b['fragments_escaped']:.1%} thrown out of Earth orbit, {b['fragments_hit_earth']:.1%} hit the Earth",
        f"   debris: " + ', '.join(f'{k} {x:.2%}' for k, x in b['debris'].items()),
        f"   rain on Earth (incl. tail): {r['mass_kg']:.3g} kg = {r['mass_frac_of_moon']:.3%} of the Moon, "
        f"{r['energy_J']:.3g} J = {r['chicxulubs_energy']:.3g} Chicxulubs = {r['ocean_boil_energies']:.2f} x the heat to boil the oceans",
        f"   first debris hits on day {r['first_debris_day']}; peak 30-day heating {r['peak_W_m2_30d']:.0f} W/m^2 "
        f"(sunlight {SUN_W:.0f}); first-year mean {r['mean_W_m2_first_year']:.0f} W/m^2",
        f"   climate: peak {c['peak_T_C']:.0f} C in year {c['year_of_peak']:.1f}; oceans boiled off {c['max_ocean_boiled']:.1%}; "
        f"{c['years_above_50C']:.0f} years above 50 C, {c['years_above_100C']:.0f} above 100 C; "
        f"{c['T_at_100y_C']:.0f} C at 100 y, {c['T_at_10000y_C']:.0f} C at 10,000 y",
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', sys.argv[1])
    fs = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    print(describe(analyse(d, f_space=fs)))
