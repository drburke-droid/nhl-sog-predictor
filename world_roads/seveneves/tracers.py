"""Stage 2: follow the debris.

Each tracer is a test particle carrying a mass (one swarm parcel of `m_quantum`, or one discrete chunk).
It feels the Earth, the Sun's tide and every fragment above `m_grav`; fragment positions come from
`field.npz` by cubic Hermite interpolation. Integration is adaptive Dormand-Prince 5(4), one tracer at
a time, in parallel (numba). Fates: hit the Earth (below 100 km altitude), hit a fragment, or leave
Earth's sphere of influence.

Debris that hits a fragment blasts out crater ejecta (Housen & Holsapple 2011 scaling). Ejecta faster
than the fragment's escape speed become the next generation of tracers: this is the "cascade" of the
novel, and `run_cascade` follows it generation by generation until it dies out (or grows).
"""
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

from constants import GM_EARTH, GM_SUN, R_ENTRY, R_ESCAPE, DAY, YEAR

# Dormand-Prince 5(4) tableau
C2, C3, C4, C5 = 1 / 5, 3 / 10, 4 / 5, 8 / 9
A21 = 1 / 5
A31, A32 = 3 / 40, 9 / 40
A41, A42, A43 = 44 / 45, -56 / 15, 32 / 9
A51, A52, A53, A54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
A61, A62, A63, A64, A65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
B1, B3, B4, B5, B6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
E1, E3, E4, E5, E6, E7 = 71 / 57600, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40


@njit(cache=True)
def locate(t, T, k):
    K = T.shape[0]
    while k + 2 < K and T[k + 1] <= t:
        k += 1
    while k > 0 and T[k] > t:
        k -= 1
    while k + 2 < K and T[k + 1] <= T[k]:      # skip zero-length (collision) intervals
        k += 1
    return k


@njit(cache=True)
def field_at(t, k, T, F, SUN, pf, gm, rad, ps):
    """Fragment positions pf[S,3], GM, radius; Sun position ps[3] at time t (interval k)."""
    dt = T[k + 1] - T[k]
    s = (t - T[k]) / dt
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = (s3 - 2 * s2 + s) * dt
    h01 = -2 * s3 + 3 * s2
    h11 = (s3 - s2) * dt
    S = F.shape[1]
    for j in range(S):
        g = F[k, j, 6]
        gm[j] = g
        if g > 0.0:
            rad[j] = F[k, j, 7]
            for c in range(3):
                pf[j, c] = h00 * F[k, j, c] + h10 * F[k, j, 3 + c] + h01 * F[k + 1, j, c] + h11 * F[k + 1, j, 3 + c]
    for c in range(3):
        ps[c] = h00 * SUN[k, c] + h10 * SUN[k, 3 + c] + h01 * SUN[k + 1, c] + h11 * SUN[k + 1, 3 + c]


@njit(cache=True)
def frag_vel(t, k, T, F, j, out):
    dt = T[k + 1] - T[k]
    s = min(max((t - T[k]) / dt, 0.0), 1.0)
    d00 = (6 * s * s - 6 * s) / dt
    d10 = 3 * s * s - 4 * s + 1
    d01 = (-6 * s * s + 6 * s) / dt
    d11 = 3 * s * s - 2 * s
    for c in range(3):
        out[c] = d00 * F[k, j, c] + d10 * F[k, j, 3 + c] + d01 * F[k + 1, j, c] + d11 * F[k + 1, j, 3 + c]


@njit(cache=True)
def accel(y, pf, gm, rad, ps, a):
    x0, x1, x2 = y[0], y[1], y[2]
    r2 = x0 * x0 + x1 * x1 + x2 * x2
    f = -GM_EARTH / (r2 * math.sqrt(r2))
    a[0] = f * x0
    a[1] = f * x1
    a[2] = f * x2
    # solar tide (direct minus the Sun's pull on the Earth)
    d0, d1, d2 = ps[0] - x0, ps[1] - x1, ps[2] - x2
    dd = d0 * d0 + d1 * d1 + d2 * d2
    fd = GM_SUN / (dd * math.sqrt(dd))
    ss = ps[0] * ps[0] + ps[1] * ps[1] + ps[2] * ps[2]
    fs = GM_SUN / (ss * math.sqrt(ss))
    a[0] += fd * d0 - fs * ps[0]
    a[1] += fd * d1 - fs * ps[1]
    a[2] += fd * d2 - fs * ps[2]
    for j in range(gm.shape[0]):
        g = gm[j]
        if g <= 0.0:
            continue
        d0, d1, d2 = pf[j, 0] - x0, pf[j, 1] - x1, pf[j, 2] - x2
        dd = d0 * d0 + d1 * d1 + d2 * d2
        r = math.sqrt(dd)
        R = rad[j]
        if r > R:
            fj = g / (dd * r)
        else:                                 # inside: uniform sphere
            fj = g / (R * R * R)
        q = pf[j, 0] * pf[j, 0] + pf[j, 1] * pf[j, 1] + pf[j, 2] * pf[j, 2]
        fq = g / (q * math.sqrt(q))           # the fragment's pull on the Earth (indirect term)
        a[0] += fj * d0 - fq * pf[j, 0]
        a[1] += fj * d1 - fq * pf[j, 1]
        a[2] += fj * d2 - fq * pf[j, 2]


@njit(cache=True)
def deriv(t, y, k, T, F, SUN, pf, gm, rad, ps, out):
    k = locate(t, T, k)
    field_at(t, k, T, F, SUN, pf, gm, rad, ps)
    out[0] = y[3]
    out[1] = y[4]
    out[2] = y[5]
    a = out[3:]
    accel(y, pf, gm, rad, ps, a)
    return k


@njit(cache=True)
def seg_min(d0, d1, R):
    """Does the straight segment d0->d1 pass within R of the origin? Return fraction s (or -1)."""
    e0, e1, e2 = d1[0] - d0[0], d1[1] - d0[1], d1[2] - d0[2]
    ee = e0 * e0 + e1 * e1 + e2 * e2
    s = 0.0
    if ee > 0:
        s = -(d0[0] * e0 + d0[1] * e1 + d0[2] * e2) / ee
        s = min(max(s, 0.0), 1.0)
    m0, m1, m2 = d0[0] + s * e0, d0[1] + s * e1, d0[2] + s * e2
    if m0 * m0 + m1 * m1 + m2 * m2 < R * R:
        # first entry along the segment
        b = d0[0] * e0 + d0[1] * e1 + d0[2] * e2
        c = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2] - R * R
        if c <= 0:
            return 0.0
        disc = b * b - ee * c
        return (-b - math.sqrt(max(disc, 0.0))) / ee
    return -1.0


@njit(parallel=True, cache=True)
def integrate(t0, y0, t_end, T, F, SUN, rtol, frame_t, frames, kick_dv, kick_tau, seed):
    n = t0.shape[0]
    S = F.shape[1]
    fate = np.zeros(n, np.int8)            # 0 alive at t_end, 1 Earth, 2 fragment, 3 escaped
    t_f = np.full(n, t_end)
    vimp = np.zeros(n)
    slot = np.full(n, -1, np.int32)
    nrm = np.zeros((n, 3))                  # Earth: impact direction; fragment: outward normal
    fgm = np.zeros(n)
    frad = np.zeros(n)
    y_end = np.zeros((n, 6))
    nsteps = np.zeros(n, np.int64)
    NF = frame_t.shape[0]
    for i in prange(n):
        np.random.seed(seed + i)
        pf = np.zeros((S, 3))
        gm = np.zeros(S)
        rad = np.zeros(S)
        ps = np.zeros(3)
        pf1 = np.zeros((S, 3))
        gm1 = np.zeros(S)
        rad1 = np.zeros(S)
        ps1 = np.zeros(3)
        y = y0[i].copy()
        yn = np.zeros(6)
        ye = np.zeros(6)
        k1 = np.zeros(6)
        k2 = np.zeros(6)
        k3 = np.zeros(6)
        k4 = np.zeros(6)
        k5 = np.zeros(6)
        k6 = np.zeros(6)
        k7 = np.zeros(6)
        tmp = np.zeros(6)
        d0 = np.zeros(3)
        d1 = np.zeros(3)
        vf = np.zeros(3)
        t = t0[i]
        k = locate(t, T, 0)
        h = 30.0
        kf = 0
        while kf < NF and frame_t[kf] < t:
            kf += 1
        next_kick = t + (-math.log(np.random.random()) * kick_tau if kick_tau > 0 else 1e300)
        k = deriv(t, y, k, T, F, SUN, pf, gm, rad, ps, k1)
        done = False
        steps = 0
        while not done:
            if t + h > t_end:
                h = t_end - t
            if t + h > next_kick:
                h = max(next_kick - t, 1.0)
            for c in range(6):
                tmp[c] = y[c] + h * A21 * k1[c]
            kk = deriv(t + C2 * h, tmp, k, T, F, SUN, pf1, gm1, rad1, ps1, k2)
            for c in range(6):
                tmp[c] = y[c] + h * (A31 * k1[c] + A32 * k2[c])
            kk = deriv(t + C3 * h, tmp, kk, T, F, SUN, pf1, gm1, rad1, ps1, k3)
            for c in range(6):
                tmp[c] = y[c] + h * (A41 * k1[c] + A42 * k2[c] + A43 * k3[c])
            kk = deriv(t + C4 * h, tmp, kk, T, F, SUN, pf1, gm1, rad1, ps1, k4)
            for c in range(6):
                tmp[c] = y[c] + h * (A51 * k1[c] + A52 * k2[c] + A53 * k3[c] + A54 * k4[c])
            kk = deriv(t + C5 * h, tmp, kk, T, F, SUN, pf1, gm1, rad1, ps1, k5)
            for c in range(6):
                tmp[c] = y[c] + h * (A61 * k1[c] + A62 * k2[c] + A63 * k3[c] + A64 * k4[c] + A65 * k5[c])
            kk = deriv(t + h, tmp, kk, T, F, SUN, pf1, gm1, rad1, ps1, k6)
            for c in range(6):
                yn[c] = y[c] + h * (B1 * k1[c] + B3 * k3[c] + B4 * k4[c] + B5 * k5[c] + B6 * k6[c])
            kk = deriv(t + h, yn, kk, T, F, SUN, pf1, gm1, rad1, ps1, k7)
            # error norm: positions relative to max(|r|, R_entry/10), velocities to max(|v|, 10 m/s)
            rn = max(math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2), 6e5)
            vn = max(math.sqrt(y[3] ** 2 + y[4] ** 2 + y[5] ** 2), 10.0)
            err = 0.0
            for c in range(6):
                ec = h * (E1 * k1[c] + E3 * k3[c] + E4 * k4[c] + E5 * k5[c] + E6 * k6[c] + E7 * k7[c])
                sc = rtol * (rn if c < 3 else vn)
                err = max(err, abs(ec) / sc)
            if err > 1.0 and h > 1e-3:
                h *= max(0.2, 0.9 * err ** -0.25)
                continue
            steps += 1
            tn = t + h
            # frames (cubic Hermite between the step ends)
            while kf < NF and frame_t[kf] <= tn:
                s = (frame_t[kf] - t) / h
                s2 = s * s
                s3 = s2 * s
                for c in range(3):
                    frames[kf, i, c] = (2 * s3 - 3 * s2 + 1) * y[c] + (s3 - 2 * s2 + s) * h * y[3 + c] + \
                        (-2 * s3 + 3 * s2) * yn[c] + (s3 - s2) * h * yn[3 + c]
                kf += 1
            # Earth
            for c in range(3):
                d0[c] = y[c]
                d1[c] = yn[c]
            s_hit = seg_min(d0, d1, R_ENTRY)
            if s_hit >= 0.0:
                r0 = math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
                v2 = y[3] ** 2 + y[4] ** 2 + y[5] ** 2 + 2 * GM_EARTH * (1 / R_ENTRY - 1 / r0)
                fate[i] = 1
                t_f[i] = t + s_hit * h
                vimp[i] = math.sqrt(max(v2, 0.0))
                for c in range(3):
                    nrm[i, c] = d0[c] + s_hit * (d1[c] - d0[c])
                break
            # fragments (pf: at t from k1 eval; pf1: at t+h from k7 eval)
            hitj = -1
            best = 2.0
            for j in range(S):
                if gm[j] <= 0.0 or gm1[j] <= 0.0:
                    continue
                for c in range(3):
                    d0[c] = y[c] - pf[j, c]
                    d1[c] = yn[c] - pf1[j, c]
                if d0[0] ** 2 + d0[1] ** 2 + d0[2] ** 2 < rad[j] ** 2:
                    continue                        # started inside (launch site): ignore
                sj = seg_min(d0, d1, rad1[j])
                if sj >= 0.0 and sj < best:
                    best = sj
                    hitj = j
            if hitj >= 0:
                th = t + best * h
                kk = locate(th, T, kk)
                field_at(th, kk, T, F, SUN, pf1, gm1, rad1, ps1)
                frag_vel(th, kk, T, F, hitj, vf)
                fate[i] = 2
                t_f[i] = th
                slot[i] = hitj
                fgm[i] = gm1[hitj]
                frad[i] = rad1[hitj]
                vv = 0.0
                for c in range(3):
                    pc = y[c] + best * (yn[c] - y[c]) - pf1[hitj, c]
                    nrm[i, c] = pc
                    dv = y[3 + c] + best * (yn[3 + c] - y[3 + c]) - vf[c]
                    vv += dv * dv
                vimp[i] = math.sqrt(vv)
                nn = math.sqrt(nrm[i, 0] ** 2 + nrm[i, 1] ** 2 + nrm[i, 2] ** 2)
                for c in range(3):
                    nrm[i, c] /= nn
                break
            t = tn
            for c in range(6):
                y[c] = yn[c]
                k1[c] = k7[c]
            for j in range(S):
                gm[j] = gm1[j]
                rad[j] = rad1[j]
                for c in range(3):
                    pf[j, c] = pf1[j, c]
            k = kk
            if y[0] ** 2 + y[1] ** 2 + y[2] ** 2 > R_ESCAPE * R_ESCAPE:
                fate[i] = 3
                t_f[i] = t
                break
            if t >= next_kick:                      # optional collisional randomisation
                for c in range(3):
                    y[3 + c] += kick_dv / math.sqrt(3.0) * np.random.normal()
                next_kick = t - math.log(np.random.random()) * kick_tau
                k = deriv(t, y, k, T, F, SUN, pf, gm, rad, ps, k1)
            if t >= t_end:
                done = True
                break
            fac = 0.9 * max(err, 1e-10) ** -0.2
            h *= min(5.0, max(0.2, fac))
            h = min(h, 0.5 * DAY)
        for c in range(6):
            y_end[i, c] = y[c]
        nsteps[i] = steps
    return fate, t_f, vimp, slot, nrm, fgm, frad, y_end, nsteps


# ---------------------------------------------------------------- crater ejecta (the cascade)
# Housen & Holsapple (2011, Icarus 211:856) ejecta scaling, M(>v)/m = (3k/4pi)[(v/(C1 U))^(-3 mu) - n1^3].
# Defaults are their "water / competent rock" fit (mu = 0.55), the most generous to a cascade among
# their materials; "sand" (mu = 0.41, C1 = 0.55, k = 0.3) gives far less escaping ejecta.
HH = dict(mu=0.55, C1=1.5, k=0.2, n1=1.2)


def escaping_ejecta(U, v_esc, hh=HH):
    """Mass of ejecta faster than v_esc per unit impactor mass."""
    x = (np.asarray(v_esc) / (hh['C1'] * np.asarray(U))) ** (-3 * hh['mu'])
    return np.maximum(3 * hh['k'] / (4 * math.pi) * (x - hh['n1'] ** 3), 0.0)


def ejecta_tracers(res, field, m_quantum, rng):
    """Spawn next-generation tracers from fragment impacts in `res`."""
    T, F = field['T'], field['F']
    hit = np.where(res['fate'] == 2)[0]
    out_t, out_y, out_m = [], [], []
    mu, C1, k_, n1 = HH['mu'], HH['C1'], HH['k'], HH['n1']
    for i in hit:
        gm, R, U = res['fgm'][i], res['frad'][i], res['vimp'][i]
        v_esc = math.sqrt(2 * gm / R)
        m_esc = res['m'][i] * float(escaping_ejecta(U, v_esc))
        if m_esc <= 0:
            continue
        n = int(m_esc / m_quantum + rng.random())
        if n == 0:
            continue
        th = res['t_f'][i]
        kk = np.searchsorted(T, th, side='right') - 1
        kk = min(max(kk, 0), len(T) - 2)
        while kk + 2 < len(T) and T[kk + 1] <= T[kk]:
            kk += 1
        s = (th - T[kk]) / (T[kk + 1] - T[kk])
        j = res['slot'][i]
        pfr = (1 - s) * F[kk, j, :3] + s * F[kk + 1, j, :3]
        vfr = (1 - s) * F[kk, j, 3:6] + s * F[kk + 1, j, 3:6]
        nrm = res['nrm'][i]
        # ejecta speeds drawn from M(>v), v in [v_esc, C1 U / n1^(1/mu)]
        x_lo = (v_esc / (C1 * U)) ** (-3 * mu)
        u = rng.random(n)
        x = n1 ** 3 + u * (x_lo - n1 ** 3)
        v = C1 * U * x ** (-1 / (3 * mu))
        # 45 degree ejecta cone around the local normal
        a = np.cross(nrm, [0.0, 0.0, 1.0] if abs(nrm[2]) < 0.9 else [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(nrm, a)
        ph = rng.random(n) * 2 * math.pi
        el = math.radians(45)
        d = (math.cos(el) * nrm[None] + math.sin(el) * (np.cos(ph)[:, None] * a + np.sin(ph)[:, None] * b))
        x0 = pfr + 1.01 * R * nrm
        out_t.append(np.full(n, th))
        out_y.append(np.hstack([np.repeat(x0[None], n, 0), vfr + v[:, None] * d]))
        out_m.append(np.full(n, m_esc / n))
    if not out_t:
        return None
    return dict(t=np.concatenate(out_t), y=np.vstack(out_y), m=np.concatenate(out_m))


def run_cascade(outdir, rtol=1e-9, frame_dt=2 * DAY, frame_years=None, kick_dv=0.0, kick_tau_years=0.0,
                max_gen=8, seed=1, subsample=1):
    field = dict(np.load(os.path.join(outdir, 'field.npz')))
    tr = np.load(os.path.join(outdir, 'tracers0.npz'))
    T, F, SUN = field['T'], field['F'], field['SUN']
    t_end = float(T[-1])
    sel = np.arange(len(tr['m']))[::subsample]
    gen = dict(t=tr['t'][sel], y=np.hstack([tr['x'][sel], tr['v'][sel]]), m=tr['m'][sel] * subsample,
               src=tr['src'][sel])
    m_quantum = float(np.median(tr['m'])) * subsample
    fy = frame_years if frame_years is not None else t_end / YEAR
    frame_t = np.arange(0, min(fy * YEAR, t_end) + 1, frame_dt)
    rng = np.random.default_rng(seed)
    results = []
    g = 0
    wall = time.time()
    while gen is not None and g <= max_gen and len(gen['m']):
        n = len(gen['m'])
        frames = np.full((len(frame_t), n, 3), np.nan, np.float32)
        out = integrate(gen['t'].astype(float), gen['y'].astype(float), t_end, T, F, SUN, rtol, frame_t, frames,
                        kick_dv, kick_tau_years * YEAR, seed * 100000 + g * 1000)
        res = dict(zip(['fate', 't_f', 'vimp', 'slot', 'nrm', 'fgm', 'frad', 'y_end', 'nsteps'], out))
        res.update(t0=gen['t'], m=gen['m'], gen=np.full(n, g, np.int16),
                   src=gen.get('src', np.full(n, 2, np.int8)))
        mass = gen['m'].sum()
        fm = {f: res['m'][res['fate'] == f].sum() / mass for f in range(4)}
        print(f'  gen {g}: {n} tracers ({mass:.3g} kg): Earth {fm[1]:.3%}, fragments {fm[2]:.1%}, '
              f'escaped {fm[3]:.1%}, still in orbit {fm[0]:.1%}; {res["nsteps"].mean():.0f} steps each, '
              f'{time.time() - wall:.0f}s', flush=True)
        np.savez_compressed(os.path.join(outdir, f'frames_gen{g}.npz'), t=frame_t, P=frames)
        results.append(res)
        nxt = ejecta_tracers(res, field, m_quantum, rng)
        if nxt is not None:
            mult = nxt['m'].sum() / max(res['m'][res['fate'] == 2].sum(), 1e-30)
            print(f'      fragment impacts blast out {nxt["m"].sum():.3g} kg of escaping ejecta '
                  f'({mult:.2f} x the mass that hit)', flush=True)
        gen = nxt
        g += 1
    keys = ['fate', 't_f', 'vimp', 'slot', 'nrm', 'fgm', 'frad', 'y_end', 'nsteps', 't0', 'm', 'gen', 'src']
    allr = {k: np.concatenate([r[k] for r in results]) for k in keys}
    np.savez_compressed(os.path.join(outdir, 'rain.npz'), **allr)
    return allr


if __name__ == '__main__':
    d = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split('=')
        kw[k] = float(v) if '.' in v or 'e' in v else int(v)
    run_cascade(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', d), **kw)
