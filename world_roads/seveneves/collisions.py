"""What happens when two Moon fragments touch.

Outcome regimes follow Leinhardt & Stewart (2012, ApJ 745:79, "Collisions between gravity-dominated
bodies. I. Outcome regimes and scaling laws"), with the merge / hit-and-run boundary of Kokubo & Genda
(2010, ApJ 714:L21). Debris is split into a power-law size distribution (cumulative N(>m) ~ m^-b).

Everything is returned as plain arrays: new N-body bodies, and new debris tracers.
"""
import math
import numpy as np
from constants import G, RHO_MOON, radius_of

RHO1 = 1000.0           # L&S reference density for R_C1


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


def random_dirs(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def spread_dirs(n, rng):
    """n well-separated directions (Fibonacci sphere, randomly rotated)."""
    if n == 0:
        return np.zeros((0, 3))
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return d @ q.T


def qstar_rd(m_tot, gamma, mu_ratio, cstar, mubar):
    """Catastrophic disruption threshold Q*'_RD (J/kg), L&S 2012 eqs. 10, 23, 15."""
    rc1 = (3 * m_tot / (4 * math.pi * RHO1)) ** (1 / 3)
    q1 = cstar * 0.8 * math.pi * RHO1 * G * rc1 ** 2
    q = q1 * (0.25 * (gamma + 1) ** 2 / gamma) ** (2 / (3 * mubar) - 1)
    return q * mu_ratio ** (2 - 1.5 * mubar)


def largest_remnant(m_tot, q_r, q_star):
    x = q_r / q_star
    if x < 1.8:
        return m_tot * (1 - 0.5 * x)
    return m_tot * 0.1 * (x / 1.8) ** -1.5


def interacting_fraction(r_big, r_small, b):
    """Fraction of the smaller body that overlaps the bigger one along the path (L&S eq. 11)."""
    l = (r_big + r_small) * (1 - b)
    if l >= 2 * r_small:
        return 1.0
    return max((3 * r_small * l ** 2 - l ** 3) / (4 * r_small ** 3), 1e-6)


def split_debris(m_rest, m_lr, cfg):
    """Rank-ordered power-law pieces: m_k = m_max k^(-1/b). Returns (bodies, chunks, swarm_mass)."""
    b = cfg['size_slope']
    m_max = min(m_rest * (1 - b) / b, m_lr)
    bodies, chunks = [], []
    left = m_rest
    k = 1
    while left > 0:
        m = m_max * k ** (-1 / b)
        if m < cfg['m_quantum'] or m > left:
            break
        (bodies if m >= cfg['m_cut'] else chunks).append(m)
        left -= m
        k += 1
    return bodies, chunks, left


def launch_speeds(n, v_esc, median, sigma, rng, cap):
    s = np.exp(np.log(median) + sigma * rng.normal(size=n))
    return np.minimum(s, cap) * v_esc


def resolve(m1, x1, v1, m2, x2, v2, cfg, rng):
    """Resolve a collision. Returns dict with 'bodies' [(m, x, v)], 'tracers' (m, x, v arrays), 'info'."""
    if m2 > m1:
        m1, x1, v1, m2, x2, v2 = m2, x2, v2, m1, x1, v1
    R1, R2 = radius_of(m1), radius_of(m2)
    M = m1 + m2
    dx, dv = x2 - x1, v2 - v1
    v_imp = float(np.linalg.norm(dv))
    nhat = _unit(dx)
    b = float(np.linalg.norm(np.cross(dx, dv)) / (np.linalg.norm(dx) * v_imp + 1e-30))
    b = min(max(b, 0.0), 1.0)
    v_esc = math.sqrt(2 * G * M / (R1 + R2))
    mu = m1 * m2 / M
    q_r = 0.5 * mu * v_imp ** 2 / M
    gamma = m2 / m1
    x_c = (m1 * x1 + m2 * x2) / M
    v_c = (m1 * v1 + m2 * v2) / M
    P = m1 * v1 + m2 * v2

    # Kokubo & Genda (2010) critical speed between merging and hit-and-run
    Gam = ((m1 - m2) / M) ** 2
    om = (1 - b) ** 2.5
    v_cr = v_esc * (2.43 * Gam * om - 0.0408 * Gam + 1.86 * om + 1.08)
    b_crit = R1 / (R1 + R2)
    hit_and_run = (b > b_crit) and (v_imp > v_cr)

    bodies, tr_m, tr_x, tr_v = [], [], [], []
    if not hit_and_run:
        alpha = interacting_fraction(R1, R2, b)
        mu_a = alpha * m2 * m1 / (alpha * m2 + m1)
        qs = qstar_rd(M, gamma, mu / mu_a, cfg['cstar'], cfg['mubar'])
        m_lr = min(largest_remnant(M, q_r, qs), M)
        regime = 'merge' if m_lr > m1 else 'erode'
        if q_r > 1.8 * qs:
            regime = 'supercatastrophic'
        m_rest = M - m_lr
        centre_m, centre_x, centre_v = m_lr, x_c, v_c
        others = []                       # (m, x, v) of second bodies that survive intact
    else:
        # target keeps its mass; projectile is disrupted by the "reverse" impact (L&S 2012 sec. 3.4)
        beta = interacting_fraction(R2, R1, b)          # fraction of the target the projectile sweeps
        m_int = min(beta * m1, m1)
        Mp = m2 + m_int
        mu_p = m2 * m_int / Mp
        qs_p = qstar_rd(Mp, min(m_int, m2) / max(m_int, m2), 1.0, cfg['cstar'], cfg['mubar'])
        q_p = 0.5 * mu_p * v_imp ** 2 / Mp
        m_slr = min(largest_remnant(Mp, q_p, qs_p), m2)
        m_lr = m1
        m_rest = m2 - m_slr
        regime = 'hit-and-run'
        # projectile remnant leaves along a deflected path
        vn = float(dv @ nhat)
        vt = dv - vn * nhat
        dv_new = 0.95 * vt + 0.5 * abs(vn) * nhat
        r_slr = radius_of(m_slr)
        centre_m, centre_x, centre_v = m1, x1, v1
        others = [(m_slr, x1 + (R1 + r_slr + 2e3) * nhat, v1 + dv_new)] if m_slr > 0 else []

    big, chunks, swarm = split_debris(m_rest, m_lr, cfg)
    r_lr = radius_of(centre_m)
    ke_imp = 0.5 * mu * v_imp ** 2

    # discrete fragments: N-body bodies (>= m_cut) and chunk tracers
    nb = len(big)
    dirs_b = spread_dirs(nb, rng)
    vinf_b = launch_speeds(nb, v_esc, cfg['v_body_median'], 0.5, rng, 3 * v_imp / v_esc)
    for m, d, vi in zip(big, dirs_b, vinf_b):
        r = r_lr + radius_of(m) + 2e3
        vl = math.sqrt(vi ** 2 + 2 * G * centre_m / r)
        bodies.append([m, centre_x + r * d, centre_v + vl * d])

    n_sw = int(swarm / cfg['m_quantum'] + rng.random())
    masses = np.array(list(chunks) + [cfg['m_quantum']] * n_sw)
    if len(masses):
        # debris leaves from near the impact point, spread over the outward hemisphere
        dirs = random_dirs(len(masses), rng)
        flip = (dirs @ nhat) < -0.2
        dirs[flip] *= -1
        vinf = launch_speeds(len(masses), v_esc, cfg['v_debris_median'], 0.7, rng, 3 * v_imp / v_esc)
        r0 = r_lr * 1.02
        v_esc_l2 = 2 * G * centre_m / r0
        ke_need = 0.5 * np.sum(masses) * v_esc_l2
        ke_inf = 0.5 * np.sum(masses * vinf ** 2)
        budget = cfg['f_energy'] * ke_imp - ke_need
        if ke_inf > budget:
            vinf *= math.sqrt(max(budget, 0.0) / ke_inf)
        vl = np.sqrt(vinf ** 2 + v_esc_l2)
        tr_m = masses
        tr_x = centre_x + r0 * dirs
        tr_v = centre_v + vl[:, None] * dirs
        tr_vinf = centre_v + vinf[:, None] * dirs      # momentum the debris carries away
    else:
        tr_m, tr_x, tr_v, tr_vinf = np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    bodies = [[centre_m, centre_x.copy(), centre_v.copy()]] + [list(o) for o in others] + bodies
    # momentum: shift all N-body bodies by the same velocity so the total is conserved
    p_out = sum(b_[0] * b_[2] for b_ in bodies) + (tr_m[:, None] * tr_vinf).sum(0)
    m_bodies = sum(b_[0] for b_ in bodies)
    dvc = (P - p_out) / m_bodies
    for b_ in bodies:
        b_[2] = b_[2] + dvc
    info = dict(regime=regime, m1=m1, m2=m2, v_imp=v_imp, v_esc=v_esc, b=b, q_r=q_r,
                m_lr=float(m_lr), debris=float(m_rest), n_bodies=len(bodies),
                n_tracers=int(len(tr_m)), ke_imp=ke_imp)
    return dict(bodies=bodies, tr_m=np.asarray(tr_m, float), tr_x=np.asarray(tr_x, float).reshape(-1, 3),
                tr_v=np.asarray(tr_v, float).reshape(-1, 3), info=info)
