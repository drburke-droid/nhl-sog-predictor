"""Rigid-body Plinko: a flat disc sliding down a tilted pegboard.

The chip is a 2D disc (position, velocity, spin) in the plane of the board. Forces and contacts:
  * gravity along the board: g cos(tilt)
  * sliding friction against the backboard: mu_back * g sin(tilt), opposing the slide (and the spin)
  * impulsive contacts with round pegs, the sawtooth side walls and the slot dividers, with a normal
    coefficient of restitution and Coulomb friction that couples slide and spin.
  * real pins are not perfect: each impact's contact normal is tilted by a small random angle
    (jitter_deg, standard deviation) and its restitution varies by +-e_jitter (uniform). This stands in for
    pin placement tolerance, pins that flex, worn chip edges and the odd brush with the plexiglass cover.

Fixed-step integrator (semi-implicit Euler, 0.1 ms) with impulse contacts and penetration correction, compiled
with numba. Every chip carries its own splitmix64 random stream seeded from its index, so results are
bit-for-bit reproducible however the threads are scheduled.
"""
import math
import numpy as np
from numba import njit, prange

import board as B

# parameter vector layout --------------------------------------------------------------------------
P_E, P_MU, P_TILT, P_MUBACK, P_DT, P_TMAX, P_EREST, P_INERTIA, P_JIT, P_EJIT = range(10)
DEFAULT = dict(e=0.45, mu=0.20, tilt_deg=B.BOARD_TILT_DEG, mu_back=0.10, dt=1e-4, tmax=40.0,
               e_rest_speed=0.05, inertia=0.6, jitter_deg=3.0, e_jitter=0.05)


def params(**kw):
    d = dict(DEFAULT)
    d.update(kw)
    return np.array([d["e"], d["mu"], math.radians(d["tilt_deg"]), d["mu_back"], d["dt"], d["tmax"],
                     d["e_rest_speed"], d["inertia"], math.radians(d["jitter_deg"]), d["e_jitter"]],
                    dtype=np.float64)


PEGS, PEG_ROWS = B.pegs()
SEGS = B.segments()
GEOM = np.array([B.SLOT_PITCH, B.ROW_SPACING, B.CHIP_DIAMETER / 2, B.PEG_DIAMETER / 2,
                 B.last_row_y() + B.DIVIDER_TOP_BELOW_LAST_ROW, B.last_row_y() + B.FLOOR_BELOW_LAST_ROW,
                 B.N_ROWS, B.DIVIDER_THICKNESS / 2], dtype=np.float64)

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


@njit(cache=True, inline="always")
def _rng_next(state):
    """splitmix64: returns (new_state, uniform in [0, 1))."""
    state = state + _GOLD
    z = state
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    z = z ^ (z >> np.uint64(31))
    return state, float(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def _rng_normal(state):
    state, u1 = _rng_next(state)
    state, u2 = _rng_next(state)
    return state, math.sqrt(-2.0 * math.log(u1 + 1e-300)) * math.cos(2.0 * math.pi * u2)


@njit(cache=True, inline="always")
def _contact(px, py, vx, vy, w, nx, ny, depth, R, inv_m, inv_I, e, mu, e_rest, jit, ejit, state):
    """Resolve one contact against a static obstacle. n points from obstacle to chip centre.

    Returns the new state plus 1 if this was an impact (approach faster than e_rest), else 0."""
    px += nx * depth
    py += ny * depth
    vn = vx * nx + vy * ny
    if vn >= 0.0:
        return px, py, vx, vy, w, state, 0
    impact = -vn > e_rest
    ee = 0.0
    if impact:
        ee = e
        if ejit > 0.0:
            state, u = _rng_next(state)
            ee = e + ejit * (2.0 * u - 1.0)
        if jit > 0.0:   # tilt the normal a little: an imperfect pin
            state, z = _rng_normal(state)
            a = jit * z
            ca, sa = math.cos(a), math.sin(a)
            nx, ny = ca * nx - sa * ny, sa * nx + ca * ny
            vn = vx * nx + vy * ny
            if vn >= 0.0:
                return px, py, vx, vy, w, state, 1
    tx, ty = -ny, nx
    vt = vx * tx + vy * ty - w * R          # slip speed of the contact point
    jn = -(1.0 + ee) * vn / inv_m
    jt = -vt / (inv_m + R * R * inv_I)
    lim = mu * jn
    if jt > lim:
        jt = lim
    elif jt < -lim:
        jt = -lim
    vx += (jn * nx + jt * tx) * inv_m
    vy += (jn * ny + jt * ty) * inv_m
    w += -R * jt * inv_I
    return px, py, vx, vy, w, state, 1 if impact else 0


@njit(cache=True)
def _seg_closest(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = ((px - x0) * dx + (py - y0) * dy) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return x0 + t * dx, y0 + t * dy


@njit(cache=True)
def simulate_one(x0, y0, vx0, vy0, w0, seed, prm, geom, segs, rec_every, rec):
    """Drop one chip. Returns (slot 1..9 or 0 if stuck, time, n_impacts, n_rec).

    If rec_every > 0, writes (t, x, y, angle, impact_flag) every rec_every steps into rec (shape (M, 5))."""
    e, mu, tilt, mu_back, dt, tmax, e_rest, kI, jit, ejit = (prm[0], prm[1], prm[2], prm[3], prm[4],
                                                             prm[5], prm[6], prm[7], prm[8], prm[9])
    W, H, R, rp, y_div, y_floor, nrows, rd = (geom[0], geom[1], geom[2], geom[3], geom[4], geom[5],
                                              int(geom[6]), geom[7])
    state = np.uint64(seed) * _M1 + _GOLD
    inv_m = 1.0
    inv_I = 1.0 / (kI * R * R)            # I = kI * m R^2 (0.5 uniform disc, 1.0 thin ring)
    g_in = 9.81 * math.cos(tilt)
    a_fric = mu_back * 9.81 * math.sin(tilt)
    a_spin = a_fric * (2.0 / 3.0) * R * inv_I   # backboard torque / I for uniform pressure
    px, py, vx, vy, w, ang = x0, y0, vx0, vy0, w0, 0.0
    t = 0.0
    impacts = 0
    still = 0.0
    nrec = 0
    step = 0
    nseg = segs.shape[0]
    reach = R + rp
    hit_flag = 0
    while t < tmax:
        # --- forces
        vy += g_in * dt
        sp = math.sqrt(vx * vx + vy * vy)
        if sp > 1e-9:
            dec = a_fric * dt
            if dec >= sp:
                vx = 0.0
                vy = 0.0
            else:
                f = (sp - dec) / sp
                vx *= f
                vy *= f
        dw = a_spin * dt
        if abs(w) <= dw:
            w = 0.0
        elif w > 0:
            w -= dw
        else:
            w += dw
        px += vx * dt
        py += vy * dt
        ang += w * dt
        t += dt
        # --- pegs (only rows near the chip)
        r_lo = int(math.floor((py - reach) / H))
        r_hi = int(math.ceil((py + reach) / H))
        for r in range(max(r_lo, 0), min(r_hi, nrows - 1) + 1):
            yy = r * H
            if r % 2 == 0:
                off = 0.5
                k_min, k_max = 1, 8
            else:
                off = 0.0
                k_min, k_max = 2, 8
            k_lo = int(math.floor((px - reach) / W - off))
            k_hi = int(math.ceil((px + reach) / W - off))
            for k in range(max(k_lo, k_min), min(k_hi, k_max) + 1):
                xx = (k + off) * W
                dx, dy = px - xx, py - yy
                d2 = dx * dx + dy * dy
                if d2 < reach * reach:
                    d = math.sqrt(d2)
                    px, py, vx, vy, w, state, imp = _contact(px, py, vx, vy, w, dx / d, dy / d, reach - d,
                                                             R, inv_m, inv_I, e, mu, e_rest, jit, ejit,
                                                             state)
                    impacts += imp
                    hit_flag |= imp
        # --- walls and dividers
        near_wall = px < 1.0 * W + R + 0.01 or px > 9.0 * W - R - 0.01
        for s in range(nseg):
            x0s, y0s, x1s, y1s = segs[s, 0], segs[s, 1], segs[s, 2], segs[s, 3]
            is_div = x0s == x1s and x0s > 0.75 * W and x0s < 9.25 * W
            if not is_div and not near_wall:
                continue
            if is_div and py < y_div - R - rd:
                continue
            ymin = min(y0s, y1s) - R - rd
            ymax = max(y0s, y1s) + R + rd
            if py < ymin or py > ymax:
                continue
            cx, cy = _seg_closest(px, py, x0s, y0s, x1s, y1s)
            dx, dy = px - cx, py - cy
            d2 = dx * dx + dy * dy
            lim = R + (rd if is_div else 0.0)
            if d2 < lim * lim and d2 > 1e-18:
                d = math.sqrt(d2)
                px, py, vx, vy, w, state, imp = _contact(px, py, vx, vy, w, dx / d, dy / d, lim - d, R,
                                                         inv_m, inv_I, e, mu, e_rest, jit, ejit, state)
                impacts += imp
                hit_flag |= imp
        if rec_every > 0 and step % rec_every == 0 and nrec < rec.shape[0]:
            rec[nrec, 0] = t
            rec[nrec, 1] = px
            rec[nrec, 2] = py
            rec[nrec, 3] = ang
            rec[nrec, 4] = hit_flag
            hit_flag = 0
            nrec += 1
        step += 1
        # --- landed? (centre below the divider tops: the slot is decided)
        if py >= y_div:
            slot = int(math.floor(px / W + 0.5))
            if slot < 1:
                slot = 1
            elif slot > 9:
                slot = 9
            if rec_every > 0:   # let it drop to the slot floor for the animation
                while py < y_floor - R and nrec < rec.shape[0]:
                    vy += g_in * dt
                    py += vy * dt
                    t += dt
                    step += 1
                    if step % rec_every == 0:
                        rec[nrec, 0] = t
                        rec[nrec, 1] = px
                        rec[nrec, 2] = min(py, y_floor - R)
                        rec[nrec, 3] = ang
                        rec[nrec, 4] = 0
                        nrec += 1
            return slot, t, impacts, nrec
        # --- stuck? (at rest somewhere on the board for 1.5 s)
        if vx * vx + vy * vy < 1e-4:
            still += dt
            if still > 1.5:
                return 0, t, impacts, nrec
        else:
            still = 0.0
    return 0, t, impacts, nrec


@njit(parallel=True, cache=True)
def simulate_batch(init, seeds, prm, geom, segs):
    """init: (N, 5) rows of (x, y, vx, vy, spin). Returns slots (N,), times (N,), impacts (N,)."""
    n = init.shape[0]
    slots = np.zeros(n, dtype=np.int64)
    times = np.zeros(n)
    hits = np.zeros(n, dtype=np.int64)
    dummy = np.zeros((1, 5))
    for i in prange(n):
        s, tt, c, _ = simulate_one(init[i, 0], init[i, 1], init[i, 2], init[i, 3], init[i, 4], seeds[i],
                                   prm, geom, segs, 0, dummy)
        slots[i] = s
        times[i] = tt
        hits[i] = c
    return slots, times, hits


# ---------------------------------------------------------------- release model
# A contestant presses the chip flat in the top row's gap over the chosen slot and lets go.
# sx, sy: hand placement scatter (m); sv: speed at release (m/s); sw: spin at release (rad/s).
RELEASE = dict(sx=0.010, sy=0.010, sv=0.05, sw=1.0, y0=0.0)


def releases(position, n, rng, sx=None, sy=None, sv=None, sw=None, y0=None, x_offset=0.0, vy_down=0.0,
             spin=0.0):
    """vy_down: a deliberate throw straight down (m/s); spin: a deliberate twist (rad/s, random sense)."""
    r = dict(RELEASE)
    for k, v in dict(sx=sx, sy=sy, sv=sv, sw=sw, y0=y0).items():
        if v is not None:
            r[k] = v
    init = np.empty((n, 5))
    init[:, 0] = B.drop_x(position) + x_offset + rng.normal(0, r["sx"], n)
    init[:, 1] = r["y0"] + rng.normal(0, r["sy"], n)
    init[:, 2] = rng.normal(0, r["sv"], n)
    init[:, 3] = vy_down + rng.normal(0, r["sv"], n)
    init[:, 4] = rng.normal(0, r["sw"], n)
    if spin:   # drawn only when asked, so default releases keep the random stream the results were made with
        init[:, 4] += spin * rng.choice([-1.0, 1.0], n)
    return init


def run(position, n, prm, seed=0, redrop_stuck=True, **rel):
    """Landing slots for n chips dropped over `position`. Stuck chips are re-dropped, as on the show."""
    rng = np.random.default_rng(seed)
    init = releases(position, n, rng, **rel)
    seeds = rng.integers(1, 2**62, n).astype(np.uint64)
    slots, times, hits = simulate_batch(init, seeds, prm, GEOM, SEGS)
    n_stuck = int((slots == 0).sum())
    tries = 0
    while redrop_stuck and (slots == 0).any() and tries < 5:
        idx = np.where(slots == 0)[0]
        s2, t2, h2 = simulate_batch(releases(position, len(idx), rng, **rel),
                                    rng.integers(1, 2**62, len(idx)).astype(np.uint64), prm, GEOM, SEGS)
        slots[idx], times[idx], hits[idx] = s2, t2, h2
        tries += 1
    return slots, times, hits, n_stuck


def trajectory(x0, y0=0.0, vx0=0.0, vy0=0.0, w0=0.0, seed=1, prm=None, fps=240):
    """One chip's path sampled at fps: columns (t, x, y, angle, impact-since-last-sample)."""
    prm = params() if prm is None else prm
    every = max(1, int(round(1.0 / (fps * prm[P_DT]))))
    rec = np.zeros((int(prm[P_TMAX] * fps) + 10, 5))
    slot, t, c, nrec = simulate_one(x0, y0, vx0, vy0, w0, np.uint64(seed), prm, GEOM, SEGS, every, rec)
    return slot, rec[:nrec].copy()


def kernel(prm, n=4000, seed=0, positions=range(1, 6), **rel):
    """Empirical 9x9 transfer matrix (rows = drop position) plus mean descent time and stuck rate.

    Positions 6..9 are filled in by mirror symmetry from 4..1 unless simulated explicitly."""
    K = np.full((9, 9), np.nan)
    tmean = np.full(9, np.nan)
    stuck = np.full(9, np.nan)
    hits = np.full(9, np.nan)
    for p in positions:
        slots, times, h, n_stuck = run(p, n, prm, seed=seed + 1000 * p, **rel)
        ok = slots > 0
        K[p - 1] = np.bincount(slots[ok], minlength=10)[1:] / ok.sum()
        tmean[p - 1] = times[ok].mean()
        stuck[p - 1] = n_stuck / n
        hits[p - 1] = h[ok].mean()
    for p in range(1, 10):
        q = 10 - p
        if np.isnan(K[p - 1, 0]) and not np.isnan(K[q - 1, 0]):
            K[p - 1] = K[q - 1, ::-1]
            tmean[p - 1], stuck[p - 1], hits[p - 1] = tmean[q - 1], stuck[q - 1], hits[q - 1]
    if 5 in positions:   # the centre row is its own mirror: symmetrise it
        K[4] = 0.5 * (K[4] + K[4, ::-1])
    return K, tmean, stuck, hits


if __name__ == "__main__":
    import time
    prm = params()
    t0 = time.time()
    run(5, 64, prm)
    print("compile", round(time.time() - t0, 1), "s")
    t0 = time.time()
    slots, times, hits, ns = run(5, 20000, prm)
    dt = time.time() - t0
    print(f"20k chips in {dt:.1f}s; mean descent {times.mean():.2f}s; impacts {hits.mean():.0f}; stuck {ns}")
    print(np.bincount(slots, minlength=10)[1:] / len(slots))
