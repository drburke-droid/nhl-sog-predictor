"""
Granular mixing simulation for the "conveyor + mixer" animation.

Six powders (tiny cubes, simulated as equal spheres) sit in lanes in a glass
tray.  Every cycle the tray rides the conveyor under a twin-paddle mixer, the
paddles plunge in, the tray is drawn through them, the paddles lift out and
the tray leaves the frame - then the same tray comes round again.

The physics is position-based dynamics (PBD) with Coulomb friction, run in the
reference frame of the tray.  Output (in ./cache by default):

  particles.pc2   point cache (Blender "Mesh Cache" modifier, PC2 format)
  meta.npz        per-particle attributes + per-frame animation channels
  meta.json       human readable parameters / mixing curve

Units: millimetres and seconds.  Blender scale is applied by build_scene.py.

    python simulate.py                 # default ~64k grains, 10 passes
    python simulate.py --grain 1.2     # finer powder (~125k grains)
    python simulate.py --passes 16
"""
import argparse
import json
import os
import struct
import time

import numpy as np

from pbd import substep

# --------------------------------------------------------------------------
# Machine geometry (mm) - shared with build_scene.py through meta.json
# --------------------------------------------------------------------------
GEO = dict(
    tray_lx=150.0,        # tray interior length (along the belt, x)
    tray_ly=90.0,         # tray interior width  (across the belt, y)
    tray_h=28.0,          # tray interior wall height
    tray_rc=28.0,         # tray interior corner radius (= paddle_r + 2: no dead corners)
    bed_depth=14.0,       # target powder depth
    paddle_r=26.0,        # paddle tip radius
    paddle_sep=34.0,      # distance between the two paddle axes (they intermesh)
    paddle_h=27.0,        # paddle frame height
    paddle_bar=4.0,       # frame bar width
    paddle_rib=3.0,       # inner rib width
    paddle_spine=2.5,     # half width of the centre spine
    paddle_clear=0.5,     # paddle bottom edge above tray floor when plunged
    lift=34.0,            # vertical travel of the mixer head
    mix_reach=47.0,       # paddles traverse tray-local x in [-reach, +reach]
    x_wrap=300.0,         # tray leaves / re-enters at world x = +-x_wrap
)

# --------------------------------------------------------------------------
# Cycle timing (s)
# --------------------------------------------------------------------------
TIMING = dict(
    t_in=1.25,            # travel in, decelerating to a stop under the mixer
    t_plunge=0.35,        # head comes down
    t_mix=1.6,            # tray drawn through the spinning paddles
    t_lift=0.35,          # head goes up
    t_out=1.25,           # accelerate away
    t_settle=0.5,         # grains keep settling this long after the lift
    rev_per_s=1.5,        # paddle speed (about 3 turns in the powder per pass)
    t_spin=0.45,          # paddle spin-up / spin-down time
    t_intro=0.6,          # extra hold on the untouched lanes at the start
    t_outro=2.0,          # hold on the final mix at the end
)

G = 9810.0  # mm/s^2


# --------------------------------------------------------------------------
# Paddle outline: a "gate" frame (flat beater).  Coordinates (r, z) with z
# measured from the paddle's bottom edge; r is signed (two wings).
# --------------------------------------------------------------------------
def paddle_solid(r, z, geo):
    """Boolean mask: is the point (r, z) inside the solid part of the paddle?"""
    R, H = geo["paddle_r"], geo["paddle_h"]
    bar, rib, spine = geo["paddle_bar"], geo["paddle_rib"], geo["paddle_spine"]
    a = np.abs(r)
    inside = (a <= R) & (z >= 0) & (z <= H)
    style = geo.get("paddle_style", "gate")
    if style == "fork":                    # tines hanging from a top bar
        tines = np.zeros_like(inside)
        for c in fork_tines(geo):
            tines |= np.abs(a - c) <= FORK_TINE_W / 2
        return inside & (tines | (z >= H - bar))
    # rounded bottom corners (radius rc)
    rc = 7.0
    corner = (a > R - rc) & (z < rc)
    inside &= ~(corner & ((a - (R - rc)) ** 2 + (z - rc) ** 2 > rc ** 2))
    # windows
    rib_c = 0.5 * (spine + (R - bar))
    win1 = (a > spine) & (a < rib_c - rib / 2) & (z > bar) & (z < H - bar)
    win2 = (a > rib_c + rib / 2) & (a < R - bar) & (z > bar) & (z < H - bar)
    # inner rounded corner of the outer window
    rci = rc - bar
    icorner = win2 & (a > R - bar - rci) & (z < bar + rci)
    win2 &= ~(icorner & ((a - (R - bar - rci)) ** 2 + (z - bar - rci) ** 2 > rci ** 2))
    if style == "plate":                   # solid blade, no windows
        return inside
    return inside & ~(win1 | win2)


FORK_TINE_W = 2.2


def fork_tines(geo, n=3):
    """Radial positions of the fork tines on one wing."""
    R = geo["paddle_r"]
    c0, c1 = 0.15 * R, R - FORK_TINE_W / 2
    return [c0 + (c1 - c0) * k / (n - 1) for k in range(n)]


def paddle_colliders(geo, d):
    """Sheet of kinematic spheres (radius d/2) filling the paddle solid."""
    R, H = geo["paddle_r"], geo["paddle_h"]
    s = 0.6 * d
    rr = np.arange(-R + d / 2, R - d / 2 + 1e-6, s)
    rr = rr - rr.mean()
    zz = np.arange(d / 2, H - d / 2 + 1e-6, s)
    r, z = np.meshgrid(rr, zz, indexing="ij")
    r, z = r.ravel(), z.ravel()
    # keep sphere centres whose sphere is (mostly) inside the plate
    m = paddle_solid(r, z, geo)
    return np.stack([r[m], z[m]], axis=1)


# --------------------------------------------------------------------------
# Motion profiles
# --------------------------------------------------------------------------
def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def ease_mix(u):
    """Mostly constant speed with short ramps (trapezoid velocity)."""
    u = np.clip(u, 0.0, 1.0)
    a = 0.12
    v = 1.0 / (1.0 - a)
    out = np.where(u < a, 0.5 * v * u * u / a,
          np.where(u > 1 - a, 1 - 0.5 * v * (1 - u) ** 2 / a,
                   0.5 * v * a + v * (u - a)))
    return out


def cycle_state(tc, geo, tm):
    """State within one cycle at local time tc (array ok).

    Returns tray_x (world), plunge (0 = head up, 1 = fully down)."""
    tc = np.asarray(tc, dtype=np.float64)
    reach, xw = geo["mix_reach"], geo["x_wrap"]
    t1 = tm["t_in"]
    t2 = t1 + tm["t_plunge"]
    t3 = t2 + tm["t_mix"]
    t4 = t3 + tm["t_lift"]
    t5 = t4 + tm["t_out"]
    L = xw - reach
    x = np.empty_like(tc)
    u = np.clip(tc / t1, 0, 1)
    x_in = -reach - L * (1 - u) ** 2
    u = ease_mix((tc - t2) / (t3 - t2))
    x_mix = -reach + 2 * reach * u
    u = np.clip((tc - t4) / (t5 - t4), 0, 1)
    x_out = reach + L * u ** 2
    x = np.where(tc < t1, x_in, np.where(tc < t4, x_mix, x_out))
    plunge = smooth((tc - t1) / (t2 - t1)) * (1 - smooth((tc - t3) / (t4 - t3)))
    return x, plunge


def cycle_len(tm):
    return tm["t_in"] + tm["t_plunge"] + tm["t_mix"] + tm["t_lift"] + tm["t_out"]


# --------------------------------------------------------------------------
# Mixing metric (Lacey index on full-depth columns)
# --------------------------------------------------------------------------
def mix_index(pos, ingr, geo, n_ingr, cell=7.5):
    """Mixedness in [0, 1] from the spread of composition between samples.

    The tray is cut into full-depth sample columns (cell x cell mm).  For each
    ingredient we take the standard deviation of its fraction across samples:
    s0 for fully segregated, sr for a perfect random mix (binomial sampling
    noise).  Index = (s0 - s) / (s0 - sr), averaged over ingredients.  (This is
    the standard-deviation form of the Lacey index; it tracks what the eye
    sees more closely than the variance form, which saturates early.)"""
    hx, hy = geo["tray_lx"] / 2, geo["tray_ly"] / 2
    ncx = int(round(2 * hx / cell))
    ncy = int(round(2 * hy / cell))
    ix = np.clip(((pos[:, 0] + hx) / (2 * hx) * ncx).astype(np.int64), 0, ncx - 1)
    iy = np.clip(((pos[:, 1] + hy) / (2 * hy) * ncy).astype(np.int64), 0, ncy - 1)
    c = ix * ncy + iy
    tot = np.bincount(c, minlength=ncx * ncy).astype(np.float64)
    ok = tot > 10
    M = []
    for k in range(n_ingr):
        cnt = np.bincount(c[ingr == k], minlength=ncx * ncy)
        frac = cnt[ok] / tot[ok]
        p = (ingr == k).mean()
        s = np.sqrt(np.average((frac - p) ** 2, weights=tot[ok]))
        s0 = np.sqrt(p * (1 - p))
        sr = np.sqrt(p * (1 - p) / tot[ok].mean())
        M.append((s0 - s) / (s0 - sr))
    return float(np.clip(np.mean(M), 0.0, 1.0))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grain", type=float, default=1.5, help="grain size in mm")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--ingredients", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--rev", type=float, default=TIMING["rev_per_s"])
    ap.add_argument("--t-mix", type=float, default=TIMING["t_mix"])
    ap.add_argument("--alternate", action="store_true",
                    help="reverse the paddles every pass (mixes worse: it partly undoes the last pass)")
    ap.add_argument("--paddles", type=int, choices=(1, 2), default=2,
                    help="2 = twin intermeshing paddles, 1 = one paddle on the centre line")
    ap.add_argument("--style", choices=("gate", "plate", "fork"), default="gate",
                    help="paddle shape: windowed frame, solid plate, or tines")
    ap.add_argument("--corotate", action="store_true",
                    help="twin paddles turn the same way instead of counter-rotating")
    ap.add_argument("--max-dt", type=float, default=1.0,
                    help="longest solver time step in ms (smaller = more accurate, slower)")
    ap.add_argument("--substep-factor", type=float, default=1.0,
                    help="multiply the solver substeps per frame (numerical convergence checks)")
    ap.add_argument("--no-cache", action="store_true",
                    help="skip the point cache (experiments that only need the mixing curve)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any GEO / TIMING entry, e.g. --set paddle_r=26")
    args = ap.parse_args()

    geo, tm = dict(GEO), dict(TIMING)
    for kv in args.set:
        k, v = kv.split("=")
        (geo if k in geo else tm)[k] = float(v)
    geo["paddle_style"] = args.style
    tm["rev_per_s"] = args.rev
    tm["t_mix"] = args.t_mix
    d = args.grain
    fps = args.fps
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    hx, hy = geo["tray_lx"] / 2, geo["tray_ly"] / 2

    # ---- colliders -------------------------------------------------------
    kloc = paddle_colliders(geo, d)            # (K0, 2): signed r, z
    K0 = len(kloc)
    npad = args.paddles
    K = npad * K0
    yc = np.array([-geo["paddle_sep"] / 2, geo["paddle_sep"] / 2]) if npad == 2 else np.array([0.0])
    b_sign = 1.0 if args.corotate else -1.0
    omega = 2 * np.pi * tm["rev_per_s"]

    def paddle_angles(t, npass=0):
        """Paddle A / B angle (rad).  B is 90 deg out of phase so the blades mesh."""
        a = np.interp(t, ang_t, ang_tab)
        return a, b_sign * a + np.pi / 2

    # intermesh sanity check
    if npad == 2:
        th = np.linspace(0, 2 * np.pi, 720)
        mind = 1e9
        edge = kloc[(np.abs(kloc[:, 0]) > geo["paddle_r"] - d)]
        for a in th:
            aa, bb = a, b_sign * a + np.pi / 2
            A = np.stack([edge[:, 0] * np.cos(aa), yc[0] + edge[:, 0] * np.sin(aa)], 1)
            B = np.stack([kloc[:, 0] * np.cos(bb), yc[1] + kloc[:, 0] * np.sin(bb)], 1)
            dd = np.sqrt(((A[:, None, :] - B[None, ::3, :]) ** 2).sum(-1)).min()
            mind = min(mind, dd)
        print(f"paddle intermesh clearance: {mind:.2f} mm")
        assert mind > 2.0, "paddles would collide"

    # ---- initial fill ----------------------------------------------------
    target = geo["bed_depth"]
    n_target = int(geo["tray_lx"] * geo["tray_ly"] * target / d ** 3 * 1.13)
    sp = 1.3 * d
    gx = np.arange(-hx + sp / 2, hx - sp / 2 + 1e-6, sp)
    gy = np.arange(-hy + sp / 2, hy - sp / 2 + 1e-6, sp)
    nz_l = int(np.ceil(n_target / (len(gx) * len(gy))))
    gz = d + sp * np.arange(nz_l)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pos += rng.uniform(-0.12, 0.12, pos.shape) * d
    rc = geo["tray_rc"]
    ax = np.abs(pos[:, 0]) - (hx - rc)
    ay = np.abs(pos[:, 1]) - (hy - rc)
    pos = pos[~((ax > 0) & (ay > 0) & (ax ** 2 + ay ** 2 > (rc - d) ** 2))]
    N = len(pos)
    print(f"grains: {N}   paddle collider spheres: {K}")

    n_all = N + K
    P = np.zeros((n_all, 3))
    Pprev = np.zeros((n_all, 3))
    V = np.zeros((N, 3))
    P[:N] = pos
    P[N:] = 1e6   # parked

    # grid
    h = d * 1.02
    inv_h = 1.0 / h
    ox, oy, oz = -hx - 2 * h, -hy - 2 * h, -h
    nx = int((2 * hx) / h) + 5
    ny = int((2 * hy) / h) + 5
    ztop = max(gz.max(), geo["lift"] + geo["paddle_h"]) + 6 * d
    nz = int(ztop / h) + 3
    cell_of = np.zeros(n_all, np.int64)
    start = np.zeros(nx * ny * nz + 1, np.int64)
    order = np.zeros(n_all, np.int64)

    # time stepping: keep paddle tip travel per substep below 0.4 grain
    tip = omega * geo["paddle_r"]
    nsub = int(np.ceil(tip / fps / (0.4 * d)))
    # ... and keep the time step short.  PBD lets stirred grains jitter by about
    # g*dt^2 per step, which mixes them on its own: at dt = 4 ms the first pass
    # read 55 % mixed, at 2 ms 34 %, at 1 ms 28 %, at 0.5 ms 25 %.
    nsub = max(nsub, int(np.ceil(1000.0 / (fps * args.max_dt))))
    nsub = int(round(nsub * args.substep_factor))
    dt = 1.0 / fps / nsub
    print(f"substeps/frame: {nsub}  dt={dt*1e3:.3f} ms")

    def step(n_active):
        substep(P, Pprev, V, N, n_active, dt, G, d, args.mu, 0.4, 2,
                0, hx, hy, geo["tray_rc"], geo["tray_h"], inv_h, ox, oy, oz, nx, ny, nz,
                cell_of, start, order, 4.0, 0.45 * d / dt)

    # ---- settle the fill ---------------------------------------------------
    t0 = time.time()
    for f in range(int(1.2 * fps)):
        for s in range(nsub):
            step(N)
    print(f"settled in {time.time()-t0:.1f}s; bed top ~{np.percentile(P[:N,2],99)+d/2:.1f} mm")

    # ingredient = lane across the belt, assigned from the settled position
    n_ing = args.ingredients
    ingr = np.clip(((P[:N, 1] + hy) / (2 * hy) * n_ing).astype(np.int32), 0, n_ing - 1)

    # ---- timeline ----------------------------------------------------------
    Tc = cycle_len(tm)
    t_intro = tm["t_intro"]
    n_frames = int(round((t_intro + args.passes * Tc + tm["t_in"] + tm["t_outro"]) * fps))
    # sim window inside each cycle
    w0 = tm["t_in"]
    w1 = tm["t_in"] + tm["t_plunge"] + tm["t_mix"] + tm["t_lift"] + tm["t_settle"]
    zb_low = geo["paddle_clear"]

    # paddle angle: spin up just before the plunge, spin down after the lift,
    # optionally reversing direction every pass.  Integrated on a fine grid.
    ang_t = np.arange(0.0, n_frames / fps + 1.0, dt / 2)
    tt_ = ang_t - t_intro
    cyc_ = np.clip(np.floor(tt_ / Tc), 0, None)
    tc_ = tt_ - cyc_ * Tc
    t_lift_end = tm["t_in"] + tm["t_plunge"] + tm["t_mix"] + tm["t_lift"]
    env = smooth((tc_ - (tm["t_in"] - tm["t_spin"])) / tm["t_spin"]) * \
          (1 - smooth((tc_ - t_lift_end) / tm["t_spin"]))
    env[(tt_ < 0) | (cyc_ >= args.passes)] = 0.0
    sgn = np.where((cyc_ % 2 == 1) & args.alternate, -1.0, 1.0)
    ang_tab = np.concatenate([[0.0], np.cumsum(omega * env * sgn)[:-1] * (dt / 2)])
    # revolutions each paddle has made while it is down in the powder
    _, pl_ = cycle_state(tc_, geo, tm)
    pl_ = np.where((tt_ < 0) | (cyc_ >= args.passes), 0.0, pl_)
    rev_tab = np.cumsum(omega * env * (pl_ > 0.5)) * (dt / 2) / (2 * np.pi)

    tray_x = np.zeros(n_frames)
    plunge = np.zeros(n_frames)
    ang_a = np.zeros(n_frames)
    ang_b = np.zeros(n_frames)
    cache_idx = np.zeros(n_frames, np.int32)
    mix = np.zeros(n_frames)
    pass_no = np.zeros(n_frames, np.int32)
    belt_s = np.zeros(n_frames)       # cumulative belt travel (for texture scroll)
    revs = np.zeros(n_frames)         # paddle revolutions in the powder so far

    pc2_path = os.devnull if args.no_cache else os.path.join(args.out, "particles.pc2")
    fpc = open(pc2_path, "wb")
    fpc.write(b"POINTCACHE2\0")
    fpc.write(struct.pack("<iiffi", 1, N, 0.0, 1.0, 0))  # sample count patched later
    n_samples = 0

    def write_sample():
        nonlocal n_samples
        fpc.write(P[:N].astype("<f4").tobytes())
        n_samples += 1

    write_sample()                      # sample 0: untouched lanes
    m_now = mix_index(P[:N], ingr, geo, n_ing)
    mix_after_pass = [m_now]
    revs_after_pass = [0.0]
    kprev_world = None
    t_start = time.time()
    s_acc = 0.0
    x_prev = None
    for f in range(n_frames):
        t = f / fps
        tt = t - t_intro
        if tt < 0:
            cyc, tc = 0, 0.0                 # intro: parked at the mixer entry
        else:
            cyc = int(tt // Tc)
            tc = tt - cyc * Tc
            if cyc >= args.passes:           # outro: come round once more, stop
                cyc, tc = args.passes, min(tt - args.passes * Tc, tm["t_in"])
        x, pl = cycle_state(tc, geo, tm)
        x, pl = float(x), float(pl)
        if cyc == 0 and tc < tm["t_in"]:
            x = -geo["mix_reach"]            # first cycle starts already parked
        if cyc >= args.passes:
            pl = 0.0
        in_window = (tt >= 0) and (cyc < args.passes) and (w0 <= tc < w1)
        if in_window:
            for s in range(nsub):
                ts = t + (s + 1) * dt
                tcs = ts - t_intro - cyc * Tc
                xs, pls = cycle_state(tcs, geo, tm)
                xs, pls = float(xs), float(pls)
                if pls > 1e-4:
                    zb = zb_low + geo["lift"] * (1 - pls)
                    aa, bb = paddle_angles(ts, cyc)
                    for b, ang in enumerate((aa, bb)[:npad]):
                        sl = slice(N + b * K0, N + (b + 1) * K0)
                        Pprev[sl] = P[sl]
                        P[sl, 0] = -xs + kloc[:, 0] * np.cos(ang)
                        P[sl, 1] = yc[b] + kloc[:, 0] * np.sin(ang)
                        P[sl, 2] = zb + kloc[:, 1]
                    if kprev_world is None:      # first active substep: no history
                        Pprev[N:] = P[N:]
                        kprev_world = True
                    step(n_all)
                else:
                    kprev_world = None
                    step(N)
            write_sample()
            m_now = mix_index(P[:N], ingr, geo, n_ing)
            if tc + 1.0 / fps >= w1 and len(mix_after_pass) == cyc + 1:
                mix_after_pass.append(m_now)
                revs_after_pass.append(float(np.interp(t + 1.0 / fps, ang_t, rev_tab)))
                el = time.time() - t_start
                print(f"pass {cyc+1:2d}: mixed {100*m_now:5.1f}%   "
                      f"bed top {np.percentile(P[:N,2],99.5)+d/2:5.1f} mm  "
                      f"max z {P[:N,2].max():5.1f}  [{el:.0f}s]", flush=True)
        tray_x[f] = x
        plunge[f] = pl
        aa, bb = paddle_angles(t, cyc)
        ang_a[f], ang_b[f] = aa, bb
        cache_idx[f] = n_samples - 1
        mix[f] = m_now
        revs[f] = np.interp(t, ang_t, rev_tab)
        pass_no[f] = min(cyc + (1 if (tt >= 0 and tc >= w0) else 0), args.passes)
        if x_prev is not None:
            dx = x - x_prev
            if dx < -geo["mix_reach"]:   # wrapped
                dx += 2 * geo["x_wrap"]
            s_acc += max(dx, 0.0)
        belt_s[f] = s_acc
        x_prev = x

    # patch sample count
    fpc.seek(12 + 16)
    fpc.write(struct.pack("<i", n_samples))
    fpc.close()

    # the intro: tray sits at the entry point (off-screen start of cycle 0).
    np.savez_compressed(
        os.path.join(args.out, "meta.npz"),
        ingredient=ingr.astype(np.int32),
        rand=rng.random((N, 4)).astype(np.float32),
        pos0=pos.astype(np.float32),
        tray_x=tray_x, plunge=plunge, ang_a=ang_a, ang_b=ang_b,
        cache_idx=cache_idx, mix=mix, pass_no=pass_no, belt_s=belt_s, revs=revs,
    )
    meta = dict(geo=geo, timing=tm, grain=d, fps=fps, passes=args.passes,
                n_grains=int(N), n_frames=int(n_frames), n_samples=int(n_samples),
                ingredients=n_ing, mix_after_pass=mix_after_pass,
                revs_after_pass=revs_after_pass, cycle_s=Tc,
                paddle_y=[float(v) for v in yc], paddle_style=args.style,
                corotate=bool(args.corotate), alternate=bool(args.alternate))
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("mix after each pass (%):", " ".join(f"{100*m:.0f}" for m in mix_after_pass))
    print("paddle revolutions in the powder per pass: %.2f" % (revs_after_pass[-1] / max(args.passes, 1)))
    if not args.no_cache:
        print(f"wrote {pc2_path}  ({os.path.getsize(pc2_path)/1e6:.0f} MB, {n_samples} samples, {n_frames} frames)")


if __name__ == "__main__":
    main()
