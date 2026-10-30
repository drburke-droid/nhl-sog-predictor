"""The dust a supersonic machine raises, simulated once and rendered at every moment its shot and its replay need.
  python flyby_dust.py x1 -> /home/user/lot/render/fly_x1_dust/d####.png, g####.png (the real-time shot's frames)
                             /home/user/lot/render/fly_x1_replay_dust/... (the replay's)

The X-1 at Mach 1.06, 6 m up: its Mach cone sweeps over the ground (it reaches a point at lateral distance d from the
flight line (x + d sqrt(M^2 - 1)) / v after the nose crossed x = 0). Right under it the boom is around 15 kPa
(the shock and its reflection off the ground; rough: a 9.4 m aircraft, near-field N-wave falling as d^-3/4), and the
air behind it jerks outward at dp / (rho c), 36 m/s for a moment, then back. Fine dust jumps off the sand and
gravel, a sheet thickest under the path, and hangs; the rocket's wake drags air backward behind it."""
import numpy as np, math, os, sys, time, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust as D, flight as F
from scipy.ndimage import zoom, gaussian_filter

NAME = sys.argv[1] if len(sys.argv) > 1 else "x1"
CFG = {
    "x1": dict(kind="boom", dp0=15000.0, d0=6.0, dp_min=3500.0, n_fine=700_000, n_sand=60_000, lift=0.1, unit=0.028,
               wake=7.0, grid=(-17, 17, 9, 31, 6), vis=(-16, 16, 9.5, 30.5, 5)),
    # the F-35 at Mach 1.6: a longer, heavier aircraft, a stronger boom (~30 kPa under it), and the afterburner's jet
    # dragging the air (and the dust) behind it
    "f35": dict(kind="boom", dp0=30000.0, d0=6.0, dp_min=4500.0, n_fine=900_000, n_sand=80_000, lift=0.1, unit=0.03,
                wake=22.0, grid=(-17, 17, 7, 33, 7), vis=(-16, 16, 7.5, 32.5, 6)),
    # the X-15's line blast (hyper.py): dust thrown off the ground when the blast reaches it, at ~6% of the impulse
    # per area in m/s (16 m/s under the path), outward and up; a core of heated air 6 m up, expanding, then rising
    "x15": dict(kind="blast", n_fine=1_300_000, n_sand=160_000, lift=0.045, unit=0.11, area=8.0, wake=12.0,
                grid=(-17, 17, 3, 32, 12), vis=(-16, 16, 3.5, 31.5, 11)),
}[NAME]
M = F.MACHINES[NAME]
V, ALT, PY = M["speed"], M["alt"], F.lane_y(NAME)
RACE = F.RACE and NAME in F.RACE_LANES
H_AIR, H_VIS = 0.15, 0.1
if RACE:            # the race's lane (race.py): the grids follow it, wide enough for the picture there, as coarse as far
    k_ = PY / F.FLY_Y
    half = 0.8 * (PY + 13) + 4
    yb = (PY - 16, PY + 12) if CFG["kind"] == "blast" else (PY - 13, PY + 13)
    CFG = dict(CFG, grid=(-half, half, yb[0], yb[1], CFG["grid"][4]),
               vis=(-half + 1, half - 1, yb[0] + 0.5, yb[1] - 0.5, CFG["vis"][4]))
    H_AIR, H_VIS = 0.15 * k_, 0.1 * k_
MACH = V / 343.0
COT = math.sqrt(max(MACH ** 2 - 1, 0.0))
SCALE = float(os.environ.get("DUST_SCALE", "1.0"))
R_ = "/home/user/lot/render"


def dp(d):
    return CFG["dp0"] * (CFG["d0"] / np.maximum(d, CFG["d0"])) ** 0.75


def arrival(x, y):
    if CFG["kind"] == "blast":
        import hyper as Hy
        return Hy.t_hit(x, y, 0.0)
    d = np.hypot(y - PY, ALT)
    return (x + d * COT) / V


def launch_speed(d):
    """m/s the ground's dust leaves at, d m from the path"""
    if CFG["kind"] == "blast":
        import hyper as Hy
        return CFG["lift"] * Hy.impulse(d)
    return CFG["lift"] * dp(d) / (1.2 * 343.0)                # a fraction of the air's jerk behind the shock


def weight(d):
    """how much of the ground's dust comes up, d m from the path"""
    if CFG["kind"] == "blast":
        return np.clip(1.3 - d / 22.0, 0, 1)
    return np.clip((dp(d) - CFG["dp_min"]) / (CFG["dp0"] - CFG["dp_min"]), 0, 1) ** 1.5


def setup():
    rng = np.random.default_rng(11)
    x0, x1, y0, y1, z1 = CFG["grid"]
    air = D.Grid(x0=x0, x1=x1, y0=y0, y1=y1, z1=z1, h=H_AIR)
    blast = CFG["kind"] == "blast"
    sim = D.Sim(air, alpha=0.0, beta=6.0 if blast else 2.0, eps=6.0 if blast else 5.0, drag=0.2 if blast else 0.25,
                cool=0.35 if blast else 0.5, settle=0.0, iters=80 if blast else 60)
    sim.salt = (1e9, 0.0, 0.0)
    X, Y, Z = air.coords()
    near = np.exp(-(Y - PY) ** 2 / (2 * (6.0 if blast else 4.0) ** 2)) * np.exp(-Z / (2.5 if blast else 1.2))
    for comp_, seed in ((air.u, 5), (air.v, 6), (air.w, 7)):
        comp_ += ((5.0 if blast else 2.0) * (D.noise3(air.n, 5, seed, 2) - 0.5) * near).astype(np.float32)
    # the rocket's wake: air dragged backward behind the plane, spreading down toward the ground
    r2 = (Y - PY) ** 2 + (Z - ALT) ** 2
    air.u -= (CFG["wake"] * np.exp(-r2 / (2 * 1.8 ** 2))).astype(np.float32)
    if NAME == "f35":                                   # the afterburner's jet spreads down to the ground behind it
        air.u -= (0.45 * CFG["wake"] * np.exp(-(Y - PY) ** 2 / (2 * 3.0 ** 2)) * np.exp(-Z / 2.5)).astype(np.float32)
        air.w += (2.5 * np.exp(-(Y - PY) ** 2 / (2 * 3.0 ** 2)) * np.exp(-Z / 1.5)).astype(np.float32)
    core = None
    if blast:
        # the heated core along the path, and behind the blast the air near the ground thrown outward
        lump = 0.5 + D.noise3(air.n, 9, 11, 3)
        air.T[:] = (2.5 * np.exp(-r2 / (2 * 1.6 ** 2)) * lump).astype(np.float32)
        SIG = 1.4
        core = (np.exp(-r2 / (2 * SIG ** 2)) / (2 * math.pi * SIG ** 2) * lump).astype(np.float32)
        dyA = Y - PY
        air.v += (10.0 * np.tanh(dyA / 3.0) * np.exp(-dyA ** 2 / (2 * 6.0 ** 2)) * np.exp(-Z / 1.2)).astype(np.float32)
        air.w += (6.0 * np.exp(-dyA ** 2 / (2 * 3.0 ** 2)) * np.exp(-Z / 3.0)).astype(np.float32)

    def avail(x):
        a = np.ones_like(x)
        a[(x > -4.2) & (x < 3.5)] = 0.35                # asphalt
        a[(x > -6.2) & (x <= -4.2)] = 0.7               # gravel shoulder
        return a

    # the ground's patchiness on a 360 x 220 lattice: 0.1 m cells over x -18..18, y 9..31 (clamped beyond), or over the
    # race lane's grid
    gx0, gx1, gy0, gy1 = (x0 - 1, x1 + 1, y0, y1) if RACE else (-18, 18, 9, 31)

    def ground_noise(seed):
        r = np.random.default_rng(seed)
        out = 0
        for cells, amp in ((0.5, 0.35), (1.5, 0.4), (4.0, 0.25)):
            nx, ny = int((gx1 - gx0) / cells) + 4, int((gy1 - gy0) / cells) + 4
            out = out + amp * zoom(r.random((nx, ny)), (360 / nx * 1.0001, 220 / ny * 1.0001), order=3)[:360, :220]
        return (out - out.mean()) / out.std()
    GN = ground_noise(23)

    def emit(n, tau, vs, gain):
        x = rng.uniform(x0, x1, n); y = rng.uniform(y0 + 0.5, y1 - 0.5, n)
        d = np.hypot(y - PY, ALT)
        keep = rng.random(n) < weight(d) * avail(x)
        x, y, d = x[keep], y[keep], d[keep]; n = len(x)
        c = GN[np.clip(((x - gx0) / (gx1 - gx0) * 360).astype(int), 0, 359),
               np.clip(((y - gy0) / (gy1 - gy0) * 220).astype(int), 0, 219)]
        sp = gain * launch_speed(d) * np.exp(0.25 * c) * rng.lognormal(0, 0.6 if blast else 0.35, n)
        out_ = 0.9 if blast else 0.5
        dirn = np.stack([(0.0 if blast else 0.33) + 0.2 * rng.standard_normal(n),
                         out_ * np.sign(y - PY) + 0.2 * rng.standard_normal(n), 1.0 + 0.3 * rng.standard_normal(n)], 1)
        dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
        P = np.stack([x, y, rng.uniform(0, 0.05, n)], 1).astype(np.float32)
        # not all at once: the ground lets go raggedly, some grains a few milliseconds after the front passes
        late = rng.exponential(0.004, n) * (1.0 if blast else 0.3)
        return P, (dirn * sp[:, None]).astype(np.float32), np.full(n, tau, np.float32), np.full(n, vs, np.float32), \
            np.exp(0.8 * c).astype(np.float32), (arrival(x, y) + late).astype(np.float32)

    fine = emit(CFG["n_fine"], 0.3, 0.08, 1.0)
    sand = emit(CFG["n_sand"], 0.9, 6.0, 0.6)
    cat = lambda k: np.concatenate([fine[k], sand[k]])
    return dict(air=air, sim=sim, P=cat(0), V=cat(1), tau=cat(2), vs=cat(3), core=core,
                mass=np.concatenate([fine[4], 0.35 * sand[4]]).astype(np.float32), tl=cat(5))


def props_tmax(R, path, static):
    p = path if os.path.exists(path) else static
    if not os.path.exists(p): return None
    d = np.load(p)
    return D.ray_tmax(R, d["depth"].astype(np.float32), d["rgba"][..., 3].astype(np.float32))


def save(out, f, rgb, tr, gs):
    cv2.imwrite(f"{out}/d{f:04d}.png", np.clip(rgb[..., ::-1] * 32767.5, 0, 65535).astype(np.uint16))
    cv2.imwrite(f"{out}/g{f:04d}.png", np.clip(np.stack([np.zeros_like(tr), gs, tr], -1) * 65535, 0, 65535).astype(np.uint16))


if __name__ == "__main__":
    S = setup()
    air, sim, P, Vp, tau, vs, mass, tl = (S[k] for k in ("air", "sim", "P", "V", "tau", "vs", "mass", "tl"))
    alive = np.ones(len(P), np.bool_)
    core = S["core"]
    t_start = float(tl.min()) if core is None else min(float(tl.min()), -0.01)
    print("parcels", len(P), "first lifts at", round(t_start, 3), flush=True)
    # every moment wanted: (flight time, output dir, frame)
    SUF = os.environ.get("OUTSUF", "")
    o_rt, o_rp = f"{R_}/fly_{NAME}_dust", f"{R_}/fly_{NAME}_replay_dust{SUF}"
    os.makedirs(o_rt, exist_ok=True); os.makedirs(o_rp, exist_ok=True)
    want = [(t, o_rt, n, f"{R_}/fly_{NAME}/p{n:04d}.npz") for n, t in F.frames(NAME) if t >= t_start]
    want += [(t, o_rp, n, f"{R_}/fly_{NAME}_replay{SUF}/p{n:04d}.npz") for n, (t, _) in enumerate(F.replay(NAME))
             if t >= t_start]
    if RACE:                                            # the race's frames instead (race.py)
        import race as RC
        o_rc = f"{R_}/fly_{NAME}_race_dust"
        os.makedirs(o_rc, exist_ok=True)
        want = [(RC.own(NAME, T), o_rc, n, f"{R_}/fly_{NAME}_race/p{n:04d}.npz") for n, (T, _) in enumerate(RC.timeline())
                if RC.own(NAME, T) >= t_start]
    want.sort(key=lambda w: w[0])
    vx0, vx1, vy0, vy1, vz1 = CFG["vis"]
    vis = D.Grid(x0=vx0, x1=vx1, y0=vy0, y1=vy1, z1=vz1, h=H_VIS)
    R = D.Renderer(vis, scale=SCALE, kappa=1.3)
    static = f"{R_}/fly_{NAME}/static.npz"
    t = t_start
    for tw, out, n, propsfile in want:
        t0 = time.time()
        while t < tw - 1e-7:
            dt = min(tw - t, 0.002 if t < 0.15 else 1 / 75)
            if core is not None:                        # the heated core expanding (as the ball's blast did)
                a = max(t, 0.0)
                sim.src = (core * (CFG["area"] / 0.05) * math.exp(-a / 0.05)).astype(np.float32) if a < 0.3 else None
            sim.step(dt)
            live = alive & (tl <= t)
            D.move_parcels(P, Vp, tau, vs, live, air.u, air.v, air.w, air.o, air.h, dt, 0.3)
            alive &= live | (tl > t)
            t += dt
        if os.path.exists(f"{out}/d{n:04d}.png"): continue
        live = alive & (tl <= t)
        vis.d[:] = 0
        D.splat(P, mass * CFG["unit"] / (vis.h ** 3) * 1e-3, live, vis.d, vis.o, vis.h)
        vis.d[:] = gaussian_filter(vis.d, 0.6)
        rgb, tr, gs = D.render_l(R, [], tmax=props_tmax(R, propsfile, static))
        save(out, n, rgb, tr, gs)
        print(f"{os.path.basename(out)} {n} t={t:.4f} live {live.sum()} maxd {vis.d.max():.2f} T.min {tr.min():.3f} "
              f"{time.time() - t0:.1f}s", flush=True)
    print("ALLDONE")
