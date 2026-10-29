"""Dust for the slow-motion replay: the same parcels as the real-time pass (blast_dust.setup), but each leaves the
ground only when the blast reaches it: the ball's passage plus the time the shock takes to reach that spot on the
ground. Just behind the shock the air moves at hundreds of metres a second, so a thin skin of dust jumps ~0.2 m in the
first millisecond (an extra displacement that saturates), then rises at its launch speed. Until 3 ms the motion is
kinematic (the air itself has barely moved); after that the air solver takes over, stepped to each replay frame.
Lit by the sun, the sky and the glowing ball.
  python replay_dust.py [first last]  -> /home/user/lot/render/replay_dust/d####.png, g####.png"""
import numpy as np, math, os, sys, time, cv2
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust as D, blast_dust as B, replay_time as T
OUT = "/home/user/lot/render/replay_dust"
os.makedirs(OUT, exist_ok=True)
T_SIM = 3.0e-3
TAU_E, K_E = 8.0e-4, 8.0
UNIT = 0.11

S = B.setup()
air, sim, core = S["air"], S["sim"], S["core"]
p0 = S["P"].copy(); Vl = S["V"].copy(); tau, vs, mass = S["tau"], S["vs"], S["mass"]
n = len(p0)
dy = p0[:, 1] - T.PY
dist = np.sqrt(T.PZ ** 2 + dy ** 2)                       # from the path to the spot on the ground
# when the shock reaches it: invert R(age)
hit = np.where(dist < T.R_WEAK, (dist / T.BLAST) ** 2, T.T_WEAK + (dist - T.R_WEAK) / T.C_SOUND)
tL = (p0[:, 0] - T.X_ENTER) / T.V + hit
spd = np.linalg.norm(Vl, axis=1) + 1e-6
dirn = Vl / spd[:, None]
_r = np.random.default_rng(31)


def ground_noise2(seed, cells=(0.25, 0.8, 2.5), amps=(0.4, 0.4, 0.2)):
    """standardized 2D value noise over the ground, x -18..18, y 6..18, 0.05 m cells"""
    from scipy.ndimage import zoom
    rr = np.random.default_rng(seed); out = 0
    for c, a in zip(cells, amps):
        nx, ny = int(36 / c) + 4, int(12 / c) + 4
        out = out + a * zoom(rr.random((nx, ny)), (720 / nx * 1.0001, 240 / ny * 1.0001), order=3)[:720, :240]
    return (out - out.mean()) / out.std()
GN2 = ground_noise2(32)
cg = GN2[np.clip(((p0[:, 0] + 18) / 0.05).astype(int), 0, 719), np.clip(((p0[:, 1] - 6) / 0.05).astype(int), 0, 239)]
# fingers and jets: some patches of ground jump much higher than others
s_inf = (K_E * spd * TAU_E * np.exp(-np.maximum(dist - 0.6, 0) / 0.8) * np.exp(1.1 * cg - 0.6) *
         _r.lognormal(0, 0.35, n)).astype(np.float32)
dir_e = dirn + 0.4 * _r.standard_normal((n, 3)).astype(np.float32)      # the early jump is ragged
dir_e[:, 2] = np.abs(dir_e[:, 2])
dir_e /= np.linalg.norm(dir_e, axis=1, keepdims=True)

vis = D.Grid(x0=-16, x1=16, y0=5, y1=20, z1=7, h=0.07)
R = D.Renderer(vis, scale=1.0, kappa=1.3)
first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 10 ** 6)

P = p0.copy(); V = np.zeros_like(Vl)
launched = np.zeros(n, np.bool_); alive = np.zeros(n, np.bool_)
t_now = None


def kinematic(t):
    a = np.maximum(t - tL, 0.0)
    on = t >= tL
    return p0 + Vl * a[:, None] * on[:, None], on


def render_at(t, Prender, live, idx):
    vis.d[:] = 0
    lift = np.clip((Prender[:, 2] - p0[:, 2]) / 0.08, 0, 1) * (1 - np.exp(-np.maximum(t - tL, 0) / 6e-4))
    lift = lift.astype(np.float32)
    D.splat(Prender, (mass * lift * UNIT / (vis.h ** 3) * 1e-3).astype(np.float32), live, vis.d, vis.o, vis.h)
    vis.d[:] = gaussian_filter(vis.d, 0.45)
    zmax = float(Prender[live, 2].max()) + 0.6 if live.any() else 0.5
    rgb, tr, gs = D.render_l(R, T.lights(t), zmax=zmax)
    cv2.imwrite(f"{OUT}/d{idx:04d}.png", np.clip(rgb[..., ::-1] * 32767.5, 0, 65535).astype(np.uint16))
    cv2.imwrite(f"{OUT}/g{idx:04d}.png", np.clip(np.stack([np.zeros_like(tr), gs, tr], -1) * 65535, 0, 65535).astype(np.uint16))
    return zmax, tr.min()


def offset(t):
    a = np.maximum(t - tL, 0.0)
    return dir_e * (s_inf * (1 - np.exp(-a / TAU_E)))[:, None]


tl = T.timeline()
for idx, (t, slow) in enumerate(tl):
    t0 = time.time()
    if t < T_SIM:
        Pk, on = kinematic(t)
        Pr = (Pk + offset(t)).astype(np.float32)
        live = on.copy()
    else:
        if t_now is None:                                  # hand the parcels to the air solver
            Pk, on = kinematic(T_SIM)
            P[:] = Pk; V[:] = Vl * on[:, None]
            launched[:] = on; alive[:] = on
            t_now = T_SIM
        while t_now < t - 1e-9:
            dt = min(1.0 / 75, t - t_now)
            sim.src = (core * (B.AREA / B.TAU) * math.exp(-t_now / B.TAU)).astype(np.float32) if t_now < 6 * B.TAU else None
            sim.step(dt)
            D.move_parcels(P, V, tau, vs, alive, air.u, air.v, air.w, air.o, air.h, dt, 0.3)
            t_now += dt
            new = (~launched) & (tL <= t_now)
            if new.any():
                P[new] = p0[new]; V[new] = Vl[new]
                launched |= new; alive |= new
        Pr = (P + offset(t)).astype(np.float32)
        live = alive.copy()
    if first <= idx <= last:
        zmax, tmin = render_at(t, Pr, live, idx)
        print(f"replay {idx} t={t * 1e3:.3f} ms live {live.sum()} zmax {zmax:.2f} Tmin {tmin:.3f} {time.time() - t0:.1f}s", flush=True)
    if idx > last: break
