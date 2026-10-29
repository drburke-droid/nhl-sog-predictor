"""Composite the slow-motion replay: the plate (frozen while time is slowed 2000x), bent by the shock and the hot
air; the ground lit by the glowing ball; the dust; the gravel (Blender, at the replay's sub-frame times); the glowing
ball, its wake and its sparks; the camera shaken when the blast reaches it 33 ms after the pass; the labels.
  python replay_comp.py [first last]  -> /home/user/lot/out/replay/r####.png"""
import numpy as np, math, os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as Cam, comp as C, dust as D, replay_time as T
OUT = "/home/user/lot/out/replay"
DUST = "/home/user/lot/render/replay_dust"
DEB = "/home/user/lot/render/replay_debris"
os.makedirs(OUT, exist_ok=True)
W, H = Cam.W, Cam.H_IMG
P0 = 57                                    # plate frame at the moment of the pass
WHITE, DIM = (255, 255, 255), (214, 220, 228)

# ---------------------------------------------------------------- per-pixel geometry, once
R0 = D.Renderer(D.Grid(h=1.0), scale=1.0)
dirs, org = R0.dirs, R0.org
dz = dirs[..., 2]
tg = np.where(dz < -1e-6, -org[2] / np.minimum(dz, -1e-6), np.inf)          # where the ray meets the ground
oy, oz = org[1] - T.PY, org[2] - T.PZ
ts = -(oy * dirs[..., 1] + oz * dirs[..., 2]) / (dirs[..., 1] ** 2 + dirs[..., 2] ** 2)
ts = np.clip(ts, 0, tg)
py = org[1] + ts * dirs[..., 1] - T.PY; pz = org[2] + ts * dirs[..., 2] - T.PZ
DMIN = np.sqrt(py ** 2 + pz ** 2).astype(np.float32)                         # closest approach to the path
XS = (org[0] + ts * dirs[..., 0]).astype(np.float32)                         # ... and where along it
gy, gx = np.gradient(DMIN)
gn = np.sqrt(gx ** 2 + gy ** 2) + 1e-9
GX, GY = (gx / gn).astype(np.float32), (gy / gn).astype(np.float32)          # outward from the path, on screen
ground = np.isfinite(tg)
G = org + np.where(ground, tg, 0)[..., None] * dirs                         # ground points
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def project(p):
    uv, z = Cam.project(np.atleast_2d(p))
    return uv, z


# ---------------------------------------------------------------- glow
def glow(t):
    out = np.zeros((H, W, 3), np.float32)
    xb = T.ball_x(t)
    u = XS - xb                                                              # ahead of the ball > 0
    r = DMIN
    # the ball: white hot inside its plasma sheath, a teardrop trailing back
    d = np.sqrt(u ** 2 + r ** 2)
    core = np.clip((0.115 - d) / 0.02, 0, 1) * 40.0
    env = np.where(u > 0, np.exp(-np.minimum(u, 50) / 0.035), np.exp(np.maximum(u, -50) / 0.30))
    sr = 0.12 + 0.10 * np.clip(-u, 0, 1.5)
    sheath = 18.0 * env * np.exp(-r ** 2 / (2 * sr ** 2))
    out += (core + sheath)[..., None] * np.array([0.86, 0.9, 1.0], np.float32)
    # the wake: the core of hot air, cooling from white to red; ages vary along the path
    a = np.maximum(-u / T.V, 0)
    mask = (u < 0) & (a < 6e-3)
    if mask.any():
        aa = a[mask]; rr = r[mask]
        eps = np.vectorize(T.wake_eps)(aa)
        col = np.array([T.wake_color(x) for x in np.geomspace(1e-6, 6e-3, 200)], np.float32)
        ci = np.clip((np.log(aa + 1e-9) - math.log(1e-6)) / (math.log(6e-3) - math.log(1e-6)) * 199, 0, 199).astype(int)
        rc = np.vectorize(T.core_radius)(aa)
        flick = 0.75 + 0.5 * NOISE1(XS[mask] * 3.0 + t * 900.0)
        val = eps * np.exp(-rr ** 2 / (2 * rc ** 2)) * flick
        out[mask] += val[:, None] * col[ci]
    return out


_rng = np.random.default_rng(5)
_n1 = _rng.random(4096).astype(np.float32)
def NOISE1(x):
    x = np.asarray(x, np.float32); i = np.floor(x).astype(int); f = x - i
    f = f * f * (3 - 2 * f)
    return _n1[i % 4096] * (1 - f) + _n1[(i + 1) % 4096] * f


# sparks: droplets of molten steel shed by the ball, slowed by the air within a few cm, cooling over milliseconds
NS = 2200
SP_tb = _rng.uniform(-6e-4, (T.X_EXIT + 3 - T.X_ENTER) / T.V, NS)
SP_v0 = T.V * _rng.uniform(0.05, 0.5, NS)
SP_ts = _rng.uniform(2e-5, 1.2e-4, NS)
SP_lat = _rng.normal(0, 1, (NS, 2)) * _rng.uniform(0.02, 0.25, NS)[:, None]
SP_b0 = _rng.lognormal(math.log(3.0), 0.9, NS)
SP_tc = np.where(_rng.random(NS) < 0.12, _rng.uniform(1e-3, 3e-3, NS), _rng.uniform(1.2e-4, 8e-4, NS))


def spark_pos(t):
    a = t - SP_tb
    on = a > 0
    a = np.maximum(a, 0)
    s = SP_v0 * SP_ts * (1 - np.exp(-a / SP_ts))
    spread = 1 - np.exp(-a / 4e-4)
    x = T.ball_x(SP_tb) + s
    y = T.PY + SP_lat[:, 0] * spread
    z = T.PZ + SP_lat[:, 1] * spread + 0.08 * spread
    return np.stack([x, y, z], 1), a, on


def sparks(t, expo):
    out = np.zeros((H, W, 3), np.float32)
    p1, a, on = spark_pos(t)
    p0, _, _ = spark_pos(t - expo)
    b = SP_b0 * np.exp(-a / SP_tc) * on
    keep = b > 0.05
    if not keep.any(): return out
    cols = np.array([T.wake_color(min(x * 0.6, 4e-3) + 1e-4) for x in a[keep]], np.float32)
    for k in range(5):                                                       # streak over the exposure
        f = k / 4
        uv, z = Cam.project(p0[keep] * (1 - f) + p1[keep] * f)
        x, y = uv[:, 0], uv[:, 1]
        ok = (x > 0) & (x < W - 1) & (y > 0) & (y < H - 1) & (z > 0)
        xi = x[ok].astype(int); yi = y[ok].astype(int)
        np.add.at(out, (yi, xi), (b[keep][ok] / 5)[:, None] * cols[ok])
    return cv2.GaussianBlur(out, (0, 0), 0.9) * 3.0


# gravel thrown off the path when the blast reaches it: ballistic, bouncing; trajectories precomputed every 0.5 ms
NR = 700
RK_x = _rng.uniform(-15, 15, NR); RK_y = T.PY + _rng.uniform(-0.8, 0.8, NR)
_d = np.sqrt(T.PZ ** 2 + (RK_y - T.PY) ** 2)
RK_tL = (RK_x - T.X_ENTER) / T.V + np.where(_d < T.R_WEAK, (_d / T.BLAST) ** 2, T.T_WEAK + (_d - T.R_WEAK) / T.C_SOUND)
_u = _rng.normal(0, 1, (NR, 3)); _u /= np.linalg.norm(_u, axis=1, keepdims=True)
RK_v = np.array([0, 0, 9.0]) + 6.0 * _u * _rng.random(NR)[:, None]
RK_r = 0.03 * (1 - 0.85 * _rng.random(NR))
RK_c = np.array([0.07, 0.055, 0.04]) * _rng.uniform(0.6, 1.4, NR)[:, None]
DT_R, NT_R = 5e-4, 2900


def _trajectories():
    p = np.stack([RK_x, RK_y, np.full(NR, 0.02)], 1); v = RK_v.copy()
    out = np.empty((NT_R, NR, 3), np.float32)
    for k in range(NT_R):
        out[k] = p
        v[:, 2] -= 9.81 * DT_R
        v *= math.exp(-0.05 * DT_R)
        p = p + v * DT_R
        low = p[:, 2] < 0
        p[low, 2] = -p[low, 2] * 0.3; v[low, 2] = -v[low, 2] * 0.4; v[low, :2] *= 0.6
    return out
RK_traj = _trajectories()


def rock_pos(t):
    a = t - RK_tL
    on = a > 0
    k = np.clip(a / DT_R, 0, NT_R - 1.001)
    i = k.astype(int); f = (k - i)[:, None]
    idx = np.arange(NR)
    return RK_traj[i, idx] * (1 - f) + RK_traj[i + 1, idx] * f, on


def rocks(t, expo):
    """straight-alpha layer of the gravel, drawn far to near, streaked over the exposure"""
    rgb = np.zeros((H * 2, W * 2, 3), np.float32); al = np.zeros((H * 2, W * 2), np.float32)
    p1, on = rock_pos(t); p0, _ = rock_pos(t - expo)
    lifted = on & (p1[:, 2] > 0.03)
    if not lifted.any(): return None
    uv1, z1 = Cam.project(p1); uv0, _ = Cam.project(p0)
    order = np.argsort(-z1)
    for i in order:
        if not lifted[i] or z1[i] <= 0: continue
        rad = max(0.6, 2 * Cam.F * RK_r[i] / z1[i])
        a0 = tuple(int(round(v * 8)) for v in uv0[i] * 2); a1 = tuple(int(round(v * 8)) for v in uv1[i] * 2)
        ln = math.hypot(a1[0] - a0[0], a1[1] - a0[1]) / 8
        op = min(1.0, (2 * rad + 1) / (ln + 2 * rad + 1))
        th = max(1, int(round(2 * rad)))
        cv2.line(al, a0, a1, op, th, cv2.LINE_AA, shift=3)
        cv2.line(rgb, a0, a1, tuple(float(v) for v in RK_c[i] * op), th, cv2.LINE_AA, shift=3)
    al = cv2.resize(al, (W, H), interpolation=cv2.INTER_AREA)[..., None]
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    c = np.where(al > 1e-4, rgb / np.maximum(al, 1e-4), 0)
    return np.clip(c, 0, 1), np.clip(al, 0, 1)


# ---------------------------------------------------------------- light on the ground, the shock, the shimmer
def ground_light(t):
    E = np.zeros((H, W, 3), np.float32)
    for (lx, ly, lz), c in T.lights(t):
        d2 = (G[..., 0] - lx) ** 2 + (G[..., 1] - ly) ** 2 + lz ** 2
        e = lz / (d2 ** 1.5)
        E += e[..., None] * np.array(c, np.float32)
    return np.where(ground[..., None], E, 0)


def displacement(t):
    """px to move the plate by: the shock front bends light where a ray grazes it; hot air shimmers inside"""
    a = T.age(XS, t)
    disp = np.zeros((H, W), np.float32)
    on = a > 0
    if on.any():
        R = np.vectorize(T.shock_r)(np.where(on, a, 0))
        amp = 7.0 * np.minimum(1.0, T.R_WEAK / np.maximum(R, 1e-3)) ** 1.1
        w = 0.05 + 0.03 * R
        q = (DMIN - R) / w
        ring = -q * np.exp(0.5 - 0.5 * q * q)
        disp = np.where(on & (R < 14), amp * ring, 0).astype(np.float32)
    dx, dy = disp * GX, disp * GY
    # shimmer: the hot core, rising and lingering
    hot = np.where(on, np.exp(-np.maximum(DMIN - 0.6, 0) / 0.6) * np.clip(np.where(on, a, 0) / 2e-3, 0, 1), 0)
    if hot.any():
        ph = t * 60.0
        nx = cv2.resize(SHIM[int(ph) % 8], (W, H), interpolation=cv2.INTER_CUBIC) * (1 - ph % 1) + \
             cv2.resize(SHIM[(int(ph) + 1) % 8], (W, H), interpolation=cv2.INTER_CUBIC) * (ph % 1)
        ny = cv2.resize(SHIM[(int(ph) + 3) % 8], (W, H), interpolation=cv2.INTER_CUBIC)
        dx = dx + 1.6 * hot * (nx - 0.5); dy = dy + 1.6 * hot * (ny - 0.5)
    return dx.astype(np.float32), dy.astype(np.float32)


SHIM = [_rng.random((H // 10, W // 10)).astype(np.float32) for _ in range(8)]


def warp(img, dx, dy):
    return cv2.remap(img, XX - dx, YY - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def label(slow, t):
    if slow >= T.SLOW - 1e-6: s = f"replay, {T.SLOW:,}× slower"
    elif slow > 1.05: s = f"replay, {slow:,.0f}× slower"
    else: s = "replay, real speed"
    ms = t * 1e3
    tt = f"{ms:.3f} ms" if ms < 10 else (f"{ms:.1f} ms" if ms < 1000 else f"{t:.2f} s")
    return [("17,500 mph", 64, "Bold", WHITE), (s, 30, "Medium", DIM), (f"t = {tt}", 26, "Medium", DIM)]


def frame(idx, t, slow, expo):
    bg = C.plate_loop(P0 + max(0, int(round(t * T.FPS))))
    dx, dy = displacement(t)
    bg = warp(bg, dx, dy)
    dl = C.dust(f"{DUST}/d{idx:04d}.png", f"{DUST}/g{idx:04d}.png")
    if dl is not None:
        bg = bg * (1 - C.SUN_FRACTION * (1 - dl[2]))
    bg = bg * (1 + ground_light(t) / 2.3)
    img = C.over(bg, rocks(t, expo), soften=0)
    if dl is not None:
        img = img * dl[1] + dl[0]
    gl = glow(t) + sparks(t, expo)
    if dl is not None: gl = gl * np.sqrt(dl[1])
    img = img + gl
    img = C.bloom(img, thresh=0.9, sigma=(4, 16, 60), gain=(0.6, 0.4, 0.3))
    img = C.shake(img, t - 0.033, amp=16.0, seed=3)
    img = C.grain(img, idx)
    pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((C.H, C.W, 3), np.uint8)
    out[C.TOP:C.TOP + C.PH] = pic
    return C.label(out, label(slow, t - T.timeline()[0][0]))


if __name__ == "__main__":
    tl = T.timeline()
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, len(tl) - 1)
    for idx in range(a, min(b, len(tl) - 1) + 1):
        t, slow = tl[idx]
        expo = 0.5 / (T.FPS * slow)                                          # a 180-degree shutter
        cv2.imwrite(f"{OUT}/r{idx:04d}.png", frame(idx, t, slow, expo)[..., ::-1])
        print("comp", idx, flush=True)
