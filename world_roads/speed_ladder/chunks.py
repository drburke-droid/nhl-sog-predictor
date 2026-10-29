"""Small flying things drawn straight into the frame (too small to be worth Blender): ballistic chunks with a bounce,
streaked over the exposure; asphalt grit where the 50 mph ball hits; the glass of the car's windows glinting."""
import numpy as np, math, cv2
import camera as Cam, motion as M
W, H = Cam.W, Cam.H_IMG


class Ballistic:
    def __init__(self, t0, P0, V0, rest=0.3, fric=0.5, drag=0.3, dt=2e-3, tmax=3.0):
        """trajectories precomputed from t0 (array per chunk) every dt"""
        n = len(P0)
        self.t0, self.dt = np.asarray(t0, float), dt
        steps = int(tmax / dt) + 1
        p, v = np.array(P0, float), np.array(V0, float)
        self.traj = np.empty((steps, n, 3), np.float32)
        for k in range(steps):
            self.traj[k] = p
            v[:, 2] -= 9.81 * dt
            v *= math.exp(-drag * dt)
            p = p + v * dt
            low = p[:, 2] < 0
            p[low, 2] = -p[low, 2] * rest
            v[low, 2] = -v[low, 2] * rest; v[low, :2] *= fric

    def at(self, t):
        a = t - self.t0
        on = a > 0
        k = np.clip(a / self.dt, 0, len(self.traj) - 1.001)
        i = k.astype(int); f = (k - i)[:, None]
        idx = np.arange(len(a))
        return self.traj[i, idx] * (1 - f) + self.traj[i + 1, idx] * f, on


def draw(p1, p0, radius, color, on, opacity=1.0):
    """straight-alpha layer (rgb, a) of chunks streaked from p0 to p1, far to near, 2x supersampled"""
    rgb = np.zeros((H * 2, W * 2, 3), np.float32); al = np.zeros((H * 2, W * 2), np.float32)
    uv1, z1 = Cam.project(p1); uv0, _ = Cam.project(p0)
    for i in np.argsort(-z1):
        if not on[i] or z1[i] <= 0: continue
        rad = max(0.5, 2 * Cam.F * radius[i] / z1[i])
        a0 = tuple(int(round(v * 16)) for v in uv0[i]); a1 = tuple(int(round(v * 16)) for v in uv1[i])
        ln = math.hypot(a1[0] - a0[0], a1[1] - a0[1]) / 8
        op = opacity * min(1.0, (2 * rad + 1) / (ln + 2 * rad + 1))
        th = max(1, int(round(2 * rad)))
        cv2.line(al, a0, a1, op, th, cv2.LINE_AA, shift=3)
        cv2.line(rgb, a0, a1, tuple(float(c) * op for c in color[i]), th, cv2.LINE_AA, shift=3)
    al = cv2.resize(al, (W, H), interpolation=cv2.INTER_AREA)[..., None]
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    if al.max() <= 1e-4: return None
    return np.clip(np.where(al > 1e-4, rgb / np.maximum(al, 1e-4), 0), 0, 1), np.clip(al, 0, 1)


# ---------------------------------------------------------------- asphalt grit at the 50 mph impact
_r = np.random.default_rng(21)
NG = 140
_ang = _r.uniform(0, 2 * math.pi, NG)
_sp = _r.lognormal(math.log(2.5), 0.5, NG)
GRIT = Ballistic(np.full(NG, M.T_LAND), np.stack([np.full(NG, M.X_LAND), np.full(NG, 12.0), np.full(NG, 0.02)], 1),
                 np.stack([np.cos(_ang) * _sp + 0.3 * M.VX1 * _r.random(NG), np.sin(_ang) * _sp, _r.uniform(0.8, 4.0, NG)], 1),
                 rest=0.25, fric=0.4, tmax=2.0)
GRIT_R = _r.uniform(0.003, 0.012, NG)
GRIT_C = np.array([0.022, 0.021, 0.020]) * _r.uniform(0.6, 1.8, NG)[:, None]


def grit(t, expo):
    p1, on = GRIT.at(t); p0, _ = GRIT.at(t - expo)
    return draw(p1, p0, GRIT_R, GRIT_C, on & (p1[:, 2] > 0.004))


# ---------------------------------------------------------------- the parked car's windows blowing in
NSH = 90
_car = M.CAR
TH = M.t_hit(_car["x"], _car["d"])
_side = _r.random(NSH) < 0.45
ry0, ry1, rz0, rz1 = _car["rear"]
sx, sy0, sy1, sz0, sz1 = _car["side"]
_px = _car["x"] + np.where(_side, sx - 0.01, _r.uniform(-0.55, 0.55, NSH))
_py = _car["y"] + np.where(_side, _r.uniform(sy0, sy1, NSH), _r.uniform(ry0, ry1, NSH))
_pz = np.where(_side, _r.uniform(sz0, sz1, NSH), _r.uniform(rz0, rz1, NSH))
_v = np.stack([np.where(_side, _r.uniform(-1.5, 0.5, NSH), _r.uniform(-2, 2, NSH)),
               np.where(_side, _r.uniform(-1, 3, NSH), _r.uniform(-2.5, 1.0, NSH)), _r.uniform(-0.5, 2.5, NSH)], 1)
SHARDS = Ballistic(TH + _r.uniform(0, 1.5e-3, NSH), np.stack([_px, _py, _pz], 1), _v, rest=0.2, fric=0.3, tmax=3.0)
_ph = _r.uniform(0, 2 * math.pi, NSH); _om = _r.uniform(20, 70, NSH)


def shards(t, expo):
    """additive glints: tempered glass cubes catching the sun as they tumble"""
    out = np.zeros((H, W, 3), np.float32)
    p, on = SHARDS.at(t)
    if not on.any(): return out
    flash = 0.08 + 6.0 * np.maximum(0, np.sin(_ph + _om * t)) ** 24
    uv, z = Cam.project(p)
    ok = on & (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < W - 1) & (uv[:, 1] > 0) & (uv[:, 1] < H - 1)
    np.add.at(out, (uv[ok, 1].astype(int), uv[ok, 0].astype(int)), flash[ok, None] * np.array([0.9, 0.95, 1.0]))
    return cv2.GaussianBlur(out, (0, 0), 0.7) * 2.0
