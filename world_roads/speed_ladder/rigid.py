"""Cones and drums tumbling on the ground: a small rigid-body simulator of my own (Blender's Bullet threw a kicked
cone 100 m into the air however it was set up). Each body is a set of contact points on its hull; where one goes below
the ground a spring-damper pushes back and Coulomb friction drags; gravity and air drag act on the centre of mass.
Semi-implicit Euler at 0.2 ms. Bodies don't collide with each other.
  python rigid.py roll|blast  -> /home/user/lot/props/sim_<scenario>.npz (t every 1 ms; per prop loc, quat w x y z)"""
import numpy as np, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion as M, replay_time as T
from layout import LAYOUT, COM

G = np.array([0, 0, -9.81])
DT = 2e-4


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Body:
    def __init__(self, kind, x, y, heading):
        a = math.radians(heading)
        self.q = np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])
        self.v = np.zeros(3); self.w = np.zeros(3)
        if kind == "cone":
            self.m, zc = 3.5, COM["cone"]
            ring = [(0.155 * math.cos(t), 0.155 * math.sin(t), 0.03) for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
            base = [(sx * 0.18, sy * 0.18, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 0.03)]
            pts = base + ring + [(0, 0, 0.71)] + [(0.09 * math.cos(t), 0.09 * math.sin(t), 0.33) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
            self.Ib = np.diag([0.075, 0.075, 0.035]) * 1.0
            self.area, self.cd = 0.12, 0.9
            self.k, self.c, self.mu = 3e5, 260.0, 0.65
        else:
            self.m, zc = 18.0, 0.465
            pts = [(0.29 * math.cos(t), 0.29 * math.sin(t), z) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False) for z in (0.0, 0.93)]
            pts += [(0.31 * math.cos(t), 0.31 * math.sin(t), z) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False) for z in (0.31, 0.62)]
            r, h = 0.29, 0.93
            self.Ib = np.diag([self.m * (3 * r * r + h * h) / 12] * 2 + [self.m * r * r / 2])
            self.area, self.cd = 0.5, 0.9
            self.k, self.c, self.mu = 1.2e6, 1500.0, 0.5
        self.pts = np.array(pts) - np.array([0, 0, zc])          # relative to the centre of mass
        self.p = np.array([x, y, zc + 0.0005])
        self.Ibi = np.linalg.inv(self.Ib)
        self.kick_t, self.kick_v, self.kick_w = None, None, None
        self.asleep = True

    def step(self, t):
        if self.asleep:
            if self.kick_t is not None and t >= self.kick_t:
                self.v, self.w, self.asleep = np.array(self.kick_v, float), np.array(self.kick_w, float), False
            else:
                return
        R = qmat(self.q)
        r = self.pts @ R.T
        wp = self.p + r
        F = self.m * G - 0.5 * 1.2 * self.cd * self.area * np.linalg.norm(self.v) * self.v
        Tq = np.zeros(3)
        pen = -wp[:, 2]
        hit = pen > 0
        if hit.any():
            rh = r[hit]
            u = self.v + np.cross(self.w, rh)
            fn = np.maximum(self.k * pen[hit] - self.c * u[:, 2], 0)
            ut = u.copy(); ut[:, 2] = 0
            sp = np.linalg.norm(ut, axis=1) + 1e-9
            ft = -np.minimum(self.mu * fn, 400.0 * self.m * sp)[:, None] * ut / sp[:, None]
            f = ft; f[:, 2] += fn
            F = F + f.sum(0)
            Tq = Tq + np.cross(rh, f).sum(0)
        self.v = self.v + DT * F / self.m
        Iw = R @ self.Ibi @ R.T
        self.w = self.w + DT * (Iw @ (Tq - np.cross(self.w, R @ self.Ib @ R.T @ self.w)))
        self.w *= math.exp(-0.05 * DT)
        self.p = self.p + DT * self.v
        wl = np.linalg.norm(self.w)
        if wl > 0:
            dq = np.r_[math.cos(wl * DT / 2), math.sin(wl * DT / 2) * self.w / wl]
            self.q = qmul(dq, self.q); self.q /= np.linalg.norm(self.q)


def build(sc):
    bodies = {}
    rng = np.random.default_rng(3)
    for name, (kind, x, y, h) in LAYOUT.items():
        if kind not in ("cone", "drum"): continue
        b = Body(kind, x, y, h)
        bodies[name] = b
        if sc == "roll":
            if name == "cone9":
                # the 33 kg ball at ~19 m/s, rising 2.5 m/s, clips the 3.5 kg cone above its centre of mass: the cone
                # takes most of the ball's speed (crumpling plastic, no rebound) and tumbles; the ball barely slows
                xc = x - (M.R_BALL + 0.07)
                b.kick_t = M.T_LAND + (xc - M.X_LAND) / M.VX1
                b.kick_v, b.kick_w = (16.0, 0.8, 3.4), (0.0, 13.0, 2.0)
        else:
            rad, top = (0.155, 0.71) if kind == "cone" else (0.29, 0.93)
            dy = abs(y - T.PY)
            d = math.hypot(max(dy - rad, 0.05), max(0.0, T.PZ - top))
            area, mass = (0.25, 3.5) if kind == "cone" else (0.51, 18.0)
            dv = min(35.0, M.impulse(d) * area / mass) * rng.uniform(0.85, 1.15)
            side = 1.0 if y >= T.PY else -1.0
            dirn = np.array([rng.uniform(-0.15, 0.15), side, rng.uniform(0.25, 0.45)]); dirn /= np.linalg.norm(dirn)
            axis = np.cross([0, 0, 1], [0, side, 0]) + np.array([rng.uniform(-0.3, 0.3), 0, rng.uniform(-0.4, 0.4)])
            axis /= np.linalg.norm(axis)
            spin = (1.6 if kind == "cone" else 0.9) * dv / top
            b.kick_t = M.t_hit(x, d)
            b.kick_v, b.kick_w = dirn * dv, axis * spin
            print(f"{name}: {d:.2f} m from the path, hit at {b.kick_t * 1e3:.2f} ms, {dv:.1f} m/s, {spin:.0f} rad/s")
    return bodies


if __name__ == "__main__":
    sc = sys.argv[1]
    T0, T1 = (0.0, 2.4) if sc == "roll" else (-0.05, 2.6)
    bodies = build(sc)
    n = int(round((T1 - T0) * 1000)) + 1
    out = {"t": T0 + np.arange(n) / 1000.0}
    rec = {k: (np.zeros((n, 3), np.float32), np.zeros((n, 4), np.float32)) for k in bodies}
    sub = int(round(1e-3 / DT))
    t = T0
    for i in range(n):
        for k, b in bodies.items():
            rec[k][0][i] = b.p; rec[k][1][i] = b.q
        for _ in range(sub):
            for b in bodies.values(): b.step(t)
            t += DT
    for k in bodies:
        out[k + "_loc"], out[k + "_quat"] = rec[k]
        moved = np.linalg.norm(rec[k][0][-1] - rec[k][0][0])
        if moved > 0.01:
            print(f"{k}: ends {moved:.1f} m away, peak {rec[k][0][:, 2].max():.2f} m")
    np.savez_compressed(f"/home/user/lot/props/sim_{sc}.npz", **out)
