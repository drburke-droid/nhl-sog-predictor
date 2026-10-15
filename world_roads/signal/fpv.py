"""First-person beacon sequence for the long video: the real terrain of fires 18 -> 19 -> 20 of the loop
(Mont Blanc -> the northern Apennines -> the Dolomites) at blue hour.

  close on Mont Blanc's beacon -> pull back 311 km to the Apennine summit, where it is a speck on the
  horizon -> the keeper lights the next fire -> pull back 244 km to the Dolomites -> light -> rise.

Terrain: fpv_terrain.py (Terrarium zoom 11 + zoom 13 around the summits), raymarched with Earth curvature
and the same refraction as the search (k = 0.13). Fires, keepers and sparks are composited sprites.

    python fpv.py still 13.0     -> out/fpv_still.png
    python fpv.py                -> out/fpv.mp4 (silent)
    python fpv.py alt            -> out/fpv_alt_tail.mp4: the rise (t >= ALT_FROM) with Video 2's last caption;
                                    Video 2 joins it to the first ALT_FROM s of fpv.mp4
"""
import os, sys, json, math, time, subprocess
import numpy as np
import numba as nb
import cv2
import fpv_terrain as FT
import style as ST

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
R = 6371.0088
REFF = R / (1 - 0.13)
W, H, FPS = 1920, 1080, 30
RS = 2                                            # terrain is raymarched at 1/RS resolution
PYRE = 0.010                                      # flame top above the summit (km), as in the search
SUN_AZ, SUN_EL = math.radians(296), math.radians(-7.5)
MOON = np.array([0.55, -0.62, 0.56]); MOON /= np.linalg.norm(MOON)


# ------------------------------------------------------------------ terrain sampling + raymarch
@nb.njit(cache=True, fastmath=True)
def bil(g, fx, fy):
    xi = int(fx); yi = int(fy)
    ax = fx - xi; ay = fy - yi
    return (g[yi, xi] * (1 - ax) + g[yi, xi + 1] * ax) * (1 - ay) + (g[yi + 1, xi] * (1 - ax) + g[yi + 1, xi + 1] * ax) * ay


@nb.njit(cache=True, fastmath=True)
def hgt(T, cx0, cy0, cd, F, fpos, fd, x, y):
    fx = (x - cx0) / cd; fy = (y - cy0) / cd
    ny, nx = T.shape
    if fx < 0 or fy < 0 or fx >= nx - 1 or fy >= ny - 1:
        hc = 0.0
    else:
        hc = bil(T, fx, fy)
    nf = F.shape[1]
    for k in range(F.shape[0]):
        gx = (x - fpos[k, 0]) / fd; gy = (y - fpos[k, 1]) / fd
        if gx >= 0 and gy >= 0 and gx < nf - 1 and gy < nf - 1:
            e = min(min(gx, gy), min(nf - 1 - gx, nf - 1 - gy))
            hf = bil(F[k], gx, gy)
            if e >= 60:
                return hf
            w = e / 60.0
            return hf * w + hc * (1 - w)
    return hc


@nb.njit(parallel=True, cache=True, fastmath=True)
def march(T, cx0, cy0, cd, F, fpos, fd, cam, dirs, hmax, tmax, out_t, out_n, out_h):
    for i in nb.prange(dirs.shape[0]):
        dx = dirs[i, 0]; dy = dirs[i, 1]; dz = dirs[i, 2]
        hxy = math.sqrt(dx * dx + dy * dy)
        t = 0.0005
        tp = 0.0
        hit = False
        for it in range(6000):
            px = cam[0] + dx * t; py = cam[1] + dy * t; pz = cam[2] + dz * t
            dh = hxy * t
            drop = dh * dh / (2 * REFF)
            above = pz - (hgt(T, cx0, cy0, cd, F, fpos, fd, px, py) - drop)
            if above <= 0:
                hit = True
                break
            if pz + drop > hmax and dz + hxy * dh / REFF >= 0:
                break
            if t > tmax:
                break
            tp = t
            t += max(above * 0.35, t * 0.0006, 0.0012)
        if not hit:
            out_t[i] = -1.0
            continue
        a = tp; b = t
        for _ in range(12):
            m = 0.5 * (a + b)
            dh = hxy * m
            if cam[2] + dz * m < hgt(T, cx0, cy0, cd, F, fpos, fd, cam[0] + dx * m, cam[1] + dy * m) - dh * dh / (2 * REFF):
                b = m
            else:
                a = m
        t = b
        px = cam[0] + dx * t; py = cam[1] + dy * t
        e = 0.012 if t < 6 else (0.04 if t < 30 else 0.09)
        hx0 = hgt(T, cx0, cy0, cd, F, fpos, fd, px - e, py); hx1 = hgt(T, cx0, cy0, cd, F, fpos, fd, px + e, py)
        hy0 = hgt(T, cx0, cy0, cd, F, fpos, fd, px, py - e); hy1 = hgt(T, cx0, cy0, cd, F, fpos, fd, px, py + e)
        nx_ = -(hx1 - hx0) / (2 * e); ny_ = -(hy1 - hy0) / (2 * e)
        nn = math.sqrt(nx_ * nx_ + ny_ * ny_ + 1)
        out_t[i] = t
        out_n[i, 0] = nx_ / nn; out_n[i, 1] = ny_ / nn; out_n[i, 2] = 1 / nn
        out_h[i] = hgt(T, cx0, cy0, cd, F, fpos, fd, px, py)


# ------------------------------------------------------------------ flame
@nb.njit(cache=True, fastmath=True)
def hsh(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 1440662683) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 65535.0


@nb.njit(cache=True, fastmath=True)
def vnoise(x, y, z):
    ix = int(math.floor(x)); iy = int(math.floor(y)); iz = int(math.floor(z))
    fx = x - ix; fy = y - iy; fz = z - iz
    ux = fx * fx * (3 - 2 * fx); uy = fy * fy * (3 - 2 * fy); uz = fz * fz * (3 - 2 * fz)
    a = hsh(ix, iy, iz) + (hsh(ix + 1, iy, iz) - hsh(ix, iy, iz)) * ux
    b = hsh(ix, iy + 1, iz) + (hsh(ix + 1, iy + 1, iz) - hsh(ix, iy + 1, iz)) * ux
    c = hsh(ix, iy, iz + 1) + (hsh(ix + 1, iy, iz + 1) - hsh(ix, iy, iz + 1)) * ux
    d = hsh(ix, iy + 1, iz + 1) + (hsh(ix + 1, iy + 1, iz + 1) - hsh(ix, iy + 1, iz + 1)) * ux
    return (a + (b - a) * uy) * (1 - uz) + (c + (d - c) * uy) * uz


@nb.njit(cache=True, fastmath=True)
def fbm(x, y, z):
    s = 0.0; amp = 0.5; f = 1.0
    for _ in range(5):
        s += amp * vnoise(x * f, y * f, z * f)
        f *= 2.03; amp *= 0.5
    return s


@nb.njit(parallel=True, cache=True, fastmath=True)
def flame_field(hp, wp, tau, seed, grow):
    """Heat of a bonfire on an hp x wp canvas (bottom = base). grow in (0, 1]: flame height."""
    out = np.zeros((hp, wp), np.float32)
    for j in nb.prange(hp):
        v = 1.0 - (j + 0.5) / hp                          # 0 bottom .. 1 top
        vg = v / max(grow, 0.05)
        for i in range(wp):
            u = (i + 0.5) / wp * 2 - 1
            qx = u * 2.4; qy = vg * 3.2 - tau * 3.0                     # rising, turbulent noise
            wx = fbm(qx * 0.9 + seed, qy * 0.6, tau * 0.5)
            wy = fbm(qx * 0.9 + 7.1, qy * 0.6 + 3.3 + seed, tau * 0.5)
            n = fbm(qx + (wx - 0.5) * 2.2, qy + (wy - 0.5) * 1.6, tau * 0.8 + seed)
            width = 0.95 * max(1 - vg * 0.9, 0.0) ** 0.75 + 0.02       # wide base, tapering, swaying
            uu = (u + 0.35 * (wx - 0.5) * vg) / width
            body = 1 - uu * uu
            heat = body * 0.9 + (n - 0.5) * 2.6 - vg * 1.05 + 0.25 * (1 - vg)
            edge = min(1.0, (1 - abs(u)) * 6) * min(1.0, (1 - v) * 8)
            out[j, i] = max(heat, 0.0) * edge
    return out


def flame_rgb(h, inten):
    a = np.clip((h - 0.04) / 0.18, 0, 1)
    r = np.clip(h * 2.2, 0, 1)
    g = np.clip((h - 0.28) * 1.7, 0, 1) * 0.76
    b = np.clip((h - 0.78) * 2.2, 0, 1) * 0.45
    lum = 0.25 + 1.35 * np.clip(h - 0.3, 0, 1.4)
    return np.stack([r, g, b], -1) * (a * lum)[..., None] * inten


@nb.njit(parallel=True, cache=True, fastmath=True)
def bump(px, py, tt, out):
    """Small-scale relief (km) for nearby terrain: gradient of fbm, fading out by 15 km."""
    for i in nb.prange(len(px)):
        fade = max(0.0, 1 - tt[i] / 15.0)
        if fade <= 0:
            out[i, 0] = 0.0; out[i, 1] = 0.0; out[i, 2] = 0.0
            continue
        x = px[i] * 60.0; y = py[i] * 60.0                       # ~17 m features
        e = 0.05
        n0 = fbm(x, y, 1.7)
        out[i, 0] = (fbm(x + e, y, 1.7) - n0) / e * 0.55 * fade
        out[i, 1] = (fbm(x, y + e, 1.7) - n0) / e * 0.55 * fade
        out[i, 2] = n0


# ------------------------------------------------------------------ scene
def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


ALT_FROM = 32.6                                   # Video 2's caption change starts at the rise


class Scene:
    def __init__(self, alt=False):
        self.alt = alt
        d = np.load(os.path.join(DATA, "fpv_terrain.npz"))
        self.T = np.ascontiguousarray(d["coarse"]); self.cx0 = float(d["cx0"]); self.cy0 = float(d["cy0"]); self.cd = float(d["cd"])
        self.F = np.ascontiguousarray(d["fine"]); self.fpos = d["fpos"].astype(np.float64); self.fd = float(d["fd"])
        self.hmax = float(self.T.max()) + 0.05
        self.fires = []
        for k in range(len(self.fpos)):
            # each fine patch is centred on its station (fpv_terrain.py), which keeps the scene fixed
            # even though loop.json has changed since (Mont Blanc, the Apennines and the Dolomites)
            x, y = float(self.fpos[k, 0] + FT.FINE_HALF), float(self.fpos[k, 1] + FT.FINE_HALF)
            # the pyre goes on the highest fine-grid point within 250 m of the station
            f = self.F[k]
            gx, gy = (x - self.fpos[k, 0]) / self.fd, (y - self.fpos[k, 1]) / self.fd
            r = int(0.25 / self.fd)
            sub = f[int(gy) - r:int(gy) + r + 1, int(gx) - r:int(gx) + r + 1]
            jy, jx = np.unravel_index(np.argmax(sub), sub.shape)
            x = self.fpos[k, 0] + (int(gx) - r + jx) * self.fd
            y = self.fpos[k, 1] + (int(gy) - r + jy) * self.fd
            self.fires.append(np.array([x, y, float(sub[jy, jx])]))
        self.dist = [float(np.hypot(*(self.fires[k + 1][:2] - self.fires[k][:2]))) for k in range(2)]
        rng = np.random.default_rng(7)
        v = rng.normal(size=(3000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
        self.stars = v[v[:, 2] > 0.02]
        self.star_mag = rng.random(len(self.stars)) ** 4
        self.sparks = [self.make_sparks(rng) for _ in self.fires]
        self.plan()

    # -------------------------------------------------------------- sightline helpers
    def ray_t(self, cam, target):
        v = target - cam
        D = float(np.linalg.norm(v))
        dirs = (v / D)[None].astype(np.float64)
        ot = np.empty(1); on = np.empty((1, 3)); oh = np.empty(1)
        march(self.T, self.cx0, self.cy0, self.cd, self.F, self.fpos, self.fd, cam.astype(np.float64), dirs,
              self.hmax, D + 1, ot, on, oh)
        return ot[0], D

    def visible(self, cam, target):
        """Is target visible from cam? (the target's own ground drop is applied relative to the camera)"""
        tgt = target.copy()
        dh = float(np.hypot(*(target[:2] - cam[:2])))
        tgt[2] -= dh * dh / (2 * REFF)
        t, D = self.ray_t(cam, tgt)
        return t < 0 or t > D - 0.004

    def eff(self, cam, p):
        """p with Earth's curvature applied as seen from cam (the renderer's flat frame)."""
        q = np.array(p, float)
        dh = float(np.hypot(*(q[:2] - cam[:2])))
        q[2] -= dh * dh / (2 * REFF)
        return q

    def hold_cam(self, k_from, k_at, side):
        """Camera beside fire k_at's pyre, looking back at fire k_from: 30 m behind, `side` m to the side,
        raised until fire k_from is in view."""
        A, B = self.fires[k_from], self.fires[k_at]
        u = (B[:2] - A[:2]) / np.linalg.norm(B[:2] - A[:2])
        perp = np.array([-u[1], u[0]])
        xy = B[:2] + u * 0.030 + perp * side
        g = hgt(self.T, self.cx0, self.cy0, self.cd, self.F, self.fpos, self.fd, xy[0], xy[1])
        z = max(g, B[2]) + 0.004
        tgt = A + np.array([0, 0, PYRE * 0.7])
        for _ in range(60):
            cam = np.array([xy[0], xy[1], z])
            if self.visible(cam, tgt):
                break
            z += 0.003
        return np.array([xy[0], xy[1], z])

    # -------------------------------------------------------------- timeline
    def plan(self):
        A, B, C = self.fires
        uAB = (B[:2] - A[:2]) / np.linalg.norm(B[:2] - A[:2])
        pAB = np.array([-uAB[1], uAB[0]])
        xy = A[:2] + uAB * 0.020 + pAB * 0.004
        g = hgt(self.T, self.cx0, self.cy0, self.cd, self.F, self.fpos, self.fd, xy[0], xy[1])
        self.camA = np.array([xy[0], xy[1], max(g + 0.0017, A[2] - 0.0015)])
        self.camB = self.hold_cam(0, 1, 0.011)
        self.camC = self.hold_cam(1, 2, -0.011)
        self.keepers = [A[:2] + uAB * 0.007 + pAB * 0.0028, None, None]
        for k, cam in ((1, self.camB), (2, self.camC)):
            Fk = self.fires[k]
            u = (Fk[:2] - cam[:2]); u /= np.linalg.norm(u)
            perp = np.array([-u[1], u[0]])
            side = 1 if k == 1 else -1
            self.keepers[k] = Fk[:2] - perp * 0.0035 * side - u * 0.0015
        self.T0 = dict(pull1=5.0, hold1=12.5, light1=16.0, pull2=19.4, hold2=26.4, light2=29.6, rise=32.6, end=41.0)
        self.lit_at = [-10.0, self.T0["light1"], self.T0["light2"]]
        self.total = self.T0["end"]
        self.caps = [
            (0.6, 4.7, [("A beacon: a bonfire on a mountaintop", 46, (240, 244, 250), "SemiBold"),
                        ("This one is on Mont Blanc", 32, (170, 186, 206), "Regular")]),
            (5.6, 11.9, [("The next beacon is 311 km away", 46, (240, 244, 250), "SemiBold"),
                         ("on a summit in Italy's northern Apennines", 32, (170, 186, 206), "Regular")]),
            (13.0, 15.7, [("From 311 km, Mont Blanc's fire is a speck on the horizon", 40, (240, 244, 250), "SemiBold")]),
            (16.3, 19.0, [("See the fire. Light your own.", 48, (255, 186, 80), "Bold")]),
            (20.0, 25.9, [("Now the keeper 244 km away, in the Dolomites, is watching", 40, (240, 244, 250), "SemiBold")]),
            (29.9, 32.3, [("Fire to fire, peak to peak", 48, (255, 186, 80), "Bold")]),
            (33.2, 40.4, [("How far could the message go?", 50, (240, 244, 250), "Bold")]),
        ]
        if self.alt:                                # Video 2: the network nothing can hack
            self.caps[-1] = (33.2, 40.4, [("Nothing to hack. Nothing to switch off.", 50, (255, 186, 80), "Bold"),
                                          ("But how fast could it reach every capital on Earth?", 36, (240, 244, 250), "SemiBold")])
        self.rings = [(13.2, 15.8, 0, "Mont Blanc · 311 km"), (26.9, 29.7, 1, "Apennines · 244 km")]

    def fire_state(self, k, t):
        """(intensity 0..1, flame growth 0..1)"""
        a = t - self.lit_at[k]
        if a < 0:
            return 0.0, 0.0
        return min(1.0, a / 0.5), min(1.0, 0.15 + a / 1.4)

    def pull(self, t, t0, t1, P0, P1, focus, bump):
        s = (t - t0) / (t1 - t0)
        e = smoother(s)
        d0 = np.linalg.norm(P0[:2] - focus[:2]); d1 = np.linalg.norm(P1[:2] - focus[:2])
        dd = d0 * (d1 / d0) ** e                                         # powers-of-ten pull back
        f = (dd - d0) / (d1 - d0)
        xy = P0[:2] + (P1[:2] - P0[:2]) * f
        z = P0[2] + (P1[2] - P0[2]) * f + bump * math.sin(math.pi * f) ** 1.2
        cam = np.array([xy[0], xy[1], z])
        g = hgt(self.T, self.cx0, self.cy0, self.cd, self.F, self.fpos, self.fd, xy[0], xy[1])
        cam[2] = max(cam[2], g + 0.0017 + 0.3 * min(f, 1 - f))
        return cam

    @staticmethod
    def blend_look(cam, p, q, w):
        """A look target whose direction blends from p to q (as seen from cam)."""
        a = (p - cam) / np.linalg.norm(p - cam); b = (q - cam) / np.linalg.norm(q - cam)
        d = a * (1 - w) + b * w
        return cam + d / np.linalg.norm(d)

    def hold(self, t, cam, far, near, t_hold, t_light):
        z = smooth((t - t_hold) / 2.6)
        fov = 40.0 - 18.0 * z
        look = far
        if t > t_light - 0.2:
            w = smooth((t - t_light + 0.2) / 2.6)
            fov += (50.0 - fov) * w
            look = self.blend_look(cam, far, near + np.r_[0, 0, PYRE * 0.3], 0.62 * w)
        return cam, look, fov

    def camera(self, t):
        A, B, C = self.fires
        T0 = self.T0
        fa = A + np.r_[0, 0, PYRE * 0.5]; fb = B + np.r_[0, 0, PYRE * 0.5]; fc = C + np.r_[0, 0, PYRE * 0.5]
        if t < T0["pull1"]:
            s = smooth(t / T0["pull1"])
            cam = A + (self.camA - A) * (1 + 0.3 * s)
            cam[2] = self.camA[2] + 0.002 * s
            return cam, fa, 54.0
        if t < T0["hold1"]:
            cam = self.pull(t, T0["pull1"], T0["hold1"], self.camA + np.r_[(self.camA - A)[:2] * 0.3, 0.002], self.camB, A, 3.0)
            return cam, fa, 54.0 - 14.0 * smooth((t - T0["pull1"]) / (T0["hold1"] - T0["pull1"]))
        if t < T0["pull2"]:
            return self.hold(t, self.camB, fa, fb, T0["hold1"], T0["light1"])
        if t < T0["hold2"]:
            cam = self.pull(t, T0["pull2"], T0["hold2"], self.camB, self.camC, B, 2.4)
            _, look0, _ = self.hold(T0["pull2"], self.camB, fa, fb, T0["hold1"], T0["light1"])
            w = smooth((t - T0["pull2"]) / 1.6)
            look = self.blend_look(cam, look0, fb, w)
            return cam, look, 50.0 - 10.0 * smooth((t - T0["pull2"]) / (T0["hold2"] - T0["pull2"]))
        if t < T0["rise"]:
            return self.hold(t, self.camC, fb, fc, T0["hold2"], T0["light2"])
        _, look0, fov0 = self.hold(T0["rise"], self.camC, fb, fc, T0["hold2"], T0["light2"])
        s = smooth((t - T0["rise"]) / (T0["end"] - T0["rise"]))
        u = (C[:2] - B[:2]) / np.linalg.norm(C[:2] - B[:2])
        cam = self.camC + np.r_[u * 45 * s, 24 * s ** 1.4]
        tgt = (fb + fc) / 2 + np.r_[-u * 40, -6.0]
        look = self.blend_look(cam, look0, tgt, s)
        return cam, look, fov0 + (56.0 - fov0) * s

    # -------------------------------------------------------------- sparks
    def make_sparks(self, rng):
        n = 1100
        return dict(t0=np.sort(rng.random(n) * 45.0), p=rng.normal(0, 1, (n, 3)) * [0.0014, 0.0014, 0.0010] + [0, 0, 0.004],
                    v=np.c_[rng.normal(0, 0.0010, (n, 2)), 0.0025 + 0.0035 * rng.random(n)],
                    life=0.8 + 1.6 * rng.random(n) ** 2, ph=rng.random(n) * 6.28, b=0.3 + rng.random(n))

    # -------------------------------------------------------------- render
    def basis(self, cam, look):
        f = look - cam; f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u

    def sky(self, d):
        el = np.arcsin(np.clip(d[..., 2], -1, 1))
        az = np.arctan2(d[..., 0], d[..., 1])
        sun = np.array([math.sin(SUN_AZ) * math.cos(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL)])
        ang = np.arccos(np.clip(d @ sun, -1, 1))
        e = np.clip(el, 0, None)
        hz = np.clip(1 - e / 0.5, 0, 1) ** 3
        zen = np.array([0.0035, 0.006, 0.020]); hor = np.array([0.020, 0.030, 0.068])
        col = zen + (hor - zen) * hz[..., None]
        glow = np.exp(-ang / 0.45) * np.exp(-e / 0.07)
        col = col + np.array([0.30, 0.11, 0.035]) * (glow * 0.6)[..., None] + np.array([0.10, 0.06, 0.06]) * (np.exp(-ang / 1.2) * np.exp(-e / 0.2) * 0.25)[..., None]
        return col

    def frame(self, t):
        cam, look, fov = self.camera(t)
        f, r, u = self.basis(cam, look)
        rw, rh = W // RS, H // RS
        tn = math.tan(math.radians(fov) / 2)
        ys, xs = np.mgrid[0:rh, 0:rw]
        sx = ((xs + 0.5) / rw * 2 - 1) * tn * rw / rh
        sy = (1 - (ys + 0.5) / rh * 2) * tn
        d = f[None, None] + sx[..., None] * r[None, None] + sy[..., None] * u[None, None]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        dirs = np.ascontiguousarray(d.reshape(-1, 3))
        n = len(dirs)
        ot = np.empty(n); on = np.empty((n, 3)); oh = np.empty(n)
        march(self.T, self.cx0, self.cy0, self.cd, self.F, self.fpos, self.fd, cam.astype(np.float64), dirs,
              self.hmax, 650.0, ot, on, oh)
        hit = ot > 0
        col = self.sky(dirs)
        if hit.any():
            dh, nh, hh, tt = dirs[hit], on[hit], oh[hit], ot[hit]
            p = cam[None] + dh * tt[:, None]
            bp = np.empty((len(tt), 3))
            bump(p[:, 0], p[:, 1], tt, bp)
            nh = nh + np.c_[-bp[:, 0], -bp[:, 1], np.zeros(len(tt))]
            nh /= np.linalg.norm(nh, axis=1, keepdims=True)
            slope = nh[:, 2]
            var = 0.75 + 0.5 * bp[:, 2]
            ss = lambda a, b, x: np.clip((x - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / (b - a), 0, 1))
            wob = 0.15 * np.sin(p[:, 0] * 3.1) * np.cos(p[:, 1] * 2.7) + 0.08 * (bp[:, 2] - 0.5)
            snow_w = ss(2.6, 2.95, hh + wob) * ss(0.45, 0.65, slope)              # snow, not on cliffs
            rock_w = ss(1.4, 2.3, hh + wob)                                       # forest/meadow -> bare rock
            cliff_w = 1 - ss(0.45, 0.65, slope)
            alb = (0.09 + 0.11 * rock_w) * (1 - cliff_w) + 0.16 * cliff_w
            alb = (alb * (1 - snow_w) + 0.62 * snow_w) * var
            water = (hh < 0.003) & (slope > 0.995)
            moon = np.clip(nh @ MOON, 0, None) * 0.075
            amb = (0.55 + 0.45 * slope) * 0.030
            light = (moon[:, None] * np.array([0.75, 0.85, 1.0]) + amb[:, None] * np.array([0.55, 0.65, 1.0]))
            # the western sky's warm bounce on slopes facing it
            sun_h = np.array([math.sin(SUN_AZ), math.cos(SUN_AZ), 0.25]); sun_h /= np.linalg.norm(sun_h)
            light += (np.clip(nh @ sun_h, 0, None) * 0.020)[:, None] * np.array([0.9, 0.55, 0.4])
            # firelight
            for k, F_ in enumerate(self.fires):
                I, _ = self.fire_state(k, t)
                if I <= 0:
                    continue
                fp = self.eff(cam, F_ + np.r_[0, 0, 0.004])
                v = fp[None] - p
                dd = np.linalg.norm(v, axis=1)
                near = dd < 1.5
                if near.any():
                    fl = 1 + 0.12 * math.sin(t * 13 + k) + 0.08 * math.sin(t * 29 + 2 * k)
                    lam = np.clip(np.einsum("ij,ij->i", nh[near], v[near] / dd[near, None]), 0, None)
                    light[near] += (I * fl * 0.9 * lam / (1 + (dd[near] / 0.012) ** 2))[:, None] * np.array([1.0, 0.45, 0.12])
            c = alb[:, None] * light
            refl = dh * np.array([1, 1, -1.0])
            c = np.where(water[:, None], self.sky(refl) * 0.45, c)
            T_ = np.exp(-tt / 140.0)[:, None]
            haze = self.sky(dh * np.array([1, 1, 0]) / np.linalg.norm(dh[:, :2], axis=1, keepdims=True).clip(1e-6) + np.array([0, 0, 0.02]))
            col[hit] = c * T_ + haze * 0.75 * (1 - T_)
        img = col.reshape(rh, rw, 3).astype(np.float32)
        depth = np.where(hit, ot, 1e9).reshape(rh, rw).astype(np.float32)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        fpx = (H / 2) / tn
        E = np.zeros_like(img)

        def proj(p):
            q = self.eff(cam, p) - cam
            z = q @ f
            if z <= 1e-5:
                return None
            return W / 2 + (q @ r) / z * fpx, H / 2 - (q @ u) / z * fpx, z

        # stars on clear sky
        sp = self.stars @ np.c_[r, u, f]
        ok = sp[:, 2] > 0.1
        sx_ = W / 2 + sp[ok, 0] / sp[ok, 2] * fpx; sy_ = H / 2 - sp[ok, 1] / sp[ok, 2] * fpx
        inb = (sx_ >= 1) & (sx_ < W - 1) & (sy_ >= 1) & (sy_ < H - 1)
        for x_, y_, m in zip(sx_[inb].astype(int), sy_[inb].astype(int), self.star_mag[ok][inb]):
            if depth[y_, x_] > 1e8:
                img[y_, x_] += 0.05 + 0.5 * m
        # pyres, keepers, flames, sparks (far to near)
        order = sorted(range(len(self.fires)), key=lambda k: -np.linalg.norm(self.fires[k] - cam))
        spots = {}
        for k in order:
            sp_ = self.draw_beacon(img, E, depth, k, t, cam, f, proj, fpx)
            if sp_ is not None:
                spots[k] = sp_
        small = cv2.resize(E, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5) * 0.30 + cv2.GaussianBlur(small, (0, 0), 10) * 0.22, (W, H))
        out = 1 - np.exp(-(img + E + bloom) * 3.0)
        out = np.clip(out, 0, 1) ** (1 / 1.9)
        fr = (out * 255).astype(np.uint8)
        for t0, t1, k, txt in self.rings:
            if k in spots:
                fr = ST.ring_label(fr, spots[k][0], spots[k][1], txt, ST.fade(t, t0, t1), side=1 if spots[k][0] < W * 0.7 else -1)
        return ST.draw_captions(fr, t, self.caps)

    def draw_beacon(self, img, E, depth, k, t, cam, f, proj, fpx):
        F_ = self.fires[k]
        I, grow = self.fire_state(k, t)
        base = proj(F_)
        top = proj(F_ + np.r_[0, 0, PYRE])
        if base is None or top is None:
            return None
        bx, by, bz = base
        tx, ty, _ = top
        hpx = by - ty                                          # projected flame height (px)
        fl = 1 + 0.15 * math.sin(t * 11 + k * 2) + 0.1 * math.sin(t * 23 + k)
        if hpx < 10:
            # a distant beacon: a point of light, if no terrain is in the way
            cx_, cy_ = (bx + tx) / 2, (by + ty) / 2
            if I <= 0 or not (0 <= cx_ < W and 0 <= cy_ < H) or not self.visible(cam, F_ + np.r_[0, 0, PYRE * 0.7]):
                return None
            amp = I * fl * float(np.clip(12.0 / max(bz, 0.05) ** 0.5, 0.9, 3.0))
            self.glow(E, cx_, cy_, 1.1, 2.4 * amp, (1.0, 0.82, 0.55))
            self.glow(E, cx_, cy_, 2.6, 0.45 * amp)
            self.glow(E, cx_, cy_, 8.0, 0.05 * amp)
            return (cx_, cy_)
        sc = hpx / PYRE                                          # px per km at the beacon
        ok = lambda z_: z_ - 0.003
        # flames (behind the logs)
        if I > 0:
            fw = max(8, int(0.0085 * sc)); fh = max(10, int(0.0135 * sc))
            q = max(1, -(-max(fw, fh) // 560))                  # cap the noise canvas resolution
            heat = flame_field(max(fh // q, 4), max(fw // q, 4), t * 1.0 + k * 3.7, float(k) * 11.3, grow)
            if q > 1:
                heat = cv2.resize(heat, (fw, fh), interpolation=cv2.INTER_CUBIC)
            rgb = flame_rgb(heat, I * fl * 0.62).astype(np.float32)
            x0 = int(bx - fw / 2); y0 = int(by - fh + 0.0003 * sc)
            X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + fw, W), min(y0 + fh, H)
            if X1 > X0 and Y1 > Y0:
                sub = rgb[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
                vis = depth[Y0:Y1, X0:X1] > ok(bz)
                E[Y0:Y1, X0:X1] += sub * vis[..., None]
            self.glow(E, bx, by - 0.0012 * sc, min(0.0022 * sc, 260), I * fl * 0.10)
        # the pyre: a teepee of logs, 3.6 m wide, 3.3 m tall
        rng = np.random.default_rng(100 + k)
        mask = np.zeros((H, W), np.uint8)
        th = max(1, int(0.00022 * sc))
        for j in range(15):
            x_b = -0.0018 + 0.0036 * j / 14 + rng.normal(0, 0.00012)
            x_t = rng.normal(0, 0.00025)
            y_t = 0.0033 + rng.normal(0, 0.0002)
            cv2.line(mask, (int(bx + x_b * sc), int(by + 0.0002 * sc)), (int(bx + x_t * sc), int(by - y_t * sc)), 1, th, cv2.LINE_AA)
        for j in range(3):                                      # a few logs leaning across
            yy_ = 0.0004 + j * 0.0007 + rng.normal(0, 0.0001)
            half = 0.0017 * (1 - yy_ / 0.0035)
            tilt = rng.normal(0, 0.0005)
            cv2.line(mask, (int(bx - half * sc), int(by - (yy_ - tilt) * sc)), (int(bx + half * 0.8 * sc), int(by - (yy_ + tilt) * sc)), 1, th, cv2.LINE_AA)
        m = (mask > 0) & (depth > ok(bz))
        wood = np.array([0.010, 0.0075, 0.006], np.float32) + I * fl * np.array([0.06, 0.016, 0.003], np.float32)
        img[m] = img[m] * 0.10 + wood
        E[m] *= 0.12
        # keeper (1.8 m), arm raised with a torch
        kp = self.keepers[k]
        kz = hgt(self.T, self.cx0, self.cy0, self.cd, self.F, self.fpos, self.fd, kp[0], kp[1])
        kb = proj(np.array([kp[0], kp[1], kz]))
        if kb is not None:
            kx, ky, kzz = kb
            s = fpx / kzz * 0.001                                # px per metre
            body = [(-0.20, 0), (-0.08, 0), (0, 0.80), (0.08, 0), (0.20, 0), (0.17, 0.85), (0.22, 1.45),
                    (0.10, 1.52), (-0.10, 1.52), (-0.22, 1.45), (-0.17, 0.85)]
            poly = np.array([(kx + x * s, ky - y * s) for x, y in body], np.int32)
            km = np.zeros((H, W), np.uint8)
            cv2.fillPoly(km, [poly], 1, cv2.LINE_AA)
            cv2.circle(km, (int(kx), int(ky - 1.66 * s)), max(1, int(0.12 * s)), 1, -1, cv2.LINE_AA)
            hand = (kx + 0.45 * s, ky - 2.05 * s)
            cv2.line(km, (int(kx + 0.15 * s), int(ky - 1.45 * s)), (int(hand[0]), int(hand[1])), 1, max(1, int(0.09 * s)), cv2.LINE_AA)
            m = (km > 0) & (depth > kzz - 0.003)
            img[m] = np.array([0.003, 0.003, 0.005], np.float32) + img[m] * 0.04
            E[m] *= 0.05
            if k > 0 and t < self.lit_at[k] + 2.5:
                self.glow(E, hand[0], hand[1] - 0.1 * s, float(np.clip(0.10 * s, 1.2, 7)), 1.5 * fl)
        if I <= 0:
            return None
        # sparks
        S = self.sparks[k]
        age = t - self.lit_at[k] - S["t0"]
        alive = (age > 0) & (age < S["life"])
        if alive.any():
            a_ = age[alive][:, None]
            p = S["p"][alive] + S["v"][alive] * a_ + np.c_[0.0009 * np.sin(a_[:, 0] * 2.1 + S["ph"][alive]),
                                                            0.0009 * np.cos(a_[:, 0] * 1.7 + S["ph"][alive]), np.zeros(len(a_))]
            br = S["b"][alive] * (1 - a_[:, 0] / S["life"][alive]) * I
            for (px, py, pz), b in zip(p, br):
                pr = proj(F_ + np.array([px, py, pz]))
                if pr is None:
                    continue
                x_, y_, z_ = pr
                if 0 <= x_ < W - 1 and 0 <= y_ < H - 1 and depth[int(y_), int(x_)] > z_ - 0.003:
                    self.glow(E, x_, y_, float(np.clip(0.00009 * fpx / z_, 0.6, 1.8)), b * 1.1, (1.0, 0.62, 0.25))
        return None

    def glow(self, E, x, y, sig, amp, col=(1.0, 0.55, 0.18)):
        rr = int(sig * 4) + 2
        cx_, cy_ = int(x), int(y)
        y0, y1, x0, x1 = max(cy_ - rr, 0), min(cy_ + rr + 1, H), max(cx_ - rr, 0), min(cx_ + rr + 1, W)
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig * sig)
        g = np.clip(np.exp(-d2) - math.exp(-8.0), 0, None) * amp
        E[y0:y1, x0:x1] += g[..., None] * np.array(col, np.float32)


def render(still=None, alt=False):
    os.makedirs(OUT, exist_ok=True)
    S = Scene(alt)
    print(f"fires at {[np.round(f_, 3).tolist() for f_ in S.fires]}, hops {np.round(S.dist, 1)} km", flush=True)
    print(f"camB {np.round(S.camB, 4)}, camC {np.round(S.camC, 4)}", flush=True)
    if still is not None:
        for s in (still if isinstance(still, list) else [still]):
            t0 = time.time()
            cv2.imwrite(os.path.join(OUT, f"fpv_still_{s:05.1f}.png"), S.frame(s)[..., ::-1])
            print(f"still {s}: {time.time() - t0:.1f}s", flush=True)
        return
    vid = os.path.join(OUT, "fpv_alt_tail.mp4" if alt else "fpv.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", vid],
                         stdin=subprocess.PIPE)
    n = int(round(S.total * FPS))
    i0 = int(round(ALT_FROM * FPS)) if alt else 0
    t0 = time.time()
    for i in range(i0, n):
        p.stdin.write(S.frame(i / FPS).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{n} {time.time() - t0:.0f}s", flush=True)
    p.stdin.close(); p.wait()
    print("wrote", vid)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "still":
        render([float(a) for a in sys.argv[2:]])
    elif len(sys.argv) > 1 and sys.argv[1] == "alt":
        render(alt=True)
    else:
        render()
