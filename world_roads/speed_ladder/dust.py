"""Dust: a small smoke solver and a volume renderer of my own (Blender's Mantaflow won't run in the Python build of
Blender available here), both on the plate's camera and sun.

Solver: stable fluids on a collocated grid (semi-Lagrangian advection, Jacobi pressure projection, vorticity
confinement), with buoyancy from heat, dust that is heavier than air and settles, and a ground that lifts sand where
the wind over it is strong (saltation). A hypersonic pass starts it with an impulse: air pushed out from the path,
hot gas along it, dust scoured off the ground under it.
Renderer: ray marching through the grid from the plate's camera; single scattering of the sun (Henyey-Greenstein,
shadowed by the dust between it and the sun) plus sky light (occluded by the dust above), and the dust's shadow on
the ground. Writes premultiplied RGB + transmittance + a ground-shadow factor per frame, for comp.py."""
import numpy as np, math, os, sys, time
from numba import njit, prange
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as C

# ---------------------------------------------------------------- grid
class Grid:
    def __init__(self, x0=-16.0, x1=16.0, y0=5.0, y1=20.0, z1=9.0, h=0.125):
        self.h = h
        self.o = np.array([x0, y0, 0.0], np.float32)
        self.n = (int(round((x1 - x0) / h)), int(round((y1 - y0) / h)), int(round(z1 / h)))
        nx, ny, nz = self.n
        z = lambda: np.zeros((nx, ny, nz), np.float32)
        self.u, self.v, self.w = z(), z(), z()
        self.d, self.T, self.p = z(), z(), z()
        self.avail = np.ones((nx, ny), np.float32)

    def coords(self):
        nx, ny, nz = self.n; h = self.h
        x = self.o[0] + (np.arange(nx) + 0.5) * h
        y = self.o[1] + (np.arange(ny) + 0.5) * h
        z = (np.arange(nz) + 0.5) * h
        return np.meshgrid(x, y, z, indexing="ij")


@njit(inline="always")
def trilin(q, x, y, z):
    nx, ny, nz = q.shape
    x = min(max(x, 0.0), nx - 1.001); y = min(max(y, 0.0), ny - 1.001); z = min(max(z, 0.0), nz - 1.001)
    i = int(x); j = int(y); k = int(z)
    fx = x - i; fy = y - j; fz = z - k
    c00 = q[i, j, k] * (1 - fx) + q[i + 1, j, k] * fx
    c10 = q[i, j + 1, k] * (1 - fx) + q[i + 1, j + 1, k] * fx
    c01 = q[i, j, k + 1] * (1 - fx) + q[i + 1, j, k + 1] * fx
    c11 = q[i, j + 1, k + 1] * (1 - fx) + q[i + 1, j + 1, k + 1] * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz


@njit(parallel=True, cache=True)
def advect(q, u, v, w, dt_h, out, wsettle, outside):
    """semi-Lagrangian; cells whose back-trace leaves the grid sideways or above take 'outside' (0 for dust)"""
    nx, ny, nz = q.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                x = i - dt_h * u[i, j, k]
                y = j - dt_h * v[i, j, k]
                z = k - dt_h * (w[i, j, k] - wsettle)
                if x < -0.5 or x > nx - 0.5 or y < -0.5 or y > ny - 0.5 or z > nz - 0.5:
                    out[i, j, k] = outside
                else:
                    out[i, j, k] = trilin(q, x, y, z)


@njit(parallel=True, cache=True)
def divergence(u, v, w, h, div):
    nx, ny, nz = u.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                ue = u[i + 1, j, k] if i < nx - 1 else u[i, j, k]
                uw = u[i - 1, j, k] if i > 0 else u[i, j, k]
                vn = v[i, j + 1, k] if j < ny - 1 else v[i, j, k]
                vs = v[i, j - 1, k] if j > 0 else v[i, j, k]
                wt = w[i, j, k + 1] if k < nz - 1 else w[i, j, k]
                wb = w[i, j, k - 1] if k > 0 else -w[i, j, k]          # the ground: no flow through it
                div[i, j, k] = (ue - uw + vn - vs + wt - wb) / (2 * h)


@njit(parallel=True, cache=True)
def jacobi(p, div, h, out):
    nx, ny, nz = p.shape
    h2 = h * h
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                s = 0.0
                s += p[i + 1, j, k] if i < nx - 1 else 0.0          # open sides and top: p = 0 beyond
                s += p[i - 1, j, k] if i > 0 else 0.0
                s += p[i, j + 1, k] if j < ny - 1 else 0.0
                s += p[i, j - 1, k] if j > 0 else 0.0
                s += p[i, j, k + 1] if k < nz - 1 else 0.0
                s += p[i, j, k - 1] if k > 0 else p[i, j, k]         # ground: dp/dz = 0
                out[i, j, k] = (s - h2 * div[i, j, k]) / 6.0


@njit(parallel=True, cache=True)
def subtract_grad(p, u, v, w, h):
    nx, ny, nz = p.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                pe = p[i + 1, j, k] if i < nx - 1 else 0.0
                pw = p[i - 1, j, k] if i > 0 else 0.0
                pn = p[i, j + 1, k] if j < ny - 1 else 0.0
                ps = p[i, j - 1, k] if j > 0 else 0.0
                pt = p[i, j, k + 1] if k < nz - 1 else 0.0
                pb = p[i, j, k - 1] if k > 0 else p[i, j, k]
                u[i, j, k] -= (pe - pw) / (2 * h)
                v[i, j, k] -= (pn - ps) / (2 * h)
                w[i, j, k] -= (pt - pb) / (2 * h)
            w[i, j, 0] = max(w[i, j, 0], 0.0) * 0.5


@njit(parallel=True, cache=True)
def vorticity(u, v, w, h, eps, dt, fx, fy, fz, om):
    nx, ny, nz = u.shape
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                ox = (w[i, j + 1, k] - w[i, j - 1, k] - v[i, j, k + 1] + v[i, j, k - 1]) / (2 * h)
                oy = (u[i, j, k + 1] - u[i, j, k - 1] - w[i + 1, j, k] + w[i - 1, j, k]) / (2 * h)
                oz = (v[i + 1, j, k] - v[i - 1, j, k] - u[i, j + 1, k] + u[i, j - 1, k]) / (2 * h)
                fx[i, j, k] = ox; fy[i, j, k] = oy; fz[i, j, k] = oz
                om[i, j, k] = math.sqrt(ox * ox + oy * oy + oz * oz)
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            for k in range(2, nz - 2):
                gx = (om[i + 1, j, k] - om[i - 1, j, k]) / (2 * h)
                gy = (om[i, j + 1, k] - om[i, j - 1, k]) / (2 * h)
                gz = (om[i, j, k + 1] - om[i, j, k - 1]) / (2 * h)
                g = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-6
                Nx = gx / g; Ny = gy / g; Nz = gz / g
                ox = fx[i, j, k]; oy = fy[i, j, k]; oz = fz[i, j, k]
                u[i, j, k] += dt * eps * h * (Ny * oz - Nz * oy)
                v[i, j, k] += dt * eps * h * (Nz * ox - Nx * oz)
                w[i, j, k] += dt * eps * h * (Nx * oy - Ny * ox)


@njit(parallel=True, cache=True)
def forces(u, v, w, d, T, dt, alpha, beta, drag, cool, d_amb):
    nx, ny, nz = u.shape
    a = math.exp(-drag * dt); c = math.exp(-cool * dt)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                w[i, j, k] += dt * (beta * T[i, j, k] - alpha * d[i, j, k])
                u[i, j, k] *= a; v[i, j, k] *= a; w[i, j, k] *= a
                T[i, j, k] *= c


@njit(parallel=True, cache=True)
def saltation(u, v, d, avail, dt, thr, rate, cap):
    """sand lifted where the wind over the ground is strong"""
    nx, ny, nz = u.shape
    for i in prange(nx):
        for j in range(ny):
            s = math.sqrt(u[i, j, 0] ** 2 + v[i, j, 0] ** 2)
            if s > thr:
                add = dt * rate * avail[i, j] * (s - thr)
                d[i, j, 0] = min(cap, d[i, j, 0] + add)
                d[i, j, 1] = min(cap, d[i, j, 1] + 0.5 * add)


class Sim:
    def __init__(self, g, alpha=0.6, beta=1.2, eps=3.0, drag=0.35, cool=0.6, settle=0.12, iters=60):
        self.g = g; self.alpha, self.beta, self.eps = alpha, beta, eps
        self.drag, self.cool, self.settle, self.iters = drag, cool, settle, iters
        self.salt = (6.0, 0.35, 3.0)
        s = g.u.shape
        self.tmp = np.zeros(s, np.float32); self.tmp2 = np.zeros(s, np.float32)
        self.f = [np.zeros(s, np.float32) for _ in range(4)]

    def step(self, dt):
        g = self.g; h = g.h
        forces(g.u, g.v, g.w, g.d, g.T, dt, self.alpha, self.beta, self.drag, self.cool, 0.0)
        vorticity(g.u, g.v, g.w, h, self.eps, dt, *self.f)
        saltation(g.u, g.v, g.d, g.avail, dt, *self.salt)
        self.project()
        nu = np.empty_like(g.u); nv = np.empty_like(g.v); nw = np.empty_like(g.w)
        advect(g.u, g.u, g.v, g.w, dt / h, nu, 0.0, 0.0)
        advect(g.v, g.u, g.v, g.w, dt / h, nv, 0.0, 0.0)
        advect(g.w, g.u, g.v, g.w, dt / h, nw, 0.0, 0.0)
        g.u, g.v, g.w = nu, nv, nw
        self.project()
        advect(g.d, g.u, g.v, g.w, dt / h, self.tmp, self.settle, 0.0); g.d, self.tmp = self.tmp, g.d
        advect(g.T, g.u, g.v, g.w, dt / h, self.tmp, 0.0, 0.0); g.T, self.tmp = self.tmp, g.T
        np.maximum(g.d, 0, out=g.d)

    def project(self):
        g = self.g
        if getattr(self, "src", None) is not None:
            divergence_minus(g.u, g.v, g.w, g.h, self.tmp2, self.src)       # expanding gas: div u = src
        else:
            divergence(g.u, g.v, g.w, g.h, self.tmp2)
        p, q = g.p, self.tmp
        for _ in range(self.iters):
            jacobi(p, self.tmp2, g.h, q); p, q = q, p
        g.p = p
        subtract_grad(p, g.u, g.v, g.w, g.h)


def noise3(shape, scale_cells, seed, octaves=3):
    """smooth value noise in [0, 1]"""
    from scipy.ndimage import zoom
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        sc = max(2.0, scale_cells / 2 ** o)
        cs = tuple(int(math.ceil(n / sc)) + 3 for n in shape)
        r = rng.random(cs).astype(np.float32)
        z = zoom(r, sc, order=3)[:shape[0], :shape[1], :shape[2]]
        out += amp * z; tot += amp; amp *= 0.5
    out /= tot
    return np.clip((out - out.mean()) / (out.std() * 3) + 0.5, 0, 1)


# ---------------------------------------------------------------- rendering
@njit(parallel=True, cache=True)
def sun_transmittance(d, h, sdir, kappa, out):
    """optical depth from every cell toward the sun -> transmittance"""
    nx, ny, nz = d.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                x = i + 0.5; y = j + 0.5; z = k + 0.5
                tau = 0.0
                for s in range(1000):
                    x += sdir[0]; y += sdir[1]; z += sdir[2]
                    if x < 0 or x >= nx or y < 0 or y >= ny or z >= nz or z < 0:
                        break
                    tau += trilin(d, x - 0.5, y - 0.5, z - 0.5)
                out[i, j, k] = math.exp(-kappa * tau * h)


@njit(parallel=True, cache=True)
def sky_occlusion(d, h, kappa, out):
    nx, ny, nz = d.shape
    for i in prange(nx):
        for j in range(ny):
            tau = 0.0
            for k in range(nz - 1, -1, -1):
                out[i, j, k] = math.exp(-kappa * tau * h)
                tau += d[i, j, k]


@njit(parallel=True, cache=True)
def march(dirs, org, d, ts, sky, gmin, h, kappa, albedo, sun_rad, sky_rad, phase_g, sundir, step, seed_img, rgb, trans, gshadow):
    H, W, _ = dirs.shape
    nx, ny, nz = d.shape
    gmax0 = gmin[0] + nx * h; gmax1 = gmin[1] + ny * h; gmax2 = nz * h
    g2 = phase_g * phase_g
    for py in prange(H):
        for px in range(W):
            dx = dirs[py, px, 0]; dy = dirs[py, px, 1]; dz = dirs[py, px, 2]
            # the box [gmin, gmax] x [0, gmax2]
            t0 = 0.0; t1 = 1e9
            for a in range(3):
                o = org[a]; dd = dirs[py, px, a]
                lo = gmin[a] if a < 2 else 0.0
                hi = gmax0 if a == 0 else (gmax1 if a == 1 else gmax2)
                if abs(dd) < 1e-9:
                    if o < lo or o > hi: t1 = -1.0
                else:
                    ta = (lo - o) / dd; tb = (hi - o) / dd
                    if ta > tb: ta, tb = tb, ta
                    t0 = max(t0, ta); t1 = min(t1, tb)
            hit_ground = False
            if dz < 0:
                tg = -org[2] / dz
                if tg < t1:
                    t1 = tg; hit_ground = True
            T = 1.0; r = 0.0; gg = 0.0; b = 0.0
            if t1 > t0:
                cosang = dx * sundir[0] + dy * sundir[1] + dz * sundir[2]
                ph = (1 - g2) / (4 * math.pi * (1 + g2 - 2 * phase_g * cosang) ** 1.5)
                t = t0 + step * seed_img[py, px]
                while t < t1 and T > 0.003:
                    x = (org[0] + t * dx - gmin[0]) / h - 0.5
                    y = (org[1] + t * dy - gmin[1]) / h - 0.5
                    z = (org[2] + t * dz) / h - 0.5
                    rho = trilin(d, x, y, z)
                    if rho > 1e-4:
                        st = kappa * rho
                        a_ = 1.0 - math.exp(-st * step)
                        tsun = trilin(ts, x, y, z); occ = trilin(sky, x, y, z)
                        for c in range(3):
                            S = albedo[c] * (sun_rad[c] * tsun * ph + sky_rad[c] * (0.25 + 0.75 * occ))
                            if c == 0: r += T * a_ * S
                            elif c == 1: gg += T * a_ * S
                            else: b += T * a_ * S
                        T *= 1.0 - a_
                    t += step
            rgb[py, px, 0] = r; rgb[py, px, 1] = gg; rgb[py, px, 2] = b
            trans[py, px] = T
            gshadow[py, px] = 1.0
            if hit_ground:
                x = (org[0] + t1 * dx - gmin[0]) / h - 0.5
                y = (org[1] + t1 * dy - gmin[1]) / h - 0.5
                if x > 0 and y > 0 and x < nx - 1 and y < ny - 1:
                    gshadow[py, px] = trilin(ts, x, y, 0.0)


class Renderer:
    def __init__(self, g, scale=1.0, kappa=2.2, albedo=(0.62, 0.45, 0.30), phase_g=0.45,
                 sun=(50.0, 20.0), sun_irr=2.6, sun_col=(1.0, 0.93, 0.84), sky=(0.36, 0.40, 0.46)):
        self.g = g
        self.W, self.H = int(C.W * scale), int(C.H_IMG * scale)
        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        xs = (xs + 0.5) / scale - 0.5; ys = (ys + 0.5) / scale - 0.5
        cam = np.stack([(xs - C.CX) / C.F, (ys - C.CY) / C.F, np.ones_like(xs)], -1)
        d = cam @ C.R()                                          # camera -> world (R is orthonormal)
        self.dirs = (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)
        self.org = np.array([C.CAM_X, 0.0, C.H], np.float32)
        az, el = math.radians(sun[0]), math.radians(sun[1])
        self.sundir = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)], np.float32)
        self.kappa = kappa
        self.albedo = np.array(albedo, np.float32)
        self.sun_rad = (sun_irr * np.array(sun_col)).astype(np.float32)
        self.sky_rad = np.array(sky, np.float32)
        self.phase_g = phase_g
        self.seed = np.random.default_rng(1).random((self.H, self.W)).astype(np.float32)

    def render(self, step=None):
        g = self.g
        ts = np.empty_like(g.d); sky = np.empty_like(g.d)
        sd = (self.sundir / np.abs(self.sundir).max()).astype(np.float32)       # one cell per step along the main axis
        sun_transmittance(g.d, g.h * float(np.linalg.norm(sd)), sd, self.kappa, ts)
        sky_occlusion(g.d, g.h, self.kappa * 0.6, sky)
        rgb = np.zeros((self.H, self.W, 3), np.float32); tr = np.ones((self.H, self.W), np.float32)
        gs = np.ones((self.H, self.W), np.float32)
        march(self.dirs, self.org, g.d, ts, sky, g.o[:2].copy(), g.h, self.kappa, self.albedo, self.sun_rad, self.sky_rad,
              self.phase_g, self.sundir, step or g.h * 0.6, self.seed, rgb, tr, gs)
        return rgb, tr, gs


# ---------------------------------------------------------------- dust parcels
@njit(parallel=True, cache=True)
def move_parcels(P, V, tau, vs, alive, u, v, w, o, h, dt, rest):
    """drag toward the air (relaxation time tau), settling at vs (gravity balanced by drag), the ground"""
    n = P.shape[0]
    nx, ny, nz = u.shape
    for i in prange(n):
        if not alive[i]: continue
        x = (P[i, 0] - o[0]) / h - 0.5; y = (P[i, 1] - o[1]) / h - 0.5; z = P[i, 2] / h - 0.5
        if x < 0 or y < 0 or x > nx - 1 or y > ny - 1 or z > nz - 1:
            ua = 0.0; va = 0.0; wa = 0.0
        else:
            ua = trilin(u, x, y, z); va = trilin(v, x, y, z); wa = trilin(w, x, y, z)
        k = 1.0 - math.exp(-dt / tau[i])
        V[i, 0] += (ua - V[i, 0]) * k
        V[i, 1] += (va - V[i, 1]) * k
        V[i, 2] += (wa - V[i, 2]) * k - vs[i] / tau[i] * dt
        P[i, 0] += V[i, 0] * dt; P[i, 1] += V[i, 1] * dt; P[i, 2] += V[i, 2] * dt
        if P[i, 2] < 0.0:
            P[i, 2] = -P[i, 2] * rest
            if tau[i] > 0.3:
                alive[i] = False                      # sand grains land and stay
            else:
                V[i, 2] = -V[i, 2] * rest; V[i, 0] *= 0.6; V[i, 1] *= 0.6


@njit(cache=True)
def splat(P, m, alive, d, o, h):
    nx, ny, nz = d.shape
    for i in range(P.shape[0]):
        if not alive[i]: continue
        x = (P[i, 0] - o[0]) / h - 0.5; y = (P[i, 1] - o[1]) / h - 0.5; z = P[i, 2] / h - 0.5
        if x < 0 or y < 0 or z < -0.5 or x >= nx - 1 or y >= ny - 1 or z >= nz - 1: continue
        if z < 0: z = 0.0
        ii = int(x); jj = int(y); kk = int(z)
        fx = x - ii; fy = y - jj; fz = z - kk
        for a in range(2):
            wx = fx if a else 1 - fx
            for b in range(2):
                wy = fy if b else 1 - fy
                for c in range(2):
                    wz = fz if c else 1 - fz
                    d[ii + a, jj + b, kk + c] += m[i] * wx * wy * wz


@njit(parallel=True, cache=True)
def divergence_minus(u, v, w, h, div, src):
    divergence(u, v, w, h, div)
    nx, ny, nz = u.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                div[i, j, k] -= src[i, j, k]


# ---------------------------------------------------------------- marching with point lights (the glowing ball)
@njit(parallel=True, cache=True)
def march_l(dirs, org, d, ts, sky, gmin, h, zmax, kappa, albedo, sun_rad, sky_rad, phase_g, sundir, step, seed_img,
            lp, lc, tmax, rgb, trans, gshadow):
    H, W, _ = dirs.shape
    nx, ny, nz = d.shape
    gmax0 = gmin[0] + nx * h; gmax1 = gmin[1] + ny * h; gmax2 = min(nz * h, zmax)
    g2 = phase_g * phase_g
    nl = lp.shape[0]
    for py in prange(H):
        for px in range(W):
            dx = dirs[py, px, 0]; dy = dirs[py, px, 1]; dz = dirs[py, px, 2]
            t0 = 0.0; t1 = 1e9
            for a in range(3):
                o = org[a]; dd = dirs[py, px, a]
                lo = gmin[a] if a < 2 else 0.0
                hi = gmax0 if a == 0 else (gmax1 if a == 1 else gmax2)
                if abs(dd) < 1e-9:
                    if o < lo or o > hi: t1 = -1.0
                else:
                    ta = (lo - o) / dd; tb = (hi - o) / dd
                    if ta > tb: ta, tb = tb, ta
                    t0 = max(t0, ta); t1 = min(t1, tb)
            hit_ground = False
            tg = 1e9
            if dz < 0:
                tg = -org[2] / dz
                if tg < t1:
                    t1 = tg; hit_ground = True
            if tmax[py, px] < t1:                     # an object in front: dust behind it is hidden
                t1 = tmax[py, px]
            T = 1.0; r = 0.0; gg = 0.0; b = 0.0
            if t1 > t0:
                cosang = dx * sundir[0] + dy * sundir[1] + dz * sundir[2]
                ph = (1 - g2) / (4 * math.pi * (1 + g2 - 2 * phase_g * cosang) ** 1.5)
                t = t0 + step * seed_img[py, px]
                while t < t1 and T > 0.003:
                    sx = org[0] + t * dx; sy = org[1] + t * dy; sz = org[2] + t * dz
                    x = (sx - gmin[0]) / h - 0.5
                    y = (sy - gmin[1]) / h - 0.5
                    z = sz / h - 0.5
                    rho = trilin(d, x, y, z)
                    if rho > 1e-4:
                        st = kappa * rho
                        a_ = 1.0 - math.exp(-st * step)
                        tsun = trilin(ts, x, y, z); occ = trilin(sky, x, y, z)
                        lr = 0.0; lg = 0.0; lb = 0.0
                        for l in range(nl):
                            ex = sx - lp[l, 0]; ey = sy - lp[l, 1]; ez = sz - lp[l, 2]
                            d2 = ex * ex + ey * ey + ez * ez + 0.01
                            dd_ = math.sqrt(d2)
                            c2 = (ex * dx + ey * dy + ez * dz) / dd_
                            pl = (1 - g2) / (4 * math.pi * (1 + g2 - 2 * phase_g * c2) ** 1.5) / d2
                            lr += lc[l, 0] * pl; lg += lc[l, 1] * pl; lb += lc[l, 2] * pl
                        amb = 0.25 + 0.75 * occ
                        r += T * a_ * albedo[0] * (sun_rad[0] * tsun * ph + sky_rad[0] * amb + lr)
                        gg += T * a_ * albedo[1] * (sun_rad[1] * tsun * ph + sky_rad[1] * amb + lg)
                        b += T * a_ * albedo[2] * (sun_rad[2] * tsun * ph + sky_rad[2] * amb + lb)
                        T *= 1.0 - a_
                    t += step
            rgb[py, px, 0] = r; rgb[py, px, 1] = gg; rgb[py, px, 2] = b
            trans[py, px] = T
            gshadow[py, px] = 1.0
            if hit_ground:
                x = (org[0] + t1 * dx - gmin[0]) / h - 0.5
                y = (org[1] + t1 * dy - gmin[1]) / h - 0.5
                if x > 0 and y > 0 and x < nx - 1 and y < ny - 1:
                    gshadow[py, px] = trilin(ts, x, y, 0.0)


def ray_tmax(R, depth, alpha):
    """per-pixel distance along the ray to the first opaque CG surface (from a z-depth pass), inf where there is none"""
    fwd = C.R()[2]
    cosf = (R.dirs @ fwd).astype(np.float32)
    t = np.where(alpha > 0.5, depth / np.maximum(cosf, 1e-3), np.inf).astype(np.float32)
    if t.shape != R.dirs.shape[:2]:
        import cv2
        t = cv2.resize(t, (R.W, R.H), interpolation=cv2.INTER_NEAREST)
    return t


def render_l(R, lights=None, zmax=1e9, step=None, tmax=None):
    """Renderer.render with point lights [(pos, rgb intensity)], the box cut at zmax, and the march stopped at CG
    surfaces (tmax, from ray_tmax)"""
    g = R.g
    ts = np.empty_like(g.d); sky = np.empty_like(g.d)
    sd = (R.sundir / np.abs(R.sundir).max()).astype(np.float32)
    sun_transmittance(g.d, g.h * float(np.linalg.norm(sd)), sd, R.kappa, ts)
    sky_occlusion(g.d, g.h, R.kappa * 0.6, sky)
    rgb = np.zeros((R.H, R.W, 3), np.float32); tr = np.ones((R.H, R.W), np.float32)
    gs = np.ones((R.H, R.W), np.float32)
    lights = lights or []
    lp = np.array([l[0] for l in lights] or [[0, 0, -100]], np.float32).reshape(-1, 3)
    lc = np.array([l[1] for l in lights] or [[0, 0, 0]], np.float32).reshape(-1, 3)
    if tmax is None:
        tmax = np.full((R.H, R.W), np.inf, np.float32)
    march_l(R.dirs, R.org, g.d, ts, sky, g.o[:2].copy(), g.h, float(zmax), R.kappa, R.albedo, R.sun_rad, R.sky_rad,
            R.phase_g, R.sundir, step or g.h * 0.6, R.seed, lp, lc, tmax, rgb, tr, gs)
    return rgb, tr, gs
