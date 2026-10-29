"""What the compositor adds to a flyby: light bent by the shock waves of a supersonic machine (the bow and tail shocks'
Mach cones sweeping out from its path), the shimmer of its hot exhaust trail, the camera shaken when the boom
reaches it, and its dust (flyby_dust.py)."""
import numpy as np, math, os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as Cam, comp as C, dust as D, flight as F

W, H = Cam.W, Cam.H_IMG
R_ = "/home/user/lot/render"
# shock refraction: px the plate moves across the bow shock where a ray grazes it 6 m from the path (falling as
# d^-3/4 like the pressure jump); shake: px, when the boom reaches the camera
FX = {"x1": dict(refr=2.2, shake=9.0, trail=1.2),
      "f35": dict(refr=3.2, shake=14.0, trail=1.8),
      "x15": dict(refr=7.0, shake=24.0, trail=1.8)}
_geo = {}


def geometry(alt):
    """per pixel: the closest approach of its ray (up to where it meets the ground) to the flight line, where along
    the line that is, and the on-screen direction away from the line"""
    if alt in _geo: return _geo[alt]
    R0 = D.Renderer(D.Grid(h=1.0), scale=1.0)
    dirs, org = R0.dirs, R0.org
    dz = dirs[..., 2]
    tg = np.where(dz < -1e-6, -org[2] / np.minimum(dz, -1e-6), np.inf)
    oy, oz = org[1] - F.FLY_Y, org[2] - alt
    ts = -(oy * dirs[..., 1] + oz * dirs[..., 2]) / (dirs[..., 1] ** 2 + dirs[..., 2] ** 2)
    ts = np.clip(ts, 0, tg)
    py = org[1] + ts * dirs[..., 1] - F.FLY_Y; pz = org[2] + ts * dirs[..., 2] - alt
    dmin = np.sqrt(py ** 2 + pz ** 2).astype(np.float32)
    xs = (org[0] + ts * dirs[..., 0]).astype(np.float32)
    gy, gx = np.gradient(dmin)
    gn = np.sqrt(gx ** 2 + gy ** 2) + 1e-9
    _geo[alt] = (dmin, xs, (gx / gn).astype(np.float32), (gy / gn).astype(np.float32))
    return _geo[alt]


_rng = np.random.default_rng(5)
SHIM = [_rng.random((H // 10, W // 10)).astype(np.float32) for _ in range(8)]


def displacement(name, t):
    """(dx, dy) px at flight time t, or None"""
    if name not in FX: return None
    m, fx = F.MACHINES[name], FX[name]
    mach = m["speed"] / 343.0
    if mach <= 1: return None
    cot = math.sqrt(mach ** 2 - 1)
    dmin, xs, gx, gy = geometry(m["alt"])
    L = m["box"][0]
    disp = np.zeros((H, W), np.float32)
    if name == "x15":
        # the line blast: a cylinder of shock opening out from the path (hyper.py), bending light most while strong
        import hyper as Hy
        a = Hy.age(xs + L / 2, t)
        on = a > 0
        if on.any():
            Rr = Hy.shock_r(np.where(on, a, 0)).astype(np.float32)
            amp = fx["refr"] * np.minimum(1.0, Hy.R_WEAK / np.maximum(Rr, 1e-3)) ** 1.1
            w = 0.05 + 0.03 * Rr
            q = (dmin - Rr) / w
            disp = np.where(on & (Rr < 30), amp * (-q * np.exp(0.5 - 0.5 * q * q)), 0).astype(np.float32)
    for x_apex, sign in (() if name == "x15" else ((m["speed"] * t + L / 2, 1.0), (m["speed"] * t - L / 2, -0.8))):
        Rr = (x_apex - xs) / cot                          # the cone's radius at each pixel's closest point
        on = Rr > 0.3
        if not on.any(): continue
        w = 0.05 + 0.012 * Rr
        q = (dmin - Rr) / w
        amp = fx["refr"] * (6.0 / np.maximum(Rr, 1.5)) ** 0.75
        disp += np.where(on & (Rr < 40), sign * amp * (-q * np.exp(0.5 - 0.5 * q * q)), 0).astype(np.float32)
    dx, dy = disp * gx, disp * gy
    # the exhaust trail: hot gas left behind the nozzles, shimmering, cooling over a second or so
    age = (m["speed"] * t - L / 2 - xs) / m["speed"]
    hot = np.where(age > 0, np.exp(-np.maximum(dmin - 0.4 - 0.8 * np.sqrt(np.maximum(age, 0)), 0) / 0.5) *
                   np.exp(-np.maximum(age, 0) / 1.2), 0).astype(np.float32)
    if hot.max() > 0.01:
        ph = t * 40.0
        nx = cv2.resize(SHIM[int(ph) % 8], (W, H), interpolation=cv2.INTER_CUBIC) * (1 - ph % 1) + \
             cv2.resize(SHIM[(int(ph) + 1) % 8], (W, H), interpolation=cv2.INTER_CUBIC) * (ph % 1)
        ny = cv2.resize(SHIM[(int(ph) + 3) % 8], (W, H), interpolation=cv2.INTER_CUBIC)
        dx = dx + fx["trail"] * hot * (nx - 0.5); dy = dy + fx["trail"] * hot * (ny - 0.5)
    return dx.astype(np.float32), dy.astype(np.float32)


_grid = None


def warp(img, d):
    global _grid
    if d is None: return img
    if _grid is None: _grid = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, _grid[1] - d[0], _grid[0] - d[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def shake_t(name, t):
    """seconds since the boom reached the camera (negative before), for comp.shake"""
    if name not in FX: return -1.0
    import fly_sound
    if not hasattr(shake_t, "c"): shake_t.c = {}
    if name not in shake_t.c:
        n0 = round(F.MACHINES[name]["pre"] * F.FPS)
        shake_t.c[name] = fly_sound.cone_arrival(name, n0)[0] - n0 / F.FPS
    return t - shake_t.c[name]


def wake_glow(name, t, expo):
    """the X-15's wake: air the bow shock and the blast heat to 1,500-2,400 K, glowing orange behind it and cooling
    within a couple of milliseconds; premultiplied linear light to add (None if nothing glows)"""
    if name != "x15": return None
    import hyper as Hy
    m = F.MACHINES[name]
    xn = m["speed"] * t - m["box"][0] / 2                      # the tail: the wake starts there
    ages = np.geomspace(2e-5, 3e-3, 40)
    img = np.zeros((H, W, 3), np.float32)
    for a in ages:
        x = xn - m["speed"] * a
        uv, z = Cam.project([[x, F.FLY_Y, m["alt"]]])
        if z[0] <= 0: continue
        u, v = uv[0]
        if u < -400 or u > W + 400: continue
        r_m = 0.3 + 0.25 * float(Hy.shock_r(a)) ** 0.5
        r_px = max(2.0, Cam.F * r_m / z[0])
        seg = m["speed"] * a * (ages[1] / ages[0] - 1)               # metres of wake this sample stands for
        blur = m["speed"] * expo * Cam.F / z[0]                      # smeared by the motion during the exposure
        L_px = Cam.F * seg / z[0] + blur
        k = math.exp(-a / 4e-4)
        col = np.array([1.0, 0.45 + 0.35 * k, 0.15 + 0.3 * k ** 2], np.float32) * 2.2 * (0.25 + 0.75 * k)
        x0, x1 = int(u - L_px - 3 * r_px), int(u + 3 * r_px)
        y0, y1 = int(v - 3 * r_px), int(v + 3 * r_px)
        xa, xb, ya, yb = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
        if xa >= xb or ya >= yb: continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        dxl = np.clip(xx - u, -L_px, 0) - (xx - u)                   # distance to the streak's segment
        prof = np.exp(-(dxl ** 2 + (yy - v) ** 2) / (2 * r_px ** 2))
        keep = (Cam.F * seg / z[0] + 2.5 * r_px) / (L_px + 2.5 * r_px)   # the same light spread over the smear
        img[ya:yb, xa:xb] += prof[..., None] * col * 0.5 * keep
    return img if img.max() > 1e-4 else None


def dust_layer(name, n, replay=False):
    p = f"{R_}/fly_{name}{'_replay' if replay else ''}_dust"
    return C.dust(f"{p}/d{n:04d}.png", f"{p}/g{n:04d}.png")
