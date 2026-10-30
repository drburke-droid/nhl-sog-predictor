"""What the compositor adds to a flyby: light bent by the shock waves of a supersonic machine (the bow and tail shocks'
Mach cones sweeping out from its path), the shimmer of its hot exhaust trail, the camera shaken when the boom
reaches it, and its dust (flyby_dust.py)."""
import numpy as np, math, os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as Cam, comp as C, dust as D, flight as F

W, H = Cam.W, Cam.H_IMG
R_ = "/home/user/lot/render"
# shake: px, when the boom reaches the camera; trail: px of shimmer in the hot exhaust trail
FX = {"x1": dict(shake=9.0, trail=1.2),
      "f35": dict(shake=14.0, trail=1.8),
      "x15": dict(shake=24.0, trail=1.8)}
# light bent by a shock: the air behind it is denser, so each shell (a Mach cone, or the X-15's blast cylinder) is the
# edge of a weak cylindrical lens. A ray passing d from the axis of a shell of radius R with the index stepping up by dn
# inside is bent toward the axis by 2 dn d / sqrt(R^2 - d^2) (the cone's slant cancels out), averaged here over the
# pixel, so the bend is strongest in the pixel just inside the shell and falls off inward. The index step comes from
# the pressure jump: the peak pressure at the camera (make_cuts.KPA, in the 20 m shots) scaled as distance^-3/4, or,
# close to the X-15's path, the strong shock of its line blast (hyper.py); density across the shock by Rankine-Hugoniot.
N_AIR = 2.6e-4                            # refractivity of the desert air, n - 1
KPA_SHOT = {"x1": 5.9, "f35": 11.8, "x15": 20.0}       # make_cuts.KPA: peak pressure at the camera in the 20 m shots
D_SHOT = math.hypot(20.0, 6.0 - 1.37)
TAIL = 0.8                                # the tail shock's jump against the bow shock's
_geo = {}


def geometry(alt, fly_y=F.FLY_Y):
    """per pixel: the closest approach of its ray (up to where it meets the ground) to the flight line, where along
    the line that is, and the on-screen direction away from the line"""
    key = (alt, fly_y)
    if key in _geo: return _geo[key]
    R0 = D.Renderer(D.Grid(h=1.0), scale=1.0)
    dirs, org = R0.dirs, R0.org
    dz = dirs[..., 2]
    tg = np.where(dz < -1e-6, -org[2] / np.minimum(dz, -1e-6), np.inf)
    oy, oz = org[1] - fly_y, org[2] - alt
    ts = -(oy * dirs[..., 1] + oz * dirs[..., 2]) / (dirs[..., 1] ** 2 + dirs[..., 2] ** 2)
    ts = np.clip(ts, 0, tg)
    py = org[1] + ts * dirs[..., 1] - fly_y; pz = org[2] + ts * dirs[..., 2] - alt
    dmin = np.sqrt(py ** 2 + pz ** 2).astype(np.float32)
    xs = (org[0] + ts * dirs[..., 0]).astype(np.float32)
    gy, gx = np.gradient(dmin)
    gn = np.sqrt(gx ** 2 + gy ** 2) + 1e-9
    _geo[key] = (dmin, xs, (gx / gn).astype(np.float32), (gy / gn).astype(np.float32))
    _geo[key + ("px",)] = gn.astype(np.float32)          # m of closest approach per px
    return _geo[key]


def index_step(name, R):
    """n inside - n outside, just behind a shock R m from the path"""
    R = np.maximum(np.asarray(R, np.float32), 0.5)
    dp = KPA_SHOT[name] * 1e3 * (D_SHOT / R) ** 0.75
    if name == "x15":
        import hyper as Hy
        ms = np.maximum(1.0, Hy.BLAST ** 2 / (2 * R) / Hy.C_SOUND)       # the strong blast's own shock speed
        dp = np.maximum(dp, Hy.P0 * 2.8 / 2.4 * (ms ** 2 - 1))
    P = 1 + dp / 1.013e5
    return N_AIR * ((2.4 * P + 0.4) / (0.4 * P + 2.4) - 1)


def bend(d, R, dn, h):
    """px a ray is bent toward the axis, its pixel spanning d - h .. d + h of closest approach to a shell of radius R"""
    a = np.clip(d - h, 0, R); b = np.clip(d + h, 0, R)
    return 2 * dn * (np.sqrt(R * R - a * a) - np.sqrt(R * R - b * b)) / (2 * h) * Cam.F


_rng = np.random.default_rng(5)
SHIM = [_rng.random((H // 10, W // 10)).astype(np.float32) for _ in range(8)]


def displacement(name, t):
    """(dx, dy) px at flight time t, or None"""
    if name not in FX: return None
    m, fx = F.MACHINES[name], FX[name]
    mach = m["speed"] / 343.0
    if mach <= 1: return None
    cot = math.sqrt(mach ** 2 - 1)
    dmin, xs, gx, gy = geometry(m["alt"], F.lane_y(name))
    h = 0.5 * _geo[(m["alt"], F.lane_y(name), "px")]
    L = m["box"][0]
    disp = np.zeros((H, W), np.float32)
    if name == "x15":
        # the line blast: a cylinder of shock opening out from the path (hyper.py), bending light most while strong
        import hyper as Hy
        a = Hy.age(xs + L / 2, t)
        on = a > 0
        if on.any():
            Rr = Hy.shock_r(np.where(on, a, 0)).astype(np.float32)
            disp = np.where(on & (Rr < 30), bend(dmin, Rr, index_step(name, Rr), h), 0).astype(np.float32)
    for x_apex, k in (() if name == "x15" else ((m["speed"] * t + L / 2, 1.0), (m["speed"] * t - L / 2, TAIL))):
        Rr = (x_apex - xs) / cot                          # the cone's radius at each pixel's closest point
        on = Rr > 0.3
        if not on.any(): continue
        Rc = np.maximum(Rr, 0.3)
        disp += np.where(on & (Rr < 40), bend(dmin, Rc, k * index_step(name, np.maximum(Rc, 1.5)), h), 0).astype(np.float32)
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
        uv, z = Cam.project([[x, F.lane_y(name), m["alt"]]])
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
        keep *= min(1.0, 0.005 / max(expo, 1e-9)) if expo > 0.012 else 1.0   # a long exposure: it glows a sliver of it
        img[ya:yb, xa:xb] += prof[..., None] * col * 0.5 * keep
    return img if img.max() > 1e-4 else None


def dust_layer(name, n, replay=False):
    p = f"{R_}/fly_{name}{'_replay' if replay else ''}_dust{F.REPLAY_SUF.get(name, '') if replay else ''}"
    return C.dust(f"{p}/d{n:04d}.png", f"{p}/g{n:04d}.png")
