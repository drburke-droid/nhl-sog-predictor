"""The finale: the Parker Solar Probe at 430,000 mph (192 km/s) in sea-level air, 5 m up, 20 m out.

It cannot survive it. The air it hits carries 0.5 rho v^3 = 4e15 W per m^2 of heat shield; even if only a tenth reached
it, carbon (60 MJ/kg to vaporize) would boil off at 30 tonnes a second: the 73 kg heat shield is gone in a few
microseconds and the 685 kg probe within about 25 us, some 5 m after it meets the air. What carries on is its vapour,
a plasma bolt still at nearly full speed, slowed only as it shoves air aside: with its momentum shared with the air it
has swept (a snowplough), v = v0 / (1 + x / L), L = mass / (rho * area) = 135 m. The kinetic energy it gives the air,
d(KE)/dx, is 9e10 J per metre where it enters (22 tonnes of TNT a metre), 6e10 J/m as it leaves the picture: 0.6 kilotons
in the 33 m in view. That is a line blast: R = (E'/rho)^(1/4) sqrt(age), ~500 sqrt(age); it reaches the ground under
the path in 0.1 ms, the car in 0.15 ms, the camera 20 m away in 1.65 ms, still moving at Mach 18 (380 atmospheres):
nothing in the picture survives it, the camera last.

Here the probe meets the air just inside the picture at the left (as if it appeared there), so it is seen whole for a
moment before it goes."""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flight as F, camera as Cam

V0 = 192227.0
MASS, AREA, RHO = 685.0, 4.15, 1.225
PY, PZ = F.FLY_Y, F.MACHINES["parker"]["alt"]
X_IN = -13.5                     # where the probe meets the air (its heat shield), just inside the picture
LS = MASS / (RHO * AREA)         # snowplough length, 135 m
KE0 = 0.5 * MASS * V0 ** 2
T_SHIELD, T_GONE = 4e-6, 25e-6   # heat shield gone; the whole probe vapour
CAM_P = np.array([Cam.CAM_X, 0.0, Cam.H])


def x_head(t):
    """where the head of the probe (then of its plasma) is at time t (t = 0 when it meets the air at X_IN)"""
    # dx/dt = v0 / (1 + s / LS), s = x - X_IN  ->  s + s^2 / (2 LS) = v0 t
    t = np.asarray(t, float)
    s = LS * (np.sqrt(1 + 2 * V0 * np.maximum(t, 0) / LS) - 1)
    return X_IN + np.where(t > 0, s, V0 * t)


def t_at(x):
    """when the head passes x"""
    s = np.asarray(x, float) - X_IN
    return np.where(s > 0, (s + s ** 2 / (2 * LS)) / V0, s / V0)


def e_line(x):
    """J per metre the plasma leaves in the air at x"""
    s = np.maximum(np.asarray(x, float) - X_IN, 0)
    return KE0 / LS / (1 + s / LS) ** 2


def blast(x):
    return (e_line(x) / RHO) ** 0.25


def shock_r(x, t):
    """radius of the blast around the path at x at time t"""
    a = np.maximum(t - t_at(x), 0)
    return blast(x) * np.sqrt(a)


def t_reach(p):
    """when the blast reaches point p (its closest point on the path, then outward)"""
    p = np.asarray(p, float)
    d = math.hypot(p[1] - PY, p[2] - PZ)
    x = max(p[0], X_IN)
    return float(t_at(x) + (d / blast(x)) ** 2)


def temperature(a):
    """K of the air the plasma has heated, a seconds after it passed: tens of thousands at first, still several
    thousand a millisecond later (the core cools by expanding and radiating; a fireball of this energy stays
    luminous for tens of milliseconds)"""
    return 40000.0 * (1 + np.asarray(a, float) / 2e-6) ** -0.35


T_CAMERA = t_reach(CAM_P)

# the replay: 2 us a frame (20,000 times slower) while the probe crosses, then each frame's step growing by the
# same factor until the blast reaches the camera
STEP0 = 2e-6


def replay():
    out = []
    t = -6 * STEP0
    t_exit = float(t_at(17.0))
    while t < t_exit:
        out.append((t, STEP0 * F.FPS)); t += STEP0
    n = 60
    lo, hi = 1.0, 1.2
    for _ in range(60):                                    # the ratio that lands the last frame on T_CAMERA
        r = (lo + hi) / 2
        if t + STEP0 * r * (r ** n - 1) / (r - 1) > T_CAMERA: hi = r
        else: lo = r
    step = STEP0
    for _ in range(n):
        step *= r
        t += step
        out.append((min(t, T_CAMERA), step * F.FPS))
    return out


if __name__ == "__main__":
    print(f"snowplough length {LS:.0f} m; E' at entry {e_line(X_IN):.2e} J/m, at x=17 {e_line(17):.2e}")
    print(f"kinetic energy {KE0:.2e} J = {KE0 / 4.184e12:.1f} kt; left in view {KE0 - KE0 / (1 + (17 - X_IN) / LS):.2e} J")
    print(f"head leaves the picture at {float(t_at(17)) * 1e6:.0f} us; blast reaches ground below x=0 at "
          f"{t_reach((0, PY, 0)) * 1e6:.0f} us, the car at {t_reach((5.6, 17.2, 1.0)) * 1e6:.0f} us, the camera at "
          f"{T_CAMERA * 1e3:.2f} ms")
    r = replay()
    print(len(r), "replay frames; last steps", [f"{s / F.FPS * 1e6:.0f} us" for _, s in r[-3:]])
