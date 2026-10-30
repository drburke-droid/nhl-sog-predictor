"""The physics of a hypersonic flyby: the X-15 at 4,520 mph (Mach 5.9 in sea-level air), 6 m up, 20 m out.

Drag: q = 0.5 rho v^2 = 2.45 MPa (355 psi; twenty times the most it was built for, see the README); with its
zero-lift drag coefficient at Mach 6 (about 0.065 on its 18.6 m^2 wing) that is about 3 MN, so it leaves 3 MJ in every
metre of air it crosses: a line blast. The blast is a strong cylindrical shock, R = BLAST sqrt(age), BLAST =
(E/rho)^(1/4), until it slows to the speed of sound (age 3.4 ms, R 2.3 m), then travels on at 343 m/s; its impulse
per area falls as 1/distance, scaled from the blast's own size R0 = sqrt(E / p0) = 5.5 m."""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flight as F

NAME = "x15"
M_ = F.MACHINES[NAME]
V, PY, PZ = M_["speed"], F.lane_y("x15"), M_["alt"]
E_LINE = 3.0e6                       # J per metre of path
RHO, P0, C_SOUND = 1.225, 1.013e5, 343.0
BLAST = (E_LINE / RHO) ** 0.25
T_WEAK = (0.5 * BLAST / C_SOUND) ** 2
R_WEAK = BLAST * math.sqrt(T_WEAK)
R0 = math.sqrt(E_LINE / P0)
REFLECT = 1.5                        # things standing on the ground feel the shock and its reflection off the ground


def age(x, t):
    """seconds since the nose passed x (negative: not yet); t = 0 as it crosses x = 0"""
    return t - x / V


def shock_r(a):
    a = np.asarray(a, float)
    return np.where(a <= 0, 0.0, np.where(a < T_WEAK, BLAST * np.sqrt(np.maximum(a, 0)),
                                          R_WEAK + C_SOUND * (a - T_WEAK)))


def travel(d):
    """seconds for the shock to get d metres from the path"""
    d = np.asarray(d, float)
    return np.where(d < R_WEAK, (d / BLAST) ** 2, T_WEAK + (d - R_WEAK) / C_SOUND)


def dist(y, z):
    return np.hypot(np.asarray(y, float) - PY, np.asarray(z, float) - PZ)


def t_hit(x, y, z):
    """when the blast reaches the point (x, y, z), on the flight's clock"""
    return x / V + travel(dist(y, z))


def impulse(d):
    """Pa s on a surface facing the path, d m from it (reflection off the ground included)"""
    return REFLECT * P0 * R0 / C_SOUND * 0.125 * R0 / np.maximum(np.asarray(d, float), 1.0)


def direction(x, y, z):
    """unit vector the blast pushes along at (x, y, z): straight away from the path"""
    v = np.array([0.0, y - PY, z - PZ]); return v / np.linalg.norm(v)


if __name__ == "__main__":
    print(f"BLAST {BLAST:.1f}, weak at {T_WEAK * 1e3:.1f} ms, R {R_WEAK:.2f} m, R0 {R0:.1f} m")
    for what, (x, y, z) in {"ground below": (0, 20, 0), "cone row": (0, 13.5, 0.35), "car roof": (5.6, 19.4, 1.4),
                            "sign panel": (4.5, 10.2, 2.0), "camera": (-1.37, 0, 1.37)}.items():
        d = float(dist(y, z))
        print(f"{what:12s} {d:5.1f} m  hit {float(travel(d)) * 1e3:6.1f} ms after the nose passes  "
              f"impulse {float(impulse(d)):5.0f} Pa s")


# ---------------------------------------------------------------- the props that don't tumble (cones and drums: rigid.py)
SIGN = dict(x=4.5, y=10.2, z=2.0, theta=math.radians(68))      # the 100 sign's panel; how far the post folds over
CAR = dict(x=5.6, y=17.2, z=1.4)                                # the BMW's roof, 4.6 m below and beside the path


def sign_bend(t):
    """angle the post leans toward the camera (radians), pivoting at its base: 155 Pa s on its panel is ~40 times the
    impulse the post can take without yielding"""
    u = t - float(t_hit(SIGN["x"], SIGN["y"], SIGN["z"]))
    if u <= 0: return 0.0
    s = 1 - math.exp(-u / 0.03) * (1 + u / 0.03)
    wob = math.radians(5) * math.exp(-u / 0.25) * math.sin(2 * math.pi * 7 * u)
    return SIGN["theta"] * s + wob * s


def car_motion(t):
    """(slide toward the camera m, pitch rad, heave m, roll rad) of the parked car: ~350 Pa s on its roof and bonnet
    (2,800 N s down, ~1,500 N s back) slams it onto its bump stops and shoves it back on its locked wheels"""
    u = t - float(t_hit(CAR["x"], CAR["y"], CAR["z"]))
    if u <= 0: return 0.0, 0.0, 0.0, 0.0
    slide = 0.16 * (1 - math.exp(-u / 0.09))
    w, z = 2 * math.pi * 1.5, 0.22
    env = math.exp(-z * w * u)
    heave = -0.09 * env * math.cos(w * u * 0.98) * (1 - math.exp(-u / 0.006))
    pitch = math.radians(3.0) * env * math.sin(w * u + 0.4)
    roll = math.radians(2.2) * env * math.sin(w * 1.1 * u + 1.2)
    return slide, pitch, heave, roll


def car_glass_broken(t):
    return t >= float(t_hit(CAR["x"], CAR["y"], CAR["z"])) + 0.002


def first_hit():
    return float(t_hit(CAR["x"], CAR["y"], CAR["z"]))
