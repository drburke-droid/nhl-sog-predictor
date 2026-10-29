"""The slow-motion replay's clock and the physics of the pass it shows.

t is real time in seconds, 0 when the ball comes into view on the left (x = X_ENTER). The replay runs
  A: 1/2000 speed (a camera at 50,000 frames per second) while the ball crosses, 20 microseconds a frame;
  B: a ramp back to real speed, each frame's step growing by the same factor, 20 us to 40 ms in 60 frames;
  C: real speed for half a second.
The blast along the path is a strong cylindrical shock, R = BLAST sqrt(age) with BLAST = (E/rho)^(1/4) for E = 1.08 MJ
per metre, until it slows to the speed of sound (age 2 ms, R 1.37 m); after that it is a sound wave."""
import math
V = 7823.0                         # 17,500 mph in m/s
PY, PZ = 12.0, 0.6                 # the path: 12 m out, 0.6 m up, along x
X_ENTER, X_EXIT = -9.8, 10.1
E_LINE = 1.08e6                    # J per metre of path (drag, 20 cm ball)
BLAST = (E_LINE / 1.225) ** 0.25
C_SOUND = 343.0
T_WEAK = (0.5 * BLAST / C_SOUND) ** 2
R_WEAK = BLAST * math.sqrt(T_WEAK)
FPS = 25
SLOW = 2000


def ball_x(t):
    return X_ENTER + V * t


def age(x, t):
    """seconds since the ball passed x (negative: not yet)"""
    return t - (x - X_ENTER) / V


def shock_r(a):
    if a <= 0: return 0.0
    return BLAST * math.sqrt(a) if a < T_WEAK else R_WEAK + C_SOUND * (a - T_WEAK)


def timeline():
    """[(real time, slow-down factor)] for every frame of the replay"""
    out = []
    t, dt = -1.0e-4, 1.0 / (FPS * SLOW)
    t_exit = (X_EXIT - X_ENTER) / V
    while t <= t_exit + 1.0e-4:
        out.append((t, SLOW)); t += dt
    n, end = 60, 1.0 / FPS
    r = (end / dt) ** (1.0 / n)
    for k in range(1, n + 1):
        step = dt * r ** k
        t += step
        out.append((t, (1.0 / FPS) / step))
    for k in range(12):
        t += 1.0 / FPS
        out.append((t, 1.0))
    return out


if __name__ == "__main__":
    tl = timeline()
    print(len(tl), "frames;", "A ends", sum(1 for _, s in tl if s == SLOW), "; last t", round(tl[-1][0], 3))
    print("shock: weak at", round(T_WEAK * 1e3, 2), "ms, R", round(R_WEAK, 2), "m; reaches the camera (12 m) at",
          round((T_WEAK + (12 - R_WEAK) / C_SOUND) * 1e3, 1), "ms")


# ---------------------------------------------------------------- what glows
def wake_eps(a):
    """radiance of the hot core a seconds after the ball passed (linear; the sky is ~0.6)"""
    if a <= 0: return 0.0
    return 9.0 * (0.7 * math.exp(-a / 1e-4) + 0.3 * math.exp(-a / 7e-4))


_KEYS = [(0.0, (0.80, 0.86, 1.00)), (3e-5, (1.0, 0.95, 0.90)), (1.5e-4, (1.0, 0.75, 0.45)),
         (5e-4, (1.0, 0.50, 0.18)), (1.5e-3, (0.85, 0.25, 0.07)), (4e-3, (0.5, 0.08, 0.02))]


def wake_color(a):
    """white-blue at the head, through yellow and orange to dull red as the air cools"""
    if a <= _KEYS[0][0]: return _KEYS[0][1]
    for (a0, c0), (a1, c1) in zip(_KEYS, _KEYS[1:]):
        if a <= a1:
            f = (a - a0) / (a1 - a0)
            return tuple(c0[i] * (1 - f) + c1[i] * f for i in range(3))
    return _KEYS[-1][1]


def core_radius(a):
    return 0.05 + 0.14 * min(shock_r(a), 3.0)


HEAD_I = 200.0                       # the plasma around the ball as a point light (sun = 2.6 W/m^2 in these units);
                                     # the real thing (a 7000 K shock layer) would be ~25x brighter still


def lights(t):
    """the glowing ball and its wake as a handful of point lights: [((x, y, z), (r, g, b))]"""
    xb = ball_x(t)
    out = [((xb, PY, PZ), tuple(HEAD_I * c for c in (0.85, 0.9, 1.0)))]
    ages = [1e-5, 6e-5, 2e-4, 6e-4, 1.5e-3, 4e-3]
    for a0, a1 in zip(ages, ages[1:]):
        a = math.sqrt(a0 * a1)
        seg = V * (a1 - a0)
        i = 5.0 * wake_eps(a) / 6.0 * seg
        if i < 0.01: continue
        x = xb - V * a
        out.append(((x, PY, PZ), tuple(i * c for c in wake_color(a))))
    return out
