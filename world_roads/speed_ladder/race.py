"""The final cut's race: every machine clears the right edge of the picture at the same instant (T = 0, the finish),
each in its own lane (flight.RACE_LANES), so they start staggered by their speeds. The Wright Flyer comes in at real
speed; as it nears the far side everything slows to 48 times slower, and stays there through the finish while the
others come in one after another. After the finish, while the booms reach the camera, it ramps back to real speed.
Run everything that uses this with RACE=1 (flight.py reads it). No Blender here.

Lanes: none of them can share one line, or they would meet at the finish. The slowest flies nearest and lowest; the
faster ones farther out, the Spitfire high, the F-35 lower, the X-15 low and farthest, so at the finish they stand
apart on the screen."""
import math, os, sys, functools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as Cam, flight as F

assert F.RACE, "run with RACE=1"
ORDER = ["wright", "spitfire", "f35", "x15"]
VERTS = "/home/user/lot/layout/verts_{}.npy"     # a sample of each model's vertices, about its centre (layout_verts.py)
SLOW = 1 / 48
FPS = F.FPS
LEAD = 0.6                  # s of empty road before the Wright Flyer appears
SLOW_FROM = -0.285          # 48x from here on (the Spitfire comes in at -0.272)
DOWN = 20                   # frames slowing down to it
AFTER = 0.025               # s at 48x after the finish (the F-35's boom reaches the camera at about +0.008)
UP = 50                     # frames ramping back to real speed (the X-15's blast arrives on the way)
TAIL = 62                   # frames at real speed to end
# PARKER=1: the race with the Parker Solar Probe (parker.py, in flight.PARKER_LANE). It crosses in 1.25 ms, so in the
# last few milliseconds everything slows again, to 1,000x, while it comes in at the left and crosses with the others;
# after the finish the frames lengthen step by step until its blast reaches the camera (57 ms after it met the air)
PARKER = F.PARKER
DEEP = 1 / 1000
PRE_P = 0.2e-3              # s at 1,000x before it meets the air
DEEP_N = 25                 # frames slowing from 48x to 1,000x
AFTER_P = 0.2e-3            # s at 1,000x after the finish
BLAST_N = 90                # frames from there to the blast reaching the camera
if PARKER:
    import parker as P


@functools.lru_cache(maxsize=None)
def verts(name):
    return np.load(VERTS.format(name))


def place(name, t):
    """world points of the machine at its own time t (t = 0 crossing x = 0)"""
    (x, y, z), (roll, pitch, yaw) = F.pose(name, t)
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    return verts(name) @ (Ry @ Rx).T + np.array([x, y, z])


def u_range(name, t):
    uv, zc = Cam.project(place(name, t))
    return uv[:, 0].min(), uv[:, 0].max()


@functools.lru_cache(maxsize=None)
def exit_time(name):
    """own time its last pixel leaves the right edge (Parker's: its head, the probe's own time counted from when it
    meets the air at the left edge)"""
    if name == "parker": return float(P.t_at(P.X_FINISH))
    v = F.MACHINES[name]["speed"]
    lo, hi = 0.0, 200.0 / v
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if u_range(name, mid)[0] < Cam.W else (lo, mid)
    return hi


@functools.lru_cache(maxsize=None)
def entry_time(name):
    """own time its first pixel comes in at the left edge"""
    if name == "parker": return 0.0
    v = F.MACHINES[name]["speed"]
    lo, hi = -200.0 / v, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if u_range(name, mid)[1] < 0 else (lo, mid)
    return hi


def own(name, T):
    """the machine's own time at race time T"""
    return T + exit_time(name)


def across(name, T):
    """how far across the picture it has come: 0 coming in, 1 gone"""
    return (own(name, T) - entry_time(name)) / (exit_time(name) - entry_time(name))


@functools.lru_cache(maxsize=None)
def timeline():
    """[(T, slow)] for every frame of the cut: T race time (s, 0 at the finish), slow the frame's step against real
    time (1 at real speed, 1/48 at 48x); exposures are a quarter of a frame's step, as in the shots"""
    step = SLOW / FPS
    n_slow = round(-SLOW_FROM / step)
    slow = [(-j * step, SLOW) for j in range(n_slow, -1, -1)] + \
           [(j * step, SLOW) for j in range(1, round(AFTER / step) + 1)]
    g = (1 / SLOW) ** (1 / DOWN)
    down, T, st = [], slow[0][0], step
    for i in range(DOWN):                     # backwards from where 48x begins: each step longer by g, to 1/25 s
        st *= g; T -= st; down.append((T, st * FPS))
    down = down[::-1]
    T0 = entry_time("wright") - exit_time("wright") - LEAD
    real, T = [], down[0][0]
    while T - 1 / FPS >= T0:
        T -= 1 / FPS; real.append((T, 1.0))
    real = real[::-1]
    if PARKER: return real + down + deep_end(slow, step)
    gu = (1 / SLOW) ** (1 / UP)
    up, T, st = [], slow[-1][0], step
    for i in range(UP):
        st *= gu; T += st; up.append((T, st * FPS))
    tail = [(T + (i + 1) / FPS, 1.0) for i in range(TAIL)]
    return real + down + slow + up + tail


def deep_end(slow, step):
    """PARKER=1: the 48x frames up to the second slowdown, the slowdown to 1,000x (each step shorter by the same
    factor, scaled a touch so that it starts on a 48x frame and ends on a 1,000x one), 1,000x through the finish, then
    each step longer by the same factor until the blast reaches the camera"""
    s1 = DEEP / FPS
    M = math.ceil((exit_time("parker") + PRE_P) / s1)
    hold = [(-m * s1, DEEP) for m in range(M, -1, -1)] + [(m * s1, DEEP) for m in range(1, round(AFTER_P / s1) + 1)]
    q = (s1 / step) ** (1 / DEEP_N)
    st = step * q ** np.arange(1, DEEP_N + 1)
    j = round((M * s1 + st.sum()) / step)
    st *= (j * step - M * s1) / st.sum()
    Ts = -j * step + np.cumsum(st)
    ramp = [(float(T), float(d * FPS)) for T, d in zip(Ts[:-1], st[:-1])]
    before = [(T, sl) for T, sl in slow if T <= -j * step + 1e-12]
    T_cam = P.T_CAMERA - exit_time("parker")
    T, span = hold[-1][0], T_cam - hold[-1][0]
    lo, hi = 1.0, 1.5
    for _ in range(60):                       # the ratio that lands the last frame on the blast reaching the camera
        r = (lo + hi) / 2
        lo, hi = (lo, r) if s1 * r * (r ** BLAST_N - 1) / (r - 1) > span else (r, hi)
    after, d = [], s1
    for i in range(BLAST_N):
        d *= r; T += d; after.append((min(T, T_cam), d * FPS))
    return before + ramp + hold + after


def finish_frame():
    return next(k for k, (T, s) in enumerate(timeline()) if abs(T) < 1e-12)


if __name__ == "__main__":
    tl = timeline()
    print(f"{len(tl)} frames ({len(tl) / FPS:.1f} s), the finish at frame {finish_frame()}")
    for n in ORDER:
        print(f"{n:9s} lane {F.RACE_LANES[n]}  in the picture {exit_time(n) - entry_time(n):.4f} s, comes in at "
              f"T = {entry_time(n) - exit_time(n):+.4f} s, frame "
              f"{next(k for k, (T, s) in enumerate(tl) if own(n, T) >= entry_time(n))}")
    for k in (0, len([1 for T, s in tl if s == 1.0 and T < 0]), finish_frame()):
        T = tl[k][0]
        print(k, f"T={T:+.4f}", "  ".join(f"{n} {100 * across(n, T):.0f}%" for n in ORDER if 0 < across(n, T) < 1))
