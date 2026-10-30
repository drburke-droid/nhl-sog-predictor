"""Where each machine is, moment by moment, as it crosses the road left to right, and which frames of its shot need it
rendered. No Blender here, so the compositor and the sound can use it too.

World: road along +y, x to the right, z up, ground z = 0; every machine flies along x at y = FLY_Y (in the final
cut's race, RACE=1, each in its own lane: RACE_LANES)."""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as Cam, scene_sun as SS

FLY_Y = 20.0
FPS = 25

# speed m/s, height of its centre m, bounding box (length, span, height) m as built (aircraft.py), seconds of the shot
# before and after it crosses the middle of the frame
MACHINES = {
    "wright":   dict(title="1903 Wright Flyer", mph=30, speed=13.41, alt=3.8, box=(6.56, 12.48, 2.94), pre=3.4, post=3.0),
    "spitfire": dict(title="1943 Spitfire", mph=370, speed=165.4, alt=4.0, box=(9.24, 11.24, 2.86), pre=2.0, post=2.0),
    "x1":       dict(title="1947 Bell X-1", mph=807, speed=360.6, alt=6.0, box=(9.42, 8.50, 2.82), pre=2.0, post=2.5),
    "f35":      dict(title="2016 F-35 Lightning II", mph=1228, speed=548.8, alt=6.0, box=(17.2, 10.84, 3.65), pre=2.0,
                     post=2.5),
    "x15":      dict(title="1967 X-15", mph=4520, speed=2020.6, alt=6.0, box=(15.86, 6.70, 3.26), pre=2.0, post=3.0),
    "parker":   dict(title="2024 Parker Solar Probe", mph=430000, speed=192227.0, alt=5.0, box=(6.62, 5.74, 5.26), pre=2.0,
                     post=3.0),
}


# the final cut's race (race.py): every machine clears the right edge at the same instant, so none can share a line;
# each has its own lane, (distance from the camera, height) m: the slow ones nearer and lower, the fast ones farther
RACE_LANES = {"wright": (13.0, 1.85), "spitfire": (22.0, 6.0), "f35": (32.0, 5.2), "x15": (48.0, 3.3)}
RACE = os.environ.get("RACE") == "1"
if RACE:
    for _n, (_y, _z) in RACE_LANES.items(): MACHINES[_n]["alt"] = _z
# the race with the Parker Solar Probe in it (PARKER=1 with RACE=1): 100 m out, so that its blast (~16 m in its first
# millisecond) reaches none of the others before the finish, and 12 m up, crossing open sky above the X-15
PARKER_LANE = (100.0, 12.0)
PARKER = RACE and os.environ.get("PARKER") == "1"
if PARKER:
    MACHINES["parker"]["alt"] = PARKER_LANE[1]


def lane_y(name):
    """the distance from the camera the machine flies past at"""
    if PARKER and name == "parker": return PARKER_LANE[0]
    return RACE_LANES[name][0] if RACE and name in RACE_LANES else FLY_Y


# the slow-motion replay: how much slower than real time while it crosses (a camera running 25 / slow frames a second)
REPLAY = {"spitfire": 1 / 16, "x1": 1 / 32, "f35": 1 / 48, "x15": 1 / 200, "parker": 1 / 20000}
REPLAY_SUF = {}                   # a replay's renders are in fly_<name>_replay<suffix>
if os.environ.get("REPLAY48") == "1":            # the cut with every replay at the same 48 times slower
    REPLAY_SUF = {k: "_r48" for k, v in REPLAY.items() if abs(v - 1 / 48) > 1e-12}
    REPLAY = {k: 1 / 48 for k in REPLAY}
RAMP, TAIL = 50, 15               # frames ramping back to real speed, then frames at real speed


def replay(name):
    """[(t, slow)]: the replay's frames, slowed while the machine is in the picture, then ramping back to real speed
    (each frame's step growing by the same factor) and on at real speed for a moment"""
    slow = REPLAY[name]
    step = slow / FPS
    t = -60.0 / MACHINES[name]["speed"]                        # find when it enters
    while not in_view(name, t, 0.0): t += step
    t0 = t - 8 * step
    while in_view(name, t, 0.0): t += step
    t1 = t + 6 * step
    out, t = [], t0
    while t < t1:
        out.append((t, slow)); t += step
    g = (1 / slow) ** (1 / RAMP)
    for i in range(RAMP):
        step *= g; t += step; out.append((t, step * FPS))
    for i in range(TAIL):
        t += 1 / FPS; out.append((t, 1.0))
    return out


def pose(name, t):
    """centre (x, y, z) and attitude (roll, pitch, yaw) in radians; t = 0 when it crosses x = 0, in front of the camera"""
    m = MACHINES[name]
    x = m["speed"] * t
    z, roll, pitch = m["alt"], 0.0, 0.0
    if name == "wright":
        # the Flyer was unstable in pitch: it flew in gentle undulations, nose leading each rise, and rocked a little
        w = 2 * math.pi / 2.4
        z += 0.22 * math.sin(w * t + 0.7)
        pitch = math.radians(2.6) * math.cos(w * t + 0.7)
        roll = math.radians(1.3) * math.sin(2 * math.pi * t / 3.3 + 1.9)
    elif name == "spitfire":
        roll = math.radians(1.0)
    return (x, lane_y(name), z), (roll, pitch, 0.0)


def frames(name):
    """[(frame, t)] for the shot; t = 0 at the crossing"""
    m = MACHINES[name]
    n0 = round(m["pre"] * FPS)
    return [(n, (n - n0) / FPS) for n in range(n0 + round(m["post"] * FPS) + 1)]


def _corners(c, box):
    L, S, Hh = box
    return np.array([[c[0] + a * L / 2, c[1] + b * S / 2, c[2] + d * Hh / 2] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])


def screen_rect(name, t, expo=0.01, margin=40):
    """the picture's pixels (x0, y0, x1, y1) that the machine, or its shadow, touches during the exposure; None if none"""
    m = MACHINES[name]
    to_sun = np.array(SS.direction())
    rect = None
    for dt in (-expo / 2, 0.0, expo / 2):
        c, _ = pose(name, t + dt)
        P = _corners(c, m["box"])
        shadow = P - np.outer(P[:, 2] / to_sun[2], to_sun)
        for Q in (P, shadow):
            uv, z = Cam.project(Q)
            if (z <= 0.1).any():                  # beside or behind the camera: far off to that side of the picture
                xc = (Q - np.array([Cam.CAM_X, 0.0, Cam.H])) @ Cam.R()[0]
                uv[z <= 0.1] = np.stack([Cam.CX + np.sign(xc[z <= 0.1]) * 1e6, np.full((z <= 0.1).sum(), Cam.CY)], 1)
            r = (uv[:, 0].min() - margin, uv[:, 1].min() - margin, uv[:, 0].max() + margin, uv[:, 1].max() + margin)
            if r[2] < 0 or r[0] > Cam.W or r[3] < 0 or r[1] > Cam.H_IMG: continue
            rect = r if rect is None else (min(rect[0], r[0]), min(rect[1], r[1]), max(rect[2], r[2]), max(rect[3], r[3]))
    if rect is None: return None
    return (max(0, int(rect[0])), max(0, int(rect[1])), min(Cam.W, int(rect[2]) + 1), min(Cam.H_IMG, int(rect[3]) + 1))


def in_view(name, t, expo=0.01):
    return screen_rect(name, t, expo) is not None
