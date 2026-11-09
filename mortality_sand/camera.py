"""The camera, shared by Blender (glass plates) and the compositor (film.py).

A pose is (tx, ty, tilt, dist, heading): the camera looks at map pixel (tx, ty) on the table from `dist`
Blender units away, tilted `tilt` degrees from straight down, turned `heading` degrees about the
vertical. The lens is fixed (the wide shot's fit, as in the migration film).

This first cut holds the wide shot for the whole film: the story is the sand, and the lifted grains rise
toward the camera and out of the frame. (Close-ups and a closing flight can be added as KEYS later, with
glass plates rendered at keyframes as in ancestry_sand/camera.py.)

World units: 1 Blender unit = 100 map px; map (x, y, z px) -> (x/100 - 9.6, 5.4 - y/100, z/100).
"""
import math
import numpy as np

W, H = 1920, 1080
FPS = 30
LENS = 45.654884338378906           # the wide shot's lens (fit to the map in blender_scene.py)
SENSOR = 36.0
FOCAL_PX = LENS / SENSOR * W
WALL_H = 14.0                       # glass wall height, map px

WIDE = (960.0, 505.0, 22.0, 26.0, 0.0)
KEYS = [(0.0, WIDE), (9999.0, WIDE)]


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def pose_at(s):
    for (s0, p0), (s1, p1) in zip(KEYS, KEYS[1:]):
        if s0 <= s < s1:
            e = ease((s - s0) / (s1 - s0))
            out = [a + (b - a) * e for a, b in zip(p0, p1)]
            out[3] = math.exp(math.log(p0[3]) + (math.log(p1[3]) - math.log(p0[3])) * e)   # dolly in log
            return tuple(out)
    return KEYS[-1][1]


def basis(pose):
    """Camera location and (right, up, forward) axes in world units."""
    tx, ty, tilt, dist, head = pose
    tgt = np.array([tx / 100.0 - 9.6, 5.4 - ty / 100.0, 0.0])
    t, h = math.radians(tilt), math.radians(head)
    back = np.array([-math.sin(h) * math.sin(t), -math.cos(h) * math.sin(t), math.cos(t)])
    loc = tgt + back * dist
    f = -back
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return loc, r, u, f


def P_of(pose):
    """3x4 matrix: (map x, map y, height, 1) in px -> homogeneous screen px."""
    loc, r, u, f = basis(pose)
    M = np.array([[0.01, 0, 0, -9.6], [0, -0.01, 0, 5.4], [0, 0, 0.01, 0]])     # map px -> world
    R = np.stack([r, -u, f])                                                     # world -> camera (x right, y down, z fwd)
    E = np.hstack([R, (-R @ loc)[:, None]]) @ np.vstack([M, [0, 0, 0, 1]])
    Kk = np.array([[FOCAL_PX, 0, W / 2], [0, FOCAL_PX, H / 2], [0, 0, 1]])
    P = Kk @ E
    return P / abs(P[2, 3])          # (never flip the sign: w > 0 stays 'in front of the camera')


def hom(P, z=0.0):
    """Homography from map px on the plane at height z to screen px."""
    Hh = np.stack([P[:, 0], P[:, 1], P[:, 2] * z + P[:, 3]], 1)
    return Hh / Hh[2, 2]
