"""The moving camera: one path, shared by Blender (glass plates) and the compositor (film.py).

A pose is (tx, ty, tilt, dist, heading): the camera looks at map pixel (tx, ty) on the table from `dist`
Blender units away, tilted `tilt` degrees from straight down, turned `heading` degrees about the
vertical (positive = looking more toward the east). The lens is fixed (the wide shot's fit), so the
camera dollies in rather than zooming, which keeps real perspective on the towers.

World units: 1 Blender unit = 100 map px; map (x, y, z px) -> (x/100 - 9.6, 5.4 - y/100, z/100).
"""
import math
import numpy as np

W, H = 1920, 1080
LENS = 45.654884338378906           # the wide shot's lens (fit to the map in blender_scene.py)
SENSOR = 36.0
FOCAL_PX = LENS / SENSOR * W

WIDE = (960.0, 505.0, 22.0, 26.0, 0.0)

# (video second, pose). Between two keys the camera eases in and out; equal neighbouring poses hold.
# STORY follows the story around the world; FINALE (the default) holds the wide shot until today,
# then flies low among the population towers of Asia for the ending.
KEYS_STORY = [
    (0.0, WIDE),
    (17.0, WIDE),
    (25.0, (1075.0, 455.0, 36.0, 14.0, 0.0)),        # East Africa: early ventures, then the exit
    (31.0, (1085.0, 445.0, 38.0, 14.0, 2.0)),
    (44.0, (1420.0, 520.0, 40.0, 15.5, 6.0)),         # along the southern route to Asia and Sahul
    (54.0, WIDE),
    (65.0, WIDE),
    (72.0, (520.0, 330.0, 36.0, 15.5, -8.0)),         # Beringia and the Americas
    (79.0, (545.0, 420.0, 38.0, 15.5, -4.0)),
    (87.0, (1060.0, 330.0, 42.0, 13.0, 4.0)),         # the first farmers, Fertile Crescent
    (99.0, (975.0, 300.0, 44.0, 13.0, -2.0)),         # farmers into Europe, then the steppe
    (108.0, (1035.0, 545.0, 42.0, 15.0, 0.0)),        # the Bantu expansion
    (117.0, WIDE),
    (129.0, WIDE),
    (136.0, (700.0, 400.0, 44.0, 14.0, -4.0)),        # 1500 CE: the Atlantic
    (141.0, (720.0, 420.0, 46.0, 14.0, 2.0)),
    (148.0, (1340.0, 420.0, 57.0, 8.6, 10.0)),        # the population explosion: towers of Asia
    (153.0, (1240.0, 395.0, 58.0, 8.6, -4.0)),
    (157.5, (1010.0, 360.0, 55.0, 9.5, -12.0)),       # across to Europe and Africa
    (163.0, WIDE),
    (999.0, WIDE),
]

KEYS_FINALE = [
    (0.0, WIDE),
    (163.5, WIDE),                                   # still for 200,000 years, until today's arrivals land
    (170.0, (1290.0, 425.0, 58.0, 7.5, 6.0)),         # then down among the towers of South Asia
    (175.0, (1420.0, 440.0, 60.0, 7.0, 22.0)),        # and across to Southeast and East Asia
    (178.5, (1460.0, 420.0, 62.0, 6.0, 28.0)),
    (999.0, (1460.0, 420.0, 62.0, 6.0, 28.0)),
]

import os as _os
KEYS = KEYS_STORY if _os.environ.get("SAND_CAMERA", "finale") == "story" else KEYS_FINALE


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


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
    return P / P[2, 3]


def hom(P, z=0.0):
    """Homography from map px on the plane at height z to screen px."""
    Hh = np.stack([P[:, 0], P[:, 1], P[:, 2] * z + P[:, 3]], 1)
    return Hh / Hh[2, 2]


# ------------------------------------------------------------------ glass plate keyframes
# Blender renders the glass only at keyframes; every other frame is reprojected from the two nearest
# keyframes through the plane halfway up the walls. A new keyframe is placed whenever that
# reprojection would be off by more than MAX_ERR px anywhere on screen (the walls' parallax).
MAX_ERR = 1.0
WALL_H = 14.0
FPS = 30
_xs, _ys = np.meshgrid(np.linspace(0, 1920, 25), np.linspace(60, 900, 15))
_G0 = np.stack([_xs.ravel(), _ys.ravel(), np.zeros(_xs.size), np.ones(_xs.size)])
_G1 = _G0.copy()
_G1[2] = WALL_H


def _wall_vectors(P):
    a, b = P @ _G0, P @ _G1
    a, b = a[:2] / a[2], b[:2] / b[2]
    vis = (a[0] > -200) & (a[0] < W + 200) & (a[1] > -200) & (a[1] < H + 200)
    return b - a, vis


def plate_keys(end=999.0):
    keys = [0.0]
    ref = _wall_vectors(P_of(pose_at(0.0)))
    f = 0
    last = KEYS[-2][0]                     # the camera is still after the last move
    while f / FPS < last:
        f += 1
        cur = _wall_vectors(P_of(pose_at(f / FPS)))
        vis = ref[1] & cur[1]
        err = 0.5 * np.abs(cur[0] - ref[0])[:, vis].max() if vis.any() else 1e9
        if err > MAX_ERR:
            keys.append((f - 1) / FPS)
            ref = _wall_vectors(P_of(pose_at(keys[-1])))
    keys.append(last)
    return [round(k, 5) for k in keys]


def plate_index():
    """([(time, pose_id)], poses): identical poses share one render."""
    poses, out = [], []
    for t in plate_keys():
        p = tuple(round(v, 6) for v in pose_at(t))
        if p not in poses:
            poses.append(p)
        out.append((t, poses.index(p)))
    return out, poses


if __name__ == "__main__":
    import json, os, sys
    keys, poses = plate_index()
    print(len(keys), "plate keyframes,", len(poses), "distinct poses")
    if "--write" in sys.argv:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        json.dump(dict(keys=keys, poses=poses), open(os.path.join(d, "plate_poses.json"), "w"))
