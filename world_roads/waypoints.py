"""Macro traversal order: a hand-authored sweep through the world's road networks.

Every road edge gets a scalar `key` = distance along this waypoint path of the nearest densified
waypoint-path point (3-D unit-vector nearest neighbour, so no antimeridian artefacts). The router
only lets the dot pick up roads whose key lies within a moving band, which yields the intended
continental progression while the local walk stays free to turn, loop and backtrack.
Sequences are separate lists so no path is densified across an ocean.
"""
import numpy as np
from scipy.spatial import cKDTree

SEQUENCES = [
    # North America: start in Alaska / western NA, sweep east
    [(-150, 64), (-135, 60), (-123, 49), (-122, 38), (-118, 34), (-112, 33), (-112, 41), (-105, 40),
     (-106, 32), (-97, 30), (-97, 38), (-97, 47), (-90, 45), (-88, 41), (-90, 35), (-84, 34), (-82, 28),
     (-80, 35), (-77, 39), (-74, 41), (-71, 43), (-79, 44), (-73, 46), (-64, 46), (-56, 49)],
    # Caribbean, Mexico and Central America
    [(-66, 18), (-72, 19), (-78, 21), (-110, 29), (-104, 22), (-99, 19), (-92, 17), (-88, 14), (-84, 10), (-80, 8.5)],
    # South America, ending in the south-east / Patagonia
    [(-75, 8), (-67, 10), (-60, 4), (-60, -3), (-48, -1), (-38, -6), (-35, -9), (-39, -14), (-43, -20),
     (-47, -16), (-56, -15), (-63, -17), (-70, -16), (-77, -12), (-79, -2), (-70, -33), (-65, -26),
     (-57, -25), (-50, -25), (-46, -23), (-56, -34), (-58, -34.6), (-62, -39), (-68, -46), (-69, -53)],
    # Europe
    [(-17, 65), (-9, 39), (-3, 40), (2, 47), (-2, 53), (-8, 53), (5, 52), (10, 51), (10, 60), (25, 62),
     (20, 52), (15, 48), (12, 43), (20, 44), (24, 38), (30, 48), (38, 55)],
    # Turkey, Levant, Africa (anticlockwise), Arabia
    [(33, 39), (36, 33), (31, 30), (10, 34), (-6, 33), (-15, 14), (3, 8), (20, 0), (25, -15), (28, -26),
     (20, -33), (32, -25), (37, -5), (38, 8), (32, 15), (45, 22), (52, 30)],
    # Asia
    [(62, 40), (60, 56), (85, 55), (105, 52), (133, 45), (127, 37), (138, 36), (125, 44), (116, 40),
     (105, 47), (85, 43), (90, 31), (70, 30), (77, 28), (78, 12), (81, 7), (90, 23), (104, 30),
     (120, 31), (113, 23), (121, 24), (121, 14), (106, 16), (100, 14), (102, 3), (100, 0), (110, -7),
     (115, 0), (120, -3)],
    # Oceania, finishing in New Zealand
    [(145, -6), (131, -13), (153, -27), (151, -34), (145, -38), (138, -35), (116, -32), (147, -42),
     (178, -18), (175, -37), (170, -45)],
]
SEQ_GAP_KM = 2000.0
STEP_KM = 25.0
R = 6371.0


def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def densified():
    pts, keys, k0 = [], [], 0.0
    for seq in SEQUENCES:
        v = _xyz(*np.array(seq, float).T)
        for a, b in zip(v[:-1], v[1:]):
            ang = np.arccos(np.clip(a @ b, -1, 1))
            n = max(2, int(ang * R / STEP_KM))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            s = np.sin(ang)
            p = (np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / s if s > 1e-9 else np.repeat(a[None], n, 0)
            pts.append(p); keys.append(k0 + t[:, 0] * ang * R)
            k0 += ang * R
        pts.append(v[-1:]); keys.append(np.array([k0]))
        k0 += SEQ_GAP_KM
    return np.concatenate(pts), np.concatenate(keys)


def sweep_key(lon, lat):
    pts, keys = densified()
    _, i = cKDTree(pts).query(_xyz(lon, lat))
    return keys[i]
