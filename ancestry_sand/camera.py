"""The moving camera: one path, shared by Blender (glass plates) and the compositor (film.py).

A pose is (tx, ty, tilt, dist, heading): the camera looks at map pixel (tx, ty) on the table from `dist`
Blender units away, tilted `tilt` degrees from straight down, turned `heading` degrees about the
vertical (positive = looking more toward the east). The lens is fixed (the wide shot's fit), so the
camera dollies in rather than zooming, which keeps real perspective on the towers.

World units: 1 Blender unit = 100 map px; map (x, y, z px) -> (x/100 - 9.6, 5.4 - y/100, z/100).
"""
import math, os
import numpy as np

W, H = 1920, 1080
FPS = 30
LENS = 45.654884338378906           # the wide shot's lens (fit to the map in blender_scene.py)
SENSOR = 36.0
FOCAL_PX = LENS / SENSOR * W

WIDE = (960.0, 505.0, 22.0, 26.0, 0.0)

# (video second, pose). Between two keys the camera eases in and out; equal neighbouring poses hold.
# STORY follows the story around the world; FINALE (the default) holds the wide shot until today,
# then flies low among the population towers of Asia, darts across the Pacific for a low tour of the
# Americas, Europe, Africa and India, and rises to reveal the whole world.
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

KEYS_FINALE = [(0.0, WIDE), (999.0, WIDE)]          # the wide shot, between the opening and the finale

# ---- the opening: diorama close-ups of the first people, then out to the whole map by 0:45
OPENING = [
    (0.0, (1005.0, 668.0, 64.0, 3.8, -12.0)),        # low over the first people of southern Africa, under the title
    (6.5, (1012.0, 575.0, 63.0, 4.2, -2.0)),          # north over Central Africa
    (12.5, (1060.0, 485.0, 60.0, 5.0, 10.0)),         # East Africa
    (18.5, (1035.0, 365.0, 52.0, 9.0, 6.0)),          # rising, looking north: Neanderthal Europe beyond the Levant
    (24.5, (1090.0, 420.0, 55.0, 7.0, 18.0)),         # the early ventures out of Africa
    (30.5, (1080.0, 462.0, 60.0, 5.0, 22.0)),         # back down to East Africa for the main exit
    (37.5, (1235.0, 470.0, 52.0, 9.5, 30.0)),         # following them along the coast toward South Asia
    (45.0, WIDE),                                      # and out to the whole map
]

# ---- the finale: down from the wide shot into the towers, then a low flight through the world's great
# cities, looking ahead, lifted over any tower in its way: Asia, a whip across the Pacific (the camera wraps
# from the map's right edge to its left between two frames), the Americas, the Atlantic, Europe, and Africa,
# where the story began; then up to reveal the whole world. x is unwrapped (+ WRAP after the Pacific).
TOUR_START = 163.5
WRAP = 1680.0                      # map width at ~38 N: from the Bering Strait on the right to the left edge
CITIES = [
    ("Mumbai", 72.9, 19.1), ("Delhi", 77.2, 28.6), ("Dhaka", 90.4, 23.8), ("Bangkok", 100.5, 13.8),
    ("Singapore", 103.8, 1.35), ("Jakarta", 106.8, -6.2), ("Manila", 121.0, 14.6), ("Hong Kong", 114.2, 22.3),
    ("Shanghai", 121.5, 31.2), ("Beijing", 116.4, 39.9), ("Seoul", 127.0, 37.6), ("Tokyo", 139.7, 35.7),
    ("Los Angeles", -118.2, 34.1), ("Mexico City", -99.1, 19.4), ("New York", -74.0, 40.7), ("Lima", -77.0, -12.0),
    ("S\u00e3o Paulo", -46.6, -23.5), ("Buenos Aires", -58.4, -34.6), ("London", -0.1, 51.5), ("Paris", 2.35, 48.9),
    ("Istanbul", 29.0, 41.0), ("Moscow", 37.6, 55.8), ("Cairo", 31.2, 30.0), ("Lagos", 3.4, 6.5),
    ("Kinshasa", 15.3, -4.3),
]
FLY_TILT = 64.0                    # degrees from straight down: looking ahead, 26 degrees below the horizon
_HDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tour_heights.npy")


def _heights():
    """Sand height per 3-px cell once the sand has settled today (grains), saved by export_blender.py."""
    global _HC
    try:
        return _HC
    except NameError:
        _HC = np.load(_HDATA).reshape(360, 640) if os.path.exists(_HDATA) else None
        return _HC


def _tower_top(x, y, r=4):
    """Tallest sand near map px (x, y), Blender units (0 off the map or without heights)."""
    H = _heights()
    if H is None:
        return 0.0
    ci, cj = int(x // 3), int(y // 3)
    win = H[max(cj - r, 0):max(cj + r + 1, 0), max(ci - r, 0):max(ci + r + 1, 0)]
    return float(win.max()) * 0.0055 if win.size else 0.0


# The flight (v3.3). The first cut toured 25 cities in 45 s and rose to the whole world in 4.6 s: it felt
# rushed, and the viewer never got their bearings. Now: the same descent over the Arabian Sea and the same
# low flight past Mumbai (every frame before 2:48 is unchanged), then Asia's great cities at about half the
# speed, a whip across the Pacific, three cities of the Americas, and a long, slow rise from New York to the
# whole world, arriving on the score's final chord and held to the end.
TOUR_ROUTE = ["Mumbai", "Delhi", "Dhaka", "Bangkok", "Hong Kong", "Shanghai", "Tokyo",
              "Los Angeles", "Mexico City", "New York"]          # Delhi second keeps Mumbai's heading
FIRST_CITY = 171.0              # Mumbai, as before
LAST_CITY = 202.5               # New York
REVEAL = 215.5                  # the whole map (the wide shot), on the final chord; held to the end
WHIP = 2.4                      # the Pacific crossing: a fast whip over empty sea


def _build_tour():
    from proj import to_px
    keys = [(TOUR_START, WIDE), (168.2, (1225.0, 470.0, 60.0, 4.2, 15.0))]   # down over the Arabian Sea
    where = {name: (lo, la) for name, lo, la in CITIES}
    pts, wrapped = [], False
    for name in TOUR_ROUTE:
        x, y = (float(v) for v in to_px(*where[name]))
        if pts and not wrapped and x < pts[-1][1] - 600:
            wrapped = True                                     # across the Pacific
        pts.append((name, x + (WRAP if wrapped else 0.0), y))
    legs = []                                                  # (is the whip, length in units)
    for a, b in zip(pts, pts[1:]):
        legs.append(((b[1] > WRAP) != (a[1] > WRAP), math.hypot(b[1] - a[1], b[2] - a[2]) / 100.0))
    w = [d + 0.35 for whip, d in legs if not whip]             # time ~ distance, plus a moment at each city
    span = LAST_CITY - FIRST_CITY - WHIP * sum(whip for whip, _ in legs)
    times, k = [FIRST_CITY], 0
    for whip, d in legs:
        if whip:
            times.append(times[-1] + WHIP)
        else:
            times.append(times[-1] + span * w[k] / sum(w)); k += 1
    heads = []
    for i in range(len(pts)):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
        hd = math.degrees(math.atan2(b[1] - a[1], -(b[2] - a[2])))
        if heads:
            while hd - heads[-1] > 180:
                hd -= 360
            while hd - heads[-1] < -180:
                hd += 360
        heads.append(hd)
    for (name, x, y), t, hd in zip(pts, times, heads):
        top = _tower_top(x % WRAP if x > 1900 else x, y)
        h = min(max(0.8 * top + 0.45, 0.65), 1.25)             # fly low: just below the tallest towers
        keys.append((t, (x, y, FLY_TILT, h / math.cos(math.radians(FLY_TILT)), hd)))
    # the rise: up and back from New York, over the Atlantic, to the whole map
    end_head = 360.0 * round(heads[-1] / 360.0)
    nx, ny = pts[-1][1], pts[-1][2]
    wx, wy = WIDE[0] + WRAP, WIDE[1]
    keys.append((LAST_CITY + 0.45 * (REVEAL - LAST_CITY),
                 (nx + 0.4 * (wx - nx), ny + 0.4 * (wy - ny), 48.0, 6.0, 0.5 * (heads[-1] + end_head))))
    keys.append((REVEAL, (wx, wy, WIDE[2], WIDE[3], end_head)))
    cities = [(name, x, y, t) for (name, x, y), t in zip(pts, times)]
    return keys, cities, REVEAL


KEYS = KEYS_STORY if os.environ.get("SAND_CAMERA", "finale") == "story" else KEYS_FINALE


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


TOUR, TOUR_CITIES, TOUR_END = _build_tour()


def _spline(K, s):
    """Cubic Hermite through keys K [(t, pose)] (Catmull-Rom tangents, at rest at both ends); distance in log."""
    T = [k[0] for k in K]
    P = [list(k[1]) for k in K]
    for p in P:
        p[3] = math.log(p[3])
    i = max(0, min(len(T) - 2, sum(1 for t in T if t <= s) - 1))
    m = []
    for j in (i, i + 1):
        if j == 0 or j == len(T) - 1:
            m.append([0.0] * 5)
        else:
            m.append([(b - a) / (T[j + 1] - T[j - 1]) for a, b in zip(P[j - 1], P[j + 1])])
    h = T[i + 1] - T[i]
    u = min(max((s - T[i]) / h, 0.0), 1.0)
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    out = [h00 * a + h10 * h * ma + h01 * b + h11 * h * mb for a, b, ma, mb in zip(P[i], P[i + 1], m[0], m[1])]
    out[3] = math.exp(out[3])
    return out


def _find_cut():
    """The wrap: when the view centre passes the map's right edge beyond Japan (x 1840), on a half frame."""
    i = next(j for j, (_, x, _, _) in enumerate(TOUR_CITIES) if x > WRAP)
    t0, t1 = TOUR_CITIES[i - 1][3], TOUR_CITIES[i][3]
    s = t0
    while s < t1 and _spline(TOUR, s)[0] < 1840.0:
        s += 0.002
    return (math.floor(s * FPS) + 0.5) / FPS


CUT = _find_cut()


def _clearance():
    """Extra camera height (as extra distance along the view) so it never passes through a tower: per
    frame, the lens and the first 0.4 units of its view stay clear of the sand, then the lift is eased."""
    n0, n1 = int(TOUR_START * FPS), int(math.ceil(TOUR_END * FPS)) + 2
    lift = np.zeros(n1 - n0 + 1)
    for f in range(n0, n1 + 1):
        p = _spline(TOUR, f / FPS)
        if f / FPS >= CUT:
            p[0] -= WRAP
        extra = 0.0
        for _ in range(3):
            q = list(p)
            q[3] = p[3] + extra / math.cos(math.radians(p[2]))
            loc, r, u, fw = basis(tuple(q))
            need = 0.0
            for d in (0.0, 0.1, 0.2, 0.3, 0.4):
                pt = loc + fw * d
                px, py = (pt[0] + 9.6) * 100.0, (5.4 - pt[1]) * 100.0
                need = max(need, _tower_top(px, py, r=2) + 0.08 - pt[2])
            if need <= 0:
                break
            extra += need
        lift[f - n0] = extra
    w = 12                                                     # ease the lifts in and out (+-0.4 s)
    pad = np.pad(lift, w, mode="edge")
    mx = np.array([pad[i:i + 2 * w + 1].max() for i in range(len(lift))])
    k = np.hanning(2 * w + 3)[1:-1]
    sm = np.convolve(np.pad(mx, w, mode="edge"), k / k.sum(), mode="valid")
    return n0, np.maximum(sm, mx * 0.0)


def _tour(s):
    out = _spline(TOUR, s)
    if s >= CUT:
        out[0] -= WRAP
    global _CL
    try:
        _CL
    except NameError:
        _CL = None
        _CL = _clearance()
    if _CL is not None:
        n0, lift = _CL
        x = s * FPS - n0
        e = float(np.interp(x, np.arange(len(lift)), lift))
        out[3] += e / math.cos(math.radians(out[2]))
    return tuple(out)


def pose_at(s):
    if KEYS is KEYS_FINALE:
        if s <= OPENING[-1][0]:
            return tuple(_spline(OPENING, s))
        if s >= TOUR_START:
            return _tour(s)
    for (s0, p0), (s1, p1) in zip(KEYS, KEYS[1:]):
        if s0 <= s < s1:
            e = ease((s - s0) / (s1 - s0))
            out = [a + (b - a) * e for a, b in zip(p0, p1)]
            out[3] = math.exp(math.log(p0[3]) + (math.log(p1[3]) - math.log(p0[3])) * e)   # dolly in log
            return tuple(out)
    return KEYS[-1][1]


def fstop_at(s):
    """Aperture: wide open (f/0.9, the miniature look) for most shots, stopped down when the camera flies
    low so the towers around it stay sharp."""
    return 0.9 * min(max(7.0 / pose_at(s)[3], 1.0), 4.0)


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


# ------------------------------------------------------------------ glass plate keyframes
# Blender renders the glass only at keyframes; every other frame is reprojected from the two nearest
# keyframes through the plane halfway up the walls. A new keyframe is placed whenever that
# reprojection would be off by more than MAX_ERR px anywhere on screen (the walls' parallax).
MAX_ERR = 1.0
WALL_H = 14.0
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
