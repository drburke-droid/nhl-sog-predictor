"""Storyboard stills for "A robot vacuum at the speed of light cleans every floor on Earth" (1920x1080).

    python frames.py [names...]      -> out/frames/NN_name.png (all frames when no names are given)

Every number on screen comes from data/nyc/stats.json, data/nyc/route_stats.json, data/ghsl/totals.json or
the constants below (sources in ACCURACY.md).
"""
import os, sys, json, math, functools
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
sys.path.insert(0, os.path.join(ROOT, "roomba"))
import style as ST                                                    # noqa: E402
import xray as X                                                      # noqa: E402

OUT = os.path.join(HERE, "out", "frames")
W, H, SS = 1920, 1080, 2
INK, INK2, MUTED, GOLD, RED = ST.INK, ST.INK2, ST.MUTED, ST.GOLD, ST.RED
CYAN = (110, 200, 255)
C_MS = 299792458.0
LANE = 0.25                                  # m between rows
WORLD_FLOOR_M2 = 273e9                       # GlobalABC / UNEP Global Status Report 2025-26: 2024
WORLD_BUILDINGS = 2.53e9                     # Overture Maps, 2026-08 release notes
WORLD_TRAVEL_S = 410.0                       # estimate (ACCURACY.md): hops between buildings + stairs
NYC = json.load(open(os.path.join(HERE, "data", "nyc", "route_stats.json")))
NYS = json.load(open(os.path.join(HERE, "data", "nyc", "stats.json")))


# ------------------------------------------------------------------ text helpers
def text(d, xy, s, size, col, a=1.0, weight="SemiBold", anchor="la"):
    if a > 0:
        d.text(xy, s, font=ST.font(size, weight), fill=tuple(col) + (int(255 * a),), anchor=anchor)


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def caption(fr, lines, y=None, anchor="bottom", cx=None):
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    ST.caption(d, lines, 1.0, y=y, anchor=anchor, cx=cx)
    return np.asarray(im)


def hud(fr, rows, x0=60, y0=56, title=None):
    """rows: [(label, value, colour, size)]"""
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    y = y0
    if title:
        text(d, (x0, y), title, 26, INK2); y += 44
    for lab, val, col, sz in rows:
        text(d, (x0, y), lab, 22, MUTED); y += 28
        text(d, (x0, y), val, sz, col, 1.0, "Bold"); y += int(sz * 1.25) + 18
    return np.asarray(im)


def tag(fr, s, x=W - 60, y=56, col=INK2, size=26, anchor="ra"):
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    text(d, (x, y), s, size, col, 1.0, "SemiBold", anchor)
    return np.asarray(im)


def save(fr, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".png")
    Image.fromarray(fr).save(p)
    print("wrote", p, flush=True)


def fmt_s(s):
    if s < 1e-3:
        return f"{s * 1e6:,.0f} µs"
    if s < 1:
        return f"{s * 1e3:,.1f} ms"
    if s < 60:
        return f"{s:,.2f} s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)} h {int(m):02d} min {int(sec):02d} s" if h else f"{int(m)} min {int(sec):02d} s"


# ------------------------------------------------------------------ the city
@functools.lru_cache(1)
def city():
    return X.City()


@functools.lru_cache(1)
def route():
    r = np.load(os.path.join(HERE, "data", "nyc", "route.npz"))
    return r["order"], r["t_start"], r["t_end"]


def levels_at(t):
    """per building: height (m) below which it is clean at light time t"""
    C = city()
    _, ts, te = route()
    lv = np.where(te <= t, 1e9, -1.0)
    act = np.nonzero((ts <= t) & (te > t))[0]
    for b in act:
        f = (t - ts[b]) / (te[b] - ts[b])
        lv[b] = f * C.top[b]
    return lv


def esb_prisms():
    C = city()
    return np.nonzero(C.pb == C.esb)[0]


def floor_slabs(C, b):
    """building b's floors bottom to top: (prism, z) pairs"""
    out = []
    for k in np.nonzero(C.pb == b)[0]:
        for j in range(C.pnfl[k]):
            out.append((k, C.pz0[k] + j * (C.pz1[k] - C.pz0[k]) / C.pnfl[k]))
    out.sort(key=lambda q: q[1])
    return out


def rows_in(poly_xy, lane=LANE):
    """boustrophedon rows across a polygon along its long axis: list of (p0, p1) world segments in order"""
    import shapely
    P = shapely.Polygon(poly_xy).buffer(-0.2)
    rect = np.asarray(P.minimum_rotated_rectangle.exterior.coords)
    e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
    ax = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    ax = ax / np.linalg.norm(ax)
    nx = np.array([-ax[1], ax[0]])
    pts = np.asarray(P.exterior.coords)
    s = pts @ nx
    segs = []
    for i, v in enumerate(np.arange(s.min() + lane / 2, s.max(), lane)):
        line = shapely.LineString([v * nx + ax * -1e4, v * nx + ax * 1e4])
        g = P.intersection(line)
        parts = [g] if g.geom_type == "LineString" else list(getattr(g, "geoms", []))
        for q in parts:
            if q.is_empty:
                continue
            c = np.asarray(q.coords)
            if i % 2:
                c = c[::-1]
            segs.append((c[0], c[-1]))
    return segs


def robot_sprite(px, heading):
    import roomba3d
    return roomba3d.sprite(px, heading)


def paste_rgba(fr, sp, x, y):
    im = Image.fromarray(fr).convert("RGBA")
    im.alpha_composite(sp, (int(round(x - sp.width / 2)), int(round(y - sp.height * 0.55))))
    return np.asarray(im.convert("RGB"))


# ================================================================== 01: one floor, X-rayed
def active_floor(C, prism, zf, f, s, cam, img, dust_n=60000, wake=True, rows_gain=0.32):
    """draw one floor mid-clean on img (supersampled float RGB): returns robot (x, y, heading_deg) in output px"""
    Hs, Ws = img.shape[:2]
    ring = C.rxy[C.ps[prism]:C.pe[prism]][:-1]
    segs = rows_in(ring)
    lens = np.array([np.linalg.norm(q - p) for p, q in segs])
    cum = np.concatenate([[0], np.cumsum(lens)])
    dist = f * cum[-1]
    k = int(np.clip(np.searchsorted(cum, dist) - 1, 0, len(segs) - 1))
    p0, p1 = segs[k]
    u = (dist - cum[k]) / max(lens[k], 1e-9)
    rx, ry = p0 + u * (p1 - p0)
    canvas = np.zeros((Hs, Ws), np.float32)
    cv2.fillPoly(canvas, X._polys(C, cam, np.array([prism]), np.array([zf])), 1.0, shift=4)
    done = [np.array([cam.project(a[0], a[1], zf), cam.project(c[0], c[1], zf)]).reshape(2, 2) for a, c in segs[:k]]
    done.append(np.array([cam.project(p0[0], p0[1], zf), cam.project(rx, ry, zf)]).reshape(2, 2))
    lw = max(2, int(LANE * s * SS * 1.02))
    rowmask = np.zeros((Hs, Ws), np.float32)
    stripe = np.zeros((Hs, Ws), np.float32)
    cv2.polylines(rowmask, [(q * 16).astype(np.int32) for q in done], False, 1.0, lw, cv2.LINE_AA, shift=4)
    if s * LANE * SS > 5:                                     # vacuum tracks: alternate rows a little lighter
        cv2.polylines(stripe, [(q * 16).astype(np.int32) for q in done[1::2]], False, 1.0, lw, cv2.LINE_AA, shift=4)
    rowmask = np.minimum(rowmask, 1) * canvas
    stripe = np.minimum(stripe, 1) * canvas
    dirtyf = canvas * (1 - rowmask)
    img += (0.16 * dirtyf)[..., None] * X.CYAN
    img += (rows_gain * (rowmask + 0.35 * stripe))[..., None] * X.GOLD
    if s * LANE * SS > 5:                                     # seams between rows, when rows are wide enough to see
        seams = np.zeros((Hs, Ws), np.float32)
        cv2.polylines(seams, [(q * 16).astype(np.int32) for q in done], False, 1.0, 1, cv2.LINE_AA, shift=4)
        img += (0.10 * seams * canvas)[..., None] * X.GOLD
    edge = np.zeros((Hs, Ws), np.uint8)
    cv2.polylines(edge, X._polys(C, cam, np.array([prism]), np.array([zf])), True, 255, 2 * SS, cv2.LINE_AA, shift=4)
    img += (edge.astype(np.float32) / 255 * 0.9)[..., None] * (0.6 * X.CYAN + 0.4)
    if dust_n:
        rng = np.random.default_rng(4)
        mn, mx = ring.min(0), ring.max(0)
        px = rng.uniform(mn[0], mx[0], dust_n); py = rng.uniform(mn[1], mx[1], dust_n)
        uu, vv = cam.project(px, py, np.full(dust_n, zf))
        ok = (uu >= 0) & (uu < Ws) & (vv >= 0) & (vv < Hs)
        uu, vv = uu[ok].astype(int), vv[ok].astype(int)
        keep = dirtyf[vv, uu] > 0.5
        dust = np.zeros((Hs, Ws), np.float32)
        dust[vv[keep], uu[keep]] = rng.uniform(0.5, 1.0, keep.sum())
        dust = cv2.GaussianBlur(dust, (0, 0), 1.6) * 6
        img += np.minimum(dust, 1.2)[..., None] * np.array([0.42, 0.40, 0.38], np.float32)
    if wake:
        w = np.zeros((Hs, Ws), np.float32)
        a = np.array(cam.project(p0[0], p0[1], zf)); c = np.array(cam.project(rx, ry, zf))
        L = np.linalg.norm(c - a)
        n = 24
        for i in range(n):                                    # brighter toward the robot
            q0 = a + (c - a) * max(0, 1 - (i + 1) / n * min(1, 900 * SS / max(L, 1)))
            q1 = a + (c - a) * max(0, 1 - i / n * min(1, 900 * SS / max(L, 1)))
            cv2.line(w, tuple((q0 * 16).astype(int)), tuple((q1 * 16).astype(int)), (1 - i / n) ** 1.5, lw, cv2.LINE_AA, shift=4)
        img += (w * 1.3)[..., None] * np.array([1.0, 0.92, 0.75], np.float32)
    ox_, oy_ = cam.project(rx, ry, zf)
    hx, hy = cam.project(rx + (p1 - p0)[0] * 0.01, ry + (p1 - p0)[1] * 0.01, zf)
    head = math.degrees(math.atan2(-(hy - oy_), hx - ox_))
    return ox_ / SS, oy_ / SS, head


def building_layers(C, cam, b, level, img, zf=None, above_gain=0.10, below_gain=0.30, others=True, other_gain=0.20,
                    radius=600, lines=True):
    """the hero building as X-ray glass (floors above faint, cleaned floors below gold), neighbours dimmer"""
    idx = X.visible(C, cam, margin=300)
    near = idx[np.hypot(C.pcx[idx] - C.cx[b], C.pcy[idx] - C.cy[b]) < radius]
    mine = near[C.pb[near] == b]
    other = near[C.pb[near] != b]
    d0, c0 = X.slabs(C, cam, level, mine)
    img += (1 - np.exp(-d0 * 0.25))[..., None] * X.CYAN * above_gain
    img += (1 - np.exp(-c0 * 0.30))[..., None] * X.GOLD * below_gain
    if lines:
        L = X.edges(C, cam, mine, level, verticals=True)
        img += (0.30 * L[0])[..., None] * X.CYAN + (0.45 * L[1])[..., None] * X.GOLD
    if others and len(other):
        d1, c1 = X.slabs(C, cam, level, other)
        img += (1 - np.exp(-d1 * 0.10))[..., None] * X.CYAN * other_gain
        img += (1 - np.exp(-c1 * 0.10))[..., None] * X.GOLD * other_gain
        L = X.edges(C, cam, other, level, floor_lines=False, verticals=False)
        img += (other_gain * 0.9 * (L[0] + L[1]))[..., None] * X.CYAN
    return img


def esb_tower():
    C = city()
    ks = esb_prisms()
    return ks[np.argmax([(C.pz1[k] - C.pz0[k]) * (C.pnfl[k] > 0) for k in ks])]


def robot_at(C, prism, f):
    ring = C.rxy[C.ps[prism]:C.pe[prism]][:-1]
    segs = rows_in(ring)
    lens = np.array([np.linalg.norm(q - p) for p, q in segs]); cum = np.cumsum(lens)
    k = int(np.searchsorted(cum, f * cum[-1]))
    p0, p1 = segs[k]
    u = (f * cum[-1] - (cum[k] - lens[k])) / lens[k]
    return p0 + u * (p1 - p0)


def magnifier(fr, inset, cx, cy, R, tx, ty, label=None, col=(235, 240, 250)):
    """paste a round inset (uint8 2R x 2R) at (cx, cy) with a leader line to (tx, ty)"""
    im = Image.fromarray(fr).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    ang = math.atan2(ty - cy, tx - cx)
    ex, ey = cx + R * math.cos(ang), cy + R * math.sin(ang)
    d.line([(ex, ey), (tx, ty)], fill=col + (200,), width=3)
    d.ellipse([tx - 9, ty - 9, tx + 9, ty + 9], outline=col + (230,), width=3)
    mask = Image.new("L", (2 * R, 2 * R), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 2 * R - 1, 2 * R - 1], fill=255)
    im.paste(Image.fromarray(inset), (int(cx - R), int(cy - R)), mask)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col + (235,), width=4)
    if label:
        text(d, (cx, cy + R + 18), label, 24, INK2, 1.0, "SemiBold", "ma")
    return np.asarray(im.convert("RGB"))


def f01():
    """The Empire State Building X-rayed, one floor mid-clean; a magnifier on the robot itself."""
    C = city()
    b = C.esb
    tower = esb_tower()
    j = 14
    zf = C.pz0[tower] + j * (C.pz1[tower] - C.pz0[tower]) / C.pnfl[tower]
    f = 0.46
    r = robot_at(C, tower, f)
    s = 2.05
    psi, el = 34, 26
    cam = X.Cam(psi, el, s, C.cx[b], C.cy[b], 215, ox=1180, oy=500)
    level = np.full(C.n, -1.0)
    level[b] = zf - 0.01
    img = np.zeros((H * SS, W * SS, 3), np.float32) + X.BG
    building_layers(C, cam, b, level, img, above_gain=0.55, below_gain=0.85, radius=900, other_gain=0.16)
    x, y, head = active_floor(C, tower, zf, f, s, cam, img, dust_n=0, wake=False, rows_gain=0.9)
    img += X.light(H * SS, W * SS, x * SS, y * SS, SS, size=0.5)
    img = X.bloom(img, SS)
    fr = to_u8(X.tonemap(X.down(img, SS)))
    # the inset: the same floor, 90x closer
    R = 230
    si = 185.0
    ci = X.Cam(psi, el, si, r[0], r[1], zf, ox=R, oy=R + 20, W=2 * R, H=2 * R, ss=SS)
    im2 = np.zeros((2 * R * SS, 2 * R * SS, 3), np.float32) + X.BG
    ix, iy, ih = active_floor(C, tower, zf, f, si, ci, im2, dust_n=400000, wake=True, rows_gain=0.30)
    im2 = X.bloom(im2, SS, k=(0.12, 0.06, 0.02))
    inset = to_u8(X.tonemap(X.down(im2, SS)))
    inset = paste_rgba(inset, robot_sprite(int(0.35 * si * 1.5), ih), ix, iy)
    fr = magnifier(fr, inset, 470, 470, R, x, y, "the robot: 35 cm across · slowed down 1 billion times")
    fr = tag(fr, "EMPIRE STATE BUILDING", col=INK2)
    fr = tag(fr, "X-ray view · gold floors are clean", y=96, col=MUTED, size=22)
    fr = caption(fr, [("This robot vacuum moves at the speed of light.", 46, INK, "SemiBold")])
    save(fr, "01_one_floor")


# ------------------------------------------------------------------ NYC progress helpers
@functools.lru_cache(4)
def nyc_progress(t):
    C = city()
    _, ts, te = route()
    done = te <= t
    act = (ts <= t) & (te > t)
    fa = C.floor_area
    area = fa[done].sum() + sum(fa[b] * (t - ts[b]) / (te[b] - ts[b]) for b in np.nonzero(act)[0])
    return int(done.sum()), float(area)


def t_radius(r):
    """light time when every building within r metres of the Empire State Building is clean"""
    C = city()
    _, ts, te = route()
    d = np.hypot(C.cx - C.cx[C.esb], C.cy - C.cy[C.esb])
    return float(te[d < r].max())


def nyc_hud(fr, t, title="NEW YORK CITY · EVERY FLOOR"):
    n, area = nyc_progress(t)
    return hud(fr, [("LIGHT TIME", fmt_s(t), GOLD, 64),
                    ("BUILDINGS CLEANED", f"{n:,} of {NYC['buildings']:,}", INK, 40),
                    ("FLOOR CLEANED", f"{area / 1e6:,.1f} million m²" if area > 1e6 else f"{area:,.0f} m²", INK, 40)],
               title=title)


def city_frame(cam, t, gains=(0.06, 0.10), lines_radius=0, sat=(0.9, 1.0), extra_fn=None, bloom_k=(0.22, 0.14, 0.08)):
    C = city()
    level = levels_at(t)
    idx = X.visible(C, cam)
    d, c = X.slabs(C, cam, level, idx)
    lines = None
    if lines_radius:
        near = idx[np.hypot(C.pcx[idx] - C.cx[C.esb], C.pcy[idx] - C.cy[C.esb]) < lines_radius]
        lines = X.edges(C, cam, near, level, floor_lines=False, verticals=False)
    extra = extra_fn(cam) if extra_fn else None
    img = X.compose(d, c, gain=gains, sat=sat, lines=lines, line_gain=(0.25, 0.35), extra=extra, ss=SS, bloom_k=bloom_k)
    return to_u8(img)


def frontier_light(t):
    """the light at the building being cleaned at time t (world position)"""
    C = city()
    _, ts, te = route()
    act = np.nonzero((ts <= t) & (te > t))[0]
    b = act[0] if len(act) else int(np.argmin(np.abs(te - t)))
    f = (t - ts[b]) / max(te[b] - ts[b], 1e-12)
    return C.cx[b], C.cy[b], f * C.top[b]


def light_layer(cam, t, size=0.7):
    x, y, z = frontier_light(t)
    u, v = cam.project(x, y, z)
    return X.light(cam.H * cam.ss, cam.W * cam.ss, u, v, cam.ss, size=size)


PSI, EL = 34, 30


# ================================================================== 02: pull back over Midtown
def f02():
    C = city()
    t = t_radius(650)
    cam = X.Cam(PSI, EL, 0.55, C.cx[C.esb] + 120, C.cy[C.esb] + 250, 0, ox=1060, oy=640)
    fr = city_frame(cam, t, gains=(0.05, 0.09), lines_radius=1400, extra_fn=lambda c: light_layer(c, t))
    fr = nyc_hud(fr, t)
    fr = tag(fr, "SLOW MOTION  ×100", col=INK2)
    fr = caption(fr, [("It's going to clean every floor of every building in New York.", 42, INK, "SemiBold")])
    save(fr, "02_midtown")


# ================================================================== 03: Manhattan, the gold spreading
def f03():
    C = city()
    t = t_radius(2600)
    cam = X.Cam(PSI, 36, 0.105, C.cx[C.esb] + 1500, C.cy[C.esb] + 2500, 0, ox=1150, oy=560)
    fr = city_frame(cam, t, gains=(0.035, 0.07), extra_fn=lambda c: light_layer(c, t, 0.5))
    fr = nyc_hud(fr, t)
    fr = tag(fr, "SLOW MOTION  ×5", col=INK2)
    save(fr, "03_manhattan")


# ================================================================== globe helpers
@functools.lru_cache(1)
def globe():
    import globe as G
    return G.Globe()


def seq_keys():
    import globe as G
    ks, k0 = [], 0.0
    for seq in G.SEQUENCES:
        v = G._xyz(*np.array(seq, float).T)
        L = sum(np.arccos(np.clip(a @ b, -1, 1)) for a, b in zip(v[:-1], v[1:])) * G.R
        ks.append((k0, k0 + L)); k0 += L + 2000.0
    return ks


def done_at_key(K):
    s = np.load(os.path.join(HERE, "data", "ghsl", "sweep.npz"))
    i = np.searchsorted(s["key"], K)
    return float(s["cum"][i - 1]) if i > 0 else 0.0


def world_t(done_m2):
    """light time when done_m2 of floor is clean (travel spread evenly)"""
    return done_m2 / LANE / C_MS * (1 + WORLD_TRAVEL_S / (WORLD_FLOOR_M2 / LANE / C_MS))


def globe_frame(g, done_m2, fire=0.0, light_lonlat=None, glow=1.0):
    import globe as G
    img, gg = globe().render(g, done_m2, ss=SS, fire=fire, glow=glow)
    if light_lonlat is not None:
        u, v, vis = G.ortho_forward(np.array([light_lonlat[0]]), np.array([light_lonlat[1]]), gg)
        if vis[0]:
            img += X.light(H * SS, W * SS, float(u[0]), float(v[0]), SS, size=0.8)
    lum = img.max(axis=2, keepdims=True)
    return to_u8(X.tonemap(X.down(img, SS)))


def world_hud(fr, done_m2, title="EVERY FLOOR ON EARTH"):
    t = world_t(done_m2)
    return hud(fr, [("LIGHT TIME", fmt_s(t), GOLD, 64),
                    ("FLOOR CLEANED", f"{done_m2 / 1e9:,.1f} billion m²" if done_m2 > 1e9 else f"{done_m2 / 1e6:,.0f} million m²", INK, 40),
                    ("OF ABOUT", "273 billion m²", INK2, 34)], title=title)


# ================================================================== 04: then every floor on Earth
def f04():
    done = NYS["floor_area_from_footprints_m2"]
    fr = globe_frame(dict(cx=1100, cy=470, r=430, lon0=-62, lat0=30), done, light_lonlat=(-74.0, 40.7))
    fr = caption(fr, [("Then every floor on Earth.", 48, INK, "SemiBold"), ("How long will it take?", 40, GOLD, "SemiBold")])
    save(fr, "04_earth")


# ================================================================== 05: then and now
def modern_route(RM, lane=LANE):
    """rows across the free floor (robot centre kept R from everything), then an edge pass round the walls and furniture.
    Returns a list of polylines (metres) in driving order."""
    ys = np.arange(RM.R, RM.RH - RM.R + 1e-6, lane)
    xs = np.arange(RM.R, RM.RW - RM.R + 1e-6, 0.02)
    paths = []
    for i, yy in enumerate(ys):
        free = np.array([not RM.blocked(x, yy) for x in xs])
        runs, start = [], None
        for x, fv in zip(xs, free):
            if fv and start is None:
                start = x
            if not fv and start is not None:
                runs.append((start, x - 0.02)); start = None
        if start is not None:
            runs.append((start, xs[-1]))
        for a, c in (runs if i % 2 == 0 else runs[::-1]):
            paths.append(np.array([[a, yy], [c, yy]]) if i % 2 == 0 else np.array([[c, yy], [a, yy]]))
    R = RM.R
    paths.append(np.array([[R, R], [RM.RW - R, R], [RM.RW - R, RM.RH - R], [R, RM.RH - R], [R, R]]))   # walls
    for _, (x0, y0, x1, y1) in RM.BOXES:
        a0, b0, a1, b1 = max(R, x0 - R), max(R, y0 - R), min(RM.RW - R, x1 + R), min(RM.RH - R, y1 + R)
        paths.append(np.array([[a0, b0], [a1, b0], [a1, b1], [a0, b1], [a0, b0]]))
    x, y, r = RM.PLANT
    th = np.linspace(0, 2 * np.pi, 40)
    paths.append(np.c_[x + (r + R) * np.cos(th), y + (r + R) * np.sin(th)])
    return paths


def f05():
    import room as RM
    import maps
    PX = 112.0
    paths = modern_route(RM)
    length = sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in paths)
    length += sum(np.linalg.norm(paths[i + 1][0] - paths[i][-1]) for i in range(len(paths) - 1))
    T = length / RM.V                                           # seconds at a Roomba's 0.3 m/s

    def room_img(ox, oy, mode):
        im = np.zeros((H, W, 3), np.float32)
        Wr, Hr = int(RM.RW * PX), int(RM.RH * PX)
        x0, y0 = int(ox), int(oy)
        im[y0:y0 + Hr, x0:x0 + Wr] = np.array([18, 22, 32]) / 255
        cov = np.zeros((H, W), np.uint8)
        tr = np.zeros((H, W), np.float32)
        wpx = int(2 * RM.R * PX * 0.92)
        if mode == "random":
            P = RM.simulate(seconds=T, dt=0.02)
            pts = np.c_[ox + P[:, 0] * PX, oy + P[:, 1] * PX].astype(np.int32)
            cv2.polylines(cov, [pts], False, 1, wpx, cv2.LINE_8)
            cv2.polylines(tr, [pts[-1500:]], False, 1.0, 2, cv2.LINE_AA)
            end, hd = P[-1], math.atan2(*(P[-1] - P[-8])[::-1])
        else:
            for p in paths:
                q = np.c_[ox + p[:, 0] * PX, oy + p[:, 1] * PX].astype(np.int32)
                cv2.polylines(cov, [q], False, 1, wpx, cv2.LINE_8)
                cv2.polylines(tr, [q], False, 0.5, 1, cv2.LINE_AA)
            end, hd = paths[-1][-1], 0.0
        yy_, xx_ = np.mgrid[y0:y0 + Hr, x0:x0 + Wr]
        mx, my = (xx_ - ox) / PX, (yy_ - oy) / PX
        obst = np.zeros_like(mx, bool)
        for _, (a0, b0, a1, b1) in RM.BOXES:
            obst |= (mx >= a0) & (mx <= a1) & (my >= b0) & (my <= b1)
        obst |= np.hypot(mx - RM.PLANT[0], my - RM.PLANT[1]) <= RM.PLANT[2]
        free = np.zeros((H, W), bool)
        free[y0:y0 + Hr, x0:x0 + Wr] = ~obst
        c = (cov.astype(bool) & free).astype(np.float32)
        pct = c.sum() / free.sum()
        g = np.array(GOLD, np.float32) / 255
        im = im * (1 - 0.45 * c[..., None]) + g * 0.45 * c[..., None] + tr[..., None] * np.array([1, 0.92, 0.75]) * 0.5
        out = Image.fromarray(to_u8(im))
        d = ImageDraw.Draw(out, "RGBA")
        px = lambda x, y: (ox + x * PX, oy + y * PX)
        for nm, (a0, b0, a1, b1) in RM.BOXES:
            d.rounded_rectangle([*px(a0, b0), *px(a1, b1)], 8, fill=(40, 46, 62), outline=(86, 96, 118), width=2)
            d.text(px((a0 + a1) / 2, (b0 + b1) / 2), nm, font=ST.font(19), fill=(130, 140, 160), anchor="mm")
        x, y, r = RM.PLANT
        d.ellipse([*px(x - r, y - r), *px(x + r, y + r)], fill=(34, 64, 48), outline=(70, 120, 90), width=2)
        d.rectangle([*px(0, 0), *px(RM.RW, RM.RH)], outline=(150, 160, 180), width=5)
        return out, pct, px(*end), hd

    L, pl, el_, hl = room_img(130, 290, "random")
    x1 = W - 130 - RM.RW * PX
    Rr, pr, er, hr = room_img(x1, 290, "rows")
    fr = np.maximum(np.asarray(L), np.asarray(Rr))
    im = Image.fromarray(fr)
    maps.draw_roomba(im, el_[0], el_[1], hl, diam=int(2 * RM.R * PX))
    maps.draw_roomba(im, er[0], er[1], hr, diam=int(2 * RM.R * PX))
    d = ImageDraw.Draw(im, "RGBA")
    mins = f"{T / 60:.1f}".rstrip("0").rstrip(".")
    text(d, (130, 190), "LAST TIME · A 2002 ROOMBA", 26, MUTED)
    text(d, (130, 226), "No map. Bump, turn, repeat.", 34, INK2)
    text(d, (x1, 190), "THIS TIME · A MODERN ROBOT VACUUM", 26, MUTED)
    text(d, (x1, 226), "Maps the room. Rows, then the edges.", 34, INK2)
    yb = 290 + RM.RH * PX + 24
    text(d, (130, yb), f"after {mins} minutes: {pl:.0%} of the floor", 30, INK, 1.0, "SemiBold")
    text(d, (x1, yb), f"after {mins} minutes: done ({pr:.0%})", 30, GOLD, 1.0, "SemiBold")
    fr = np.asarray(im)
    fr = caption(fr, [("This one has a map. Rows 25 cm apart, every square metre once.", 40, INK, "SemiBold")])
    save(fr, "05_then_and_now")
    print("modern route", length, "m", T / 60, "min; random", pl, "modern", pr)


# ================================================================== 06: the rules, on one real floor
def f06():
    import shapely
    C = city()
    tower = esb_tower()
    ring = C.rxy[C.ps[tower]:C.pe[tower]][:-1]
    P = shapely.Polygon(ring)
    area = P.area
    segs = rows_in(ring)
    rows_m = sum(np.linalg.norm(q - p) for p, q in segs)
    mn, mx = ring.min(0), ring.max(0)
    s = 640 / max(mx - mn)
    ox, oy = 560, 560
    tf = lambda xy: (ox + (xy[0] - (mn[0] + mx[0]) / 2) * s, oy - (xy[1] - (mn[1] + mx[1]) / 2) * s)
    im = Image.new("RGB", (W, H), (5, 8, 15))
    d = ImageDraw.Draw(im, "RGBA")
    d.polygon([tf(p) for p in ring], fill=(14, 26, 44))
    k_now = int(len(segs) * 0.62)
    for i, (p, q) in enumerate(segs[:k_now]):
        col = (255, 186, 80, 150 if i % 2 else 205)
        d.line([tf(p), tf(q)], fill=col, width=max(1, int(LANE * s)))
    for i, (p, q) in enumerate(segs[:k_now - 1]):              # the turn at the end of each row
        a, b = segs[i][1], segs[i + 1][0]
        d.line([tf(a), tf(b)], fill=(255, 186, 80, 120), width=1)
    p, q = segs[k_now]
    m = p + 0.55 * (q - p)
    d.line([tf(p), tf(m)], fill=(255, 240, 210, 255), width=max(2, int(LANE * s)))
    d.polygon([tf(p) for p in ring], outline=(150, 200, 240), width=3)
    fr = np.asarray(im).copy()
    import maps
    im = Image.fromarray(fr)
    hx, hy = tf(q)
    mx_, my_ = tf(m)
    maps.draw_roomba(im, mx_, my_, math.atan2(hy - my_, hx - mx_), diam=max(8, int(0.35 * s * 1.6)))
    d = ImageDraw.Draw(im, "RGBA")
    text(d, (ox, oy + 360), f"One floor of the Empire State Building's tower · {area:,.0f} m²", 26, INK2, 1.0, "SemiBold", "ma")
    x0 = 1060
    text(d, (x0, 150), "THE RULES", 26, MUTED)
    rules = [("Rows 25 cm apart.", "Every square metre of floor, once."),
             ("Every floor of every building.", "Bottom to top, then on to the next building."),
             ("Light speed the whole way.", "No slowing down for corners. Or stairs."),
             ("Floors only.", "Not walls, not ceilings, not furniture.")]
    y = 200
    for a_, b_ in rules:
        text(d, (x0, y), a_, 38, INK, 1.0, "SemiBold"); text(d, (x0, y + 50), b_, 28, INK2, 1.0, "Regular"); y += 128
    text(d, (x0, y + 20), f"This floor: {rows_m / 1e3:.1f} km of rows", 34, GOLD, 1.0, "SemiBold")
    text(d, (x0, y + 66), f"Light needs {rows_m / C_MS * 1e6:.0f} microseconds.", 34, GOLD, 1.0, "SemiBold")
    save(np.asarray(im), "06_the_rules")


# ================================================================== 07: one Empire State Building
def f07():
    C = city()
    b = C.esb
    t = float(route()[2][b])
    cam = X.Cam(PSI, 26, 2.05, C.cx[b], C.cy[b], 215, ox=1180, oy=500)
    level = levels_at(t)
    img = np.zeros((H * SS, W * SS, 3), np.float32) + X.BG
    building_layers(C, cam, b, level, img, above_gain=0.55, below_gain=0.85, radius=900, other_gain=0.16)
    img = X.bloom(img, SS)
    fr = to_u8(X.tonemap(X.down(img, SS)))
    fr = hud(fr, [("LIGHT TIME", fmt_s(NYC["esb_light_s"]), GOLD, 72),
                  ("FLOOR", f"{NYC['esb_floor_area_m2']:,.0f} m²", INK, 40),
                  ("ROWS DRIVEN", f"{NYC['esb_rows_km']:,.0f} km", INK, 40)], title="EMPIRE STATE BUILDING · 102 FLOORS")
    blink = 0.1 / NYC["esb_light_s"]
    fr = caption(fr, [("The whole Empire State Building: 3.5 milliseconds.", 44, INK, "SemiBold"),
                      (f"In the blink of an eye (about 0.1 s), it could clean {blink:.0f} of them.", 34, GOLD, "SemiBold")])
    save(fr, "07_empire_state")


# ================================================================== 08: Manhattan
def f08():
    C = city()
    t = t_radius(5200)
    cam = X.Cam(PSI, 36, 0.105, C.cx[C.esb] + 1500, C.cy[C.esb] + 2500, 0, ox=1150, oy=560)
    fr = city_frame(cam, t, gains=(0.035, 0.07), extra_fn=lambda c: light_layer(c, t, 0.5))
    fr = nyc_hud(fr, t)
    mn = NYS["floor_area_by_borough_m2"]["MN"]
    fr = caption(fr, [(f"Manhattan alone: {mn / 1e6:,.0f} million m² of floor.", 42, INK, "SemiBold"),
                      (f"Light needs {mn / LANE / C_MS:.1f} seconds to vacuum it.", 34, GOLD, "SemiBold")])
    save(fr, "08_manhattan_half")


# ================================================================== 09: every floor in New York
def f09():
    C = city()
    t = NYC["total_light_s"] + 1e-6
    cam = X.Cam(PSI, 40, 0.036, C.cx[C.esb] + 6000, C.cy[C.esb] - 5000, 0, ox=1240, oy=560)
    fr = city_frame(cam, t, gains=(0.02, 0.10), bloom_k=(0.25, 0.16, 0.10))
    moon = NYC["rows_km"] / 384400
    fr = hud(fr, [("LIGHT TIME", fmt_s(NYC["total_light_s"]), GOLD, 72),
                  ("BUILDINGS", f"{NYC['buildings']:,}", INK, 40),
                  ("FLOOR", f"{NYC['floor_area_m2'] / 1e6:,.0f} million m²", INK, 40),
                  ("ROWS DRIVEN", f"{NYC['rows_km'] / 1e6:.2f} million km", INK, 40)], title="NEW YORK CITY · EVERY FLOOR")
    fr = caption(fr, [(f"Every floor in New York: {NYC['total_light_s']:.1f} seconds.", 46, GOLD, "Bold"),
                      (f"{NYC['rows_km'] / 1e6:.1f} million km of rows: {moon:.1f} times the distance to the Moon.", 32, INK2, "Regular")])
    save(fr, "09_new_york_done")


# ================================================================== 10: the Americas done
def f10():
    ks = seq_keys()
    done = done_at_key(ks[3][1])                     # through South America
    fr = globe_frame(dict(cx=1240, cy=470, r=430, lon0=-75, lat0=12), done, light_lonlat=(-69, -53))
    fr = world_hud(fr, done)
    fr = caption(fr, [("Earth has about 273 billion m² of floor.", 44, INK, "SemiBold"),
                      ("About the area of New Zealand, mostly in homes.", 34, GOLD, "SemiBold")])
    save(fr, "10_the_americas")


# ================================================================== 11: Asia
def f11():
    import globe as G
    ks = seq_keys()
    K = ks[6][0] + 0.45 * (ks[6][1] - ks[6][0])
    done = done_at_key(K)
    pts, keys = G.densified()
    i = int(np.searchsorted(keys, K)); v = pts[min(i, len(pts) - 1)]
    lon, lat = math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(v[2]))
    g = dict(cx=1240, cy=560, r=470, lon0=100, lat0=25)
    fr = globe_frame(g, done, light_lonlat=(lon, lat))
    fr = world_hud(fr, done)
    cj = json.load(open(os.path.join(HERE, "data", "ghsl", "cities.json")))
    cities = {"Tokyo": (139.69, 35.69), "Beijing": (116.40, 39.90), "Shanghai": (121.47, 31.23), "Delhi": (77.21, 28.61)}
    im = Image.fromarray(fr); d = ImageDraw.Draw(im, "RGBA")
    for nm, (lo, la) in cities.items():
        u, v_, vis = G.ortho_forward(np.array([lo]), np.array([la]), g)
        u, v_ = float(u[0]), float(v_[0])
        side = 1 if nm in ("Tokyo", "Shanghai") else -1
        d.ellipse([u - 7, v_ - 7, u + 7, v_ + 7], outline=(240, 244, 250, 230), width=2)
        lx = u + side * 26
        lab = f"{nm} · {cj[nm] / LANE / C_MS:.0f} s"
        text(d, (lx, v_ - 16), lab, 26, INK, 1.0, "SemiBold", "la" if side > 0 else "ra")
    fr = np.asarray(im)
    cc = json.load(open(os.path.join(HERE, "data", "ghsl", "countries.json")))["top_countries_m2"]
    china = cc["People's Republic of China"] / WORLD_FLOOR_M2
    fr = caption(fr, [(f"China has about a fifth of Earth's floor ({china:.0%}).", 42, INK, "SemiBold"),
                      ("Greater Tokyo is the biggest single city: 24 seconds of light.", 34, GOLD, "SemiBold")])
    save(fr, "11_asia")


# ================================================================== 12: how far
def f12():
    AU = 149597870.7
    rows_km = WORLD_FLOOR_M2 / LANE / 1e3
    im = Image.new("RGB", (W, H), (5, 8, 15))
    d = ImageDraw.Draw(im, "RGBA")
    x0, x1, y = 170, 1760, 520
    sc = (x1 - x0) / 10.0
    d.line([(x0, y), (x1, y)], fill=(60, 74, 96), width=2)
    for k in range(1, 11):
        d.line([(x0 + k * sc, y - 8), (x0 + k * sc, y + 8)], fill=(60, 74, 96), width=2)
        text(d, (x0 + k * sc, y + 22), f"{k} AU", 20, MUTED, 1.0, "Regular", "ma")
    d.ellipse([x0 - 34, y - 34, x0 + 34, y + 34], fill=(255, 210, 120))
    text(d, (x0, y - 70), "Sun", 26, INK2, 1.0, "SemiBold", "ma")
    for nm, au, r in (("Mercury", 0.387, 5), ("Venus", 0.723, 8), ("Earth", 1.0, 8), ("Mars", 1.524, 6),
                      ("Jupiter", 5.204, 20), ("Saturn", 9.537, 17)):
        x = x0 + au * sc
        d.ellipse([x - r, y - r, x + r, y + r], fill=(170, 186, 206))
        if nm in ("Earth", "Jupiter", "Saturn", "Mars"):
            text(d, (x, y - r - 40), nm, 26, INK2, 1.0, "SemiBold", "ma")
    xe = x0 + rows_km / AU * sc
    d.line([(x0, y - 120), (xe, y - 120)], fill=(255, 186, 80, 255), width=10)
    d.ellipse([xe - 12, y - 132, xe + 12, y - 108], fill=(255, 240, 210))
    text(d, (xe, y - 190), f"{rows_km / 1e9:.2f} billion km", 40, GOLD, 1.0, "Bold", "ma")
    text(d, ((x0 + xe) / 2, y - 100), "every row on every floor, end to end", 26, INK2, 1.0, "Regular", "ma")
    fr = caption(np.asarray(im), [(f"{rows_km / 1e9:.2f} billion km of rows: farther than Jupiter is from the Sun.", 42, INK, "SemiBold")])
    save(fr, "12_how_far")


# ================================================================== 13: the last floor
def f13():
    import globe as G
    total = WORLD_FLOOR_M2 * 1.0001
    g = dict(cx=1240, cy=440, r=420, lon0=150, lat0=-64)
    fr = globe_frame(g, total, light_lonlat=(0.0, -89.99))
    u, v, _ = G.ortho_forward(np.array([0.0]), np.array([-89.99]), g)
    fr = ST.ring_label(fr, int(u[0]), int(v[0]), "Amundsen–Scott South Pole Station: the last floor", 1.0, side=-1)
    T = WORLD_FLOOR_M2 / LANE / C_MS + WORLD_TRAVEL_S
    fr = hud(fr, [("LIGHT TIME", "1 h 08 min", GOLD, 72),
                  ("VACUUMING", fmt_s(WORLD_FLOOR_M2 / LANE / C_MS), INK, 40),
                  ("GETTING THERE", "≈ 7 min", INK, 40),
                  ("BUILDINGS", "≈ 2.5 billion", INK, 40)], title="EVERY FLOOR ON EARTH")
    fr = caption(fr, [("Every floor on Earth: about 1 hour 8 minutes.", 46, GOLD, "Bold"),
                      ("An hour of vacuuming, and 7 minutes getting between 2.5 billion buildings.", 32, INK2, "Regular")])
    save(fr, "13_last_floor")


# ================================================================== 14: the dustbin (brief)
def f14():
    g = dict(cx=1240, cy=560, r=470, lon0=20, lat0=20)
    fr = globe_frame(g, WORLD_FLOOR_M2 * 1.0001, glow=0.6)
    fr = (fr.astype(np.float32) * 0.55).astype(np.uint8)
    dust_t = WORLD_FLOOR_M2 * 0.3 / 1e6
    bins = WORLD_FLOOR_M2 * 0.3 / 150
    yrs = bins * 10 / 3.156e7
    fr = hud(fr, [("DUST PICKED UP", f"≈ {dust_t / 1e3:,.0f},000 tonnes", INK, 52),
                  ("BIN EMPTIED", f"≈ {bins / 1e6:,.0f} million times", INK, 52),
                  ("AT 10 S PER EMPTYING", f"≈ {yrs:,.0f} years", GOLD, 52)], title="THE DUSTBIN", y0=300, x0=110)
    fr = caption(fr, [("Oh, and the dust: about half a billion bin-fulls.", 42, INK, "SemiBold"),
                      ("The robot is fast. The dock that empties it is not.", 34, GOLD, "SemiBold")])
    save(fr, "14_dustbin")


# ================================================================== 15: the U-turn
def f15():
    C = city()
    tower = esb_tower()
    j = 14
    zf = C.pz0[tower] + j * (C.pz1[tower] - C.pz0[tower]) / C.pnfl[tower]
    ring = C.rxy[C.ps[tower]:C.pe[tower]][:-1]
    segs = rows_in(ring)
    lens = np.array([np.linalg.norm(q - p) for p, q in segs]); cum = np.cumsum(lens)
    k = int(len(segs) * 0.40)
    f = (cum[k] - 0.2) / cum[-1]                                 # the end of a row, at the wall
    r = segs[k][1]
    si = 120.0
    cam = X.Cam(PSI, 40, si, r[0], r[1], zf, ox=960, oy=500)
    img = np.zeros((H * SS, W * SS, 3), np.float32) + X.BG
    x, y, hd = active_floor(C, tower, zf, f, si, cam, img, dust_n=500000, wake=True, rows_gain=0.30)
    img += X.light(H * SS, W * SS, x * SS, y * SS, SS, size=6.0)
    img = X.bloom(img, SS, k=(0.3, 0.25, 0.2))
    fr = to_u8(X.tonemap(X.down(img, SS)))
    m, c2 = 4.0, C_MS ** 2
    gam = 1 / math.sqrt(1 - 0.99 ** 2)
    ke = (gam - 1) * m * c2
    tsar = 50 * 4.184e15
    turns = WORLD_FLOOR_M2 / LANE / 4.0
    fr = hud(fr, [("ONE STOP AT 99% OF LIGHT SPEED", f"{ke / tsar:.0f} Tsar Bombas", GOLD, 56),
                  ("U-TURNS, EVERY FLOOR ON EARTH", f"≈ {turns / 1e9:,.0f} billion", INK, 44)], title="PHYSICS FOOTNOTE")
    fr = caption(fr, [("Nothing with mass can actually reach light speed.", 40, INK, "SemiBold"),
                      ("At 99% of it, every U-turn at the end of a row releases about 10 Tsar Bombas.", 34, GOLD, "SemiBold")])
    save(fr, "15_u_turn")


# ================================================================== 16: the end
def f16():
    g = dict(cx=960, cy=500, r=400, lon0=-40, lat0=25)
    fr = globe_frame(g, WORLD_FLOOR_M2 * 1.0001, fire=1.6)
    turns = WORLD_FLOOR_M2 / LANE / 4.0
    gam = 1 / math.sqrt(1 - 0.99 ** 2)
    E = turns * (gam - 1) * 4.0 * C_MS ** 2
    sun_min = E / 3.828e26 / 60
    fr = tag(fr, f"{turns / 1e9:.0f} billion U-turns ≈ {sun_min:.0f} minutes of the Sun's entire output", x=960, y=40, col=INK2, size=26, anchor="ma")
    fr = caption(fr, [("The floors would be spotless. There just wouldn't be any floors.", 42, INK, "SemiBold"),
                      ("What should it clean next?", 40, GOLD, "Bold")])
    save(fr, "16_spotless")


# ================================================================== thumbnails (1280x720)
def big(d, xy, s, size, col, anchor="la", font="Anton-Regular.ttf", stroke=0):
    from PIL import ImageFont
    f = ImageFont.truetype(os.path.join(ROOT, "thumb_fonts", font), size)
    d.text(xy, s, font=f, fill=col, anchor=anchor, stroke_width=stroke, stroke_fill=(5, 8, 15))


def tA():
    """the X-ray Empire State Building half cleaned, the light on its floor; EVERY FLOOR / ON EARTH"""
    C = city()
    b = C.esb
    tower = esb_tower()
    zf = C.pz0[tower] + 20 * (C.pz1[tower] - C.pz0[tower]) / C.pnfl[tower]
    cam = X.Cam(PSI, 24, 2.25, C.cx[b], C.cy[b], 200, ox=1380, oy=520)
    level = np.full(C.n, -1.0); level[b] = zf - 0.01
    img = np.zeros((H * SS, W * SS, 3), np.float32) + X.BG
    building_layers(C, cam, b, level, img, above_gain=0.65, below_gain=1.0, radius=900, other_gain=0.12)
    x, y, _ = active_floor(C, tower, zf, 0.5, 2.25, cam, img, dust_n=0, wake=False, rows_gain=1.0)
    img += X.light(H * SS, W * SS, x * SS, y * SS, SS, size=1.3)
    img = X.bloom(img, SS)
    fr = to_u8(X.tonemap(X.down(img, SS)))
    im = Image.fromarray(fr).resize((1280, 720), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    big(d, (56, 70), "EVERY FLOOR", 150, (255, 255, 255))
    big(d, (56, 250), "ON EARTH", 150, (255, 196, 84))
    big(d, (60, 450), "A ROBOT VACUUM", 58, (220, 230, 245))
    big(d, (60, 520), "AT LIGHT SPEED", 58, (220, 230, 245))
    os.makedirs(OUT, exist_ok=True)
    im.save(os.path.join(OUT, "thumb_A.png")); print("wrote thumb_A")


def tB():
    """the globe half gold, the light at the frontier; HOW LONG TO VACUUM EARTH?"""
    ks = seq_keys()
    done = done_at_key(ks[5][1])
    import globe as G
    g = dict(cx=1250, cy=560, r=560, lon0=45, lat0=22)
    pts, keys = G.densified()
    i = int(np.searchsorted(keys, ks[5][1])); v = pts[min(i, len(pts) - 1)]
    lon, lat = math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(v[2]))
    fr = globe_frame(g, done, light_lonlat=(lon, lat))
    im = Image.fromarray(fr).resize((1280, 720), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    big(d, (50, 60), "VACUUMING", 120, (255, 255, 255))
    big(d, (50, 200), "EVERY FLOOR", 120, (255, 255, 255))
    big(d, (50, 340), "ON EARTH", 120, (255, 196, 84))
    big(d, (54, 500), "AT THE SPEED OF LIGHT", 54, (220, 230, 245))
    im.save(os.path.join(OUT, "thumb_B.png")); print("wrote thumb_B")


FRAMES = {"01": f01, "02": f02, "03": f03, "04": f04, "05": f05, "06": f06, "07": f07, "08": f08, "09": f09,
          "10": f10, "11": f11, "12": f12, "13": f13, "14": f14, "15": f15, "16": f16, "tA": tA, "tB": tB}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FRAMES)
    for nm in names:
        FRAMES[nm]()
