"""Assemble the Blender edition: plates + chip passes + cover + LED panel + motion graphics -> film.

    python3 composite3d.py still <shot> <t> [...]   # PNG checks -> out/c3d_<shot>_<t>.png
    python3 composite3d.py film --workers 4         # -> out/plinko_blender.mp4

Layers per frame (all registered to the Blender camera, cropped exactly as the chip pass was rendered):
  plate (studio + board, no cover)  ->  contact shadows from the chips' alpha  ->  chips  ->  plexiglass cover
  ->  lit LED cells with bloom  ->  projected callouts, trails and captions.
"""
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import cairo
import cv2
import numpy as np

import board as B
import draw as D

OUT = "out"
FPS = 30
OW, OH = 1920, 1080
META = json.load(open(f"{OUT}/anim3d/shots.json"))
PATHS = np.load(f"{OUT}/anim3d/paths.npz")
STRAT = json.load(open("results/strategy.json"))
TECH = json.load(open("results/technique.json"))["styles"]
REAL = json.load(open("data/real_outcomes.json"))
RMAP = np.load("results/releasemap.npz")
RMAP_J = json.load(open("results/releasemap.json"))
CHAOS_TBL = [("1 micron", 0.143), ("1 nanometre", 0.170), ("1 atom (0.1 nm)", 0.237), ("1/100 of an atom", 0.476)]
W, H = B.SLOT_PITCH, B.ROW_SPACING
TILT = math.radians(B.BOARD_TILT_DEG)
FLOOR_Y = B.last_row_y() + B.FLOOR_BELOW_LAST_ROW
RIG_Z = 0.85 + FLOOR_Y * math.cos(TILT)
DEPTH = B.CHIP_THICKNESS / 2 + 0.0015
VAL_COL = {100: (0.565, 0.522, 0.914), 500: (0.098, 0.620, 0.439), 1000: (0.224, 0.529, 0.898),
           0: (0.32, 0.32, 0.34), 10000: (0.929, 0.631, 0.0)}


# ----------------------------------------------------------------------------------------------- geometry
def board_world(x, y, depth=0.0):
    """Board coordinates -> world (same transform as blender/scene.py)."""
    lx, ly, lz = x - 5 * W, -depth, -y
    c, s = math.cos(-TILT), math.sin(-TILT)
    return np.array([lx, ly * c - lz * s, ly * s + lz * c + RIG_Z])


class Cam:
    def __init__(self, name):
        j = json.load(open(f"{OUT}/plates/{name}.json"))
        self.j = j
        self.Minv = np.linalg.inv(np.array(j["matrix_world"]))
        self.k = j["lens"] / j["sensor_width"]
        self.PW, self.PH = j["res"]
        self.pts = j["points"]
        for pw, ndc in zip(j["probe_world"], j["probe_ndc"]):     # check against Blender's own projection
            u, v = self.ndc(pw)
            assert abs(u - ndc[0]) < 1e-4 and abs(v - ndc[1]) < 1e-4, (u, v, ndc)

    def ndc(self, p):
        q = self.Minv @ np.array([p[0], p[1], p[2], 1.0])
        x = 0.5 + self.k * q[0] / -q[2] - self.j["shift_x"]
        y = 0.5 + (self.k * q[1] / -q[2] - self.j["shift_y"]) * (self.PW / self.PH)
        return x, y

    def px(self, p, crop):
        """World point -> output pixel for crop (s, cx, cy)."""
        s, cx, cy = crop
        x, y = self.ndc(p)
        return ((x - cx) / s + 0.5) * OW, (((1 - y) - cy) / s + 0.5) * OH


_cams, _plates = {}, {}


def cam(name):
    if name not in _cams:
        _cams[name] = Cam(name)
    return _cams[name]


def plate(name, kind):
    key = (name, kind)
    if key not in _plates:
        im = cv2.imread(f"{OUT}/plates/{name}_{kind}.png", cv2.IMREAD_UNCHANGED)
        im = im.astype(np.float32) / 255.0
        if im.shape[2] == 4:
            im[..., :3] *= im[..., 3:4]                       # premultiply
        _plates[key] = cv2.GaussianBlur(im, (0, 0), 0.45)
    return _plates[key]


def crop_img(img, crop):
    s, cx, cy = crop
    PH, PW = img.shape[:2]
    sc = OW / (s * PW)
    M = np.array([[sc, 0, OW / 2 - cx * PW * sc], [0, sc, OH / 2 - cy * PH * sc]], np.float32)
    return cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_at(shot, t):
    ks = shot["crop"]
    if len(ks) == 1 or t <= ks[0][0]:
        return ks[0][1:]
    for a, b in zip(ks[:-1], ks[1:]):
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0])
            f = f * f * (3 - 2 * f)
            return [a[i] + (b[i] - a[i]) * f for i in (1, 2, 3)]
    return ks[-1][1:]


# ----------------------------------------------------------------------------------------------- base frame
def base_frame(shot, fr, t):
    crop = crop_at(shot, t)
    c = shot["cam"]
    img = crop_img(plate(c, "set"), crop)
    chip_path = f"{OUT}/chips/{shot['name']}/f{fr:04d}.png"
    static = f"{OUT}/chips/{shot['name']}/static.png"
    if not os.path.exists(chip_path) and os.path.exists(static):
        fj = json.load(open(f"{OUT}/chips/{shot['name']}/frames.json"))
        if fr >= fj.get("static_from", 10**9):
            chip_path = static
    if os.path.exists(chip_path):
        ch = cv2.imread(chip_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        a = ch[..., 3:4]
        # contact shadow: the chip's own silhouette, softened and nudged down-right, darkens the board
        chip_px = max(8.0, chip_size_px(shot, crop))
        sh = cv2.GaussianBlur(a[..., 0], (0, 0), chip_px * 0.10)
        sh = np.roll(np.roll(sh, int(chip_px * 0.05), 1), int(chip_px * 0.08), 0)
        img *= (1 - 0.55 * sh)[..., None]
        img = ch[..., :3] * a + img * (1 - a)
    cov = crop_img(plate(c, "cover"), crop)
    img = cov[..., :3] + img * (1 - cov[..., 3:4])
    return img, crop


def chip_size_px(shot, crop):
    cm = cam(shot["cam"])
    a = cm.px(board_world(5 * W, 6 * H, DEPTH), crop)
    b = cm.px(board_world(5 * W + B.CHIP_DIAMETER, 6 * H, DEPTH), crop)
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------------------------------- overlays
def to_surface(img):
    """float BGR image -> cairo ARGB32 surface."""
    a = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    buf = np.empty((OH, OW, 4), np.uint8)
    buf[..., :3] = a
    buf[..., 3] = 255
    s = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, OW, OH)
    return s, buf


def glow_layer():
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, OW, OH)
    return s, cairo.Context(s)


def add_glow(img, surf, sigma=9.0, strength=1.2):
    """Add a cairo-drawn emissive layer and its bloom onto the float image (BGR)."""
    surf.flush()
    g = np.ndarray((OH, OW, 4), np.uint8, surf.get_data()).astype(np.float32)[..., :3] / 255.0
    bloom = cv2.GaussianBlur(g, (0, 0), sigma) + 0.5 * cv2.GaussianBlur(g, (0, 0), sigma * 3)
    return np.clip(img + g + bloom * strength, 0, 1.4)


def quad(ctx, pts):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def led_quad(cm, crop, k, r, frac=1.0):
    (x0, y0, z0), (x1, y1, z1) = cm.pts["led_cells"][k * 20 + r]
    z1 = z0 + (z1 - z0) * frac
    return [cm.px((x0, y0, z0), crop), cm.px((x1, y0, z0), crop), cm.px((x1, y0, z1), crop),
            cm.px((x0, y0, z1), crop)]


def draw_leds(ctx, cm, crop, levels, color, alpha=1.0):
    """Light each column up to `levels[k]` cells (fractional top cell allowed)."""
    for k in range(9):
        lvl = max(0.0, min(20.0, levels[k]))
        for r in range(int(math.ceil(lvl))):
            frac = min(1.0, lvl - r)
            quad(ctx, led_quad(cm, crop, k, r, 1.0))
            ctx.set_source_rgba(*color, alpha * (0.25 + 0.75 * frac))
            ctx.fill()


def draw_caps(ctx, cm, crop, levels, color, alpha=1.0, width=3.0):
    """A bright line at height `levels[k]` cells over each column (a reference value)."""
    for k in range(9):
        lvl = max(0.0, min(20.0, levels[k]))
        r = min(19, int(lvl))
        (x0, y0, z0), (x1, y1, z1) = cm.pts["led_cells"][k * 20 + r]
        pitch = (cm.pts["led_cells"][1][0][2] - cm.pts["led_cells"][0][0][2])
        z = cm.pts["led_cells"][k * 20][0][2] + lvl * pitch
        a = cm.px((x0 - 0.01, y0, z), crop)
        b = cm.px((x1 + 0.01, y0, z), crop)
        ctx.move_to(*a)
        ctx.line_to(*b)
        ctx.set_source_rgba(*color, alpha)
        ctx.set_line_width(width)
        ctx.stroke()


def scrim(ctx, x1=900, alpha=0.78):
    """Darken the left of the frame so captions read over the studio."""
    g = cairo.LinearGradient(0, 0, x1, 0)
    g.add_color_stop_rgba(0, 0.02, 0.025, 0.035, alpha)
    g.add_color_stop_rgba(0.7, 0.02, 0.025, 0.035, alpha * 0.75)
    g.add_color_stop_rgba(1, 0.02, 0.025, 0.035, 0)
    ctx.rectangle(0, 0, x1, OH)
    ctx.set_source(g)
    ctx.fill()


def money(x):
    return f"${x:,.0f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


LX, LW = 96, 720


def head(ctx, s, t, t0=0.3, y=150, sub=None, t1=1e9, size=56):
    a = D.fade(t, t0, t1, 0.5)
    D.text(ctx, s, LX, y, size, D.INK, a, "black")
    if sub:
        D.paragraph(ctx, sub, LX, y + size * 0.95, 27, LW, D.INK2, a)


def coin_levels():
    from galton import kernel
    return kernel()[4] / 0.0125


def real_levels():
    s = np.array(REAL["totals"]["slots_left_to_right"], float)
    return s / s.sum() / 0.0125


# ----------------------------------------------------------------------------------------------- shots
def ov_open(ctx, gctx, t, shot, cm, crop):
    path = PATHS["open"]
    t_land = 2.0 + path[-1, 0]
    if t > t_land:
        k = int(round(path[-1, 1] / W))
        x, y = cm.px(board_world(k * W, B.last_row_y() + 0.2 * W, 0.05), crop)
        D.pill(ctx, D.SLOT_LABELS[k - 1], x, y - 60, 46, D.INK, alpha=D.fade(t, t_land, 99, 0.25))
    a = D.fade(t, t_land + 0.7, 99, 0.6)
    if a > 0:
        scrim(ctx, 760, 0.55 * a)
        D.text(ctx, "Where should you", 90, 440, 60, D.INK, a, "black")
        D.text(ctx, "drop a Plinko chip?", 90, 512, 60, D.INK, a, "black")
        D.paragraph(ctx, "I measured the board, rebuilt it in a physics engine, and checked it against 1,029 "
                         "real chips from the show.", 92, 575, 26, 560, D.INK2, a)


def ov_measure(ctx, gctx, t, shot, cm, crop):
    scrim(ctx)
    head(ctx, "The board, measured", t, 0.4, 150)
    items = [(1.5, "9 slots, 7.0 in (17.8 cm) apart"),
             (4.5, "13 rows of steel pins: 98 in all"),
             (8.0, "Sawtooth walls: 6 teeth a side stand in for the end pins"),
             (11.5, "Chip: 5.1 in (13 cm) across, about 3/4 in thick"),
             (14.5, "Playfield: 63 in wide, 73 in from the top row of pins to the bottom row")]
    y = 250
    for ts, s in items:
        a = D.fade(t, ts, 99, 0.5)
        if a <= 0:
            continue
        ctx.arc(LX + 8, y - 9, 6, 0, 2 * math.pi)
        ctx.set_source_rgba(*D.GOLD, a)
        ctx.fill()
        y = D.paragraph(ctx, s, LX + 30, y, 30, LW - 40, D.INK, a, "medium") + 22
    a = D.fade(t, 1.5, 4.4, 0.4)
    if a > 0:       # slot pitch: bracket over the $10,000 slot
        p0 = cm.px(board_world(4.5 * W, FLOOR_Y + 0.05, 0.05), crop)
        p1 = cm.px(board_world(5.5 * W, FLOOR_Y + 0.05, 0.05), crop)
        D.dim_line(gctx, p0[0], p0[1] + 34, p1[0], p1[1] + 34, "7.0 in", 22, a, D.GOLD, side=-1)
    a = D.fade(t, 4.5, 8.0, 0.3)
    if a > 0:       # pins, row by row
        row = min(12, int((t - 4.5) * 5))
        pins, rows = B.pegs()
        for (x, y_), r in zip(pins, rows):
            if r <= row:
                px_, py_ = cm.px(board_world(x, y_, 0.044), crop)
                gctx.arc(px_, py_, 4.5, 0, 2 * math.pi)
                gctx.set_source_rgba(*D.GOLD, a * (1.0 if r == row else 0.55))
                gctx.fill()
    a = D.fade(t, 8.0, 11.4, 0.3)
    if a > 0:       # wall teeth
        for p in cm.pts["wall_teeth"]:
            px_, py_ = cm.px(p, crop)
            gctx.arc(px_, py_, 7, 0, 2 * math.pi)
            gctx.set_source_rgba(*D.GOLD, a)
            gctx.fill()
    a = D.fade(t, 12.0, 99, 0.4)
    if a > 0:       # chip size
        c0 = cm.px(board_world(5 * W - B.CHIP_DIAMETER / 2, -0.02 * W - 0.10, 0.05), crop)
        c1 = cm.px(board_world(5 * W + B.CHIP_DIAMETER / 2, -0.02 * W - 0.10, 0.05), crop)
        D.dim_line(gctx, c0[0], c0[1], c1[0], c1[1], "5.1 in", 20, a, D.GOLD, side=1)
    a = D.fade(t, 14.5, 99, 0.5)
    if a > 0:       # field size
        tl = cm.px(board_world(0.5 * W, 0, 0.05), crop)
        bl = cm.px(board_world(0.5 * W, B.last_row_y(), 0.05), crop)
        D.dim_line(gctx, tl[0] - 46, tl[1], bl[0] - 46, bl[1], "73 in", 22, a, D.GOLD, side=-1)
    a = D.fade(t, 15.5, 99, 0.6)
    D.paragraph(ctx, "No blueprint of the TV board is public: measured from broadcast frames, scaled by the chip, "
                     "cross-checked against the host's height and a published diagram.", LX, 1000, 20, LW, D.INK3, a)


def ov_textbook(ctx, gctx, t, shot, cm, crop):
    scrim(ctx)
    head(ctx, "The textbook answer", t, 0.3, 150,
         "Treat every pin as a coin flip: half a slot left or right. A chip meets 12 pins on the way down.")
    p = PATHS["coin"]
    tt = t - 1.5
    n = int(np.clip(tt * 60, 0, len(p) - 1)) if tt > 0 else 0
    if tt > 0:          # glowing path behind the chip
        gctx.set_line_width(3)
        gctx.set_source_rgba(*D.S2, 0.9)
        for i in range(0, n + 1, 2):
            q = cm.px(board_world(p[i, 1], p[i, 2], 0.03), crop)
            (gctx.move_to if i == 0 else gctx.line_to)(*q)
        gctx.stroke()
    flips = np.load(f"{OUT}/anim.npz")["flips"]
    nflip = int(np.clip(tt / 0.42, 0, 12))
    D.text(ctx, " ".join("R" if f else "L" for f in flips[:nflip]), LX, 330, 30, D.INK, D.fade(t, 1.5, 99, 0.3),
           "semibold")
    a = D.fade(t, 8.0, 99, 0.6)
    if a > 0:
        D.paragraph(ctx, "12 flips make 4,096 equally likely paths. 924 of them end in the middle slot.", LX, 410,
                    30, LW, D.INK, a, "medium")
        D.text(ctx, "924 / 4,096 = 22.6%", LX, 520, 52, D.GOLD, a, "black")
        lv = coin_levels() * min(1.0, (t - 8.0) / 1.5)
        draw_leds(gctx, cm, crop, lv, D.S2, a)
        D.text(ctx, "LED panel: share of chips per slot, 1.25% a cell", LX, 590, 22, D.INK2, a)
    a = D.fade(t, 13.0, 99, 0.6)
    if a > 0:
        ev = STRAT["models"]["coin-flip"]["ev"]
        D.paragraph(ctx, f"Drop in the middle and coin flips pay {money(ev[4])} a chip on average. At the edge: "
                         f"{money(ev[0])}.", LX, 680, 30, LW, D.INK, a, "medium")


def ov_reality(ctx, gctx, t, shot, cm, crop):
    scrim(ctx)
    real = np.array(REAL["totals"]["slots_left_to_right"], float)
    share = real / real.sum()
    head(ctx, "What really happens on the show", t, 0.3, 150,
         f"{int(real.sum()):,} chips, 2015 to 2026, logged by fans at tpirstats.com.", size=50)
    f = D.ease(min(1.0, max(0.0, (t - 1.0) / 1.5)))
    lv = coin_levels() * (1 - f) + real_levels() * f
    draw_leds(gctx, cm, crop, lv, D.S1 if f > 0.5 else D.S2, 1.0)
    if t > 2.5:
        draw_caps(gctx, cm, crop, coin_levels(), D.S2, D.fade(t, 2.5, 99, 0.5), 3)
    a = D.fade(t, 3.0, 99, 0.5)
    # legend
    ctx.rectangle(LX, 300 - 14, 18, 18)
    ctx.set_source_rgba(*D.S1, a)
    ctx.fill()
    D.text(ctx, "The show", LX + 28, 300, 22, D.INK2, a)
    ctx.rectangle(LX + 170, 300 - 6, 26, 3)
    ctx.set_source_rgba(*D.S2, a)
    ctx.fill()
    D.text(ctx, "Coin flips, dropped in the middle", LX + 206, 300, 22, D.INK2, a)
    lines = [(4.0, f"$10,000 slot: {pct(share[4])} of real chips, not 22.6%"),
             (7.0, f"$100 slots: {pct(share[0] + share[8])}. Coin flips allow 3.2%"),
             (10.0, f"Real average: {money(float(real @ B.SLOT_VALUES / real.sum()))} a chip, not $2,558")]
    y = 390
    for ts, s in lines:
        D.paragraph(ctx, s, LX, y, 32, LW, D.INK, D.fade(t, ts, 99, 0.5), "semibold")
        y += 80
    D.paragraph(ctx, "The real board scatters chips much further than coin flips. To see why, I needed real "
                     "physics.", LX, 650, 27, LW, D.INK2, D.fade(t, 12.0, 99, 0.6))


def rain_landed(t):
    shot = [s for s in META["shots"] if s["name"] == "rain"][0]
    counts = np.zeros(9)
    for c, sl in zip(shot["chips"], META["rain_slots"]):
        if t >= c["t0"] + PATHS[c["path"]][-1, 0]:
            counts[sl - 1] += 1
    return counts


def ov_rain(ctx, gctx, t, shot, cm, crop):
    scrim(ctx)
    head(ctx, "So I rebuilt it in a physics engine", t, 0.3, 150)
    items = ["A 5.1 in disc that spins, slides and bounces", "98 round steel pins, sawtooth walls, dividers",
             "Bounce and friction at every contact; pins a little imperfect", "The board leans back 10 degrees"]
    y = 240
    for i, s in enumerate(items):
        a = D.fade(t, 0.8 + 0.5 * i, 99, 0.4)
        ctx.arc(LX + 8, y - 9, 6, 0, 2 * math.pi)
        ctx.set_source_rgba(*D.S3, a)
        ctx.fill()
        D.text(ctx, s, LX + 30, y, 27, D.INK, a)
        y += 44
    cnt = rain_landed(t)
    n = cnt.sum()
    if n > 0:
        draw_leds(gctx, cm, crop, cnt / n / 0.0125, D.S3, 1.0)
    D.text(ctx, f"{int(n):,} simulated chips, all dropped in the middle", LX, 470, 26, D.INK2,
           D.fade(t, 3.5, 99, 0.4))
    a = D.fade(t, 25.0, 99, 0.5)
    if a > 0:
        draw_caps(gctx, cm, crop, real_levels(), (1, 1, 1), a, 3)
        ctx.rectangle(LX, 540 - 6, 26, 3)
        ctx.set_source_rgba(1, 1, 1, a)
        ctx.fill()
        D.text(ctx, "The show's 1,029 real chips", LX + 36, 540, 22, D.INK2, a)
        K = np.load("results/kernel_final.npz")["K"]
        D.paragraph(ctx, f"I tuned the one thing I couldn't measure, the bounce, until the two matched: "
                         f"{pct(K[4, 4])} of simulated chips hit the $10,000 (show: 15.2%).", LX, 610, 29, LW,
                    D.INK, a, "medium")


def ov_question(ctx, gctx, t, shot, cm, crop):
    # the chip sits left of centre here, so the question goes on the right
    g = cairo.LinearGradient(OW, 0, 1180, 0)
    g.add_color_stop_rgba(0, 0.02, 0.025, 0.035, 0.88)
    g.add_color_stop_rgba(0.75, 0.02, 0.025, 0.035, 0.7)
    g.add_color_stop_rgba(1, 0.02, 0.025, 0.035, 0)
    ctx.rectangle(1180, 0, OW - 1180, OH)
    ctx.set_source(g)
    ctx.fill()
    a = D.fade(t, 0.5, 99, 0.6)
    x = 1300
    D.text(ctx, "With micron", x, 400, 62, D.INK, a, "black")
    D.text(ctx, "accuracy, could you", x, 474, 62, D.INK, a, "black")
    D.text(ctx, "hit $10,000", x, 548, 62, D.INK, a, "black")
    D.text(ctx, "every time?", x, 622, 62, D.GOLD, a, "black")
    D.paragraph(ctx, "Two chips, released 0.001 mm apart. A perfect board: no imperfect pins, nothing random.",
                x, 694, 27, 560, D.INK2, D.fade(t, 2.5, 99, 0.5))


def ov_twins(ctx, gctx, t, shot, cm, crop):
    scrim(ctx)
    head(ctx, "0.001 mm apart", t, 0.2, 150, "Same board, same bounce, same everything. Slowed to half speed.")
    ra, rb = PATHS["twin_a"], PATHS["twin_b"]
    tt = max(0.0, (t - 0.6) * 0.5)
    for rec, col in ((rb, D.S1), (ra, D.CHIP)):
        n = int(np.clip(tt * 60, 0, len(rec) - 1))
        gctx.set_line_width(3)
        gctx.set_source_rgba(*col, 0.85)
        for i in range(0, n + 1):
            q = cm.px(board_world(rec[i, 1], rec[i, 2], 0.03), crop)
            (gctx.move_to if i == 0 else gctx.line_to)(*q)
        gctx.stroke()
    m = min(len(ra), len(rb))
    i = int(np.clip(tt * 60, 0, m - 1))
    sep = math.hypot(ra[i, 1] - rb[i, 1], ra[i, 2] - rb[i, 2])
    s = f"{sep * 1e6:,.1f} µm apart" if sep < 1e-3 else (f"{sep * 1e3:.1f} mm apart" if sep < 0.01
                                                                else f"{sep * 100:.0f} cm apart")
    D.text(ctx, s, LX, 380, 64, D.INK, D.fade(t, 0.6, 99, 0.3), "black")
    if tt > max(ra[-1, 0], rb[-1, 0]):
        a = D.fade(t, 0.6 + 2 * max(ra[-1, 0], rb[-1, 0]), 99, 0.4)
        for rec in (ra, rb):
            k = int(round(rec[-1, 1] / W))
            x, y = cm.px(board_world(k * W, B.last_row_y() + 0.2 * W, 0.05), crop)
            D.pill(ctx, D.SLOT_LABELS[k - 1], x, y - 56, 34, D.GOLD if k == 5 else D.INK, alpha=a)
        D.paragraph(ctx, "One wins $10,000. Its twin, released a thousandth of a millimetre away, wins $100.", LX,
                    470, 30, LW, D.INK, a, "medium")


STUCK_COL = (0.86, 0.86, 0.84)


def slot_rgb(sl):
    return VAL_COL[int(B.SLOT_VALUES[sl - 1])] if sl > 0 else STUCK_COL


def barcode(ctx, x0, y0, w, h, slots, alpha=1.0):
    n = len(slots)
    cols = np.array([slot_rgb(s) for s in range(10)])
    img = cols[np.asarray(slots)]                                     # n x 3
    # draw as an image strip, one column per sample, resampled to w pixels
    idx = np.clip((np.arange(int(w)) * n / w).astype(int), 0, n - 1)
    strip = (img[idx] * 255).astype(np.uint8)
    buf = np.zeros((int(h), int(w), 4), np.uint8)
    buf[..., 0], buf[..., 1], buf[..., 2] = strip[:, 2], strip[:, 1], strip[:, 0]
    buf[..., 3] = 255
    s = cairo.ImageSurface.create_for_data(memoryview(np.ascontiguousarray(buf)), cairo.FORMAT_ARGB32, int(w), int(h))
    ctx.set_source_surface(s, x0, y0)
    ctx.paint_with_alpha(alpha)


LEVEL_NAMES = [("9 cm", "the whole drop gap"), ("1 mm", "a grain of sand"), ("10 µm", "a bacterium"),
               ("100 nm", "a virus"), ("1 nm", "a few atoms")]


def ov_zoom(ctx, gctx, t, shot, cm, crop, img=None):
    scrim(ctx, 1920, 0.80)
    D.text(ctx, "Where you let go, and where it lands", LX, 130, 50, D.INK, D.fade(t, 0.3, 99, 0.5), "black")
    D.paragraph(ctx, "Each sliver is one release point on a perfect board, coloured by the slot it lands in.", LX,
                182, 26, 1400, D.INK2, D.fade(t, 0.5, 99, 0.5))
    # legend
    lx = LX
    for col, name in ((VAL_COL[10000], "$10,000"), (VAL_COL[1000], "$1,000"), (VAL_COL[500], "$500"),
                      (VAL_COL[100], "$100"), (VAL_COL[0], "$0"), (STUCK_COL, "balanced on a pin (stuck)")):
        ctx.rectangle(lx, 232, 18, 18)
        ctx.set_source_rgba(*col, D.fade(t, 0.5, 99, 0.5))
        ctx.fill()
        lx += 28 + D.text(ctx, name, lx + 26, 248, 20, D.INK2, D.fade(t, 0.5, 99, 0.5)) + 24
    # the strip first sits across the real 9 cm drop gap on the board, then lifts into a chart;
    # then five zoom levels, 2.5 s each
    bx, by, bw, bh = 120, 330, 1680, 150
    g0 = cm.px(board_world(4.5 * W + B.PEG_DIAMETER / 2, 0.0, 0.03), crop)
    g1 = cm.px(board_world(5.5 * W - B.PEG_DIAMETER / 2, 0.0, 0.03), crop)
    if 1.0 <= t < 4.5:
        f = D.ease((t - 3.5) / 1.0) if t > 3.5 else 0.0
        x0 = g0[0] + (bx - g0[0]) * f
        x1 = g1[0] + (bx + bw - g1[0]) * f
        y0 = g0[1] - 9 + (by - (g0[1] - 9)) * f
        hh = 18 + (bh - 18) * f
        barcode(ctx, x0, y0, x1 - x0, hh, RMAP["s0"], D.fade(t, 1.0, 99, 0.4))
        if t < 3.5:
            D.text(ctx, "every release point across the middle gap", (g0[0] + g1[0]) / 2, g0[1] - 30, 22, D.INK,
                   D.fade(t, 1.3, 3.4, 0.3), "semibold", "center")
    seg = 2.5
    k = int(np.clip((t - 4.5) // seg, 0, 4))
    tl = t - 4.5 - k * seg
    slots = RMAP[f"s{k}"]
    a = D.fade(t, 4.5, 99, 0.2)
    if t >= 4.5:
        barcode(ctx, bx, by, bw, bh, slots, a)
        lev = RMAP_J["levels"][k]
        name, ref = LEVEL_NAMES[k]
        D.text(ctx, f"Width shown: {name}", bx, by + bh + 46, 34, D.INK, a, "black")
        D.text(ctx, f"({ref})", bx + D.text_width(ctx, f"Width shown: {name}", 34, "black") + 16, by + bh + 46, 28,
               D.INK2, a)
        D.text(ctx, f"{pct(lev['p10k'], 0)} of these release points win $10,000. Widest winning stretch: "
                    f"{fmt_len(lev['widest_10k_run_m'])}", bx, by + bh + 90, 26, D.INK2, a)
        if k < 4 and tl > 1.3:          # where the next zoom goes
            nxt = RMAP_J["levels"][k + 1]
            c = (nxt["centre_m"] - lev["centre_m"] + lev["half_width_m"]) / (2 * lev["half_width_m"])
            wfrac = max(2 * nxt["half_width_m"] / (2 * lev["half_width_m"]), 0.002)
            xa = bx + bw * (c - wfrac / 2)
            f = D.ease((tl - 1.3) / 1.1)
            ctx.set_source_rgba(1, 1, 1, 0.9)
            ctx.set_line_width(3)
            ctx.rectangle(xa - 4 - (xa - bx) * f, by - 6, bw * wfrac + 8 + (bw - bw * wfrac) * f, bh + 12)
            ctx.stroke()
    a = D.fade(t, 17.0, 99, 0.5)
    if a > 0:
        D.text(ctx, "Release twin chips this far apart: how often do they land in the same slot?", LX, 680, 28,
               D.INK, a, "medium")
        y = 740
        for (lab, v) in CHAOS_TBL + [("two unrelated chips", 0.122)]:
            D.text(ctx, lab, LX, y, 28, D.INK2 if lab.startswith("two") else D.INK, a)
            D.text(ctx, pct(v, 0), LX + 470, y, 28, D.INK, a, "semibold", "right")
            ctx.rectangle(LX + 500, y - 20, 600 * v, 22)
            ctx.set_source_rgba(*(D.INK3 if lab.startswith("two") else D.S1), a)
            ctx.fill()
            y += 44
    a = D.fade(t, 20.5, 99, 0.5)
    if a > 0:
        D.paragraph(ctx, "No. Micron accuracy is no better than random. Even twins a hundredth of an atom apart share "
                         "a slot only half the time, and heat alone jiggles the pins' atoms more than that.",
                    1220, 760, 27, 600, D.GOLD, a, "semibold")


def fmt_len(m):
    if m is None:
        return "-"
    for unit, f in (("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9), ("pm", 1e-12)):
        if m >= f:
            return f"{m / f:.3g} {unit}"
    return f"{m:.1e} m"


def ov_strategy(ctx, gctx, t, shot, cm, crop):
    scrim(ctx)
    ev_p = np.array(STRAT["models"]["physics tuned"]["ev"])
    ev_c = np.array(STRAT["models"]["coin-flip"]["ev"])
    if t < 10.5:
        head(ctx, "So where should you drop it?", t, 0.3, 150, t1=10.5)
        f = D.ease(min(1, max(0, (t - 0.8) / 1.2)))
        draw_leds(gctx, cm, crop, ev_p / 150 * f, D.GOLD, D.fade(t, 0.8, 10.5, 0.4))
        draw_caps(gctx, cm, crop, ev_c / 150, D.S2, D.fade(t, 2.0, 10.5, 0.4), 3)
        a = D.fade(t, 0.8, 10.5, 0.4)
        ctx.rectangle(LX, 260 - 14, 18, 18)
        ctx.set_source_rgba(*D.GOLD, a)
        ctx.fill()
        D.text(ctx, "Physics, tuned to the show", LX + 28, 260, 22, D.INK2, a)
        ctx.rectangle(LX + 330, 260 - 6, 26, 3)
        ctx.set_source_rgba(*D.S2, a)
        ctx.fill()
        D.text(ctx, "Coin flips", LX + 366, 260, 22, D.INK2, a)
        D.text(ctx, "LED panel now: average win per chip by drop position, $150 a cell", LX, 300, 22, D.INK3, a)
        a = D.fade(t, 3.0, 10.5, 0.5)
        D.paragraph(ctx, f"The middle, in both models. On the tuned board the middle three tie at about "
                         f"{money(ev_p[4])} a chip; the far edge pays {money(ev_p[0])}.", LX, 380, 32, LW, D.INK, a,
                    "semibold")
        D.paragraph(ctx, f"Coin flips say {money(ev_c[4])} against {money(ev_c[0])}. The real board scrambles chips "
                         f"too much for that.", LX, 540, 27, LW, D.INK2, D.fade(t, 5.5, 10.5, 0.5))
    elif t < 17.5:
        tt = t - 10.5
        head(ctx, "How you let go doesn't matter", tt, 0.2, 150, t1=7)
        rows = [("Steady hand (2 mm)", "steady hand (2 mm scatter)"), ("Sloppy hand (3 cm)", "sloppy hand (3 cm scatter)"),
                ("Thrown down", "thrown down at 1 m/s"), ("Spun hard", "spun hard (10 rad/s)"),
                ("Aimed 2 cm off centre", "aimed 2 cm off centre")]
        y = 270
        D.text(ctx, "Average per chip from the middle, 24,000 drops each", LX, 230, 22, D.INK3, D.fade(tt, 0.4, 7, 0.4))
        for i, (lab, key) in enumerate(rows):
            a = D.fade(tt, 0.6 + 0.4 * i, 7, 0.3)
            D.text(ctx, lab, LX, y + 30, 32, D.INK, a, "medium")
            D.text(ctx, money(TECH[key]["ev"]), LX + 600, y + 30, 32, D.INK, a, "semibold", "right")
            D.text(ctx, f"±{money(TECH[key]['se'])}", LX + 616, y + 30, 22, D.INK3, a)
            y += 60
        evs = [v["ev"] for v in TECH.values()]
        D.paragraph(ctx, f"All within {money(max(evs) - min(evs))} of each other, about what chance alone gives. "
                         f"Aim at the slot; don't try to finesse it.", LX, y + 50, 27, LW, D.INK2,
                    D.fade(tt, 3.0, 7, 0.4))
    else:
        tt = t - 17.5
        head(ctx, "What does pay: more chips", tt, 0.2, 150)
        sp = STRAT["small_prizes"]
        five = STRAT["five_chips_calibrated"]["all centre"]
        D.paragraph(ctx, f"Each chip is worth about {money(ev_p[4])}. You win up to four more by saying which digit of "
                         f"a wrong price is right.", LX, 250, 30, LW, D.INK, D.fade(tt, 0.5, 99, 0.4), "medium")
        a = D.fade(tt, 2.0, 99, 0.4)
        D.text(ctx, f"The last digit was right {pct(sp['last_digit_correct_share'], 0)} of the time", LX, 400, 34,
               D.GOLD, a, "black")
        D.text(ctx, f"({sp['items']:,} small prizes, 2015 to 2026). Unsure? Keep the last digit.", LX, 444, 24,
               D.INK2, a)
        a = D.fade(tt, 3.8, 99, 0.4)
        D.text(ctx, "Five chips, all in the middle:", LX, 540, 28, D.INK, a, "medium")
        D.text(ctx, f"{money(five['mean'])} on average", LX, 604, 50, D.INK, a, "black")
        D.text(ctx, f"{pct(five['p_at_least_one_10k'], 0)} chance of at least one $10,000", LX, 656, 28, D.INK2, a)


def ov_outro(ctx, gctx, t, shot, cm, crop):
    path = PATHS["outro"]
    t_land = 1.2 + path[-1, 0]
    if t > t_land:          # the $10,000 sign lights up
        a = D.fade(t, t_land, 99, 0.2) * (0.55 + 0.45 * math.sin((t - t_land) * 5) ** 2)
        yc = FLOOR_Y + 0.03 + 0.16
        pts = [cm.px(board_world(5 * W + dx, yc + dy, 0.104), crop) for dx, dy in
               ((-0.075, -0.035), (0.075, -0.035), (0.075, 0.035), (-0.075, 0.035))]
        quad(gctx, pts)
        gctx.set_source_rgba(*D.GOLD, 0.45 * a)
        gctx.fill()
    a = D.fade(t, t_land + 0.4, 99, 0.5)
    if a > 0:
        scrim(ctx, 760, 0.55 * a)
        D.text(ctx, "Drop it in", 90, 440, 64, D.INK, a, "black")
        D.text(ctx, "the middle.", 90, 514, 64, D.GOLD, a, "black")
        D.paragraph(ctx, "Then let chaos decide.", 92, 580, 30, 500, D.INK2, a, "medium")
    if t > shot["dur"] - 1.2:
        ctx.set_source_rgba(0, 0, 0, D.ease((t - (shot["dur"] - 1.2)) / 1.2))
        ctx.paint()


OVERLAY = dict(open=ov_open, measure=ov_measure, textbook=ov_textbook, reality=ov_reality, rain=ov_rain,
               question=ov_question, twins=ov_twins, zoom=ov_zoom, strategy=ov_strategy, outro=ov_outro)


_vig = None


def vignette():
    """Darken the corners (and the floor spill at the bottom left) like a lens would."""
    global _vig
    if _vig is None:
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        r = np.hypot((xx - OW * 0.55) / (OW * 0.62), (yy - OH * 0.45) / (OH * 0.62))
        v = 1 - 0.42 * np.clip(r - 0.55, 0, 1) ** 1.5 / 0.45 ** 1.5
        bl = np.clip((yy / OH - 0.62) / 0.38, 0, 1) * np.clip(1 - xx / (OW * 0.5), 0, 1)
        _vig = (np.clip(v, 0.45, 1) * (1 - 0.45 * bl))[..., None]
    return _vig


def render(shot, fr):
    t = (fr - 1) / FPS
    img, crop = base_frame(shot, fr, t)
    img = img * vignette()
    cm = cam(shot["cam"])
    if shot["name"] == "zoom":
        img = cv2.GaussianBlur(img, (0, 0), 2 + 6 * D.ease(min(1, t / 1.5))) * 0.55
    if shot["name"] == "question":
        img = img * 0.85
    if shot["name"] == "open":
        img *= D.ease(min(1.0, t / 1.2))
    gsurf, gctx = glow_layer()
    surf, buf = to_surface(np.zeros_like(img))
    # overlays: drawn on a transparent layer, then composited
    osurf = cairo.ImageSurface(cairo.FORMAT_ARGB32, OW, OH)
    octx = cairo.Context(osurf)
    OVERLAY[shot["name"]](octx, gctx, t, shot, cm, crop)
    img = add_glow(img, gsurf)
    osurf.flush()
    o = np.ndarray((OH, OW, 4), np.uint8, osurf.get_data()).astype(np.float32) / 255.0
    img = o[..., :3] + img * (1 - o[..., 3:4])                # cairo is premultiplied BGRA
    # scene edges: a short dip to black between shots
    edge = min(t, shot["dur"] - t)
    if edge < 0.2 and shot["name"] not in ("open",):
        img *= edge / 0.2
    return np.clip(img, 0, 1)


def shot_by_name(n):
    return [s for s in META["shots"] if s["name"] == n][0]


def write_shot(args):
    name, path = args
    shot = shot_by_name(name)
    n = int(round(shot["dur"] * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r",
           str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-profile:v", "high", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in range(1, n + 1):
        img = render(shot, fr)
        p.stdin.write((img * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    return path


def film(workers=4, out="plinko_blender.mp4", only=None, assemble=True):
    jobs = [(s["name"], f"{OUT}/c3d_seg_{i:02d}_{s['name']}.mp4") for i, s in enumerate(META["shots"])]
    todo = [j for j in jobs if (only is None or j[0] in only)]
    with Pool(workers) as pool:
        pool.map(write_shot, todo)
    if not assemble:
        return
    segs = [j[1] for j in jobs]
    with open(f"{OUT}/c3d_segs.txt", "w") as f:
        for s in segs:
            f.write(f"file '{os.path.basename(s)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{OUT}/c3d_segs.txt",
                    "-c", "copy", "-movflags", "+faststart", f"{OUT}/{out}"], check=True)
    print("wrote", f"{OUT}/{out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "still":
        shot = shot_by_name(a[1])
        for tt in a[2:]:
            fr = int(round(float(tt) * FPS)) + 1
            img = render(shot, fr)
            cv2.imwrite(f"{OUT}/c3d_{a[1]}_{float(tt):05.1f}.png", (img * 255).astype(np.uint8))
            print("still", a[1], tt)
    elif a[0] == "film":
        film(int(a[a.index("--workers") + 1]) if "--workers" in a else 4)
    elif a[0] == "segments":            # composite some shots now: segments <names...> [--workers n]
        names = [x for x in a[1:] if not x.startswith("--") and not x.isdigit()]
        film(int(a[a.index("--workers") + 1]) if "--workers" in a else 2, only=names, assemble=False)
    elif a[0] == "assemble":
        film(1, only=[])
