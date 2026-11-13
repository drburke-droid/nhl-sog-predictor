"""The chaos-theory cut: "Same start, different finish". Built on composite3d's layers and the Blender plates.

    python3 composite_chaos.py still <shot> <t> [...]     # PNG checks -> out/cc_<shot>_<t>.png
    python3 composite_chaos.py segments <shots...> [--workers n]
    python3 composite_chaos.py assemble                   # -> out/plinko_chaos.mp4

Every number on screen comes from results/chaos2_*.json (chaos2.py) or the earlier results.
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

import composite3d as C
import draw as D

OUT, FPS, OW, OH = C.OUT, C.FPS, C.OW, C.OH
META = json.load(open(f"{OUT}/anim3d_chaos/shots.json"))
HP = C._load_npz(f"{OUT}/anim3d_chaos/paths.npz")
GROW = json.load(open("results/chaos2_growth.json"))
REP = json.load(open("results/chaos2_replay.json"))
FILM = json.load(open("results/chaos2_film.json"))
EXTRA = json.load(open("results/chaos2_extra.json"))
MAPS = C._load_npz("results/chaos2_maps.npz")
REG = json.load(open("results/chaos2_region.json"))
MSUM = json.load(open("results/chaos2_region_map_summary.json"))
ONE = json.load(open("results/chaos2_onevar.json"))
CUES = json.load(open("results/narration_cues.json"))


def cue(shot, k, j=0, end=False):
    """Film-shot time of narration line k (within the shot), phrase j: when it starts (or ends)."""
    ph = CUES[shot]["lines"][k]["phrases"][j]
    return ph[1] if end else ph[0]
W = C.W
LX, LW = C.LX, C.LW
FAC = GROW["per_bounce_factor"]


def head2(ctx, s, t, sub=None, y=140, size=54, width=1600, t0=0.3):
    """Heading plus a full-width subtitle, for the shots with charts across the frame."""
    a = D.fade(t, t0, 1e9, 0.5)
    D.text(ctx, s, LX, y, size, D.INK, a, "black")
    if sub:
        D.paragraph(ctx, sub, LX, y + size * 0.95, 27, width, D.INK2, a)


def board_label(ctx, cm, crop, k, text, gold=False, alpha=1.0, dy=-56, size=40):
    x, y = cm.px(C.board_world(k * W, C.B.last_row_y() + 0.2 * W, 0.05), crop)
    D.pill(ctx, text, x, y + dy, size, D.GOLD if gold else D.INK, alpha=alpha)


def trail(gctx, cm, crop, rec, n, col, width=3.0, alpha=0.9):
    if n < 1:
        return
    gctx.set_line_width(width)
    gctx.set_source_rgba(*col, alpha)
    for i in range(0, n + 1):
        q = cm.px(C.board_world(rec[i, 1], rec[i, 2], 0.03), crop)
        (gctx.move_to if i == 0 else gctx.line_to)(*q)
    gctx.stroke()


# ----------------------------------------------------------------------------------------------- 1 hook
def ov_hook(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx)
    win, rep = HP["hook_win"], HP["hook_replay"]
    t0w, t0r = 1.0, 7.2
    nw = int(np.clip((t - t0w) * 60, 0, len(win) - 1)) if t > t0w else 0
    nr = int(np.clip((t - t0r) * 60, 0, len(rep) - 1)) if t > t0r else 0
    trail(gctx, cm, crop, win, nw, D.GOLD, 3.5, 0.85)
    trail(gctx, cm, crop, rep, nr, D.S1, 3.0, 0.95)
    if t > t0w + win[-1, 0]:
        board_label(ctx, cm, crop, 5, "$10,000", True, D.fade(t, t0w + win[-1, 0], 99, 0.25))
    if t > t0r + rep[-1, 0]:
        k = int(round(rep[-1, 1] / W))
        board_label(ctx, cm, crop, k, C.D.SLOT_LABELS[k - 1], False, D.fade(t, t0r + rep[-1, 0], 99, 0.25))
    # where the two paths part
    ts = t0r + FILM["hook_split_t"]
    if t > ts:
        i = int(FILM["hook_split_t"] * 60)
        x, y = cm.px(C.board_world(rep[i, 1], rep[i, 2], 0.03), crop)
        a = D.fade(t, ts, ts + 2.5, 0.3)
        r = 18 + 30 * D.ease_out(min(1.0, (t - ts) / 0.8))
        gctx.arc(x, y, r, 0, 2 * math.pi)
        gctx.set_source_rgba(1, 1, 1, 0.8 * a)
        gctx.set_line_width(3)
        gctx.stroke()
    a = D.fade(t, 0.4, 99, 0.5)
    D.text(ctx, "This drop wins $10,000.", LX, 180, 50, D.INK, a, "black")
    a = D.fade(t, cue("hook", 1) - 0.3, 99, 0.5)
    D.paragraph(ctx, "Now the same drop again: same spot, same everything, matched to the width of an atom.", LX, 262,
                30, LW - 40, D.INK, a, "medium")
    for j, (s_, col, y) in enumerate((("Same start.", D.INK, 520), ("Different finish.", D.GOLD, 600),
                                      ("That's chaos.", D.INK, 690))):
        a = D.fade(t, cue("hook", 2, j) - 0.1, 99, 0.4)
        if a > 0:
            D.text(ctx, s_, LX, y, 66 if j < 2 else 54, col, a, "black")


# ----------------------------------------------------------------------------------------------- 2 deterministic
def ov_deterministic(ctx, gctx, t, shot, cm, crop):
    a = D.fade(t, 0.5, 99, 0.6)
    C.scrim(ctx, 800, 0.6 * a)
    a = D.fade(t, cue("deterministic", 0, 1) - 0.2, 99, 0.6)
    D.text(ctx, "There are no dice", 90, 420, 58, D.INK, a, "black")
    D.text(ctx, "inside the board.", 90, 490, 58, D.INK, a, "black")
    D.paragraph(ctx, "Every bounce follows Newton's laws. The same start must give the same finish.", 92, 560, 28,
                560, D.INK2, D.fade(t, cue("deterministic", 0, 2) - 0.2, 99, 0.5), "medium")
    D.paragraph(ctx, "So if the replay lost, its start can't have been quite the same.", 92, 660, 28, 560, D.GOLD,
                D.fade(t, cue("deterministic", 1) - 0.2, 99, 0.5), "semibold")


# ----------------------------------------------------------------------------------------------- 3 twins
def ov_twins(ctx, gctx, t, shot, cm, crop):
    C.ov_twins(ctx, gctx, t, shot, cm, crop)
    a = D.fade(t, cue("twins", 2) - 0.2, 99, 0.6)
    D.text(ctx, "Sensitive dependence on initial conditions", LX, 640, 30, D.GOLD, a, "black")
    D.paragraph(ctx, "A difference far too small to see, and a completely different result.", LX, 690, 26, LW,
                D.INK2, a)


# ----------------------------------------------------------------------------------------------- 3b chaos theory
def _lorenz(s0, T, dt=0.002, sig=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = map(float, s0)
    n = int(round(T / dt))
    out = np.empty((n + 1, 3))
    out[0] = x, y, z
    for i in range(n):
        def f(x, y, z):
            return sig * (y - x), x * (rho - z) - y, x * y - beta * z
        k1 = f(x, y, z)
        k2 = f(x + dt / 2 * k1[0], y + dt / 2 * k1[1], z + dt / 2 * k1[2])
        k3 = f(x + dt / 2 * k2[0], y + dt / 2 * k2[1], z + dt / 2 * k2[2])
        k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        out[i + 1] = x, y, z
    return out


LZ_RATE = 1.5                  # Lorenz time units per second of film, for the two runs on the attractor


def lorenz_data():
    """Lorenz's 1963 equations (sigma 10, rho 28, beta 8/3): the attractor, two 'forecasts' that differ only in
    one starting number (0.506127 vs 0.506), and two runs on the attractor that part when the narration says so."""
    path = f"{OUT}/anim3d_chaos/lorenz.npz"
    if os.path.exists(path):
        return C._load_npz(path)
    bg = _lorenz((1.0, 1.0, 1.0), 110.0)[5000:]
    i = int(np.argmin(np.abs(bg[:, 0] - 0.5) + 10 * (np.diff(bg[:, 0], append=0) < 0)))
    _, y0, z0 = bg[i]
    fa = _lorenz((0.506127, y0, z0), 25.0)
    fb = _lorenz((0.506, y0, z0), 25.0)
    p0 = bg[20000]
    t_part = (cue("chaos", 2, 2) - cue("chaos", 2, 0)) * LZ_RATE
    best = None
    for d0 in np.logspace(-9, -2, 36):
        ra = _lorenz(p0, 14.0)
        rb = _lorenz(p0 + np.array([d0, 0, 0]), 14.0)
        sep = np.linalg.norm(ra - rb, axis=1)
        k = np.argmax(sep > 3.0)
        tc = k * 0.002 if sep[k] > 3.0 else 99.0
        if best is None or abs(tc - t_part) < abs(best[0] - t_part):
            best = (tc, d0)
    T = (CUES["chaos"]["need"] + 2.0 - cue("chaos", 2, 0)) * LZ_RATE
    pa = _lorenz(p0, T)
    pb = _lorenz(p0 + np.array([best[1], 0, 0]), T)
    np.savez_compressed(path, bg=bg[::5], fa=fa, fb=fb, pa=pa, pb=pb, d0=best[1])
    return C._load_npz(path)


LZ = None


def ov_chaos(ctx, gctx, t, shot, cm, crop):
    global LZ
    if LZ is None:
        LZ = lorenz_data()
    C.scrim(ctx, 1920, 0.80)
    head2(ctx, "What is chaos theory?", t)
    # 0: exact rules, no randomness, still unpredictable
    out0 = 1 - D.ease(min(1.0, max(0.0, (t - cue("chaos", 1) + 0.6) / 0.6)))
    for j, (s_, col, y, tc) in enumerate((("Exact rules.", D.INK, 430, cue("chaos", 0, 1)),
                                          ("No randomness at all.", D.INK, 510, cue("chaos", 0, 1) + 1.6),
                                          ("Still unpredictable.", D.GOLD, 590, cue("chaos", 0, 2)))):
        a = D.fade(t, tc, 99, 0.4) * out0
        if a > 0:
            D.text(ctx, s_, LX, y, 60, col, a, "black")
    # 1: Lorenz, 1961: two forecasts that differ in one rounded number
    t1, t2 = cue("chaos", 1), cue("chaos", 2)
    out1 = 1 - D.ease(min(1.0, max(0.0, (t - t2 + 0.8) / 0.7)))
    a = D.fade(t, cue("chaos", 1, 1) - 0.2, 99, 0.4) * out1
    if a > 0:
        D.text(ctx, "1961", LX, 380, 64, D.GOLD, a, "black")
        D.paragraph(ctx, "Edward Lorenz, a meteorologist at MIT, reruns a weather simulation.", LX, 430, 28, 680,
                    D.INK, a, "medium")
        a2 = D.fade(t, cue("chaos", 1, 2), 99, 0.4) * out1
        D.text(ctx, "first run", LX, 560, 22, D.INK3, a2)
        D.text(ctx, "0.506127", LX, 612, 54, D.GOLD, a2, "black")
        D.text(ctx, "rerun, typed from a printout", LX, 670, 22, D.INK3, a2)
        D.text(ctx, "0.506", LX, 722, 54, D.S1, a2, "black")
        a3 = D.fade(t, cue("chaos", 1, 3), 99, 0.4) * out1
        D.paragraph(ctx, "a difference of about 1 part in 4,000", LX, 790, 28, 680, D.INK, a3, "semibold")
        # the two forecasts: one variable of Lorenz's equations over time
        gx, gy, gw, gh = 900, 380, 900, 420
        a4 = D.fade(t, cue("chaos", 1, 2), 99, 0.4) * out1
        f = D.ease(min(1.0, max(0.0, (t - cue("chaos", 1, 2)) / (t2 - 1.0 - cue("chaos", 1, 2)))))
        ctx.new_path()
        ctx.move_to(gx, gy + gh / 2)
        ctx.line_to(gx + gw, gy + gh / 2)
        ctx.set_source_rgba(*D.GRID, a4)
        ctx.set_line_width(1)
        ctx.stroke()
        D.text(ctx, "simulated weather, day after day  →", gx + gw / 2, gy + gh + 36, 20, D.INK3, a4,
               align="center")
        n = len(LZ["fa"])
        for run, col in ((LZ["fb"], D.S1), (LZ["fa"], D.GOLD)):
            m = max(2, int(f * (n - 1)))
            xs = gx + gw * np.arange(0, m, 10) / (n - 1)
            ys = gy + gh / 2 - run[:m:10, 0] / 22.0 * (gh / 2)
            ctx.new_path()
            ctx.move_to(xs[0], ys[0])
            for xx, yy in zip(xs[1:], ys[1:]):
                ctx.line_to(xx, yy)
            ctx.set_source_rgba(*col, a4)
            ctx.set_line_width(3)
            ctx.stroke()
        a5 = D.fade(t, cue("chaos", 1, 4), 99, 0.4) * out1
        D.pill(ctx, "two months in: nothing in common", gx + gw * 0.78, gy - 18, 22, D.INK, alpha=a5,
               weight="semibold")
        D.text(ctx, "Illustrated with Lorenz's 1963 equations", gx + gw, gy + gh + 70, 16, D.INK3, a4,
               align="right")
    # 2: two runs on the attractor, together, then apart
    a = D.fade(t, t2 - 0.4, 99, 0.6) * (1 - D.ease(min(1.0, max(0.0, (t - cue("chaos", 4) + 0.2) / 0.8))))
    if a > 0:
        cx, cy, sx, sy = 1290, 640, 15.5, 10.5
        P = lambda q: (cx + q[..., 0] * sx, cy - (q[..., 2] - 25.0) * sy)
        bx, by = P(LZ["bg"])
        ctx.new_path()
        ctx.move_to(bx[0], by[0])
        for xx, yy in zip(bx[1:4000], by[1:4000]):
            ctx.line_to(xx, yy)
        ctx.set_source_rgba(1, 1, 1, 0.10 * a)
        ctx.set_line_width(1)
        ctx.stroke()
        k = int(max(0.0, t - t2) * LZ_RATE / 0.002)
        for run, col in ((LZ["pb"], D.S1), (LZ["pa"], D.GOLD)):
            k1 = min(k, len(run) - 1)
            k0 = max(0, k1 - int(2.2 / 0.002))
            seg = run[k0:k1 + 1:4]
            if len(seg) > 1:
                xs, ys = P(seg)
                for c0 in range(0, len(xs) - 1, 25):
                    c1 = min(len(xs) - 1, c0 + 26)
                    ctx.new_path()
                    ctx.move_to(xs[c0], ys[c0])
                    for xx, yy in zip(xs[c0 + 1:c1 + 1], ys[c0 + 1:c1 + 1]):
                        ctx.line_to(xx, yy)
                    ctx.set_source_rgba(*col, a * (0.15 + 0.85 * c1 / (len(xs) - 1)))
                    ctx.set_line_width(3)
                    ctx.stroke()
            hx, hy = P(run[k1])
            ctx.new_path()
            ctx.arc(hx, hy, 8, 0, 2 * math.pi)
            ctx.set_source_rgba(*col, a)
            ctx.fill()
        a2 = D.fade(t, t2, 99, 0.4)
        d0 = float(LZ["d0"])
        dtxt = f"{d0:.{max(1, -int(math.floor(math.log10(d0))))}f}"
        D.paragraph(ctx, f"Two runs of Lorenz's equations, starting {dtxt} apart in one number.", LX, 400, 28, 640,
                    D.INK, a2 * a, "medium")
        D.text(ctx, "they move as one", LX, 520, 34, D.INK, D.fade(t, cue("chaos", 2, 1), 99, 0.4) * a, "black")
        D.text(ctx, "then they part, for good", LX, 580, 34, D.GOLD, D.fade(t, cue("chaos", 2, 2), 99, 0.4) * a,
               "black")
        a3 = D.fade(t, cue("chaos", 3) - 0.2, 99, 0.5) * a
        D.text(ctx, "Sensitive dependence", LX, 700, 40, D.GOLD, a3, "black")
        D.text(ctx, "on initial conditions", LX, 748, 40, D.GOLD, a3, "black")
        a4 = D.fade(t, cue("chaos", 3, 1) - 0.1, 99, 0.5) * a
        D.text(ctx, "the butterfly effect", LX, 820, 34, D.INK, a4, "black")
        D.paragraph(ctx, "Lorenz's 1972 talk: “Does the flap of a butterfly's wings in Brazil set off a tornado "
                         "in Texas?”", LX, 870, 22, 660, D.INK2, a4)
    # 4: and Plinko
    a = D.fade(t, cue("chaos", 4) - 0.1, 99, 0.5)
    if a > 0:
        D.text(ctx, "A Plinko board is a perfect example.", 960, 560, 56, D.INK, a, "black", "center")


# ----------------------------------------------------------------------------------------------- 4 mechanism
def _hit(p, d, c, r):
    """Distance along ray p + s d to circle (c, r), or None."""
    f = p - c
    b = f @ d
    q = f @ f - r * r
    disc = b * b - q
    if disc < 0:
        return None
    s_ = -b - math.sqrt(disc)
    return s_ if s_ > 1e-6 else None


def _trace(p, d, circles, L):
    """A ray reflecting off circles (perfect mirror), up to total length L: list of points, and the hits."""
    pts, hits = [p.copy()], []
    left = L
    while left > 0:
        best = None
        for c, r in circles:
            s_ = _hit(p, d, c, r)
            if s_ is not None and (best is None or s_ < best[0]):
                best = (s_, c, r)
        if best is None or best[0] > left:
            pts.append(p + d * left)
            break
        s_, c, r = best
        p = p + d * s_
        nrm = (p - c) / r
        hits.append((p.copy(), nrm.copy(), len(pts)))
        d = d - 2 * (d @ nrm) * nrm
        pts.append(p.copy())
        left -= s_
    return pts, hits


def _draw_path(ctx, pts, upto, col, alpha, width=3.0):
    """Polyline pts drawn up to arc length `upto`."""
    ctx.new_path()
    ctx.move_to(*pts[0])
    left = upto
    for a_, b_ in zip(pts[:-1], pts[1:]):
        seg = float(np.linalg.norm(b_ - a_))
        if left >= seg:
            ctx.line_to(*b_)
            left -= seg
        else:
            q = a_ + (b_ - a_) * (left / seg)
            ctx.line_to(*q)
            break
    ctx.set_source_rgba(*col, alpha)
    ctx.set_line_width(width)
    ctx.stroke()


def _arrow(ctx, p, d, L, col, alpha, width=2.5):
    q = p + d * L
    ctx.new_path()
    ctx.move_to(*p)
    ctx.line_to(*q)
    nx_, ny_ = -d[1], d[0]
    ctx.move_to(*q)
    ctx.line_to(q[0] - d[0] * 12 + nx_ * 7, q[1] - d[1] * 12 + ny_ * 7)
    ctx.move_to(*q)
    ctx.line_to(q[0] - d[0] * 12 - nx_ * 7, q[1] - d[1] * 12 - ny_ * 7)
    ctx.set_source_rgba(*col, alpha)
    ctx.set_line_width(width)
    ctx.stroke()


R_PIN = 110.0
C_PIN = np.array([1300.0, 430.0])


def _zigzag(t_frac, s0=6.0, grow=2.2):
    """A falling path bending past three pins, drawn as two paths whose gap grows x2.2 past every pin."""
    V = [np.array(v, float) for v in ((1150, 620), (1290, 710), (1160, 800), (1300, 890), (1170, 980))]
    rs = 26.0
    pins = []
    for k in range(1, len(V) - 1):
        d_in = (V[k] - V[k - 1]) / np.linalg.norm(V[k] - V[k - 1])
        d = (V[k + 1] - V[k]) / np.linalg.norm(V[k + 1] - V[k])
        out = d_in - d
        out /= np.linalg.norm(out)
        pins.append((V[k], out, k))
    # each path has its own corners, side by side; the gap between them grows x2.2 at every pin
    gaps = [s0, s0] + [s0 * grow ** k for k in range(1, len(V) - 1)]
    gaps = gaps[:len(V)]
    paths = [[V[k] + np.array([sg * gaps[k] / 2, 0.0]) for k in range(len(V))] for sg in (-1, 1)]
    pins = [(V[k] + out * (rs + gaps[k] / 2 + 8), k) for V_k, out, k in pins]
    return V, pins, paths, gaps, rs


def ov_mechanism(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx, 1920, 0.84)
    head2(ctx, "Why so touchy? Round pins.", t)
    g = 9.0
    # left: a flat wall
    fa = D.fade(t, cue("mechanism", 1) - 0.3, 99, 0.4) * (1 - D.ease(min(1.0, max(0.0, (t - cue("mechanism", 4)
                                                                                         + 0.4) / 0.6))))
    if fa > 0:
        wy, wx0, wx1 = 690.0, 170.0, 830.0
        ctx.new_path()
        ctx.move_to(wx0, wy)
        ctx.line_to(wx1, wy)
        ctx.set_source_rgba(0.85, 0.86, 0.9, fa)
        ctx.set_line_width(6)
        ctx.stroke()
        for xx in np.arange(wx0, wx1, 22):
            ctx.move_to(xx, wy + 4)
            ctx.line_to(xx - 14, wy + 22)
        ctx.set_source_rgba(0.6, 0.62, 0.66, 0.6 * fa)
        ctx.set_line_width(1.5)
        ctx.stroke()
        D.text(ctx, "a flat wall", (wx0 + wx1) / 2, wy + 58, 24, D.INK, fa, "semibold", "center")
        din = np.array([math.cos(math.radians(55)), math.sin(math.radians(55))])
        dout = np.array([din[0], -din[1]])
        prog = 1400 * D.ease(min(1.0, max(0.0, (t - cue("mechanism", 1)) / 3.0)))
        for k, col in ((0, D.GOLD), (1, D.S1)):
            h = np.array([500.0 + k * g / math.sin(math.radians(55)), wy - 3])
            _draw_path(ctx, [h - din * 380, h, h + dout * 380], prog, col, fa)
        an = D.fade(t, cue("mechanism", 1, 2) - 0.2, 99, 0.4) * fa
        if an > 0:
            for k in (0, 1):
                h = np.array([500.0 + k * g / math.sin(math.radians(55)), wy - 3])
                _arrow(ctx, h, np.array([0.0, -1.0]), 80, (1, 1, 1), an)
            D.paragraph(ctx, "It faces the same way everywhere, so the two leave exactly as far apart as they came "
                             "in.", 170, 800, 26, 660, D.INK, an, "medium")
    # right: a round pin
    ra = D.fade(t, cue("mechanism", 2) - 0.3, 99, 0.4)
    if ra > 0:
        c1 = C_PIN
        ctx.new_path()
        ctx.arc(c1[0], c1[1], R_PIN, 0, 2 * math.pi)
        ctx.set_source_rgba(0.55, 0.57, 0.62, 0.45 * ra)
        ctx.fill()
        ctx.arc(c1[0], c1[1], R_PIN, 0, 2 * math.pi)
        ctx.set_source_rgba(0.85, 0.86, 0.9, ra)
        ctx.set_line_width(3)
        ctx.stroke()
        D.text(ctx, "a round pin", c1[0], c1[1] + 8, 24, D.INK, ra, "semibold", "center")
        t_in, t_out = cue("mechanism", 2), cue("mechanism", 2, 2)
        prog = 205 * D.ease(min(1.0, max(0.0, (t - t_in) / 1.4)))
        prog += 420 * D.ease(min(1.0, max(0.0, (t - t_out) / 2.4)))
        hits = []
        for k, col in ((0, D.GOLD), (1, D.S1)):
            p0 = np.array([c1[0] + 0.74 * R_PIN + k * g, c1[1] - 200.0])
            pts, hh = _trace(p0, np.array([0.0, 1.0]), [(c1, R_PIN)], 640)
            hits.append(hh[0])
            _draw_path(ctx, pts, prog, col, ra)
        an = D.fade(t, cue("mechanism", 2, 1) - 0.2, 99, 0.4)
        if an > 0:
            for h in hits:
                _arrow(ctx, h[0], h[1], 80, (1, 1, 1), an)
            D.paragraph(ctx, "but a curve faces a different way at every spot", 1450, 270, 24, 360, D.INK, an,
                        "medium")
        ad = D.fade(t, cue("mechanism", 2, 3) - 0.2, 99, 0.4)
        D.paragraph(ctx, "so they bounce off in different directions", 1450, 610, 24, 360, D.INK, ad, "medium")
        D.text(ctx, "two paths a hair apart (drawn much wider)", c1[0] + 90, c1[1] - 215, 18, D.INK3, ra,
               align="center")
    # the next pin does it again: the gap grows at every pin
    az = D.fade(t, cue("mechanism", 3) - 0.2, 99, 0.4)
    if az > 0:
        V, pins, paths, gaps, rs = _zigzag(0)
        f = D.ease(min(1.0, max(0.0, (t - cue("mechanism", 3)) / 4.0)))
        for (c, k) in pins:
            ctx.new_path()
            ctx.arc(c[0], c[1], rs, 0, 2 * math.pi)
            ctx.set_source_rgba(0.85, 0.86, 0.9, az * D.fade(t, cue("mechanism", 3) + 0.8 * (k - 1), 99, 0.3))
            ctx.fill()
        for pts, col in zip(paths, (D.GOLD, D.S1)):
            total = sum(float(np.linalg.norm(q1 - q0)) for q0, q1 in zip(pts[:-1], pts[1:]))
            _draw_path(ctx, pts, total * f, col, az, 2.5)
        for (c, k) in pins:
            at = az * D.fade(t, cue("mechanism", 3) + 0.8 * k, 99, 0.3)
            D.text(ctx, f"gap x {FAC ** k:.0f}" if k > 1 else f"gap x {FAC:.1f}", c[0] + (40 if c[0] > 1220 else -40),
                   c[1] + 8, 20, D.INK, at, "semibold", "left" if c[0] > 1220 else "right")
        D.paragraph(ctx, "The further they travel, the wider the gap. Then the next pin widens it again.", 1520,
                    800, 24, 340, D.INK, D.fade(t, cue("mechanism", 3) + 0.4, 99, 0.4), "medium")
    # the multiplier, bounce after bounce (takes the flat wall's place)
    ta = cue("mechanism", 4)
    a = D.fade(t, ta - 0.2, 99, 0.5)
    if a > 0:
        xs = 120
        D.text(ctx, f"In the simulator, a bounce multiplies the gap by about {FAC:.1f}", xs, 360, 28, D.INK, a,
               "semibold")
        rows = [(1, "1 bounce"), (5, "5 bounces"), (10, "10 bounces"), (20, "20 bounces"),
                (round(GROW["bounces_per_drop"]), f"a whole drop ({round(GROW['bounces_per_drop'])} bounces)")]
        y = 440
        for i, (n_, lab) in enumerate(rows):
            tr = ta + 0.6 + 0.7 * i if i < 4 else cue("mechanism", 4, 1)
            ar = D.fade(t, tr, 99, 0.35)
            v = FAC ** n_
            s_ = f"x {v:,.0f}" if v < 1e6 else (f"x {v / 1e6:,.0f} million" if v < 1e9 else
                                                 f"x {v / 1e9:,.0f} billion")
            D.text(ctx, lab, xs, y, 30, D.INK2, ar, "medium")
            D.text(ctx, s_, xs + 720, y, 34, D.GOLD if i == 4 else D.INK, ar, "black", "right")
            y += 62
        a2 = D.fade(t, cue("mechanism", 5) - 0.2, 99, 0.5)
        D.paragraph(ctx, f"Doubling, over and over: the gap doubles about every "
                         f"{1000 * GROW['doubling_time_s']:.0f} milliseconds. That's exponential growth, the engine "
                         f"of chaos.", xs, 790, 28, 720, D.GOLD, a2, "semibold")


# ----------------------------------------------------------------------------------------------- 5 horizon
def ov_horizon(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx, 1920, 0.82)
    head2(ctx, "How far ahead can you see?", t,
          "Know the chip's start to within this, and you can follow it for about this many bounces.")
    hb = FILM["horizon_bounces"]
    rows = [("1 mm", hb["1 mm"]), ("1 micron", hb["1 um"]), ("1 nanometre", hb["1 nm"]), ("1 atom (0.1 nm)", hb["1 atom"]),
            ("an atomic nucleus", hb["an atomic nucleus"])]
    x0, bw = LX + 420, 1150
    full = 38.0
    y = 360
    whole = GROW["bounces_per_drop"]
    a0 = D.fade(t, 1.0, 99, 0.4)
    xw = x0 + bw * whole / full
    ctx.move_to(xw, 330)
    ctx.line_to(xw, 360 + 5 * 78 - 20)
    ctx.set_source_rgba(*D.INK3, a0)
    ctx.set_line_width(1.5)
    ctx.stroke()
    D.text(ctx, f"a whole drop: about {whole:.0f} bounces", xw - 8, 320, 20, D.INK3, a0, align="right")
    for i, (lab, n) in enumerate(rows):
        a = D.fade(t, 1.4 + 0.8 * i, 99, 0.35)
        f = D.ease_out(min(1.0, max(0.0, (t - 1.4 - 0.8 * i) / 0.9)))
        D.text(ctx, lab, LX, y + 34, 30, D.INK, a, "medium")
        ctx.rectangle(x0, y + 8, bw * min(n, full) / full * f, 34)
        ctx.set_source_rgba(*(D.GOLD if i == 4 else D.S1), a)
        ctx.fill()
        D.text(ctx, f"{n:.0f}", x0 + bw * min(n, full) / full * f + 14, y + 36, 26, D.INK, a, "semibold")
        y += 78
    a = D.fade(t, 6.5, 99, 0.5)
    D.paragraph(ctx, f"Every 1,000-fold gain in precision buys only about {math.log(1000) / math.log(FAC):.0f} more "
                     f"bounces. To call a whole drop you'd need the start to about "
                     f"{FILM['whole_drop_precision_m'] * 1e15:.0f} femtometres: the size of an atomic nucleus.",
                LX, 820, 30, 1500, D.INK, a, "medium")


# ----------------------------------------------------------------------------------------------- 6 crux
PREC_LABEL = {1e-2: "1 cm", 1e-3: "1 mm", 1e-6: "1 µm", 1e-9: "1 nm", 1e-10: "1 atom", 1e-12: "1 pm",
              1e-15: "1 fm"}


def ov_crux(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx, 1920, 0.84)
    head2(ctx, "Find a drop that wins. Replay it.", t,
          "Winning drops from anywhere across the top, each replayed from exactly the same spot, as precisely as "
          "you like. How often do they win again?", width=1700)
    d = np.array([float(k) for k in REG["position"]])
    p = np.array(list(REG["position"].values()))
    base = REG["base_rate"]
    gx, gy, gw, gh = 230, 330, 1440, 470
    lx_ = lambda dd: gx + gw * (-2 - math.log10(dd)) / 13.0       # 1 cm (left) ... 1 fm (right)
    ly_ = lambda pp: gy + gh * (1 - pp)
    a = D.fade(t, 1.0, 99, 0.4)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        ctx.move_to(gx, ly_(v))
        ctx.line_to(gx + gw, ly_(v))
        ctx.set_source_rgba(*D.GRID, a)
        ctx.set_line_width(1)
        ctx.stroke()
        D.text(ctx, f"{100 * v:.0f}%", gx - 14, ly_(v), 18, D.INK3, a, align="right", anchor="middle")
    for dd, lab in PREC_LABEL.items():
        D.text(ctx, lab, lx_(dd), gy + gh + 34, 20, D.INK2, a, align="center")
    D.text(ctx, "replay precision  →  finer", gx + gw / 2, gy + gh + 70, 20, D.INK3, a, align="center")
    # a fresh drop
    ctx.move_to(gx, ly_(base))
    ctx.line_to(gx + gw, ly_(base))
    ctx.set_source_rgba(*D.INK3, a)
    ctx.set_line_width(2)
    ctx.stroke()
    D.text(ctx, f"a fresh drop: {100 * base:.0f}%", gx + gw - 4, ly_(base) + 26, 20, D.INK3, a, align="right")
    # the replay curve, drawn left to right
    f = D.ease(min(1.0, max(0.0, (t - 2.5) / 6.0)))
    xmax = gx + gw * f
    ctx.set_line_width(3)
    ctx.set_source_rgba(*D.GOLD, a)
    pts = [(lx_(dd), ly_(pp)) for dd, pp in zip(d, p)]
    started = False
    for (x, y) in pts:
        if x > xmax:
            break
        (ctx.line_to if started else ctx.move_to)(x, y)
        started = True
    ctx.stroke()
    for (x, y) in pts:
        if x <= xmax:
            ctx.arc(x, y, 5, 0, 2 * math.pi)
            ctx.set_source_rgba(*D.GOLD, a)
            ctx.fill()
    for dd, lab, ts in ((1e-6, "1 µm", 5.0), (1e-10, "1 atom", 7.0), (1e-15, "1 fm", 8.6)):
        if t > ts:
            pp = REG["position"][f"{dd:g}"]
            D.pill(ctx, f"{lab}: {100 * pp:.0f}%", lx_(dd) - (40 if dd == 1e-15 else 0), ly_(pp) - 40, 24, D.INK,
                   alpha=D.fade(t, ts, 99, 0.3), weight="semibold")
    pos = REG["position"]
    a = D.fade(t, 10.5, 99, 0.5)
    D.paragraph(ctx, f"Replayed to a micron, it wins again {100 * pos['1e-06']:.0f}% of the time, barely better than a "
                     f"fresh drop. To an atom: {100 * pos['1e-10']:.0f}%, still worse than a coin flip. To a "
                     f"hundred-thousandth of an atom, 1 replay in {1 / (1 - pos['1e-15']):.0f} still misses.",
                LX, 940, 28, 1700, D.INK, a, "medium")
    a = D.fade(t, 18.0, 99, 0.5)
    D.paragraph(ctx, f"Below a nanometre, each tenfold gain in precision removes only about "
                     f"{100 * (1 - REG['miss_cut_per_10x']):.0f}% of the misses left. Certainty never arrives.",
                LX, 1030, 28, 1700, D.GOLD, a, "semibold")


# ----------------------------------------------------------------------------------------------- 7 islands
ISL_LEVELS = [("2 mm", "a grain of sand"), ("20 \u00b5m", "a fifth of a hair"), ("200 nm", "a virus"),
              ("2 nm", "a few atoms"), ("20 pm", "a fifth of an atom")]
GOLD_C = C.VAL_COL[10000]
GREY_C = (0.34, 0.35, 0.39)
OVER = C._load_npz("results/chaos2_overview.npz")


def strip_bin(ctx, x, y, w, h, slots, reveal=1.0, alpha=1.0):
    """One sliver per release point (nearest sample per pixel): gold if it wins $10,000, grey otherwise."""
    if w < 1 or h < 1 or alpha <= 0:
        return
    w_i, h_i = max(1, int(round(w))), max(1, int(round(h)))
    n = len(slots)
    idx = np.clip(((np.arange(w_i) + 0.5) * n / w_i).astype(int), 0, n - 1)
    win = np.asarray(slots)[idx] == 5
    cols = np.where(win[:, None], np.array(GOLD_C), np.array(GREY_C))
    buf = np.zeros((h_i, w_i, 4), np.uint8)
    rgb = (cols * 255).astype(np.uint8)
    buf[..., 0], buf[..., 1], buf[..., 2] = rgb[:, 2], rgb[:, 1], rgb[:, 0]
    buf[..., 3] = 255
    buf[:, int(w_i * max(0.0, min(1.0, reveal))):] = 0
    s = cairo.ImageSurface.create_for_data(memoryview(np.ascontiguousarray(buf)), cairo.FORMAT_ARGB32, w_i, h_i)
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(w / w_i, h / h_i)
    ctx.set_source_surface(s, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_NEAREST)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


def zoom_lines(ctx, x0, x1, ytop, X0, X1, ybot, alpha):
    """A magnifier: a bracket over [x0, x1] on one strip, opened out to [X0, X1] on the strip below."""
    ctx.set_source_rgba(1, 1, 1, 0.10 * alpha)
    ctx.move_to(x0, ytop)
    ctx.line_to(x1, ytop)
    ctx.line_to(X1, ybot)
    ctx.line_to(X0, ybot)
    ctx.close_path()
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, 0.85 * alpha)
    ctx.set_line_width(1.5)
    ctx.move_to(x0, ytop)
    ctx.line_to(X0, ybot)
    ctx.move_to(x1, ytop)
    ctx.line_to(X1, ybot)
    ctx.stroke()
    ctx.set_line_width(2.5)
    ctx.move_to(x0, ytop - 8)
    ctx.line_to(x0, ytop)
    ctx.line_to(x1, ytop)
    ctx.line_to(x1, ytop - 8)
    ctx.stroke()


def tag(ctx, s, x, y, size=20, col=D.INK, alpha=1.0, weight="semibold", align="center"):
    w = D.text_width(ctx, s, size, weight)
    if align == "center":
        x = min(max(x, 40 + w / 2), 1880 - w / 2)
    xl = x - w / 2 if align == "center" else (x - w if align == "right" else x)
    C.D.rounded_rect(ctx, xl - 8, y - size - 4, w + 16, size + 12, 6)
    ctx.set_source_rgba(0.03, 0.035, 0.05, 0.85 * alpha)
    ctx.fill()
    D.text(ctx, s, x, y, size, col, alpha, weight, align)


_MAP = {}


def region_surface():
    """The whole-top release map as a cairo surface, 1 px per grid point: gold wins $10,000, grey doesn't, dark
    where a chip can't be let go (inside a wall or a top pin)."""
    if "s" not in _MAP:
        m = np.load("results/chaos2_region_map.npz")
        sl = m["slots"]
        rgb = np.where((sl == 5)[..., None], np.array(GOLD_C), np.array(GREY_C))
        buf = np.zeros(sl.shape + (4,), np.uint8)
        c = (rgb * 255).astype(np.uint8)
        buf[..., 0], buf[..., 1], buf[..., 2], buf[..., 3] = c[..., 2], c[..., 1], c[..., 0], 255
        buf[sl < 0] = (20, 15, 13, 255)
        buf = np.ascontiguousarray(buf)
        _MAP["buf"] = buf
        _MAP["s"] = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, sl.shape[1], sl.shape[0])
        _MAP["gold"] = float((sl == 5).sum() / (sl >= 0).sum())
    return _MAP["s"]


ISL_SHIFT = 7.5          # the strip-and-zoom part starts this much later than in the earlier cut


def ov_islands(ctx, gctx, t, shot, cm, crop):
    GAP = FILM["release_half_range_m"]
    TOP = REG["top_m"]
    C.scrim(ctx, 1920, 0.84)
    head2(ctx, "Every place you could let go", t,
          "A player can let go anywhere across the top: right at the top pins, or up to 30 cm (12 in) above "
          "them, wall to wall. Each dot is one release point. Gold: that drop wins $10,000. Grey: it lands "
          "anywhere else.", y=110, size=48, width=1700)
    al = D.fade(t, 0.8, 99, 0.4)
    lx = 1800
    legend = [(GREY_C, "lands anywhere else"), (GOLD_C, "wins $10,000")]
    if t < 13.4:
        legend.insert(0, ((0.05, 0.06, 0.08), "no room for the chip"))
    for col, name in legend:
        w = D.text_width(ctx, name, 20, "medium")
        D.text(ctx, name, lx, 104, 20, D.INK2, al, "medium", "right")
        ctx.new_path()
        ctx.rectangle(lx - w - 28, 88, 18, 18)
        ctx.set_source_rgba(*col, al)
        ctx.fill_preserve()
        ctx.set_source_rgba(1, 1, 1, 0.35 * al)
        ctx.set_line_width(1)
        ctx.stroke()
        lx -= w + 64
    # --- phase 0: the whole top, as a map of release points
    surf = region_surface()
    mw_i, mh_i = surf.get_width(), surf.get_height()
    mx, my, mw = 120, 300, 1680
    mh = mw * mh_i / mw_i
    xm = lambda x: mx + mw * (x - 0.5 * W) / (9 * W)               # board x -> map px
    ma = D.fade(t, 0.8, 99, 0.4) * (1 - D.ease(min(1.0, max(0.0, (t - 12.6) / 1.0))))
    if ma > 0:
        sweep0 = D.ease(min(1.0, max(0.0, (t - 1.0) / 3.5)))
        ctx.save()
        ctx.rectangle(mx, my, mw * sweep0, mh)
        ctx.clip()
        ctx.translate(mx, my)
        ctx.scale(mw / mw_i, mh / mh_i)
        ctx.set_source_surface(surf, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        ctx.paint_with_alpha(ma)
        ctx.restore()
        for k in range(1, 9):                                           # the top row of pins
            ctx.new_path()
            ctx.arc(xm((k + 0.5) * W), my + mh, max(3.0, mw * C.B.PEG_DIAMETER / 2 / (9 * W)), 0, 2 * math.pi)
            ctx.set_source_rgba(0.85, 0.87, 0.9, ma)
            ctx.fill()
        ctx.new_path()
        ctx.rectangle(mx, my, mw, mh)
        ctx.set_source_rgba(1, 1, 1, 0.35 * ma)
        ctx.set_line_width(1.5)
        ctx.stroke()
        D.text(ctx, f"{100 * TOP:.0f} cm", mx - 10, my + 16, 18, D.INK3, ma, align="right")
        D.text(ctx, "above", mx - 10, my + 38, 16, D.INK3, ma, align="right")
        D.text(ctx, "pins", mx - 10, my + mh, 18, D.INK3, ma, align="right")
        D.text(ctx, "wall", mx, my + mh + 30, 16, D.INK3, ma, align="center")
        D.text(ctx, "wall", mx + mw, my + mh + 30, 16, D.INK3, ma, align="center")
        pa = ma * (1 - D.ease(min(1.0, max(0.0, (t - 8.3) / 0.5))))
        for k in range(1, 10):
            D.text(ctx, f"{k}", xm(k * W), my + mh + 30, 20, D.INK2, pa, "semibold", "center")
        ca = D.fade(t, 4.6, 8.5, 0.4) * ma
        if ca > 0:
            D.paragraph(ctx, f"{100 * MSUM['gold']:.0f}% of all release points win $10,000, sprinkled over the whole "
                             f"top: {100 * MSUM['middle']:.0f}% of the spots over the middle win, "
                             f"{100 * MSUM['walls']:.0f}% of those by the walls. Up close it's pure speckle: move a "
                             f"hair, and the prize changes.",
                        120, 740, 30, 1680, D.INK, ca, "medium")
    # --- phase 1: one line through it, the middle gap right at the pins, blown up into a strip
    ox, oy, ow, oh = ov_rect = (120, 262, 1680, 46)
    big_y = 880
    bx0, bx1 = xm(5 * W - GAP), xm(5 * W + GAP)
    ba = D.fade(t, 8.5, 13.4, 0.4)
    if ba > 0:
        ctx.new_path()
        ctx.rectangle(bx0 - 2, my + mh - 6, bx1 - bx0 + 4, 8)
        ctx.set_source_rgba(1, 1, 1, ba)
        ctx.set_line_width(2)
        ctx.stroke()
        tag(ctx, "one line through it: the middle gap, right at the pins", (bx0 + bx1) / 2, my + mh + 48, 22,
            D.INK, ba)
        zoom_lines(ctx, bx0, bx1, my + mh + 60, ox, ox + ow, big_y, ba * D.fade(t, 9.4, 99, 0.3))
    sweep = D.ease(min(1.0, max(0.0, (t - 9.6) / 2.6)))
    lift = D.ease(min(1.0, max(0.0, (t - 12.8) / 1.2)))
    sy = big_y + (oy - big_y) * lift
    if t > 9.6:
        strip_bin(ctx, ox, sy, ow, oh, OVER["slots"], sweep if lift <= 0 else 1.0, D.fade(t, 9.6, 99, 0.3))
    t = t - ISL_SHIFT
    # --- phase 2: the overview with a scale, and two winning spots
    oa = D.fade(t, 6.2, 99, 0.4)
    if oa > 0:
        xx = ox + ow / 2
        ctx.move_to(xx, oy + oh)
        ctx.line_to(xx, oy + oh + 8)
        ctx.set_source_rgba(*D.INK3, oa)
        ctx.set_line_width(1.5)
        ctx.stroke()
        D.text(ctx, "centre", xx, oy + oh + 28, 18, D.INK3, oa, align="center")
        D.text(ctx, f"\u2212{100 * GAP:.1f} cm", ox - 10, oy + oh / 2 + 7, 18, D.INK3, oa, align="right")
        D.text(ctx, f"+{100 * GAP:.1f} cm", ox + ow + 10, oy + oh / 2 + 7, 18, D.INK3, oa)
    spots = sorted([("knife", FILM["knife_x"]), ("robust", FILM["robust_x"])], key=lambda s: s[1])
    names = {"knife": "the winning drop from the start", "robust": "another winning drop"}
    cols = {}
    colx = {0: (120, 890), 1: (1030, 1800)}
    for ci, (key, x) in enumerate(spots):
        cols[key] = ci
        mx = ox + ow * (x + GAP) / (2 * GAP)
        a = D.fade(t, 7.0 + 0.6 * ci, 99, 0.4)
        if a <= 0:
            continue
        ctx.move_to(mx, oy - 4)
        ctx.line_to(mx - 9, oy - 18)
        ctx.line_to(mx + 9, oy - 18)
        ctx.close_path()
        ctx.set_source_rgba(1, 1, 1, a)
        ctx.fill()
        tag(ctx, f"Spot {ci + 1}: {names[key]}", mx, oy - 26, 20, D.INK, a)
    for t0, t1, msg in ((8.0, 12.9, "Both spots win $10,000. Now zoom in on each, a hundredfold at a time."),
                        (13.3, 15.5, "Down to the size of a virus, both spots look like any random drop: mostly grey."),
                        (15.9, 20.9, f"Zoom to a few atoms across and spot {cols['robust'] + 1} turns gold. "
                                     f"Spot {cols['knife'] + 1} never does.")):
        a = D.fade(t, t0, t1, 0.4)
        if a > 0:
            D.paragraph(ctx, msg, 120, 1010, 30, 1680, D.INK, a, "medium")
    # --- phase 3: zoom ladders, a hundredfold per row
    rows_y = [372, 484, 596, 708, 820]
    t_row = [9.0 + 2.2 * k for k in range(5)]
    for key, x in spots:
        ci = cols[key]
        x0c, x1c = colx[ci]
        cw = x1c - x0c
        mx = ox + ow * (x + GAP) / (2 * GAP)
        half0 = 1e-3 / (2 * GAP) * ow                 # the 2 mm window on the overview
        for k in range(5):
            if t < t_row[k]:
                break
            f = D.ease(min(1.0, (t - t_row[k]) / 0.9))
            y = rows_y[k]
            if k == 0:
                zoom_lines(ctx, mx - half0, mx + half0, oy + oh, x0c + cw / 2 - cw / 2 * f, x0c + cw / 2 + cw / 2 * f,
                           y, D.fade(t, t_row[k], 99, 0.3))
            else:
                yp = rows_y[k - 1] + 40
                c = x0c + cw / 2
                zoom_lines(ctx, c - cw * 0.005 - 3, c + cw * 0.005 + 3, yp, c - cw / 2 * f, c + cw / 2 * f, y,
                           D.fade(t, t_row[k], 99, 0.3))
            sl = MAPS[f"{key}{k}"]
            strip_bin(ctx, x0c + cw / 2 - cw / 2 * f, y, cw * f, 40, sl, 1.0, D.fade(t, t_row[k] + 0.3, 99, 0.3))
            a = D.fade(t, t_row[k] + 0.6, 99, 0.3)
            ctx.move_to(x0c + cw / 2, y - 6)
            ctx.line_to(x0c + cw / 2, y + 46)
            ctx.set_source_rgba(1, 1, 1, 0.9 * a)
            ctx.set_line_width(1.5)
            ctx.stroke()
            D.text(ctx, f"{100 * float((sl == 5).mean()):.0f}% gold", x1c, y + 64, 18, D.INK, a, "semibold", "right")
            if ci == 0:
                nm, ref = ISL_LEVELS[k]
                D.text(ctx, nm, 960, y + 22, 24, D.INK, a, "black", "center")
                D.text(ctx, ref, 960, y + 46, 15, D.INK3, a, align="center")
    # --- phase 4: what the two ladders show
    a = D.fade(t, 21.2, 99, 0.5)
    if a > 0:
        for key, title, col in (("robust", "an island, solid gold within a few atoms", GOLD_C),
                                ("knife", "a knife edge, still mixed at a fifth of an atom", D.S1)):
            x0c, x1c = colx[cols[key]]
            tag(ctx, f"Spot {cols[key] + 1}: {title}", (x0c + x1c) / 2, 950, 24, col, a, "black")
        n_isl = round(100 * REG["share_robust"])
        D.paragraph(ctx, f"Only {n_isl} in 100 winning spots are islands like spot {cols['robust'] + 1}. "
                         f"One win can't tell you which kind you found.", 120, 1020, 30, 1680, D.INK,
                    D.fade(t, 22.5, 99, 0.5), "semibold")


# ----------------------------------------------------------------------------------------------- 8 one variable
def deg(dT):
    """0.000057 -> '0.00006 °C'."""
    d = max(0, -int(math.floor(math.log10(dT))))
    return f"{dT:.{d}f} \u00b0C"


HEAT = None


def heat_data():
    """The first drop's energy, sample by sample (60 per second of drop): what's left as height, what's speed,
    and what has already become heat; and the pins it hit."""
    rec = HP["warm_win_full"]
    tau, x, y, ang = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    m, R = 0.25, C.B.CHIP_DIAMETER / 2
    inertia = 0.6 * m * R * R
    ct = math.cos(math.radians(C.B.BOARD_TILT_DEG))
    vx, vy, w = np.gradient(x, tau), np.gradient(y, tau), np.gradient(ang, tau)
    e_tot = ONE["drop_energy_J"]
    pe_lost = np.clip(m * 9.81 * (y - y[0]) * ct, 0, e_tot)
    ke = 0.5 * m * (vx * vx + vy * vy) + 0.5 * inertia * w * w
    heat = np.clip(np.maximum.accumulate(np.clip(pe_lost - ke, 0, None)), 0, e_tot)
    heat[-1] = e_tot
    speed = np.clip(pe_lost - heat, 0, None)
    speed[-1] = 0.0
    height = e_tot - pe_lost
    pegs, _ = C.B.pegs()
    hits = []
    for i in np.flatnonzero(rec[:, 4] > 0):
        d = np.hypot(pegs[:, 0] - x[i], pegs[:, 1] - y[i])
        j = int(np.argmin(d))
        if d[j] < R + C.B.PEG_DIAMETER / 2 + 0.02:
            hits.append((float(tau[i]), float(pegs[j, 0]), float(pegs[j, 1])))
    return dict(tau=tau, height=height, speed=speed, heat=heat, hits=hits, e_tot=e_tot)


def fmt_len(v):
    for unit, k in (("cm", 1e-2), ("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9)):
        if v >= k:
            return f"{v / k:.0f} {unit}" if v / k >= 10 else f"{v / k:.1f} {unit}"
    return f"{v * 1e9:.1f} nm"


def ov_heat(ctx, gctx, t, shot, cm, crop):
    global HEAT
    if HEAT is None:
        HEAT = heat_data()
    C.scrim(ctx)
    slow = shot["slowmo"]
    t_fall = shot["chips"][0]["t0"]
    tau_now = max(0.0, (t - t_fall) * slow)
    k = int(np.searchsorted(HEAT["tau"], tau_now))
    k = min(k, len(HEAT["tau"]) - 1)
    # titles
    D.text(ctx, "Make the start perfect.", LX, 180, 50, D.INK, D.fade(t, 0.3, 99, 0.5), "black")
    D.text(ctx, "Change just one thing:", LX, 244, 50, D.INK, D.fade(t, cue("heat", 0, 1) - 0.2, 99, 0.5),
           "black")
    D.text(ctx, "the board's temperature.", LX, 308, 50, D.GOLD, D.fade(t, cue("heat", 0, 2) - 0.2, 99, 0.5),
           "black")
    a = D.fade(t, cue("heat", 1) - 0.2, 99, 0.4)
    w_ = D.text(ctx, "Where would heat come from?", LX, 390, 30, D.INK, a, "semibold")
    D.text(ctx, "The chips themselves.", LX + w_ + 14, 390, 30, D.GOLD,
           D.fade(t, cue("heat", 1, 1) - 0.2, 99, 0.4), "semibold")
    # one chip's energy: height -> speed -> heat
    a = D.fade(t, cue("heat", 2) - 0.2, 99, 0.4)
    if a > 0:
        e = HEAT["e_tot"]
        parts = [(HEAT["height"][k], D.S1, "height"), (HEAT["speed"][k], D.INK, "speed"),
                 (HEAT["heat"][k], D.S2, "heat")]
        if tau_now <= 0:
            parts = [(e, D.S1, "height"), (0.0, D.INK, "speed"), (0.0, D.S2, "heat")]
        D.text(ctx, f"One chip's energy: {e:.1f} joules", LX, 470, 24, D.INK2, a, "semibold")
        bx, by, bw, bh = LX, 488, 680, 34
        x = bx
        for v, col, _ in parts:
            wv = bw * v / e
            ctx.new_path()
            ctx.rectangle(x, by, wv, bh)
            ctx.set_source_rgba(*col, a)
            ctx.fill()
            x += wv
        ctx.new_path()
        ctx.rectangle(bx, by, bw, bh)
        ctx.set_source_rgba(1, 1, 1, 0.4 * a)
        ctx.set_line_width(1.5)
        ctx.stroke()
        x = bx
        for v, col, lab in parts:
            x += 8 + D.text(ctx, f"{lab} {v:.1f} J", x, by + bh + 32, 22, col, a, "semibold") + 28
    # glows where the chip hits pins: each impact leaves heat behind
    warm_all = D.ease(min(1.0, max(0.0, (t - cue("heat", 4)) / 2.0)))
    for tau_i, px, py in HEAT["hits"]:
        t_i = t_fall + tau_i / slow
        if t < t_i:
            continue
        q = cm.px(C.board_world(px, py, 0.03), crop)
        fresh = math.exp(-(t - t_i) / 0.6)
        rad = 9 + 9 * (1 - fresh) + 6 * warm_all
        g_ = cairo.RadialGradient(q[0], q[1], 0, q[0], q[1], rad)
        alpha = (0.22 + 0.5 * fresh) * (1 - 0.5 * warm_all)
        g_.add_color_stop_rgba(0, 1.0, 0.42, 0.08, alpha)
        g_.add_color_stop_rgba(1, 0.9, 0.2, 0.02, 0.0)
        gctx.new_path()
        gctx.arc(q[0], q[1], rad, 0, 2 * math.pi)
        gctx.set_source(g_)
        gctx.fill()
    # the heat spreads through the whole board
    if warm_all > 0:
        Wd = C.W
        pts = [cm.px(C.board_world(xx, yy, 0.0), crop) for xx, yy in
               ((0.5 * Wd, -0.05), (9.5 * Wd, -0.05), (9.5 * Wd, C.B.last_row_y() + 0.1),
                (0.5 * Wd, C.B.last_row_y() + 0.1))]
        C.quad(ctx, pts)
        ctx.set_source_rgba(*D.S2, 0.16 * warm_all)
        ctx.fill()
    a = D.fade(t, cue("heat", 3) - 0.2, 99, 0.4)
    D.text(ctx, "Stopped: every joule is now heat.", LX, 610, 26, D.S2, a, "semibold")
    a = D.fade(t, cue("heat", 4) - 0.1, 99, 0.4)
    D.paragraph(ctx, f"Spread through a ~{ONE['board_kg']:.0f} kg board: {deg(ONE['drop_dT_C'])} warmer.", LX, 670,
                26, LW, D.INK, a, "medium")
    # warm wood and steel expand: every pin moves, away from where the board is held
    a = D.fade(t, cue("heat", 4, 1) - 0.1, 99, 0.4)
    if a > 0:
        Wd = C.W
        pegs, rows = C.B.pegs()
        for (px, py), r in zip(pegs, rows):
            if r % 3 != 0 or r == 0:
                continue
            dvec = np.array([px - 5 * Wd, py])
            dvec /= np.linalg.norm(dvec)
            p0 = np.array(cm.px(C.board_world(px, py, 0.03), crop))
            p1 = np.array(cm.px(C.board_world(px + dvec[0] * 0.06, py + dvec[1] * 0.06, 0.03), crop))
            dd = p1 - p0
            L = float(np.linalg.norm(dd))
            _arrow(ctx, p0, dd / L, L, D.S2, a, 2.2)
        D.paragraph(ctx, f"The board grows about {1e9 * ONE['drop_expansion_m']:.0f} nm, the width of ten atoms. "
                         "Every pin shifts with it.", LX, 740, 26, LW, D.INK,
                    D.fade(t, cue("heat", 4, 2) - 0.1, 99, 0.4), "medium")
        D.text(ctx, "(arrows drawn 100 million times bigger than the real shift)", LX, 812, 18, D.INK3, a)
    # a nanometre, doubled by every bounce
    a = D.fade(t, cue("heat", 5) - 0.1, 99, 0.4)
    if a > 0:
        f = D.ease(min(1.0, max(0.0, (t - cue("heat", 5, 1)) / (cue("heat", 5, 2) - cue("heat", 5, 1)))))
        n = int(round(21 * f))
        v = 1e-9 * FAC ** n if n < 21 else 1e-2
        D.text(ctx, f"after {n:2d} bounces:", LX, 870, 30, D.INK2, a, "medium")
        D.text(ctx, fmt_len(v), LX + 300, 870, 40, D.GOLD if n >= 21 else D.INK, a, "black")
        D.text(ctx, "a different slot", LX, 930, 34, D.GOLD, D.fade(t, cue("heat", 5, 3) - 0.1, 99, 0.4), "black")


def ov_warm(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx)
    win, rep = HP["warm_win"], HP["warm_replay"]
    slow, t0 = shot["slowmo"], 2.5
    tt = max(0.0, (t - t0) * slow)
    for rec, col, wd in ((win, D.GOLD, 3.5), (rep, D.S1, 3.0)):
        trail(gctx, cm, crop, rec, int(np.clip(tt * 60, 0, len(rec) - 1)), col, wd, 0.9)
    ts = t0 + ONE["pair_split_t"] / slow
    if t > ts:
        i = int(ONE["pair_split_t"] * 60)
        x, y = cm.px(C.board_world(rep[i, 1], rep[i, 2], 0.03), crop)
        a = D.fade(t, ts, ts + 2.5, 0.3)
        gctx.arc(x, y, 18 + 30 * D.ease_out(min(1.0, (t - ts) / 0.8)), 0, 2 * math.pi)
        gctx.set_source_rgba(1, 1, 1, 0.8 * a)
        gctx.set_line_width(3)
        gctx.stroke()
    adjacent = abs(round(win[-1, 1] / W) - round(rep[-1, 1] / W)) == 1
    for rec, gold in ((win, True), (rep, False)):
        tl = t0 + rec[-1, 0] / slow
        if t > tl:
            k = int(round(rec[-1, 1] / W))
            board_label(ctx, cm, crop, k, C.D.SLOT_LABELS[k - 1], gold, D.fade(t, tl, 99, 0.25),
                        dy=-124 if (gold and adjacent) else -56)
    dT = deg(ONE["drop_dT_C"])
    D.text(ctx, "Same drop.", LX, 180, 50, D.INK, D.fade(t, 0.3, 99, 0.5), "black")
    D.text(ctx, "Warmer board.", LX, 244, 50, D.GOLD, D.fade(t, 0.9, 99, 0.5), "black")
    D.paragraph(ctx, f"Exactly the same release, to the last atom. The only difference: the board is {dT} "
                     f"warmer, from the heat of the first drop.", LX, 314, 28, LW - 20, D.INK,
                D.fade(t, 1.6, 99, 0.5), "medium")
    a = D.fade(t, cue("warm", 0, 1) - 0.2, 99, 0.4)
    y = 470
    for col, lab in ((D.GOLD, "the first drop"), (D.S1, f"the replay, board {dT} warmer")):
        ctx.new_path()
        ctx.arc(LX + 10, y - 8, 9, 0, 2 * math.pi)
        ctx.set_source_rgba(*col, a)
        ctx.fill()
        D.text(ctx, lab, LX + 32, y, 24, D.INK2, a, "medium")
        y += 40
    D.paragraph(ctx, "They fall as one, until the shifted pins catch up with them.", LX, 600, 26, LW - 20,
                D.INK, D.fade(t, cue("warm", 1) - 0.2, 99, 0.4), "medium")
    a = D.fade(t, cue("warm", 2) - 0.1, 99, 0.5)
    if a > 0:
        k = int(round(rep[-1, 1] / W))
        D.text(ctx, f"{dT} warmer:", LX, 760, 44, D.INK, a, "black")
        D.text(ctx, f"$10,000 became {C.D.SLOT_LABELS[k - 1]}.", LX, 818, 44, D.GOLD, a, "black")


def ov_realworld(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx, 1920, 0.84)
    head2(ctx, "How much warmer is too warm?", t,
          f"A perfect release, everything else fixed. Warm the board, then replay {REG['warm_n_winners']} winning drops "
          f"from across the top, exactly. How many still win?", width=1700)
    dts = np.array(REG["warm_dT_C"])
    p = np.array(REG["warm_wins_again"])
    base = REG["base_rate"]
    gx, gy, gw, gh = 230, 330, 1440, 470
    lx_ = lambda d: gx + gw * (math.log10(d) + 10) / 10.0          # 1e-10 C (left) ... 1 C (right)
    ly_ = lambda v: gy + gh * (1 - v)
    a = D.fade(t, 1.0, 99, 0.4)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        ctx.move_to(gx, ly_(v))
        ctx.line_to(gx + gw, ly_(v))
        ctx.set_source_rgba(*D.GRID, a)
        ctx.set_line_width(1)
        ctx.stroke()
        D.text(ctx, f"{100 * v:.0f}%", gx - 14, ly_(v), 18, D.INK3, a, align="right", anchor="middle")
    for d, lab in ((1e-9, "a billionth"), (1e-6, "a millionth"), (1e-3, "a thousandth"), (1.0, "1 degree")):
        D.text(ctx, lab, lx_(d), gy + gh + 34, 20, D.INK2, a, align="center")
    D.text(ctx, "how much warmer the board is (\u00b0C)  \u2192", gx + gw / 2, gy + gh + 70, 20, D.INK3, a,
           align="center")
    ctx.move_to(gx, ly_(base))
    ctx.line_to(gx + gw, ly_(base))
    ctx.set_source_rgba(*D.INK3, a)
    ctx.set_line_width(2)
    ctx.stroke()
    D.text(ctx, f"a fresh drop: {100 * base:.0f}%", gx + gw - 4, ly_(base) + 26, 20, D.INK3, a, align="right")
    # one drop's heat
    xd = lx_(ONE["drop_dT_C"])
    ad = D.fade(t, 6.5, 99, 0.4)
    if ad > 0:
        ctx.set_dash([6, 6])
        ctx.move_to(xd, gy)
        ctx.line_to(xd, gy + gh)
        ctx.set_source_rgba(*D.S1, ad)
        ctx.set_line_width(2)
        ctx.stroke()
        ctx.set_dash([])
        D.text(ctx, "the heat of one drop", xd, gy - 12, 20, D.S1, ad, "semibold", "center")
    f = D.ease(min(1.0, max(0.0, (t - 2.5) / 5.0)))
    xmax = gx + gw * f
    pts = [(lx_(d), ly_(v)) for d, v in zip(dts, p)]
    ctx.set_line_width(3)
    ctx.set_source_rgba(*D.GOLD, a)
    started = False
    for (x, y) in pts:
        if x > xmax:
            break
        (ctx.line_to if started else ctx.move_to)(x, y)
        started = True
    ctx.stroke()
    for (x, y) in pts:
        if x <= xmax:
            ctx.arc(x, y, 5, 0, 2 * math.pi)
            ctx.set_source_rgba(*D.GOLD, a)
            ctx.fill()
    pick = {f"{d:.3g}": v for d, v in zip(dts, p)}
    for d, ts, dy in ((1e-9, 4.0, -40), (ONE["drop_dT_C"], 7.0, -40), (1.0, 8.0, -40)):
        if t > ts:
            v = pick[f"{d:.3g}"]
            D.pill(ctx, f"{100 * v:.0f}%", lx_(d) - (30 if d == 1.0 else 0), ly_(v) + dy, 24, D.INK,
                   alpha=D.fade(t, ts, 99, 0.3), weight="semibold")
    v_drop = pick[f"{ONE['drop_dT_C']:.3g}"]
    a = D.fade(t, cue("realworld", 0, 1), 99, 0.5)
    D.paragraph(ctx, f"Even a billionth of a degree spoils {100 * (1 - pick['1e-09']):.0f} replays in 100, and one "
                     f"drop's heat spoils {100 * (1 - v_drop):.0f}. No change is too small "
                     f"to matter.",
                LX, 925, 28, 1700, D.INK, a, "medium")
    a = D.fade(t, cue("realworld", 1), 99, 0.5)
    D.paragraph(ctx, "And every drop warms the board: in practice, a replay is barely better than a fresh drop.",
                LX, 985, 28, 1700, D.GOLD, a, "semibold")


# ----------------------------------------------------------------------------------------------- 9 statistics
def ov_statistics(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx)
    C.head(ctx, "One drop can't be predicted.", t, 0.3, 150, size=50)
    D.text(ctx, "Four hundred can.", LX, 218, 50, D.GOLD, D.fade(t, 1.5, 99, 0.5), "black")
    trn = t + shot.get("frame_offset", 0) / FPS
    cnt = C.rain_landed(trn)
    n = cnt.sum()
    if n > 0:
        C.draw_leds(gctx, cm, crop, cnt / n / 0.0125, D.S3, 1.0)
    D.text(ctx, f"{int(n):,} simulated chips, all dropped in the middle", LX, 300, 26, D.INK2, D.fade(t, 1.0, 99, 0.4))
    D.paragraph(ctx, "Chaos scrambles every single path, but the pile they make is steady: drop after drop, the same "
                     "bell-shaped spread.", LX, 380, 28, LW, D.INK, D.fade(t, 5.0, 99, 0.5), "medium")
    a = D.fade(t, 15.0, 99, 0.5)
    if a > 0:
        C.draw_caps(gctx, cm, crop, C.real_levels(), (1, 1, 1), a, 3)
        ctx.rectangle(LX, 540 - 6, 26, 3)
        ctx.set_source_rgba(1, 1, 1, a)
        ctx.fill()
        D.text(ctx, "The show's 1,029 real chips", LX + 36, 540, 22, D.INK2, a)
        ev = C.STRAT["models"]["physics tuned"]["ev"]
        D.paragraph(ctx, f"That's why the middle still pays best: about {C.money(ev[4])} a chip, against "
                         f"{C.money(ev[0])} at the edge.", LX, 610, 29, LW, D.INK, a, "semibold")


# ----------------------------------------------------------------------------------------------- 10 outro
def ov_outro(ctx, gctx, t, shot, cm, crop):
    path = C.PATHS["outro"]
    t_land = 1.2 + path[-1, 0]
    if t > t_land:
        a = D.fade(t, t_land, 99, 0.2) * (0.55 + 0.45 * math.sin((t - t_land) * 5) ** 2)
        yc = C.FLOOR_Y + 0.19
        pts = [cm.px(C.board_world(5 * W + dx, yc + dy, 0.104), crop) for dx, dy in
               ((-0.075, -0.035), (0.075, -0.035), (0.075, 0.035), (-0.075, 0.035))]
        C.quad(gctx, pts)
        gctx.set_source_rgba(*D.GOLD, 0.45 * a)
        gctx.fill()
    a = D.fade(t, cue("outro", 0, 1) - 0.2, 99, 0.5)
    if a > 0:
        C.scrim(ctx, 760, 0.55 * a)
        for j, (s_, col, y) in enumerate((("Deterministic.", D.INK, 430), ("Unpredictable.", D.INK, 504),
                                          ("That's chaos.", D.GOLD, 578))):
            D.text(ctx, s_, 90, y, 62, col, D.fade(t, cue("outro", 1, j) - 0.1, 99, 0.4), "black")
        D.paragraph(ctx, "Drop it in the middle, and let chaos decide.", 92, 640, 26, 520, D.INK2, a)
    if t > shot["dur"] - 1.2:
        ctx.set_source_rgba(0, 0, 0, D.ease((t - (shot["dur"] - 1.2)) / 1.2))
        ctx.paint()


OVERLAY = dict(hook=ov_hook, deterministic=ov_deterministic, twins=ov_twins, mechanism=ov_mechanism,
               horizon=ov_horizon, crux=ov_crux, islands=ov_islands, heat=ov_heat, warm=ov_warm, realworld=ov_realworld, chaos=ov_chaos,
               statistics=ov_statistics, outro=ov_outro)
DARK = {"chaos": (8.0, 0.35, 0.0), "mechanism": (6.0, 0.45, 0.0), "horizon": (6.0, 0.5, 0.0), "islands": (6.0, 0.5, 0.0),
        "crux": (5.0, 0.6, 0.0), "realworld": (5.0, 0.6, 0.0)}


def render(shot, fr):
    t = (fr - 1) / FPS
    img, crop = C.base_frame(shot, fr, t)
    img = img * C.vignette()
    cm = C.cam(shot["cam"])
    if shot["name"] in DARK:
        blur, dim, t0 = DARK[shot["name"]]
        f = D.ease(min(1.0, max(0.0, (t - t0) / 1.2)))
        if t0 > 0:                                  # keep the board sharp until the effect starts
            img = img * (0.8 + (dim - 0.8) * f)
            if f > 0:
                img = cv2.GaussianBlur(img, (0, 0), 0.3 + blur * f)
        else:
            img = cv2.GaussianBlur(img, (0, 0), 2 + blur * f) * dim
    if shot["name"] == "hook":
        img *= D.ease(min(1.0, t / 1.0))
    gsurf, gctx = C.glow_layer()
    osurf = cairo.ImageSurface(cairo.FORMAT_ARGB32, OW, OH)
    octx = cairo.Context(osurf)
    OVERLAY[shot["name"]](octx, gctx, t, shot, cm, crop)
    img = C.add_glow(img, gsurf)
    osurf.flush()
    o = np.ndarray((OH, OW, 4), np.uint8, osurf.get_data()).astype(np.float32) / 255.0
    img = o[..., :3] + img * (1 - o[..., 3:4])
    edge = min(t, shot["dur"] - t)
    if edge < 0.2 and shot["name"] != "hook":
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
        p.stdin.write((render(shot, fr) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    return path


def seg_path(i, name):
    return f"{OUT}/cc_seg_{i:02d}_{name}.mp4"


def segments(names, workers=2):
    jobs = [(s["name"], seg_path(i, s["name"])) for i, s in enumerate(META["shots"]) if s["name"] in names]
    with Pool(workers) as pool:
        pool.map(write_shot, jobs)


def assemble(out="plinko_chaos.mp4"):
    lst = f"{OUT}/cc_segs.txt"
    with open(lst, "w") as f:
        for i, s in enumerate(META["shots"]):
            f.write(f"file '{os.path.basename(seg_path(i, s['name']))}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", f"{OUT}/{out}"], check=True)
    print("wrote", f"{OUT}/{out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "still":
        shot = shot_by_name(a[1])
        for tt in a[2:]:
            img = render(shot, int(round(float(tt) * FPS)) + 1)
            cv2.imwrite(f"{OUT}/cc_{a[1]}_{float(tt):05.1f}.png", (img * 255).astype(np.uint8))
            print("still", a[1], tt)
    elif a[0] == "segments":
        names = [x for x in a[1:] if not x.startswith("--") and not x.isdigit()]
        segments(names, int(a[a.index("--workers") + 1]) if "--workers" in a else 2)
    elif a[0] == "assemble":
        assemble()
