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
    a = D.fade(t, 6.0, 99, 0.5)
    D.paragraph(ctx, "Now the same drop again: same spot, same everything, matched to the width of an atom.", LX, 262,
                30, LW - 40, D.INK, a, "medium")
    a = D.fade(t, 12.0, 99, 0.6)
    if a > 0:
        D.text(ctx, "Same start.", LX, 520, 66, D.INK, a, "black")
        D.text(ctx, "Different finish.", LX, 600, 66, D.GOLD, a, "black")
        D.paragraph(ctx, "A short film about chaos, told with a Plinko board.", LX, 670, 28, LW, D.INK2,
                    D.fade(t, 13.0, 99, 0.6))


# ----------------------------------------------------------------------------------------------- 2 deterministic
def ov_deterministic(ctx, gctx, t, shot, cm, crop):
    a = D.fade(t, 0.5, 99, 0.6)
    C.scrim(ctx, 800, 0.6 * a)
    D.text(ctx, "There are no dice", 90, 420, 58, D.INK, a, "black")
    D.text(ctx, "inside the board.", 90, 490, 58, D.INK, a, "black")
    D.paragraph(ctx, "Every bounce follows Newton's laws. The same start must give the same finish.", 92, 560, 28,
                560, D.INK2, D.fade(t, 3.0, 99, 0.5), "medium")
    D.paragraph(ctx, "So if the replay lost, its start can't have been quite the same.", 92, 660, 28, 560, D.GOLD,
                D.fade(t, 6.5, 99, 0.5), "semibold")


# ----------------------------------------------------------------------------------------------- 3 twins
def ov_twins(ctx, gctx, t, shot, cm, crop):
    C.ov_twins(ctx, gctx, t, shot, cm, crop)
    a = D.fade(t, 9.5, 99, 0.6)
    D.text(ctx, "Sensitive dependence on initial conditions", LX, 640, 30, D.GOLD, a, "black")
    D.paragraph(ctx, "Edward Lorenz hit it in 1961, rerunning a weather model with 0.506 typed in for 0.506127: "
                     "the forecast came out completely different. It became known as the butterfly effect.", LX, 690,
                26, LW, D.INK2, a)


# ----------------------------------------------------------------------------------------------- 4 mechanism
def ov_mechanism(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx, 1920, 0.82)
    head2(ctx, "Why so touchy? Round pins.", t,
          "A chip bouncing off a round pin is like light off a convex mirror: two paths that arrive almost "
          "together leave further apart.", width=1500)
    # diagram: two parallel paths hitting a pin (the circle the chip's centre can't enter)
    cx, cy, R = 470, 700, 130
    a = D.fade(t, 1.2, 99, 0.4)
    ctx.arc(cx, cy, R, 0, 2 * math.pi)
    ctx.set_source_rgba(0.55, 0.56, 0.6, 0.35 * a)
    ctx.fill()
    ctx.arc(cx, cy, 16, 0, 2 * math.pi)
    ctx.set_source_rgba(0.8, 0.81, 0.85, a)
    ctx.fill()
    D.text(ctx, "pin", cx, cy + 40, 20, D.INK3, a, align="center")
    for off, col in ((52, D.GOLD), (62, D.S1)):
        x0 = cx + off
        yh = cy - math.sqrt(R * R - off * off)
        f_in = D.ease(min(1.0, max(0.0, (t - 1.6) / 1.4)))
        y_in = 400 + (yh - 400) * f_in
        ctx.move_to(x0, 400)
        ctx.line_to(x0, y_in)
        ctx.set_source_rgba(*col, a)
        ctx.set_line_width(3)
        ctx.stroke()
        if t > 3.0:
            nx, ny = (x0 - cx) / R, (yh - cy) / R
            dx, dy = 0.0, 1.0
            dd = dx * nx + dy * ny
            rx, ry = dx - 2 * dd * nx, dy - 2 * dd * ny
            L = 420 * D.ease(min(1.0, (t - 3.0) / 1.5))
            ctx.move_to(x0, yh)
            ctx.line_to(x0 + rx * L, yh + ry * L)
            ctx.stroke()
    D.paragraph(ctx, "almost together going in", cx + 70, 420, 22, 260, D.INK2, D.fade(t, 2.0, 99, 0.4))
    D.paragraph(ctx, "much further apart coming out", cx + 220, 820, 22, 300, D.INK2, D.fade(t, 4.3, 99, 0.4))
    # the multiplier, bounce after bounce
    xs = 1040
    a = D.fade(t, 6.0, 99, 0.5)
    D.text(ctx, f"In the simulator, a typical bounce multiplies the gap by about {FAC:.1f}", xs, 360, 28, D.INK, a,
           "semibold")
    rows = [(1, "1 bounce"), (5, "5 bounces"), (10, "10 bounces"), (20, "20 bounces"),
            (round(GROW["bounces_per_drop"]), "a whole drop")]
    y = 430
    for i, (n, lab) in enumerate(rows):
        a = D.fade(t, 7.0 + 0.9 * i, 99, 0.35)
        v = FAC ** n
        s = f"x {v:,.0f}" if v < 1e6 else (f"x {v / 1e6:,.0f} million" if v < 1e9 else f"x {v / 1e9:,.0f} billion")
        D.text(ctx, lab, xs, y, 30, D.INK2, a, "medium")
        D.text(ctx, s, xs + 640, y, 34, D.GOLD if i == 4 else D.INK, a, "black", "right")
        y += 58
    a = D.fade(t, 12.5, 99, 0.5)
    D.paragraph(ctx, f"The gap doubles about every {1000 * GROW['doubling_time_s']:.0f} milliseconds. Growth like "
                     f"that is exponential, and it is the signature of chaos.", xs, 760, 27, 700, D.INK2, a)


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
          "From exactly the same spot, as precisely as you like. How often does it win again?")
    d = np.array([float(k) for k in REP["position"]])
    p = np.array(list(REP["position"].values()))
    base = REP["base_rate"]
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
    for dd, lab, ts in ((1e-6, "1 µm: 57%", 5.5), (1e-10, "1 atom: 71%", 7.0), (1e-15, "96%", 8.6)):
        if t > ts:
            pp = REP["position"][f"{dd:g}"]
            D.pill(ctx, f"{lab.split(':')[0]}: {100 * pp:.0f}%" if ":" in lab else f"{100 * pp:.0f}%",
                   lx_(dd), ly_(pp) - 40, 24, D.INK, alpha=D.fade(t, ts, 99, 0.3), weight="semibold")
    a = D.fade(t, 10.5, 99, 0.5)
    D.paragraph(ctx, f"Atomic accuracy wins again {100 * REP['position']['1e-10']:.0f}% of the time: much better than "
                     f"a fresh drop, and still not a sure thing. At a hundred-thousandth of an atom, 1 replay in "
                     f"{1 / (1 - REP['position']['1e-15']):.0f} still misses.", LX, 940, 28, 1700, D.INK, a, "medium")
    a = D.fade(t, 18.0, 99, 0.5)
    D.paragraph(ctx, f"Each tenfold gain in precision cuts the misses by only about {1 / FILM['miss_cut_per_10x']:.2f} "
                     f"times. Certainty never arrives.", LX, 1020, 28, 1700, D.GOLD, a, "semibold")


# ----------------------------------------------------------------------------------------------- 7 islands
ISL_LEVELS = [("2 mm", "a grain of sand"), ("20 µm", "a fifth of a hair"), ("200 nm", "a virus"),
              ("2 nm", "a few atoms"), ("20 pm", "a fifth of an atom")]


def ov_islands(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx, 1920, 0.84)
    head2(ctx, "Why never certain? Two kinds of winning spot.", t,
          "Each strip is the release points around one winning spot, coloured by where they land. Zoom in on each.",
          y=130)
    lx = LX
    al = D.fade(t, 0.8, 99, 0.4)
    for col, name in ((C.VAL_COL[10000], "$10,000"), (C.VAL_COL[1000], "$1,000"), (C.VAL_COL[500], "$500"),
                      (C.VAL_COL[100], "$100"), (C.VAL_COL[0], "$0"), (C.STUCK_COL, "stuck")):
        ctx.rectangle(lx, 232, 18, 18)
        ctx.set_source_rgba(*col, al)
        ctx.fill()
        lx += 28 + D.text(ctx, name, lx + 26, 248, 20, D.INK2, al) + 24
    seg = 2.8
    k = int(np.clip((t - 1.5) // seg, 0, 4))
    a = D.fade(t, 1.5, 99, 0.3)
    bx, bw, bh = 120, 1680, 110
    for row, (name, title, y) in enumerate((("robust", "An island", 330), ("knife", "A knife edge", 560))):
        if t < 1.5:
            continue
        sl = MAPS[f"{name}{k}"]
        C.barcode(ctx, bx, y, bw, bh, sl, a)
        ctx.move_to(bx + bw / 2, y - 10)
        ctx.line_to(bx + bw / 2, y + bh + 10)
        ctx.set_source_rgba(1, 1, 1, 0.9 * a)
        ctx.set_line_width(2)
        ctx.stroke()
        gold = float((sl == 5).mean())
        D.text(ctx, title, bx, y - 20, 30, D.GOLD if row == 0 else D.S1, a, "black")
        D.text(ctx, f"{100 * gold:.0f}% gold", bx + bw, y - 20, 24, D.INK, a, "semibold", "right")
    if t >= 1.5:
        nm, ref = ISL_LEVELS[k]
        D.text(ctx, f"Width shown: {nm}", 120, 760, 32, D.INK, a, "black")
        D.text(ctx, f"({ref})", 120 + D.text_width(ctx, f"Width shown: {nm}", 32, "black") + 14, 760, 26, D.INK2, a)
    a = D.fade(t, 16.0, 99, 0.5)
    D.paragraph(ctx, f"{100 * FILM['share_robust']:.0f}% of winning spots sit on islands: within an atom's width, "
                     f"nearly every release wins. {100 * FILM['share_knife']:.0f}% sit on knife edges: still mixed at "
                     f"every zoom we tried, down to a fifth of an atom. One win can't tell you which kind you found.",
                120, 830, 29, 1680, D.INK, a, "medium")


# ----------------------------------------------------------------------------------------------- 8 real world
def ov_realworld(ctx, gctx, t, shot, cm, crop):
    C.scrim(ctx, 1920, 0.84)
    head2(ctx, "And a real board is never the same twice.", t,
          "Replay a winning drop with the release perfect, and change only the things nobody can hold still.")
    rows = [("Release replayed to within 1 atom", REP["position"]["1e-10"]),
            ("Perfect release; only the chip's own heat jiggle differs", REP["real_world"]["chip's own thermal motion"]),
            ("Perfect release; board 0.001 °C warmer or cooler", EXTRA["board 0.001 C"]),
            ("Perfect release; board 0.01 °C", EXTRA["board 0.01 C"]),
            ("Perfect release; board 1 °C", EXTRA["board 1 C"]),
            ("a fresh drop", REP["base_rate"])]
    x0, bw = LX + 860, 760
    y = 300
    for i, (lab, v) in enumerate(rows):
        a = D.fade(t, 1.0 + 0.7 * i, 99, 0.35)
        f = D.ease_out(min(1.0, max(0.0, (t - 1.0 - 0.7 * i) / 0.8)))
        D.text(ctx, lab, LX, y + 32, 28, D.INK3 if i == 5 else D.INK, a, "medium")
        ctx.rectangle(x0, y + 8, bw * v * f, 32)
        ctx.set_source_rgba(*(D.INK3 if i == 5 else (D.GOLD if i == 0 else D.S1)), a)
        ctx.fill()
        D.text(ctx, f"{100 * v:.0f}%", x0 + bw * v * f + 14, y + 34, 26, D.INK, a, "semibold")
        y += 70
    D.text(ctx, "chance the replay wins again", x0, 290, 20, D.INK3, D.fade(t, 1.0, 99, 0.4))
    a = D.fade(t, 6.5, 99, 0.5)
    D.paragraph(ctx, "A thousandth of a degree stretches the board by about 20 nanometres, a couple of hundred atoms. No studio "
                     "holds a board to a hundredth of a degree, so in practice a replay is barely better than a "
                     "fresh drop.", LX, 800, 29, 1680, D.INK, a, "medium")


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
    a = D.fade(t, t_land + 0.4, 99, 0.5)
    if a > 0:
        C.scrim(ctx, 760, 0.55 * a)
        D.text(ctx, "Deterministic.", 90, 430, 62, D.INK, a, "black")
        D.text(ctx, "Unpredictable.", 90, 504, 62, D.INK, a, "black")
        D.text(ctx, "That's chaos.", 90, 578, 62, D.GOLD, a, "black")
        D.paragraph(ctx, "Drop it in the middle, and let chaos decide.", 92, 640, 26, 520, D.INK2,
                    D.fade(t, t_land + 1.5, 99, 0.5))
    if t > shot["dur"] - 1.2:
        ctx.set_source_rgba(0, 0, 0, D.ease((t - (shot["dur"] - 1.2)) / 1.2))
        ctx.paint()


OVERLAY = dict(hook=ov_hook, deterministic=ov_deterministic, twins=ov_twins, mechanism=ov_mechanism,
               horizon=ov_horizon, crux=ov_crux, islands=ov_islands, realworld=ov_realworld,
               statistics=ov_statistics, outro=ov_outro)
DARK = {"mechanism": (6.0, 0.5), "horizon": (6.0, 0.5), "islands": (6.0, 0.5), "crux": (5.0, 0.6),
        "realworld": (5.0, 0.6)}


def render(shot, fr):
    t = (fr - 1) / FPS
    img, crop = C.base_frame(shot, fr, t)
    img = img * C.vignette()
    cm = C.cam(shot["cam"])
    if shot["name"] in DARK:
        blur, dim = DARK[shot["name"]]
        img = cv2.GaussianBlur(img, (0, 0), 2 + blur * D.ease(min(1, t / 1.2))) * dim
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
