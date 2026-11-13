"""Render the Plinko video (16:9 1920x1080 and a vertical short) from the simulator and results/.

    python3 render.py prepare            # trajectories for the animations -> out/anim.npz
    python3 render.py still 3 40 120     # PNG stills at those times (seconds) -> out/  (add --short)
    python3 render.py video --workers 4  # full film -> out/plinko_optimal_play.mp4
    python3 render.py short --workers 4  # vertical cut -> out/plinko_optimal_play_SHORT.mp4

Everything drawn here is computed: the board is the measured geometry, every chip path comes from
physics.py, every number on screen comes from results/*.json or data/real_outcomes.json.
"""
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

import board as B
import draw as D
import galton
import physics as P

FPS = 30
OUT = "out"
W_, H_ = 1920, 1080

# ------------------------------------------------------------------------------------------- data
_cache = {}


def data():
    if "d" in _cache:
        return _cache["d"]
    d = {}
    d["fit"] = json.load(open("results/fit.json"))
    d["strategy"] = json.load(open("results/strategy.json"))
    d["sens"] = json.load(open("results/sensitivity.json")) if os.path.exists("results/sensitivity.json") else None
    d["chaos"] = json.load(open("results/chaos.json")) if os.path.exists("results/chaos.json") else None
    d["tech"] = json.load(open("results/technique.json"))
    d["real"] = json.load(open("data/real_outcomes.json"))
    z = np.load("results/kernels_e.npz")
    d["kern"] = {k: z[k] for k in z.files}
    a = np.load(f"{OUT}/anim.npz", allow_pickle=True)
    d["anim"] = {k: a[k] for k in a.files}
    d["e"] = d["strategy"]["calibrated_e"]
    d["tuned"] = np.load("results/kernel_final.npz")["K"]
    d["coin"] = galton.kernel()
    _cache["d"] = d
    return d


def money(x):
    return f"${x:,.0f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


# ------------------------------------------------------------------------------------------- prepare
def find_hero(e, want_slots, n_try=400, seed0=1, release_sx=0.004):
    prm = P.params(e=e)
    rng = np.random.default_rng(seed0)
    for i in range(n_try):
        x0 = B.drop_x(5) + rng.normal(0, release_sx)
        w0 = rng.normal(0, 0.5)
        slot, rec = P.trajectory(x0, 0.0, 0.0, 0.0, w0, seed=1000 + i, prm=prm, fps=FPS)
        if slot in want_slots and 2.6 < rec[-1, 0] < 5.5:
            return slot, rec
    raise RuntimeError("no hero path")


def prepare():
    e = json.load(open("results/strategy.json"))["calibrated_e"]
    prm = P.params(e=e)
    out = {}
    s, rec = find_hero(e, {4, 6}, seed0=3)                    # cold open: lands in a $0 next to the $10k
    out["hero_open"] = rec
    s, rec = find_hero(e, {5}, seed0=8)                       # outro: lands in the $10k
    out["hero_outro"] = rec
    # rain: 600 chips from the centre, drawn as ghosts
    rng = np.random.default_rng(42)
    init = P.releases(5, 600, rng)
    paths, slots = [], []
    for i in range(600):
        sl, rec = P.trajectory(*init[i], seed=int(rng.integers(1, 2**31)), prm=prm, fps=FPS)
        if sl == 0:
            continue
        paths.append(rec[:, 1:4].astype(np.float32))
        slots.append(sl)
    lens = np.array([len(p) for p in paths])
    flat = np.zeros((len(paths), lens.max(), 3), np.float32)
    for i, p in enumerate(paths):
        flat[i, :len(p)] = p
        flat[i, len(p):] = p[-1]
    out["rain"] = flat
    out["rain_len"] = lens
    out["rain_slot"] = np.array(slots)
    # butterfly twins: no pin jitter, identical except 0.001 mm in x; want a long shared stretch then a
    # split into far-apart slots
    prm0 = P.params(e=e, jitter_deg=0.0, e_jitter=0.0)
    best = None
    rng = np.random.default_rng(5)
    for i in range(300):
        x0 = B.drop_x(5) + rng.normal(0, 0.004)
        w0 = rng.normal(0, 0.5)
        sa, ra = P.trajectory(x0, 0.0, 0, 0, w0, seed=7, prm=prm0, fps=FPS)
        sb, rb = P.trajectory(x0 + 1e-6, 0.0, 0, 0, w0, seed=7, prm=prm0, fps=FPS)
        if sa == 0 or sb == 0:
            continue
        m = min(len(ra), len(rb))
        sep = np.hypot(ra[:m, 1] - rb[:m, 1], ra[:m, 2] - rb[:m, 2])
        k = np.argmax(sep > 0.01)      # frame where they visibly part (1 cm)
        score = abs(sa - sb) * 10 + (ra[k, 2] / B.last_row_y()) * 5
        if abs(sa - sb) >= 2 and 0.35 < ra[k, 2] / B.last_row_y() < 0.75 and (best is None or score > best[0]):
            best = (score, ra, rb, sa, sb)
    out["twin_a"], out["twin_b"] = best[1], best[2]
    # textbook coin-flip path: a fixed sequence that ends in the $10k
    rng = np.random.default_rng(12)
    while True:
        flips = rng.integers(0, 2, 12)
        u = 10
        ok = True
        for r in range(12):
            u += 1 if flips[r] else -1
            if u < 2 or u > 18:
                ok = False
        if ok and u == 10:
            break
    out["flips"] = flips
    np.savez_compressed(f"{OUT}/anim.npz", **out)
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in out.items()})


# ------------------------------------------------------------------------------------------- layout
LBL = 0.55 * B.SLOT_PITCH
V_CENTER = D.fit_view(560, 40, 1360, 1000, margin_bottom_labels=LBL)
V_RIGHT = D.fit_view(1080, 40, 1860, 1000, margin_bottom_labels=LBL)
LEFT_X, LEFT_W = 100, 860


def lerp_view(a, b, t):
    t = D.ease(t)
    return D.View(a.cx + (b.cx - a.cx) * t, a.top + (b.top - a.top) * t, a.s + (b.s - a.s) * t, a.y_top)


_board_cache = {}


def board_layer(ctx, v, key=None, **kw):
    """Draw the static board; cached as an image when the view is fixed."""
    import cairo
    if key is None:
        D.draw_board(ctx, v, **kw)
        return
    if key not in _board_cache:
        s, c = D.new_surface(W_, H_)
        c.set_source_rgba(0, 0, 0, 0)
        c.set_operator(cairo.OPERATOR_CLEAR)
        c.paint()
        c.set_operator(cairo.OPERATOR_OVER)
        s2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, W_, H_)
        c2 = cairo.Context(s2)
        D.draw_board(c2, v, **kw)
        _board_cache[key] = s2
    ctx.set_source_surface(_board_cache[key], 0, 0)
    ctx.paint()


def sample(rec, t):
    """Position on a recorded path at time t (seconds since release); clamps to the ends."""
    i = t * FPS
    if i <= 0:
        return rec[0]
    if i >= len(rec) - 1:
        return rec[-1]
    k = int(i)
    f = i - k
    return rec[k] * (1 - f) + rec[k + 1] * f


def heading(ctx, s, t, t0=0.2, y=150, sub=None, size=58, t1=1e9):
    a = D.fade(t, t0, t1, 0.5)
    D.text(ctx, s, LEFT_X, y, size, D.INK, a, "black")
    if sub:
        D.paragraph(ctx, sub, LEFT_X, y + size * 0.95, 28, LEFT_W, D.INK2, a)


def bar_chart(ctx, x, y, w, h, series, labels, colors, t_grow=1.0, alpha=1.0, ymax=None, fmt=pct,
              highlight=None, value_labels=None, legend=None, bar_w=None, tick_fmt=None, ticks=None,
              label_size=20):
    """Grouped columns. series: list of arrays (same length as labels)."""
    n = len(labels)
    m = len(series)
    ymax = ymax or max(max(s) for s in series) * 1.15
    base = y + h
    ctx.set_line_width(1)
    if ticks:
        for tv in ticks:
            yy = base - h * tv / ymax
            ctx.move_to(x, yy)
            ctx.line_to(x + w, yy)
            ctx.set_source_rgba(*D.rgba(D.GRID, alpha))
            ctx.stroke()
            D.text(ctx, (tick_fmt or fmt)(tv), x - 10, yy, 17, D.INK3, alpha, align="right", anchor="middle")
    ctx.move_to(x, base)
    ctx.line_to(x + w, base)
    ctx.set_source_rgba(*D.rgba(D.INK3, alpha))
    ctx.stroke()
    band = w / n
    bw = bar_w or min(24 * 1.6, band * 0.8 / m - 2)
    group = bw * m + 2 * (m - 1)
    for i in range(n):
        gx = x + band * i + (band - group) / 2
        top = 0.0
        for j in range(m):
            val = series[j][i]
            hh = h * min(val, ymax) / ymax * D.ease_out(t_grow)
            top = max(top, hh)
            a = alpha * (1.0 if highlight is None or i in highlight else 0.35)
            D.column(ctx, gx + j * (bw + 2), base, bw, hh, colors[j], a)
        if value_labels and t_grow > 0.95 and (highlight is None or i in highlight):
            labs = [(j, value_labels[j](series[j][i])) for j in range(m) if value_labels[j] is not None]
            for k, (j, s) in enumerate(labs[::-1]):
                D.text(ctx, s, gx + group / 2, base - top - 10 - k * 24, 19,
                       D.INK if j == 0 else D.INK2, alpha * min(1, (t_grow - 0.95) * 20), "semibold", "center")
        D.text(ctx, labels[i], x + band * (i + 0.5), base + 30, label_size, D.INK2, alpha, align="center")
    if legend:
        lx = x
        for j, (name, col) in enumerate(zip(legend, colors)):
            ctx.rectangle(lx, y - 34, 16, 16)
            ctx.set_source_rgba(*D.rgba(col, alpha))
            ctx.fill()
            lx += 24 + D.text(ctx, name, lx + 24, y - 20, 19, D.INK2, alpha) + 28


# ------------------------------------------------------------------------------------------- scenes
def sc_open(ctx, t, d):
    v = V_CENTER
    board_layer(ctx, v, "center")
    rec = d["anim"]["hero_open"]
    t0 = 1.2
    if t < t0:
        a = D.fade(t, 0.3, 99, 0.6)
        D.draw_chip(ctx, v, rec[0, 1], rec[0, 2], rec[0, 3], alpha=a)
    else:
        r = sample(rec, t - t0)
        D.draw_chip(ctx, v, r[1], r[2], r[3])
    t_land = t0 + rec[-1, 0]
    if t > t_land:
        a = D.fade(t, t_land, 99, 0.25)
        k = int(round(rec[-1, 1] / B.SLOT_PITCH))
        lx, ly = v.p(k * B.SLOT_PITCH, B.last_row_y())
        D.pill(ctx, D.SLOT_LABELS[k - 1], lx, ly - 40, 56, D.GOLD if k == 5 else D.INK, alpha=a)
    a = D.fade(t, t_land + 0.8, 99, 0.6)
    D.text(ctx, "Where should", 90, 420, 64, D.INK, a, "black")
    D.text(ctx, "you drop", 90, 494, 64, D.INK, a, "black")
    D.text(ctx, "the chip?", 90, 568, 64, D.INK, a, "black")
    D.paragraph(ctx, "A physics simulation of the real board, checked against 1,029 chips from the show.",
                90, 640, 26, 400, D.INK2, a)


def measure_layout(t):
    return lerp_view(V_CENTER, V_RIGHT, t / 1.2)


def sc_measure(ctx, t, d):
    v = measure_layout(t)
    if t < 1.25:
        D.draw_board(ctx, v)
    else:
        board_layer(ctx, v, "right")
    W = B.SLOT_PITCH
    heading(ctx, "The board, measured", t, 0.8, 150)
    inch = 0.0254
    items = [
        (2.0, "9 slots, 7.0 in (17.8 cm) apart", "pitch"),
        (5.0, f"13 rows of steel pins: {len(D.PEG_XY)} pins", "rows"),
        (8.5, "Sawtooth side walls: 6 teeth a side stand in for the missing end pins", "teeth"),
        (12.0, f"Chip {B.CHIP_DIAMETER/inch:.1f} in (13 cm) across, about 3/4 in thick", "chip"),
        (15.0, f"Playfield {9*W/inch:.0f} in wide, {B.last_row_y()/inch:.0f} in from top pin row to bottom",
         "field"),
    ]
    y = 260
    for (ts, s, key) in items:
        a = D.fade(t, ts, 99, 0.5)
        if a > 0:
            ctx.arc(LEFT_X + 9, y - 9, 6, 0, 2 * math.pi)
            ctx.set_source_rgba(*D.rgba(D.GOLD, a))
            ctx.fill()
            y = D.paragraph(ctx, s, LEFT_X + 32, y, 32, LEFT_W - 40, D.INK, a, "medium") + 26
    # callouts on the board
    yb = B.last_row_y() + B.FLOOR_BELOW_LAST_ROW
    a = D.fade(t, 2.0, 99, 0.5)
    if a > 0:
        xa, ya = v.p(4.5 * W, yb + 0.0)
        xb, _ = v.p(5.5 * W, yb)
        k = int(min(1, (t - 2.0) / 0.6) * 9)
        D.dim_line(ctx, xa, ya + 70, xb, ya + 70, "7.0 in", 22, a, D.GOLD, side=-1)
    a = D.fade(t, 5.0, 8.4, 0.3)
    if a > 0:   # flash pins row by row
        row = int((t - 5.0) * 5) % 13
        for (x, yy), r in zip(D.PEG_XY, D.PEG_ROW):
            if r <= row:
                px, py = v.p(x, yy)
                ctx.arc(px, py, 6, 0, 2 * math.pi)
                ctx.set_source_rgba(*D.rgba(D.GOLD, a * (1.0 if r == row else 0.45)))
                ctx.fill()
    a = D.fade(t, 8.5, 11.8, 0.3)
    if a > 0:
        for poly in D.WALLS:
            for r in range(1, 13, 2):
                (x, yy) = poly[r + 1]
                px, py = v.p(x, yy)
                ctx.arc(px, py, 9, 0, 2 * math.pi)
                ctx.set_source_rgba(*D.rgba(D.GOLD, a))
                ctx.fill()
    a = D.fade(t, 12.0, 99, 0.5)
    if a > 0:
        cx, cy = 5 * W, -0.05 * W
        D.draw_chip(ctx, v, cx, cy, 0.4, a)
        xa, ya = v.p(cx - B.CHIP_DIAMETER / 2, cy)
        xb, _ = v.p(cx + B.CHIP_DIAMETER / 2, cy)
        D.dim_line(ctx, xa, ya - v.L(B.CHIP_DIAMETER / 2) - 18, xb, ya - v.L(B.CHIP_DIAMETER / 2) - 18,
                   "5.1 in", 20, a, D.GOLD, side=1)
    a = D.fade(t, 15.0, 99, 0.5)
    if a > 0:
        xa, ya = v.p(0.5 * W, 0)
        _, yb2 = v.p(0.5 * W, B.last_row_y())
        D.dim_line(ctx, xa - 40, ya, xa - 40, yb2, "73 in", 22, a, D.GOLD, side=-1)
    a = D.fade(t, 17.0, 99, 0.6)
    D.paragraph(ctx, "No blueprint of the TV board is public. These come from broadcast frames, scaled by the "
                     "chip (5 in replicas; 13.5 cm against a hand) and cross-checked against the host's height.",
                LEFT_X, 860, 22, LEFT_W, D.INK3, a)


def sc_textbook(ctx, t, d):
    v = V_RIGHT
    board_layer(ctx, v, "right")
    W = B.SLOT_PITCH
    heading(ctx, "The textbook answer", t, 0.2, 150,
            "Treat every pin as a coin flip: half a slot left or right. A chip meets 12 pins on the way down.")
    flips = d["anim"]["flips"]
    # animate the discrete path
    step_t = 0.55
    t0 = 1.5
    u = 10
    pts = [(5 * W, 0.0)]
    for r in range(12):
        u += 1 if flips[r] else -1
        pts.append((u / 2 * W, (r + 1) * B.ROW_SPACING - 0.0 * W))
    pts.append((pts[-1][0], B.last_row_y() + B.FLOOR_BELOW_LAST_ROW - B.CHIP_DIAMETER / 2))
    k = (t - t0) / step_t
    if t > t0 - 0.5:
        n_done = int(max(0, min(len(pts) - 1, k)))
        # trail
        ctx.set_line_width(4)
        ctx.set_source_rgba(*D.rgba(D.CHIP, 0.55))
        ctx.move_to(*v.p(*pts[0]))
        for p in pts[1:n_done + 1]:
            ctx.line_to(*v.p(*p))
        if 0 <= k < len(pts) - 1:
            f = D.ease(k - int(k))
            a, b = pts[int(k)], pts[int(k) + 1]
            cur = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            ctx.line_to(*v.p(*cur))
        else:
            cur = pts[-1] if k >= len(pts) - 1 else pts[0]
        ctx.stroke()
        D.draw_chip(ctx, v, cur[0], cur[1], 0.0)
        # flip letters
        s = " ".join("R" if f else "L" for f in flips[:max(0, min(12, int(k)))])
        D.text(ctx, s, LEFT_X, 330, 30, D.INK, D.fade(t, t0, 12.2, 0.3), "semibold")
    # the count of paths
    a = D.fade(t, 9.0, 99, 0.6)
    if a > 0:
        D.paragraph(ctx, "12 flips make 4,096 equally likely paths. 924 of them end in the middle slot.",
                    LEFT_X, 400, 30, LEFT_W, D.INK, a, "medium")
        D.text(ctx, "924 / 4,096 = 22.6%", LEFT_X, 520, 54, D.GOLD, a, "black")
    # EV per drop position
    a = D.fade(t, 14.0, 99, 0.6)
    if a > 0:
        ev, _ = galton.ev_table(d["coin"])
        bar_chart(ctx, LEFT_X + 60, 640, LEFT_W - 80, 270, [ev], [str(i) for i in range(1, 10)], [D.S2],
                  t_grow=(t - 14.0) / 1.2, alpha=a, ymax=3000, fmt=money, ticks=[1000, 2000, 3000],
                  value_labels=[lambda x: money(x)], highlight=[4, 0])
        D.text(ctx, "Average win per chip, by drop position (1 = far left)", LEFT_X, 610, 22, D.INK2, a)
    a = D.fade(t, 19.0, 99, 0.6)
    D.text(ctx, "Middle: $2,558 a chip. Far edge: $780.", LEFT_X, 1010, 30, D.INK, a, "semibold")


def sc_reality(ctx, t, d):
    v = V_RIGHT
    board_layer(ctx, v, "right")
    real = np.array(d["real"]["totals"]["slots_left_to_right"], float)
    n = real.sum()
    share = real / n
    coin = d["coin"][4]
    heading(ctx, "What really happens on the show", t, 0.2, 150,
            f"{int(n):,} chips from 11 seasons (2015 to 2026), logged by tpirstats.com.")
    a = D.fade(t, 1.5, 99, 0.6)
    if a > 0:
        bar_chart(ctx, LEFT_X + 60, 360, LEFT_W - 80, 380, [share, coin], D.SLOT_LABELS, [D.S1, D.S2],
                  t_grow=(t - 1.5) / 1.2, alpha=a, ymax=0.45, ticks=[0.1, 0.2, 0.3, 0.4],
                  tick_fmt=lambda x: f"{100*x:.0f}%", legend=["Show, every drop", "Coin flips, dropped in the middle"],
                  label_size=17, value_labels=None,
                  highlight=[4] if 6 < t < 11 else ([0, 8] if 11 <= t < 16 else None))
    a = D.fade(t, 6.0, 99, 0.6)
    if 6 <= t < 11:
        D.text(ctx, f"The $10,000 slot caught {pct(share[4])} of real chips, not 22.6%.", LEFT_X, 850, 28,
               D.INK, a, "medium")
    elif 11 <= t < 16:
        D.text(ctx, f"The $100 slots caught {pct(share[0] + share[8])}; coin flips allow "
                    f"{pct(coin[0] + coin[8])}.", LEFT_X, 850, 28, D.INK, D.fade(t, 11, 99, 0.4), "medium")
    per = float(real @ B.SLOT_VALUES / n)
    a = D.fade(t, 16.0, 99, 0.6)
    if a > 0:
        D.text(ctx, f"Real average: {money(per)} a chip", LEFT_X, 860, 40, D.GOLD, a, "black")
        D.paragraph(ctx, "Either contestants drop all over the board, or the real board is wilder than "
                         "coin flips. Probably both. To tell them apart, I needed real physics.",
                    LEFT_X, 915, 26, LEFT_W, D.INK2, a)


def rain_state(d, t, start_gap=0.045):
    """Positions of rain chips at time t, and how many have landed per slot."""
    rain, lens, slots = d["anim"]["rain"], d["anim"]["rain_len"], d["anim"]["rain_slot"]
    n = len(lens)
    starts = np.arange(n) * start_gap
    fi = ((t - starts) * FPS).astype(int)
    active = (fi >= 0) & (fi < lens)
    landed = fi >= lens
    counts = np.bincount(slots[landed], minlength=10)[1:]
    pos = rain[np.arange(n)[active], np.minimum(fi[active], lens[active] - 1)]
    return pos, counts, landed.sum()


def sc_physics(ctx, t, d):
    v = V_RIGHT
    board_layer(ctx, v, "right")
    heading(ctx, "So I rebuilt it in a physics engine", t, 0.2, 150)
    items = ["A 5.1 in disc that spins, slides and bounces", "98 round steel pins, sawtooth walls, dividers",
             "Bounce and friction at every contact; pins a little imperfect",
             "Board leaning back 10 degrees; the chip slides on it"]
    y = 250
    for i, s in enumerate(items):
        a = D.fade(t, 0.8 + i * 0.6, 99, 0.4)
        ctx.arc(LEFT_X + 9, y - 9, 6, 0, 2 * math.pi)
        ctx.set_source_rgba(*D.rgba(D.S3, a))
        ctx.fill()
        D.text(ctx, s, LEFT_X + 32, y, 27, D.INK, a)
        y += 44
    t_rain = t - 4.0
    if t_rain > 0:
        pos, counts, nl = rain_state(d, t_rain)
        for p in pos:
            D.draw_chip(ctx, v, p[0], p[1], p[2], alpha=0.8, r_scale=1.0)
        share = counts / max(1, counts.sum())
        a = D.fade(t, 4.5, 99, 0.5)
        bar_chart(ctx, LEFT_X + 60, 520, LEFT_W - 80, 300, [share], D.SLOT_LABELS, [D.S3],
                  t_grow=1, alpha=a, ymax=0.45, ticks=[0.1, 0.2, 0.3, 0.4],
                  tick_fmt=lambda x: f"{100*x:.0f}%", label_size=17,
                  value_labels=[pct] if t > 26 else None, highlight=[4] if t > 26 else None)
        D.text(ctx, f"Simulated chips dropped in the middle: {int(nl)}", LEFT_X, 490, 24, D.INK2, a)
    a = D.fade(t, 26.0, 99, 0.5)
    if a > 0:
        K = d["tuned"]
        D.paragraph(ctx, f"After 24,000 simulated drops in the middle: {pct(K[4, 4])} land in the $10,000, "
                         f"against 22.6% for coin flips. The real board scatters chips much further.",
                    LEFT_X, 905, 28, LEFT_W, D.INK, a, "medium")


def sc_chaos(ctx, t, d):
    v = V_RIGHT
    board_layer(ctx, v, "right")
    heading(ctx, "Why nobody can aim", t, 0.2, 150,
            "Two chips, released 0.001 mm apart. Same bounce, same friction, no imperfect pins.")
    ra, rb = d["anim"]["twin_a"], d["anim"]["twin_b"]
    t0 = 2.0
    tt = max(0.0, t - t0)
    for rec, col in ((rb, D.S1), (ra, D.CHIP)):
        n = int(min(len(rec) - 1, tt * FPS))
        ctx.set_line_width(3)
        ctx.set_source_rgba(*D.rgba(col, 0.55))
        for i in range(n + 1):
            x, y = v.p(rec[i, 1], rec[i, 2])
            (ctx.move_to if i == 0 else ctx.line_to)(x, y)
        ctx.stroke()
        r = sample(rec, tt)
        D.draw_chip(ctx, v, r[1], r[2], r[3], color=col)
    # separation readout
    m = min(len(ra), len(rb))
    i = int(min(m - 1, tt * FPS))
    sep = math.hypot(ra[i, 1] - rb[i, 1], ra[i, 2] - rb[i, 2])
    a = D.fade(t, t0, 99, 0.4)
    if sep < 1e-3:
        s = f"{sep*1e6:,.1f} µm apart" if sep < 1e-4 else f"{sep*1e3:.2f} mm apart"
    else:
        s = f"{sep*100:.1f} cm apart"
    D.text(ctx, s, LEFT_X, 420, 64, D.INK, a, "black")
    # separation growth: log plot
    gx, gy, gw, gh = LEFT_X + 60, 500, LEFT_W - 100, 260
    a = D.fade(t, t0, 99, 0.4)
    ctx.set_line_width(1)
    for dec in range(-6, 1):
        yy = gy + gh * (1 - (dec + 6) / 6)
        ctx.move_to(gx, yy)
        ctx.line_to(gx + gw, yy)
        ctx.set_source_rgba(*D.rgba(D.GRID, a))
        ctx.stroke()
        lab = {-6: "1 µm", -5: "10 µm", -4: "0.1 mm", -3: "1 mm", -2: "1 cm", -1: "10 cm", 0: "1 m"}[dec]
        D.text(ctx, lab, gx - 10, yy, 17, D.INK3, a, align="right", anchor="middle")
    T = ra[m - 1, 0]
    ctx.set_line_width(2)
    ctx.set_source_rgba(*D.rgba(D.GOLD, a))
    for j in range(0, i + 1):
        sj = max(1e-6, math.hypot(ra[j, 1] - rb[j, 1], ra[j, 2] - rb[j, 2]))
        xx = gx + gw * ra[j, 0] / T
        yy = gy + gh * (1 - (math.log10(sj) + 6) / 6)
        (ctx.move_to if j == 0 else ctx.line_to)(xx, yy)
    ctx.stroke()
    D.text(ctx, "Separation over the drop", gx, gy - 14, 20, D.INK2, a)
    a = D.fade(t, 12.0, 99, 0.6)
    ch = d["chaos"]
    D.paragraph(ctx, f"Each bounce multiplies a tiny difference. Released a whole millimetre apart, twin "
                     f"chips still share a slot only {pct(ch['same_slot']['0.001'], 0)} of the time "
                     f"(two unrelated chips: {pct(ch['independent_same_slot'], 0)}).",
                LEFT_X, 850, 27, LEFT_W, D.INK, a, "medium")


def sc_calibrate(ctx, t, d):
    v = V_RIGHT
    board_layer(ctx, v, "right")
    heading(ctx, "Tuning the bounce", t, 0.2, 150,
            "Bounciness is the one thing I could not measure from video, so the show's own 1,029 chips pick it. "
            "Simulated chips here are all dropped in the middle.")
    es = d["kern"]["e"]
    Ks = d["kern"]["K"]
    fit = d["fit"]["models"]
    e_best = d["e"]
    # sweep e then settle
    if t < 3:
        e_now = es[0]
    elif t < 13:
        f = (t - 3) / 10
        e_now = es[0] + (es[-1] - es[0]) * (0.5 - 0.5 * math.cos(2 * math.pi * f * 0.75))
    else:
        f = D.ease(min(1, (t - 13) / 2))
        e13 = es[0] + (es[-1] - es[0]) * (0.5 - 0.5 * math.cos(2 * math.pi * 0.75))
        e_now = e13 + (e_best - e13) * f
    # interpolate kernels
    j = np.clip(np.searchsorted(es, e_now) - 1, 0, len(es) - 2)
    f = (e_now - es[j]) / (es[j + 1] - es[j])
    K = Ks[j] * (1 - f) + Ks[j + 1] * f
    # what the fitted drop habit would produce
    key = f"physics e={e_best:.3f}"
    lam = fit[key]["best"]["lam"]
    import analyze as A
    q = A.drop_mix(lam)
    real = np.array(d["real"]["totals"]["slots_left_to_right"], float)
    share = real / real.sum()
    sim = q @ K
    a = D.fade(t, 1.0, 99, 0.5)
    bar_chart(ctx, LEFT_X + 60, 470, LEFT_W - 80, 330, [share, sim], D.SLOT_LABELS, [D.S1, D.S3],
              t_grow=1, alpha=a, ymax=0.30, ticks=[0.1, 0.2, 0.3], tick_fmt=lambda x: f"{100*x:.0f}%",
              legend=["Show", "Simulator"], label_size=17)
    # slider
    sx, sy, sw = LEFT_X, 985, LEFT_W - 60
    ctx.set_line_width(4)
    ctx.set_line_cap(1)
    ctx.move_to(sx, sy)
    ctx.line_to(sx + sw, sy)
    ctx.set_source_rgba(*D.rgba(D.GRID, a))
    ctx.stroke()
    fx = (e_now - es[0]) / (es[-1] - es[0])
    ctx.arc(sx + sw * fx, sy, 11, 0, 2 * math.pi)
    ctx.set_source_rgba(*D.rgba(D.GOLD, a))
    ctx.fill()
    D.text(ctx, "less bounce", sx, sy + 40, 20, D.INK3, a)
    D.text(ctx, "more bounce", sx + sw, sy + 40, 20, D.INK3, a, align="right")
    D.text(ctx, f"bounciness e = {e_now:.2f}", sx + sw / 2, sy - 22, 24, D.INK, a, "semibold", "center")
    a = D.fade(t, 15.5, 99, 0.5)
    if a > 0:
        D.text(ctx, f"Best match: e = {e_best:.2f}", LEFT_X, 892, 34, D.GOLD, a, "black")


def sc_strategy(ctx, t, d):
    v = V_RIGHT
    board_layer(ctx, v, "right")
    st = d["strategy"]
    e = d["e"]
    phys = st["models"]["physics tuned"]
    coin = st["models"]["coin-flip"]
    ev_p, ev_c = np.array(phys["ev"]), np.array(coin["ev"])
    if t < 13:
        heading(ctx, "So where should you drop it?", t, 0.2, 150, t1=13)
        a = D.fade(t, 0.8, 13, 0.5)
        bar_chart(ctx, LEFT_X + 60, 330, LEFT_W - 80, 380, [ev_p, ev_c], [str(i) for i in range(1, 10)],
                  [D.S3, D.S2], t_grow=(t - 0.8) / 1.2, alpha=a, ymax=3000, fmt=money,
                  ticks=[1000, 2000, 3000], legend=["Physics, tuned to the show", "Coin flips"],
                  value_labels=None, highlight=[4, 0] if t > 4 else None)
        D.text(ctx, "Average win per chip, by drop position", LEFT_X, 800, 22, D.INK2, a)
        a = D.fade(t, 5.0, 13, 0.5)
        D.paragraph(ctx, f"Still the middle, in both models. On the tuned board the middle three are a tie at "
                         f"about {money(ev_p[4])} a chip; the far edge pays {money(ev_p[0])}. That is "
                         f"{money(ev_p[4]-ev_p[0])} a chip, not the textbook's {money(ev_c[4]-ev_c[0])}.",
                    LEFT_X, 860, 28, LEFT_W, D.INK, a, "medium")
    elif t < 24:
        tt = t - 13
        heading(ctx, "What does not help", tt, 0.2, 150, t1=11)
        sty = d["tech"]["styles"]
        rows = [("Surgeon-steady hand (2 mm)", "steady hand (2 mm scatter)"),
                ("Sloppy hand (3 cm)", "sloppy hand (3 cm scatter)"),
                ("Thrown down at 1 m/s", "thrown down at 1 m/s"), ("Spun hard", "spun hard (10 rad/s)"),
                ("Aimed 2 cm off centre", "aimed 2 cm off centre")]
        y = 290
        D.text(ctx, "Average per chip, 24,000 drops each, all over the middle", LEFT_X, y - 50, 22, D.INK2,
               D.fade(tt, 0.6, 11, 0.4))
        for i, (lab, key) in enumerate(rows):
            a = D.fade(tt, 0.8 + i * 0.6, 11, 0.4)
            D.text(ctx, lab, LEFT_X, y, 32, D.INK, a, "medium")
            D.text(ctx, money(sty[key]["ev"]), LEFT_X + 640, y, 32, D.INK, a, "semibold", "right")
            D.text(ctx, f"\u00b1{money(sty[key]['se'])}", LEFT_X + 660, y, 24, D.INK3, a)
            y += 62
        evs = [sty[k]["ev"] for _, k in rows] + [sty["normal release (1 cm scatter)"]["ev"]]
        a = D.fade(tt, 4.5, 11, 0.5)
        D.paragraph(ctx, f"Every style lands within {money(max(evs) - min(evs))} of the others, about what you "
                         f"get from chance alone. Past the first few pins, chaos takes over. Aim at the slot; "
                         f"don't try to finesse it.", LEFT_X, y + 30, 28, LEFT_W, D.INK2, a)
    else:
        tt = t - 24
        heading(ctx, "What does help: more chips", tt, 0.2, 150)
        sp = st["small_prizes"]
        five = st["five_chips_calibrated"]["all centre"]
        a = D.fade(tt, 0.8, 99, 0.5)
        D.paragraph(ctx, f"Each chip is worth about {money(ev_p[4])}. You earn up to four by picking which digit "
                         f"of a wrong price is right.", LEFT_X, 270, 30, LEFT_W, D.INK, a, "medium")
        a = D.fade(tt, 3.0, 99, 0.5)
        D.text(ctx, f"The last digit was the right one {pct(sp['last_digit_correct_share'], 0)} of the time",
               LEFT_X, 420, 32, D.GOLD, a, "black")
        D.text(ctx, f"({sp['items']:,} small prizes, 2015 to 2026). Unsure? Keep the last digit.", LEFT_X, 462,
               24, D.INK2, a)
        a = D.fade(tt, 6.0, 99, 0.5)
        D.text(ctx, "Five chips, all in the middle:", LEFT_X, 570, 30, D.INK, a, "medium")
        D.text(ctx, f"{money(five['mean'])} on average", LEFT_X, 640, 54, D.INK, a, "black")
        D.text(ctx, f"{pct(five['p_at_least_one_10k'], 0)} chance of at least one $10,000", LEFT_X, 700, 30,
               D.INK2, a, "medium")
        D.text(ctx, f"{pct(five['p_zero'], 1)} chance that all five land on $0", LEFT_X, 745, 30, D.INK2, a,
               "medium")


def sc_outro(ctx, t, d):
    v = lerp_view(V_RIGHT, V_CENTER, t / 1.2)
    if t < 1.25:
        D.draw_board(ctx, v)
    else:
        board_layer(ctx, v, "center")
    rec = d["anim"]["hero_outro"]
    t0 = 1.5
    r = sample(rec, max(0.0, t - t0))
    D.draw_chip(ctx, v, r[1], r[2], r[3], alpha=D.fade(t, 1.0, 99, 0.4))
    t_land = t0 + rec[-1, 0]
    a = D.fade(t, t_land + 0.3, 99, 0.4)
    D.text(ctx, "Drop it in", 90, 440, 64, D.INK, a, "black")
    D.text(ctx, "the middle.", 90, 514, 64, D.GOLD, a, "black")
    D.paragraph(ctx, "Then let chaos decide.", 90, 580, 30, 420, D.INK2, a, "medium")
    a = D.fade(t, t_land + 1.2, 99, 0.5)
    D.paragraph(ctx, "Sources and method: README and ACCURACY.md in the project folder.", 1440, 980, 18, 420,
                D.INK3, a)


SCENES = [("open", 9.0, sc_open), ("measure", 22.0, sc_measure), ("textbook", 26.0, sc_textbook),
          ("reality", 24.0, sc_reality), ("physics", 30.0, sc_physics), ("chaos", 22.0, sc_chaos),
          ("calibrate", 20.0, sc_calibrate), ("strategy", 38.0, sc_strategy), ("outro", 10.0, sc_outro)]


# ------------------------------------------------------------------------------------------- vertical short
V_SHORT = D.fit_view(70, 760, 1010, 1830, margin_bottom_labels=LBL)
SX, SW = 80, 920


def short_board(ctx, d):
    board_layer(ctx, V_SHORT, "short")


def sh_hook(ctx, t, d):
    short_board(ctx, d)
    rec = d["anim"]["hero_open"]
    r = sample(rec, max(0.0, t - 0.8))
    D.draw_chip(ctx, V_SHORT, r[1], r[2], r[3])
    a = D.fade(t, 0.1, 99, 0.4)
    D.text(ctx, "Where should you", SX, 260, 78, D.INK, a, "black")
    D.text(ctx, "drop a Plinko chip?", SX, 350, 78, D.INK, a, "black")
    D.paragraph(ctx, "I measured the board, rebuilt it in a physics engine and checked it against 1,029 real "
                     "chips from the show.", SX, 440, 34, SW, D.INK2, a)
    if t > 0.8 + rec[-1, 0]:
        k = int(round(rec[-1, 1] / B.SLOT_PITCH))
        lx, ly = V_SHORT.p(k * B.SLOT_PITCH, B.last_row_y())
        D.pill(ctx, D.SLOT_LABELS[k - 1], lx, ly - 40, 64, D.GOLD if k == 5 else D.INK,
               alpha=D.fade(t, 0.8 + rec[-1, 0], 99, 0.2))


def sh_textbook(ctx, t, d):
    short_board(ctx, d)
    a = D.fade(t, 0.1, 99, 0.4)
    D.text(ctx, "The textbook says:", SX, 200, 46, D.INK2, a, "medium")
    D.text(ctx, "the middle.", SX, 280, 80, D.GOLD, a, "black")
    ev, _ = galton.ev_table(d["coin"])
    bar_chart(ctx, SX + 60, 400, SW - 80, 250, [ev], [str(i) for i in range(1, 10)], [D.S2],
              t_grow=(t - 0.6) / 1.0, alpha=D.fade(t, 0.5, 99, 0.4), ymax=3000, fmt=money,
              ticks=[1000, 2000, 3000], value_labels=[money], highlight=[4, 0], label_size=24)
    D.text(ctx, "Average win per chip if every pin is a coin flip", SX, 370, 26, D.INK2, D.fade(t, 0.5, 99, 0.4))


def sh_reality(ctx, t, d):
    short_board(ctx, d)
    real = np.array(d["real"]["totals"]["slots_left_to_right"], float)
    n = real.sum()
    per = float(real @ B.SLOT_VALUES / n)
    a = D.fade(t, 0.1, 99, 0.4)
    D.text(ctx, "The show says otherwise.", SX, 200, 56, D.INK, a, "black")
    D.paragraph(ctx, f"{int(n):,} chips, 2015 to 2026:", SX, 280, 36, SW, D.INK2, a)
    a = D.fade(t, 1.2, 99, 0.4)
    D.text(ctx, f"{pct(real[4] / n)} hit the $10,000", SX, 390, 60, D.GOLD, a, "black")
    D.text(ctx, "coin flips predict 22.6%", SX, 440, 32, D.INK2, a)
    a = D.fade(t, 3.0, 99, 0.4)
    D.text(ctx, f"{money(per)} a chip", SX, 560, 60, D.INK, a, "black")
    D.text(ctx, "coin flips predict $2,558", SX, 610, 32, D.INK2, a)


def sh_rain(ctx, t, d):
    short_board(ctx, d)
    a = D.fade(t, 0.1, 99, 0.4)
    D.text(ctx, "Real physics, tuned to the show:", SX, 190, 44, D.INK, a, "black")
    pos, counts, nl = rain_state(d, t - 0.5, start_gap=0.022)
    for p in pos:
        D.draw_chip(ctx, V_SHORT, p[0], p[1], p[2], alpha=0.85)
    share = counts / max(1, counts.sum())
    bar_chart(ctx, SX + 60, 300, SW - 80, 300, [share], D.SLOT_LABELS, [D.S3], t_grow=1, alpha=a, ymax=0.45,
              ticks=[0.1, 0.2, 0.3, 0.4], tick_fmt=lambda x: f"{100*x:.0f}%", label_size=19)
    D.text(ctx, f"{int(nl)} chips dropped in the middle", SX, 690, 30, D.INK2, a)


def sh_verdict(ctx, t, d):
    short_board(ctx, d)
    st = d["strategy"]
    ev = np.array(st["models"]["physics tuned"]["ev"])
    sp = st["small_prizes"]
    lines = [(0.2, "Still drop in the middle", D.GOLD, 58, "black"),
             (0.9, f"Middle three: {money(ev[4])} a chip. Edge: {money(ev[0])}", D.INK, 38, "semibold"),
             (3.5, "Spin it, toss it, aim it:", D.INK, 44, "black"),
             (4.1, "no measurable effect. It's chaos.", D.INK2, 38, "medium"),
             (6.8, "What pays: more chips.", D.INK, 44, "black"),
             (7.4, f"Last digit was right {pct(sp['last_digit_correct_share'], 0)} of the time.", D.INK2, 38,
              "medium")]
    y = 190
    for (ts, txt, col, size, wt) in lines:
        D.text(ctx, txt, SX, y, size, col, D.fade(t, ts, 99, 0.4), wt)
        y += size * 1.45 + (24 if wt == "semibold" or (wt == "medium") else 0)


def sh_outro(ctx, t, d):
    short_board(ctx, d)
    rec = d["anim"]["hero_outro"]
    r = sample(rec, max(0.0, t - 0.3))
    D.draw_chip(ctx, V_SHORT, r[1], r[2], r[3])
    a = D.fade(t, 0.2, 99, 0.4)
    D.text(ctx, "Drop it in the middle.", SX, 300, 74, D.GOLD, a, "black")
    D.text(ctx, "Then let chaos decide.", SX, 390, 52, D.INK, a, "black")


SHORT_SCENES = [("hook", 6.5, sh_hook), ("textbook", 8.0, sh_textbook), ("reality", 8.5, sh_reality),
                ("rain", 15.0, sh_rain), ("verdict", 13.0, sh_verdict), ("outro", 6.5, sh_outro)]
FORMATS = {"wide": (1920, 1080, SCENES), "short": (1080, 1920, SHORT_SCENES)}
FMT = {"name": "wide"}


def set_format(name):
    global W_, H_
    FMT["name"] = name
    W_, H_ = FORMATS[name][0], FORMATS[name][1]


def total_seconds(scenes=None):
    return sum(s[1] for s in (scenes or FORMATS[FMT["name"]][2]))


def render_frame(t, scenes=None):
    scenes = scenes or FORMATS[FMT["name"]][2]
    d = data()
    s, ctx = D.new_surface(W_, H_)
    D.background(ctx, W_, H_)
    acc = 0.0
    for name, dur, fn in scenes:
        if t < acc + dur or name == scenes[-1][0]:
            tl = t - acc
            fn(ctx, tl, d)
            edge = min(tl, dur - tl)          # quick dip at scene cuts
            if edge < 0.25:
                ctx.set_source_rgba(D.BG0[0], D.BG0[1], D.BG0[2], 1 - edge / 0.25)
                ctx.paint()
            break
        acc += dur
    return s


# ------------------------------------------------------------------------------------------- output
def _worker(args):
    fmt, i0, i1, path = args
    set_format(fmt)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W_}x{H_}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt",
           "yuv420p", "-profile:v", "high", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(D.frame_to_bytes(render_frame(i / FPS)))
    p.stdin.close()
    p.wait()
    return path


def video(fmt="wide", workers=4):
    name = {"wide": "plinko_optimal_play.mp4", "short": "plinko_optimal_play_SHORT.mp4"}[fmt]
    set_format(fmt)
    n = int(round(total_seconds() * FPS))
    chunks = np.linspace(0, n, workers * 3 + 1).astype(int)
    jobs = [(fmt, chunks[k], chunks[k + 1], f"{OUT}/seg_{fmt}_{k:02d}.mp4") for k in range(len(chunks) - 1)]
    with Pool(workers) as pool:
        segs = pool.map(_worker, jobs)
    lst = f"{OUT}/segs_{fmt}.txt"
    with open(lst, "w") as f:
        for sgm in segs:
            f.write(f"file '{os.path.basename(sgm)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", "-movflags", "+faststart", f"{OUT}/{name}"], check=True)
    for sgm in segs:
        os.remove(sgm)
    os.remove(lst)
    print("wrote", f"{OUT}/{name}", n, "frames", f"{n / FPS:.1f}s")


def stills(times, fmt="wide"):
    set_format(fmt)
    for tt in times:
        s = render_frame(float(tt))
        s.write_to_png(f"{OUT}/still_{fmt}_{float(tt):06.1f}.png")
        print("still", fmt, tt)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "video"
    w = int(args[args.index("--workers") + 1]) if "--workers" in args else 4
    if cmd == "prepare":
        prepare()
    elif cmd == "still":
        fmt = "short" if "--short" in args else "wide"
        stills([a for a in args[1:] if not a.startswith("--")], fmt)
    elif cmd == "video":
        video("wide", w)
    elif cmd == "short":
        video("short", w)
    else:
        raise SystemExit(__doc__)
