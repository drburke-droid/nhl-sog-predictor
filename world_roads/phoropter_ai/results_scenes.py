"""Result sections: the scoreboard, the learning curve, and how the AI spends an exam.

Chart conventions (dataviz skill): one highlight hue per entity (AI blue, textbook orange), everything else
neutral grey; direct labels; recessive grid; text in ink colours, never the series colour; no dual axes.
Series colours validated against the video background (#080a10) with validate_palette.js.
"""
import os, json, csv, math
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
import ui
from ui import W, H, BG, INK, INK2, MUTED, GOLD, CARD, LINE, PANEL, text, rr, smooth, fade
from scenes import Section, para, CHIP

HERE = ui.HERE
AI_BLUE = (0x33, 0x8C, 0xED)
TB_ORANGE = (0xE7, 0x66, 0x24)
GREY = (0x6B, 0x76, 0x89)
GRID = (0x26, 0x2D, 0x3A)
AXIS = (0x47, 0x4F, 0x5E)


def load_results(name):
    return json.load(open(os.path.join(HERE, "data", f"results_{name}.json")))


def load_curve(path):
    xs, ys, sh = [], [], []
    for r in csv.DictReader(open(path)):
        if r["eval_within"]:
            xs.append(int(r["exams"]))
            ys.append(float(r["eval_within"]))
            sh.append(float(r.get("shaping") or 0))
    return np.array(xs), np.array(ys), np.array(sh)


def pct(x):
    return f"{100 * x:.0f}%"


# ======================================================================== scoreboard
class Scoreboard(Section):
    """rows: [(label, summary dict, colour, emphasis)]"""

    def __init__(self, title, subtitle, rows, dur=14.0, note=None, highlight_col=None):
        self.title, self.subtitle, self.rows, self.dur, self.note = title, subtitle, rows, dur, note
        self.highlight_col = highlight_col
        self.sfx = [(0.9 + 0.35 * i, "tick") for i in range(len(rows))]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (80, 56), self.title, 44, INK, "Bold", a=a)
        text(d, (80, 114), self.subtitle, 26, MUTED, "Medium", a=a)
        x0, x1 = 700, 1240
        cols = [("EXACT", "exact", pct), ("MOVES", "steps", lambda v: f"{v:.0f}"),
                ("OVER-MINUSED\nUNDER 40", "over_minus_young", pct)]
        cx = [1420, 1580, 1760]
        top = 270
        text(d, (x0, top - 70), "WITHIN 0.25 D OF THE TRUE Rx", 20, MUTED, "SemiBold", a=a)
        for (lab, _, _), c in zip(cols, cx):
            for j, ln in enumerate(lab.split("\n")):
                hl = self.highlight_col == lab
                text(d, (c, top - 70 + 24 * j - (12 if "\n" in lab else 0)), ln, 20, GOLD if hl else MUTED,
                     "SemiBold", "ma", a)
        rh, gap = 50, 46
        # grid
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + (x1 - x0) * v
            d.line([(x, top - 22), (x, top + len(self.rows) * (rh + gap) - gap + 10)], fill=GRID + (int(255 * a),),
                   width=1)
            text(d, (x, top + len(self.rows) * (rh + gap) - gap + 18), pct(v), 18, MUTED, "Medium", "ma", a)
        for i, (lab, s, col, emph) in enumerate(self.rows):
            y = top + i * (rh + gap)
            ai = fade(t, 0.5 + 0.35 * i, 999, 0.4, 0.1) * a
            if emph:
                rr(d, [60, y - 18, 1860, y + rh + 18], 14, fill=(20, 28, 44, int(255 * ai)))
            text(d, (x0 - 24, y + rh / 2), lab, 28, INK if emph else INK2, "SemiBold" if emph else "Medium", "rm", ai)
            g = smooth((t - 0.9 - 0.35 * i) / 1.1)
            v = s["within_025"] * g
            xe = x0 + (x1 - x0) * v
            if xe > x0 + 2:
                d.rounded_rectangle([x0, y, xe, y + rh], 4, fill=col + (int(255 * ai),),
                                    corners=(False, True, True, False))
            text(d, (xe + 14, y + rh / 2), pct(v), 30, INK, "Bold" if emph else "SemiBold", "lm", ai * min(1, g * 3))
            for (_, key, f), c in zip(cols, cx):
                aa = ai * smooth((t - 1.6 - 0.35 * i) / 0.4)
                text(d, (c, y + rh / 2), f(s[key]), 30, INK if emph else INK2, "SemiBold", "mm", aa)
        if self.note:
            yb = top + len(self.rows) * (rh + gap) + 50
            para(d, 80, yb, self.note, 24, 1700, MUTED, "Medium", fade(t, 2.5, self.dur, 0.6, 0.5))
        return np.asarray(im)


# ======================================================================== learning curve
class Curve(Section):
    """series: [(label, xs, ys, colour, width)], refs: [(label, y, colour)]; notes: [(t0, t1, x_exams, text)]"""

    def __init__(self, title, subtitle, series, refs, dur=14.0, notes=(), draw_from=0.8, draw_secs=8.0, shade=None):
        self.title, self.subtitle, self.series, self.refs, self.dur = title, subtitle, series, refs, dur
        self.notes, self.draw_from, self.draw_secs, self.shade = notes, draw_from, draw_secs, shade
        self.X0, self.X1, self.Y0, self.Y1 = 260, 1520, 900, 230
        self.lx0, self.lx1 = 3.0, max(math.log10(max(s[1].max() for s in series)) + 0.15, 6.5)

    def px(self, ex, v):
        lx = math.log10(max(ex, 10 ** self.lx0))
        x = self.X0 + (self.X1 - self.X0) * (lx - self.lx0) / (self.lx1 - self.lx0)
        y = self.Y0 + (self.Y1 - self.Y0) * v
        return x, y

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (80, 56), self.title, 44, INK, "Bold", a=a)
        text(d, (80, 114), self.subtitle, 26, MUTED, "Medium", a=a)
        X0, X1, Y0, Y1 = self.X0, self.X1, self.Y0, self.Y1
        if self.shade:
            xa, _ = self.px(self.shade[0], 0)
            xb, _ = self.px(self.shade[1], 0)
            d.rectangle([xa, Y1 - 10, xb, Y0], fill=(255, 186, 80, int(22 * a)))
            text(d, ((xa + xb) / 2, Y1 - 6), self.shade[2], 20, GOLD, "SemiBold", "ma", a)
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            _, y = self.px(1, v)
            d.line([(X0, y), (X1, y)], fill=(GRID if v > 0 else AXIS) + (int(255 * a),), width=1 if v > 0 else 2)
            text(d, (X0 - 16, y), pct(v), 20, MUTED, "Medium", "rm", a)
        for e in range(int(math.ceil(self.lx0)), int(self.lx1) + 1):
            x, _ = self.px(10 ** e, 0)
            lab = {3: "1,000", 4: "10,000", 5: "100,000", 6: "1 million", 7: "10 million", 8: "100 million"}[e]
            d.line([(x, Y0), (x, Y0 + 8)], fill=AXIS + (int(255 * a),), width=2)
            text(d, (x, Y0 + 16), lab, 20, MUTED, "Medium", "ma", a)
        text(d, ((X0 + X1) / 2, Y0 + 56), "PRACTICE EXAMS (log scale)", 20, MUTED, "SemiBold", "ma", a)
        text(d, (X0 - 16, Y1 - 48), "WITHIN 0.25 D", 20, MUTED, "SemiBold", "ra", a)
        for lab, v, col in self.refs:
            _, y = self.px(1, v)
            for xx in range(X0, X1, 22):
                d.line([(xx, y), (min(xx + 12, X1), y)], fill=col + (int(220 * a),), width=3)
            text(d, (X1 + 14, y), f"{lab} {pct(v)}", 22, INK2, "SemiBold", "lm", a)
        u = smooth((t - self.draw_from) / self.draw_secs)
        for k, (lab, xs, ys, col, wid) in enumerate(self.series):
            lx = np.log10(np.maximum(xs, 10 ** self.lx0))
            lim = self.lx0 + (lx.max() - self.lx0) * (u if k == len(self.series) - 1 else 1.0)
            pts = [self.px(x, y) for x, y, l in zip(xs, ys, lx) if l <= lim + 1e-9]
            if len(pts) >= 2:
                d.line(pts, fill=col + (int(255 * a),), width=wid, joint="curve")
            if pts:
                hx, hy = pts[-1]
                r = wid + 4
                d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=col + (int(255 * a),), outline=BG + (255,), width=2)
                if k == len(self.series) - 1 or True:
                    text(d, (hx + 14, hy - 22), f"{lab} {pct(ys[len(pts) - 1])}", 24, INK, "SemiBold", "lm", a)
        for t0, t1, ex, msg, vy in self.notes:
            al = fade(t, t0, t1, 0.4, 0.4) * a
            if al <= 0:
                continue
            x, y = self.px(ex, vy)
            f = ui.font(24, "SemiBold")
            lines = msg.split("\n")
            tw = max(d.textlength(l, font=f) for l in lines)
            bx = min(max(x - tw / 2 - 20, 80), 1840 - tw - 40)
            by = y - 40 - 36 * len(lines)
            rr(d, [bx, by, bx + tw + 40, by + 36 * len(lines) + 20], 12, fill=PANEL + (int(230 * al),),
               outline=GOLD + (int(255 * al),), width=2)
            for j, l in enumerate(lines):
                text(d, (bx + 20, by + 12 + 36 * j), l, 24, INK, "SemiBold", a=al)
            d.line([(x, by + 36 * len(lines) + 20), (x, y - 6)], fill=GOLD + (int(255 * al),), width=2)
        return np.asarray(im)


# ======================================================================== how it spends an exam
def typical(actions):
    """The most common complete move sequence, and how many exams used exactly it."""
    c = Counter(tuple(a) for a in actions)
    seq, n = c.most_common(1)[0]
    return list(seq), n


class Habits(Section):
    """Typical exam strips plus a few counted habits. strips: [(label, seq, share, colour)];
    facts: [(big, small)]"""

    def __init__(self, title, subtitle, strips, facts, dur=16.0):
        self.title, self.subtitle, self.strips, self.facts, self.dur = title, subtitle, strips, facts, dur

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (80, 56), self.title, 44, INK, "Bold", a=a)
        text(d, (80, 114), self.subtitle, 26, MUTED, "Medium", a=a)
        y = 200
        f = ui.font(18, "SemiBold")
        for i, (lab, seq, share, col) in enumerate(self.strips):
            al = fade(t, 0.5 + 1.2 * i, 999, 0.5, 0.1) * a
            text(d, (80, y), lab, 26, INK, "SemiBold", a=al)
            text(d, (1840, y + 4), share, 20, MUTED, "Medium", "ra", al)
            y += 44
            x = 80
            for j, m in enumerate(seq):
                show = smooth((t - 0.8 - 1.2 * i - 0.06 * j) / 0.2) * al
                lab2, c = CHIP[m]
                w = d.textlength(lab2, font=f) + 18
                if x + w > 1840:
                    x, y = 80, y + 40
                rr(d, [x, y, x + w, y + 32], 8, fill=tuple(cc // 4 for cc in c) + (int(255 * show),),
                   outline=c + (int(255 * show),), width=2)
                d.text((x + w / 2, y + 16), lab2, font=f, fill=c + (int(255 * show),), anchor="mm")
                x += w + 6
            y += 76
        y = max(y + 10, 640)
        n = len(self.facts)
        cw = (1760 - 30 * (n - 1)) / n
        for i, (big, small) in enumerate(self.facts):
            al = fade(t, 3.0 + 1.0 * i, 999, 0.5, 0.1) * a
            x = 80 + i * (cw + 30)
            rr(d, [x, y, x + cw, y + 250], 18, fill=CARD + (int(255 * al),), outline=LINE + (int(255 * al),), width=2)
            text(d, (x + 30, y + 26), big, 56, GOLD, "Bold", a=al)
            para(d, x + 30, y + 110, small, 24, cw - 60, INK2, "Medium", al)
        return np.asarray(im)


# ======================================================================== prescriptions as vectors
def cyl_xy(C, ax):
    """Cylinder as an arrow in the 'cylinder plane': length |C|, direction 2 x axis (so 0 and 180 coincide)."""
    t = math.radians(2 * ax)
    return abs(C) * math.cos(t), abs(C) * math.sin(t)


def plane(d, cx, cy, R, rng_d, a, ring=0.25, labels=True, title=None):
    """Axes of a cylinder plane centred at (cx, cy), radius R pixels for rng_d dioptres."""
    A = int(255 * a)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=GRID + (A,), width=2)
    d.line([(cx - R, cy), (cx + R, cy)], fill=GRID + (A,), width=1)
    d.line([(cx, cy - R), (cx, cy + R)], fill=GRID + (A,), width=1)
    if ring:
        r = R * ring / rng_d
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=GOLD + (int(200 * a),), width=2)
    if labels:
        for txt, (dx, dy, anc) in (("axis 180", (R + 10, 0, "lm")), ("axis 45", (0, -R - 10, "md")),
                                   ("axis 90", (-R - 10, 0, "rm")), ("axis 135", (0, R + 10, "ma"))):
            text(d, (cx + dx, cy + dy), txt, 18, MUTED, "Medium", anc, a)
    if title:
        text(d, (cx, cy - R - 54), title, 26, INK, "SemiBold", "md", a)


def arrow(d, x0, y0, x1, y1, col, w=5, a=1.0, head=16):
    A = int(255 * a)
    d.line([(x0, y0), (x1, y1)], fill=col + (A,), width=w)
    ang = math.atan2(y1 - y0, x1 - x0)
    if math.hypot(x1 - x0, y1 - y0) > head:
        pts = [(x1, y1), (x1 - head * math.cos(ang - 0.4), y1 - head * math.sin(ang - 0.4)),
               (x1 - head * math.cos(ang + 0.4), y1 - head * math.sin(ang + 0.4))]
        d.polygon(pts, fill=col + (A,))


class VectorIdea(Section):
    """A prescription is a point: spherical equivalent, plus a cylinder arrow. The error is the gap."""

    def __init__(self, true_rx, ai_rx, ai_label, dur=24.0):
        self.T, self.Ai, self.lab, self.dur = true_rx, ai_rx, ai_label, dur

    def frame(self, t):
        from scenes import para
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (80, 56), "EVERY PRESCRIPTION IS A VECTOR", 44, INK, "Bold", a=a)
        text(d, (80, 114), "How the score measures an error", 26, MUTED, "Medium", a=a)
        S, C, ax = self.T
        S2, C2, ax2 = self.Ai
        se, se2 = S + C / 2, S2 + C2 / 2
        # --- sphere line (left)
        X, Y0, Y1 = 330, 260, 940
        lo, hi = se - 1.0, se + 1.0
        ypos = lambda v: Y1 + (Y0 - Y1) * (v - lo) / (hi - lo)
        a1 = fade(t, 0.6, 999, 0.6, 0.1) * a
        d.line([(X, Y0), (X, Y1)], fill=AXIS + (int(255 * a1),), width=3)
        for v in np.arange(math.ceil(lo * 4) / 4, hi + 1e-6, 0.25):
            y = ypos(v)
            major = abs(v * 2 - round(v * 2)) < 1e-6
            d.line([(X - (10 if major else 5), y), (X + (10 if major else 5), y)], fill=AXIS + (int(255 * a1),), width=2)
            if major:
                text(d, (X - 18, y), f"{v:+.2f}", 18, MUTED, "Medium", "rm", a1)
        text(d, (X, Y0 - 50), "SPHERE PART", 22, INK2, "SemiBold", "md", a1)
        text(d, (X, Y0 - 24), "spherical equivalent, S + C/2", 18, MUTED, "Medium", "md", a1)
        # --- cylinder plane (centre)
        cx, cy, R, rd = 960, 560, 280, 1.2
        a2 = fade(t, 3.0, 999, 0.6, 0.1) * a
        plane(d, cx, cy, R, rd, a2, ring=0)
        text(d, (cx, cy - R - 64), "CYLINDER PART", 22, INK2, "SemiBold", "md", a2)
        text(d, (cx, cy - R - 38), "arrow length = power, direction = 2 × axis", 18, MUTED, "Medium", "md", a2)
        k = R / rd
        tx, ty = cyl_xy(C, ax)
        px, py = cyl_xy(C2, ax2)
        # --- true Rx
        a3 = fade(t, 5.5, 999, 0.5, 0.1) * a
        arrow(d, cx, cy, cx + tx * k, cy - ty * k, INK, 5, a3)
        d.ellipse([X - 10, ypos(se) - 10, X + 10, ypos(se) + 10], fill=INK + (int(255 * a3),))
        text(d, (X + 84, ypos(se) + (-16 if se2 < se else 16)), f"true  {se:+.2f}", 22, INK, "SemiBold", "lm", a3)
        d.line([(cx - 200, cy + R + 64), (cx - 160, cy + R + 64)], fill=INK + (int(255 * a3),), width=5)
        text(d, (cx - 146, cy + R + 64), f"true Rx: cylinder {abs(C):.2f} × {ax:03.0f}", 22, INK, "SemiBold", "lm", a3)
        # --- AI Rx
        a4 = fade(t, 8.5, 999, 0.5, 0.1) * a
        arrow(d, cx, cy, cx + px * k, cy - py * k, AI_BLUE, 5, a4)
        d.ellipse([X - 10, ypos(se2) - 10, X + 10, ypos(se2) + 10], fill=AI_BLUE + (int(255 * a4),))
        text(d, (X + 84, ypos(se2) + (16 if se2 < se else -16)), f"{self.lab}  {se2:+.2f}", 22, INK2, "SemiBold", "lm", a4)
        d.line([(cx - 200, cy + R + 100), (cx - 160, cy + R + 100)], fill=AI_BLUE + (int(255 * a4),), width=5)
        text(d, (cx - 146, cy + R + 100), f"{self.lab}: cylinder {abs(C2):.2f} × {ax2:03.0f}", 22, INK2, "SemiBold",
             "lm", a4)
        # --- the gaps
        a5 = fade(t, 11.5, 999, 0.5, 0.1) * a
        if a5 > 0:
            for i in range(0, 100, 8):
                f0, f1 = i / 100, min(1, (i + 4) / 100)
                d.line([(cx + (px + (tx - px) * f0) * k, cy - (py + (ty - py) * f0) * k),
                        (cx + (px + (tx - px) * f1) * k, cy - (py + (ty - py) * f1) * k)], fill=GOLD + (int(255 * a5),),
                       width=5)
            d.line([(X + 40, ypos(se)), (X + 40, ypos(se2))], fill=GOLD + (int(255 * a5),), width=5)
        dse = se2 - se
        cres = math.hypot(px - tx, py - ty)
        err = math.hypot(dse, cres)
        r2 = lambda x: math.floor(abs(x) * 100 + 0.5) / 100
        x3 = 1380
        a6 = fade(t, 13.0, 999, 0.5, 0.1) * a
        text(d, (x3, 300), "SPHERE GAP", 20, MUTED, "SemiBold", a=a6)
        text(d, (x3, 328), f"{r2(dse):.2f} D", 44, GOLD, "SemiBold", a=a6)
        text(d, (x3, 410), "CYLINDER GAP (leftover cylinder)", 20, MUTED, "SemiBold", a=a6)
        text(d, (x3, 438), f"{r2(cres):.2f} D", 44, GOLD, "SemiBold", a=a6)
        a7 = fade(t, 15.5, 999, 0.5, 0.1) * a
        text(d, (x3, 530), "ERROR = LENGTH OF BOTH GAPS TOGETHER", 20, MUTED, "SemiBold", a=a7)
        text(d, (x3, 558), f"√({r2(dse):.2f}² + {r2(cres):.2f}²) = {r2(err):.2f} D", 40, INK, "SemiBold", a=a7)
        a8 = fade(t, 18.0, 999, 0.6, 0.1) * a
        para(d, x3, 660, "A 14° axis error on a 1.00 D cylinder leaves almost half a dioptre of cylinder.",
             26, 470, INK2, "Medium", a8)
        return np.asarray(im)


class ErrorClouds(Section):
    """Small multiples: every patient's leftover-cylinder vector (dots) and spherical-equivalent error
    (histogram), one panel per examiner. panels: [(title, vecs (n,3), colour, stats_line)]"""

    def __init__(self, title, subtitle, panels, dur=14.0, note=None):
        self.title, self.subtitle, self.panels, self.dur, self.note = title, subtitle, panels, dur, note

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (80, 56), self.title, 44, INK, "Bold", a=a)
        text(d, (80, 114), self.subtitle, 26, MUTED, "Medium", a=a)
        n = len(self.panels)
        W_ = 1760 / n
        R, rd = 190 if n >= 3 else 220, 1.0
        for i, (ttl, v, col, stat) in enumerate(self.panels):
            cx = 80 + W_ * (i + 0.5)
            cy = 470
            ai = fade(t, 0.4 + 0.5 * i, 999, 0.5, 0.1) * a
            plane(d, cx, cy, R, rd, ai, ring=0.25, labels=(i == 0), title=ttl)
            g = smooth((t - 0.9 - 0.5 * i) / 2.5)
            m = int(len(v) * g)
            if m > 0:
                lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dl = ImageDraw.Draw(lay)
                xs = cx + np.clip(v[:m, 1], -rd * 1.05, rd * 1.05) * R / rd
                ys = cy - np.clip(v[:m, 2], -rd * 1.05, rd * 1.05) * R / rd
                for x, y in zip(xs, ys):
                    dl.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=col + (int(70 * ai),))
                im.paste(lay, (0, 0), lay)
                d = ImageDraw.Draw(im, "RGBA")
            # spherical-equivalent error histogram
            hx0, hx1, hy = cx - R, cx + R, 920
            bins = np.linspace(-1.0, 1.0, 17)
            h, _ = np.histogram(np.clip(v[:, 0], -0.999, 0.999), bins)
            hmax = len(v) * 0.6
            bw = (hx1 - hx0) / (len(bins) - 1)
            for j, c in enumerate(h):
                bh = 150 * min(c / hmax, 1) * g
                if bh > 0.5:
                    d.rounded_rectangle([hx0 + j * bw + 1, hy - bh, hx0 + (j + 1) * bw - 1, hy], 3,
                                        fill=col + (int(220 * ai),), corners=(True, True, False, False))
            d.line([(hx0, hy), (hx1, hy)], fill=AXIS + (int(255 * ai),), width=2)
            for v_ in (-1, -0.5, 0, 0.5, 1):
                x = hx0 + (v_ + 1) / 2 * (hx1 - hx0)
                text(d, (x, hy + 10), f"{v_:+.1f}" if v_ else "0", 17, MUTED, "Medium", "ma", ai)
            if i == 0:
                text(d, (hx0 - 14, hy - 75), "SPHERICAL", 17, MUTED, "SemiBold", "rm", ai)
                text(d, (hx0 - 14, hy - 53), "EQUIVALENT", 17, MUTED, "SemiBold", "rm", ai)
                text(d, (hx0 - 14, hy - 31), "ERROR (D)", 17, MUTED, "SemiBold", "rm", ai)
            text(d, (cx, cy + R + 34), stat, 22, INK2, "SemiBold", "ma", ai * smooth((t - 2.5 - 0.5 * i) / 0.6))
        if self.note:
            text(d, (80, 1010), self.note, 22, MUTED, "Medium", a=fade(t, 2.0, self.dur, 0.6, 0.5))
        return np.asarray(im)
