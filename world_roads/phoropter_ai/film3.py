"""The clinic film: one AI that knows nothing about eyes opens an eye clinic.

    python film3.py list                 section names and lengths
    python film3.py render <name>        render one section to out/f_<name>.mp4
    python film3.py all                  render every section, then assemble
    python film3.py assemble             join sections and add music -> out/ai_optometry.mp4
"""
import os, sys, json, csv, math, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ui
import clinic
import env3 as E3
import clinic_ui as CU
from ui import W, H, BG, INK, INK2, MUTED, GOLD, GREEN, RED, CARD, LINE, PANEL, text, rr, fade, smooth, fmt_rx
from scenes import Section, para, wrap, big_title
from scenes3 import Exam3
from film3_intro import Hook3, Toolbox, PatientSays, Motivation, HowItLearns, fade_frame, ui_arrow

OUT = ui.OUT
XF = 0.5
DATA = os.path.join(HERE, "data")
TR = os.path.join(DATA, "traces")
AI_COL = (0x33, 0x8C, 0xED)
TB_COL = (0xE7, 0x66, 0x24)
GREY = (0x6B, 0x76, 0x89)
GRID = (0x26, 0x2D, 0x3A)
OUTC = [CU.OUT_COL["happy"], CU.OUT_COL["annoyed"], CU.OUT_COL["recheck"], CU.OUT_COL["review"], (120, 60, 70)]
OUTL = ["Happy", "Annoyed", "Rx check", "Bad review", "Walked out"]


def trace(name):
    return json.load(open(os.path.join(TR, f"{name}.json")))


def fmt_n(n):
    return f"{int(n):,}"


def pts(v):
    return f"{v:+.0f}"


# ======================================================================== text card
class Card(Section):
    def __init__(self, title, items, dur, kicker=None, y0=250):
        self.title, self.items, self.dur, self.kicker, self.y0 = title, items, dur, kicker, y0

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        y = self.y0
        if self.kicker:
            text(d, (960, y - 70), self.kicker, 26, GOLD, "SemiBold", "ma", a)
        text(d, (960, y), self.title, 60, INK, "Bold", "ma", a)
        y += 130
        for t0, s, size, col, wt in self.items:
            al = fade(t, t0, 999, 0.6, 0.1) * a
            for ln in wrap(d, s, size, 1400, wt):
                text(d, (960, y), ln, size, col, wt, "ma", al)
                y += size * 1.35
            y += 26
        return np.asarray(im)


# ======================================================================== outcome grids
class OutcomeGrid(Section):
    """panels: [(label, counts over the 5 outcomes, net points per patient, minutes)]: one square per patient."""

    def __init__(self, title, subtitle, panels, dur=14.0, n=1000, note=None):
        self.title, self.subtitle, self.panels, self.dur, self.n, self.note = title, subtitle, panels, dur, n, note
        self.sfx = [(0.8 + 1.2 * i, "tick") for i in range(len(panels))]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), self.title, 40, INK, "Bold", a=a)
        text(d, (60, 86), self.subtitle, 24, MUTED, "Medium", a=a)
        k = len(self.panels)
        pw = (1800 - 40 * (k - 1)) / k
        cols = 40 if k == 1 else (25 if k == 2 else 20)
        n = self.n
        rows = int(math.ceil(n / cols))
        cell = min((pw - 20) / cols, 560 / rows)
        for i, (lab, counts, net, mins) in enumerate(self.panels):
            x0 = 60 + i * (pw + 40)
            ai = fade(t, 0.6 + 1.2 * i, 999, 0.5, 0.1) * a
            text(d, (x0, 150), lab, 26, INK, "Bold", a=ai)
            c = np.array(counts, float)
            c = np.round(c / c.sum() * n).astype(int)
            c[0] += n - c.sum()
            colors = np.repeat(np.arange(5), c)
            fill = smooth((t - 0.8 - 1.2 * i) / 2.5)
            m = int(n * fill)
            gx0 = x0 + (pw - cols * cell) / 2
            for j in range(m):
                r_, c_ = divmod(j, cols)
                x, y = gx0 + c_ * cell, 200 + r_ * cell
                d.rectangle([x, y, x + cell - 2, y + cell - 2], fill=OUTC[colors[j]] + (int(255 * ai),))
            yb = 200 + rows * cell + 24
            aa = ai * smooth((t - 2.6 - 1.2 * i) / 0.5)
            for o in range(5):
                if c[o] == 0:
                    continue
                text(d, (x0, yb), "■", 22, OUTC[o], "Bold", a=aa)
                text(d, (x0 + 30, yb), f"{OUTL[o]}", 22, INK2, "Medium", a=aa)
                text(d, (x0 + pw, yb), f"{100 * c[o] / n:.0f}%", 22, INK, "SemiBold", "ra", aa)
                yb += 32
            yb += 10
            text(d, (x0, yb), "Points per patient", 22, INK2, "Medium", a=aa)
            text(d, (x0 + pw, yb - 4), pts(net), 32, INK, "Bold", "ra", aa)
            if mins is not None:
                yb += 40
                m_, s_ = divmod(int(round(mins * 60)), 60)
                text(d, (x0, yb), "Chair time per patient", 22, INK2, "Medium", a=aa)
                text(d, (x0 + pw, yb - 2), f"{m_}:{s_:02d}", 28, INK, "SemiBold", "ra", aa)
        if self.note:
            al = fade(t, 3.5, 999, 0.6, 0.1) * a
            big_title(d, W / 2, 1010, self.note, 28, al, GOLD, "SemiBold")
        return np.asarray(im)


# ======================================================================== the practice curve
TERM1_END = 1_630_473          # practice term 1 ended here (the axis dial always started at 180)


def practice_curve():
    """Both practice terms, joined: runs/f up to the end of term 1, then runs/f2."""
    xs, ys, mins, sh = [], [], [], []
    for path, lo, hi in ((os.path.join(HERE, "runs", "f", "logs", "train.csv"), 0, TERM1_END),
                         (os.path.join(HERE, "runs", "f2", "logs", "train.csv"), TERM1_END + 1, 10 ** 9)):
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r.get("train_points") and lo <= int(r["exams"]) <= hi:
                xs.append(int(r["exams"]))
                ys.append(float(r["train_points"]) - clinic.POINTS_PER_MIN * float(r["train_minutes"]))
                mins.append(float(r["train_minutes"]))
                sh.append(float(r["shaping"]))
    xs, ys, mins, sh = map(np.array, (xs, ys, mins, sh))
    k = 9
    ker = np.ones(k) / k
    pad = lambda v: np.concatenate([np.full(k // 2, v[0]), v, np.full(k // 2, v[-1])])
    sm = lambda v: np.convolve(pad(v), ker, "valid")
    t1 = xs <= TERM1_END                                  # smooth each term on its own (no blur across the change)
    ys = np.concatenate([sm(ys[t1]), sm(ys[~t1])]) if (~t1).any() else sm(ys)
    mins = np.concatenate([sm(mins[t1]), sm(mins[~t1])]) if (~t1).any() else sm(mins)
    return xs, ys, mins, sh


class PracticeCurve(Section):
    """Net clinic points per patient (points for the glasses minus chair time) while it practised."""

    def __init__(self, refs, notes=(), dur=34.0, draw_secs=14.0, x_from=None, x_to=None, title="LEARNING ON THE JOB"):
        self.xs, self.ys, self.mins, self.sh = practice_curve()
        self.refs, self.notes, self.dur, self.draw_secs = refs, notes, dur, draw_secs
        self.x_from, self.x_to, self.title = x_from, x_to, title
        self.X0, self.X1, self.Y0, self.Y1 = 230, 1500, 900, 220
        self.lx0, self.lx1 = 3.0, math.log10(self.xs.max()) + 0.08
        self.v0, self.v1 = -160, 80

    def px(self, ex, v):
        lx = math.log10(max(ex, 10 ** self.lx0))
        return (self.X0 + (self.X1 - self.X0) * (lx - self.lx0) / (self.lx1 - self.lx0),
                self.Y0 + (self.Y1 - self.Y0) * (v - self.v0) / (self.v1 - self.v0))

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), self.title, 40, INK, "Bold", a=a)
        text(d, (60, 86), "Points per patient while it practised (glasses minus chair time), every simulated patient a new one",
             24, MUTED, "Medium", a=a)
        X0, X1, Y0, Y1 = self.X0, self.X1, self.Y0, self.Y1
        # mentor shading: each stretch where the hint was on
        on = self.sh > 0
        j = 0
        while j < len(on):
            if on[j]:
                k2 = j
                while k2 + 1 < len(on) and on[k2 + 1]:
                    k2 += 1
                xa, _ = self.px(self.xs[j] if j else 10 ** self.lx0, 0)
                xb, _ = self.px(self.xs[k2], 0)
                if self.x_to is None or self.xs[j] <= self.x_to:
                    d.rectangle([xa, Y1, xb, Y0], fill=(255, 186, 80, int(18 * a)))
                    text(d, ((xa + xb) / 2, Y1 + 8), "mentor", 18, GOLD, "SemiBold", "ma", a * 0.9)
                j = k2 + 1
            else:
                j += 1
        if (self.xs > TERM1_END).any() and (self.x_to is None or self.x_to > TERM1_END):
            xm, _ = self.px(TERM1_END, 0)
            d.line([(xm, Y1), (xm, Y0)], fill=INK2 + (int(200 * a),), width=2)
            text(d, (xm - 8, Y1 + 36), "dial no longer", 17, INK2, "SemiBold", "ra", a)
            text(d, (xm - 8, Y1 + 58), "starts at 180°", 17, INK2, "SemiBold", "ra", a)
        for v in range(-150, 81, 50):
            _, y = self.px(1e4, v)
            d.line([(X0, y), (X1, y)], fill=GRID + (int(255 * a),), width=1)
            text(d, (X0 - 14, y), f"{v:+d}" if v else "0", 18, MUTED, "Medium", "rm", a)
        for e in range(3, int(self.lx1) + 1):
            x, _ = self.px(10 ** e, 0)
            text(d, (x, Y0 + 14), {3: "1,000", 4: "10,000", 5: "100,000", 6: "1 million", 7: "10 million"}[e], 18,
                 MUTED, "Medium", "ma", a)
        text(d, ((X0 + X1) / 2, Y0 + 46), "practice patients so far", 20, INK2, "Medium", "ma", a)
        for lab, v, col in self.refs:
            _, y = self.px(1e4, v)
            for xx in range(X0, X1, 18):
                d.line([(xx, y), (xx + 9, y)], fill=tuple(col) + (int(200 * a),), width=2)
            text(d, (X1 + 14, y), lab, 20, INK2, "SemiBold", "lm", a)
            text(d, (X1 + 14, y + 26), f"{v:+.0f}", 20, MUTED, "Medium", "lm", a)
        prog = smooth((t - 1.0) / self.draw_secs)
        lxs = np.log10(np.maximum(self.xs, 10 ** self.lx0))
        c0 = math.log10(self.x_from) if self.x_from else self.lx0
        c1 = math.log10(self.x_to) if self.x_to else self.lx1
        cut = c0 + (c1 - c0) * prog
        m = lxs <= cut
        P = [self.px(x, y) for x, y in zip(self.xs[m], self.ys[m])]
        if len(P) > 1:
            d.line(P, fill=AI_COL + (int(255 * a),), width=5, joint="curve")
            d.ellipse([P[-1][0] - 8, P[-1][1] - 8, P[-1][0] + 8, P[-1][1] + 8], fill=AI_COL + (int(255 * a),))
            ex_now = self.xs[m][-1]
            text(d, (P[-1][0] + 12, P[-1][1] - 30), f"{self.ys[m][-1]:+.0f}", 24, INK, "Bold", "la", a)
            text(d, (1560, 140), f"{fmt_n(ex_now)} patients", 26, INK, "Bold", "la", a)
        # notes at exam counts: (exams, text)
        for ex_n, s in self.notes:
            lx = math.log10(max(ex_n, 10 ** self.lx0))
            if cut >= lx:
                j = int(np.argmin(np.abs(self.xs - ex_n)))
                x, y = self.px(self.xs[j], self.ys[j])
                al = smooth((cut - lx) / 0.15) * a
                d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=INK + (int(255 * al),), width=2)
        # the current note at the right
        cur = [s for ex_n, s in self.notes if cut >= math.log10(max(ex_n, 10 ** self.lx0))]
        if cur:
            y = 230
            for s in cur[-4:]:
                y = para(d, 1560, y, s, 21, 330, INK2, "Medium", a, lh=1.25) + 18
        return np.asarray(im)


# ======================================================================== the textbook, spelled out
class TextbookCard(Section):
    dur = 36.0
    STEPS = [("Starting point", "Retinoscopy or an autorefractor. Here there is none: the script finds a starting sphere "
                                "by trying 1.00 D steps, as you would with a trial frame."),
             ("1  Best sphere (MPMVA)", "Fog +0.75 so the patient loses 2–3 lines, then take plus off 0.25 at a time, "
                                        "reading the chart. Keep the most plus that gives the best acuity."),
             ("2  Cross-cylinder", "One line above their best. Find any cylinder; axis first if it's 1.00 D or more, "
                                   "otherwise power first; axis steps narrow 15° → 5° → 2°; keep the spherical equivalent."),
             ("3  Sphere again", "If the cylinder moved by 0.50 D or the axis by 10° or more, redo the best sphere."),
             ("4  The other eye", "Steps 1–3 on the left eye."),
             ("5  Binocular balance", "+0.75 both eyes, 20/40 line, prism splits it in two. Add +0.25 to whichever "
                                      "eye is clearer until they match."),
             ("6  Both eyes together", "Take plus off both eyes 0.25 at a time; no more minus without better acuity. "
                                       "Prescribe.")]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "WHAT THE TEXTBOOK SAYS", 40, TB_COL, "Bold", a=a)
        text(d, (60, 86), "The standard sequence (Borish; as taught in Wilkinson 2016, University of Iowa), "
                          "scripted exactly. The script uses the same buttons and sees the same patients.", 22, MUTED,
             "Medium", a=a)
        y = 150
        for i, (h, s) in enumerate(self.STEPS):
            al = fade(t, 1.0 + 2.6 * i, 999, 0.5, 0.1) * a
            rr(d, [60, y, 1860, y + 112], 14, fill=CARD + (int(255 * al),), outline=(TB_COL if i else LINE) + (int(200 * al),),
               width=2)
            text(d, (90, y + 18), h, 26, INK, "Bold", a=al)
            para(d, 520, y + 20, s, 23, 1300, INK2, "Medium", al, lh=1.3)
            y += 124
        return np.asarray(im)


# ======================================================================== two exams side by side, in time
class Timelines(Section):
    """The same patient: every button the AI pressed and every button the textbook pressed, laid out in chair
    time, coloured by phoropter part. notes: [(t0, text)] commentary shown in turn."""

    def __init__(self, ai_trace, tb_trace, notes, dur=46.0):
        self.rows = [("AI", ai_trace, AI_COL), ("TEXTBOOK", tb_trace, TB_COL)]
        self.notes, self.dur = notes, dur
        self.tmax = max(tr["steps"][-1]["secs"] for _, tr, _ in self.rows)

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "THE SAME PATIENT: WHAT EACH ONE DID, MINUTE BY MINUTE", 36, INK, "Bold", a=a)
        X0, X1 = 260, 1860
        sx = (X1 - X0) / max(self.tmax, 60)
        prog = smooth((t - 0.8) / 7.0)
        for i, (lab, tr, col) in enumerate(self.rows):
            y0 = 170 + i * 250
            text(d, (60, y0 + 40), lab, 30, col, "Bold", "lm", a)
            last = tr["steps"][-1]
            m_, s_ = divmod(int(round(last["secs"])), 60)
            text(d, (60, y0 + 84), f"{m_}:{s_:02d}  ·  {len(tr['steps'])} buttons", 20, INK2, "Medium", "lm", a)
            err = last["err"]
            ok = all(e <= 0.25 + 1e-9 for e in err)
            text(d, (60, y0 + 116), "both eyes within 0.25 D" if ok else f"off {max(err):.2f} D", 20,
                 GREEN if ok else RED, "SemiBold", "lm", a)
            s0 = 0.0
            stage_x = []
            for s in tr["steps"]:
                dur_ = max(E3.SECS[s["a"]], 0.6)
                if s0 > self.tmax * prog:
                    break
                x = X0 + s0 * sx
                g = CU.BTN[s["a"]][5]
                h = 80 if g in ("chart", "sphere") and s["a"] in (E3.READ, E3.READ_LINE, E3.CMP_P, E3.CMP_M) else 56
                yy = y0 + (80 - h) / 2 + 10
                d.rectangle([x, yy, x + max(dur_ * sx - 1, 1.5), yy + h], fill=tuple(CU.col_of(s["a"])) + (int(235 * a),))
                if s["a"] == E3.NEXT:
                    stage_x.append(x)
                s0 += E3.SECS[s["a"]]
            # stage labels
            xs = [X0] + stage_x + [X0 + min(s0, self.tmax * prog) * sx]
            for j, lab2 in enumerate(("RIGHT EYE", "LEFT EYE", "BOTH")):
                if j + 1 < len(xs) and xs[j + 1] - xs[j] > 60:
                    text(d, ((xs[j] + xs[j + 1]) / 2, y0 + 112), lab2, 16, MUTED, "Bold", "ma", a)
                if j + 1 < len(xs) - 1:
                    d.line([(xs[j + 1], y0 - 4), (xs[j + 1], y0 + 104)], fill=INK + (int(200 * a),), width=2)
        for m in range(0, int(self.tmax / 60) + 2):
            x = X0 + m * 60 * sx
            if x <= X1 + 1:
                text(d, (x, 670), f"{m} min", 17, MUTED, "Medium", "ma", a)
        # legend
        lx = 260
        for g in ("sphere", "cyl", "axis", "jcc", "prism", "occ", "chart"):
            name = {"sphere": "sphere", "cyl": "cylinder", "axis": "axis", "jcc": "cross-cyl", "prism": "balance",
                    "occ": "next eye", "chart": "chart"}[g]
            text(d, (lx, 706), "■ " + name, 19, CU.GROUP_COL[g], "SemiBold", a=a)
            lx += 190
        text(d, (lx + 10, 708), "(tall = a question)", 17, MUTED, "Medium", a=a)
        # commentary
        y = 770
        for t0, s in self.notes:
            al = fade(t, t0, 999, 0.6, 0.1) * a
            if al <= 0:
                continue
            text(d, (64, y + 2), "•", 26, GOLD, "Bold", a=al)
            y = para(d, 96, y, s, 25, 1760, INK, "Medium", al, lh=1.3) + 14
        return np.asarray(im)


# ======================================================================== habits over 5,000 patients
class HabitsTable(Section):
    def __init__(self, R, notes, dur=30.0):
        self.R, self.notes, self.dur = R, notes, dur

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "HOW EACH ONE SPENDS AN EXAM", 40, INK, "Bold", a=a)
        text(d, (60, 86), f"Average per patient, {fmt_n(self.R['meta']['test_patients'])} new simulated patients", 24,
             MUTED, "Medium", a=a)
        H_ai, H_tb = self.R["ai"]["habits"], self.R["textbook"]["habits"]
        rows = [("Chart readings", "chart reading", CU.CHART), ("Sphere “1 or 2?”", "sphere 1-or-2", CU.SPHERE),
                ("Sphere lens clicks", "sphere lens", CU.SPHERE), ("Cross-cylinder “1 or 2?”", "cross-cylinder", CU.JCC),
                ("Cylinder clicks", "cylinder", CU.CYL), ("Axis turns", "axis", CU.AXIS),
                ("Balance questions", "balance", CU.PRISM)]
        X = (760, 1000)
        text(d, (X[0], 150), "AI", 26, AI_COL, "Bold", "ma", a)
        text(d, (X[1], 150), "TEXTBOOK", 26, TB_COL, "Bold", "ma", a)
        y = 200
        for i, (lab, key, col) in enumerate(rows):
            al = fade(t, 0.8 + 0.5 * i, 999, 0.4, 0.1) * a
            text(d, (100, y), "■", 24, col, "Bold", a=al)
            text(d, (140, y), lab, 26, INK2, "Medium", a=al)
            for x, Hh in zip(X, (H_ai, H_tb)):
                text(d, (x, y), f"{Hh['mean_count'][key]:.1f}", 28, INK, "SemiBold", "ma", al)
            y += 52
        al = fade(t, 4.6, 999, 0.4, 0.1) * a
        y += 10
        d.line([(100, y), (1080, y)], fill=LINE + (int(255 * al),), width=2)
        y += 20
        for lab, fn in (("Minutes: right eye", lambda h: f"{h['stage_minutes'][0]:.1f}"),
                        ("Minutes: left eye", lambda h: f"{h['stage_minutes'][1]:.1f}"),
                        ("Minutes: both eyes", lambda h: f"{h['stage_minutes'][2]:.1f}"),
                        ("Switches between sphere and cylinder work", lambda h: f"{h['sph_cyl_switches']:.1f}"),
                        ("Exams that reach the both-eyes stage", lambda h: f"{100 * h['reached_stage'][2]:.0f}%")):
            text(d, (140, y), lab, 24, INK2, "Medium", a=al)
            for x, Hh in zip(X, (H_ai, H_tb)):
                text(d, (x, y), fn(Hh), 26, INK, "SemiBold", "ma", al)
            y += 48
        y = 200
        for t0, s in self.notes:
            al = fade(t, t0, 999, 0.6, 0.1) * a
            if al <= 0:
                continue
            rr(d, [1160, y - 14, 1860, y + 150], 14, fill=CARD + (int(255 * al),), outline=LINE + (int(255 * al),), width=2)
            yy = para(d, 1186, y, s, 23, 650, INK, "Medium", al, lh=1.3)
            y = max(yy, y + 150) + 30
        return np.asarray(im)


# ======================================================================== results
class Results3(Section):
    """Plano, the textbook and the AI on the same new patients."""

    def __init__(self, R, dur=34.0, note=None):
        self.R, self.dur, self.note = R, dur, note
        self.sfx = [(1.0 + 0.4 * i, "tick") for i in range(3)]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        R = self.R
        text(d, (60, 30), f"{fmt_n(R['meta']['test_patients'])} NEW PATIENTS, ALL STARTING AT PLANO", 40, INK, "Bold", a=a)
        text(d, (60, 86), "None of them seen in practice. Same patients, same buttons, same rules for every examiner.",
             24, MUTED, "Medium", a=a)
        ex = [("Plano for everyone", R["plano"], GREY), ("Textbook script", R["textbook"], TB_COL),
              ("AI", R["ai"], AI_COL)]
        metrics = [("Points per patient", lambda s: s["net"], lambda v: f"{v:+.0f}", (-150, 100)),
                   ("Chair time", lambda s: s["minutes"], mmss, (0, 12)),
                   ("Eyes: spherical equivalent\nwithin 0.25 D", lambda s: s["se_within_025"], lambda v: f"{100 * v:.0f}%", (0, 1)),
                   ("Eyes: whole prescription\nwithin 0.25 D", lambda s: s["vec_within_025"], lambda v: f"{100 * v:.0f}%", (0, 1))]
        cw = 420
        for j, (lab, fn, f, (lo, hi)) in enumerate(metrics):
            x0 = 60 + j * (cw + 40)
            aj = fade(t, 0.6 + 0.8 * j, 999, 0.5, 0.1) * a
            for li, ln in enumerate(lab.split("\n")):
                text(d, (x0, 150 + 28 * li), ln, 23, INK2, "SemiBold", a=aj)
            for i, (name, e, col) in enumerate(ex):
                y = 230 + i * 74
                v = fn(e["summary"])
                g = smooth((t - 1.0 - 0.8 * j - 0.3 * i) / 1.0)
                z = (0 - lo) / (hi - lo) if lo < 0 else 0
                xz = x0 + 150 + (cw - 230) * z
                xv = x0 + 150 + (cw - 230) * (v * g - lo) / (hi - lo)
                text(d, (x0, y + 22), name.replace(" for everyone", "").replace(" script", ""), 20, INK2, "Medium", "lm", aj)
                if abs(xv - xz) > 1:
                    d.rectangle([min(xz, xv), y + 4, max(xz, xv), y + 40], fill=col + (int(255 * aj),))
                text(d, (max(xz, xv) + 10 if v >= 0 or lo >= 0 else min(xz, xv) - 10, y + 22), f(v * g if g < 1 else v),
                     22, INK, "Bold", "lm" if (v >= 0 or lo >= 0) else "rm", aj * min(1, g * 2))
        # outcome bars
        ao = fade(t, 5.0, 999, 0.6, 0.1) * a
        text(d, (60, 500), "WHAT HAPPENED NEXT (ONE OUTCOME SAMPLED PER PATIENT)", 22, MUTED, "Bold", a=ao)
        for i, (name, e, col) in enumerate(ex):
            y = 550 + i * 110
            text(d, (60, y + 30), name, 26, INK, "SemiBold", "lm", ao)
            c = np.array(e["outcomes"], float)
            c = c / c.sum()
            x = 460
            g = smooth((t - 5.4 - 0.4 * i) / 1.2)
            for o in range(5):
                w = 1380 * c[o] * g
                if w > 0.5:
                    d.rectangle([x, y, x + w, y + 60], fill=OUTC[o] + (int(255 * ao),))
                    if w > 60:
                        text(d, (x + w / 2, y + 30), f"{100 * c[o]:.0f}%", 22, BG, "Bold", "mm", ao)
                x += w
        lx = 460
        for o in range(5):
            text(d, (lx, 900), "■ " + OUTL[o], 21, OUTC[o], "SemiBold", a=ao)
            lx += 240
        if self.note:
            al = fade(t, 9.0, 999, 0.6, 0.1) * a
            para(d, 60, 960, self.note, 25, 1800, GOLD, "SemiBold", al, lh=1.3)
        return np.asarray(im)


# ======================================================================== the real world
class RealWorld(Section):
    def __init__(self, R, dur=46.0):
        self.R, self.dur = R, dur

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        R = self.R
        text(d, (60, 30), "HOW DOES THIS COMPARE WITH REAL OPTOMETRISTS?", 40, INK, "Bold", a=a)
        text(d, (60, 86), "Real studies can't know a patient's true prescription, so they measure how well two refractions "
                          "agree. We can do the same.", 24, MUTED, "Medium", a=a)
        rows = [
            ("Two refractions agree within 0.25 D (spherical equivalent)",
             [("Real optometrists", "about 80%", "Goss & Grosvenor 1996 review; Shah et al. 2009: 81%"),
              ("Simulated textbook script", f"{100 * R['textbook']['repeat']['within_025']:.0f}%", "same patients, fresh answers"),
              ("AI", f"{100 * R['ai']['repeat']['within_025']:.0f}%", "same patients, fresh answers")]),
            ("Time for a subjective refraction",
             [("Real optometrists", "about 5 min", "Carracedo 2018: 4:50; Otero 2019: 4:37; Kabat 2019: 5:38"),
              ("Simulated textbook script", mmss(R["textbook"]["summary"]["minutes"]), "from plano, no retinoscopy"),
              ("AI", mmss(R["ai"]["summary"]["minutes"]), "from plano, no retinoscopy")]),
            ("Glasses that come back",
             [("Real clinics", "about 2%", "Freeman & Evans 2010: 1.8%; Hrynchak 2006: 1.6%; Bist 2021: 2.1%"),
              ("Simulated textbook script", f"{100 * (R['textbook']['summary']['recheck'] + R['textbook']['summary']['review']):.0f}%",
               "our made-up scorecard; exams start from nothing"),
              ("AI", f"{100 * (R['ai']['summary']['recheck'] + R['ai']['summary']['review']):.0f}%",
               "our made-up scorecard; exams start from nothing")]),
        ]
        y = 160
        for i, (h, items) in enumerate(rows):
            al = fade(t, 0.8 + 4.0 * i, 999, 0.6, 0.1) * a
            rr(d, [60, y, 1860, y + 236], 16, fill=CARD + (int(255 * al),), outline=LINE + (int(255 * al),), width=2)
            text(d, (90, y + 18), h, 26, INK, "Bold", a=al)
            for j, (who, v, src) in enumerate(items):
                yy = y + 70 + j * 52
                col = INK2 if j == 0 else (TB_COL if j == 1 else AI_COL)
                text(d, (110, yy), who, 24, col, "SemiBold", a=al)
                text(d, (720, yy - 2), v, 28, INK, "Bold", a=al)
                text(d, (960, yy + 2), src, 20, MUTED, "Medium", a=al)
            y += 256
        al = fade(t, 14.0, 999, 0.6, 0.1) * a
        para(d, 60, 945, "Real exams start from retinoscopy or an autorefractor, and real clinicians use judgement a "
                         "script doesn't have. Ours start from nothing, with patients that are harder to read. So compare "
                         "the AI with the textbook script, not with your own optometrist.", 25, 1800, GOLD, "SemiBold",
             al, lh=1.3)
        return np.asarray(im)


def mmss(m):
    m_, s_ = divmod(int(round(m * 60)), 60)
    return f"{m_}:{s_:02d}"


# ======================================================================== the shortcut
class Shortcut(Section):
    """End of practice term 1: great at sphere, never turns the axis, because the dial always started at 180."""

    def __init__(self, R, dur=34.0):
        self.T = R["term1"]
        self.dur = dur
        self.n1 = R["meta"]["term1_end"]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        A, B = self.T["dial180"], self.T["random_dial"]
        text(d, (60, 30), "THE SHORTCUT", 40, GOLD, "Bold", a=a)
        text(d, (60, 86), f"After {fmt_n(self.n1)} practice patients: fast, and very good at sphere. But…", 24, MUTED,
             "Medium", a=a)
        rows = [("Eyes with the spherical equivalent within 0.25 D", f"{100 * A['summary']['se_within_025']:.0f}%", INK),
                ("Times it turned the axis, per exam", f"{A['axis_turns']:.2f}", RED),
                ("Cylinders it prescribed at exactly 180°", f"{100 * A['cyl_at_180']:.1f}%" if A['cyl_at_180'] > 0.99
                 else f"{100 * A['cyl_at_180']:.0f}%", RED),
                ("With-the-rule astigmatism (axis near 180°): within 0.25 D", f"{100 * A['wtr_within']:.0f}%", INK2),
                ("Against-the-rule (near 90°): within 0.25 D", f"{100 * A['atr_within']:.0f}%", RED),
                ("Oblique: within 0.25 D", f"{100 * A['obl_within']:.0f}%", RED)]
        y = 160
        for i, (lab, v, col) in enumerate(rows):
            al = fade(t, 0.8 + 1.2 * i, 999, 0.5, 0.1) * a
            text(d, (100, y), lab, 27, INK2, "Medium", a=al)
            text(d, (1180, y - 4), v, 36, col, "Bold", "ra", al)
            y += 64
        # the dial
        cx, cy, r = 1580, 320, 150
        al = fade(t, 2.0, 999, 0.6, 0.1) * a
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=CU.AXIS + (int(255 * al),), width=6)
        for deg in range(0, 180, 15):
            tt = math.radians(deg)
            for sg in (1, -1):
                d.line([(cx + sg * (r - 14) * math.cos(tt), cy - sg * (r - 14) * math.sin(tt)),
                        (cx + sg * r * math.cos(tt), cy - sg * r * math.sin(tt))], fill=INK2 + (int(200 * al),), width=3)
        d.line([(cx - r * 0.9, cy), (cx + r * 0.9, cy)], fill=CU.AXIS + (int(255 * al),), width=10)
        text(d, (cx, cy + r + 18), "the dial always started at 180°", 22, INK2, "SemiBold", "ma", al)
        al = fade(t, 9.5, 999, 0.6, 0.1) * a
        para(d, 100, 590, "The phoropter's axis dial started at 180° for every patient, and most young people's "
                          "astigmatism sits near 180°. Turning the dial only pays off after several moves in a row, so "
                          "it never found out it should.", 27, 1760, INK, "Medium", al, lh=1.35)
        al = fade(t, 16.0, 999, 0.6, 0.1) * a
        para(d, 100, 720, "So the clinic changed one thing: the dial now starts wherever the last patient left it, as it "
                          "would in a real practice. Overnight its score fell from "
                          f"{A['summary']['net']:+.0f} to {B['summary']['net']:+.0f} points per patient.", 27, 1760, INK,
             "Medium", al, lh=1.35)
        al = fade(t, 23.0, 999, 0.6, 0.1) * a
        para(d, 100, 870, "The mentor came back for a while. Same AI, more practice.", 30, 1760, GOLD, "SemiBold", al)
        return np.asarray(im)


# ======================================================================== step-by-step, side by side
ACT_KIND = {}
for _a in (E3.SP25, E3.SM25, E3.SP100, E3.SM100, E3.OU_P, E3.OU_M):
    ACT_KIND[_a] = "sphere"
ACT_KIND[E3.R_P] = ACT_KIND[E3.L_P] = "balance"
for _a in (E3.CMP_P, E3.CMP_M):
    ACT_KIND[_a] = "sphere?"
for _a in (E3.READ, E3.READ_LINE):
    ACT_KIND[_a] = "chart"
for _a in (E3.CYL_MORE, E3.CYL_LESS, E3.AXP2, E3.AXM2, E3.AXP5, E3.AXM5, E3.AXP15, E3.AXM15, E3.JCC_AX, E3.JCC_PW):
    ACT_KIND[_a] = "cyl"
ACT_KIND[E3.BALANCE] = "balance"


def summarize(tr):
    """Turn an exam into a short list of steps per stage: runs of sphere work, chart work, cylinder work and
    balancing, with what changed. Returns [(stage, [(kind, text)])]."""
    stages = [[], [], []]
    cur, run = 0, []

    def flush(stage, run):
        if not run:
            return
        kinds = [ACT_KIND.get(s["a"]) for s in run]
        k0 = run[0]
        b, a = k0["before"], run[-1]["after"]
        e = min(stage, 1)
        n = len(run)
        nq = sum(1 for s in run if s["a"] in (E3.READ, E3.READ_LINE, E3.CMP_P, E3.CMP_M, E3.JCC_AX, E3.JCC_PW, E3.BALANCE))
        if "cyl" in kinds:
            kind = "cyl"
            if stage < 2:
                txt = (f"Cylinder: {nq} cross-cyl question{'s' if nq != 1 else ''} → "
                       + (fmt_rx(*a[e]) if a[e][1] < 0 else "no cylinder"))
            else:
                txt = "Cylinder"
        elif "balance" in kinds:
            kind = "balance"
            nb = sum(1 for s in run if s["a"] == E3.BALANCE)
            if nb:
                txt = f"Balance: {nb} prism question{'s' if nb != 1 else ''} → R {ui.fmt_d(a[0][0])}, L {ui.fmt_d(a[1][0])}"
            else:
                txt = f"+0.25 to one eye, no prism → R {ui.fmt_d(a[0][0])}, L {ui.fmt_d(a[1][0])}"
        else:
            nchart = sum(1 for s in run if s["a"] in (E3.READ, E3.READ_LINE))
            ncmp = sum(1 for s in run if s["a"] in (E3.CMP_P, E3.CMP_M))
            parts = []
            if nchart:
                parts.append(f"{nchart} chart reading{'s' if nchart != 1 else ''}")
            if ncmp:
                parts.append(f"{ncmp} “1 or 2?”")
            kind = "sphere"
            if stage < 2:
                txt = f"Sphere: {' + '.join(parts) or 'lens clicks'} → {ui.fmt_d(a[e][0])}"
            else:
                txt = f"Both eyes: {' + '.join(parts) or 'lens clicks'} → R {ui.fmt_d(a[0][0])}, L {ui.fmt_d(a[1][0])}"
        stages[stage].append((kind, txt, n))

    for s in tr["steps"]:
        if s["a"] in (E3.NEXT, E3.RX):
            flush(cur, run)
            run = []
            cur = min(cur + 1, 2) if s["a"] == E3.NEXT else cur
            continue
        k = ACT_KIND.get(s["a"])
        prev = [ACT_KIND.get(x["a"]) for x in run]
        rtype = "cyl" if "cyl" in prev else ("balance" if "balance" in prev else ("sphere" if prev else None))
        brk = ((rtype == "cyl" and k in ("sphere?", "chart")) or (rtype == "sphere" and k in ("cyl", "balance")) or
               (rtype == "balance" and k in ("sphere?", "chart")))
        if run and brk:
            flush(cur, run)
            run = []
        run.append(s)
    flush(cur, run)
    return stages


class SideBySide(Section):
    """What the textbook does and what the AI did, stage by stage, for the same patient."""

    def __init__(self, ai_trace, tb_trace, notes=(), dur=40.0):
        self.ai, self.tb = summarize(ai_trace), summarize(tb_trace)
        self.ai_tr, self.tb_tr = ai_trace, tb_trace
        self.notes, self.dur = notes, dur

    def col(self, d, x, stages, col, t0, t, title, tr):
        a0 = fade(t, t0, 999, 0.5, 0.1)
        text(d, (x, 150), title, 30, col, "Bold", a=a0)
        last = tr["steps"][-1]
        text(d, (x, 192), f"{len(tr['steps'])} buttons · {mmss(last['secs'] / 60)}", 21, MUTED, "Medium", a=a0)
        y = 236
        k = 0
        for si, (lab, items) in enumerate(zip(("RIGHT EYE", "LEFT EYE", "BOTH EYES"), stages)):
            a1 = fade(t, t0 + 0.4 + 2.0 * si, 999, 0.5, 0.1)
            text(d, (x, y), lab, 19, MUTED, "Bold", a=a1)
            y += 32
            for kind, txt, n in items:
                c = {"sphere": CU.SPHERE, "cyl": CU.JCC, "balance": CU.PRISM}[kind]
                rr(d, [x, y + 4, x + 8, y + 30], 3, fill=tuple(c) + (int(255 * a1),))
                text(d, (x + 22, y + 2), txt, 22, INK, "Medium", a=a1)
                y += 36
                k += 1
            if not items:
                text(d, (x + 22, y + 2), "skipped", 22, MUTED, "Medium", a=a1)
                y += 36
            y += 12
        return y

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "STEP BY STEP: THE TEXTBOOK AND THE AI, SAME PATIENT", 36, INK, "Bold", a=a)
        text(d, (60, 82), "Blue: sphere and chart work.  Red: cylinder (cross-cylinder, axis, power).  Orange: balancing "
                          "the two eyes.", 22, MUTED, "Medium", a=a)
        y1 = self.col(d, 60, self.tb, TB_COL, 0.6, t, "THE TEXTBOOK", self.tb_tr)
        y2 = self.col(d, 980, self.ai, AI_COL, 6.5, t, "THE AI", self.ai_tr)
        y = max(y1, y2) + 10
        for t0, s in self.notes:
            al = fade(t, t0, 999, 0.6, 0.1) * a
            if al <= 0:
                continue
            text(d, (64, y + 2), "•", 26, GOLD, "Bold", a=al)
            y = para(d, 96, y, s, 24, 1760, GOLD, "SemiBold", al, lh=1.3) + 10
        return fade_frame(im, a)


# ======================================================================== the film
def load(name):
    return json.load(open(os.path.join(DATA, name)))


def sections():
    R = load("results_f.json")
    S = {}
    S["hook"] = Hook3()
    S["toolbox"] = Toolbox()
    S["patient"] = PatientSays()
    S["wants"] = Motivation()
    S["learn"] = HowItLearns()
    # day one
    S["day1_title"] = Card("Day one", [(0.8, "Its first patient. Every exam starts at plano: no autorefractor, no "
                                             "retinoscopy.", 34, INK2, "Medium")], 6.0, kicker="AI OPTOMETRY")
    d1 = trace("f_day1")
    S["day1_exam"] = Exam3(d1, "DAY ONE", "its first patient, before any practice", speed=1.6, outro=6.0,
                           notes={0: "It has no idea what any button does. It presses them at random."})
    st0 = R["stages"][0]
    S["day1_grid"] = OutcomeGrid("1,000 PATIENTS ON DAY ONE", "Random buttons, then “Prescribe”: two eyes in three leave "
                                 "within half a dioptre of plano. One square per patient.", [("Day one", st0["sampled_outcomes"], st0["sampled"]["net"],
                                                             st0["sampled"]["minutes"])], dur=12.0,
                                 note="Most people need real glasses. Plano-ish isn't glasses.")
    # learning
    tb_net = R["textbook"]["summary"]["net"]
    refs = [("Plano for everyone", R["plano"]["summary"]["net"], GREY), ("Textbook script", tb_net, TB_COL)]
    S["curve"] = PracticeCurve(refs, notes=NOTES.get("curve", []), x_to=TERM1_END, dur=30.0, draw_secs=13.0)
    S["shortcut"] = Shortcut(R)
    S["curve2"] = PracticeCurve(refs, notes=NOTES.get("curve2", []), x_from=TERM1_END, dur=18.0, draw_secs=8.0,
                                title="MORE PRACTICE, A REALISTIC DIAL")
    stg = {r["exams"]: r for r in R["stages"]}
    picks = [e for e in STAGE_PICKS if e in stg] or [r["exams"] for r in R["stages"][1::max(1, len(R["stages"]) // 3)]][:3]
    S["grids"] = OutcomeGrid("THE SAME CLINIC, LATER", "500 patients at each stage of practice, one square each",
                             [(f"After {fmt_n(e)} patients", stg[e]["sampled_outcomes"], stg[e]["sampled"]["net"],
                               stg[e]["sampled"]["minutes"]) for e in picks], dur=14.0, n=500)
    # the graduate
    ai_tr = trace("f_ai")
    tb_tr = trace("f_textbook")
    NOTES.update(make_notes(R, ai_tr, tb_tr))
    S["curve"].notes = NOTES.get("curve", [])
    S["curve2"].notes = NOTES.get("curve2", [])
    S["ai_title"] = Card("After " + fmt_n(R["meta"]["exams"]) + " patients",
                         [(0.8, "The same first patient, again.", 34, INK2, "Medium"),
                          (2.4, "Hidden from the AI: right −2.25 −1.00 × 170, left −2.50 −0.75 × 010.", 28, MUTED,
                           "Medium")], 7.0, kicker="GRADUATION DAY")
    S["ai_exam"] = Exam3(ai_tr, "THE AI", f"after {fmt_n(R['meta']['exams'])} practice patients", speed=0.8,
                         notes=NOTES.get("ai_exam", {}), outro=8.0)
    # the textbook
    S["tb_card"] = TextbookCard()
    k_l = next(k for k, s_ in enumerate(tb_tr["steps"]) if s_["a"] == E3.NEXT)
    k_b = next(k for k, s_ in enumerate(tb_tr["steps"]) if s_["a"] == E3.NEXT and k > k_l)
    S["tb_exam"] = Exam3(tb_tr, "THE TEXTBOOK", "the same patient, the textbook script", who="TEXTBOOK",
                         show_ai_probs=False, panel_title="THE SAME BUTTONS", speed=0.36,
                         fast=[(k_l + 2, k_b, 0.4)], notes=NOTES.get("tb_exam", {}), outro=7.0)
    S["sidebyside"] = SideBySide(ai_tr, tb_tr, NOTES.get("sidebyside", []))
    S["timelines"] = Timelines(ai_tr, tb_tr, NOTES.get("timelines", []))
    S["habits"] = HabitsTable(R, NOTES.get("habits", []))
    hp = lambda e: 100 * R[e]["outcomes"][0] / sum(R[e]["outcomes"])
    S["results"] = Results3(R, note=f"The textbook's glasses are better: {hp('textbook'):.0f}% happy patients against "
                                    f"{hp('ai'):.0f}%. The AI wins on points by saving "
                                    f"{R['textbook']['summary']['minutes'] - R['ai']['summary']['minutes']:.0f} minutes "
                                    f"per patient. That's what the scorecard paid for.")
    S["real"] = RealWorld(R)
    S["outro"] = Card("The fine print", [
        (0.6, "Every patient is simulated. The optics are real (power vectors, accommodation, a wave-optics model of the "
              "blur); the answers are a model. The clinic scorecard is made up.", 26, INK2, "Medium"),
        (3.0, "The AI is reinforcement learning (PPO): a small neural network that practised on simulated patients on a "
              "4-core computer. It has never examined a real person. This is a demonstration, not a clinical tool.", 26,
         INK2, "Medium"),
        (5.6, "If your vision changes, see your optometrist.", 30, GOLD, "SemiBold")], 16.0, y0=200)
    return S


NOTES = {}
STAGE_PICKS = [50_564, 500_299, 2_697_295]


def make_notes(R, ai_tr, tb_tr):
    """Captions and commentary, built from the results so every number matches the data."""
    A, T = R["ai"], R["textbook"]
    Ah, Th = A["habits"]["mean_count"], T["habits"]["mean_count"]
    As, Ts = A["summary"], T["summary"]
    st1 = R["term1"]["dial180"]
    N = {}
    # the graduate's exam: find the moments to caption
    steps = ai_tr["steps"]
    first = lambda cond, lo=0: next((k for k in range(lo, len(steps)) if cond(steps[k])), None)
    k_jump = first(lambda s: s["a"] == E3.SM100)
    k_axis = first(lambda s: s["a"] in (E3.AXP15, E3.AXM15, E3.AXP5, E3.AXM5, E3.AXP2, E3.AXM2))
    k_jcc = first(lambda s: s["a"] == E3.JCC_PW)
    k_plus = first(lambda s: s["a"] == E3.SP100, (k_jcc or 0))
    k_next = first(lambda s: s["a"] == E3.NEXT)
    k_bal = first(lambda s: s["a"] == E3.BALANCE)
    k_ou = first(lambda s: s["a"] == E3.OU_M, (k_bal or 0))
    k_cmp = first(lambda s: s["a"] in (E3.CMP_M, E3.CMP_P), (k_bal or 0))
    k_rx = len(steps) - 1
    ex = {}
    if k_jump is not None:
        ex[k_jump + 2] = "Big −1.00 steps, far past what this eye needs. Too much minus still looks clear: the eye just focuses harder."
    if k_axis is not None:
        ex[k_axis] = "It turns the dial back toward 180°, where most astigmatism sits. It never asks which way."
    if k_jcc is not None:
        ex[k_jcc] = "Cross-cylinder: more cylinder or less? It keeps adding while the answer is “one”."
    if k_plus is not None:
        ex[k_plus] = "Now plus back, 1.00 at a time, reading the chart, until it blurs. It leaves the eye fogged on purpose."
    if k_next is not None:
        ex[k_next] = "Left eye: the same routine."
    if k_bal is not None:
        ex[k_bal] = "Both eyes open, both fogged. Prism split: +0.25 to whichever eye is clearer…"
    if k_ou is not None:
        ex[k_ou] = "…and −0.25 on both. It balances the two eyes while it unfogs them."
    if k_cmp is not None:
        ex[k_cmp] = "Last: “clearer with −0.25?” It takes more minus only while the answer is a confident yes."
    N["ai_exam"] = ex
    tsteps = tb_tr["steps"]
    tfirst = lambda cond, lo=0: next((k for k in range(lo, len(tsteps)) if cond(tsteps[k])), None)
    kt_fog = tfirst(lambda s: s["a"] == E3.SP25)
    kt_jcc = tfirst(lambda s: s["a"] in (E3.JCC_PW, E3.JCC_AX))
    kt_ax = tfirst(lambda s: s["a"] == E3.JCC_AX)
    kt_next = tfirst(lambda s: s["a"] == E3.NEXT)
    kt_bal = tfirst(lambda s: s["a"] == E3.BALANCE)
    kt_ou = tfirst(lambda s: s["a"] == E3.OU_M, (kt_bal or 0))
    N["tb_exam"] = {k: v for k, v in {
        0: "Step 0: no autorefractor, so it finds a starting sphere 1.00 D at a time, reading the chart.",
        kt_fog: "Step 1: fog with plus, then take it off a quarter at a time, reading the chart after every click.",
        kt_jcc: "Step 2: one line bigger, then the cross-cylinder. First it searches for any cylinder.",
        kt_ax: "Cross-cylinder axis: “1 or 2?” says which way to turn the dial.",
        kt_next: "Step 4: the left eye, all over again.",
        kt_bal: "Step 5: fog both eyes, split the line with prism, balance them.",
        kt_ou: "Step 6: take the fog off both eyes together. Then prescribe."}.items() if k is not None}
    # side by side and timelines
    N["sidebyside"] = [
        (14.0, "The textbook finds each eye's sphere on its own, by reading the chart after every quarter step. The AI "
               "never looks for a one-eye endpoint: it leaves each eye fogged and finds the sphere with both eyes open."),
        (20.0, "Both eyes open is where the eyes focus most naturally (less of the extra focusing behind a phoropter), so "
               "its balanced unfog lands closer to the true sphere."),
        (26.0, "Astigmatism: the textbook asks the axis question and turns the dial. The AI only turns the dial back "
               "toward 180° and asks about power.")]
    N["timelines"] = [
        (8.0, f"The textbook reads the chart {Th['chart reading']:.0f} times per patient: at 8 seconds a reading, about "
              f"{Th['chart reading'] * 8 / 60:.0f} minutes. The AI reads it {Ah['chart reading']:.0f} times."),
        (15.0, f"Does interleaving sphere and cylinder cost time? The AI switches between them more often than the "
               f"textbook ({A['habits']['sph_cyl_switches']:.1f} times per exam against {T['habits']['sph_cyl_switches']:.1f}), "
               f"but its switches are one-second dial turns while it adds minus, not extra questions. Its questions "
               f"come in blocks: cylinder questions, then chart readings."),
        (24.0, "Where the AI does interleave is with both eyes open: balance and unfog in the same few clicks, instead "
               "of balancing first and unfogging after.")]
    N["habits"] = [
        (6.0, f"The AI asks the cross-cylinder power question {Ah['cross-cylinder']:.0f} times per patient and never the "
              f"axis question. The textbook asks {Th['cross-cylinder']:.0f} cross-cylinder questions, axis and power."),
        (12.0, f"It turns the axis only to bring the dial back near 180°. With-the-rule astigmatism gets corrected; "
               f"against-the-rule and oblique astigmatism mostly don't."),
        (18.0, f"It spends {Ah['balance']:.0f} prism questions per patient balancing the eyes; the textbook "
               f"{Th['balance']:.0f}.")]
    # the practice curve: what it learned when (from data/stages_f.json)
    N["curve"] = CURVE_NOTES
    N["curve2"] = [(1_700_000, "The realistic dial cost it about 20 points per patient."),
                   (2_000_000, "It learned to turn the dial back toward 180° before adding cylinder."),
                   (2_600_000, f"Final, on new patients: {As['net']:+.0f} points per patient, {mmss(As['minutes'])} per exam.")]
    return N


CURVE_NOTES = [(20_000, "About 20,000 patients: it starts asking “1 or 2?”, ten times an exam."),
               (50_000, "50,000: it never hands out plano any more. Half the eyes get the sphere within 0.25 D."),
               (200_000, "200,000: cross-cylinder power questions. Cylinder, always at 180°."),
               (1_000_000, "A million: it balances the two eyes with prism and drops most “1 or 2?” questions. "
                           "Exams down to 3 minutes.")]


ORDER = ["hook", "toolbox", "patient", "wants", "learn", "day1_title", "day1_exam", "day1_grid", "curve", "shortcut",
         "curve2", "grids",
         "ai_title", "ai_exam", "tb_card", "tb_exam", "sidebyside", "timelines", "habits", "results", "real", "outro"]


def order():
    S = sections()
    return [n for n in ORDER if n in S], S


def render(name):
    _, S = order()
    S[name].render("f_" + name)


def render_all(names=None, par=3):
    names = names or order()[0]

    def one(nm):
        log = open(os.path.join(OUT, f"f_{nm}.log"), "w")
        r = subprocess.run([sys.executable, "film3.py", "render", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        print(f"  {nm}: exit {r.returncode}", flush=True)
        return r.returncode

    _, S = order()
    names = sorted(names, key=lambda n: -S[n].dur)
    with ThreadPoolExecutor(par) as ex:
        codes = list(ex.map(one, names))
    assert not any(codes), "a section failed"


def assemble():
    import film
    names, S = order()
    paths = [os.path.join(OUT, f"f_{s}.mp4") for s in names]
    ds = [film.dur(p) for p in paths]
    off = [0.0]
    for dd in ds[:-1]:
        off.append(off[-1] + dd - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(names, off))
    for s, o, dd in zip(names, off, ds):
        print(f"  {s:12s} {o:7.2f} + {dd:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video3.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    fc, last = [], "0:v"
    for k in range(1, len(names)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    sys.path.insert(0, os.path.join(ROOT, "flights"))
    import score as SC
    events = []
    for s in names:
        events += [(O[s] + tt, k) for tt, k in getattr(S[s], "sfx", [])]
    g = lambda k, dflt=0.0: O.get(k, dflt)
    energy = [(0, 0.3), (g("toolbox"), 0.35), (g("patient"), 0.35), (g("wants"), 0.4), (g("learn"), 0.35),
              (g("day1_title"), 0.45), (g("day1_exam"), 0.3), (g("day1_grid"), 0.4), (g("curve"), 0.6),
              (g("shortcut"), 0.35), (g("curve2"), 0.6), (g("grids"), 0.65), (g("ai_title"), 0.5), (g("ai_exam"), 0.4), (g("tb_card"), 0.35), (g("tb_exam"), 0.35),
              (g("timelines"), 0.45), (g("habits"), 0.45), (g("results"), 0.8), (g("real"), 0.5), (g("outro"), 0.4),
              (total, 0.0)]
    energy = sorted(energy)
    impacts = [(g("curve"), 0.6), (g("results"), 0.8)]
    risers = [(g("ai_title") - 2.0, g("ai_title")), (g("results") - 2.0, g("results"))]
    sc = SC.Score(total, energy, impacts, risers, transpose_at=g("ai_title"), final_at=g("outro") + 0.5, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.7
    mix = mix + film.sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score3.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "ai_optometry.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], dd] for s, dd in zip(names, ds)}, "total": total},
              open(os.path.join(OUT, "timeline3.json"), "w"), indent=1)
    print("wrote", final)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        names, S = order()
        tot = 0
        for n in names:
            print(f"{n:12s} {S[n].dur:6.1f}s")
            tot += S[n].dur - XF
        print(f"total ~{tot + XF:.0f}s")
    elif cmd == "render":
        render(sys.argv[2])
    elif cmd == "frame":
        _, S = order()
        S[sys.argv[2]].frame(float(sys.argv[3]))
        Image.fromarray(S[sys.argv[2]].frame(float(sys.argv[3]))).save(os.path.join(OUT, f"frame_{sys.argv[2]}.png"))
    elif cmd == "all":
        render_all()
        assemble()
    elif cmd == "assemble":
        assemble()

