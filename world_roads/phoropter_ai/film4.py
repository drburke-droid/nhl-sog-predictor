"""The clinic film, simpler version: one idea per screen; the final exams start at the autorefractor.

    python film4.py list | render <name> | frame <name> <t> | all | assemble     -> out/ai_optometry_v2.mp4
"""
import os, sys, json, math, subprocess
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
from scenes4 import Exam4
from film3_intro import Hook3, fade_frame
from film4_intro import Toolbox4, PatientSays4, Wants4, Learn4
import film3 as F3

OUT = ui.OUT
XF = 0.5
DATA = os.path.join(HERE, "data")
TR = os.path.join(DATA, "traces")
AI_COL, TB_COL, AR_COL = (0x33, 0x8C, 0xED), (0xE7, 0x66, 0x24), (0x8A, 0x96, 0xA8)
TERM1_END, TERM2_END = 1_630_473, 2_697_295


def trace(name):
    return json.load(open(os.path.join(TR, f"{name}.json")))


def fmt_n(n):
    return f"{int(n):,}"


def mmss(m):
    a, b = divmod(int(round(m * 60)), 60)
    return f"{a}:{b:02d}"


# ======================================================================== simple centred text card
class Say(Section):
    """Big centred lines appearing one at a time. lines: [(t0, text, size, colour, weight)]"""

    def __init__(self, lines, dur, y0=None):
        self.lines, self.dur, self.y0 = lines, dur, y0

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        blocks = []
        for t0, s, size, col, wt in self.lines:
            blocks.append((t0, wrap(d, s, size, 1560, wt), size, col, wt))
        tot = sum(len(b[1]) * b[2] * 1.3 + 34 for b in blocks)
        y = self.y0 if self.y0 is not None else (H - tot) / 2
        for t0, lns, size, col, wt in blocks:
            al = fade(t, t0, 999, 0.6, 0.1) * a
            for ln in lns:
                text(d, (W / 2, y), ln, size, col, wt, "ma", al)
                y += size * 1.3
            y += 34
        return np.asarray(im)


# ======================================================================== the three changes to the clinic
class Changes(Section):
    ITEMS = [("It never turned the axis.", "The dial always started at 180°, where most astigmatism sits.",
              "Now the dial starts wherever the last patient left it."),
             ("It only turned the dial back to 180°.", "Then asked about power, never about the axis.",
              "A run of practice patients with astigmatism at every angle."),
             ("The clinic bought an autorefractor.", "Like a real practice.",
              "Every exam now starts at the autorefractor's reading.")]
    STEP = 8.0

    def __init__(self):
        self.dur = 3.0 + self.STEP * len(self.ITEMS) + 3.0
        self.sfx = [(3.0 + self.STEP * i, "tick") for i in range(len(self.ITEMS))]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 60, "Along the way, the clinic changed three times", 50, fade(t, 0.3, 999, 0.5, 0.1) * a)
        y = 200
        for i, (head, why, fix) in enumerate(self.ITEMS):
            t0 = 3.0 + self.STEP * i
            al = fade(t, t0, 999, 0.5, 0.1) * a
            al2 = fade(t, t0 + 3.2, 999, 0.5, 0.1) * a
            ui.badge(d, 150, y + 40, str(i + 1), GOLD, a=al, r=36)
            text(d, (220, y + 4), head, 40, INK, "Bold", a=al)
            text(d, (220, y + 60), why, 30, INK2, "Medium", a=al)
            text(d, (220, y + 112), "→  " + fix, 32, GOLD, "SemiBold", a=al2)
            y += 250
        return np.asarray(im)


# ======================================================================== the practice curve, all terms, one caption at a time
class Curve4(F3.PracticeCurve):
    def __init__(self, refs, notes, dur, draw_secs, x_from=None, x_to=None, title="LEARNING ON THE JOB", term3_end=None):
        super().__init__(refs, notes=(), dur=dur, draw_secs=draw_secs, x_from=x_from, x_to=x_to, title=title)
        self.cap_notes = notes
        self.term3_end = term3_end

    def frame(self, t):
        fr = super().frame(t)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a = fade(t, 0, self.dur, 0.5, 0.5)
        if self.term3_end and (self.x_to is None or self.x_to > self.term3_end) and (self.xs > self.term3_end).any():
            xm, _ = self.px(self.term3_end, 0)
            d.line([(xm, self.Y1), (xm, self.Y0)], fill=INK2 + (int(200 * a),), width=2)
            text(d, (xm + 8, self.Y1 + 80), "autorefractor", 17, INK2, "SemiBold", "la", a)
        prog = smooth((t - 1.0) / self.draw_secs)
        c0 = math.log10(self.x_from) if self.x_from else self.lx0
        c1 = math.log10(self.x_to) if self.x_to else self.lx1
        cut = c0 + (c1 - c0) * prog
        cur = [s for ex_n, s in self.cap_notes if cut >= math.log10(max(ex_n, 10 ** self.lx0))]
        if cur:
            big_title(d, W / 2, 985, cur[-1], 34, a, GOLD, "SemiBold")
        return np.asarray(im)


class LinearCurve(Section):
    """The later practice terms on a linear axis: where the clinic changed, and how the AI recovered each time."""

    def __init__(self, notes, marks, x_from, dur=24.0, draw_secs=13.0):
        self.xs, self.ys, self.mins, self.sh = F3.practice_curve()
        m = self.xs >= x_from
        self.xs, self.ys, self.sh = self.xs[m], self.ys[m], self.sh[m]
        self.notes, self.marks, self.dur, self.draw_secs = notes, marks, dur, draw_secs
        self.x0, self.x1 = x_from, self.xs.max()
        self.X0, self.X1, self.Y0, self.Y1 = 200, 1720, 880, 200
        self.v0, self.v1 = -120, 80

    def px(self, ex, v):
        return (self.X0 + (self.X1 - self.X0) * (ex - self.x0) / (self.x1 - self.x0),
                self.Y0 + (self.Y1 - self.Y0) * (v - self.v0) / (self.v1 - self.v0))

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "MORE PRACTICE", 40, INK, "Bold", a=a)
        text(d, (60, 86), "Points per patient while it practised (glasses minus chair time)", 24, MUTED, "Medium", a=a)
        X0, X1, Y0, Y1 = self.X0, self.X1, self.Y0, self.Y1
        for v in range(-100, 81, 50):
            _, y = self.px(self.x0, v)
            d.line([(X0, y), (X1, y)], fill=F3.GRID + (int(255 * a),), width=1)
            text(d, (X0 - 14, y), f"{v:+d}" if v else "0", 20, MUTED, "Medium", "rm", a)
        for e in np.arange(1.5e6, self.x1 + 1, 0.5e6):
            x, _ = self.px(e, 0)
            text(d, (x, Y0 + 14), f"{e / 1e6:.1f} million", 19, MUTED, "Medium", "ma", a)
        prog = smooth((t - 1.0) / self.draw_secs)
        cut = self.x0 + (self.x1 - self.x0) * prog
        for i, (xe, lab) in enumerate(self.marks):
            if cut >= xe:
                xm, _ = self.px(xe, 0)
                d.line([(xm, Y1 - 10), (xm, Y0)], fill=GOLD + (int(220 * a),), width=3)
                ui.badge(d, xm, Y1 - 40, str(i + 1), GOLD, a=a, r=26)
        m = self.xs <= cut
        P = [self.px(x, y) for x, y in zip(self.xs[m], self.ys[m])]
        if len(P) > 1:
            d.line(P, fill=AI_COL + (int(255 * a),), width=5, joint="curve")
            d.ellipse([P[-1][0] - 8, P[-1][1] - 8, P[-1][0] + 8, P[-1][1] + 8], fill=AI_COL + (int(255 * a),))
            text(d, (P[-1][0] + 14, P[-1][1] - 34), f"{self.ys[m][-1]:+.0f}", 30, INK, "Bold", "la", a)
        cur = [s_ for ex_n, s_ in self.notes if cut >= ex_n]
        if cur:
            big_title(d, W / 2, 960, cur[-1], 34, a, GOLD, "SemiBold")
        return np.asarray(im)


# ======================================================================== the textbook, short
class Textbook4(Section):
    STEPS = [("Start", "at the autorefractor reading"),
             ("1", "Fog with plus, then take it off a quarter at a time, reading the chart each time"),
             ("2", "Cross-cylinder: turn the axis, then set the power"),
             ("3", "Check the sphere again"),
             ("4", "The other eye"),
             ("5", "Fog both eyes, split the chart with prism, make the two eyes equal"),
             ("6", "Take the fog off both eyes together. Prescribe.")]
    dur = 30.0

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 46, "What the textbook says", 54, fade(t, 0.3, 999, 0.5, 0.1) * a, TB_COL)
        text(d, (W / 2, 120), "The standard sequence (Borish), as a script: same buttons, same patients.", 26, MUTED,
             "Medium", "ma", a)
        y = 210
        for i, (n, s) in enumerate(self.STEPS):
            al = fade(t, 1.4 + 2.2 * i, 999, 0.5, 0.1) * a
            text(d, (360, y), n, 36, TB_COL, "Bold", "ra", al)
            text(d, (400, y), s, 36, INK, "Medium", a=al)
            y += 100
        return np.asarray(im)


# ======================================================================== results, one number at a time
class Results4(Section):
    def __init__(self, R, beats):
        self.R, self.beats = R, beats
        self.STEP = 8.5
        self.dur = self.STEP * len(beats) + 2.0
        self.sfx = [(self.STEP * i + 0.8, "tick") for i in range(len(beats))]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        i = min(int(t // self.STEP), len(self.beats) - 1)
        tt = t - self.STEP * i
        title, sub, fn, fmt, lo, hi = self.beats[i]
        al = fade(tt, 0, self.STEP, 0.4, 0.4) * a if i < len(self.beats) - 1 else fade(tt, 0, 999, 0.4, 0.1) * a
        text(d, (W / 2, 40), f"{fmt_n(self.R['meta']['test_patients'])} NEW PATIENTS · EVERY EXAM STARTS AT THE "
                             f"AUTOREFRACTOR", 24, MUTED, "Bold", "ma", a)
        big_title(d, W / 2, 120, title, 56, al)
        text(d, (W / 2, 205), sub, 28, INK2, "Medium", "ma", al)
        rows = [("Autorefractor alone", self.R["ar"], AR_COL), ("Textbook", self.R["textbook"], TB_COL),
                ("AI", self.R["ai"], AI_COL)]
        g = smooth((tt - 0.6) / 1.2)
        X0, X1 = 640, 1700
        for j, (name, e, col) in enumerate(rows):
            y = 330 + j * 190
            v = fn(e)
            text(d, (X0 - 40, y + 50), name, 40, INK, "SemiBold", "rm", al)
            z = X0 + (X1 - X0) * (0 - lo) / (hi - lo)
            xv = X0 + (X1 - X0) * (v * g - lo) / (hi - lo)
            if abs(xv - z) > 1:
                d.rectangle([min(z, xv), y, max(z, xv), y + 100], fill=col + (int(255 * al),))
            lab = fmt(v)
            if v >= 0 or lo >= 0:
                text(d, (max(z, xv) + 20, y + 50), lab, 52, INK, "Bold", "lm", al * min(1, g * 2))
            else:
                text(d, (min(z, xv) - 20, y + 50), lab, 52, INK, "Bold", "rm", al * min(1, g * 2))
        return np.asarray(im)


class Real4(Section):
    def __init__(self, R, dur=36.0):
        self.R, self.dur = R, dur

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        R = self.R
        big_title(d, W / 2, 46, "And real optometrists?", 54, fade(t, 0.3, 999, 0.5, 0.1) * a)
        items = [("Two refractions of the same eye agree within 0.25 D",
                  [("Real optometrists", "about 80%"), ("Textbook script", f"{100 * R['textbook']['repeat']['within_025']:.0f}%"),
                   ("AI", f"{100 * R['ai']['repeat']['within_025']:.0f}%")]),
                 ("A subjective refraction takes",
                  [("Real optometrists", "about 5 min"), ("Textbook script", mmss(R["textbook"]["summary"]["minutes"])),
                   ("AI", mmss(R["ai"]["summary"]["minutes"]))])]
        y = 170
        for i, (h, vals) in enumerate(items):
            al = fade(t, 1.0 + 6.0 * i, 999, 0.6, 0.1) * a
            text(d, (W / 2, y), h, 36, INK, "Bold", "ma", al)
            y += 70
            for j, (who, v) in enumerate(vals):
                x = 360 + j * 600
                col = INK2 if j == 0 else (TB_COL if j == 1 else AI_COL)
                text(d, (x, y), who, 28, col, "SemiBold", "ma", al)
                text(d, (x, y + 44), v, 54, INK, "Bold", "ma", al)
            y += 210
        al = fade(t, 13.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 700, "Sources: Goss & Grosvenor 1996; Shah et al. 2009; Carracedo et al. 2018; Otero et al. 2019.",
                  22, al, MUTED, "Medium")
        al = fade(t, 17.0, 999, 0.6, 0.1) * a
        for k, ln in enumerate(wrap(d, "Our simulated patients are harder to refract than real ones, and our scorecard "
                                       "is made up. Compare the AI with the textbook script, not with your own "
                                       "optometrist.", 32, 1500, "SemiBold")):
            text(d, (W / 2, 800 + 44 * k), ln, 32, GOLD, "SemiBold", "ma", al)
        return np.asarray(im)


# ======================================================================== the film
def slow_steps(tr, n_first=3):
    """Play slowly: the first few uses of each kind of question in each stage, and every stage change."""
    seen, out = {}, set()
    for k, s in enumerate(tr["steps"]):
        key = (s["phase_before"], CU.BTN[s["a"]][5], s["a"] in F3.E3.__dict__ and s["a"])
        if s["a"] in (E3.NEXT, E3.RX):
            out.add(k)
            continue
        kk = (s["phase_before"], s["a"])
        seen[kk] = seen.get(kk, 0) + 1
        if s["a"] in (E3.CMP_P, E3.CMP_M, E3.JCC_AX, E3.JCC_PW, E3.BALANCE, E3.READ) and seen[kk] <= n_first:
            out.add(k)
    return out


def sections():
    R = json.load(open(os.path.join(DATA, "results_g.json")))
    RF = json.load(open(os.path.join(DATA, "results_f.json")))
    S = {}
    S["hook"] = Hook3()
    S["toolbox"] = Toolbox4()
    S["patient"] = PatientSays4()
    S["wants"] = Wants4()
    S["learn"] = Learn4()
    S["day1_title"] = Say([(0.5, "Day one", 72, INK, "Bold"),
                           (1.6, "Its first patient. It has never seen an eye.", 38, INK2, "Medium")], 6.0)
    d1 = trace("f_day1")
    S["day1_exam"] = Exam4(d1, "DAY ONE", "before any practice", slow=range(len(d1["steps"])),
                           notes={0: "It has no idea what any button does. It presses them at random."}, outro=6.0)
    st0 = RF["stages"][0]
    S["day1_grid"] = F3.OutcomeGrid("1,000 PATIENTS ON DAY ONE", "Random buttons, then “Prescribe”. One square per "
                                    "patient.", [("Day one", st0["sampled_outcomes"], st0["sampled"]["net"],
                                                  st0["sampled"]["minutes"])], dur=11.0,
                                    note="Most people need real glasses. Random ones aren't glasses.")
    refs = [("Plano for everyone", RF["plano"]["summary"]["net"], (0x6B, 0x76, 0x89)),
            ("Textbook (from plano)", RF["textbook"]["summary"]["net"], TB_COL)]
    term3_end = R["meta"].get("term3_end", F3.TERM3_END)
    S["curve1"] = Curve4(refs, [(20_000, "It starts asking “1 or 2?”"),
                                (50_000, "It stops handing out plano"),
                                (200_000, "It starts using the cross-cylinder"),
                                (1_000_000, "It balances the two eyes with prism, and gets faster")],
                         dur=26.0, draw_secs=16.0, x_to=TERM1_END, term3_end=term3_end)
    S["changes"] = Changes()
    S["curve2"] = LinearCurve([(1_400_000, "Fast, good at sphere, never turns the axis"),
                               (TERM1_END, "1: the dial starts anywhere. It crashes, then recovers"),
                               (TERM2_END, "2: astigmatism at every angle. It learns to refine the cylinder"),
                               (term3_end, "3: the autorefractor. Every exam starts closer")],
                              [(TERM1_END, "dial"), (TERM2_END, "astigmatism"), (term3_end, "autorefractor")],
                              x_from=1_400_000, dur=26.0, draw_secs=15.0)
    ai = trace("g_ai")
    tb = trace("g_textbook")
    NOTES.update(make_notes(R, ai, tb))
    S["ai_title"] = Say([(0.4, f"After {R['meta']['exams'] / 1e6:.1f} million patients", 60, INK, "Bold"),
                         (1.6, "The same first patient. Now the exam starts at the autorefractor reading.", 34, INK2,
                          "Medium")], 6.5)
    S["ai_exam"] = Exam4(ai, "THE AI", f"after {fmt_n(R['meta']['exams'])} practice patients",
                         notes=NOTES.get("ai_exam", {}), slow=slow_steps(ai, 2))
    S["tb_card"] = Textbook4()
    S["tb_exam"] = Exam4(tb, "THE TEXTBOOK", "the same patient", who="TEXTBOOK", who_col=TB_COL,
                         notes=NOTES.get("tb_exam", {}), slow=slow_steps(tb, 1), fast_beat=0.24)
    S["sidebyside"] = F3.SideBySide(ai, tb, NOTES.get("sidebyside", []), dur=34.0)
    S["results"] = Results4(R, [
        ("Time in the chair", "per patient", lambda e: e["summary"]["minutes"], mmss, 0, 10),
        ("Glasses within 0.25 D", "of the true prescription, every eye counted",
         lambda e: e["summary"]["vec_within_025"], lambda v: f"{100 * v:.0f}%", 0, 1),
        ("Happy patients", "sees clearly, comes back, refers a friend", lambda e: e["summary"]["happy"],
         lambda v: f"{100 * v:.0f}%", 0, 1),
        ("Points per patient", "glasses minus chair time", lambda e: e["summary"]["net"], lambda v: f"{v:+.0f}", -40,
         80)])
    S["real"] = Real4(R)
    S["outro"] = F3.Card("The fine print", [
        (0.6, "Every patient is simulated. The optics are real; the answers are a model; the scorecard is made up.",
         30, INK2, "Medium"),
        (3.0, "The AI is reinforcement learning on a 4-core computer. It has never examined a real person. This is a "
              "demonstration, not a clinical tool.", 30, INK2, "Medium"),
        (5.6, "If your vision changes, see your optometrist.", 34, GOLD, "SemiBold")], 15.0, y0=230)
    return S


NOTES = {}


def make_notes(R, ai, tb):
    def first(tr, cond, lo=0):
        st = tr["steps"]
        return next((k for k in range(lo, len(st)) if cond(st[k])), None)
    N = {}
    A = R["ai"]["habits"]["mean_count"]
    T = R["textbook"]["habits"]["mean_count"]
    k_pw = first(ai, lambda s: s["a"] == E3.JCC_PW)
    k_ax = first(ai, lambda s: s["a"] == E3.JCC_AX)
    k_fog = first(ai, lambda s: s["a"] == E3.SP100)
    k_next = first(ai, lambda s: s["a"] == E3.NEXT)
    k_bal = first(ai, lambda s: s["a"] == E3.BALANCE)
    k_ou = first(ai, lambda s: s["a"] == E3.OU_M, k_bal or 0)
    p = ai["patient"]
    N["ai_exam"] = {k: v for k, v in {
        0: f"It starts from the autorefractor's reading: right eye {fmt_rx(p['S0'], p['C0'], p['ax0'])}.",
        k_pw: "Cross-cylinder, power: “1 or 2?” More cylinder, or less?",
        k_ax: "Cross-cylinder, axis: the answer says which way to turn the dial.",
        k_fog: "Then +1.00 at a time, reading the chart, until it blurs. It leaves the eye fogged on purpose.",
        k_next: "The left eye.",
        k_bal: "Both eyes open, both fogged. Prism split: +0.25 to whichever eye is clearer…",
        k_ou: "…and −0.25 on both. It balances and unfogs at the same time."}.items() if k is not None}
    t_fog = 0
    t_jcc = first(tb, lambda s: s["a"] in (E3.JCC_AX, E3.JCC_PW))
    t_next = first(tb, lambda s: s["a"] == E3.NEXT)
    t_bal = first(tb, lambda s: s["a"] == E3.BALANCE)
    t_ou = first(tb, lambda s: s["a"] == E3.OU_M, t_bal or 0)
    N["tb_exam"] = {k: v for k, v in {
        t_fog: "Step 1: fog with plus, then take it off a quarter at a time, reading the chart after every click.",
        t_jcc: "Step 2: one line bigger, then the cross-cylinder: axis and power.",
        t_next: "Step 4: the left eye, all over again.",
        t_bal: "Step 5: fog both eyes, split the chart with prism, make the two eyes equal.",
        t_ou: "Step 6: take the fog off both eyes together. Then prescribe."}.items() if k is not None}
    N["sidebyside"] = [
        (12.0, f"Both start at the autorefractor. The textbook reads the chart {T['chart reading']:.0f} times per patient, "
               f"after almost every quarter step. The AI reads it {A['chart reading']:.0f} times."),
        (18.0, "The textbook finds each eye's best sphere on its own. The AI leaves each eye fogged and finds the sphere "
               "with both eyes open, balancing as it goes."),
        (24.0, f"Both refine the cylinder with the cross-cylinder: {T['cross-cylinder']:.0f} questions per patient for "
               f"the textbook, {A['cross-cylinder']:.0f} for the AI.")]
    return N


ORDER = ["hook", "toolbox", "patient", "wants", "learn", "day1_title", "day1_exam", "day1_grid", "curve1", "changes",
         "curve2", "ai_title", "ai_exam", "tb_card", "tb_exam", "sidebyside", "results", "real", "outro"]


def order():
    S = sections()
    return [n for n in ORDER if n in S], S


def render(name):
    _, S = order()
    S[name].render("g_" + name)


def render_all(names=None, par=4):
    names = names or order()[0]

    def one(nm):
        log = open(os.path.join(OUT, f"g_{nm}.log"), "w")
        r = subprocess.run([sys.executable, "film4.py", "render", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
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
    paths = [os.path.join(OUT, f"g_{s}.mp4") for s in names]
    ds = [film.dur(p) for p in paths]
    off = [0.0]
    for dd in ds[:-1]:
        off.append(off[-1] + dd - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(names, off))
    for s, o, dd in zip(names, off, ds):
        print(f"  {s:12s} {o:7.2f} + {dd:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video4.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    fc, last = [], "0:v"
    for k in range(1, len(names)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    sys.path.insert(0, os.path.join(ROOT, "flights"))
    import score as SC
    events = []
    for s in names:
        events += [(O[s] + tt, k) for tt, k in getattr(S[s], "sfx", [])]
    g = lambda k, dflt=0.0: O.get(k, dflt)
    energy = [(0, 0.3), (g("toolbox"), 0.35), (g("patient"), 0.35), (g("wants"), 0.4), (g("learn"), 0.35),
              (g("day1_title"), 0.45), (g("day1_exam"), 0.3), (g("day1_grid"), 0.4), (g("curve1"), 0.6),
              (g("changes"), 0.4), (g("curve2"), 0.6), (g("ai_title"), 0.5), (g("ai_exam"), 0.4), (g("tb_card"), 0.35),
              (g("tb_exam"), 0.35), (g("sidebyside"), 0.45), (g("results"), 0.8), (g("real"), 0.5), (g("outro"), 0.4),
              (total, 0.0)]
    sc = SC.Score(total, sorted(energy), [(g("curve1"), 0.6), (g("results"), 0.8)],
                  [(g("ai_title") - 2.0, g("ai_title")), (g("results") - 2.0, g("results"))],
                  transpose_at=g("ai_title"), final_at=g("outro") + 0.5, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.7
    mix = mix + film.sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score4.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "ai_optometry_v2.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], dd] for s, dd in zip(names, ds)}, "total": total},
              open(os.path.join(OUT, "timeline4.json"), "w"), indent=1)
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
        Image.fromarray(S[sys.argv[2]].frame(float(sys.argv[3]))).save(os.path.join(OUT, f"frame4_{sys.argv[2]}.png"))
    elif cmd == "all":
        render_all()
        assemble()
    elif cmd == "assemble":
        assemble()
