"""The clinic film, version 4 (env6): the third video's AI walks into a clinic with patients more like real ones
(no smile, a rough spin, only the occasional remark, some unreliable), and practises there. Its sequence against
the Borish method's; consistency on the same patients with a reliability dial; where the points come from.

    python film6.py list | render <name> | frame <name> <t> | all | assemble     -> out/ai_optometry_v4.mp4
"""
import os, sys, json, math, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import film5 as F5                                   # (also installs the version-5 buttons in the shared scenes)
from film5 import (Toolbox5, PatientSays5, Wants5, Learn5, Practice5, TextbookSteps, Sequences, Results5, Hook3, Say,
                   F3, CU, SQ, E5, ui, clinic, mmss, fmt_n, AI_COL, TB_COL, PL_COL, XF, OUT, fade_frame)
from film4 import Changes
from ui import W, H, BG, INK, INK2, MUTED, GOLD, GREEN, RED, CARD, LINE, PANEL, text, rr, fade, smooth
from scenes import Section, wrap, big_title
from scenes5 import Exam5

# env6: the cross-cylinder axis button also turns the axis toward the clearer side
_ax = (E5.JCC_AX, "Axis?", "which way?", "Cylinder AXIS: which way?",
       "cross-cylinder either side of the axis: the axis turns toward the clearer side", "jcc")
CU.BUTTONS[:] = [_ax if b[0] == E5.JCC_AX else b for b in CU.BUTTONS]
CU.BTN[E5.JCC_AX] = _ax

DATA = os.path.join(HERE, "data")
TR = os.path.join(DATA, "traces")
RUN = os.path.join(HERE, "runs", "m")
BORISH = "Borish method"
DIAL_NAMES = {"0.5": "RELIABLE", "1.0": "TYPICAL", "2.0": "UNRELIABLE"}


def trace(name):
    return json.load(open(os.path.join(TR, f"{name}.json")))


# ======================================================================== the opening
class Hook6(Hook3):
    dur = 16.0
    LINES = [(4.2, 150, "Last time, an AI worked out its own eye exam.", 60, INK, "Bold"),
             (6.4, 270, "But its patients made it easy.", 44, INK2, "SemiBold"),
             (7.8, 340, "They smiled whenever the letters got clearer,", 40, INK2, "Medium"),
             (9.2, 400, "and stopped a spinning axis right where it was sharpest.", 40, INK2, "Medium"),
             (11.4, 520, "Today: patients more like real ones.", 56, GOLD, "Bold")]

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.6, 0.5)
        lit = None
        for t0, b in self.flash:
            if t0 <= t < t0 + 0.32:
                lit = b
        g = CU.BTN[lit][5] if lit is not None else None
        hot = (g, int(self.rng.integers(0, 2))) if g not in (None, "chart", "rx") else None
        CU.draw_phoropter(im, 450, 30, 0.7, None, hot=hot, rings=True)
        d = ImageDraw.Draw(im, "RGBA")
        CU.draw_buttons(d, 40, 800, 1840, 260, pressed=lit, mask=None, title=False)
        rr(d, [0, 0, W, 790], 0, fill=BG + (int(222 * smooth((t - 4.0) / 1.0)),))
        for t0, y, s_, size, col, wt in self.LINES:
            big_title(d, W / 2, y, s_, size, fade(t, t0, 99, 0.6, 0.1), col, wt)
        return fade_frame(im, a_all)


class Changes6(Changes):
    ITEMS = [("Patients don't smile.", "Last time, the patient smiled whenever the letters got clearer: a running score.",
              "Now and then one says “Oh, that's clear!” or “Hmm, not as clear.” That's all."),
             ("Spinning the axis was too easy.", "The patient stopped the dial almost exactly where it was sharpest.",
              "Now “stop!” comes late and rough, as it does watching a turning dial."),
             ("Every patient answered alike.", "The same amount of guessing for everyone.",
              "Now some patients are steady and some aren't: a reliability dial.")]

    def frame(self, t):
        fr = super().frame(t)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a = fade(t, 0, self.dur, 0.5, 0.5)
        rr(d, [0, 20, W, 130], 0, fill=BG + (255,))
        big_title(d, W / 2, 60, "What changed in the clinic", 50, fade(t, 0.3, 999, 0.5, 0.1) * a)
        return np.asarray(im)


class Grid6(F3.OutcomeGrid):
    """The outcome grid with its note under the subtitle (two panels fill the bottom of the frame)."""

    def frame(self, t):
        note, self.note = self.note, None
        fr = super().frame(t)
        self.note = note
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        if note:
            al = fade(t, 3.5, 999, 0.6, 0.1) * fade(t, 0, self.dur, 0.5, 0.5)
            text(d, (W - 60, 92), note, 26, GOLD, "SemiBold", "ra", al)
        return np.asarray(im)


# ======================================================================== explainers
class PatientSays6(PatientSays5):
    ANSWERS = [("“One, much clearer.”  “Two… a little.”  “They look the same.”", CU.SPHERE),
               ("“I can read down to 20/40.”", CU.CHART),
               ("“Stop! …there, I think.”  (as the axis spins: a rough answer)", CU.AXIS),
               ("“My left eye is still blurry.”  (the final check)", CU.RXC),
               ("Now and then: “Oh, that’s clear!”  “Hmm, not as clear.”", INK2)]
    GOLD_Y = 930
    T_INTRO = 10.0


class Learn6(Learn5):
    STEPS = [Learn5.STEPS[0],
             ("2", "It gets the score", "Points for the glasses, minus the chair time. Along the way, a patient's "
                                        "occasional “Oh, that’s clear!” or “Hmm, that’s not as clear” counts a little too."),
             Learn5.STEPS[2]]
    LAST = "No smile, no mentor: just a remark now and then, and a patient who won't leave seeing blurry."


class BorishSteps(TextbookSteps):
    TITLE = "The Borish method"
    SUB = "The standard sequence taught in optometry school. Written as a script: same buttons, same patients as the AI."
    STAGES = [("RIGHT EYE  ·  left eye covered", [
                  ("1", "sph_big", "Find the sphere: trial lenses, then 1.00 D steps, reading the chart"),
                  ("2", "fog", "Best sphere: fog with plus until the letters blur, then off a quarter at a time"),
                  ("3", "chk", "Astigmatism check: cross-cylinder at 180/90, then at 45/135"),
                  ("4", "axis", "Cylinder AXIS: flip either side of the axis, turn toward the clearer one"),
                  ("5", "power", "Cylinder POWER: flip along the axis, more or less  (small cylinder: power first)"),
                  ("6", "sph_fine", "Best sphere again, then “+0.25: 1 or 2?”  Keep the plus if it's no worse")]),
              ("LEFT EYE  ·  right eye covered", [("", "sph_fine", "Steps 1 to 6 again")]),
              ("BOTH EYES OPEN", [("7", "balance", "Fog both eyes, split the chart with prism, make the two eyes equal"),
                                  ("8", "ou_clear", "Clear the fog together, then “+0.25: 1 or 2?” once more"),
                                  ("", "rx_no", "Final check “still blurry”? Redo that eye, then both eyes again")])]


class Sequences6(Sequences):
    TB_NAME = "THE BORISH METHOD"
    TITLE = "The same patient: the Borish method's steps and the AI's"


class Results6(Results5):
    NAMES = ("Day one", BORISH, "AI")


# ======================================================================== where the points come from
class PointsSplit(Section):
    """Points per patient, split: what the glasses earn (happy, annoyed, re-checks, reviews, walkouts) and what
    the chair time costs (4 a minute). The net is the difference."""

    def __init__(self, R, dur=20.0):
        self.R, self.dur = R, dur
        self.rows = [("Day one", R["day1"], PL_COL), (BORISH, R["textbook"], TB_COL), ("AI", R["ai"], AI_COL)]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (W / 2, 40), f"{fmt_n(self.R['meta']['test_patients'])} NEW PATIENTS  ·  POINTS PER PATIENT", 24, MUTED,
             "Bold", "ma", a)
        big_title(d, W / 2, 100, "Where the points come from", 56, fade(t, 0.3, 999, 0.5, 0.1) * a)
        X0, X1, lo, hi = 700, 1500, -180, 100
        px = lambda v: X0 + (X1 - X0) * (v - lo) / (hi - lo)
        z = px(0)
        d.line([(z, 230), (z, 860)], fill=MUTED + (int(160 * a),), width=2)
        for j, (name, e, col) in enumerate(self.rows):
            y = 250 + j * 210
            al = fade(t, 1.0 + 2.5 * j, 999, 0.5, 0.1) * a
            s = e["summary"]
            glasses, time_ = s["points"], -clinic.POINTS_PER_MIN * s["minutes"]
            g1 = smooth((t - 1.4 - 2.5 * j) / 1.0)
            g2 = smooth((t - 2.4 - 2.5 * j) / 1.0)
            text(d, (X0 - 60, y + 70), name, 40, col, "Bold", "rm", al)
            # the glasses
            xg = px(glasses * g1)
            gcol = GREEN if glasses >= 0 else (176, 122, 122)          # walkouts and bad reviews: below zero
            d.rectangle([min(z, xg), y, max(z, xg), y + 56], fill=gcol + (int(220 * al),))
            lab = f"glasses {glasses:+.0f}"
            if glasses >= 0:
                text(d, (max(z, xg) + 14, y + 28), lab, 28, INK, "SemiBold", "lm", al * min(1, g1 * 2))
            else:
                text(d, (min(z, xg) - 14, y + 28), lab, 28, INK, "SemiBold", "rm", al * min(1, g1 * 2))
            # the chair time
            xt = px(time_ * g2)
            d.rectangle([xt, y + 70, z, y + 126], fill=RED + (int(220 * al),))
            text(d, (xt - 14, y + 98), f"{mmss(s['minutes'])} in the chair  {time_:+.0f}", 28, INK, "SemiBold", "rm",
                 al * min(1, g2 * 2))
            text(d, (1840, y + 63), f"= {s['net']:+.0f}", 56, INK, "Bold", "rm", al * min(1, g2 * 2))
        tb = self.R["textbook"]["summary"]
        tt = -clinic.POINTS_PER_MIN * tb["minutes"]
        if tb["net"] < 0:
            s_ = (f"The {BORISH}'s glasses earn {tb['points']:+.0f} a patient. Its {mmss(tb['minutes'])} in the chair "
                  f"costs {abs(tt):.0f}: that's why it ends up below zero.")
        else:
            s_ = (f"The {BORISH}'s glasses earn {tb['points']:+.0f} a patient, but {mmss(tb['minutes'])} in the chair "
                  f"costs {abs(tt):.0f} of that.")
        al = fade(t, 9.5, 999, 0.6, 0.1) * a
        for k, ln in enumerate(wrap(d, s_ + " Every minute costs 4 points.", 32, 1600, "SemiBold")):
            text(d, (W / 2, 900 + 44 * k), ln, 32, GOLD, "SemiBold", "ma", al)
        return np.asarray(im)


# ======================================================================== consistency: the same patient twice
class Consistency(Section):
    """Each patient examined twice (fresh answers, same eyes): how often the two prescriptions agree within
    0.25 D (spherical equivalent, every eye), with the patients made more or less reliable."""

    def __init__(self, R, dur=26.0, key="se_025"):
        self.R, self.dur, self.key = R, dur, key
        self.dials = [k for k in ("0.5", "1.0", "2.0") if k in R["ai"]["by_dial"]]

    @staticmethod
    def dial_icon(d, cx, cy, r, frac, al):
        d.arc([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=INK2 + (int(255 * al),), width=5)
        ang = math.pi * (1 - frac)
        d.line([(cx, cy), (cx + 0.85 * r * math.cos(ang), cy - 0.85 * r * math.sin(ang))],
               fill=GOLD + (int(255 * al),), width=6)
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=GOLD + (int(255 * al),))

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 34, "Same patient, tested twice", 56, fade(t, 0.3, 999, 0.5, 0.1) * a)
        text(d, (W / 2, 112), "Each patient examined twice, fresh answers. Do the two prescriptions agree within 0.25 D?",
             28, INK2, "Medium", "ma", fade(t, 0.8, 999, 0.5, 0.1) * a)
        base, top = 860, 380
        for j, dk in enumerate(self.dials):
            cx = 420 + 540 * j
            al = fade(t, 2.0 + 4.0 * j, 999, 0.5, 0.1) * a
            self.dial_icon(d, cx, 250, 50, {"0.5": 0.15, "1.0": 0.5, "2.0": 0.85}[dk], al)
            text(d, (cx, 280), DIAL_NAMES.get(dk, dk) + " PATIENTS", 26, INK, "Bold", "ma", al)
            for k, (who, col) in enumerate(((BORISH, TB_COL), ("AI", AI_COL))):
                ent = self.R["textbook" if k == 0 else "ai"]["by_dial"][dk]["agree"]
                v = ent[self.key]
                g = smooth((t - 2.4 - 4.0 * j - 0.4 * k) / 1.2)
                x0 = cx - 190 + k * 200
                hgt = (base - top) * v * g
                d.rectangle([x0, base - hgt, x0 + 180, base], fill=tuple(col) + (int(255 * al),))
                text(d, (x0 + 90, base - hgt - 14), f"{100 * v:.0f}%", 44, INK, "Bold", "md", al * min(1, g * 2))
                text(d, (x0 + 90, base + 14), who if k else "Borish", 26, col, "SemiBold", "ma", al)
        al = fade(t, 15.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 940, "Real optometrists, real patients: about 80% (published studies).", 32, al, INK2,
                  "Medium")
        al = fade(t, 18.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 996, "Consistent isn't the same as right (plano for everyone would agree every time): "
                                 "the happy patients say whether it's right.", 30, al, GOLD, "SemiBold")
        return np.asarray(im)


class Real6(Section):
    def __init__(self, R, dur=30.0):
        self.R, self.dur = R, dur

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        R = self.R
        big_title(d, W / 2, 46, "And real optometrists?", 54, fade(t, 0.3, 999, 0.5, 0.1) * a)
        items = [("Two refractions of the same eye agree within 0.25 D (typical patients)",
                  [("Real optometrists", "about 80%"), ("Borish script", f"{100 * R['textbook']['repeat']['within_025']:.0f}%"),
                   ("AI", f"{100 * R['ai']['repeat']['within_025']:.0f}%")]),
                 ("A subjective refraction takes",
                  [("Real optometrists", "about 5 min"), ("Borish script", mmss(R["textbook"]["summary"]["minutes"])),
                   ("AI", mmss(R["ai"]["summary"]["minutes"]))])]
        y = 170
        for i, (h_, vals) in enumerate(items):
            al = fade(t, 1.0 + 6.0 * i, 999, 0.6, 0.1) * a
            text(d, (W / 2, y), h_, 36, INK, "Bold", "ma", al)
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
        al = fade(t, 16.0, 999, 0.6, 0.1) * a
        for k, ln in enumerate(wrap(d, "The script reads the chart after almost every click, so it is slower than an "
                                       "optometrist. Our patients are simulated and our scorecard is made up: compare "
                                       "the AI with the script, not with your own optometrist.", 32, 1560, "SemiBold")):
            text(d, (W / 2, 790 + 44 * k), ln, 32, GOLD, "SemiBold", "ma", al)
        return np.asarray(im)


# ======================================================================== captions
def axis_err(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def make_notes(R, ai, tb, d1):
    """Captions, placed by finding the moments in the traces and the practice checkpoints."""
    def first(tr, cond, lo=0):
        st = tr["steps"]
        return next((k for k in range(lo, len(st)) if cond(st[k])), None)

    def nth_next(tr, n):
        ks = [k for k, s in enumerate(tr["steps"]) if s["a"] == E5.NEXT]
        return ks[n] if len(ks) > n else None

    def true_axis(tr, k):
        eye = tr["steps"][k]["phase"]
        return tr["patient"]["axis" if eye == 0 else "axis_os"]

    N = {}
    # ---- the Borish exam
    t_trial = first(tb, lambda s: s["a"] == E5.SM100)
    t_fog = first(tb, lambda s: s["a"] == E5.SP25)
    t_chk = first(tb, lambda s: s["a"] in (E5.CHK_180, E5.CHK_45))
    t_pw = first(tb, lambda s: s["a"] == E5.JCC_PW)
    t_ax = first(tb, lambda s: s["a"] == E5.JCC_AX)
    t_push = first(tb, lambda s: s["a"] == E5.CMP_P)
    t_l, t_b = nth_next(tb, 0), nth_next(tb, 1)
    t_bal = first(tb, lambda s: s["a"] == E5.BALANCE)
    t_clear = first(tb, lambda s: s["a"] == E5.OU_M, t_bal or 0)
    N["tb_exam"] = {k: v for k, v in {
        0: "Right eye first, left eye covered. “Read the smallest line you can.”",
        t_trial: "Trial lenses: −2.00, then +2.00. Then a whole dioptre at a time, reading the chart.",
        t_fog: "Best sphere: plus until the letters blur, then minus a quarter at a time, reading the chart each time.",
        t_chk: "Astigmatism check: the cross-cylinder at 180° or 90°?",
        t_pw: "Cylinder POWER: more or less? (Small cylinder: power first.)",
        t_ax: "Cylinder AXIS: “1 or 2?” The axis turns toward the clearer side, until the answers reverse.",
        t_push: "“+0.25: 1 or 2?” If the extra plus is no worse, keep it: no more minus than the eye needs.",
        t_l: "The left eye, right eye covered: all of it again.",
        t_b: "Both eyes open: fog both, split the chart with prism, make the two eyes equal…",
        t_clear: "…then clear the fog together, and the final check."}.items() if k is not None}
    N["tb_slow"] = tuple(range(0, 4))
    # ---- day one: the third video's AI, as it was
    n1 = {0: "Last video's AI, exactly as it left. Both eyes open first, then each eye."}
    for k, st in enumerate(d1["steps"]):
        if st["a"] == E5.SPIN and st["code"] == 5:
            n1[k] = (f"Cylinder in, spin the axis. “Stop!” …at {st['spin_ax']:.0f}°. "
                     f"(The true axis: {true_axis(d1, k):.0f}°.)")
            break
    N["day1_exam"] = n1
    # ---- the AI after practice here
    def ax_after(tr, k):
        st = tr["steps"][k]
        return int(round(st["after"][min(st["phase"], 1)][2])) % 180 or 180

    na = {0: "Its first patient again. Both eyes open first: minus a quarter at a time while the letters improve, "
             "and then some more."}
    for eye, name in ((0, "Right eye"), (1, "Left eye")):
        k_sp = first(ai, lambda s: s["a"] == E5.SPIN and s["code"] == 5 and s["phase"] == eye)
        if k_sp is None:
            continue
        ks = [k for k, s in enumerate(ai["steps"]) if s["a"] == E5.JCC_AX and s["phase"] == eye and k > k_sp]
        tax = true_axis(ai, k_sp)
        na[k_sp] = (f"{name}: a little cylinder, then spin the axis. “Stop!” …at {ax_after(ai, k_sp)}°. "
                    f"(The true axis: {tax:.0f}°.)")
        if ks:
            na[ks[0]] = (f"Then the cross-cylinder for the axis: “1 or 2?”, and the axis turns toward the clearer "
                         f"side. It ends at {ax_after(ai, ks[-1])}°.")
    k_l = max((k for k, s in enumerate(ai["steps"]) if s["phase"] == 1), default=0)     # after the left eye
    k_back = first(ai, lambda s: s["a"] == E5.OU_P and s["phase"] == 2, k_l)
    if k_back is not None:
        na[k_back] = "Both eyes again: the extra minus comes off, a quarter at a time. Then the final check."
    N["ai_exam"], N["ai_slow"], N["sequences"] = na, tuple(range(0, 4)), []
    # ---- the practice curve
    st = R["stages"]
    cv = [(1000, f"Day one here: {R['day1']['summary']['net']:+.0f} points a patient "
                 f"(last video's patients: {R['h_ai_net']:+.0f}).")]
    jcc = [(x["exams"], x["habits"]["used"]["cyl axis (cross-cyl)"], x["axis"]["within_10"]) for x in st]
    k = next((ex for ex, u, _ in jcc if u >= 0.9 and ex > 0), None)
    if jcc and jcc[0][0] == 0:
        cv.append((3000, f"It tries the cross-cylinder for the axis now and then (the nudge), and turns the dial "
                         f"by hand. Axis within 10°: {100 * jcc[0][2]:.0f}% of astigmatic eyes."))
    if k:
        w10 = next(w for ex, _, w in jcc if ex == k)
        cv.append((k, f"By {fmt_n(round(k, -4))} patients: the cross-cylinder for the axis in every exam, right after the spin. "
                      f"Axis within 10°: {100 * w10:.0f}%."))
    N["curve"] = cv
    return N


# ======================================================================== the film
def sections():
    R = json.load(open(os.path.join(DATA, "results_m.json")))
    H3 = json.load(open(os.path.join(DATA, "results_h.json")))
    R["h_ai_net"] = H3["ai"]["summary"]["net"]
    S = {}
    S["hook"] = Hook6()
    S["changes"] = Changes6()
    S["toolbox"] = Toolbox5()
    S["patient"] = PatientSays6()
    S["wants"] = Wants5()
    S["learn"] = Learn6()
    sc = R["scratch"]
    S["scratch"] = Say([(0.5, "Why not start from nothing again?", 56, INK, "Bold"),
                        (2.2, "We tried: a new AI, with only the occasional remark.", 36, INK2, "Medium"),
                        (4.4, f"After {sc['exams'] / 1000:.0f},000 patients it was stuck at a shortcut: minus with both "
                              f"eyes open, and prescribe. {100 * sc['summary']['happy']:.0f}% happy patients.", 36, INK2,
                         "Medium"),
                        (8.0, "The smile had been doing a lot of the teaching.", 40, GOLD, "SemiBold"),
                        (10.5, f"So: last video's AI, after {H3['meta']['exams'] / 1e6:.1f} million patients, walks into "
                               "the new clinic.", 38, INK, "SemiBold")], 16.0)
    S["nudge"] = Say([(0.5, "One button changed", 56, INK, "Bold"),
                      (2.2, "The cross-cylinder for the axis: “1 or 2?”, flipped either side of the axis.", 36, INK2,
                       "Medium"),
                      (4.6, "Last video's AI had stopped pressing it. Its spin was good enough.", 36, INK2, "Medium"),
                      (7.4, "Now the button also turns the axis toward the side the patient picks, the way the spin "
                            "sets the axis at “stop!”.", 36, INK2, "Medium"),
                      (11.0, "And at the start, the AI is nudged to try it now and then.", 36, INK2, "Medium"),
                      (13.4, "Practice decides whether it keeps it.", 42, GOLD, "SemiBold")], 18.0)
    d1 = trace("m_day1")
    ai, tb = trace("m_ai"), trace("m_borish")
    NOTES = make_notes(R, ai, tb, d1)
    S["day1_title"] = Say([(0.5, "Day one in the new clinic", 72, INK, "Bold"),
                           (1.6, "Last video's AI. Its first patient here.", 38, INK2, "Medium")], 6.0)
    S["day1_exam"] = Exam5(d1, "DAY ONE", "last video's AI, in the new clinic", notes=NOTES["day1_exam"],
                           slow=range(4), outro=6.0)
    h_ai, d1s = H3["ai"], R["day1"]
    S["day1_grid"] = Grid6(
        "LAST VIDEO'S AI, 1,000 PATIENTS EACH", "One square per patient.",
        [("Last video's patients", h_ai["outcomes"], h_ai["summary"]["net"], h_ai["summary"]["minutes"]),
         ("The new patients", d1s["outcomes"], d1s["summary"]["net"], d1s["summary"]["minutes"])], dur=13.0,
        note=(f"Same AI. With the new patients, {100 * d1s['axis']['off_10']:.0f}% of astigmatic eyes end up "
              f"with the axis more than 10° off."))
    refs = [(BORISH, R["textbook"]["summary"]["net"], TB_COL)]
    S["curve"] = Practice5(refs, NOTES.get("curve", []), max_exams=R["meta"]["exams"], run=RUN)
    S["curve"].v0, S["curve"].v1 = -50, 80
    S["ai_title"] = Say([(0.4, f"After {fmt_n(R['meta']['exams'])} patients in the new clinic", 60, INK, "Bold"),
                         (1.6, "The same first patient, again.", 36, INK2, "Medium")], 6.0)
    S["ai_exam"] = Exam5(ai, "THE AI", f"after {fmt_n(R['meta']['exams'])} patients in the new clinic",
                         notes=NOTES.get("ai_exam", {}), slow=NOTES.get("ai_slow", ()))
    S["tb_card"] = BorishSteps()
    S["tb_exam"] = Exam5(tb, "THE BORISH METHOD", "the same patient", who="BORISH", who_col=TB_COL,
                         notes=NOTES.get("tb_exam", {}), slow=NOTES.get("tb_slow", ()), fast_beat=0.2)
    S["sequences"] = Sequences6(tb, ai, NOTES.get("sequences", []))
    S["results"] = Results6(R, [
        ("Time in the chair", "per patient", lambda e: e["summary"]["minutes"], mmss, 0, 16),
        ("Astigmatism axis within 10°", "eyes with 0.75 D of cylinder or more", lambda e: e["axis"]["within_10"],
         lambda v: f"{100 * v:.0f}%", 0, 1),
        ("Happy patients", "sees clearly, comes back, refers a friend", lambda e: e["summary"]["happy"],
         lambda v: f"{100 * v:.0f}%", 0, 1)])
    S["points"] = PointsSplit(R)
    S["consistency"] = Consistency(R)
    S["real"] = Real6(R)
    S["outro"] = F3.Card("The fine print", [
        (0.6, "Every patient is simulated. The optics are real; the answers are a model; the scorecard is made up.",
         30, INK2, "Medium"),
        (3.0, "The AI is reinforcement learning on a 4-core computer. It has never examined a real person. This is a "
              "demonstration, not a clinical tool.", 30, INK2, "Medium"),
        (5.6, "If your vision changes, see your optometrist.", 34, GOLD, "SemiBold")], 15.0, y0=230)
    return S


ORDER = ["hook", "changes", "toolbox", "patient", "wants", "learn", "scratch", "nudge", "day1_title", "day1_exam",
         "day1_grid",
         "curve", "ai_title",
         "ai_exam", "tb_card", "tb_exam", "sequences", "results", "points", "consistency", "real", "outro"]


def order():
    S = sections()
    return [n for n in ORDER if n in S], S


def render(name):
    _, S = order()
    S[name].render("m_" + name)


def render_all(names=None, par=4):
    names = names or order()[0]

    def one(nm):
        log = open(os.path.join(OUT, f"m_{nm}.log"), "w")
        r = subprocess.run([sys.executable, "film6.py", "render", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
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
    paths = [os.path.join(OUT, f"m_{s}.mp4") for s in names]
    ds = [film.dur(p) for p in paths]
    off = [0.0]
    for dd in ds[:-1]:
        off.append(off[-1] + dd - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(names, off))
    for s, o, dd in zip(names, off, ds):
        print(f"  {s:12s} {o:7.2f} + {dd:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video6.mp4")
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
              (g("day1_title"), 0.45), (g("day1_exam"), 0.3), (g("day1_grid"), 0.4), (g("curve"), 0.6),
              (g("ai_title"), 0.5), (g("ai_exam"), 0.4), (g("tb_card"), 0.35), (g("tb_exam"), 0.35),
              (g("sequences"), 0.5), (g("results"), 0.8), (g("points"), 0.6), (g("consistency"), 0.6),
              (g("real"), 0.5), (g("outro"), 0.4), (total, 0.0)]
    sc = SC.Score(total, sorted(energy), [(g("curve"), 0.6), (g("results"), 0.8)],
                  [(g("ai_title") - 2.0, g("ai_title")), (g("results") - 2.0, g("results"))],
                  transpose_at=g("ai_title"), final_at=g("outro") + 0.5, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.7
    mix = mix + film.sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score6.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "ai_optometry_v4.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], dd] for s, dd in zip(names, ds)}, "total": total},
              open(os.path.join(OUT, "timeline6.json"), "w"), indent=1)
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
        Image.fromarray(S[sys.argv[2]].frame(float(sys.argv[3]))).save(os.path.join(OUT, f"frame6_{sys.argv[2]}.png"))
    elif cmd == "all":
        render_all()
        assemble()
    elif cmd == "assemble":
        assemble()
