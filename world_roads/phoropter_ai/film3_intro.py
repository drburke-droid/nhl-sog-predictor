"""Opening and explainer scenes for the clinic video: the premise, the AI's buttons, what the patient can say
(and how sure they are), what the AI wants (the clinic score), and how practice works.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
import ui
import chart as CH
import clinic
import env3 as E3
import clinic_ui as CU
from ui import W, H, BG, INK, INK2, MUTED, GOLD, RED, GREEN, CYAN, PANEL, CARD, LINE, text, rr, fmt_d, smooth, fade
from scenes import Section, para, big_title
from record import fixed_pair, SHOWCASE2

AI_COL = (51, 140, 237)
TB_COL = (231, 102, 36)


def fade_frame(im, a):
    fr = np.asarray(im).astype(np.float32)
    return (fr * a + np.array(BG) * (1 - a)).astype(np.uint8)


# ======================================================================== the premise
class Hook3(Section):
    dur = 13.0

    def __init__(self):
        self.rng = np.random.default_rng(4)
        self.flash = [(0.8 + 0.42 * i, int(self.rng.integers(0, E3.N_ACT3))) for i in range(26)]
        self.sfx = [(t, "tick") for t, _ in self.flash]

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
        rr(d, [0, 0, W, 790], 0, fill=BG + (int(150 * smooth((t - 4.0) / 1.0)),))
        big_title(d, W / 2, 210, "This AI has never seen an eye.", 64, fade(t, 4.2, 99, 0.6, 0.1))
        big_title(d, W / 2, 310, "It doesn't know what a phoropter is.", 44, fade(t, 6.0, 99, 0.6, 0.1), INK2, "SemiBold")
        big_title(d, W / 2, 430, "Today it opens an eye clinic.", 56, fade(t, 8.2, 99, 0.6, 0.1), GOLD, "Bold")
        return fade_frame(im, a_all)


# ======================================================================== the buttons
class Toolbox(Section):
    """The phoropter with its parts labelled, then every button, group by group, with what it does."""
    ORDER = ["sphere", "cyl", "axis", "jcc", "prism", "occ", "chart", "rx"]
    dur = 40.0
    T0 = 4.0
    STEP = 3.4

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        k = int((t - self.T0) // self.STEP) if t >= self.T0 else -1
        k = min(k, len(self.ORDER) - 1)
        show = self.ORDER[:k + 1] if k >= 0 else []
        cur = self.ORDER[k] if 0 <= k < len(self.ORDER) and t < self.T0 + self.STEP * len(self.ORDER) else None
        hot = (cur, None) if cur not in (None, "chart", "rx") else None
        CU.draw_phoropter(im, 30, 70, 0.56, None, hot=hot, rings=True, labels=True,
                          legend_only=[g for g in show if g not in ("chart", "rx")] or ["none"])
        d = ImageDraw.Draw(im, "RGBA")
        text(d, (40, 20), "WHAT THE AI HAS: 27 BUTTONS", 32, INK, "Bold")
        text(d, (40 + d.textlength("WHAT THE AI HAS: 27 BUTTONS", font=ui.font(32, "Bold")) + 20, 30),
             "each colour is a part of the phoropter", 22, MUTED, "Medium")
        # the full panel on the right, group by group, with descriptions
        x0, y0, wcol = 870, 80, 1020
        y = y0
        f_lab = ui.font(20, "Bold")
        for gi, g in enumerate(self.ORDER):
            a = fade(t, self.T0 + self.STEP * gi, 99, 0.5, 0.1) * a_all
            if a <= 0:
                break
            col = CU.GROUP_COL[g]
            is_cur = g == cur
            text(d, (x0, y), CU.GROUP_NAME[g], 19, col, "Bold", a=a)
            y += 28
            items = [b for b in CU.BUTTONS if b[5] == g]
            # pairs of +/- buttons share a description line
            x = x0
            line_h = 0
            for b in items:
                idx, chip, sub, lab, desc, _ = b
                w = d.textlength(lab, font=f_lab) + 26
                if x + w > x0 + wcol:
                    x, y = x0, y + 46
                rr(d, [x, y, x + w, y + 38], 9, fill=tuple(col) + (int((220 if is_cur else 40) * a),),
                   outline=tuple(col) + (int(255 * a),), width=2)
                text(d, (x + w / 2, y + 19), lab, 20, BG if is_cur else INK, "Bold", "mm", a)
                x += w + 10
            y += 46
            descs = []
            for b in items:
                if b[4] and b[4] not in descs:
                    descs.append(b[4])
            dl = "  ·  ".join(descs)
            y = para(d, x0 + 4, y, dl, 18, wcol - 10, INK2, "Medium", a, lh=1.25) + 12
        a = fade(t, self.T0 + self.STEP * len(self.ORDER) + 0.6, 99, 0.7, 0.1) * a_all
        big_title(d, 450, 760, "Nobody tells it what any of them do.", 40, a, GOLD, "SemiBold")
        para(d, 90, 830, "Some only work at certain moments: the cylinder and cross-cylinder with one eye covered, "
                         "the both-eye buttons with both open. “Prescribe” works any time: it hands over whatever "
                         "is in the phoropter.", 22, 740, INK2, "Medium", a)
        return np.asarray(im)


# ======================================================================== what the patient can say
DEMO = dict(SHOWCASE2, age=55, S=-2.75, C=0.0, axis=180, amp=1.0, prox=0.0, bias=0.04, sigma=0.03, tau=0.03,
            lapse=0.02, pupil=4.0, S_os=-2.75, C_os=0.0, axis_os=180)


def demo_odds():
    """A 55-year-old who needs -2.75: the odds of each answer to "1 or 2? (2 = -0.25)" as the lens moves from
    plano to -4.00, asked on the smallest line they can just read."""
    env = E3.ClinicEnv(1, seed=0, sampler=fixed_pair(**DEMO))
    rows = []
    for S in np.arange(0.0, -4.01, -0.25):
        env.S[0, 0], env.C[0, 0], env.ax[0, 0] = S, 0.0, 180.0
        T = float(env.mono_T(0, env.lens_pv(0))[0])
        env.line[0] = np.clip(np.round(T + 0.1, 1), -0.3, 1.0)
        P, right = env.compare_probs(E3.CMP_M)
        rows.append(dict(S=float(S), T=T, line=float(env.line[0]), P=[float(x) for x in P[0]], right=int(right[0])))
    return rows


class PatientSays(Section):
    dur = 46.0
    T_DEMO = 11.0

    def __init__(self):
        self.rows = demo_odds()
        self.step = 1.7
        self.sfx = [(self.T_DEMO + 1.0 + i * self.step, "tick") for i in range(len(self.rows))]

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "WHAT THE PATIENT CAN SAY", 34, INK, "Bold", a=a_all)
        # the answer bin
        a = fade(t, 0.6, 99, 0.5, 0.1) * a_all
        bins = [("To “1 or 2?”", ["“1 is clearer.”", "“2 is clearer.”", "“They look the same.”"], CU.SPHERE),
                ("To the cross-cylinder", ["“1.”", "“2.”", "“They look the same.”"], CU.JCC),
                ("To “read this line”", ["“All of them.”", "“Most of them.”", "“A few.”", "“I can’t read any.”"], CU.CHART),
                ("To “read the smallest line”", ["“Down to 20/40…”", "“I can’t read any of it.”"], CU.CHART),
                ("To the prism split", ["“Top.”", "“Bottom.”", "“They look equal.”"], CU.PRISM)]
        x = 60
        for i, (h, items, col) in enumerate(bins):
            ai = fade(t, 0.8 + 0.7 * i, 99, 0.5, 0.1) * a_all
            rr(d, [x, 100, x + 346, 380], 14, fill=CARD + (int(255 * ai),), outline=tuple(col) + (int(200 * ai),), width=2)
            text(d, (x + 18, 118), h, 21, col, "Bold", a=ai)
            for j, it in enumerate(items):
                text(d, (x + 22, 168 + j * 48), it, 22, INK, "Medium", a=ai)
            x += 362
        a = fade(t, 5.0, 99, 0.6, 0.1) * a_all
        para(d, 60, 408, "Nothing else. And how sure they are depends on how far the lenses are from their real "
                         "prescription.", 28, 1800, INK2, "SemiBold", a)
        # the demo
        if t < self.T_DEMO - 0.5:
            return fade_frame(im, a_all)
        ad = fade(t, self.T_DEMO - 0.5, 99, 0.6, 0.1) * a_all
        i = int(np.clip((t - self.T_DEMO - 1.0) // self.step, 0, len(self.rows) - 1))
        r = self.rows[i]
        rr(d, [60, 480, 900, 968], 16, fill=CARD + (int(255 * ad),), outline=LINE + (int(255 * ad),), width=2)
        text(d, (84, 500), "A 55-YEAR-OLD WHO NEEDS −2.75", 20, MUTED, "Bold", a=ad)
        text(d, (84, 540), "Lens in front of the eye", 24, INK2, "Medium", a=ad)
        text(d, (870, 532), fmt_d(r["S"]).replace("Plano", "0.00"), 48, CU.SPHERE, "Bold", "ra", ad)
        text(d, (84, 606), f"They can just read  {CH.snellen_of(r['line'])}", 24, INK2, "Medium", a=ad)
        CU.draw_answers(d, 84, 646, 792, 300, E3.CMP_M, probs=r["P"], chosen=None, right=r["right"], alpha=ad,
                        header="THE ODDS OF EACH ANSWER")
        # the curve: chance of the right answer, and of "the same"
        X0, Y0, X1, Y1 = 1010, 560, 1840, 860
        rr(d, [940, 480, 1880, 968], 16, fill=CARD + (int(255 * ad),), outline=LINE + (int(255 * ad),), width=2)
        text(d, (964, 500), "CHANCE OF EACH ANSWER, AS THE LENS GETS CLOSER", 20, MUTED, "Bold", a=ad)
        px = lambda S: X0 + (X1 - X0) * (-S / 4.0)
        py = lambda p: Y1 - (Y1 - Y0) * p
        for p in (0, 0.5, 1):
            d.line([(X0, py(p)), (X1, py(p))], fill=LINE + (int(255 * ad),), width=1)
            text(d, (X0 - 10, py(p)), f"{int(p * 100)}%", 17, MUTED, "Medium", "rm", ad)
        for S in (0, -1, -2, -2.75, -4):
            text(d, (px(S), Y1 + 14), fmt_d(S).replace("Plano", "0.00"), 17, GOLD if S == -2.75 else MUTED, "SemiBold",
                 "ma", ad)
        d.line([(px(-2.75), Y0), (px(-2.75), Y1)], fill=GOLD + (int(160 * ad),), width=2)
        text(d, (px(-2.75) + 8, Y0), "their prescription", 17, GOLD, "SemiBold", "la", ad)
        series = [("right answer", lambda r_: r_["P"][r_["right"]], (110, 220, 140)),
                  ("“the same”", lambda r_: r_["P"][2], MUTED),
                  ("wrong answer", lambda r_: r_["P"][1 - r_["right"]] if r_["right"] < 2 else 0.0, RED)]
        for j, (lab, fn, col) in enumerate(series):
            pts = [(px(rw["S"]), py(fn(rw))) for rw in self.rows[:i + 1]]
            if len(pts) > 1:
                d.line(pts, fill=tuple(col) + (int(255 * ad),), width=4, joint="curve")
            for p_ in pts[-1:]:
                d.ellipse([p_[0] - 6, p_[1] - 6, p_[0] + 6, p_[1] + 6], fill=tuple(col) + (int(255 * ad),))
            text(d, (X1, Y0 - 34 + 0 * j), "", 10)
        text(d, (X0 - 30, Y1 + 50), "—  the right answer", 19, (110, 220, 140), "SemiBold", "la", ad)
        text(d, (X0 + 230, Y1 + 50), "—  “they look the same”", 19, MUTED, "SemiBold", "la", ad)
        text(d, (X0 + 520, Y1 + 50), "—  the wrong answer", 19, RED, "SemiBold", "la", ad)
        # commentary
        S = r["S"]
        if S > -1.6:
            msg = "Far off: the letters are a blur, so a quarter step barely shows. Often “the same.”"
        elif S > -2.6:
            msg = "Closer, on letters they can just read: the difference is obvious. Confident and right."
        elif S > -3.6:
            msg = "Right on it: “Umm… the same?” And sometimes the wrong lens: extra minus they can focus through."
        else:
            msg = "Past it: the extra minus is too much to focus through. Confident again."
        big_title(d, W / 2, 1000, msg, 30, ad, INK, "SemiBold")
        return np.asarray(im)


# ======================================================================== what the AI wants
class Motivation(Section):
    dur = 40.0

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "WHAT THE AI WANTS: CLINIC POINTS", 34, INK, "Bold", a=a_all)
        a = fade(t, 0.6, 99, 0.5, 0.1) * a_all
        para(d, 60, 90, "It's told one thing per patient: the score. The same things an optometrist cares about.", 26,
             1800, INK2, "Medium", a)
        rows = [("happy", "Sees clearly. Comes back next year and refers a friend.", "+100"),
                ("annoyed", "“Not quite crisp.” Might not come back.", "+30"),
                ("recheck", "Comes back for an Rx check: 15 minutes of chair time nobody pays for. Won't return.",
                 "−80"),
                ("review", "Rx check, never returns, leaves a one-star review.", "−150")]
        y = 160
        for i, (o, desc, pts) in enumerate(rows):
            ai = fade(t, 1.8 + 1.6 * i, 99, 0.5, 0.1) * a_all
            col = CU.OUT_COL[o]
            rr(d, [60, y, 1000, y + 112], 14, fill=CARD + (int(255 * ai),), outline=tuple(col) + (int(255 * ai),), width=3)
            text(d, (90, y + 14), clinic.LABELS[o].split(":")[0].upper(), 22, col, "Bold", a=ai)
            para(d, 90, y + 46, desc, 21, 700, INK2, "Medium", ai, lh=1.25)
            text(d, (975, y + 56), pts, 44, col, "Bold", "rm", ai)
            y += 124
        ai = fade(t, 9.0, 99, 0.5, 0.1) * a_all
        rr(d, [60, y, 1000, y + 112], 14, fill=CARD + (int(255 * ai),), outline=CU.CHART + (int(255 * ai),), width=3)
        text(d, (90, y + 14), "CHAIR TIME", 22, CU.CHART, "Bold", a=ai)
        para(d, 90, y + 46, "Every minute in the chair costs what a minute of appointment earns. Over 15 minutes, "
                            "the patient walks out (−150).", 21, 700, INK2, "Medium", ai, lh=1.25)
        text(d, (975, y + 56), "−4/min", 40, CU.CHART, "Bold", "rm", ai)
        # how error turns into outcomes: stacked chances
        ac = fade(t, 12.0, 99, 0.6, 0.1) * a_all
        X0, Y0, X1, Y1 = 1130, 220, 1840, 760
        rr(d, [1070, 160, 1880, 900], 16, fill=CARD + (int(255 * ac),), outline=LINE + (int(255 * ac),), width=2)
        text(d, (1094, 178), "HOW FAR OFF  →  HOW THE PATIENT FEELS", 20, MUTED, "Bold", a=ac)
        es = np.linspace(0, 1.5, 121)
        P = clinic.outcome_probs(es)
        cum = np.cumsum(P, 1)
        px = lambda e: X0 + (X1 - X0) * e / 1.5
        py = lambda p: Y1 - (Y1 - Y0) * p
        prog = smooth((t - 12.5) / 4.0)
        n = max(2, int(len(es) * prog))
        for j, o in enumerate(clinic.OUTCOMES):
            lo = cum[:, j - 1] if j > 0 else np.zeros(len(es))
            hi = cum[:, j]
            poly = [(px(e), py(h)) for e, h in zip(es[:n], hi[:n])] + [(px(e), py(l)) for e, l in zip(es[:n][::-1], lo[:n][::-1])]
            d.polygon(poly, fill=tuple(CU.OUT_COL[o]) + (int(210 * ac),))
        for e in (0, 0.25, 0.5, 0.75, 1.0, 1.5):
            text(d, (px(e), Y1 + 12), f"{e:.2f} D", 17, MUTED, "Medium", "ma", ac)
        text(d, ((X0 + X1) / 2, Y1 + 40), "how far the glasses are from the true prescription (mostly the worse eye)", 18,
             INK2, "Medium", "ma", ac)
        for j, (o, lab) in enumerate(zip(clinic.OUTCOMES, ("Happy", "Annoyed", "Rx check", "Bad review"))):
            text(d, (1110 + j * 180, 828), "■ " + lab, 19, CU.OUT_COL[o], "SemiBold", a=ac)
        text(d, (1094, 864), "A plausible scorecard we made up; not measured data.", 17, MUTED, "Medium", a=ac)
        ab = fade(t, 21.0, 99, 0.6, 0.1) * a_all
        big_title(d, W / 2, 940, "Its only goal: the most points per patient.", 40, ab, GOLD, "SemiBold")
        ab2 = fade(t, 25.0, 99, 0.6, 0.1) * a_all
        big_title(d, W / 2, 1000, "Right glasses, happy patients, quick exams. Nobody tells it how.", 28, ab2, INK2,
                  "Medium")
        return np.asarray(im)


# ======================================================================== how practice works
class HowItLearns(Section):
    dur = 64.0
    BEATS = [0.5, 9.0, 18.0, 28.0, 38.0, 48.0]

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (60, 30), "HOW PRACTICE WORKS (REINFORCEMENT LEARNING, IN PLAIN ENGLISH)", 32, INK, "Bold", a=a_all)
        B = self.BEATS
        # the loop diagram
        boxes = [("WHAT IT SEES", "the lenses in the phoropter, the answers so far, its last 16 buttons", 90, CU.CHART),
                 ("ITS BRAIN", "a small neural network: 886 numbers in, 27 out", 520, AI_COL),
                 ("THE ODDS", "a chance for each button; it picks one", 950, CU.SPHERE),
                 ("THE PATIENT", "answers, or the lenses change", 1380, CU.PRISM)]
        a1 = fade(t, B[0], 99, 0.6, 0.1) * a_all
        for i, (h, sub, x, col) in enumerate(boxes):
            ai = fade(t, B[0] + 0.9 * i, 99, 0.5, 0.1) * a_all
            rr(d, [x, 120, x + 380, 300], 16, fill=CARD + (int(255 * ai),), outline=tuple(col) + (int(255 * ai),), width=3)
            text(d, (x + 22, 140), h, 24, col, "Bold", a=ai)
            para(d, x + 22, 186, sub, 22, 340, INK2, "Medium", ai)
            if i < 3:
                ui_arrow(d, x + 388, 210, x + 424, 210, INK2, ai, head=16)
        ai = fade(t, B[0] + 4.0, 99, 0.5, 0.1) * a_all
        d.line([(1570, 306), (1570, 360), (280, 360), (280, 330)], fill=INK2 + (int(255 * ai),), width=4)
        ui_arrow(d, 280, 340, 280, 308, INK2, ai, head=16)
        text(d, (925, 372), "…repeat until it presses Prescribe", 22, INK2, "SemiBold", "ma", ai)
        # the score and the nudge
        ai = fade(t, B[1], 99, 0.6, 0.1) * a_all
        rr(d, [90, 440, 900, 640], 16, fill=CARD + (int(255 * ai),), outline=GOLD + (int(255 * ai),), width=3)
        text(d, (112, 460), "AFTER EACH PATIENT: THE SCORE", 24, GOLD, "Bold", a=ai)
        para(d, 112, 504, "Clinic points for the glasses, minus the chair time. It also keeps a guess of what score "
                          "it expected from each moment of the exam.", 22, 770, INK2, "Medium", ai)
        ai = fade(t, B[2], 99, 0.6, 0.1) * a_all
        rr(d, [960, 440, 1850, 640], 16, fill=CARD + (int(255 * ai),), outline=GREEN + (int(255 * ai),), width=3)
        text(d, (982, 460), "THE NUDGE", 24, GREEN, "Bold", a=ai)
        para(d, 982, 504, "Better than expected: the buttons it pressed get a little more likely next time it's in "
                          "a similar spot. Worse than expected: a little less likely.", 22, 850, INK2, "Medium", ai)
        # why so many practice exams
        ai = fade(t, B[3], 99, 0.6, 0.1) * a_all
        rr(d, [90, 680, 900, 900], 16, fill=CARD + (int(255 * ai),), outline=LINE + (int(255 * ai),), width=2)
        text(d, (112, 700), "WHY IT NEEDS SO MANY PATIENTS", 24, INK, "Bold", a=ai)
        para(d, 112, 744, "One exam is one noisy hint: a lucky answer can make a bad move look good. Thousands of "
                          "exams average the luck out. It practises on simulated patients, 1,024 exam rooms at once.",
             22, 770, INK2, "Medium", ai)
        ai = fade(t, B[4], 99, 0.6, 0.1) * a_all
        rr(d, [960, 680, 1850, 900], 16, fill=CARD + (int(255 * ai),), outline=LINE + (int(255 * ai),), width=2)
        text(d, (982, 700), "TRAINING WHEELS", 24, INK, "Bold", a=ai)
        para(d, 982, 744, "For the first 15% of practice, a mentor also whispered “warmer” or “colder” after each "
                          "button: did the lenses get closer to the truth? Then the mentor left. After that: only "
                          "the clinic score.", 22, 850, INK2, "Medium", ai)
        ai = fade(t, B[5], 99, 0.6, 0.1) * a_all
        big_title(d, W / 2, 950, "No rules about eyes. No textbook. Just practice and a score.", 36, ai, GOLD, "SemiBold")
        return np.asarray(im)


def ui_arrow(d, x0, y0, x1, y1, col, a=1.0, w=4, head=14):
    d.line([(x0, y0), (x1, y1)], fill=tuple(col) + (int(255 * a),), width=w)
    ang = math.atan2(y1 - y0, x1 - x0)
    pts = [(x1, y1), (x1 - head * math.cos(ang - 0.45), y1 - head * math.sin(ang - 0.45)),
           (x1 - head * math.cos(ang + 0.45), y1 - head * math.sin(ang + 0.45))]
    d.polygon(pts, fill=tuple(col) + (int(255 * a),))


if __name__ == "__main__":
    import os, sys
    os.makedirs(ui.OUT, exist_ok=True)
    for S, ts in ((Hook3(), (3.0, 10.0)), (Toolbox(), (9.0, 33.0)), (PatientSays(), (8.0, 20.0, 34.0)),
                  (Motivation(), (30.0,)), (HowItLearns(), (60.0,))):
        for t in ts:
            Image.fromarray(S.frame(t)).save(os.path.join(ui.OUT, f"test_{type(S).__name__}_{int(t)}.png"))
    print("ok")
