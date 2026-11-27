"""Simpler explainer scenes: one idea per screen, big text, things appearing one at a time."""
import math
import numpy as np
from PIL import Image, ImageDraw
import ui
import chart as CH
import clinic
import env3 as E3
import clinic_ui as CU
from ui import W, H, BG, INK, INK2, MUTED, GOLD, RED, GREEN, CARD, LINE, PANEL, text, rr, fmt_d, smooth, fade
from scenes import Section, para, big_title
from record import fixed_pair
from film3_intro import DEMO, fade_frame, ui_arrow

AI_COL = (0x33, 0x8C, 0xED)


# ======================================================================== what it can do, one part at a time
class Toolbox4(Section):
    PARTS = [("sphere", "SPHERE", "How strong the lens is. Minus for near-sighted eyes, plus for far-sighted ones."),
             ("cyl", "CYLINDER", "Extra power in one direction, for astigmatism."),
             ("axis", "AXIS", "Which direction that extra power points."),
             ("jcc", "CROSS-CYLINDER", "A flip lens. “1 or 2?” tells you whether the cylinder needs to turn, or "
                                      "grow, or shrink."),
             ("prism", "PRISM", "Splits the chart into two copies, one per eye, to compare the eyes."),
             ("occ", "OCCLUDER", "Covers one eye: right eye first, then left, then both open."),
             ("chart", "THE CHART", "“Read the smallest line you can.” Bigger or smaller letters."),
             ("rx", "PRESCRIBE", "Done. Whatever is in the phoropter becomes the glasses.")]
    T0, STEP = 3.0, 4.2

    def __init__(self):
        self.dur = self.T0 + self.STEP * len(self.PARTS) + 4.5
        self.sfx = [(self.T0 + self.STEP * i, "tick") for i in range(len(self.PARTS))]

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        i = int((t - self.T0) // self.STEP) if t >= self.T0 else -1
        i = min(i, len(self.PARTS) - 1)
        g = self.PARTS[i][0] if i >= 0 else None
        hot = (g, None) if g not in (None, "chart", "rx") else None
        CU.draw_phoropter(im, 470, 60, 0.67, None, hot=hot, legend_only=[g] if hot else ["none"])
        d = ImageDraw.Draw(im, "RGBA")
        if t < self.T0 + 0.3:
            big_title(d, W / 2, 820, "Everything the AI can do", 54, fade(t, 0.3, self.T0 + 0.3, 0.5, 0.4) * a_all)
            big_title(d, W / 2, 900, "is press buttons on a phoropter.", 40, fade(t, 0.9, self.T0 + 0.3, 0.5, 0.4) * a_all,
                      INK2, "SemiBold")
        if i >= 0 and t < self.T0 + self.STEP * len(self.PARTS):
            key, name, desc = self.PARTS[i]
            tt = t - self.T0 - self.STEP * i
            al = fade(tt, 0, self.STEP, 0.35, 0.35) * a_all
            col = CU.GROUP_COL[key]
            text(d, (W / 2, 790), name, 56, col, "Bold", "ma", al)
            big_title(d, W / 2, 866, desc, 32, al, INK, "Medium")
            chips = [b for b in CU.BUTTONS if b[5] == key]
            f = ui.font(22, "Bold")
            ws = [d.textlength(b[3], font=f) + 34 for b in chips]
            gap = 12
            tot = sum(ws) + gap * (len(ws) - 1)
            scale = min(1.0, 1800 / tot)
            x = W / 2 - tot * scale / 2
            for b, w in zip(chips, ws):
                w *= scale
                rr(d, [x, 950, x + w, 1000], 10, fill=tuple(col) + (int(45 * al),), outline=tuple(col) + (int(255 * al),),
                   width=2)
                text(d, (x + w / 2, 975), b[3], int(22 * scale) if scale < 1 else 22, INK, "Bold", "mm", al)
                x += w + gap * scale
        if t >= self.T0 + self.STEP * len(self.PARTS):
            al = fade(t, self.T0 + self.STEP * len(self.PARTS), 999, 0.5, 0.1) * a_all
            big_title(d, W / 2, 840, "27 buttons. Nobody tells it what any of them do.", 46, al, GOLD, "SemiBold")
        return fade_frame(im, a_all)


# ======================================================================== what the patient says, three moments
def demo_rows():
    """The 55-year-old who needs -2.75, asked "1 or 2? (2 adds -0.25)" at three lenses, on the smallest line they
    can just read: far off, close, and right at their prescription."""
    env = E3.ClinicEnv(1, seed=0, sampler=fixed_pair(**DEMO), rand_dial=False)
    out = []
    for S, label, msg in ((-0.50, "FAR OFF", "A quarter step barely changes letters this blurry. Often: “they look "
                                             "the same.”"),
                          (-2.00, "CLOSE", "On letters they can just read, the difference is obvious. Confident, and "
                                           "right."),
                          (-2.75, "RIGHT ON IT", "No real difference any more. “Umm… the same?” And sometimes the "
                                                 "wrong one.")):
        env.S[0, 0], env.C[0, 0], env.ax[0, 0] = S, 0.0, 180.0
        T = float(env.mono_T(0, env.lens_pv(0))[0])
        env.line[0] = np.clip(np.round(T + 0.1, 1), -0.3, 1.0)
        P, right = env.compare_probs(E3.CMP_M)
        out.append(dict(S=S, label=label, msg=msg, line=float(env.line[0]), P=[float(x) for x in P[0]],
                        right=int(right[0])))
    return out


class PatientSays4(Section):
    T_INTRO = 9.0
    STEP = 11.0

    def __init__(self):
        self.rows = demo_rows()
        self.dur = self.T_INTRO + self.STEP * 3 + 1.5
        self.sfx = [(self.T_INTRO + self.STEP * i, "tick") for i in range(3)]
        from optics import to_pv
        self.E = to_pv(DEMO["S"], DEMO["C"], DEMO["axis"])

    def chart_pair(self, S, line):
        from optics import to_pv
        import scenes2
        which = max(0, int(np.argmin(np.abs(CH.LINES - min(line, 1.0)))) - 1)
        imgs = []
        for s_ in (S, S - 0.25):
            L = to_pv(s_, 0.0, 180.0)
            R = self.E[0] - L[0]
            A = scenes2.acc(R, 0.0, DEMO["amp"])
            imgs.append(ui._view(round(R - A, 3), 0.0, 0.0, round(DEMO["pupil"], 2), which))
        return imgs, which

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        if t < self.T_INTRO:
            al = fade(t, 0.3, self.T_INTRO, 0.5, 0.5)
            big_title(d, W / 2, 200, "The patient can only answer", 54, al)
            answers = getattr(self, "ANSWERS", None) or [
                ("“1.”  “2.”  “They look the same.”", CU.SPHERE), ("“I can read down to 20/40.”", CU.CHART),
                ("“The top one.”  “The bottom one.”", CU.PRISM)]
            for j, (s_, col) in enumerate(answers):
                aj = fade(t, 1.4 + 1.1 * j, self.T_INTRO, 0.5, 0.5)
                big_title(d, W / 2, 360 + 110 * j, s_, 48, aj, col, "SemiBold")
            big_title(d, W / 2, 760, "How sure they are depends on how close the lenses are.", 38,
                      fade(t, 5.0, self.T_INTRO, 0.5, 0.5), GOLD, "SemiBold")
            return fade_frame(im, a_all)
        i = min(int((t - self.T_INTRO) // self.STEP), 2)
        tt = t - self.T_INTRO - self.STEP * i
        r = self.rows[i]
        al = fade(tt, 0, self.STEP, 0.4, 0.4) * a_all
        text(d, (60, 40), f"A 55-YEAR-OLD WHO NEEDS −2.75   ·   {r['label']}", 34, GOLD, "Bold", a=al)
        text(d, (60, 96), f"In front of the eye: {fmt_d(r['S'])}.   “Which is clearer, 1 or 2?”  (2 adds −0.25)", 30,
             INK, "SemiBold", a=al)
        imgs, which = self.chart_pair(r["S"], r["line"])
        rows = ui.proj_chart(which)[1]
        li = int(np.argmin(np.abs(CH.LINES - r["line"])))
        row = [rw for rw in rows if rw[0] == li]
        yc, hh = (row[0][1], row[0][2]) if row else (ui.CHART_H / 2, 40)
        half = max(4.0 * hh, 230)
        y0c, y1c = int(max(0, yc - half)), int(min(ui.CHART_H, yc + half))
        for j, (img, lab) in enumerate(zip(imgs, ("1", "2"))):
            x0 = 90 + j * 530
            crop = Image.fromarray(img).convert("RGB").crop((0, y0c, ui.CHART_W, y1c))
            sc = min(480 / crop.size[0], 620 / crop.size[1])
            src = crop.resize((int(crop.size[0] * sc), int(crop.size[1] * sc)), Image.LANCZOS)
            ytop = int(480 - src.size[1] / 2)
            if al > 0.02:
                tile = Image.new("RGB", src.size, BG)
                tile = Image.blend(tile, src, min(1, al))
                im.paste(tile, (x0, ytop))
            d = ImageDraw.Draw(im, "RGBA")
            ui.badge(d, x0 + 40, ytop + 40, lab, GOLD, a=al, r=32)
        # the answer odds as one segmented bar: right / same / wrong
        P, right = r["P"], r["right"]
        p_right = P[right]
        p_same = P[2] if right != 2 else 0.0
        p_wrong = 1 - p_right - p_same
        x0, y0, w = 1180, 260, 660
        text(d, (x0, 190), "WHAT THEY'LL SAY", 24, MUTED, "Bold", a=al)
        g = smooth((tt - 0.8) / 1.2)
        segs = [(p_right, GREEN, "right"), (p_same, MUTED, "“the same”"), (p_wrong, RED, "wrong")]
        x = x0
        for p_, col, lab in segs:
            ww = w * p_ * g
            if ww > 1:
                d.rectangle([x, y0, x + ww, y0 + 70], fill=tuple(col) + (int(230 * al),))
            x += ww
        y = y0 + 100
        for p_, col, lab in segs:
            text(d, (x0, y), "■", 30, col, "Bold", a=al * g)
            text(d, (x0 + 44, y), lab, 30, INK, "SemiBold", a=al * g)
            text(d, (x0 + w, y), f"{100 * p_:.0f}%", 34, INK, "Bold", "ra", al * g)
            y += 56
        am = fade(tt, 2.6, self.STEP, 0.5, 0.4) * a_all
        para(d, x0, 600, r["msg"], 30, 680, GOLD, "SemiBold", am, lh=1.3)
        big_title(d, W / 2, 900, ["Far off", "Close", "Right on it"][i], 40, al * 0.0)
        return fade_frame(im, a_all)


# ======================================================================== what it wants
class Wants4(Section):
    dur = 30.0
    ROWS = [("happy", "Happy", "comes back and refers a friend", "+100"),
            ("annoyed", "Annoyed", "“not quite crisp”, might not come back", "+30"),
            ("recheck", "Rx check", "comes back, 15 unpaid minutes, never returns", "−80"),
            ("review", "Bad review", "Rx check and a one-star review", "−150")]

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 50, "What the AI wants: points", 54, fade(t, 0.3, 999, 0.5, 0.1) * a_all)
        y = 180
        for i, (o, name, desc, pts) in enumerate(self.ROWS):
            al = fade(t, 1.5 + 1.6 * i, 999, 0.5, 0.1) * a_all
            col = CU.OUT_COL[o]
            rr(d, [260, y, 1660, y + 110], 18, fill=CARD + (int(255 * al),), outline=tuple(col) + (int(255 * al),), width=3)
            text(d, (300, y + 55), name, 40, col, "Bold", "lm", al)
            text(d, (620, y + 55), desc, 30, INK2, "Medium", "lm", al)
            text(d, (1620, y + 55), pts, 52, col, "Bold", "rm", al)
            y += 130
        al = fade(t, 8.5, 999, 0.5, 0.1) * a_all
        rr(d, [260, y, 1660, y + 110], 18, fill=CARD + (int(255 * al),), outline=CU.CHART + (int(255 * al),), width=3)
        text(d, (300, y + 55), "Every minute", 40, CU.CHART, "Bold", "lm", al)
        text(d, (620, y + 55), "in the chair", 30, INK2, "Medium", "lm", al)
        text(d, (1620, y + 55), "−4", 52, CU.CHART, "Bold", "rm", al)
        al = fade(t, 12.0, 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 870, "The closer the glasses, the more likely “happy”.", 40, al, INK, "SemiBold")
        al = fade(t, 16.0, 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 950, "Right glasses, happy patients, quick exams. Nobody tells it how.", 34, al, GOLD,
                  "SemiBold")
        return np.asarray(im)


# ======================================================================== how practice works
class Learn4(Section):
    dur = 40.0
    STEPS = [("1", "It tries something", "Presses buttons, asks questions, and at some point prescribes."),
             ("2", "It gets the score", "Points for the glasses, minus the chair time. One number per patient."),
             ("3", "It adjusts", "Buttons it pressed before a better-than-expected score become a little more likely "
                                 "next time. Worse than expected: a little less likely.")]

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 50, "How practice works", 54, fade(t, 0.3, 999, 0.5, 0.1) * a_all)
        y = 190
        for i, (n, h, s) in enumerate(self.STEPS):
            al = fade(t, 1.5 + 4.0 * i, 999, 0.5, 0.1) * a_all
            ui.badge(d, 330, y + 50, n, AI_COL, a=al, r=44)
            text(d, (410, y + 8), h, 46, INK, "Bold", a=al)
            para(d, 410, y + 72, s, 30, 1180, INK2, "Medium", al, lh=1.3)
            y += 200
        al = fade(t, 14.5, 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 800, "One patient teaches it almost nothing. Millions do.", 42, al, GOLD, "SemiBold")
        al = fade(t, 20.0, 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 880, "Every patient is simulated: 1,024 exam rooms at once, on one computer.", 32, al, INK2,
                  "Medium")
        al = fade(t, 25.0, 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 950, getattr(self, "LAST", "Early on, a mentor also whispered “warmer” or “colder” after "
                                                       "each button."), 32, al, INK2, "Medium")
        return np.asarray(im)


if __name__ == "__main__":
    import os
    for S, ts in ((Toolbox4(), (1.5, 8.0, 20.0, 37.5)), (PatientSays4(), (5.0, 14.0, 26.0, 37.0)), (Wants4(), (20.0,)),
                  (Learn4(), (30.0,))):
        for t in ts:
            Image.fromarray(S.frame(t)).save(os.path.join(ui.OUT, f"test4_{type(S).__name__}_{int(t)}.png"))
    print("ok", round(Toolbox4().dur, 1), round(PatientSays4().dur, 1))
