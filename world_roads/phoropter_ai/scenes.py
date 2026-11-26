"""Video sections. Each section has .dur and .frame(t) -> uint8 HxWx3, and is rendered to out/<name>.mp4.

    python scenes.py <section>       render one section
"""
import os, sys, json, math, subprocess
import numpy as np
from PIL import Image, ImageDraw
import ui
from ui import (W, H, FPS, BG, INK, INK2, MUTED, GOLD, RED, GREEN, CYAN, PANEL, CARD, LINE, text, rr, fmt_rx, fmt_d,
                smooth, fade)
from optics import to_pv, snellen, pv_dist, rx_error

HERE = ui.HERE
DATA = os.path.join(HERE, "data")

SAY = {0: "“One.”", 1: "“Two.”", 2: "“They look the same.”"}
DOING = ["Sphere +0.25", "Sphere −0.25", "Sphere +1.00", "Sphere −1.00", "Cylinder −0.25 (more)",
         "Cylinder +0.25 (less)", "Rotate axis +5°", "Rotate axis −5°", "Rotate axis +15°",
         "Rotate axis −15°", "“Read the smallest line you can.”", "“Which is clearer, 1 or 2?”",
         "“Which is clearer, 1 or 2?”", "“Which is clearer, 1 or 2?”",
         "“Which is clearer, 1 or 2?”", "Done: prescribe this", "Show a smaller line", "Show a bigger line",
         "“Read this line.”"]
NOTE = {11: "2 adds +0.25 sphere", 12: "2 adds −0.25 sphere", 13: "cross-cylinder flip, straddling the axis",
        14: "cross-cylinder flip, on the axis"}
CHIP = {0: ("SPH +", CYAN), 1: ("SPH \u2212", CYAN), 2: ("SPH +1", CYAN), 3: ("SPH \u22121", CYAN),
        4: ("CYL \u2212", GREEN), 5: ("CYL +", GREEN), 6: ("AX +5", GOLD), 7: ("AX \u22125", GOLD), 8: ("AX +15", GOLD),
        9: ("AX \u221215", GOLD), 10: ("READ", INK), 11: ("1/2 +", INK2), 12: ("1/2 \u2212", INK2),
        13: ("JCC ax", RED), 14: ("JCC pwr", RED), 15: ("DONE", GREEN), 16: ("LINE \u2212", MUTED),
        17: ("LINE +", MUTED), 18: ("READ LINE", INK)}

CH_X, CH_Y = 110, 214                               # patient-view chart position


def chart_line_for(va):
    """The chart line a reported logMAR corresponds to (nearest line at or above it)."""
    best = None
    for k, (lm, name, _) in enumerate(ui.CHART_LINES):
        if lm >= va - 0.051:
            best = k
    return best


class Section:
    dur = 1.0
    sfx = []

    def frame(self, t):
        raise NotImplementedError

    def render(self, name):
        os.makedirs(ui.OUT, exist_ok=True)
        path = os.path.join(ui.OUT, f"{name}.mp4")
        n = int(round(self.dur * FPS))
        p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                              "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        for i in range(n):
            fr = self.frame(i / FPS)
            p.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
        p.stdin.close()
        p.wait()
        print("wrote", path, f"{n / FPS:.1f}s")


# ======================================================================== the exam player
class Exam(Section):
    """Plays a recorded exam. trace: from record.py. speed: <1 is faster.
    notes: [(step_index, text)] commentary shown while that step plays (step_index counts from 0)."""

    def __init__(self, trace, title, subtitle, speed=1.0, notes=(), intro=1.6, outro=4.0, max_steps=None,
                 show_probs=False):
        self.tr = trace
        self.title, self.subtitle = title, subtitle
        self.p = trace["patient"]
        self.E = tuple(trace["true_pv"])
        steps = trace["steps"][:max_steps] if max_steps else trace["steps"]
        self.steps = steps
        self.truncated = max_steps is not None and max_steps < len(trace["steps"])
        self.notes = dict(notes)
        self.show_probs = show_probs
        self.letters = bool(steps[0].get("letters"))      # attempt 3: projector chart, read letter by letter
        self.beats = []
        t = intro
        for k, s in enumerate(steps):
            a = s["a"]
            if a < 10 or a in (16, 17):
                d = 0.9
            elif a in (10, 18):
                d = 1.9
            elif a < 15:
                d = 2.7
            else:
                d = 1.2
            d *= speed
            if k in self.notes:
                d = max(d, 2.6)
            self.beats.append((t, t + d, k))
            t += d
        self.t_end_steps = t
        self.dur = t + outro
        self.sfx = []
        for t0, t1, k in self.beats:
            a = steps[k]["a"]
            self.sfx.append((t0, "tick"))
        self.sfx.append((self.t_end_steps, "bell"))

    # ---------------------------------------------------------------- state at time t
    def state(self, t):
        if t < self.beats[0][0] if self.beats else True:
            S, C, ax = self.steps[0]["before"]
            return dict(k=-1, u=0, lens=(S, C, ax), view="cur", a=None)
        for t0, t1, k in self.beats:
            if t0 <= t < t1:
                return dict(k=k, u=(t - t0) / (t1 - t0), dt=t - t0, d=t1 - t0, lens=self.steps[k]["before"],
                            a=self.steps[k]["a"])
        k = len(self.steps) - 1
        return dict(k=k, u=1.0, dt=99, d=1, lens=self.steps[k]["after"], a=self.steps[k]["a"], done=True)

    def view(self, lens_pv, screen=None):
        which = None
        if self.letters:
            import chart as CH
            which = max(0, int(np.argmin(np.abs(CH.LINES - min(screen, 1.0)))) - 1)
        img, acc = ui.patient_view(self.E, lens_pv, self.p["pupil"], self.p["amp"], which)
        return img, acc

    def screens(self, k):
        """Projector line before and after step k."""
        before = self.steps[k - 1].get("screen", 0.3) if k > 0 else 0.3
        return before, (self.steps[k].get("screen", 0.3) if k >= 0 else 0.3)

    # ---------------------------------------------------------------- drawing
    def frame(self, t):
        st = self.state(t)
        im, d = ui.new_frame()
        k, a = st["k"], st["a"]
        step = self.steps[k] if k >= 0 else None

        # which lens the patient is looking through, and (letter chart) which line is on the screen
        badge = None
        sb, sa = self.screens(k) if k >= 0 else (0.3, 0.3)
        if st.get("done"):
            sb = sa
        screen = sb
        if a is None:
            L = to_pv(*st["lens"])
            img, acc = self.view(L, sb)
        elif a < 10 or a in (15, 16, 17):
            L0, L1 = to_pv(*step["before"]), to_pv(*step["after"])
            i0, a0 = self.view(L0, sb)
            i1, a1 = self.view(L1, sa)
            m = smooth((st.get("dt", 99) - 0.1) / 0.3)
            img = (i0 * (1 - m) + i1 * m).astype(np.uint8)
            acc = a1 if m > 0.5 else a0
            screen = sa if m > 0.5 else sb
        elif a in (10, 18):
            late = st["dt"] > st["d"] * 0.45
            screen = sa if (late and a == 10) else sb
            img, acc = self.view(tuple(step["L"]), screen)
        else:
            dt = st["dt"]
            seg = st["d"] / 3
            iA, aA = self.view(tuple(step["A"]), sb)
            iB, aB = self.view(tuple(step["B"]), sb)
            iC, aC = self.view(tuple(step["L"]), sb)
            if dt < seg:
                m = smooth(dt / 0.25)
                img = (iC * (1 - m) + iA * m).astype(np.uint8)
                acc, badge = aA, "1"
            elif dt < 2 * seg:
                m = smooth((dt - seg) / 0.2)
                img = (iA * (1 - m) + iB * m).astype(np.uint8)
                acc, badge = aB, "2"
            else:
                m = smooth((dt - 2 * seg) / 0.3)
                img = (iB * (1 - m) + iC * m).astype(np.uint8)
                acc = aC
        # ---- left: the patient's view
        text(d, (CH_X, CH_Y - 42), "WHAT THE PATIENT SEES", 22, MUTED, "SemiBold")
        rr(d, [CH_X - 10, CH_Y - 10, CH_X + ui.CHART_W + 10, CH_Y + ui.CHART_H + 10], 14, fill=(30, 36, 48))
        im.paste(Image.fromarray(img), (CH_X, CH_Y))
        d = ImageDraw.Draw(im, "RGBA")
        if self.letters:
            import chart as CH
            top = max(0, int(np.argmin(np.abs(CH.LINES - min(screen, 1.0)))) - 1)
            for li, yc, h in ui.proj_chart(top)[1]:
                on = abs(CH.LINES[li] - screen) < 0.01
                text(d, (CH_X + ui.CHART_W + 24, CH_Y + yc), CH.snellen_of(CH.LINES[li]), 19, GOLD if on else MUTED,
                     "SemiBold" if on else "Medium", "lm")
                if on:
                    strong = a in (10, 18) and st["dt"] > st["d"] * 0.45
                    d.rounded_rectangle([CH_X - 4, CH_Y + yc - h / 2 - 10, CH_X + ui.CHART_W + 4, CH_Y + yc + h / 2 + 10],
                                        10, outline=GOLD + (230 if strong else 120,), width=4 if strong else 2)
        else:
            ui.snellen_labels(d, CH_X + ui.CHART_W + 24, CH_Y, size=19)
        if badge:
            ui.badge(d, CH_X + 46, CH_Y + 46, badge, GOLD, r=34)
        if a == 10 and not self.letters and st["dt"] > 0.7 * st["d"] * 0.45:
            li = chart_line_for(step["va"])
            if li is not None:
                _, ys = ui.chart()
                y = CH_Y + ys[li]
                h = 5 * 10 ** ui.CHART_LINES[li][0] / ui.PX
                d.rounded_rectangle([CH_X - 4, y - h / 2 - 10, CH_X + ui.CHART_W + 4, y + h / 2 + 10], 10,
                                    outline=GOLD + (230,), width=4)
        if acc >= 0.12:
            text(d, (CH_X, CH_Y + ui.CHART_H + 30), f"Patient is focusing {math.floor(acc * 100 + 0.5) / 100:.2f} D "
                 "to cancel the extra minus", 22, GOLD, "SemiBold")

        # ---- top: titles and the hidden answer
        text(d, (80, 50), self.title, 44, INK, "Bold")
        text(d, (80, 108), self.subtitle, 26, MUTED, "Medium")
        rr(d, [1390, 40, 1850, 132], 16, fill=CARD, outline=LINE, width=2)
        text(d, (1410, 56), "HIDDEN FROM THE AI: TRUE Rx", 17, MUTED, "SemiBold")
        text(d, (1410, 82), fmt_rx(self.p["S"], self.p["C"], self.p["axis"]), 32, INK2, "SemiBold")

        # ---- right: phoropter and readout
        S, C, ax = (step["after"] if (a is not None and (a < 10 or a >= 15) and st.get("dt", 99) > 0.15)
                    else (step["before"] if step else st["lens"]))
        jcc = None
        if a in (13, 14):
            dt, seg = st["dt"], st["d"] / 3
            off = (45, -45) if a == 13 else (0, 90)
            if dt < seg:
                jcc = ax + off[0]
            elif dt < 2 * seg:
                jcc = ax + off[1]
        RX = 1300
        ui.phoropter(d, RX, 300, 0.78, ax, C, jcc=jcc)
        y0 = 450
        rr(d, [820, y0, 1780, y0 + 128], 18, fill=CARD, outline=LINE, width=2)
        for i, (lab, val) in enumerate((("SPHERE", fmt_d(S)), ("CYLINDER", fmt_d(C, plus=False) if C < 0 else "0.00"),
                                         ("AXIS", f"{int(round(ax)):03d}°" if C < 0 else "—"))):
            cx = 980 + i * 320
            text(d, (cx, y0 + 22), lab, 18, MUTED, "SemiBold", "ma")
            text(d, (cx, y0 + 52), val, 50, INK, "SemiBold", "ma")
        # action and answer
        if step is not None and a is not None:
            text(d, (820, 618), "AI", 22, CYAN, "Bold")
            text(d, (870, 610), DOING[a], 36, INK, "SemiBold")
            note = NOTE.get(a)
            if self.letters and a in (16, 17, 18):
                import chart as CH
                note = f"the {CH.snellen_of(sa if a != 18 else sb)} line"
            if note:
                text(d, (870, 660), note, 22, MUTED, "Medium")
            show_ans = (a in (10, 18) and st["dt"] > st["d"] * 0.45) or (10 < a < 15 and st["dt"] > 2 * st["d"] / 3)
            if show_ans:
                text(d, (820, 712), "PATIENT", 22, GOLD, "Bold")
                if a == 10 and self.letters:
                    import chart as CH
                    ln, kk, nl = step["line"], step["k"], step["nl"]
                    if ln > 1.05:
                        ans = "“I can’t read the top line.”"
                    elif kk == nl:
                        ans = f"“All of the {CH.snellen_of(ln)} line.”"
                    else:
                        ans = f"“The {CH.snellen_of(ln)} line, but I missed {nl - kk}.”"
                elif a == 10:
                    li = chart_line_for(step["va"])
                    ans = (f"“I can read {ui.CHART_LINES[li][1]}.”" if li is not None
                           else "“I can’t read any of it.”")
                elif a == 18:
                    ans = f"“I got {step['k']} of the {step['nl']}.”"
                else:
                    ans = SAY[step["code"]]
                text(d, (950, 702), ans, 36, GOLD, "SemiBold")
        # move strip
        nq = sum(1 for s in self.steps[:k + 1] if 10 <= s["a"] < 15 or s["a"] == 18) if k >= 0 else 0
        text(d, (820, 790), f"MOVE {max(k + 1, 0)}" + (f"  ·  {nq} QUESTION{'S' if nq != 1 else ''} ASKED" if nq else ""),
             20, MUTED, "SemiBold")
        x, y = 820, 826
        for i, s in enumerate(self.steps[:k + 1]):
            lab, col = CHIP[s["a"]]
            f = ui.font(17, "SemiBold")
            w = d.textlength(lab, font=f) + 20
            if x + w > 1790:
                x, y = 820, y + 38
            alpha = 255 if i == k else 150
            rr(d, [x, y, x + w, y + 30], 8, fill=tuple(c // 4 for c in col) + (alpha,), outline=col + (alpha,), width=2)
            d.text((x + w / 2, y + 15), lab, font=f, fill=col + (alpha,), anchor="mm")
            x += w + 8

        # commentary
        if k in self.notes and a is not None:
            al = fade(st["dt"], 0.1, st["d"], 0.25, 0.3)
            ui.ST.caption(d, [(self.notes[k], 30, INK, "SemiBold")], al, y=None, W=W, H=H, anchor="bottom", cx=1300)

        # result card: what is left over, in clinical terms
        if t >= self.t_end_steps:
            al = smooth((t - self.t_end_steps) / 0.4)
            last = self.steps[-1]
            S, C, ax = last["after"]
            L = to_pv(S, C, ax)
            dse = L[0] - self.E[0]
            cres = 2 * math.hypot(L[1] - self.E[1], L[2] - self.E[2])
            err = rx_error(L, self.E)
            ok = err < 0.13
            col = GREEN if ok else (GOLD if err <= 0.25 + 1e-9 else RED)
            rr(d, [800, 596, 1800, 1040], 22, fill=BG + (int(255 * al),))
            rr(d, [820, 600, 1780, 1010], 22, fill=PANEL + (int(255 * al),), outline=col + (int(255 * al),), width=3)
            text(d, (860, 628), "PRESCRIBED", 20, MUTED, "SemiBold", a=al)
            text(d, (860, 654), fmt_rx(S, C, ax), 44, INK, "SemiBold", a=al)
            text(d, (860, 724), "TRUE Rx", 20, MUTED, "SemiBold", a=al)
            text(d, (860, 750), fmt_rx(self.p["S"], self.p["C"], self.p["axis"]), 44, INK2, "SemiBold", a=al)
            r2 = lambda x: math.floor(abs(x) * 100 + 0.5) / 100
            se_txt = ("spot on" if abs(dse) < 0.07 else
                      f"{r2(dse):.2f} D too much {'minus' if dse < 0 else 'plus'}")
            cy_txt = "none" if cres < 0.07 else f"{r2(cres):.2f} D"
            text(d, (860, 836), "SPHERICAL EQUIVALENT", 20, MUTED, "SemiBold", a=al)
            text(d, (860, 862), se_txt, 32, INK, "SemiBold", a=al)
            text(d, (1330, 836), "LEFTOVER CYLINDER", 20, MUTED, "SemiBold", a=al)
            text(d, (1330, 862), cy_txt, 32, INK, "SemiBold", a=al)
            verdict = "Exact" if ok else ("Within 0.25 D" if err <= 0.25 + 1e-9 else f"Off by {r2(err):.2f} D")
            text(d, (1740, 640), verdict, 44, col, "Bold", "ra", al)
            text(d, (1740, 700), f"{len(self.steps)} move{'s' if len(self.steps) != 1 else ''}", 28, MUTED,
                 "SemiBold", "ra", al)
            text(d, (860, 940), "Error = length of (spherical-equivalent error, leftover cylinder)", 20, MUTED,
                 "Medium", a=al)
        return np.asarray(im)


# ======================================================================== text helpers
def wrap(d, s, size, width, weight="Medium"):
    f = ui.font(size, weight)
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def para(d, x, y, s, size, width, col=INK, weight="Medium", a=1.0, lh=1.32, anchor="la"):
    for ln in wrap(d, s, size, width, weight):
        text(d, (x, y), ln, size, col, weight, anchor, a)
        y += size * lh
    return y


def big_title(d, cx, y, s, size, a, col=INK, weight="Bold"):
    text(d, (cx, y), s, size, col, weight, "ma", a)


# ======================================================================== hook
class Hook(Section):
    """The phoropter dials toward a clear chart while the premise appears."""
    dur = 11.0
    LENSES = [(0.0, 0.0, 170), (-1.0, 0.0, 170), (-1.75, 0.0, 170), (-2.25, 0.0, 170), (-2.25, -0.5, 170),
              (-2.25, -1.0, 170)]
    TIMES = [0.0, 1.6, 2.8, 4.0, 5.4, 6.8]

    def __init__(self):
        from record import SHOWCASE as P
        self.P = P
        self.E = to_pv(P["S"], P["C"], P["axis"])
        self.sfx = [(t, "tick") for t in self.TIMES[1:]]

    def frame(self, t):
        im, d = ui.new_frame()
        k = max(i for i, tt in enumerate(self.TIMES) if t >= tt)
        i1, _ = ui.patient_view(self.E, to_pv(*self.LENSES[k]), self.P["pupil"], self.P["amp"])
        if k > 0:
            i0, _ = ui.patient_view(self.E, to_pv(*self.LENSES[k - 1]), self.P["pupil"], self.P["amp"])
            m = smooth((t - self.TIMES[k]) / 0.35)
            img = (i0 * (1 - m) + i1 * m).astype(np.uint8)
        else:
            img = i1
        a_all = fade(t, 0, self.dur, 0.6, 0.4)
        rr(d, [CH_X - 10, CH_Y - 10, CH_X + ui.CHART_W + 10, CH_Y + ui.CHART_H + 10], 14, fill=(30, 36, 48))
        im.paste(Image.fromarray(img), (CH_X, CH_Y))
        d = ImageDraw.Draw(im, "RGBA")
        S, C, ax = self.LENSES[k]
        ui.phoropter(d, 1300, 260, 0.78, ax, C)
        rr(d, [920, 410, 1680, 500], 16, fill=CARD, outline=LINE, width=2)
        for i, (lab, val) in enumerate((("SPHERE", fmt_d(S)), ("CYLINDER", fmt_d(C, False) if C < 0 else "0.00"),
                                         ("AXIS", f"{ax:03d}\u00b0" if C < 0 else "\u2014"))):
            cx = 1046 + i * 254
            text(d, (cx, 424), lab, 16, MUTED, "SemiBold", "ma")
            text(d, (cx, 446), val, 36, INK, "SemiBold", "ma")
        big_title(d, 1300, 600, "I gave an AI a phoropter.", 66, fade(t, 0.5, 99, 0.6, 0.1))
        big_title(d, 1300, 700, "It has never seen an eye, a lens,", 40, fade(t, 3.6, 99, 0.6, 0.1), INK2, "SemiBold")
        big_title(d, 1300, 752, "or an eye chart.", 40, fade(t, 3.6, 99, 0.6, 0.1), INK2, "SemiBold")
        big_title(d, 1300, 860, "Can it teach itself to refract?", 44, fade(t, 7.2, 99, 0.6, 0.1), GOLD, "SemiBold")
        fr = np.asarray(im).astype(np.float32)
        return (fr * a_all + np.array(BG) * (1 - a_all)).astype(np.uint8)


# ======================================================================== the rules
class Rules(Section):
    dur = 36.0
    T = (0.4, 12.0, 24.0)

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (80, 56), "THE RULES", 44, INK, "Bold", a=a_all)
        text(d, (80, 114), "Everything below was set up before training started.", 26, MUTED, "Medium", a=a_all)
        xs = (80, 680, 1280)
        heads = ("THE PATIENT", "THE AI\u2019S 16 BUTTONS", "THE SCORE")
        for i, (x, h) in enumerate(zip(xs, heads)):
            a = fade(t, self.T[i], 99, 0.6, 0.1) * a_all
            if a <= 0:
                continue
            hi = self.T[i] <= t < (self.T[i + 1] if i < 2 else 99)
            rr(d, [x, 180, x + 560, 1000], 22, fill=CARD + (int(255 * a),),
               outline=(GOLD if hi else LINE) + (int(255 * a),), width=3 if hi else 2)
            text(d, (x + 36, 214), h, 26, GOLD if hi else INK2, "Bold", a=a)
        # card 1
        b1 = ["A simulated person with a hidden prescription.",
              "The phoropter starts at their autorefractor reading.",
              "Real optics: leftover sphere and astigmatism blur the chart.",
              "They focus like a real eye, so too much minus can still look clear.",
              "Their answers are a bit noisy, and some think extra minus looks \u201ccrisper.\u201d"]
        y = 280
        for j, b in enumerate(b1):
            a = fade(t, self.T[0] + 0.8 + 1.5 * j, 99, 0.5, 0.1) * a_all
            text(d, (116, y + 4), "\u2022", 26, GOLD, "Bold", a=a)
            y = para(d, 144, y, b, 26, 470, INK, "Medium", a) + 22
        # card 2: buttons
        groups = [("CHANGE A LENS", ["SPH +0.25", "SPH \u22120.25", "SPH +1.00", "SPH \u22121.00", "CYL \u22120.25",
                                     "CYL +0.25", "AXIS +5\u00b0", "AXIS \u22125\u00b0", "AXIS +15\u00b0",
                                     "AXIS \u221215\u00b0"], CYAN),
                  ("ASK THE PATIENT", ["Read the chart", "1 or 2? (+0.25)", "1 or 2? (\u22120.25)", "Cross-cyl: axis",
                                       "Cross-cyl: power"], GOLD),
                  ("FINISH", ["DONE"], GREEN)]
        y = 284
        f = ui.font(21, "SemiBold")
        n = 0
        for gname, items, col in groups:
            a = fade(t, self.T[1] + 0.6 + 1.6 * [g[0] for g in groups].index(gname), 99, 0.5, 0.1) * a_all
            text(d, (716, y), gname, 18, MUTED, "SemiBold", a=a)
            y += 34
            x = 716
            for it in items:
                w = d.textlength(it, font=f) + 28
                if x + w > 1206:
                    x, y = 716, y + 50
                rr(d, [x, y, x + w, y + 40], 10, fill=tuple(c // 5 for c in col) + (int(255 * a),),
                   outline=col + (int(255 * a),), width=2)
                d.text((x + w / 2, y + 20), it, font=f, fill=col + (int(255 * a),), anchor="mm")
                x += w + 10
            y += 76
        a = fade(t, self.T[1] + 6.0, 99, 0.6, 0.1) * a_all
        para(d, 716, 860, "Nobody tells it what any answer means.", 28, 480, GOLD, "SemiBold", a)
        # card 3
        a = fade(t, self.T[2] + 0.6, 99, 0.5, 0.1) * a_all
        para(d, 1316, 280, "Nothing during the exam. Only when it presses DONE:", 26, 480, INK, "Medium", a)
        para(d, 1316, 372, "How far is its prescription from the true one?", 30, 480, INK, "SemiBold", a)
        rows = [("Exact", "full marks", GREEN), ("0.25 D off", "40%", GOLD), ("0.50 D off", "nothing", RED)]
        for j, (l, r_, col) in enumerate(rows):
            a = fade(t, self.T[2] + 2.2 + 0.7 * j, 99, 0.4, 0.1) * a_all
            yy = 490 + 62 * j
            text(d, (1316, yy), l, 30, col, "SemiBold", a=a)
            text(d, (1800, yy), r_, 30, INK2, "Medium", "ra", a)
        a = fade(t, self.T[2] + 4.8, 99, 0.5, 0.1) * a_all
        para(d, 1316, 700, "Every question costs a little chair time.", 26, 480, INK2, "Medium", a)
        a = fade(t, self.T[2] + 6.6, 99, 0.5, 0.1) * a_all
        para(d, 1316, 800, "Then the next patient. Millions of times.", 30, 480, GOLD, "SemiBold", a)
        return np.asarray(im)


# ======================================================================== practice counter (between stages)
class Counter(Section):
    def __init__(self, n0, n1, line, dur=3.2):
        self.n0, self.n1, self.line, self.dur = n0, n1, line, dur

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.35, 0.35)
        u = smooth((t - 0.3) / (self.dur - 1.0))
        n = int(round(np.exp(np.log(max(self.n0, 1)) + (np.log(self.n1) - np.log(max(self.n0, 1))) * u))) \
            if self.n0 > 0 else int(round(self.n1 * u ** 3))
        text(d, (960, 380), "PRACTICE EXAMS", 30, MUTED, "SemiBold", "ma", a)
        text(d, (960, 440), f"{n:,}", 120, INK, "Bold", "ma", a)
        text(d, (960, 640), self.line, 36, INK2, "SemiBold", "ma", a)
        return np.asarray(im)


if __name__ == "__main__":
    from policy import load
    from record import record
    agent, meta = load(sys.argv[1])
    tr = record(agent, greedy=True)
    ex = Exam(tr, "TEST EXAM", f"after {meta.get('exams', 0):,} practice exams")
    os.makedirs(ui.OUT, exist_ok=True)
    for t in (0.5, ex.beats[0][0] + 0.5, ex.beats[1][0] + 1.0, ex.beats[1][0] + 2.0, ex.dur - 1):
        Image.fromarray(ex.frame(t)).save(os.path.join(ui.OUT, f"test_exam_{t:.1f}.png"))
    print("beats", len(ex.beats), "dur", ex.dur)
