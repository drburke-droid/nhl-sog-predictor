"""The clinic exam player: the real phoropter (colour-coded), what the patient sees, the patient's answer box with
the odds of every answer, the clinic scorecard, and every button the AI has (with its odds for each).
"""
import math
import numpy as np
from PIL import Image, ImageDraw
import ui
import chart as CH
import clinic
import env3 as E3
import clinic_ui as CU
from ui import (W, H, BG, INK, INK2, MUTED, GOLD, RED, GREEN, CYAN, PANEL, CARD, LINE, text, rr, fmt_rx, fmt_d, smooth,
                fade)
from scenes import Section
from scenes2 import acc
from optics import to_pv, from_pv, rx_error, jcc_pv

EYE = {0: "right eye", 1: "left eye", 2: "both eyes"}
QUESTIONS = (E3.READ, E3.READ_LINE, E3.CMP_P, E3.CMP_M, E3.JCC_AX, E3.JCC_PW, E3.BALANCE)
COMPARES = (E3.CMP_P, E3.CMP_M, E3.JCC_AX, E3.JCC_PW, E3.BALANCE)


def doing(a, ph):
    e = EYE[min(ph, 2)]
    return {E3.SP25: f"+0.25 sphere, {e}", E3.SM25: f"−0.25 sphere, {e}", E3.SP100: f"+1.00 sphere, {e}",
            E3.SM100: f"−1.00 sphere, {e}", E3.CYL_MORE: f"More cylinder (−0.25), {e}",
            E3.CYL_LESS: f"Less cylinder (+0.25), {e}", E3.AXP2: "Turns the axis +2°", E3.AXM2: "Turns the axis −2°",
            E3.AXP5: "Turns the axis +5°", E3.AXM5: "Turns the axis −5°", E3.AXP15: "Turns the axis +15°",
            E3.AXM15: "Turns the axis −15°", E3.READ: "“Read the smallest line you can.”",
            E3.READ_LINE: "“Read this line.”", E3.LINE_SM: "Shows a smaller line", E3.LINE_LG: "Shows a bigger line",
            E3.CMP_P: "“Which is clearer, 1 or 2?”", E3.CMP_M: "“Which is clearer, 1 or 2?”",
            E3.JCC_AX: "Cross-cylinder: “1 or 2?”", E3.JCC_PW: "Cross-cylinder: “1 or 2?”",
            E3.NEXT: {0: "Covers the right eye, tests the left", 1: "Opens both eyes"}.get(ph, "Next"),
            E3.OU_P: "+0.25 on both eyes", E3.OU_M: "−0.25 on both eyes",
            E3.BALANCE: "Prism split: “Top or bottom: which is clearer?”", E3.R_P: "+0.25 on the right eye only",
            E3.L_P: "+0.25 on the left eye only", E3.RX: "Prescribes"}[a]


def note_for(a, ph, scr):
    return {E3.CMP_P: "1 = as it is,  2 = with +0.25" + (" on both eyes" if ph == 2 else ""),
            E3.CMP_M: "1 = as it is,  2 = with −0.25" + (" on both eyes" if ph == 2 else ""),
            E3.JCC_AX: "the cross-cylinder flips: straddling the cylinder axis",
            E3.JCC_PW: "the cross-cylinder flips: along the cylinder axis",
            E3.BALANCE: "prism makes two copies of the line: top = right eye, bottom = left eye",
            E3.READ_LINE: f"the {CH.snellen_of(scr)} line"}.get(a)


def say(s):
    """The patient's reply, hesitant when they are unsure."""
    a, code = s["a"], s["code"]
    if a == E3.READ:
        ln, kk, nl = s["line"], s["k"], s["nl"]
        if ln > 1.05:
            return "“I can’t read any of it.”"
        return f"“Down to {CH.snellen_of(ln)}…”" + ("" if kk == nl else f" (missed {nl - kk})")
    if a == E3.READ_LINE:
        return f"“I got {s['k']} of the {s['nl']}.”"
    p = s.get("ans_probs")
    conf = 1.0 if p is None else p[code]
    if a == E3.BALANCE:
        base = {0: "top", 1: "bottom", 2: "they look the same"}[code]
    else:
        base = {0: "one", 1: "two", 2: "they look the same"}[code]
    if conf >= 0.8:
        return f"“{base.capitalize()}.”"
    if conf >= 0.5:
        return f"“Umm… {base}.”"
    return f"“Hmm… {base}… I think?”"


BEAT = {E3.READ: 1.6, E3.READ_LINE: 1.5, E3.CMP_P: 2.2, E3.CMP_M: 2.2, E3.JCC_AX: 2.2, E3.JCC_PW: 2.2,
        E3.BALANCE: 2.2, E3.NEXT: 1.2, E3.RX: 1.0}


class Exam3(Section):
    """speed: scales every beat; fast: [(k0, k1, factor)] to speed up stretches of the exam."""

    def __init__(self, trace, title, subtitle, speed=1.0, fast=(), notes=(), intro=1.4, outro=6.5, who="AI",
                 show_ai_probs=True, max_steps=None, panel_title="THE AI'S BUTTONS"):
        self.tr = trace
        self.title, self.subtitle, self.who = title, subtitle, who
        self.p = trace["patient"]
        self.E = [tuple(x) for x in trace["true_pv"]]
        self.steps = trace["steps"][:max_steps] if max_steps else trace["steps"]
        self.notes = dict(notes)
        self.show_ai_probs = show_ai_probs
        self.panel_title = panel_title
        self.beats, t = [], intro
        for k, s in enumerate(self.steps):
            d = BEAT.get(s["a"], 0.55) * speed
            for k0, k1, f in fast:
                if k0 <= k < k1:
                    d *= f
            if k in self.notes:
                d = max(d, 3.0)
            self.beats.append((t, t + d, k))
            t += d
        self.t_end_steps = t
        self.dur = t + outro
        self.sfx = [(t0, "tick") for t0, t1, _ in self.beats if t1 - t0 > 0.2] + [(self.t_end_steps, "bell")]

    # ---------------------------------------------------------------- optics of what the patient sees
    def resid(self, e, lens):
        L = to_pv(*lens)
        E = self.E[e]
        return E[0] - L[0], E[1] - L[1], E[2] - L[2]

    def view(self, lenses, both, screen, eye):
        p = self.p
        which = max(0, int(np.argmin(np.abs(CH.LINES - min(screen, 1.0)))) - 1)
        R = [self.resid(e, lenses[e]) for e in (0, 1)]
        if not both:
            A = acc(R[eye][0], p["prox"], p["amp"])
        else:
            A = min(acc(R[0][0], 0.3 * p["prox"], p["amp"]), acc(R[1][0], 0.3 * p["prox"], p["amp"]))
        imgs = [ui._view(round(R[e][0] - A, 3), round(R[e][1], 3), round(R[e][2], 3), round(p["pupil"], 2), which)
                for e in (0, 1)]
        shown = eye
        if both:
            tA = [math.hypot(math.hypot(R[e][0] - A, R[e][1]), R[e][2]) for e in (0, 1)]
            shown = int(np.argmin(tA))
        return imgs, shown, which

    def state(self, t):
        if not self.beats or t < self.beats[0][0]:
            return dict(k=-1, a=None, dt=0, d=1)
        for t0, t1, k in self.beats:
            if t0 <= t < t1:
                return dict(k=k, a=self.steps[k]["a"], dt=t - t0, d=t1 - t0)
        return dict(k=len(self.steps) - 1, a=self.steps[-1]["a"], dt=99, d=1, done=True)

    def outcome_now(self, lenses):
        e = [rx_error(to_pv(*lenses[i]), self.E[i]) for i in (0, 1)]
        ept = clinic.patient_error(e[0], e[1])
        return clinic.outcome_probs(np.array([ept]))[0], float(clinic.expected_points(np.array([ept]))[0])

    # ---------------------------------------------------------------- drawing
    def frame(self, t):
        st = self.state(t)
        im, d = ui.new_frame()
        k, a = st["k"], st["a"]
        done = st.get("done", False)
        s = self.steps[k] if k >= 0 else None
        first = self.steps[0]
        lenses_b = s["before"] if s else first["before"]
        lenses_a = s["after"] if s else first["before"]
        ph_b = s["phase_before"] if s else 0
        ph_a = s["phase"] if s else 0
        scr_b = s["screen_before"] if s else first["screen_before"]
        scr_a = s["screen"] if s else first["screen_before"]
        late = st["dt"] > st["d"] * 0.45
        changed = a is not None and a not in QUESTIONS and st["dt"] > 0.12
        lenses = lenses_a if (changed or done) else lenses_b
        ph = ph_a if (changed or done) else ph_b
        both = ph == 2
        screen = scr_a if (changed or (a == E3.READ and late) or done) else scr_b
        eye = min(ph, 1)

        # ---- the patient's view
        CX, CY, CW, CHh = 930, 96, 360, 556
        badge, label = None, None
        asking = a in COMPARES and not done
        if a == E3.BALANCE and not done:
            imgs, _, which = self.view(lenses, True, screen, 0)
            rows = ui.proj_chart(which)[1]
            li = int(np.argmin(np.abs(CH.LINES - screen)))
            row = [rw for rw in rows if rw[0] == li]
            yc, h = (row[0][1], row[0][2]) if row else (ui.CHART_H / 2, 40)
            half = int(min(ui.CHART_H / 2 - 20, h * 0.5 + 14))
            top = imgs[0][int(max(0, yc - half)):int(yc + half)]
            bot = imgs[1][int(max(0, yc - half)):int(yc + half)]
            img = np.full((ui.CHART_H, ui.CHART_W), imgs[0][0, 0], np.uint8)
            img[ui.CHART_H // 4 - top.shape[0] // 2: ui.CHART_H // 4 - top.shape[0] // 2 + top.shape[0]] = top
            img[3 * ui.CHART_H // 4 - bot.shape[0] // 2: 3 * ui.CHART_H // 4 - bot.shape[0] // 2 + bot.shape[0]] = bot
            label = "PRISM SPLIT: TOP RIGHT EYE, BOTTOM LEFT"
        elif asking:
            seg = st["d"] / 3
            lens_cur = [tuple(x) for x in lenses_b]
            S, C, ax = lens_cur[eye]
            if a in (E3.CMP_P, E3.CMP_M):
                ds = 0.25 if a == E3.CMP_P else -0.25
                lA = lens_cur
                if both:
                    lB = [(x[0] + ds, x[1], x[2]) for x in lens_cur]
                else:
                    lB = list(lens_cur)
                    lB[eye] = (S + ds, C, ax)
                vA, eA, _ = self.view(lA, both, screen, eye)
                vB, eB, _ = self.view(lB, both, screen, eye)
                iA, iB = vA[eA], vB[eB]
            else:
                off = (45, -45) if a == E3.JCC_AX else (0, 90)
                M, J0, J45 = to_pv(S, C, ax)
                out = []
                for o in off:
                    _, j0, j45 = jcc_pv(ax + o)
                    Sx, Cx, axx = from_pv(M, J0 + j0, J45 + j45)
                    ll = list(lens_cur)
                    ll[eye] = (float(Sx), float(Cx), float(axx))
                    v, _, _ = self.view(ll, False, screen, eye)
                    out.append(v[eye])
                iA, iB = out
            vC, eC, _ = self.view(lens_cur, both, screen, eye)
            iC = vC[eC]
            dt = st["dt"]
            if dt < seg:
                m = smooth(dt / 0.25)
                img, badge = (iC * (1 - m) + iA * m).astype(np.uint8), "1"
            elif dt < 2 * seg:
                m = smooth((dt - seg) / 0.2)
                img, badge = (iA * (1 - m) + iB * m).astype(np.uint8), "2"
            else:
                m = smooth((dt - 2 * seg) / 0.3)
                img = (iB * (1 - m) + iC * m).astype(np.uint8)
            label = "BOTH EYES" if both else EYE[eye].upper()
        else:
            imgs, shown, which = self.view(lenses, both, screen, eye)
            img = imgs[shown]
            label = "BOTH EYES OPEN" if both else (EYE[eye].upper() + " (OTHER COVERED)")
        text(d, (CX - 10, CY - 34), "WHAT THE PATIENT SEES", 18, MUTED, "Bold")
        text(d, (CX - 10, CY + CHh + 14), label, 16, MUTED, "SemiBold")
        rr(d, [CX - 8, CY - 8, CX + CW + 8, CY + CHh + 8], 12, fill=(30, 36, 48))
        crop = Image.fromarray(img).resize((CW, CHh), Image.LANCZOS)
        im.paste(crop, (CX, CY))
        d = ImageDraw.Draw(im, "RGBA")
        sx, sy = CW / ui.CHART_W, CHh / ui.CHART_H
        if a != E3.BALANCE or done:
            top_i = max(0, int(np.argmin(np.abs(CH.LINES - min(screen, 1.0)))) - 1)
            for li, yc, h in ui.proj_chart(top_i)[1]:
                on = abs(CH.LINES[li] - screen) < 0.01
                if on:
                    strong = a in (E3.READ, E3.READ_LINE) and late
                    d.rounded_rectangle([CX - 4, CY + (yc - h / 2 - 8) * sy, CX + CW + 4, CY + (yc + h / 2 + 8) * sy], 8,
                                        outline=GOLD + (230 if strong else 110,), width=4 if strong else 2)
                    text(d, (CX + CW - 6, CY + (yc - h / 2 - 8) * sy - 4), CH.snellen_of(CH.LINES[li]), 16,
                         GOLD, "Bold", "rd")
        if badge:
            ui.badge(d, CX + 40, CY + 40, badge, CU.col_of(a), r=30)

        # ---- the phoropter
        hot = None
        if a is not None and not done and st["dt"] < st["d"] * 0.9:
            g = CU.BTN[a][5]
            e_hot = None if (both or g in ("occ", "prism")) else eye
            if a in (E3.R_P,):
                e_hot = 0
            if a in (E3.L_P,):
                e_hot = 1
            if g == "sphere" and a in (E3.CMP_P, E3.CMP_M):
                e_hot = None if both else eye
            hot = (g, e_hot) if g not in ("chart", "rx") else None
        jcc = None
        if a in (E3.JCC_AX, E3.JCC_PW) and not done:
            seg = st["d"] / 3
            jcc = 1 if st["dt"] < seg else (2 if st["dt"] < 2 * seg else None)
        state = dict(S=[x[0] for x in lenses], C=[x[1] for x in lenses], ax=[x[2] for x in lenses], phase=ph, jcc=jcc,
                     prism=(a == E3.BALANCE and not done))
        CU.draw_phoropter(im, 14, 60, 0.6, state, hot=hot)
        d = ImageDraw.Draw(im, "RGBA")

        # ---- title, the hidden truth
        text(d, (24, 14), self.title, 30, INK, "Bold")
        tw = d.textlength(self.title, font=ui.font(30, "Bold"))
        text(d, (24 + tw + 18, 22), self.subtitle, 21, MUTED, "Medium")
        text(d, (CX - 10, 640 + 0), "", 10)

        # ---- what the AI is doing, and the reply
        yA = 690
        if s is not None and a is not None and not done:
            text(d, (24, yA - 4), self.who, 20, CYAN, "Bold")
            xw = 24 + d.textlength(self.who, font=ui.font(20, "Bold")) + 14
            text(d, (xw, yA - 10), doing(a, ph_b), 27, INK, "SemiBold")
            nt = note_for(a, ph_b, scr_b)
            show = (a in (E3.READ, E3.READ_LINE) and late) or (a in COMPARES and st["dt"] > 2 * st["d"] / 3)
            if show:
                rep = say(s)
                tw = d.textlength(doing(a, ph_b), font=ui.font(27, "SemiBold"))
                text(d, (xw + tw + 24, yA - 4), "PATIENT", 18, GOLD, "Bold")
                text(d, (xw + tw + 110, yA - 10), rep, 27, GOLD, "SemiBold")
            elif nt:
                tw = d.textlength(doing(a, ph_b), font=ui.font(27, "SemiBold"))
                text(d, (xw + tw + 20, yA - 3), nt, 19, MUTED, "Medium")

        # ---- the answer box
        AX, AY, AWd, AH = 1330, 14, 576, 352
        last_q = None
        for j in range(k, -1, -1):
            if self.steps[j]["a"] in QUESTIONS:
                last_q = j
                break
        if last_q is None:
            CU.draw_answers(d, AX, AY, AWd, AH)
        else:
            sq = self.steps[last_q]
            current = last_q == k and not done
            dim = 1.0 if current else 0.55
            rev = smooth(st["dt"] / (st["d"] * 0.4)) if current else 1.0
            shown_ans = (not current) or (st["dt"] > (2 * st["d"] / 3 if sq["a"] in COMPARES else st["d"] * 0.45))
            if sq["a"] == E3.READ:
                CU.draw_answers(d, AX, AY, AWd, AH, read={float(k_): v for k_, v in sq.get("read_dist", {}).items()},
                                chosen=sq["line"] if shown_ans else None, reveal=rev, alpha=dim)
            else:
                chosen = (sq.get("cat") if sq["a"] == E3.READ_LINE else sq["code"]) if shown_ans else None
                CU.draw_answers(d, AX, AY, AWd, AH, sq["a"], probs=sq.get("ans_probs"), chosen=chosen,
                                right=sq.get("right"), reveal=rev, alpha=dim)

        # ---- the clinic scorecard
        secs = s["secs"] if (s is not None and (changed or done or st["dt"] > st["d"] * 0.5)) else (
            s["secs_before"] if s else 0.0)
        pr, pts = self.outcome_now(lenses)
        last = self.steps[-1]
        if done:
            al = smooth((t - self.t_end_steps) / 0.5)
            pr = np.array(last["outcome_p"])
            CU.draw_scorecard(d, AX, 382, AWd, 300, secs / 60, probs=pr, outcome=int(np.argmax(pr)) if al > 0.5 else None,
                              points=last["points"] - clinic.POINTS_PER_MIN * secs / 60, title="THE VERDICT")
        else:
            CU.draw_scorecard(d, AX, 382, AWd, 300, secs / 60, probs=pr, title="IF IT PRESCRIBED NOW")

        # ---- the buttons, with the AI's odds
        probs = s.get("ai_probs") if (s is not None and self.show_ai_probs and not done) else None
        CU.draw_buttons(d, 14, 730, 1892, 336, pressed=a if (a is not None and not done) else None, probs=probs,
                        mask=E3.MASK[min(ph_b, 2)], heading=self.panel_title)
        # moves so far, as coloured ticks
        mx0, my0 = 1040, 922
        nm = max(k + 1, 0)
        nq = sum(1 for x in self.steps[:k + 1] if x["a"] in QUESTIONS) if k >= 0 else 0
        text(d, (mx0, my0), f"{nm} BUTTONS PRESSED  ·  {nq} QUESTIONS", 17, MUTED, "Bold")
        per = 60
        for i_, sx in enumerate(self.steps[:k + 1]):
            r_, c_ = divmod(i_, per)
            x = mx0 + c_ * 14.3
            y = my0 + 30 + r_ * 22
            if y > 1056:
                break
            col = CU.col_of(sx["a"])
            d.rectangle([x, y, x + 10.5, y + 16], fill=tuple(col) + (255 if i_ == k else 190,))

        # ---- result card
        if done:
            al = smooth((t - self.t_end_steps) / 0.4)
            rr(d, [8, 686, 1912, 1074], 20, fill=BG + (int(250 * al),))
            rr(d, [18, 696, 1902, 1064], 20, fill=PANEL + (int(255 * al),), outline=LINE + (int(255 * al),), width=3)
            r2 = lambda v: math.floor(abs(v) * 100 + 0.5) / 100
            for e in (0, 1):
                S, C, ax = last["after"][e]
                L = to_pv(S, C, ax)
                E = self.E[e]
                err = rx_error(L, E)
                col = GREEN if err < 0.13 else (GOLD if err <= 0.25 + 1e-9 else RED)
                y = 724 + e * 150
                tr_ = (self.p["S"], self.p["C"], self.p["axis"]) if e == 0 else (self.p["S_os"], self.p["C_os"], self.p["axis_os"])
                text(d, (56, y), "RIGHT EYE" if e == 0 else "LEFT EYE", 19, MUTED, "Bold", a=al)
                text(d, (56, y + 28), "Prescribed  " + fmt_rx(S, C, ax), 34, INK, "SemiBold", a=al)
                text(d, (56, y + 76), "True  " + fmt_rx(*tr_), 24, INK2, "Medium", a=al)
                verdict = "Exact" if err < 0.13 else ("Within 0.25 D" if err <= 0.25 + 1e-9 else f"Off by {r2(err):.2f} D")
                text(d, (960, y + 34), verdict, 38, col, "Bold", "ra", al)
            m_, s_ = divmod(int(round(last["secs"])), 60)
            text(d, (1040, 724), f"{len(self.steps)} BUTTONS  ·  {m_}:{s_:02d} IN THE CHAIR", 19, MUTED, "Bold", a=al)
            per = 56
            for i_, sx in enumerate(self.steps):
                r_, c_ = divmod(i_, per)
                y = 760 + r_ * 24
                if y > 1040:
                    break
                col = CU.col_of(sx["a"])
                d.rectangle([1040 + c_ * 14.6, y, 1040 + c_ * 14.6 + 11, y + 18], fill=tuple(col) + (int(220 * al),))
        if k in self.notes and a is not None and not done:
            al = fade(st["dt"], 0.1, st["d"], 0.25, 0.3)
            ui.ST.caption(d, [(self.notes[k], 28, INK, "SemiBold")], al, W=W, H=H, anchor="bottom", cx=660)
        return np.asarray(im)


if __name__ == "__main__":
    import os, sys
    from record3 import record3
    from clinicians3 import textbook3
    tr = record3(script=textbook3)
    ex = Exam3(tr, "THE TEXTBOOK", "the same patient, examined by the textbook script", who="TEXTBOOK",
               show_ai_probs=False, speed=0.6, panel_title="THE SAME BUTTONS")
    ks = {}
    for k, s in enumerate(tr["steps"]):
        ks.setdefault(s["a"], k)
    for nm, a in (("bal", E3.BALANCE), ("jcc", E3.JCC_PW), ("cmp", E3.CMP_M), ("read", E3.READ), ("sph", E3.SM100)):
        if a in ks:
            t0, t1, _ = ex.beats[ks[a]]
            Image.fromarray(ex.frame(t0 + (t1 - t0) * 0.8)).save(os.path.join(ui.OUT, f"test3_{nm}.png"))
    Image.fromarray(ex.frame(ex.dur - 0.5)).save(os.path.join(ui.OUT, "test3_end.png"))
    print("dur", ex.dur)
