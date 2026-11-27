"""The exam player for the version-5 clinic exam (env5.py): one idea at a time, both eyes' status always shown,
and the cylinder steps named plainly (astigmatism check, cylinder axis, cylinder power, axis spin).

Based on scenes4.py.

* Left: the real phoropter, only the part being used lit up, with the lens values on it.
* Right: what the patient sees. For "1 or 2?" both choices sit side by side, labelled 1 and 2.
* Bottom: one line of dialogue, what the examiner did or asked, then what the patient said and how sure.
* Top: chair time, and a progress bar through the stages (right eye, left eye, both eyes; an eye can come
  back after a complaint).
* Key moments play slowly with a caption; repetitive stretches fast-forward with a ▶▶ badge.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
import ui
import chart as CH
import clinic
import env5 as E3
import clinic_ui5 as CU
from ui import W, H, BG, INK, INK2, MUTED, GOLD, RED, GREEN, CARD, LINE, PANEL, text, rr, fmt_rx, fmt_d, smooth, fade
from scenes3 import Exam3, EYE
QUESTIONS = (E3.READ, E3.READ_LINE, E3.CMP_P, E3.CMP_M, E3.JCC_AX, E3.JCC_PW, E3.CHK_180, E3.CHK_45, E3.BALANCE, E3.SPIN,
             E3.RX)                                       # a refused prescription: the patient's complaint
COMPARES = (E3.CMP_P, E3.CMP_M, E3.JCC_AX, E3.JCC_PW, E3.CHK_180, E3.CHK_45, E3.BALANCE)
from optics import to_pv, from_pv, rx_error, jcc_pv

AI_COL = (0x33, 0x8C, 0xED)
TB_COL = (0xE7, 0x66, 0x24)


def said(s):
    """What the patient said, in their own words (how sure they were is part of the answer)."""
    a, code = s["a"], s["code"]
    if a == E3.READ:
        ln, kk, nl = s["line"], s["k"], s["nl"]
        if ln > 1.05:
            return "“I can’t read any of it.”", None
        return f"“I can read down to {CH.snellen_of(ln)}.”", None
    if a == E3.READ_LINE:
        return f"“I got {s['k']} of the {s['nl']} letters.”", None
    if a == E3.SPIN:
        return ("“Stop! There.”", None) if code == 5 else ("“Nothing changes.”", None)
    if a == E3.RX:
        return {7: "“Wait, my right eye is still blurry.”", 8: "“Wait, my left eye is still blurry.”",
                9: "“Wait, it’s still blurry. Both eyes.”"}.get(code, "“Thanks!”"), None
    st = s.get("strength", 1)
    if a == E3.BALANCE:
        word = {0: "The top one", 1: "The bottom one", 2: "They look the same"}[code]
    else:
        word = {0: "One", 1: "Two", 2: "They look the same"}[code]
    if code == 2:
        return f"“Umm… {word.lower()}.”", None
    return (f"“{word}, much clearer.”", None) if st == 2 else (f"“{word}… a little clearer.”", None)


def action_line(a, ph, before, after):
    e = min(ph, 1)
    eye = {0: "right eye", 1: "left eye", 2: "both eyes"}[min(ph, 2)]
    if a in (E3.SP25, E3.SM25, E3.SP100, E3.SM100):
        return f"Sphere {CU.BTN[a][1]}  →  {eye} now {ui.fmt_d(after[e][0]).replace('Plano', '0.00')}"
    if a in (E3.CYL_MORE, E3.CYL_LESS):
        return f"Cylinder {CU.BTN[a][1]}  →  {eye} now {ui.fmt_d(after[e][1], False) if after[e][1] < 0 else '0.00'}"
    if a in (E3.AXP2, E3.AXM2, E3.AXP5, E3.AXM5, E3.AXP15, E3.AXM15):
        return f"Turns the cylinder axis {CU.BTN[a][1]}  →  {int(round(after[e][2])) % 180 or 180}°"
    if a in (E3.OU_P, E3.OU_M):
        return f"{CU.BTN[a][1]} on both eyes  →  R {ui.fmt_d(after[0][0])}, L {ui.fmt_d(after[1][0])}"
    if a in (E3.R_P, E3.L_P):
        return f"+0.25 on the {'right' if a == E3.R_P else 'left'} eye only"
    if a == E3.SPIN:
        return f"Spins the cylinder axis: “Say stop when it’s sharpest”  →  {int(round(after[e][2])) % 180 or 180}°"
    return {E3.READ: "“Read the smallest line you can.”", E3.READ_LINE: "“Read this line.”",
            E3.LINE_SM: "Shows smaller letters", E3.LINE_LG: "Shows bigger letters",
            E3.CMP_P: "Sphere: “Which is clearer, 1 or 2?”   (2 adds +0.25)",
            E3.CMP_M: "Sphere: “Which is clearer, 1 or 2?”   (2 adds −0.25)",
            E3.JCC_AX: "Cylinder AXIS: “1 or 2?”   (tells it which way to turn)",
            E3.JCC_PW: "Cylinder POWER: “1 or 2?”   (more or less cylinder)",
            E3.CHK_180: "Astigmatism check, 180° or 90°: “1 or 2?”",
            E3.CHK_45: "Astigmatism check, 45° or 135°: “1 or 2?”",
            E3.NEXT: {0: "Covers the right eye: now the left eye", 1: "Opens both eyes",
                      2: "Covers the left eye: back to the right eye"}.get(ph, "Next"),
            E3.BALANCE: "Prism splits the chart: “Top or bottom, which is clearer?”",
            E3.RX: "Prescribes: “Here are your new glasses.”"}[a]


class Exam5(Exam3):
    """slow: step indices to play at full speed (others fast-forward). notes: {k: caption} (always slow)."""

    def __init__(self, trace, title, subtitle, who="AI", who_col=AI_COL, notes=(), slow=(), intro=1.6, outro=7.0,
                 fast_beat=0.32):
        self.tr = trace
        self.title, self.subtitle, self.who, self.who_col = title, subtitle, who, who_col
        self.p = trace["patient"]
        self.E = [tuple(x) for x in trace["true_pv"]]
        self.steps = trace["steps"]
        self.notes = dict(notes)
        slow = set(slow) | set(self.notes)
        self.beats, t = [], intro
        self.fastk = set()
        for k, s in enumerate(self.steps):
            a = s["a"]
            if k in slow:
                d = 3.0 if a in COMPARES else (2.6 if a in (E3.READ, E3.READ_LINE) else 1.6)
                if k in self.notes:
                    d = max(d, 4.2)
            else:
                d = fast_beat
                self.fastk.add(k)
            self.beats.append((t, t + d, k))
            t += d
        self.t_end_steps = t
        self.dur = t + outro
        self.sfx = [(t0, "tick") for t0, t1, k in self.beats if k not in self.fastk] + [(self.t_end_steps, "bell")]
        self.total_secs = self.steps[-1]["secs"]

    def frame(self, t):
        st = self.state(t)
        im, d = ui.new_frame()
        k, a = st["k"], st["a"]
        done = st.get("done", False)
        s = self.steps[k] if k >= 0 else None
        first = self.steps[0]
        fast = k in self.fastk and not done
        lenses_b = s["before"] if s else first["before"]
        lenses_a = s["after"] if s else first["before"]
        ph_b = s["phase_before"] if s else 0
        ph_a = s["phase"] if s else 0
        scr_b = s["screen_before"] if s else first["screen_before"]
        scr_a = s["screen"] if s else first["screen_before"]
        changed = a is not None and a not in QUESTIONS and (fast or st["dt"] > 0.25)
        lenses = lenses_a if (changed or done) else lenses_b
        ph = ph_a if (changed or done) else ph_b
        both = ph == 2
        screen = scr_a if (changed or (a == E3.READ and st["dt"] > st["d"] * 0.4) or done) else scr_b
        eye = min(ph, 1)

        # ---- top bar
        text(d, (40, 26), self.title, 36, INK, "Bold")
        tw = d.textlength(self.title, font=ui.font(36, "Bold"))
        text(d, (40 + tw + 18, 38), self.subtitle, 22, MUTED, "Medium")
        secs = (s["secs"] if s is not None and (changed or done or st["dt"] > st["d"] * 0.5) else
                (s["secs_before"] if s else 0.0))
        m_, s_ = divmod(int(round(secs)), 60)
        text(d, (1880, 26), f"{m_}:{s_:02d}", 40, INK, "Bold", "ra")
        text(d, (1880 - d.textlength(f"{m_}:{s_:02d}", font=ui.font(40, "Bold")) - 14, 40), "chair time", 20, MUTED,
             "Medium", "ra")
        # progress through the stages: one segment per stretch between occluder changes (the occluder cycles
        # right -> left -> both -> right, so an eye can come back)
        x0, x1, yb = 40, 1880, 92
        tot = max(self.total_secs, 1)
        bounds, phs, p_ = [0.0], [0], 0
        for sx in self.steps:
            if sx["a"] == E3.NEXT:
                bounds.append(sx["secs"])
                p_ = (p_ + 1) % 3
                phs.append(p_)
        bounds.append(tot)
        labs = ["RIGHT EYE", "LEFT EYE", "BOTH EYES"]
        for j in range(len(bounds) - 1):
            xa = x0 + (x1 - x0) * bounds[j] / tot
            xb = x0 + (x1 - x0) * bounds[j + 1] / tot
            on = bounds[j] <= secs < bounds[j + 1] or (j == len(bounds) - 2 and secs >= bounds[j])
            rr(d, [xa + 2, yb, max(xa + 3, xb - 2), yb + 30], 6, fill=(255, 255, 255, 40 if on else 14))
            fillw = (x1 - x0) * min(max(secs - bounds[j], 0), bounds[j + 1] - bounds[j]) / tot
            if fillw > 1:
                rr(d, [xa + 2, yb, xa + 2 + fillw, yb + 30], 6, fill=tuple(self.who_col) + (150,))
            lab = labs[phs[j]]
            if xb - xa > d.textlength(lab, font=ui.font(16, "Bold")) + 12:
                text(d, ((xa + xb) / 2, yb + 15), lab, 16, INK if on else MUTED, "Bold", "mm")

        # ---- which eye is doing what
        for e, (lab, x) in enumerate((("RIGHT EYE", 60), ("LEFT EYE", 520))):
            if both:
                stt, col = "open", GREEN
            elif e == eye:
                stt, col = "being tested", GREEN
            else:
                stt, col = "covered", MUTED
            rr(d, [x, 140, x + 420, 186], 23, fill=tuple(col) + (40,), outline=tuple(col) + (255,), width=2)
            text(d, (x + 22, 163), lab, 22, INK, "Bold", "lm")
            text(d, (x + 400, 163), stt, 22, col, "SemiBold", "rm")

        # ---- the phoropter: only the part in use lit up
        hot = None
        if a is not None and not done:
            g = CU.BTN[a][5]
            e_hot = None if (both or g in ("occ", "prism")) else eye
            if a == E3.R_P:
                e_hot = 0
            if a == E3.L_P:
                e_hot = 1
            hot = (g, e_hot) if g not in ("chart", "rx") else None
        jcc = None
        if a in (E3.JCC_AX, E3.JCC_PW) and not done:
            jcc = "1 / 2"
        state = dict(S=[x[0] for x in lenses], C=[x[1] for x in lenses], ax=[x[2] for x in lenses], phase=ph,
                     jcc=None, prism=(a == E3.BALANCE and not done))
        CU.draw_phoropter(im, 50, 200, 0.56, state, hot=hot, legend_only=[hot[0]] if hot else ["none"])
        d = ImageDraw.Draw(im, "RGBA")

        # ---- what the patient sees
        PX0, PY0, PW, PH = 1010, 172, 870, 600
        text(d, (PX0, PY0 - 8), "WHAT THE PATIENT SEES", 20, MUTED, "Bold", "ld")
        asking = a in COMPARES and not done
        if a == E3.BALANCE and not done:
            imgs, _, which = self.view(lenses, True, screen, 0)
            rows = ui.proj_chart(which)[1]
            li = int(np.argmin(np.abs(CH.LINES - screen)))
            row = [rw for rw in rows if rw[0] == li]
            yc, h = (row[0][1], row[0][2]) if row else (ui.CHART_H / 2, 40)
            half = int(min(ui.CHART_H / 4 - 10, h * 0.5 + 14))
            top = imgs[0][int(max(0, yc - half)):int(yc + half)]
            bot = imgs[1][int(max(0, yc - half)):int(yc + half)]
            tile = np.full((ui.CHART_H // 2 * 2, ui.CHART_W), imgs[0][0, 0], np.uint8)
            tile[ui.CHART_H // 4 - top.shape[0] // 2: ui.CHART_H // 4 - top.shape[0] // 2 + top.shape[0]] = top
            tile[3 * ui.CHART_H // 4 - bot.shape[0] // 2: 3 * ui.CHART_H // 4 - bot.shape[0] // 2 + bot.shape[0]] = bot
            self._panel(im, d, tile, PX0 + (PW - 400) / 2, PY0 + 4, 400, PH - 8, None)
            text(d, (PX0 + (PW + 400) / 2 + 16, PY0 + PH * 0.25), "TOP: right eye", 22, GOLD, "SemiBold", "lm")
            text(d, (PX0 + (PW + 400) / 2 + 16, PY0 + PH * 0.75), "BOTTOM: left eye", 22, GOLD, "SemiBold", "lm")
        elif asking:
            iA, iB = self._options(a, lenses_b, both, eye, screen)
            self._panel(im, d, iA, PX0, PY0 + 4, 420, PH - 8, "1")
            self._panel(im, d, iB, PX0 + 450, PY0 + 4, 420, PH - 8, "2")
        else:
            imgs, shown, which = self.view(lenses, both, screen, eye)
            self._panel(im, d, imgs[shown], PX0 + (PW - 420) / 2, PY0 + 4, 420, PH - 8, None, screen=screen,
                        strong=a in (E3.READ, E3.READ_LINE) and st["dt"] > st["d"] * 0.4)
            lab = "both eyes open" if both else f"{EYE[eye]} only (the other covered)"
            text(d, (PX0 + PW / 2, PY0 + PH + 14), lab, 20, MUTED, "SemiBold", "ma")
        d = ImageDraw.Draw(im, "RGBA")

        # ---- the dialogue
        DY = 830
        if s is not None and a is not None and not done:
            col = CU.col_of(a)
            rr(d, [40, DY - 8, 1880, DY + 206], 18, fill=CARD + (235,), outline=LINE + (255,), width=2)
            text(d, (70, DY + 14), self.who, 26, self.who_col, "Bold")
            xw = 70 + d.textlength(self.who, font=ui.font(26, "Bold")) + 24
            rr(d, [xw, DY + 10, xw + 16, DY + 46], 4, fill=tuple(col) + (255,))
            text(d, (xw + 32, DY + 8), action_line(a, ph_b, lenses_b, lenses_a), 34, INK, "SemiBold")
            reply_at = st["d"] * (0.62 if a in COMPARES else 0.45)
            if a in QUESTIONS and (fast or st["dt"] > reply_at):
                rep, sure = said(s)
                text(d, (70, DY + 96), "PATIENT", 26, GOLD, "Bold")
                text(d, (230, DY + 90), rep, 38, GOLD, "SemiBold")

        if fast:
            rr(d, [880, 20, 1040, 76], 14, fill=(255, 255, 255, 36))
            text(d, (960, 48), "▶▶  fast", 26, INK, "Bold", "mm")
        if k in self.notes and a is not None and not done:
            al = fade(st["dt"], 0.1, st["d"], 0.3, 0.3)
            rr(d, [70, DY + 156, 78, DY + 186], 3, fill=(190, 215, 255, int(255 * al)))
            text(d, (94, DY + 150), self.notes[k], 28, (190, 215, 255), "SemiBold", "la", al)

        # ---- the result
        if done:
            al = smooth((t - self.t_end_steps) / 0.5)
            last = self.steps[-1]
            rr(d, [40, 640, 1880, 1060], 20, fill=PANEL + (int(250 * al),), outline=LINE + (int(255 * al),), width=3)
            r2 = lambda v: math.floor(abs(v) * 100 + 0.5) / 100
            if last.get("walked"):
                n_c = sum(x["code"] in (7, 8, 9) for x in self.steps)
                text(d, (W / 2, 700), "The patient walks out. No glasses.", 56, RED, "Bold", "ma", al)
                text(d, (W / 2, 790), f"It tried “Prescribe” {n_c} times. Every time: “It’s still blurry.”" if n_c else
                     "It never prescribed.", 34, INK2, "SemiBold", "ma", al)
                m2, s2 = divmod(int(round(last["secs"])), 60)
                net = last["points"] - clinic.POINTS_PER_MIN * last["secs"] / 60
                text(d, (80, 950), f"{len(self.steps)} buttons  ·  {m2}:{s2:02d} in the chair", 28, INK2, "SemiBold", a=al)
                text(d, (1840, 950), f"{net:+.0f} points", 44, INK, "Bold", "ra", al)
                return np.asarray(im)
            for e in (0, 1):
                S, C, ax = last["after"][e]
                err = rx_error(to_pv(S, C, ax), self.E[e])
                colr = GREEN if err < 0.13 else (GOLD if err <= 0.25 + 1e-9 else RED)
                tr_ = (self.p["S"], self.p["C"], self.p["axis"]) if e == 0 else (self.p["S_os"], self.p["C_os"], self.p["axis_os"])
                y = 670 + e * 130
                text(d, (80, y), "RIGHT EYE" if e == 0 else "LEFT EYE", 20, MUTED, "Bold", a=al)
                text(d, (80, y + 28), "Glasses  " + fmt_rx(S, C, ax), 36, INK, "SemiBold", a=al)
                text(d, (80, y + 76), "Truth     " + fmt_rx(*tr_), 26, INK2, "Medium", a=al)
                verdict = "Exact" if err < 0.13 else ("Within 0.25 D" if err <= 0.25 + 1e-9 else f"Off by {r2(err):.2f} D")
                text(d, (1000, y + 36), verdict, 38, colr, "Bold", "la", al)
            m2, s2 = divmod(int(round(last["secs"])), 60)
            net = last["points"] - clinic.POINTS_PER_MIN * last["secs"] / 60
            text(d, (80, 950), f"{len(self.steps)} buttons  ·  {m2}:{s2:02d} in the chair", 28, INK2, "SemiBold", a=al)
            text(d, (1840, 950), f"{net:+.0f} points", 44, INK, "Bold", "ra", al)
        return np.asarray(im)

    def _options(self, a, lenses_b, both, eye, screen):
        lens_cur = [tuple(x) for x in lenses_b]
        S, C, ax = lens_cur[eye]
        if a in (E3.CMP_P, E3.CMP_M):
            ds = 0.25 if a == E3.CMP_P else -0.25
            if both:
                lB = [(x[0] + ds, x[1], x[2]) for x in lens_cur]
            else:
                lB = list(lens_cur)
                lB[eye] = (S + ds, C, ax)
            vA, eA, _ = self.view(lens_cur, both, screen, eye)
            vB, eB, _ = self.view(lB, both, screen, eye)
            return vA[eA], vB[eB]
        if a in E3.CHK_ORIENT:
            angs = E3.CHK_ORIENT[a]
        else:
            off = (45, -45) if a == E3.JCC_AX else (0, 90)
            angs = (ax + off[0], ax + off[1])
        M, J0, J45 = to_pv(S, C, ax)
        out = []
        for ang in angs:
            _, j0, j45 = jcc_pv(ang)
            Sx, Cx, axx = from_pv(M, J0 + j0, J45 + j45)
            ll = list(lens_cur)
            ll[eye] = (float(Sx), float(Cx), float(axx))
            v, _, _ = self.view(ll, False, screen, eye)
            out.append(v[eye])
        return out

    def _panel(self, im, d, img, x, y, w, h, badge, screen=None, strong=False):
        src = Image.fromarray(img)
        sw, sh = src.size
        scale = min(w / sw, h / sh)
        tw, th = int(sw * scale), int(sh * scale)
        ox, oy = int(x + (w - tw) / 2), int(y + (h - th) / 2)
        d.rounded_rectangle([ox - 6, oy - 6, ox + tw + 6, oy + th + 6], 10, fill=(30, 36, 48))
        im.paste(src.resize((tw, th), Image.LANCZOS), (ox, oy))
        d2 = ImageDraw.Draw(im, "RGBA")
        if screen is not None:
            top_i = max(0, int(np.argmin(np.abs(CH.LINES - min(screen, 1.0)))) - 1)
            for li, yc, hh in ui.proj_chart(top_i)[1]:
                if abs(CH.LINES[li] - screen) < 0.01:
                    d2.rounded_rectangle([ox - 2, oy + (yc - hh / 2 - 8) * scale, ox + tw + 2, oy + (yc + hh / 2 + 8) * scale],
                                         8, outline=GOLD + (230 if strong else 110,), width=4 if strong else 2)
                    text(d2, (ox + tw + 14, oy + yc * scale), CH.snellen_of(CH.LINES[li]), 22, GOLD, "Bold", "lm")
        if badge:
            ui.badge(d2, ox + 44, oy + 44, badge, GOLD, r=34)


if __name__ == "__main__":
    import os, json
    tr = json.load(open(os.path.join(ui.HERE, "data", "traces", "h_textbook.json")))
    slow = [k for k, s in enumerate(tr["steps"]) if s["a"] in QUESTIONS][:6]
    ex = Exam5(tr, "THE AI", "test", notes={3: "A caption about this moment."}, slow=slow)
    ks = {}
    for k, s in enumerate(tr["steps"]):
        ks.setdefault(s["a"], k)
    for nm, a in (("jcc", E3.JCC_PW), ("chk", E3.CHK_180), ("read", E3.READ), ("bal", E3.BALANCE), ("sph", E3.SM100)):
        if a in ks:
            t0, t1, _ = ex.beats[ks[a]]
            Image.fromarray(ex.frame(t0 + (t1 - t0) * 0.85)).save(os.path.join(ui.OUT, f"test5_{nm}.png"))
    Image.fromarray(ex.frame(ex.beats[3][0] + 1.0)).save(os.path.join(ui.OUT, "test5_note.png"))
    Image.fromarray(ex.frame(ex.dur - 0.3)).save(os.path.join(ui.OUT, "test5_end.png"))
    print("dur", round(ex.dur, 1))
