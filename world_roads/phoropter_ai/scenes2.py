"""The two-eye exam player (attempt 4 and the scripted examiners in the two-eye exam room)."""
import math
import numpy as np
from PIL import Image, ImageDraw
import ui
import phoropter_art as PA
import chart as CH
from ui import (W, H, BG, INK, INK2, MUTED, GOLD, RED, GREEN, CYAN, PANEL, CARD, LINE, text, rr, fmt_rx, fmt_d, smooth,
                fade)
from scenes import Section, CHIP as CHIP1, DOING as DOING1, CH_X, CH_Y
from optics import to_pv, rx_error, jcc_pv

DOING = DOING1 + ["Test the right eye", "Test the left eye", "Both eyes open", "+0.25 on both eyes",
                  "−0.25 on both eyes", "“Top or bottom: which is clearer?”"]
CHIP = dict(CHIP1)
CHIP.update({19: ("R EYE", CYAN), 20: ("L EYE", CYAN), 21: ("BOTH", GREEN), 22: ("OU +", GOLD), 23: ("OU −", GOLD),
             24: ("BALANCE", RED)})
SAY = {0: "“One.”", 1: "“Two.”", 2: "“They look the same.”"}
SAY_BAL = {0: "“Top.”", 1: "“Bottom.”", 2: "“They look the same.”"}
QUESTION = (10, 11, 12, 13, 14, 18, 24)


def acc(R, floor, amp):
    fl = floor * np.clip(1 + R / 0.5, 0, 1)
    return float(np.clip(max(R, fl), 0, amp))


def phoropter2(d, cx, cy, s, eyes, occluded, cur, jcc=None, a=1.0):
    """Both apertures. eyes: [(axis, cyl), (axis, cyl)] for right, left. occluded: None, 0 or 1."""
    A = int(255 * a)
    body = (34, 42, 58, A)
    rr(d, [cx - 330 * s, cy - 120 * s, cx + 330 * s, cy + 120 * s], int(110 * s), fill=body, outline=(70, 84, 108, A),
       width=max(1, int(3 * s)))
    rr(d, [cx - 40 * s, cy - 150 * s, cx + 40 * s, cy - 100 * s], int(14 * s), fill=body)
    for k, ex in enumerate((-170, 170)):
        x, y, r = cx + ex * s, cy, 88 * s
        ring = GOLD if (k == cur and occluded is not None) else (90, 104, 128)
        d.ellipse([x - r - 14 * s, y - r - 14 * s, x + r + 14 * s, y + r + 14 * s], fill=(22, 28, 40, A),
                  outline=ring + (A,), width=max(2, int(4 * s)))
        if occluded == k:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(10, 14, 20, A))
            text(d, (x, y), "COVERED", int(20 * s), MUTED, "Medium", "mm", a)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(14, 30, 44, A), outline=(60, 120, 150, A), width=max(1, int(2 * s)))
            ax, cyl = eyes[k]
            if cyl < 0:
                t = math.radians(ax)
                dx, dy = math.cos(t) * r * 0.92, -math.sin(t) * r * 0.92
                d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=GOLD + (A,), width=max(2, int(5 * s)))
            if jcc is not None and k == cur:
                t = math.radians(jcc)
                for sgn in (1, -1):
                    for f in (0.35, 0.62):
                        px, py = x + sgn * math.cos(t) * r * f, y - sgn * math.sin(t) * r * f
                        d.ellipse([px - 9 * s, py - 9 * s, px + 9 * s, py + 9 * s], fill=(235, 70, 60, A))
                        qx, qy = x + sgn * math.cos(t + math.pi / 2) * r * f, y - sgn * math.sin(t + math.pi / 2) * r * f
                        d.ellipse([qx - 9 * s, qy - 9 * s, qx + 9 * s, qy + 9 * s], fill=(240, 240, 240, A))
        text(d, (x, cy + 128 * s), "RIGHT EYE" if k == 0 else "LEFT EYE", int(18 * s), MUTED, "Medium", "ma", a)


class Exam2(Section):
    def __init__(self, trace, title, subtitle, speed=1.0, notes=(), intro=1.4, outro=6.0, chips_max=40):
        self.tr = trace
        self.title, self.subtitle = title, subtitle
        self.p = trace["patient"]
        self.E = [tuple(x) for x in trace["true_pv"]]
        self.steps = trace["steps"]
        self.notes = dict(notes)
        self.chips_max = chips_max
        self.beats, t = [], intro
        for k, s in enumerate(self.steps):
            a = s["a"]
            if a in (10, 18):
                d = 1.5
            elif a in (11, 12, 13, 14, 24):
                d = 2.1
            elif a == 15:
                d = 1.0
            else:
                d = 0.6
            d *= speed
            if k in self.notes:
                d = max(d, 2.8)
            self.beats.append((t, t + d, k))
            t += d
        self.t_end_steps = t
        self.dur = t + outro
        self.sfx = [(t0, "tick") for t0, _, _ in self.beats] + [(self.t_end_steps, "bell")]

    # ---------------------------------------------------------------- optics of what the patient sees
    def resid(self, e, lens):
        L = to_pv(*lens)
        E = self.E[e]
        return E[0] - L[0], E[1] - L[1], E[2] - L[2]

    def view(self, lenses, mode, screen, eye=None):
        """mode: 'mono' (eye) or 'bin'. Returns (image, accommodation, eye shown)."""
        p = self.p
        which = max(0, int(np.argmin(np.abs(CH.LINES - min(screen, 1.0)))) - 1)
        R = [self.resid(e, lenses[e]) for e in (0, 1)]
        if mode == "mono":
            A = acc(R[eye][0], p["prox"], p["amp"])
            shown = eye
        else:
            A = min(acc(R[0][0], 0.3 * p["prox"], p["amp"]), acc(R[1][0], 0.3 * p["prox"], p["amp"]))
            shown = eye
        imgs = []
        for e in (0, 1):
            r = R[e]
            mar0 = p["mar0"] if e == 0 else p["mar0_os"]
            key = (round(r[0] - A, 3), round(r[1], 3), round(r[2], 3), round(p["pupil"], 2), which)
            imgs.append(ui._view(*key))
        if mode == "bin" and eye is None:
            # both eyes open: show the clearer eye's view (binocular vision is about as good as the better eye)
            tA = [np.hypot(np.hypot(R[e][0] - A, R[e][1]), R[e][2]) for e in (0, 1)]
            shown = int(np.argmin(tA))
        return imgs, A, shown, which

    def state(self, t):
        if not self.beats or t < self.beats[0][0]:
            return dict(k=-1, a=None, dt=0, d=1)
        for t0, t1, k in self.beats:
            if t0 <= t < t1:
                return dict(k=k, a=self.steps[k]["a"], dt=t - t0, d=t1 - t0)
        return dict(k=len(self.steps) - 1, a=self.steps[-1]["a"], dt=99, d=1, done=True)

    # ---------------------------------------------------------------- drawing
    def frame(self, t):
        st = self.state(t)
        im, d = ui.new_frame()
        k, a = st["k"], st["a"]
        s = self.steps[k] if k >= 0 else None
        first = self.steps[0]
        lenses_b = s["before"] if s else first["before"]
        lenses_a = s["after"] if s else first["before"]
        cur_b = s["cur_before"] if s else 0
        bin_b = s["bin_before"] if s else 0.0
        cur_a = s["cur"] if s else 0
        bin_a = s["bin"] if s else 0.0
        scr_b = s["screen_before"] if s else first["screen_before"]
        scr_a = s["screen"] if s else first["screen_before"]
        late = st["dt"] > st["d"] * 0.45
        changed = a is not None and (a < 10 or a in (15, 16, 17, 19, 20, 21, 22, 23)) and st["dt"] > 0.12
        lenses = lenses_a if (changed or st.get("done")) else lenses_b
        cur = cur_a if (changed or st.get("done") or a == 24) else cur_b
        binm = bin_a if (changed or st.get("done") or a == 24) else bin_b
        screen = scr_a if (changed or (a == 10 and late) or st.get("done")) else scr_b

        # ---- left: what the patient sees
        badge, label = None, None
        mode = "bin" if binm else "mono"
        if a == 24:                                          # prism-split balance: two copies of the line
            imgs, A, _, which = self.view(lenses, "bin", screen)
            rows = ui.proj_chart(which)[1]
            li = int(np.argmin(np.abs(CH.LINES - screen)))
            row = [rw for rw in rows if rw[0] == li]
            yc, h = (row[0][1], row[0][2]) if row else (ui.CHART_H / 2, 40)
            half = int(min(ui.CHART_H / 2 - 20, h * 0.5 + 14))       # just this line, no neighbours
            top = imgs[0][int(max(0, yc - half)):int(yc + half)]
            bot = imgs[1][int(max(0, yc - half)):int(yc + half)]
            canvas = np.full((ui.CHART_H, ui.CHART_W), 0, np.uint8)
            canvas[:] = imgs[0][0, 0]
            canvas[ui.CHART_H // 4 - top.shape[0] // 2: ui.CHART_H // 4 - top.shape[0] // 2 + top.shape[0]] = top
            canvas[3 * ui.CHART_H // 4 - bot.shape[0] // 2: 3 * ui.CHART_H // 4 - bot.shape[0] // 2 + bot.shape[0]] = bot
            img = canvas
            label = "PRISM-SPLIT: TOP = RIGHT EYE, BOTTOM = LEFT EYE"
        elif a in (11, 12, 13, 14) and not st.get("done"):
            seg = st["d"] / 3
            lens_cur = list(lenses_b)
            e = cur_b
            S, C, ax = lens_cur[e]
            if a in (11, 12):
                ds = 0.25 if a == 11 else -0.25
                if binm:
                    lA, lB = lens_cur, [(x[0] + ds, x[1], x[2]) for x in lens_cur]
                else:
                    lA = lens_cur
                    lB = [tuple(x) for x in lens_cur]
                    lB[e] = (S + ds, C, ax)
                vA, *_ = self.view(lA, mode, screen, None if binm else e)
                vB, *_ = self.view(lB, mode, screen, None if binm else e)
                eA = eB = e
                if binm:
                    _, _, eA, _ = self.view(lA, "bin", screen)
                    _, _, eB, _ = self.view(lB, "bin", screen)
                iA, iB = vA[eA], vB[eB]
            else:
                off = (45, -45) if a == 13 else (0, 90)
                M, J0, J45 = to_pv(S, C, ax)
                out = []
                for o in off:
                    _, j0, j45 = jcc_pv(ax + o)
                    Sx, Cx, axx = __import__("optics").from_pv(M, J0 + j0, J45 + j45)
                    ll = [tuple(x) for x in lens_cur]
                    ll[e] = (float(Sx), float(Cx), float(axx))
                    v, *_ = self.view(ll, "mono", screen, e)
                    out.append(v[e])
                iA, iB = out
            vC, *_ = self.view(lens_cur, mode, screen, None if binm else e)
            _, _, eC, _ = self.view(lens_cur, mode, screen, None if binm else e)
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
            label = ("BOTH EYES" if binm else ("RIGHT EYE" if e == 0 else "LEFT EYE"))
        else:
            imgs, A, shown, which = self.view(lenses, mode, screen, None if binm else cur)
            img = imgs[shown]
            label = "BOTH EYES OPEN" if binm else ("RIGHT EYE (LEFT COVERED)" if cur == 0 else "LEFT EYE (RIGHT COVERED)")
        text(d, (CH_X, CH_Y - 42), "WHAT THE PATIENT SEES · " + label, 20, MUTED, "SemiBold")
        rr(d, [CH_X - 10, CH_Y - 10, CH_X + ui.CHART_W + 10, CH_Y + ui.CHART_H + 10], 14, fill=(30, 36, 48))
        im.paste(Image.fromarray(img), (CH_X, CH_Y))
        d = ImageDraw.Draw(im, "RGBA")
        if a != 24:
            top_i = max(0, int(np.argmin(np.abs(CH.LINES - min(screen, 1.0)))) - 1)
            for li, yc, h in ui.proj_chart(top_i)[1]:
                on = abs(CH.LINES[li] - screen) < 0.01
                text(d, (CH_X + ui.CHART_W + 24, CH_Y + yc), CH.snellen_of(CH.LINES[li]), 19, GOLD if on else MUTED,
                     "SemiBold" if on else "Medium", "lm")
                if on:
                    strong = a in (10, 18) and late
                    d.rounded_rectangle([CH_X - 4, CH_Y + yc - h / 2 - 10, CH_X + ui.CHART_W + 4, CH_Y + yc + h / 2 + 10],
                                        10, outline=GOLD + (230 if strong else 120,), width=4 if strong else 2)
        else:
            text(d, (CH_X + ui.CHART_W + 24, CH_Y + ui.CHART_H // 4), "RIGHT", 19, GOLD, "SemiBold", "lm")
            text(d, (CH_X + ui.CHART_W + 24, CH_Y + 3 * ui.CHART_H // 4), "LEFT", 19, GOLD, "SemiBold", "lm")
        if badge:
            ui.badge(d, CH_X + 46, CH_Y + 46, badge, GOLD, r=34)

        # ---- top: titles and the hidden answer
        text(d, (80, 50), self.title, 44, INK, "Bold")
        text(d, (80, 108), self.subtitle, 26, MUTED, "Medium")
        rr(d, [1500, 30, 1850, 150], 16, fill=CARD, outline=LINE, width=2)
        text(d, (1520, 44), "HIDDEN FROM THE AI: TRUE Rx", 17, MUTED, "SemiBold")
        text(d, (1520, 72), "R  " + fmt_rx(self.p["S"], self.p["C"], self.p["axis"]), 26, INK2, "SemiBold")
        text(d, (1520, 106), "L  " + fmt_rx(self.p["S_os"], self.p["C_os"], self.p["axis_os"]), 26, INK2, "SemiBold")

        # ---- right: phoropter and both readouts
        jcc = None
        if a in (13, 14) and not st.get("done"):
            seg = st["d"] / 3
            off = (45, -45) if a == 13 else (0, 90)
            axc = lenses_b[cur_b][2]
            if st["dt"] < seg:
                jcc = axc + off[0]
            elif st["dt"] < 2 * seg:
                jcc = axc + off[1]
        occl = None if binm else (1 - cur)
        PA.draw(im, 1270, 286, 0.58, [tuple(x) for x in lenses], occluded=occl, cur=cur,
                jcc=(cur_b, jcc) if jcc is not None else None, prism=(a == 24 and not st.get("done")))
        y0 = 440
        rr(d, [820, y0, 1780, y0 + 150], 18, fill=CARD, outline=LINE, width=2)
        for j, lab in enumerate(("SPHERE", "CYLINDER", "AXIS")):
            text(d, (1040 + j * 260, y0 + 14), lab, 16, MUTED, "SemiBold", "ma")
        for e in (0, 1):
            S, C, ax = lenses[e]
            yy = y0 + 42 + e * 54
            active = (binm and True) or (e == cur)
            col = INK if active else MUTED
            text(d, (850, yy), "R" if e == 0 else "L", 34, GOLD if (e == cur and not binm) else col, "Bold")
            vals = (fmt_d(S), fmt_d(C, False) if C < 0 else "0.00", f"{int(round(ax)):03d}°" if C < 0 else "—")
            for j, v in enumerate(vals):
                text(d, (1040 + j * 260, yy - 2), v, 38, col, "SemiBold", "ma")
        if s is not None and a is not None and not st.get("done"):
            text(d, (820, 622), "AI" if self.tr.get("who", "AI") == "AI" else self.tr["who"], 22, CYAN, "Bold")
            text(d, (900, 614), DOING[a], 34, INK, "SemiBold")
            note = None
            if a in (11, 12):
                note = ("2 adds " + ("+0.25" if a == 11 else "−0.25") + (" on both eyes" if bin_b else ""))
            elif a == 13:
                note = "cross-cylinder flip, straddling the axis"
            elif a == 14:
                note = "cross-cylinder flip, on the axis"
            elif a == 24:
                note = "vertical prism: two copies of the line, top = right eye"
            elif a in (16, 17):
                note = f"the {CH.snellen_of(scr_a)} line"
            elif a == 18:
                note = f"the {CH.snellen_of(scr_b)} line"
            if note:
                text(d, (900, 660), note, 21, MUTED, "Medium")
            show = (a in (10, 18) and late) or (a in (11, 12, 13, 14, 24) and st["dt"] > 2 * st["d"] / 3)
            if show:
                text(d, (820, 708), "PATIENT", 22, GOLD, "Bold")
                if a == 10:
                    ln, kk, nl = s["line"], s["k"], s["nl"]
                    ans = ("“I can’t read the top line.”" if ln > 1.05 else
                           f"“All of the {CH.snellen_of(ln)} line.”" if kk == nl else
                           f"“The {CH.snellen_of(ln)} line, but I missed {nl - kk}.”")
                elif a == 18:
                    ans = f"“I got {s['k']} of the {s['nl']}.”"
                elif a == 24:
                    ans = SAY_BAL[s["code"]]
                else:
                    ans = SAY[s["code"]]
                text(d, (950, 698), ans, 34, GOLD, "SemiBold")
        # move strip (most recent moves)
        nq = sum(1 for x in self.steps[:k + 1] if x["a"] in QUESTION) if k >= 0 else 0
        text(d, (820, 778), f"MOVE {max(k + 1, 0)}" + (f"  ·  {nq} QUESTION{'S' if nq != 1 else ''}" if nq else ""), 20,
             MUTED, "SemiBold")
        lo = max(0, k + 1 - self.chips_max)
        x, y = 820, 812
        f = ui.font(16, "SemiBold")
        if lo > 0:
            text(d, (x, y + 15), "…", 20, MUTED, "SemiBold", "lm")
            x += 26
        for i_, sx in enumerate(self.steps[lo:k + 1]):
            lab, col = CHIP[sx["a"]]
            w = d.textlength(lab, font=f) + 16
            if x + w > 1790:
                x, y = 820, y + 34
            al = 255 if lo + i_ == k else 150
            rr(d, [x, y, x + w, y + 28], 7, fill=tuple(c // 4 for c in col) + (al,), outline=col + (al,), width=2)
            d.text((x + w / 2, y + 14), lab, font=f, fill=col + (al,), anchor="mm")
            x += w + 6
        if k in self.notes and a is not None and not st.get("done"):
            al = fade(st["dt"], 0.1, st["d"], 0.25, 0.3)
            ui.ST.caption(d, [(self.notes[k], 28, INK, "SemiBold")], al, W=W, H=H, anchor="bottom", cx=1300)

        # ---- result card
        if t >= self.t_end_steps:
            al = smooth((t - self.t_end_steps) / 0.4)
            last = self.steps[-1]
            rr(d, [800, 596, 1800, 1040], 22, fill=BG + (int(255 * al),))
            rr(d, [820, 604, 1780, 1016], 22, fill=PANEL + (int(255 * al),), outline=LINE + (int(255 * al),), width=3)
            r2 = lambda v: math.floor(abs(v) * 100 + 0.5) / 100
            for e in (0, 1):
                S, C, ax = last["after"][e]
                L = to_pv(S, C, ax)
                E = self.E[e]
                dse = L[0] - E[0]
                cres = 2 * math.hypot(L[1] - E[1], L[2] - E[2])
                err = rx_error(L, E)
                col = GREEN if err < 0.13 else (GOLD if err <= 0.25 + 1e-9 else RED)
                y = 636 + e * 180
                tr_ = (self.p["S"], self.p["C"], self.p["axis"]) if e == 0 else (self.p["S_os"], self.p["C_os"], self.p["axis_os"])
                text(d, (850, y), "RIGHT EYE" if e == 0 else "LEFT EYE", 20, MUTED, "SemiBold", a=al)
                text(d, (850, y + 28), fmt_rx(S, C, ax), 36, INK, "SemiBold", a=al)
                text(d, (850, y + 76), "true " + fmt_rx(*tr_), 24, INK2, "Medium", a=al)
                se_txt = "SE spot on" if abs(dse) < 0.07 else f"SE {r2(dse):.2f} D too much {'minus' if dse < 0 else 'plus'}"
                cy_txt = "no leftover cyl" if cres < 0.07 else f"leftover cyl {r2(cres):.2f} D"
                text(d, (850, y + 112), se_txt + "  ·  " + cy_txt, 20, MUTED, "Medium", a=al)
                verdict = "Exact" if err < 0.13 else ("Within 0.25 D" if err <= 0.25 + 1e-9 else f"Off by {r2(err):.2f} D")
                text(d, (1750, y + 30), verdict, 38, col, "Bold", "ra", al)
            text(d, (1750, 990), f"{len(self.steps)} moves", 24, MUTED, "SemiBold", "rd", al)
        return np.asarray(im)


if __name__ == "__main__":
    import os, sys
    from record import record2
    from clinicians2 import mine
    tr = record2(script=mine)
    tr["who"] = "YOU"
    ex = Exam2(tr, "TEST", "the optometrist's routine", speed=0.5)
    ks = {s["a"]: k for k, s in enumerate(tr["steps"])}
    for nm, a in (("bal", 24), ("jcc", 14), ("cmp", 13), ("read", 10)):
        if a in ks:
            t0, t1, _ = ex.beats[ks[a]]
            Image.fromarray(ex.frame(t0 + (t1 - t0) * 0.5)).save(os.path.join(ui.OUT, f"test2_{nm}.png"))
    Image.fromarray(ex.frame(ex.dur - 0.5)).save(os.path.join(ui.OUT, "test2_end.png"))
    print("dur", ex.dur)
