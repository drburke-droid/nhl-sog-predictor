"""Drawing for the clinic video: the real phoropter (colour-coded to the AI's buttons), the buttons panel, the
patient's answer box (with live odds), and the clinic scorecard.

Colour code (the same everywhere: phoropter parts, buttons, move chips):
    sphere blue, cylinder green, axis violet, cross-cylinder red, prism (balance) orange, occluder grey,
    chart white, prescribe gold.
"""
import os, math
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import ui
from ui import W, H, BG, INK, INK2, MUTED, CARD, LINE, text, rr, fmt_d
import env3 as E3

HERE = ui.HERE
SPHERE, CYL, AXIS, JCC, PRISM, OCC, CHART, RXC = ((70, 150, 255), (60, 200, 120), (176, 124, 255), (255, 88, 88),
                                                  (255, 150, 40), (196, 206, 222), (236, 240, 248), (255, 205, 80))
GROUP_COL = {"sphere": SPHERE, "cyl": CYL, "axis": AXIS, "jcc": JCC, "prism": PRISM, "occ": OCC, "chart": CHART,
             "rx": RXC}
GROUP_NAME = {"sphere": "SPHERE: lens power", "cyl": "CYLINDER: astigmatism power", "axis": "AXIS: turn the cylinder",
              "jcc": "CROSS-CYLINDER: fine-tune the cylinder", "prism": "PRISM: compare the two eyes",
              "occ": "OCCLUDER: which eye", "chart": "PROJECTOR CHART", "rx": "FINISH"}

# every button: (index, chip line 1, chip line 2, panel label, what it does, group)
BUTTONS = [
    (E3.SP25, "+0.25", "sphere", "Sphere +0.25", "a quarter step more plus", "sphere"),
    (E3.SM25, "−0.25", "sphere", "Sphere −0.25", "a quarter step more minus", "sphere"),
    (E3.SP100, "+1.00", "sphere", "Sphere +1.00", "a big jump more plus", "sphere"),
    (E3.SM100, "−1.00", "sphere", "Sphere −1.00", "a big jump more minus", "sphere"),
    (E3.CMP_P, "1 or 2?", "with +0.25?", "Ask: “1 or 2?” (+0.25)", "is it clearer with +0.25 (2) or without (1)?",
     "sphere"),
    (E3.CMP_M, "1 or 2?", "with −0.25?", "Ask: “1 or 2?” (−0.25)", "is it clearer with −0.25 (2) or without (1)?",
     "sphere"),
    (E3.OU_P, "+0.25", "both eyes", "Both eyes +0.25", "more plus on both eyes together", "sphere"),
    (E3.OU_M, "−0.25", "both eyes", "Both eyes −0.25", "more minus on both eyes together", "sphere"),
    (E3.R_P, "+0.25", "right eye", "Right eye +0.25", "more plus on the right eye only", "sphere"),
    (E3.L_P, "+0.25", "left eye", "Left eye +0.25", "more plus on the left eye only", "sphere"),
    (E3.CYL_MORE, "−0.25", "cylinder", "Cylinder −0.25", "more astigmatism correction", "cyl"),
    (E3.CYL_LESS, "+0.25", "cylinder", "Cylinder +0.25", "less astigmatism correction", "cyl"),
    (E3.AXP2, "+2°", "axis", "Axis +2°", "turn the cylinder a touch", "axis"),
    (E3.AXM2, "−2°", "axis", "Axis −2°", "", "axis"),
    (E3.AXP5, "+5°", "axis", "Axis +5°", "turn it a bit", "axis"),
    (E3.AXM5, "−5°", "axis", "Axis −5°", "", "axis"),
    (E3.AXP15, "+15°", "axis", "Axis +15°", "turn it a lot", "axis"),
    (E3.AXM15, "−15°", "axis", "Axis −15°", "", "axis"),
    (E3.JCC_AX, "1 or 2?", "turn which way?", "Cross-cyl: axis", "flip the cross-cylinder: which way should the axis turn?",
     "jcc"),
    (E3.JCC_PW, "1 or 2?", "more or less cyl?", "Cross-cyl: power",
     "flip the cross-cylinder: more or less cylinder?", "jcc"),
    (E3.BALANCE, "Top or", "bottom clearer?", "Balance: top or bottom?",
     "prism splits the line in two: right eye on top, left eye below", "prism"),
    (E3.NEXT, "Next eye", "R → L → both", "Next eye", "cover one eye: right, then left, then both open", "occ"),
    (E3.READ, "Read the", "smallest line", "Ask: “Read the smallest line”", "how far down the chart can they read?",
     "chart"),
    (E3.READ_LINE, "Read", "this line", "Ask: “Read this line”", "how many letters can they get on this line?", "chart"),
    (E3.LINE_SM, "Smaller", "letters", "Smaller line", "show smaller letters", "chart"),
    (E3.LINE_LG, "Bigger", "letters", "Bigger line", "show bigger letters", "chart"),
    (E3.RX, "Prescribe", "done", "Prescribe", "the exam is over: these are the glasses", "rx"),
]
BTN = {b[0]: b for b in BUTTONS}
assert len(BUTTONS) == E3.N_ACT3


def col_of(a):
    return GROUP_COL[BTN[a][5]]


# ---------------------------------------------------------------- the phoropter photo
SRC = os.path.join(HERE, "assets", "phoropter.webp")
SRC_W, SRC_H = 1457, 1080
# part centres and radii in the photo (examiner's view: the patient's right eye is on the left)
PARTS = {
    "occ": [(192, 478, 112), (1297, 478, 112)],
    "sphere": [(196, 707, 46), (1293, 707, 46)],
    "sphere_knob": [(187, 940, 52), (1305, 940, 52)],
    "axis": [(355, 750, 108), (1157, 750, 108)],
    "aperture": [(355, 750, 52), (1157, 750, 52)],
    "jcc": [(455, 540, 92), (1060, 540, 92)],
    "prism": [(607, 700, 100), (908, 700, 100)],
    "cyl": [(485, 925, 108), (1022, 925, 108)],
}
PART_GROUP = {"occ": "occ", "sphere": "sphere", "sphere_knob": "sphere", "axis": "axis", "jcc": "jcc",
              "prism": "prism", "cyl": "cyl"}


@lru_cache(8)
def photo(scale):
    im = Image.open(SRC).convert("RGBA")
    return im.resize((int(SRC_W * scale), int(SRC_H * scale)), Image.LANCZOS)


def _ring(d, x, y, r, col, w, a):
    d.ellipse([x - r, y - r, x + r, y + r], outline=tuple(col) + (int(255 * a),), width=max(1, int(w)))


def draw_phoropter(im, x0, y0, scale, state=None, hot=None, a=1.0, rings=True, labels=False, legend_only=None):
    """Paste the photo at (x0, y0) and colour-code its parts.

    state: dict with S, C, ax (two eyes each), phase (0 right, 1 left, 2 both), jcc (None or 1/2 shown),
           prism (bool).  hot: (group, eye or None) to glow.  legend_only: draw rings only for these groups."""
    ph = photo(scale)
    if a < 1:
        ph = ph.copy()
        ph.putalpha(ph.getchannel("A").point(lambda v: int(v * a)))
    im.paste(ph, (int(x0), int(y0)), ph)
    d = ImageDraw.Draw(im, "RGBA")
    P = lambda cx, cy: (x0 + cx * scale, y0 + cy * scale)
    phase = state.get("phase", 0) if state else 0
    # occluder: cover the eye not being tested
    if state is not None and phase < 2:
        cov = 1 - phase
        cx, cy, r = PARTS["aperture"][cov]
        x, y = P(cx, cy)
        rr_ = r * scale
        d.ellipse([x - rr_, y - rr_, x + rr_, y + rr_], fill=(4, 5, 8, int(250 * a)))
        text(d, (x, y), "COVERED", max(10, int(15 * scale / 0.6)), MUTED, "Bold", "mm", a)
    if rings:
        for part, lst in PARTS.items():
            g = PART_GROUP.get(part)
            if g is None or (legend_only is not None and g not in legend_only):
                continue
            for e, (cx, cy, r) in enumerate(lst):
                x, y = P(cx, cy)
                is_hot = hot is not None and hot[0] == g and (hot[1] is None or hot[1] == e)
                w = (7 if is_hot else 3.2) * scale / 0.6
                aa = a * (1.0 if is_hot else 0.85)
                if is_hot:
                    for k in range(4):
                        _ring(d, x, y, r * scale + (4 + 4 * k) * scale / 0.6, GROUP_COL[g], 3, a * (0.35 - 0.08 * k))
                _ring(d, x, y, r * scale + 3 * scale / 0.6, GROUP_COL[g], w, aa)
    if state is not None:
        for e in (0, 1):
            S, C, ax = state["S"][e], state["C"][e], state["ax"][e]
            # sphere readout over the window
            cx, cy, r = PARTS["sphere"][e]
            x, y = P(cx, cy)
            bw, bh = 116 * scale / 0.6, 44 * scale / 0.6
            rr(d, [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], int(8 * scale / 0.6), fill=(6, 10, 20, int(235 * a)),
               outline=SPHERE + (int(255 * a),), width=2)
            text(d, (x, y + 1), fmt_d(S, True).replace("Plano", "0.00"), int(24 * scale / 0.6), INK, "Bold", "mm", a)
            # cylinder readout on the cylinder knob
            cx, cy, r = PARTS["cyl"][e]
            x, y = P(cx, cy)
            rr(d, [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], int(8 * scale / 0.6), fill=(6, 10, 20, int(235 * a)),
               outline=CYL + (int(255 * a),), width=2)
            text(d, (x, y + 1), "0.00" if C > -0.01 else fmt_d(C), int(24 * scale / 0.6), INK, "Bold", "mm", a)
            # axis: a line across the aperture, and the number on the ring
            cx, cy, r = PARTS["axis"][e]
            x, y = P(cx, cy)
            if C < -0.01 and not (phase < 2 and e == 1 - phase):
                t = math.radians(ax)
                rx_ = PARTS["aperture"][e][2] * scale * 0.95
                d.line([(x - math.cos(t) * rx_, y + math.sin(t) * rx_), (x + math.cos(t) * rx_, y - math.sin(t) * rx_)],
                       fill=AXIS + (int(255 * a),), width=max(2, int(5 * scale / 0.6)))
            t = math.radians(225 if e == 0 else -45)
            tx, ty = x + math.cos(t) * (r + 30) * scale, y - math.sin(t) * (r + 30) * scale
            rr(d, [tx - 40 * scale / 0.6, ty - 18 * scale / 0.6, tx + 40 * scale / 0.6, ty + 18 * scale / 0.6],
               int(6 * scale / 0.6), fill=(6, 10, 20, int(235 * a)), outline=AXIS + (int(255 * a),), width=2)
            text(d, (tx, ty + 1), f"{int(round(ax)) % 180 or 180}°", int(20 * scale / 0.6), INK, "Bold", "mm", a)
        if state.get("jcc"):                               # cross-cylinder flipped in front of the tested eye
            e = min(phase, 1)
            cx, cy, r = PARTS["aperture"][e]
            x, y = P(cx, cy)
            rr_ = r * scale
            d.ellipse([x - rr_, y - rr_, x + rr_, y + rr_], outline=JCC + (int(255 * a),), width=max(3, int(6 * scale / 0.6)))
            text(d, (x, y), f"“{state['jcc']}”", int(30 * scale / 0.6), JCC, "Bold", "mm", a)
        if state.get("prism"):
            for e in (0, 1):
                cx, cy, r = PARTS["prism"][e]
                x, y = P(cx, cy)
                text(d, (x, y + r * scale + 18 * scale / 0.6), "3Δ " + ("UP" if e == 0 else "DOWN"),
                     int(18 * scale / 0.6), PRISM, "Bold", "mm", a)
    if labels:
        lab = [("occ", 0, "OCCLUDER", (-1, -1)), ("sphere", 0, "SPHERE", (-1, 0)), ("axis", 0, "AXIS", (-1, 1)),
               ("jcc", 0, "CROSS-CYL", (0, -1)), ("prism", 0, "PRISM", (1, -1)), ("cyl", 0, "CYLINDER", (0, 1))]
        for part, e, s, _ in lab:
            cx, cy, r = PARTS[part][e]
            x, y = P(cx, cy)
            g = PART_GROUP[part]
            if legend_only is not None and g not in legend_only:
                continue
            fs = int(22 * scale / 0.6)
            tw = d.textlength(s, font=ui.font(fs, "Bold"))
            ly = y - r * scale - 30 * scale / 0.6 if part != "cyl" else y + r * scale + 26 * scale / 0.6
            rr(d, [x - tw / 2 - 10, ly - fs * 0.75, x + tw / 2 + 10, ly + fs * 0.75], 8,
               fill=tuple(GROUP_COL[g]) + (int(235 * a),))
            text(d, (x, ly + 1), s, fs, BG, "Bold", "mm", a)
    return d


# ---------------------------------------------------------------- the buttons panel
PANEL_ROWS = [
    ["sphere"],
    ["cyl", "axis"],
    ["jcc", "prism", "occ"],
    ["chart", "rx"],
]


def draw_buttons(d, x0, y0, w, h, pressed=None, probs=None, mask=None, a=1.0, describe=False, title=True,
                 heading="THE AI'S BUTTONS"):
    """Every button the AI has, grouped and colour-coded. pressed: the button just pressed (lit).
    probs: the AI's chances for each button (drawn as a fill bar). mask: which buttons work right now."""
    if title:
        text(d, (x0, y0), heading, 20, MUTED, "Bold", "la", a)
        if probs is not None:
            text(d, (x0 + w, y0), "bar = how likely the AI is to press it next", 17, MUTED, "Medium", "ra", a)
        y0 += 30
        h -= 30
    groups = {}
    for b in BUTTONS:
        groups.setdefault(b[5], []).append(b)
    rows = PANEL_ROWS
    rh = h / len(rows)
    for ri, row in enumerate(rows):
        y = y0 + ri * rh
        n = sum(len(groups[g]) for g in row)
        gap_g = 22
        bw = (w - gap_g * (len(row) - 1)) / max(n, 1)
        bw = min(bw, 190 if not describe else 300)
        x = x0
        for g in row:
            col = GROUP_COL[g]
            for b in groups[g]:
                idx, chip, sub, lab, desc, _ = b
                on = mask is None or mask[idx]
                lit = pressed == idx
                bx0, by0, bx1, by1 = x + 3, y + 3, x + bw - 3, y + rh - 4
                fill = tuple(col) + (int((235 if lit else 40 if on else 14) * a),)
                rr(d, [bx0, by0, bx1, by1], 9, fill=fill, outline=tuple(col) + (int((255 if on else 70) * a),),
                   width=3 if lit else 2)
                tc = BG if lit else (INK if on else MUTED)
                if describe:
                    text(d, (bx0 + 12, by0 + 10), lab, 21, tc, "Bold", "la", a * (1 if on else 0.5))
                    if desc:
                        ui_wrap(d, bx0 + 12, by0 + 38, desc, 16, bx1 - bx0 - 22, BG if lit else INK2, a)
                else:
                    cy_ = by0 + (by1 - by0) * (0.40 if probs is not None else 0.47)
                    aa = a * (1 if on else 0.45)
                    text(d, ((bx0 + bx1) / 2, cy_ - 10), chip, 19 if bw > 110 else 16, tc, "Bold", "mm", aa)
                    text(d, ((bx0 + bx1) / 2, cy_ + 12), sub, 14 if bw > 110 else 12, BG if lit else INK2, "Medium", "mm",
                         aa)
                if probs is not None and on:
                    p = float(probs[idx])
                    bxL, bxR = bx0 + 8, bx1 - 8
                    yb = by1 - 10
                    d.rectangle([bxL, yb - 3, bxR, yb + 3], fill=(255, 255, 255, int(30 * a)))
                    if p > 0.002:
                        d.rectangle([bxL, yb - 3, bxL + (bxR - bxL) * min(1, p), yb + 3],
                                    fill=(BG if lit else col) + (int(255 * a),))
                x += bw
            x += gap_g


def ui_wrap(d, x, y, s, size, width, col, a=1.0, lh=1.25, weight="Medium"):
    f = ui.font(size, weight)
    words, line, yy = s.split(), "", y
    for w_ in words:
        t = (line + " " + w_).strip()
        if d.textlength(t, font=f) > width and line:
            text(d, (x, yy), line, size, col, weight, "la", a)
            yy += size * lh
            line = w_
        else:
            line = t
    if line:
        text(d, (x, yy), line, size, col, weight, "la", a)
        yy += size * lh
    return yy


# ---------------------------------------------------------------- the patient's answer box
def answer_options(a):
    if a in (E3.CMP_P, E3.CMP_M, E3.JCC_AX, E3.JCC_PW):
        return "compare", list(E3.ANSWERS["compare"])
    if a == E3.BALANCE:
        return "balance", list(E3.ANSWERS["balance"])
    if a == E3.READ_LINE:
        return "line", list(E3.ANSWERS["line"])
    return None, []


def draw_answers(d, x0, y0, w, h, a_now=None, probs=None, chosen=None, right=None, read=None, reveal=1.0, alpha=1.0,
                 header="WHAT THE PATIENT CAN SAY"):
    """The answer box. a_now: the question being asked. probs: chance of each answer given the lenses now and
    the patient's true prescription. chosen: the answer given (index). right: the answer that points to the true
    prescription. read: for 'read the smallest line', a dict line -> chance."""
    rr(d, [x0, y0, x0 + w, y0 + h], 16, fill=CARD + (int(240 * alpha),), outline=LINE + (int(255 * alpha),), width=2)
    text(d, (x0 + 20, y0 + 16), header, 18, MUTED, "Bold", "la", alpha)
    y = y0 + 52
    if read is not None:
        text(d, (x0 + 20, y), "“Read the smallest line you can.”", 22, INK, "SemiBold", "la", alpha)
        y += 40
        items = sorted(read.items(), key=lambda kv: -kv[1])[:4]
        items = sorted(items, key=lambda kv: -kv[0])
        for line, p in items:
            lab = "I can't read any of it" if line > 1.05 else f"“I can read down to {CH_snellen(line)}”"
            is_c = chosen is not None and abs(line - chosen) < 1e-6
            _bar(d, x0 + 20, y, w - 40, lab, p * reveal, INK, is_c, None, alpha)
            y += 50
        return
    kind, opts = answer_options(a_now) if a_now is not None else (None, [])
    if not opts:
        for k, (kk, oo) in enumerate((("compare", E3.ANSWERS["compare"]), ("line", E3.ANSWERS["line"][:3]))):
            for o in oo:
                text(d, (x0 + 24, y), "·  " + o, 20, INK2, "Medium", "la", alpha)
                y += 32
            y += 6
        text(d, (x0 + 24, y), "·  “I can read down to 20/…”", 20, INK2, "Medium", "la", alpha)
        y += 32
        text(d, (x0 + 24, y), "·  Top is clearer / Bottom / Equal", 20, INK2, "Medium", "la", alpha)
        return
    q = {E3.CMP_P: "“Which is clearer, 1 or 2?”  (2 = +0.25)", E3.CMP_M: "“Which is clearer, 1 or 2?”  (2 = −0.25)",
         E3.JCC_AX: "“1 or 2?”  (cross-cylinder, axis)", E3.JCC_PW: "“1 or 2?”  (cross-cylinder, power)",
         E3.BALANCE: "“Top or bottom: which is clearer?”", E3.READ_LINE: "“Read this line.”"}[a_now]
    text(d, (x0 + 20, y), q, 22, INK, "SemiBold", "la", alpha)
    y += 40
    for k, o in enumerate(opts):
        p = 0.0 if probs is None else float(probs[k])
        _bar(d, x0 + 20, y, w - 40, o, p * reveal, col_of(a_now), chosen == k, (right == k) if right is not None else None,
             alpha)
        y += 50
    if right is not None:
        text(d, (x0 + 20, y + 2), "✓ = the answer that points to their true prescription", 16, MUTED, "Medium", "la", alpha)


def _bar(d, x, y, w, lab, p, col, chosen, right, alpha):
    rr(d, [x, y, x + w, y + 40], 8, fill=(255, 255, 255, int((36 if chosen else 12) * alpha)),
       outline=tuple(col) + (int((255 if chosen else 0) * alpha),), width=3 if chosen else 1)
    bw = (w - 90) * min(max(p, 0), 1)
    if bw > 1:
        rr(d, [x + 2, y + 2, x + 2 + bw, y + 38], 7, fill=tuple(col) + (int(95 * alpha),))
    text(d, (x + 14, y + 20), lab, 19, INK, "SemiBold" if chosen else "Medium", "lm", alpha)
    text(d, (x + w - 12, y + 20), f"{100 * p:.0f}%", 19, INK, "Bold", "rm", alpha)
    if right:
        text(d, (x + w - 70, y + 20), "✓", 22, (110, 220, 140), "Bold", "rm", alpha)


def CH_snellen(L):
    import chart
    return chart.snellen_of(L)


# ---------------------------------------------------------------- the clinic scorecard
OUT_COL = {"happy": (110, 220, 140), "annoyed": (255, 205, 80), "recheck": (255, 150, 40), "review": (255, 88, 88)}


def draw_scorecard(d, x0, y0, w, h, minutes, probs=None, outcome=None, points=None, alpha=1.0, title="CLINIC SCORE"):
    import clinic
    rr(d, [x0, y0, x0 + w, y0 + h], 16, fill=CARD + (int(240 * alpha),), outline=LINE + (int(255 * alpha),), width=2)
    text(d, (x0 + 20, y0 + 16), title, 18, MUTED, "Bold", "la", alpha)
    y = y0 + 50
    text(d, (x0 + 20, y), "Chair time", 20, INK2, "Medium", "la", alpha)
    m, s = divmod(int(round(minutes * 60)), 60)
    text(d, (x0 + w - 20, y), f"{m}:{s:02d}   {-clinic.POINTS_PER_MIN * minutes:+.0f} pts", 20, INK, "Bold", "ra", alpha)
    y += 38
    for k, o in enumerate(clinic.OUTCOMES):
        p = None if probs is None else float(probs[k])
        lab = clinic.LABELS[o]
        is_o = outcome == k
        rr(d, [x0 + 16, y - 4, x0 + w - 16, y + 30], 7, fill=tuple(OUT_COL[o]) + (int((200 if is_o else 0) * alpha),))
        text(d, (x0 + 24, y + 13), lab, 18, BG if is_o else INK2, "SemiBold", "lm", alpha)
        val = f"{clinic.POINTS[k]:+.0f}"
        if p is not None:
            val = f"{100 * p:.0f}%   " + val
        text(d, (x0 + w - 24, y + 13), val, 18, BG if is_o else INK, "Bold", "rm", alpha)
        y += 38
    if points is not None:
        text(d, (x0 + 20, y + 6), "This patient", 20, INK2, "Medium", "la", alpha)
        text(d, (x0 + w - 20, y + 2), f"{points:+.0f} pts", 28, INK, "Bold", "ra", alpha)


if __name__ == "__main__":
    os.makedirs(ui.OUT, exist_ok=True)
    im, d = ui.new_frame()
    st = dict(S=[-2.25, -1.75], C=[-1.0, -0.5], ax=[170, 15], phase=0, jcc=1)
    draw_phoropter(im, 20, 60, 0.6, st, hot=("jcc", 0))
    d = ImageDraw.Draw(im, "RGBA")
    probs = np.full(E3.N_ACT3, 0.0)
    probs[E3.JCC_AX], probs[E3.SM25], probs[E3.READ] = 0.6, 0.25, 0.1
    draw_buttons(d, 20, 740, 1880, 320, pressed=E3.JCC_AX, probs=probs, mask=E3.MASK[0])
    draw_answers(d, 1330, 60, 570, 330, E3.JCC_AX, probs=[0.62, 0.21, 0.17], chosen=0, right=0)
    draw_scorecard(d, 1330, 410, 570, 300, 3.4, probs=[0.75, 0.2, 0.05, 0.0])
    im.save(os.path.join(ui.OUT, "test_clinic_ui.png"))
    im, d = ui.new_frame()
    draw_phoropter(im, 260, 40, 0.75, None, labels=True)
    d = ImageDraw.Draw(im, "RGBA")
    im.save(os.path.join(ui.OUT, "test_clinic_labels.png"))
    im, d = ui.new_frame()
    draw_buttons(d, 40, 40, 1840, 1000, describe=True, title=False)
    im.save(os.path.join(ui.OUT, "test_clinic_buttons.png"))
    print("ok")
