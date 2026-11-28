"""The standalone eye-exam film: an optometrist hires an AI to do the refractions. The textbook sequence
(based on Borish's Clinical Refraction) as numbered steps and how well it works (published data); patients who
aren't perfect; the AI learning from its mistakes, written in the same step numbers; its finished exam next to
the textbook's on the same patient.

    python film7.py list | render <name> | frame <name> <t> | all | assemble     -> out/ai_eye_exam.mp4

Data: data/montage7.json (montage7.py), data/results_m.json (evaluate6.py), data/results_h.json (evaluate5.py),
traces h_day1 (the AI's first practice patient), m_ai, m_borish (traces6.py).
"""
import os, sys, json, math, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import film6 as F6                                   # (installs the version-5 buttons and the env6 axis button)
from film6 import (Exam5, Say, F3, CU, E5, ui, clinic, mmss, fmt_n, AI_COL, TB_COL, OUT, XF, Hook6, Wants5, Learn6,
                   fade_frame, trace)
from film4_intro import PatientSays4, DEMO
from ui import W, H, BG, INK, INK2, MUTED, GOLD, GREEN, RED, CARD, LINE, PANEL, text, rr, fade, smooth
from scenes import Section, wrap, big_title, para
from record import fixed_pair
from env6 import ClinicEnv6
from steps7 import LEGEND, STEP, STEP_COL, collapse

DATA = os.path.join(HERE, "data")
PREFIX = "n_"
TOTAL_PRACTICE = 1812949 + 585322
OUT_COL = dict(CU.OUT_COL, walked=(150, 70, 80))
OUT_TEXT = {"happy": "Happy: sees clearly, comes back", "annoyed": "Annoyed: “not quite crisp”",
            "recheck": "Back for a re-check", "review": "Back for a re-check, and a bad review",
            "walked": "Walked out after 15 minutes, no glasses"}
OUT_SHORT = [("happy", "Happy"), ("annoyed", "Annoyed"), ("recheck", "Re-check"), ("review", "Bad review"),
             ("walked", "Walked out")]
TEXTBOOK_SEQ = [("RIGHT EYE", "1 2 4 5 6 2 3".split()), ("LEFT EYE", "1 2 4 5 6 2 3".split()),
                ("BOTH EYES", "1 2 7 2 3 8".split())]


# ======================================================================== small drawing helpers
def chip(d, x, y, lab, a=1.0, s=34, ring=None, dim=1.0):
    col = STEP_COL[lab] if lab in STEP_COL else (120, 120, 130)
    col = tuple(int(c * dim + 30 * (1 - dim)) for c in col)
    d.rounded_rectangle([x, y, x + s, y + s], int(s * 0.22), fill=col + (int(255 * a),))
    text(d, (x + s / 2, y + s / 2 + 1), lab, int(s * 0.62), BG, "Bold", "mm", a)
    if ring is not None:
        d.rounded_rectangle([x - 4, y - 4, x + s + 4, y + s + 4], int(s * 0.3), outline=tuple(ring) + (int(255 * a),),
                            width=3)


def legend(d, x, y, a, rh=44, s=32, size=22, hi=None):
    for i, (lab, txt) in enumerate(LEGEND):
        al = a if hi is None or lab in hi else a * 0.35
        chip(d, x, y + i * rh, lab, al, s)
        text(d, (x + s + 16, y + i * rh + s / 2), txt, size, INK, "SemiBold", "lm", al)


def outcome_of_err(err, mins, walked):
    """The patient's most likely reaction to the glasses, and the score as the exam screen shows it: the points
    these glasses earn on average (or the walkout), minus 4 a minute in the chair."""
    if walked:
        return "walked", clinic.WALKOUT - clinic.POINTS_PER_MIN * mins
    e = clinic.patient_error(np.array([err[0]]), np.array([err[1]]))
    k = int(np.argmax(clinic.outcome_probs(e)[0]))
    return (("happy", "annoyed", "recheck", "review")[k],
            float(clinic.expected_points(e)[0]) - clinic.POINTS_PER_MIN * mins)


def outcome_of(tr):
    last = tr["steps"][-1]
    return outcome_of_err(last.get("err", [9, 9]), last["secs"] / 60, last.get("walked", False))


def stage_groups(tr):
    """A recorded exam as (stage label, step numbers) groups, one per occluder setting (pass-throughs dropped)."""
    groups = []
    for st in tr["steps"]:
        ph = st["phase"]
        lab = ("RIGHT EYE", "LEFT EYE", "BOTH EYES")[ph]
        if not groups or groups[-1][0] != ph:
            groups.append([ph, lab, []])
        groups[-1][2].append(st["a"])
    out = []
    for ph, lab, acts in groups:
        seq = collapse(acts)
        if seq != ["1"]:
            out.append((lab, seq))
    return out


# ======================================================================== the opening
class Hook7(Hook6):
    dur = 15.0
    LINES = [(4.2, 170, "“Which is clearer: 1 or 2?”", 66, INK, "Bold"),
             (6.6, 300, "Behind that question is a precise sequence", 44, INK2, "SemiBold"),
             (7.4, 362, "that optometrists learn in school.", 44, INK2, "SemiBold"),
             (10.2, 500, "Could an AI work out a sequence of its own?", 56, GOLD, "Bold")]


class Phoropter7(Section):
    dur = 28.0
    ITEMS = [("sphere", "Lens power", "Focuses the eye: minus for short sight, plus for long sight."),
             ("cyl", "Astigmatism strength", "For an eye that focuses more in one direction than another."),
             ("axis", "Astigmatism angle", "Which direction, from 1° to 180°."),
             ("jcc", "Cross-cylinder", "Flips between two lenses: “Which is clearer, 1 or 2?”")]
    T0, STEP = 3.0, 4.5

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        k = int((t - self.T0) // self.STEP) if t >= self.T0 else -1
        g = self.ITEMS[k][0] if 0 <= k < len(self.ITEMS) else None
        CU.draw_phoropter(im, 30, 170, 0.6, None, hot=(g, None) if g else None, rings=True,
                          legend_only=[g] if g else ["none"])
        d = ImageDraw.Draw(im, "RGBA")
        big_title(d, W / 2, 40, "The phoropter", 54, fade(t, 0.3, 999, 0.5, 0.1) * a_all)
        y = 220
        for i, (grp, head, sub) in enumerate(self.ITEMS):
            al = fade(t, self.T0 + self.STEP * i, 999, 0.5, 0.1) * a_all
            col = CU.GROUP_COL[grp]
            rr(d, [1010, y, 1022, y + 96], 4, fill=tuple(col) + (int(255 * al),))
            text(d, (1046, y + 4), head, 40, INK, "Bold", a=al)
            para(d, 1046, y + 58, sub, 28, 820, INK2, "Medium", al, lh=1.25)
            y += 150
        al = fade(t, self.T0 + self.STEP * len(self.ITEMS), 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 930, "The patient looks at a chart through it. Their answers are all the examiner has.",
                  36, al, GOLD, "SemiBold")
        return fade_frame(im, a_all)


class TextbookSteps7(Section):
    """The textbook sequence as numbered steps: the legend used for the rest of the film."""
    dur = 42.0

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 34, "The textbook sequence", 54, fade(t, 0.3, 999, 0.5, 0.1) * a)
        al = fade(t, 1.2, 999, 0.6, 0.1) * a
        para(d, 160, 118, "In 1949 Dr. Irvin Borish published Clinical Refraction, written from his teaching notes "
                          "and his years in practice. Optometry schools still teach a sequence built on it. "
                          "We'll call it the textbook.", 30, 1600, INK2, "Medium", al, lh=1.3)
        # the legend, one step at a time
        for i, (lab, txt) in enumerate(LEGEND):
            ai = fade(t, 7.0 + 1.1 * i, 999, 0.4, 0.1) * a
            chip(d, 120, 300 + i * 62, lab, ai, 44)
            text(d, (184, 300 + i * 62 + 22), txt, 28, INK, "SemiBold", "lm", ai)
        # the textbook's order
        x0, y0 = 1080, 330
        al = fade(t, 18.5, 999, 0.6, 0.1) * a
        text(d, (x0, y0 - 40), "THE TEXTBOOK'S ORDER", 22, MUTED, "Bold", a=al)
        y = y0
        n = 0
        for j, (lab, seq) in enumerate(TEXTBOOK_SEQ):
            aj = fade(t, 19.5 + 3.0 * j, 999, 0.5, 0.1) * a
            text(d, (x0, y), lab, 24, GOLD if j < 2 else (120, 200, 255), "Bold", a=aj)
            for k, lab2 in enumerate(seq):
                ak = fade(t, 19.8 + 3.0 * j + 0.25 * k, 999, 0.3, 0.1) * a
                chip(d, x0 + k * 62, y + 40, lab2, ak, 50)
            y += 140
        al = fade(t, 30.0, 999, 0.6, 0.1) * a
        para(d, x0, y + 6, "Each eye on its own, the other covered; then both eyes together. Between steps, "
                           "“Which is clearer, 1 or 2?”, again and again.", 28, 780, GOLD, "SemiBold", al, lh=1.3)
        return np.asarray(im)


class TextbookWorks7(Section):
    """How well the textbook works: published data only."""
    dur = 22.0
    STATS = [("81%", "overall lens power", "within ¼ dioptre"), ("97%", "overall lens power", "within ½ dioptre"),
             ("93%", "astigmatism strength", "within ¼ dioptre")]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 40, "How well does it work?", 54, fade(t, 0.3, 999, 0.5, 0.1) * a)
        al = fade(t, 1.2, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 130, "Optometrists' prescriptions compared with expert benchmark prescriptions, real patients:",
                  30, al, INK2, "Medium")
        for i, (big, l1, l2) in enumerate(self.STATS):
            ai = fade(t, 2.5 + 1.8 * i, 999, 0.5, 0.1) * a
            cx = 380 + 580 * i
            rr(d, [cx - 250, 220, cx + 250, 560], 24, fill=CARD + (int(255 * ai),), outline=LINE + (int(255 * ai),),
               width=2)
            text(d, (cx, 250), big, 120, GREEN, "Bold", "ma", ai)
            text(d, (cx, 420), l1, 32, INK, "SemiBold", "ma", ai)
            text(d, (cx, 466), l2, 32, INK2, "Medium", "ma", ai)
        al = fade(t, 8.5, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 640, "A quarter dioptre is the usual smallest click of the phoropter's dial.", 32, al, INK,
                  "SemiBold")
        al = fade(t, 12.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 740, "Very precise, from a person comparing two blurry charts.", 40, al, GOLD, "SemiBold")
        al = fade(t, 14.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 900, "Shah R, Edgar DF, Rabbetts R, Harle DE, Evans BJW (2009). Standardized patient "
                                 "methodology to assess refractive error reproducibility.", 22, al, MUTED, "Medium")
        big_title(d, W / 2, 934, "Optometry and Vision Science 86(5): 517–528. Optometrists in England, standardised "
                                 "patients.", 22, al, MUTED, "Medium")
        return np.asarray(im)


# ======================================================================== patients aren't perfect
def answer_cases():
    """A 55-year-old who needs -2.75, asked "1 or 2?" (2 adds -0.25), on the smallest line they can just read:
    a clear difference, almost no difference, and a clear difference for a less reliable patient."""
    out = []
    clear = ("pick 1, the blurrier one", "pick 2, the clearer one")
    for S, dial, title, msg, labs in (
            (-2.00, 1.0, "A CLEAR DIFFERENCE", "Lens 2 really is clearer. Most patients say so.", clear),
            (-2.75, 1.0, "ALMOST NO DIFFERENCE", "Lens 1 is already right; lens 2 is a touch too strong. It barely "
                                                 "looks different, so they guess.", ("pick 1, the right lens",
                                                                                      "pick 2, a touch too strong")),
            (-2.00, 2.0, "THE SAME LENSES, A LESS SURE PATIENT", "Some patients guess more than others.", clear)):
        env = ClinicEnv6(1, seed=0, sampler=fixed_pair(**DEMO), dial=dial, rand_dial=False)
        env.S[0, 0], env.C[0, 0], env.ax[0, 0] = S, 0.0, 180.0
        T = float(env.mono_T(0, env.lens_pv(0))[0])
        env.line[0] = np.clip(np.round(T + 0.1, 1), -0.3, 1.0)
        P, right = env.compare_probs(E5.CMP_M)
        out.append(dict(S=S, title=title, msg=msg, line=float(env.line[0]), P=[float(x) for x in P[0]],
                        right=int(right[0]), labs=labs))
    return out


def of_ten(P):
    """Probabilities -> counts out of 10 (largest remainder)."""
    x = np.array(P) * 10
    n = np.floor(x).astype(int)
    for i in np.argsort(-(x - n))[:10 - n.sum()]:
        n[i] += 1
    return n


class PatientAnswers7(PatientSays4):
    """Two charts with a green glow as bright as the chance the patient picks each one, and ten patients' answers."""
    T_INTRO = 7.0
    STEP = 10.0

    def __init__(self):
        super().__init__()
        self.rows = answer_cases()
        self.dur = self.T_INTRO + self.STEP * len(self.rows) + 5.0

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        if t < self.T_INTRO:
            al = fade(t, 0.3, self.T_INTRO, 0.5, 0.5)
            big_title(d, W / 2, 300, "Patients aren't perfect", 64, al)
            big_title(d, W / 2, 420, "When two lenses look almost the same, people guess,", 40,
                      fade(t, 1.5, self.T_INTRO, 0.5, 0.5), INK2, "SemiBold")
            big_title(d, W / 2, 476, "and sometimes they point the examiner the wrong way.", 40,
                      fade(t, 2.3, self.T_INTRO, 0.5, 0.5), INK2, "SemiBold")
            big_title(d, W / 2, 620, "Here's how our simulated patients answer.", 36, fade(t, 3.6, self.T_INTRO, 0.5, 0.5),
                      GOLD, "SemiBold")
            return fade_frame(im, a_all)
        if t >= self.T_INTRO + self.STEP * len(self.rows):
            al = fade(t, self.T_INTRO + self.STEP * len(self.rows), 999, 0.5, 0.1)
            big_title(d, W / 2, 440, "The textbook and the AI face the same patients,", 44, al, INK, "SemiBold")
            big_title(d, W / 2, 510, "and the same guesses.", 44, al, GOLD, "SemiBold")
            return fade_frame(im, a_all)
        i = int((t - self.T_INTRO) // self.STEP)
        tt = t - self.T_INTRO - self.STEP * i
        r = self.rows[i]
        al = fade(tt, 0, self.STEP, 0.4, 0.4) * a_all
        text(d, (60, 36), r["title"], 34, GOLD, "Bold", a=al)
        text(d, (60, 90), "“Which is clearer, 1 or 2?”   Green glow: how likely the patient is to pick it.", 28, INK,
             "SemiBold", a=al)
        imgs, which = self.chart_pair(r["S"], r["line"])
        rows = ui.proj_chart(which)[1]
        li = int(np.argmin(np.abs(CH_LINES() - r["line"])))
        row = [rw for rw in rows if rw[0] == li]
        yc, hh = (row[0][1], row[0][2]) if row else (ui.CHART_H / 2, 40)
        half = max(4.0 * hh, 230)
        y0c, y1c = int(max(0, yc - half)), int(min(ui.CHART_H, yc + half))
        g = smooth((tt - 0.8) / 1.0)
        for j, (img, lab) in enumerate(zip(imgs, ("1", "2"))):
            x0 = 90 + j * 530
            crop = Image.fromarray(img).convert("RGB").crop((0, y0c, ui.CHART_W, y1c))
            sc = min(460 / crop.size[0], 600 / crop.size[1])
            src = crop.resize((int(crop.size[0] * sc), int(crop.size[1] * sc)), Image.LANCZOS)
            ytop = int(500 - src.size[1] / 2)
            if al > 0.02:
                tile = Image.blend(Image.new("RGB", src.size, BG), src, min(1, al))
                im.paste(tile, (x0, ytop))
            d = ImageDraw.Draw(im, "RGBA")
            p = r["P"][j] * g * al
            for k in range(8):
                aa = int(255 * p * (0.95 - 0.11 * k))
                if aa > 4:
                    d.rounded_rectangle([x0 - 8 - 4 * k, ytop - 8 - 4 * k, x0 + src.size[0] + 8 + 4 * k,
                                         ytop + src.size[1] + 8 + 4 * k], 12 + 4 * k, outline=(110, 230, 140, aa), width=4)
            ui.badge(d, x0 + 40, ytop + 40, lab, GOLD, a=al, r=32)
        # ten patients like this one
        n1, n2, ns = of_ten(r["P"])
        clearer = r["right"]
        cats = []
        for k, nk in ((0, n1), (1, n2)):
            if clearer == 2:
                col = (150, 160, 180)
            else:
                col = GREEN if k == clearer else RED
            cats.append((nk, col, r["labs"][k]))
        cats.append((ns, (120, 125, 140), "say “they look the same”"))
        x0, y0 = 1200, 230
        text(d, (x0, y0 - 50), "TEN PATIENTS LIKE THIS ONE", 22, MUTED, "Bold", a=al)
        people = [c for nk, c, _ in cats for _ in range(nk)]
        for m, col in enumerate(people):
            am = al * smooth((tt - 1.2 - 0.12 * m) / 0.4)
            cx = x0 + 30 + (m % 5) * 110
            cy = y0 + 40 + (m // 5) * 150
            d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=tuple(col) + (int(255 * am),))
            d.rounded_rectangle([cx - 34, cy + 26, cx + 34, cy + 96], 26, fill=tuple(col) + (int(255 * am),))
        y = y0 + 330
        for nk, col, lab in cats:
            ay = al * smooth((tt - 2.6) / 0.6)
            d.rounded_rectangle([x0, y, x0 + 28, y + 28], 6, fill=tuple(col) + (int(255 * ay),))
            text(d, (x0 + 44, y + 14), f"{nk}  {lab}", 30, INK, "SemiBold", "lm", ay)
            y += 50
        am = fade(tt, 3.6, self.STEP, 0.5, 0.4) * a_all
        big_title(d, W / 2, 960, r["msg"], 36, am, GOLD, "SemiBold")
        return fade_frame(im, a_all)


def CH_LINES():
    import chart
    return chart.LINES


# ======================================================================== how it learns
class Learn7(Learn6):
    STEPS = [("1", "It examines a patient", "At first every choice is random: it doesn't know what any button does."),
             ("2", "It gets a score", "From the patient's reaction to the glasses, minus the time in the chair."),
             ("3", "It adjusts", "Choices from an exam that went worse than usual become a little less likely next "
                                 "time. Better than usual: a little more likely.")]
    LAST = "Its first 1.8 million practice patients smiled when the letters got clearer. After that: no smiles."


# ======================================================================== the learning montage
class Montage7(Section):
    """The same patient (A) examined at checkpoints through practice: the exam as step numbers, the reaction, the
    score, and what it does with it; 100 test patients at each checkpoint."""
    CAPTIONS = {
        0: "Patient #1. Random buttons. After 15 minutes the patient gives up and walks out.",
        2551: "Still guessing, and for a while it gets worse: nearly everyone walks out.",
        10114: "One lesson learned: prescribe before the 15 minutes are up. The glasses are still wrong.",
        20200: "A shortcut: open both eyes, prescribe at once. “Still blurry”? More power, try again.",
        751332: "This patient still gets the shortcut. Others now get each eye covered in turn.",
        1000094: "Its own routine appears: both eyes, then each eye, with astigmatism lens and a spin (S).",
        1812949: "Happy. Until now, practice patients smiled when the letters got clearer: a hint.",
        "real": "From here, realistic patients: no smiles, a rough “stop!”. We nudged it to try step 5, "
                "and made that button turn the angle for it.",
        1913144: "It keeps step 5: after the spin, “1 or 2?” for the angle (5) and strength (6).",
        "final": "After 2.4 million practice patients: its own sequence.",
    }

    def __init__(self, day1, ai):
        st = json.load(open(os.path.join(DATA, "montage7.json")))
        keep = [0, 2551, 10114, 20200, 751332, 1000094, 1812949]
        self.stages = []
        for s in st:
            r = s["rows"][0]
            real = s["run"] == "m"
            key = s["total"] if not real else ("real" if s["total"] == 1812949 else
                                              ("final" if s.get("final") else s["total"]))
            if not real and s["total"] not in keep:
                continue
            o, sc = outcome_of_err(r["err"], r["minutes"], r["outcome"] == "walked")
            row = dict(steps=r["steps"], buttons=r["buttons"], outcome=o, score=sc)
            if s["total"] == 0:                                  # the exam shown in full: the AI's first patient
                o, sc = outcome_of(day1)
                row = dict(steps=collapse([x["a"] for x in day1["steps"]]), buttons=len(day1["steps"]), outcome=o,
                           score=sc)
            if s.get("final"):                                   # the exam shown in full after practice
                o, sc = outcome_of(ai)
                row = dict(steps=collapse([x["a"] for x in ai["steps"]]), buttons=len(ai["steps"]), outcome=o,
                           score=sc)
            self.stages.append(dict(total=s["total"], real=real, final=bool(s.get("final")), per100=s["per100"],
                                    usual=s["net"], row=row, caption=self.CAPTIONS[key]))
        self.durs = [14.0] + [8.5] * (len(self.stages) - 1)
        for i, s in enumerate(self.stages):
            if s["caption"] == self.CAPTIONS["real"]:
                self.durs[i] = 12.0
        self.starts = np.cumsum([0.0] + self.durs[:-1])
        self.dur = float(sum(self.durs)) + 1.0
        self.sfx = [(float(x) + 0.3, "tick") for x in self.starts]

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        i = int(np.searchsorted(self.starts, t, side="right") - 1)
        i = max(0, min(i, len(self.stages) - 1))
        tt = t - self.starts[i]
        s = self.stages[i]
        r = s["row"]
        al = fade(tt, 0, self.durs[i], 0.35, 0.35) if i < len(self.stages) - 1 else fade(tt, 0, 999, 0.35, 0.1)
        al *= a_all
        # header
        text(d, (60, 28), "LEARNING FROM MISTAKES", 40, INK, "Bold", a=a_all)
        text(d, (1860, 26), "PRACTICE PATIENTS SO FAR", 20, MUTED, "Bold", "ra", a_all)
        n_show = s["total"] if s["total"] > 0 else 1
        text(d, (1860, 52), f"{n_show:,}", 56, INK, "Bold", "ra", al)
        tag = "after practice" if s["final"] else ("realistic patients" if s["real"] else "patients who smile")
        text(d, (1860, 116), tag, 22, GOLD if s["real"] else INK2, "SemiBold", "ra", al)
        legend(d, 60, 170, a_all, rh=46, s=32, size=21)
        # the exam, as step numbers
        X0, X1, Y0 = 640, 1860, 190
        text(d, (X0, Y0 - 22), "PATIENT A  ·  24, short-sighted with astigmatism  ·  the same patient every time", 20,
             MUTED, "Bold", a=al)
        cs, pitch, per_line, max_lines = 30, 36, int((X1 - X0) // 36), 3
        seq = r["steps"]
        cap = per_line * max_lines
        shown = seq if len(seq) <= cap else seq[:cap - 1] + ["…"]
        k_vis = int(len(shown) * smooth((tt - 0.3) / 2.2))
        cue = r["score"] > s["usual"]
        cue_on = smooth((tt - 4.6) / 0.6) if not s["final"] else 0.0
        for k, lab in enumerate(shown[:k_vis]):
            x = X0 + (k % per_line) * pitch
            y = Y0 + 16 + (k // per_line) * 40
            chip(d, x, y, lab if lab != "…" else "…", al, cs)
            if cue_on > 0:
                col = (110, 230, 140) if cue else (255, 90, 90)
                d.rounded_rectangle([x - 2, y - 2, x + cs + 2, y + cs + 2], 7,
                                    fill=col + (int(110 * cue_on * al),))
        yb = Y0 + 16 + max_lines * 40 + 10
        ao = al * smooth((tt - 2.8) / 0.5)
        m_ = r["buttons"]
        text(d, (X0, yb), f"{m_} buttons" + (f"  (the first {cap - 1} steps shown)" if len(seq) > cap else ""), 22,
             INK2, "Medium", a=ao)
        col = OUT_COL[r["outcome"]]
        d.rounded_rectangle([X0, yb + 44, X0 + 30, yb + 74], 6, fill=tuple(col) + (int(255 * ao),))
        text(d, (X0 + 46, yb + 59), OUT_TEXT[r["outcome"]], 32, col, "Bold", "lm", ao)
        text(d, (X1, yb + 59), f"score {r['score']:+.0f}", 36, INK, "Bold", "rm", ao)
        if not s["final"]:
            ac = al * cue_on
            if cue:
                msg, ccol = "▲ Better than usual: these choices become a little more likely", (110, 230, 140)
            else:
                msg, ccol = "▼ Worse than usual: these choices become a little less likely", (255, 110, 110)
            text(d, (X0, yb + 104), msg, 28, ccol, "SemiBold", a=ac)
        else:
            text(d, (X0, yb + 104), "Practice is over: from here it always takes its most likely choice.", 28, INK2,
                 "SemiBold", a=al * smooth((tt - 4.0) / 0.6))
        # 100 test patients at this point
        gy = 640
        text(d, (X0, gy), "100 TEST PATIENTS AT THIS POINT", 20, MUTED, "Bold", a=al)
        cols = []
        for (key, _), nk in zip(OUT_SHORT, s["per100"]):
            cols += [OUT_COL[key]] * nk
        cols = (cols + [OUT_COL["walked"]] * 100)[:100]
        cell = 26
        for m, c in enumerate(cols):
            x = X0 + (m % 20) * (cell + 4)
            y = gy + 36 + (m // 20) * (cell + 4)
            d.rounded_rectangle([x, y, x + cell, y + cell], 4, fill=tuple(c) + (int(255 * al),))
        lx = X0 + 20 * (cell + 4) + 40
        for j, ((key, lab), nk) in enumerate(zip(OUT_SHORT, s["per100"])):
            y = gy + 36 + j * 30
            d.rounded_rectangle([lx, y + 4, lx + 20, y + 24], 4, fill=tuple(OUT_COL[key]) + (int(255 * al),))
            text(d, (lx + 32, y + 14), f"{lab}", 22, INK, "SemiBold", "lm", al)
            text(d, (lx + 330, y + 14), f"{nk}", 24, INK, "Bold", "rm", al)
        # the caption
        ac = al * smooth((tt - 1.0) / 0.6)
        lines = wrap(d, s["caption"], 32, 1780, "SemiBold")
        for q, ln in enumerate(lines[:2]):
            text(d, (W / 2, 960 + 44 * q), ln, 32, GOLD, "SemiBold", "ma", ac)
        return np.asarray(im)


# ======================================================================== the two sequences side by side
class Compare7(Section):
    """The same patient: the textbook's steps and the AI's, in step numbers, and the differences one by one."""
    POINTS = [
        ("The AI starts with both eyes open, adding power while the chart improves.", "first_both"),
        ("No astigmatism check (4): it puts in some astigmatism lens (6) and spins the angle (S).", "spin"),
        ("Then the textbook's own tools: the cross-cylinder for the angle (5) and the strength (6).", "jcc"),
        ("No fogging and no balancing (7): it finishes with both eyes together, taking extra power off.", "end"),
    ]

    def __init__(self, tb, ai, dur=46.0):
        self.tb, self.ai = stage_groups(tb), stage_groups(ai)
        self.dur = dur
        self.sfx = [(8.0 + 8.0 * i, "tick") for i in range(len(self.POINTS))]

    def hi(self, which, row, gi, lab, n_groups):
        if which == "first_both":
            return row == "ai" and gi == 0
        if which == "spin":
            return (row == "tb" and lab == "4") or (row == "ai" and lab == "S")
        if which == "jcc":
            return lab in ("5", "6")
        if which == "end":
            return (row == "tb" and lab == "7") or (row == "ai" and gi == n_groups - 1)
        return False

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 30, "The same patient: the textbook's steps and the AI's", 46, fade(t, 0.3, 999, 0.5, 0.1) * a)
        legend(d, 40, 150, a * 0.95, rh=42, s=30, size=19)
        k = int((t - 8.0) // 8.0) if t >= 8.0 else -1
        k = min(k, len(self.POINTS) - 1)
        which = self.POINTS[k][1] if k >= 0 else None
        X0 = 560
        y = 140
        for row, name, groups, col in (("tb", "THE TEXTBOOK", self.tb, TB_COL), ("ai", "THE AI", self.ai, AI_COL)):
            ar = fade(t, 1.0 if row == "tb" else 3.5, 999, 0.5, 0.1) * a
            text(d, (X0, y), name, 30, col, "Bold", a=ar)
            y += 50
            for gi, (lab, seq) in enumerate(groups):
                text(d, (X0, y + 20), lab, 22, GOLD if lab != "BOTH EYES" else (120, 200, 255), "Bold", "lm", ar)
                for j, s_ in enumerate(seq):
                    on = which is not None and self.hi(which, row, gi, s_, len(groups))
                    dim = 1.0 if which is None or on else 0.45
                    chip(d, X0 + 190 + j * 48, y, s_, ar, 40, ring=(255, 255, 255) if on else None, dim=dim)
                y += 54
            y += 34
        if k >= 0:
            for q in range(k + 1):
                aq = fade(t, 8.0 + 8.0 * q, 999, 0.5, 0.1) * a * (1.0 if q == k else 0.55)
                for m, ln in enumerate(wrap(d, f"{q + 1}.  {self.POINTS[q][0]}", 28, 1300, "SemiBold")[:2]):
                    text(d, (X0, 790 + 70 * q + 32 * m - (0 if m == 0 else 0)), ln, 26 if q != k else 28,
                         GOLD if q == k else INK2, "SemiBold", a=aq)
        return np.asarray(im)


# ======================================================================== results (simulated patients)
class Results7(Section):
    dur = 24.0

    def __init__(self, R):
        ai = R["ai"]
        self.stats = [(f"{100 * ai['summary']['se_within_025']:.0f}%", "overall lens power", "within ¼ D of the truth"),
                      (f"{100 * ai['summary']['se_within_050']:.0f}%", "overall lens power", "within ½ D of the truth"),
                      (f"{100 * ai['axis']['within_10']:.0f}%", "astigmatism angle", "within 10° (0.75 D or more)"),
                      (f"{100 * ai['summary']['happy']:.0f} of 100", "happy patients", ""),
                      (mmss(ai["summary"]["minutes"]), "in the chair", "on average")]
        self.n = R["meta"]["test_patients"]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 40, "How close did the AI get?", 54, fade(t, 0.3, 999, 0.5, 0.1) * a)
        big_title(d, W / 2, 124, f"{fmt_n(self.n)} new simulated patients, typical reliability. In a simulation the true "
                                 "prescription is known.", 28, fade(t, 1.0, 999, 0.5, 0.1) * a, INK2, "Medium")
        for i, (big, l1, l2) in enumerate(self.stats):
            ai_ = fade(t, 2.5 + 1.6 * i, 999, 0.5, 0.1) * a
            row, col = divmod(i, 3)
            cx = 360 + 600 * col + (300 if row == 1 else 0)
            y0 = 210 + 330 * row
            rr(d, [cx - 270, y0, cx + 270, y0 + 290], 24, fill=CARD + (int(255 * ai_),), outline=LINE + (int(255 * ai_),),
               width=2)
            text(d, (cx, y0 + 26), big, 96, AI_COL, "Bold", "ma", ai_)
            text(d, (cx, y0 + 170), l1, 32, INK, "SemiBold", "ma", ai_)
            if l2:
                text(d, (cx, y0 + 214), l2, 28, INK2, "Medium", "ma", ai_)
        al = fade(t, 12.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 920, "Simulated patients, not real ones: the real test of any refraction is a real patient.",
                  32, al, GOLD, "SemiBold")
        return np.asarray(im)


class Reliability7(Section):
    """The AI's consistency on the same patients examined twice, with more and less reliable patients."""
    dur = 22.0

    def __init__(self, R):
        self.vals = [(dk, R["ai"]["by_dial"][dk]["agree"]["se_025"]) for dk in ("0.5", "1.0", "2.0")]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 40, "And when patients are less sure?", 54, fade(t, 0.3, 999, 0.5, 0.1) * a)
        big_title(d, W / 2, 124, "Each simulated patient examined twice by the AI, fresh answers each time. "
                                 "Do the two prescriptions agree within ¼ D?", 28, fade(t, 1.0, 999, 0.5, 0.1) * a, INK2,
                  "Medium")
        base, top = 830, 420
        for j, (dk, v) in enumerate(self.vals):
            cx = 480 + 480 * j
            aj = fade(t, 2.5 + 2.5 * j, 999, 0.5, 0.1) * a
            F6.Consistency.dial_icon(d, cx, 240, 50, {"0.5": 0.15, "1.0": 0.5, "2.0": 0.85}[dk], aj)
            text(d, (cx, 270), {"0.5": "STEADY", "1.0": "TYPICAL", "2.0": "UNSURE"}[dk] + " PATIENTS", 26, INK, "Bold",
                 "ma", aj)
            g = smooth((t - 2.9 - 2.5 * j) / 1.2)
            hgt = (base - top) * v * g
            d.rectangle([cx - 110, base - hgt, cx + 110, base], fill=tuple(AI_COL) + (int(255 * aj),))
            text(d, (cx, base - hgt - 14), f"{100 * v:.0f}%", 52, INK, "Bold", "md", aj * min(1, g * 2))
        al = fade(t, 11.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 900, "Shakier answers cost some consistency, for any examiner.", 36, al, GOLD, "SemiBold")
        return np.asarray(im)


# ======================================================================== the film
def step_no(a):
    s = STEP[int(a)]
    return s, STEP_COL[s]


def exam(tr, *args, **kw):
    ex = Exam5(tr, *args, **kw)
    ex.halo, ex.step_no = True, step_no
    return ex


def sections():
    R = json.load(open(os.path.join(DATA, "results_m.json")))
    H3 = json.load(open(os.path.join(DATA, "results_h.json")))
    d1, ai, tb = trace("h_day1"), trace("m_ai"), trace("m_borish")
    R6 = dict(R, h_ai_net=H3["ai"]["summary"]["net"])
    N = F6.make_notes(R6, ai, tb, trace("m_day1"))
    S = {}
    S["hook"] = Hook7()
    S["hired"] = Say([(0.5, "An optometrist hires an AI", 64, INK, "Bold"),
                      (2.0, "Its job: the refraction, the part of the eye exam that measures your glasses prescription.",
                       38, INK2, "Medium"),
                      (5.5, "That leaves the optometrist more time for the rest of the exam: the health of your eyes, "
                            "from the front to the retina, how your two eyes work together, contact lenses, and more.",
                       34, INK2, "Medium"),
                      (11.0, "Every patient in this video is simulated. The AI has never examined a real person.", 32,
                       GOLD, "SemiBold")], 16.0)
    S["phoropter"] = Phoropter7()
    S["textbook"] = TextbookSteps7()
    S["works"] = TextbookWorks7()
    S["patients"] = PatientAnswers7()
    S["ai_start"] = Say([(0.5, "The AI starts with nothing", 64, INK, "Bold"),
                         (2.0, "It doesn't know what a lens does, or what the questions mean.", 38, INK2, "Medium"),
                         (4.4, "It has the same phoropter buttons as the optometrist, and it hears the patient's answers.",
                          38, INK2, "Medium"),
                         (7.4, "After each patient, it gets a score.", 44, GOLD, "SemiBold")], 11.0)
    S["wants"] = Wants5()
    S["learn"] = Learn7()
    k1 = next((k for k, x in enumerate(d1["steps"]) if x["code"] in (7, 8, 9)), None)
    n1 = {0: "Its very first practice patient. It has no idea what any button does: every choice is random."}
    if k1 is not None:
        n1[k1] = "It tries prescribing. The patient: “It’s still blurry.” The exam goes on."
    S["day1_exam"] = exam(d1, "PATIENT #1", "before any practice", slow=range(6), notes=n1, outro=6.0, fast_beat=0.05)
    S["montage"] = Montage7(d1, ai)
    S["ai_title"] = Say([(0.4, "After 2.4 million practice patients", 60, INK, "Bold"),
                         (1.6, "Patient A again, at full speed.", 36, INK2, "Medium")], 6.0)
    na = dict(N["ai_exam"])
    na[0] = ("Patient A again. Both eyes open first: power a quarter at a time while the letters improve, and then "
             "some more.")
    S["ai_exam"] = exam(ai, "THE AI", f"after {fmt_n(TOTAL_PRACTICE)} practice patients", notes=na,
                        slow=N.get("ai_slow", ()))
    S["tb_title"] = Say([(0.4, "The textbook, same patient", 60, INK, "Bold"),
                         (1.6, "Written as a script: the same buttons, the same patient, the same kind of answers.", 36,
                          INK2, "Medium")], 6.0)
    S["tb_exam"] = exam(tb, "THE TEXTBOOK", "the same patient, step by step", who="TEXTBOOK", who_col=TB_COL,
                        notes=N.get("tb_exam", {}), slow=N.get("tb_slow", ()), fast_beat=0.12)
    S["compare"] = Compare7(tb, ai)
    S["results"] = Results7(R)
    S["reliability"] = Reliability7(R)
    S["outro"] = Say([(0.5, "Dr. Borish built his sequence from years of teaching and practice.", 42, INK, "SemiBold"),
                      (3.0, "The AI needed 2.4 million simulated patients to find one of its own,", 42, INK, "SemiBold"),
                      (5.2, "and it ended up using the textbook's own tools: the cross-cylinder and “1 or 2?”.", 42,
                       GOLD, "SemiBold"),
                      (9.0, "Every patient here is simulated. The AI has never examined a real person: this is a "
                            "demonstration, not a clinical tool. If your vision changes, see your optometrist.", 28,
                       INK2, "Medium")], 17.0)
    return S


ORDER = ["hook", "hired", "phoropter", "textbook", "works", "patients", "ai_start", "wants", "learn", "day1_exam",
         "montage", "ai_title", "ai_exam", "tb_title", "tb_exam", "compare", "results", "reliability", "outro"]


def order():
    S = sections()
    return [n for n in ORDER if n in S], S


def render(name):
    _, S = order()
    S[name].render(PREFIX + name)


def render_all(names=None, par=4):
    names = names or order()[0]

    def one(nm):
        log = open(os.path.join(OUT, f"{PREFIX}{nm}.log"), "w")
        r = subprocess.run([sys.executable, "film7.py", "render", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
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
    paths = [os.path.join(OUT, f"{PREFIX}{s}.mp4") for s in names]
    ds = [film.dur(p) for p in paths]
    off = [0.0]
    for dd in ds[:-1]:
        off.append(off[-1] + dd - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(names, off))
    for s, o, dd in zip(names, off, ds):
        print(f"  {s:12s} {o:7.2f} + {dd:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video7.mp4")
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
    energy = [(0, 0.3), (g("phoropter"), 0.35), (g("textbook"), 0.35), (g("works"), 0.4), (g("patients"), 0.35),
              (g("ai_start"), 0.45), (g("learn"), 0.4), (g("day1_exam"), 0.35), (g("montage"), 0.6),
              (g("ai_title"), 0.5), (g("ai_exam"), 0.4), (g("tb_exam"), 0.35), (g("compare"), 0.55),
              (g("results"), 0.7), (g("reliability"), 0.55), (g("outro"), 0.4), (total, 0.0)]
    sc = SC.Score(total, sorted(energy), [(g("montage"), 0.6), (g("results"), 0.7)],
                  [(g("ai_title") - 2.0, g("ai_title")), (g("results") - 2.0, g("results"))],
                  transpose_at=g("ai_title"), final_at=g("outro") + 0.5, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.7
    mix = mix + film.sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score7.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "ai_eye_exam.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], dd] for s, dd in zip(names, ds)}, "total": total},
              open(os.path.join(OUT, "timeline7.json"), "w"), indent=1)
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
        Image.fromarray(S[sys.argv[2]].frame(float(sys.argv[3]))).save(os.path.join(OUT, f"frame7_{sys.argv[2]}.png"))
    elif cmd == "all":
        render_all()
        assemble()
    elif cmd == "assemble":
        assemble()
