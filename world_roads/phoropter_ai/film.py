"""The whole film: sections in order, rendering, and assembly with the score.

    python film.py list                 section names and lengths
    python film.py render <name>        render one section to out/<name>.mp4
    python film.py all                  render every section (3 at a time), then assemble
    python film.py assemble             join sections (cross-fades) and add music -> out/phoropter_ai.mp4
"""
import os, sys, json, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ui
from ui import INK, INK2, MUTED, GOLD, GREEN, RED, CARD, LINE, text, rr, fade, smooth
from scenes import Section, Hook, Rules, Counter, Exam, para, wrap
import results_scenes as RS
from scenes2 import CHIP as CHIP2

OUT = ui.OUT
XF = 0.5
TR = os.path.join(HERE, "data", "traces")


def trace(name):
    return json.load(open(os.path.join(TR, f"{name}.json")))


def fmt_n(n):
    return f"{int(n):,}"


# ======================================================================== text cards
class Card(Section):
    """A title and paragraphs that appear one by one. items: [(t0, text, size, colour, weight)]"""

    def __init__(self, title, items, dur, kicker=None):
        self.title, self.items, self.dur, self.kicker = title, items, dur, kicker

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        y = 250
        if self.kicker:
            text(d, (960, y - 70), self.kicker, 26, GOLD, "SemiBold", "ma", a)
        text(d, (960, y), self.title, 60, INK, "Bold", "ma", a)
        y += 130
        for t0, s, size, col, wt in self.items:
            al = fade(t, t0, 999, 0.6, 0.1) * a
            lines = wrap(d, s, size, 1300, wt)
            for ln in lines:
                text(d, (960, y), ln, size, col, wt, "ma", al)
                y += size * 1.35
            y += 26
        return np.asarray(im)


class SamePatient(Section):
    """One two-eyed patient, three examiners: final prescriptions and how far each eye is from the truth."""

    def __init__(self, entries, dur=13.0):
        self.entries, self.dur = entries, dur

    def frame(self, t):
        from optics import to_pv, rx_error
        import math
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        p = self.entries[0][1]["patient"]
        text(d, (80, 56), "THE SAME PATIENT, THREE EXAMINERS", 44, INK, "Bold", a=a)
        text(d, (80, 114), "True Rx  R " + ui.fmt_rx(p["S"], p["C"], p["axis"]) + "   L " +
             ui.fmt_rx(p["S_os"], p["C_os"], p["axis_os"]), 26, MUTED, "Medium", a=a)
        E = [to_pv(p["S"], p["C"], p["axis"]), to_pv(p["S_os"], p["C_os"], p["axis_os"])]
        r2 = lambda v: math.floor(abs(v) * 100 + 0.5) / 100
        for j, h in enumerate(("RIGHT EYE", "LEFT EYE")):
            text(d, (700 + j * 560, 200), h, 20, MUTED, "SemiBold", a=a)
        text(d, (1820, 200), "MOVES", 20, MUTED, "SemiBold", "ra", a)
        for i, (name, tr, col) in enumerate(self.entries):
            al = fade(t, 0.6 + 1.0 * i, 999, 0.5, 0.1) * a
            y = 270 + i * 190
            rr(d, [60, y - 30, 1860, y + 140], 18, fill=CARD + (int(255 * al),), outline=col + (int(255 * al),), width=3)
            text(d, (100, y + 40), name, 32, INK, "SemiBold", "lm", al)
            last = tr["steps"][-1]["after"]
            for e in (0, 1):
                S, C, ax = last[e]
                L = to_pv(S, C, ax)
                err = rx_error(L, E[e])
                dse = L[0] - E[e][0]
                c2 = GREEN if err < 0.13 else (GOLD if err <= 0.25 + 1e-9 else RED)
                x = 700 + e * 560
                text(d, (x, y + 8), ui.fmt_rx(S, C, ax), 34, INK, "SemiBold", a=al)
                txt = "exact" if err < 0.13 else f"off {r2(err):.2f} D"
                if err >= 0.13 and abs(dse) >= 0.12:
                    txt += f" ({'too minus' if dse < 0 else 'too plus'})"
                text(d, (x, y + 62), txt, 26, c2, "SemiBold", a=al)
            text(d, (1820, y + 40), str(len(tr["steps"])), 36, INK2, "SemiBold", "rm", al)
        return np.asarray(im)


# ======================================================================== the film
def results(name):
    return RS.load_results(name)


def sections():
    A = results("a")
    has_c = all(os.path.exists(os.path.join(HERE, "data", f)) for f in ("results_c.json", "results_c_plano.json",
                                                                         "errors_c.npz", "traces/c_final.json",
                                                                         "traces/c_noauto.json"))
    C3 = results("c") if has_c else None
    S = {}
    S["hook"] = Hook()
    S["rules"] = Rules()

    # ---------------- attempt 1
    S["a_title"] = Card("Attempt 1", [(0.8, "Scored only at the end of each exam.", 36, INK2, "SemiBold")], 4.5,
                        kicker="PRACTICE ROUND")
    t0 = trace("a_0")
    S["a_exam0"] = Exam(t0, "EXAM 1", "Untrained: a random network pressing random buttons", speed=0.55,
                        notes=[(0, "It has no idea what any button does."),
                               (5, "It asks the patient to read the chart… and learns nothing from it.")])
    ta = trace("a_exams_107079")
    S["a_c1"] = Counter(0, ta["meta"]["exams"], "practice exams later…")
    S["a_exam1"] = Exam(ta, f"AFTER {fmt_n(ta['meta']['exams'])} PRACTICE EXAMS", "Attempt 1", outro=5.5,
                        notes=[(0, "Its first discovery: press DONE at once and prescribe the autorefractor reading.")])
    tb = trace("a_exams_301936")
    S["a_c2"] = Counter(ta["meta"]["exams"], tb["meta"]["exams"], "practice exams later…")
    S["a_exam2"] = Exam(tb, f"AFTER {fmt_n(tb['meta']['exams'])} PRACTICE EXAMS", "Attempt 1",
                        notes=[(1, "Now it asks a question: is +0.25 clearer?"),
                               (2, "“The same,” so it takes the plus."),
                               (3, "Is more plus clearer? No. It stops there.")])
    tc = trace("a_exams_1002464")
    S["a_c3"] = Counter(tb["meta"]["exams"], tc["meta"]["exams"], "practice exams later…")
    S["a_exam3"] = Exam(tc, f"AFTER {fmt_n(tc['meta']['exams'])} PRACTICE EXAMS", "Attempt 1", outro=5.5,
                        notes=[(0, "The cross-cylinder: this patient wants more cylinder…"),
                               (1, "…but it never learned to add cylinder or turn the axis."),
                               (3, "Instead it checks the sphere, again and again.")])
    xa, ya, _ = RS.load_curve(os.path.join(HERE, "runs", "a", "logs", "train.csv"))
    Ah = {k: A[k]["summary_half"] for k in ("autoref", "pleaser", "textbook", "ai")}
    Af = {k: A[k]["summary"] for k in ("autoref", "pleaser", "textbook", "ai")}
    S["a_curve"] = RS.Curve("ATTEMPT 1: THE LEARNING CURVE", "Share of 1,000 check patients within 0.25 D, "
                            "on the score it was trained with",
                            [("AI", xa, ya, RS.AI_BLUE, 5)],
                            [("Textbook script", Ah["textbook"]["within_025"], RS.TB_ORANGE),
                             ("Autorefractor alone", Ah["autoref"]["within_025"], RS.GREY)], dur=13.0, draw_secs=7.0,
                            notes=[(2.2, 6.0, 4e4, "Shortcut: press DONE,\nprescribe the autorefractor", 0.36),
                                   (6.4, 12.6, 6e5, "Learns to ask questions", 0.72)])
    rows = lambda D: [("Autorefractor reading alone", D["autoref"], RS.GREY, False),
                      ("Script: give them what they prefer", D["pleaser"], RS.GREY, False),
                      ("Script: textbook technique", D["textbook"], RS.TB_ORANGE, False),
                      ("AI, attempt 1", D["ai"], RS.AI_BLUE, True)]
    S["a_score"] = RS.Scoreboard("5,000 NEW PATIENTS", "Simulated patients it never practised on. Every examiner saw "
                                 "the same 5,000.", rows(Ah), dur=13.0,
                                 note="Scored the way attempt 1 was trained. Exact: within half a click. "
                                      "Over-minused: at least 0.25 D more minus than the truth.")
    st = A["ai"]["strategy"]
    turned = sum(st["used_pct"][k] for k in ("Axis +5°", "Axis -5°", "Axis +15°", "Axis -15°"))
    S["a_habits"] = habits(A, "ATTEMPT 1: ITS HABITS", "AI, attempt 1", "a", [
        (f"{int(round(5000 * turned))}", "exams out of 5,000 where it turned the axis dial"),
        (f"{int(round(5000 * st['used_pct']['Cylinder -0.25']))}", "exams where it added any cylinder"),
        (f"{st['mean_count']['1 or 2? (+0.25)']:.1f}\u00d7", "per exam it asks \u201cis +0.25 clearer?\u201d")])
    S["a_twist"] = Card("Not so fast", [
        (0.6, "My score measured the error as a vector, but it counted the leftover cylinder at half weight.",
         34, INK2, "Medium"),
        (4.6, "So a 0.25 D cylinder error still counted as \u201cexact,\u201d and a 14\u00b0 axis error on a "
              "\u22121.00 cylinder looked half as bad as it really is.", 34, INK2, "Medium"),
        (9.6, "It didn\u2019t skip the axis by accident. My score told it the axis barely mattered.", 36, GOLD,
         "SemiBold")], 14.5, kicker="ATTEMPT 1")
    S["a_rescore"] = RS.Scoreboard("THE SAME EXAMS, SCORED FAIRLY", "Leftover cylinder now counts in full dioptres. "
                                   "Same 5,000 patients, same prescriptions.", rows(Af), dur=13.0,
                                   note="Error = length of (spherical-equivalent error, leftover cylinder). "
                                        "An axis error of d\u00b0 on cylinder C leaves 2\u00b7C\u00b7sin d of cylinder.")

    ta3 = trace("a_exams_1002464")["steps"][-1]["after"]
    S["vector"] = RS.VectorIdea((-2.25, -1.00, 170), tuple(ta3), "AI", dur=23.0)

    # ---------------- attempt 2: the fair score
    stat = lambda v: f"leftover cylinder under 0.25 D: {100 * np.mean(np.hypot(v[:, 1], v[:, 2]) < 0.25):.0f}%"
    ea = np.load(os.path.join(HERE, "data", "errors_a.npz"))
    B = results("b")
    eb = np.load(os.path.join(HERE, "data", "errors_b.npz"))
    S["b_title"] = Card("Attempt 2: the fair score", [
        (0.8, "Same patients, same buttons. Only the score changed: leftover cylinder now counts in full dioptres.",
         34, INK2, "Medium"),
        (4.4, f"Practice from scratch: {fmt_n(B['ai']['meta']['exams'])} exams.", 34, GOLD, "SemiBold")], 8.5,
        kicker="PRACTICE ROUND")
    S["b_clouds"] = RS.ErrorClouds("WHERE THE ERRORS LAND", "Each dot is one of 5,000 test patients: the cylinder "
                                   "left over after the exam (fair score)",
                                   [("Textbook script", ea["textbook"], RS.TB_ORANGE, stat(ea["textbook"])),
                                    ("AI, attempt 1", ea["ai"], RS.GREY, stat(ea["ai"])),
                                    ("AI, attempt 2", eb["ai"], RS.AI_BLUE, stat(eb["ai"]))], dur=15.0,
                                   note="Gold ring: 0.25 D. Bars: spherical-equivalent error (prescription minus "
                                        "truth). Attempt 2 still never turned the axis.")
    S["b_why"] = Card("The score wasn\u2019t the whole problem", [
        (0.6, "Turning the axis only pays off as a pair of moves: ask the cross-cylinder, then turn the way the "
              "patient points.", 34, INK2, "Medium"),
        (4.8, "Turn it at random and the prescription gets worse. Early in practice, before it could know the "
              "pair existed, it learned to leave the axis alone, and never went back.", 34, INK2, "Medium"),
        (10.4, "A fair score is necessary. It isn\u2019t sufficient.", 36, GOLD, "SemiBold")], 14.5)

    # ---------------- attempt 3: training wheels (+ half its practice with no autorefractor)
    if C3:
        Cf = C3["ai"]["summary"]
        S["c_title"] = Card("Attempt 3: training wheels", [
            (0.8, "Harder practice patients at first: bigger cylinders, autorefractor axes further off.", 30, INK2,
             "Medium"),
            (3.4, "And a \u201cwarmer\u201d or \u201ccolder\u201d after every move.", 30, INK2, "Medium"),
            (5.6, "Both fade to nothing by 30% of practice. The final score is the fair one.", 30, INK2, "Medium"),
            (8.4, "A real projector chart, read letter by letter, with three new buttons: show a smaller line, "
                  "show a bigger line, \u201cread this line.\u201d", 30, INK2, "Medium"),
            (12.4, "And half of its practice exams had no autorefractor reading at all.", 30, GOLD, "SemiBold")],
            17.0, kicker="PRACTICE ROUND")
        tf = trace("c_final")
        S["c_exam"] = Exam(tf, f"AFTER {fmt_n(tf['meta']['exams'])} PRACTICE EXAMS", "Attempt 3, training wheels "
                           "long gone", notes=notes("c_final"), outro=6.0, speed=0.6)
        xc, yc, shc = RS.load_curve(os.path.join(HERE, "runs", "c", "logs", "train.csv"))
        end_hint = xc[np.argmax(shc <= 0.0)] if (shc <= 0).any() else xc[-1]
        S["c_curve"] = RS.Curve("ATTEMPT 3: THE LEARNING CURVE", "Share of 1,000 check patients within 0.25 D, fair "
                                "score, with an autorefractor reading",
                                [("AI", xc, yc, RS.AI_BLUE, 5)],
                                [("Textbook script", Af["textbook"]["within_025"], RS.TB_ORANGE),
                                 ("Autorefractor alone", Af["autoref"]["within_025"], RS.GREY)],
                                dur=12.0, draw_secs=7.0, shade=(xc[0], end_hint, "TRAINING WHEELS FADING"))
        rows3 = [("Autorefractor reading alone", Af["autoref"], RS.GREY, False),
                 ("Script: textbook (letter chart)", C3["textbook"]["summary"], RS.TB_ORANGE, False),
                 ("AI, attempt 1", Af["ai"], RS.GREY, False),
                 ("AI, attempt 2", B["ai"]["summary"], RS.GREY, False),
                 ("AI, attempt 3", Cf, RS.AI_BLUE, True)]
        S["c_score"] = RS.Scoreboard("5,000 NEW PATIENTS, FAIR SCORE", "Starting from the autorefractor reading. The "
                                     "same 5,000 simulated patients as before.", rows3, dur=14.0,
                                     note="Attempt 3 and its textbook script read the letter chart; attempts 1\u20132 "
                                          "read whole lines. Exact: within half a click (0.13 D).")
        ec = np.load(os.path.join(HERE, "data", "errors_c.npz"))
        S["c_clouds"] = RS.ErrorClouds("WHERE THE ERRORS LAND", "Each dot is one of 5,000 test patients: the cylinder "
                                       "left over after the exam",
                                       [("Textbook script", ec["textbook"], RS.TB_ORANGE, stat(ec["textbook"])),
                                        ("AI, attempt 2", eb["ai"], RS.GREY, stat(eb["ai"])),
                                        ("AI, attempt 3", ec["ai"], RS.AI_BLUE, stat(ec["ai"]))], dur=14.0,
                                       note="Gold ring: 0.25 D. Bars: spherical-equivalent error (prescription minus "
                                            "truth).")
        Cp = results("c_plano")
        Bp = results("b_plano")
        S["n_title"] = Card("No autorefractor", [
            (0.8, "The phoropter starts at plano. Nothing to refine: the whole prescription has to be found.", 34,
             INK2, "Medium"),
            (4.4, "The textbook script brackets the sphere with \u00b11.00 D lenses and the chart, then fogs, searches "
                  "for cylinder at four axes, and refines.", 30, MUTED, "Medium"),
            (8.6, "Exams may now run to 80 moves.", 30, MUTED, "Medium")], 12.0, kicker="THE SECOND TEST")
        tn = trace("c_noauto")
        S["n_exam"] = Exam(tn, "NO AUTOREFRACTOR", "Attempt 3, same patient, starting at plano", speed=0.42,
                           notes=notes("c_noauto"), outro=6.0)
        rows_n = [("Prescribe nothing (plano)", Cp["start"]["summary"], RS.GREY, False),
                  ("Script: textbook, from scratch", Cp["textbook"]["summary"], RS.TB_ORANGE, False),
                  ("AI, attempt 2 (never practised this)", Bp["ai"]["summary"], RS.GREY, False),
                  ("AI, attempt 3", Cp["ai"]["summary"], RS.AI_BLUE, True)]
        S["n_score"] = RS.Scoreboard("NO AUTOREFRACTOR: 5,000 PATIENTS", "The same patients, starting at plano.",
                                     rows_n, dur=14.0, note="Fair score. Exact: within half a click (0.13 D).")
        S["c_habits"] = habits(C3, "ATTEMPT 3: ITS HABITS", "AI, attempt 3", "c", facts("c"))
    # ---------------- attempt 4: the real exam (two eyes, one line at a time, binocular balance)
    has_d = all(os.path.exists(os.path.join(HERE, "data", f)) for f in ("results_d.json", "errors_d.npz",
                                                                         "traces/d_ai.json", "traces/d_mine.json",
                                                                         "traces/d_textbook.json"))
    if has_d:
        from scenes2 import Exam2
        D = results("d")
        MINE = (0x00, 0xA8, 0x88)
        S["d_title"] = Card("Attempt 4: the real exam", [
            (0.8, "Two eyes. One projected line at a time.", 32, INK2, "Medium"),
            (3.0, "How well a patient can judge blur depends on the letters: sphere is easiest near the smallest line "
                  "they can read, the cross-cylinder a couple of lines bigger.", 30, INK2, "Medium"),
            (7.6, "Behind the phoropter, a covered-up patient keeps focusing a little, so one-eye results drift toward "
                  "minus. That is what binocular balancing is for.", 30, INK2, "Medium"),
            (12.4, "Three examiners on the same patients: the textbook, my own routine, and a fresh AI.", 32, GOLD,
             "SemiBold")], 17.0, kicker="THE OPTOMETRIST\u2019S NOTES")
        tm = trace("d_mine")
        tm["who"] = "ME"
        S["d_mine"] = Exam2(tm, "MY ROUTINE", "Scripted exactly as I do it, on a new two-eyed patient", speed=0.3,
                            notes=notes("d_mine"))
        ta = trace("d_ai")
        S["d_exam"] = Exam2(ta, f"AI, ATTEMPT 4: AFTER {fmt_n(ta['meta']['exams'])} PRACTICE EXAMS",
                            "The same patient", speed=0.3, notes=notes("d_ai"))
        S["d_same"] = SamePatient([("Textbook script", trace("d_textbook"), RS.TB_ORANGE),
                                   ("My routine", tm, MINE), ("AI, attempt 4", ta, RS.AI_BLUE)])
        xd, yd, shd = RS.load_curve(os.path.join(HERE, "runs", "d", "logs", "train.csv"))
        end_d = xd[np.argmax(shd <= 0.0)] if (shd <= 0).any() else xd[-1]
        Dt, Dm = D["textbook"]["summary"], D["mine"]["summary"]
        S["d_curve"] = RS.Curve("ATTEMPT 4: THE LEARNING CURVE", "Share of eyes within 0.25 D, 1,000 two-eyed check "
                                "patients", [("AI", xd, yd, RS.AI_BLUE, 5)],
                                [("Textbook script", Dt["within_025"], RS.TB_ORANGE),
                                 ("My routine", Dm["within_025"], MINE)],
                                dur=12.0, draw_secs=7.0, shade=(xd[0], end_d, "TRAINING WHEELS FADING"))
        rows_d = [("Autorefractor reading alone", D["autoref"]["summary"], RS.GREY, False),
                  ("Script: textbook", Dt, RS.TB_ORANGE, False),
                  ("Script: my routine", Dm, MINE, False),
                  ("AI, attempt 4", D["ai"]["summary"], RS.AI_BLUE, True)]
        S["d_score"] = RS.Scoreboard("5,000 TWO-EYED PATIENTS", "The full exam: both eyes, binocular balance, "
                                     "one projected line at a time.", rows_d, dur=15.0,
                                     note="Within 0.25 D and exact: per eye. Over-minused: eyes of patients under 40, "
                                          "at least 0.25 D too minus.")
        ed = np.load(os.path.join(HERE, "data", "errors_d.npz"))
        S["d_clouds"] = RS.ErrorClouds("WHERE THE ERRORS LAND", "Each dot is one eye of 5,000 two-eyed patients",
                                       [("Textbook script", ed["textbook"], RS.TB_ORANGE, stat(ed["textbook"])),
                                        ("My routine", ed["mine"], MINE, stat(ed["mine"])),
                                        ("AI, attempt 4", ed["ai"], RS.AI_BLUE, stat(ed["ai"]))], dur=14.0,
                                       note="Gold ring: 0.25 D. Bars: spherical-equivalent error (prescription minus "
                                            "truth); left of 0 = too much minus.")
        S["d_habits"] = RS.Habits("ATTEMPT 4: ITS HABITS", "Each chip is one move",
                                  [("AI, attempt 4: its most common exam", D["ai"]["habits"]["common_seq"],
                                    f"{D['ai']['habits']['common_n']} of 5,000 exams went exactly like this",
                                    RS.AI_BLUE),
                                   ("My routine: a typical exam", D["mine"]["habits"]["common_seq"],
                                    f"{D['mine']['habits']['common_n']} of 5,000 exams went exactly like this", MINE)],
                                  facts("d"), dur=16.0, chip=CHIP2)
    S["outro"] = Card("The fine print", [
        (0.6, "Every patient here was simulated. The optics are real physics; the people are a model.", 34, INK2,
         "Medium"),
        (4.0, "Real patients get tired, guess, change their minds, and have two eyes.", 34, INK2, "Medium"),
        (7.4, "This AI has never examined a real person. It is a demonstration, not a clinical tool.", 34, GOLD,
         "SemiBold"),
        (11.0, "Code, every assumption, and the numbers behind each chart: see the description.", 26, MUTED,
         "Medium")], 16.0)
    return S


def habits(R, title, ai_label, tag, facts):
    from collections import Counter as C
    sa = json.load(open(os.path.join(HERE, "data", f"seqs_{tag}.json")))
    strips = [(f"{ai_label}: its most common exam", sa["ai"]["seq"],
               f"{sa['ai']['n']} of 5,000 exams went exactly like this", RS.AI_BLUE),
              ("Textbook script: a typical exam", sa["textbook"]["seq"], "fog, unfog, cross-cylinder, re-check", RS.TB_ORANGE)]
    return RS.Habits(title, "Each chip is one move", strips, facts, dur=15.0)


def notes(name):
    """Commentary for a recorded exam, written after watching it: data/notes_<name>.json = [[step, text], ...]"""
    p = os.path.join(HERE, "data", f"notes_{name}.json")
    return [tuple(x) for x in json.load(open(p))] if os.path.exists(p) else []


def facts(tag):
    p = os.path.join(HERE, "data", f"facts_{tag}.json")
    return [tuple(x) for x in json.load(open(p))] if os.path.exists(p) else []


ORDER = ["hook", "rules", "a_title", "a_exam0", "a_c1", "a_exam1", "a_c2", "a_exam2", "a_c3", "a_exam3", "a_curve",
         "a_score", "a_habits", "a_twist", "vector", "a_rescore", "b_title", "b_clouds", "b_why", "c_title", "c_exam",
         "c_curve", "c_score", "c_clouds", "n_title", "n_exam", "n_score", "c_habits", "d_title", "d_mine", "d_exam",
         "d_same", "d_curve", "d_score", "d_clouds", "d_habits", "outro"]


def order():
    S = sections()
    return [k for k in ORDER if k in S], S


def render(name):
    _, S = order()
    S[name].render(name)


def render_all(names=None, par=3):
    names = names or order()[0]

    def one(nm):
        log = open(os.path.join(OUT, f"{nm}.log"), "w")
        r = subprocess.run([sys.executable, "film.py", "render", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        print(f"  {nm}: exit {r.returncode}", flush=True)
        return r.returncode

    _, S = order()
    names = sorted(names, key=lambda n: -S[n].dur)
    with ThreadPoolExecutor(par) as ex:
        codes = list(ex.map(one, names))
    assert not any(codes), "a section failed"


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def sfx(N, sr, events):
    """soft wooden ticks for each move, and a bell when an exam ends"""
    out = np.zeros((2, N), np.float32)
    for t, kind in events:
        i0 = int(t * sr)
        if kind == "tick":
            n = int(0.05 * sr)
            tt = np.arange(n) / sr
            s = (np.sin(2 * np.pi * 1850 * tt) * 0.6 + np.sin(2 * np.pi * 920 * tt) * 0.4) * np.exp(-tt / 0.008) * 0.10
        else:
            n = int(1.4 * sr)
            tt = np.arange(n) / sr
            s = (np.sin(2 * np.pi * 1318.5 * tt) + 0.5 * np.sin(2 * np.pi * 1975.5 * tt) * np.exp(-tt / 0.25)) \
                * np.exp(-tt / 0.45) * np.minimum(1, tt / 0.003) * 0.11
        m = min(n, N - i0)
        if m > 0:
            out[:, i0:i0 + m] += s[:m]
    return out


def assemble():
    names, S = order()
    paths = [os.path.join(OUT, f"{s}.mp4") for s in names]
    ds = [dur(p) for p in paths]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(names, off))
    for s, o, d in zip(names, off, ds):
        print(f"  {s:10s} {o:7.2f} + {d:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video.mp4")
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        args += ["-i", p]
    fc, last = [], "0:v"
    for k in range(1, len(names)):
        fc.append(f"[{last}][{k}:v]xfade=transition=fade:duration={XF}:offset={off[k]:.3f}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(fc), "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
             "-pix_fmt", "yuv420p", "-r", "30", vid]
    subprocess.run(args, check=True)
    # ---- music (the channel's synthesizer): calm under the exams, lifting for results and the second attempt
    sys.path.insert(0, os.path.join(ROOT, "flights"))
    import score as SC
    events = []
    for s in names:
        events += [(O[s] + t, k) for t, k in getattr(S[s], "sfx", [])]
    g = lambda k, d=0.0: O.get(k, d)
    energy = [(0, 0.25), (g("rules"), 0.35), (g("a_exam0"), 0.3), (g("a_curve"), 0.55), (g("a_score"), 0.62),
              (g("a_twist"), 0.3), (g("vector"), 0.35), (g("a_rescore"), 0.45), (g("b_title"), 0.4),
              (g("b_why"), 0.3), (g("c_title", total), 0.45), (g("c_exam", total), 0.4), (g("c_curve", total), 0.7),
              (g("c_score", total), 0.8), (g("c_clouds", total), 0.65), (g("n_title", total), 0.45),
              (g("n_exam", total), 0.4), (g("n_score", total), 0.75), (g("c_habits", total), 0.55),
              (g("outro"), 0.45), (total, 0.0)]
    energy = sorted(energy)
    impacts = [(g("a_curve"), 0.6)] + ([(g("c_curve"), 0.8), (g("n_score"), 0.6)] if "c_curve" in O else [])
    risers = [(g("a_curve") - 2.0, g("a_curve"))] + ([(g("c_curve") - 2.0, g("c_curve"))] if "c_curve" in O else [])
    sc = SC.Score(total, energy, impacts, risers, transpose_at=O.get("c_title"), final_at=g("outro") + 0.5,
                  intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.7
    mix = mix + sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "phoropter_ai.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], d] for s, d in zip(names, ds)}, "total": total},
              open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print("wrote", final)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        names, S = order()
        tot = 0
        for n in names:
            print(f"{n:10s} {S[n].dur:6.1f}s")
            tot += S[n].dur - XF
        print(f"total ~{tot + XF:.0f}s")
    elif cmd == "render":
        render(sys.argv[2])
    elif cmd == "all":
        render_all()
        assemble()
    elif cmd == "assemble":
        assemble()
