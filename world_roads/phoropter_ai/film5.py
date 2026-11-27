"""The clinic film, version 3: one AI, plano start, no mentor. Can it figure out a refraction, and how does its
sequence compare with the textbook's?

    python film5.py list | render <name> | frame <name> <t> | all | assemble     -> out/ai_optometry_v3.mp4
"""
import os, sys, json, math, csv, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ui
import clinic
import clinic_ui as CUmain
import clinic_ui5 as CU
# every shared scene draws the version-5 buttons
CUmain.BUTTONS, CUmain.BTN, CUmain.col_of = CU.BUTTONS, CU.BTN, CU.col_of
import env5 as E5
from ui import W, H, BG, INK, INK2, MUTED, GOLD, GREEN, RED, CARD, LINE, PANEL, text, rr, fade, smooth, fmt_rx
from scenes import Section, para, wrap, big_title
from scenes5 import Exam5
from film3_intro import Hook3, fade_frame
from film4_intro import Toolbox4, PatientSays4, Wants4, Learn4
import film3 as F3
from film4 import Say, Results4, Real4, mmss, fmt_n
import sequence5 as SQ

OUT = ui.OUT
XF = 0.5
DATA = os.path.join(HERE, "data")
TR = os.path.join(DATA, "traces")
RUN = os.path.join(HERE, "runs", "h")
AI_COL, TB_COL, PL_COL = (0x33, 0x8C, 0xED), (0xE7, 0x66, 0x24), (0x8A, 0x96, 0xA8)


def trace(name):
    return json.load(open(os.path.join(TR, f"{name}.json")))


# ======================================================================== explainers, version 5
class Toolbox5(Toolbox4):
    PARTS = [("sphere", "SPHERE", "How strong the lens is. Minus for near-sighted eyes, plus for far-sighted ones."),
             ("cyl", "CYLINDER", "Extra power in one direction, for astigmatism."),
             ("axis", "AXIS", "Which direction that extra power points. Turn it, or spin it and let the patient say "
                              "stop."),
             ("jcc", "CROSS-CYLINDER", "A flip lens for three questions: is there astigmatism? Which way should the "
                                       "axis turn? More or less cylinder?"),
             ("prism", "PRISM", "Splits the chart into two copies, one per eye, to compare the eyes."),
             ("occ", "OCCLUDER", "Covers one eye: right eye first, then left, then both open."),
             ("chart", "THE CHART", "“Read the smallest line you can.” Bigger or smaller letters."),
             ("rx", "FINAL CHECK", "The patient reads with each eye. Both clear: these are the glasses. If not, "
                                   "they say which eye is still blurry.")]

    def frame(self, t):
        fr = super().frame(t)
        if t >= self.T0 + self.STEP * len(self.PARTS):
            im = Image.fromarray(fr)
            d = ImageDraw.Draw(im, "RGBA")
            d.rectangle([0, 800, W, 900], fill=BG)
            al = fade(t, self.T0 + self.STEP * len(self.PARTS), 999, 0.5, 0.1) * fade(t, 0, self.dur, 0.5, 0.5)
            big_title(d, W / 2, 840, "30 buttons. Nobody tells it what any of them do.", 46, al, GOLD, "SemiBold")
            return np.asarray(im)
        return fr


class PatientSays5(PatientSays4):
    ANSWERS = [("“One, much clearer.”  “Two… a little.”  “They look the same.”", CU.SPHERE),
               ("“I can read down to 20/40.”", CU.CHART), ("“Stop! There.”  (as the axis spins)", CU.AXIS),
               ("“My left eye is still blurry.”  (the final check)", CU.RXC)]
    GOLD_Y = 850


class Wants5(Wants4):
    """Wants4 plus the walkout: the patient who says it's still blurry stays in the chair, up to 15 minutes."""

    def frame(self, t):
        im, d = ui.new_frame()
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 40, "What the AI wants: points", 54, fade(t, 0.3, 999, 0.5, 0.1) * a_all)
        rows = [(o, name, desc, pts, CU.OUT_COL[o]) for o, name, desc, pts in self.ROWS]
        rows += [("min", "Every minute", "in the chair", "−4", CU.CHART),
                 ("walk", "Walks out", "still in the chair at 15 minutes, no glasses", "−150", (200, 70, 70))]
        y = 150
        for i, (o, name, desc, pts, col) in enumerate(rows):
            al = fade(t, 1.5 + 1.5 * i, 999, 0.5, 0.1) * a_all
            rr(d, [260, y, 1660, y + 100], 18, fill=CARD + (int(255 * al),), outline=tuple(col) + (int(255 * al),),
               width=3)
            text(d, (300, y + 50), name, 40, col, "Bold", "lm", al)
            text(d, (620, y + 50), desc, 30, INK2, "Medium", "lm", al)
            text(d, (1620, y + 50), pts, 52, col, "Bold", "rm", al)
            y += 114
        al = fade(t, 11.5, 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 870, "The closer the glasses, the more likely “happy”.", 40, al, INK, "SemiBold")
        al = fade(t, 15.5, 999, 0.6, 0.1) * a_all
        big_title(d, W / 2, 950, "Right glasses, happy patients, quick exams. Nobody tells it how.", 34, al, GOLD,
                  "SemiBold")
        return np.asarray(im)


class Learn5(Learn4):
    LAST = "The patient smiles when the letters get clearer, and won't leave seeing blurry. That's all the help it gets."



# ======================================================================== the practice curve, one run
class Practice5(F3.PracticeCurve):
    """Points per patient while it practised (glasses minus chair time), one caption at a time."""

    def __init__(self, refs, notes, dur=30.0, draw_secs=18.0):
        xs, ys, mins = [], [], []
        for r in csv.DictReader(open(os.path.join(RUN, "logs", "train.csv"))):
            if r.get("train_points"):
                xs.append(int(r["exams"]))
                ys.append(float(r["train_points"]) - clinic.POINTS_PER_MIN * float(r["train_minutes"]))
                mins.append(float(r["train_minutes"]))
        k = 9
        ker = np.ones(k) / k
        pad = lambda v: np.concatenate([np.full(k // 2, v[0]), v, np.full(k // 2, v[-1])])
        self.xs = np.array(xs)
        self.ys = np.convolve(pad(np.array(ys)), ker, "valid")
        self.mins = np.convolve(pad(np.array(mins)), ker, "valid")
        self.sh = np.zeros(len(xs))
        self.refs, self.notes, self.dur, self.draw_secs = refs, (), dur, draw_secs
        self.x_from = self.x_to = None
        self.title = "LEARNING ON THE JOB"
        self.X0, self.X1, self.Y0, self.Y1 = 230, 1500, 900, 220
        self.lx0, self.lx1 = 3.0, math.log10(self.xs.max()) + 0.08
        self.v0, self.v1 = -230, 90
        self.cap_notes = notes

    def frame(self, t):
        fr = super().frame(t)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a = fade(t, 0, self.dur, 0.5, 0.5)
        prog = smooth((t - 1.0) / self.draw_secs)
        cut = self.lx0 + (self.lx1 - self.lx0) * prog
        cur = [s for ex_n, s in self.cap_notes if cut >= math.log10(max(ex_n, 10 ** self.lx0))]
        if cur:
            big_title(d, W / 2, 985, cur[-1], 34, a, GOLD, "SemiBold")
        return np.asarray(im)


# ======================================================================== the textbook, step by step
class TextbookSteps(Section):
    """The textbook sequence from plano, grouped by what each eye is doing, coloured like the side-by-side."""
    STAGES = [("RIGHT EYE  ·  left eye covered", [
                  ("1", "sph_big", "Find the sphere: 1.00 D steps, reading the chart"),
                  ("2", "fog", "Best sphere: fog with plus, then take it off a quarter at a time"),
                  ("3", "chk", "Astigmatism check: cross-cylinder at 180/90, then at 45/135"),
                  ("4", "axis", "Cylinder AXIS: flip either side of the axis, turn toward the clearer one"),
                  ("5", "power", "Cylinder POWER: flip along the axis, more or less  (small cylinder: power first)"),
                  ("6", "sph_fine", "Best sphere again")]),
              ("LEFT EYE  ·  right eye covered", [("", "sph_fine", "Steps 1 to 6 again")]),
              ("BOTH EYES OPEN", [("7", "balance", "Fog both eyes, split the chart with prism, make the two eyes equal"),
                                  ("8", "ou_clear", "Clear the fog, both eyes together. Prescribe."),
                                  ("", "rx_no", "Final check “still blurry”? Redo that eye, then both eyes again")])]
    dur = 32.0

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 34, "What the textbook says", 54, fade(t, 0.3, 999, 0.5, 0.1) * a, TB_COL)
        text(d, (W / 2, 106), "The standard sequence, written as a script: same buttons, same patients as the AI.", 26,
             MUTED, "Medium", "ma", a)
        y, k = 170, 0
        for head, rows in self.STAGES:
            ah = fade(t, 1.0 + 1.6 * k, 999, 0.5, 0.1) * a
            text(d, (200, y), head, 26, GOLD, "Bold", a=ah)
            y += 48
            for n, kind, s in rows:
                al = fade(t, 1.0 + 1.6 * k, 999, 0.5, 0.1) * a
                col = CU.GROUP_COL[SQ.KIND_GROUP[kind]]
                rr(d, [200, y, 1720, y + 58], 10, fill=CARD + (int(255 * al),))
                d.rectangle([200, y, 210, y + 58], fill=tuple(col) + (int(255 * al),))
                text(d, (262, y + 29), n, 30, tuple(col), "Bold", "rm", al)
                text(d, (290, y + 29), s, 30, INK, "Medium", "lm", al)
                y += 68
                k += 1
            y += 18
        return np.asarray(im)


# ======================================================================== the centrepiece: the two sequences
class Sequences(Section):
    """The textbook's steps and the AI's for the same patient, each column a sequence of occluder visits (which eye
    is open, which covered), with the time spent on each visit. Rows scale to fit."""
    TOP, BOTTOM = 190, 944                            # rows live here; the commentary box sits below
    HEAD = 40

    def __init__(self, tb, ai, notes, dur=None, reveal=(1.5, 14.0)):
        self.tb_tr, self.ai_tr = tb, ai
        self.cols = [self.blocks(tb), self.blocks(ai)]
        self.notes = notes
        self.reveal = reveal                         # (textbook starts, AI starts)
        self.dur = dur or 50.0
        need = [sum(self.HEAD + 8 + len(sts) * 1.0 for _, sts, _ in c) for c in self.cols]
        n_rows = [sum(len(sts) for _, sts, _ in c) for c in self.cols]
        n_heads = [len(c) for c in self.cols]
        avail = self.BOTTOM - self.TOP
        self.rh = min(50.0, min((avail - nh * (self.HEAD + 8)) / max(1, nr) for nr, nh in zip(n_rows, n_heads)))

    @staticmethod
    def blocks(tr):
        """[(stage, steps, seconds)] per occluder visit, with the chair time of the visit."""
        st = SQ.steps_of(tr)
        vis = SQ.visits(st)
        # seconds per visit from the trace: count NEXT presses
        secs, v, prev = {}, 0, 0.0
        for s in tr["steps"]:
            secs[v] = secs.get(v, 0.0) + s["secs"] - prev
            prev = s["secs"]
            if s["a"] == E5.NEXT:
                v += 1
        out, vi = [], 0
        for stg, sts in vis:
            out.append((stg, sts, secs.get(sts[0]["visit"], 0.0)))
        return out

    def col(self, d, x, w, blocks, tr, title, colr, t, t0):
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        last = tr["steps"][-1]
        text(d, (x, 112), title, 32, colr, "Bold", a=a_all * fade(t, t0 - 0.8, 999, 0.5, 0.1))
        text(d, (x + w, 120), f"{len(tr['steps'])} buttons  ·  {mmss(last['secs'] / 60)} in the chair", 22, MUTED,
             "Medium", "ra", a_all * fade(t, t0 - 0.8, 999, 0.5, 0.1))
        rh = self.rh
        fs = int(max(16, min(21, rh * 0.44)))
        y, k = self.TOP, 0
        for stg, sts, secs in blocks:
            ah = fade(t, t0 + 0.45 * k, 999, 0.4, 0.1) * a_all
            eye_col = GOLD if stg < 2 else (120, 200, 255)
            text(d, (x, y + 6), SQ.STAGES[stg], 22, eye_col, "Bold", a=ah)
            text(d, (x + w, y + 6), mmss(secs / 60), 22, colr, "Bold", "ra", ah)
            d.line([x, y + self.HEAD - 4, x + w, y + self.HEAD - 4], fill=eye_col + (int(110 * ah),), width=1)
            y += self.HEAD
            for st in sts:
                k += 1
                aj = fade(t, t0 + 0.45 * k, 999, 0.4, 0.1) * a_all
                col = CU.GROUP_COL[SQ.KIND_GROUP[st["kind"]]]
                rr(d, [x, y, x + w, y + rh - 5], 6, fill=CARD + (int(255 * aj),))
                d.rectangle([x, y, x + 7, y + rh - 5], fill=tuple(col) + (int(255 * aj),))
                text(d, (x + 20, y + (rh - 5) / 2), SQ.KIND_NAME[st["kind"]], fs, INK, "SemiBold", "lm", aj)
                text(d, (x + w - 12, y + (rh - 5) / 2), SQ.detail(st).split("→")[-1].strip(), fs - 1, INK2, "Medium",
                     "rm", aj)
                y += rh
            y += 8
            k += 1

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 26, "The same patient: the textbook's steps and the AI's", 44, a)
        self.col(d, 60, 860, self.cols[0], self.tb_tr, "THE TEXTBOOK", TB_COL, t, self.reveal[0])
        self.col(d, 1000, 860, self.cols[1], self.ai_tr, "THE AI", AI_COL, t, self.reveal[1])
        cur = [(t0, s) for t0, s in self.notes if t >= t0]
        if cur:
            t0, s = cur[-1]
            al = fade(t, t0, 999, 0.5, 0.1) * a
            rr(d, [60, 958, 1860, 1062], 16, fill=(10, 14, 24, int(240 * al)), outline=GOLD + (int(255 * al),), width=2)
            lines = wrap(d, s, 28, 1720, "SemiBold")
            yy = 1010 - 18 * len(lines)
            for ln in lines:
                text(d, (W / 2, yy), ln, 28, GOLD, "SemiBold", "ma", al)
                yy += 36
        return np.asarray(im)


class Results5(Results4):
    """Results4 for the plano-start exam: day one, the textbook and the AI."""
    HEAD = "EVERY EXAM STARTS AT PLANO"

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        i = min(int(t // self.STEP), len(self.beats) - 1)
        tt = t - self.STEP * i
        title, sub, fn, fmt, lo, hi = self.beats[i]
        al = fade(tt, 0, self.STEP, 0.4, 0.4) * a if i < len(self.beats) - 1 else fade(tt, 0, 999, 0.4, 0.1) * a
        text(d, (W / 2, 40), f"{fmt_n(self.R['meta']['test_patients'])} NEW PATIENTS  ·  {self.HEAD}", 24, MUTED,
             "Bold", "ma", a)
        big_title(d, W / 2, 120, title, 56, al)
        text(d, (W / 2, 205), sub, 28, INK2, "Medium", "ma", al)
        rows = [("Day one", self.R["day1"], PL_COL), ("Textbook", self.R["textbook"], TB_COL),
                ("AI", self.R["ai"], AI_COL)]
        g = smooth((tt - 0.6) / 1.2)
        X0, X1 = 640, 1700
        for j, (name, e, col) in enumerate(rows):
            y = 330 + j * 190
            v = fn(e)
            text(d, (X0 - 40, y + 50), name, 40, INK, "SemiBold", "rm", al)
            vc = min(max(v, lo), hi)
            z = X0 + (X1 - X0) * (0 - lo) / (hi - lo)
            xv = X0 + (X1 - X0) * (vc * g - lo) / (hi - lo)
            if abs(xv - z) > 1:
                d.rectangle([min(z, xv), y, max(z, xv), y + 100], fill=col + (int(255 * al),))
            lab = fmt(v)
            if v >= 0 or lo >= 0:
                text(d, (max(z, xv) + 20, y + 50), lab, 52, INK, "Bold", "lm", al * min(1, g * 2))
            else:
                text(d, (min(z, xv) - 20, y + 50), lab, 52, INK, "Bold", "rm", al * min(1, g * 2))
        return np.asarray(im)


# ======================================================================== the film
NOTES = {}


def sections():
    R = json.load(open(os.path.join(DATA, "results_h.json")))
    S = {}
    S["hook"] = Hook3()
    S["toolbox"] = Toolbox5()
    S["patient"] = PatientSays5()
    S["wants"] = Wants5()
    S["learn"] = Learn5()
    S["day1_title"] = Say([(0.5, "Day one", 72, INK, "Bold"),
                           (1.6, "Its first patient. It has never seen an eye.", 38, INK2, "Medium")], 6.0)
    d1 = trace("h_day1")
    k1 = next((k for k, x in enumerate(d1["steps"]) if x["code"] in (7, 8, 9)), None)
    d1_notes = {0: "It has no idea what any button does. It presses them at random."}
    if k1 is not None:
        d1_notes[k1] = "It tries the final check. The patient can't see: the exam goes on."
    S["day1_exam"] = Exam5(d1, "DAY ONE", "before any practice", slow=range(8), notes=d1_notes, outro=6.0,
                           fast_beat=0.06)
    st0 = R["stages"][0]
    S["day1_grid"] = F3.OutcomeGrid("1,000 PATIENTS ON DAY ONE", "Random buttons. One square per patient.",
                                    [("Day one", st0["sampled_outcomes"], st0["sampled"]["net"],
                                      st0["sampled"]["minutes"])], dur=11.0,
                                    note="Random glasses never pass the final check. Almost everyone walks out.")
    refs = [("Textbook", R["textbook"]["summary"]["net"], TB_COL)]
    S["curve"] = Practice5(refs, NOTES.get("curve", []))
    ai, tb = trace("h_ai"), trace("h_textbook")
    S["ai_title"] = Say([(0.4, f"After {R['meta']['exams'] / 1e6:.1f} million patients", 60, INK, "Bold"),
                         (1.6, "The same first patient, again.", 36, INK2, "Medium")], 6.0)
    S["ai_exam"] = Exam5(ai, "THE AI", f"after {fmt_n(R['meta']['exams'])} practice patients",
                         notes=NOTES.get("ai_exam", {}), slow=NOTES.get("ai_slow", ()))
    S["tb_card"] = TextbookSteps()
    S["tb_exam"] = Exam5(tb, "THE TEXTBOOK", "the same patient", who="TEXTBOOK", who_col=TB_COL,
                         notes=NOTES.get("tb_exam", {}), slow=NOTES.get("tb_slow", ()), fast_beat=0.2)
    S["sequences"] = Sequences(tb, ai, NOTES.get("sequences", []))
    Rr = dict(R, day1=dict(summary=R["stages"][0]["sampled"]))
    res = Results5(Rr, [
        ("Time in the chair", "per patient", lambda e: e["summary"]["minutes"], mmss, 0, 16),
        ("Glasses within 0.25 D", "of the true prescription, every eye counted",
         lambda e: e["summary"]["vec_within_025"], lambda v: f"{100 * v:.0f}%", 0, 1),
        ("Happy patients", "sees clearly, comes back, refers a friend", lambda e: e["summary"]["happy"],
         lambda v: f"{100 * v:.0f}%", 0, 1),
        ("Points per patient", "glasses minus chair time", lambda e: e["summary"]["net"], lambda v: f"{v:+.0f}", -230,
         90)])
    S["results"] = res
    S["real"] = Real4(R)
    S["outro"] = F3.Card("The fine print", [
        (0.6, "Every patient is simulated. The optics are real; the answers are a model; the scorecard is made up.",
         30, INK2, "Medium"),
        (3.0, "The AI is reinforcement learning on a 4-core computer. It has never examined a real person. This is a "
              "demonstration, not a clinical tool.", 30, INK2, "Medium"),
        (5.6, "If your vision changes, see your optometrist.", 34, GOLD, "SemiBold")], 15.0, y0=230)
    return S


ORDER = ["hook", "toolbox", "patient", "wants", "learn", "day1_title", "day1_exam", "day1_grid", "curve", "ai_title",
         "ai_exam", "tb_card", "tb_exam", "sequences", "results", "real", "outro"]


def order():
    S = sections()
    return [n for n in ORDER if n in S], S


def render(name):
    _, S = order()
    S[name].render("h_" + name)


def render_all(names=None, par=4):
    names = names or order()[0]

    def one(nm):
        log = open(os.path.join(OUT, f"h_{nm}.log"), "w")
        r = subprocess.run([sys.executable, "film5.py", "render", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
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
    paths = [os.path.join(OUT, f"h_{s}.mp4") for s in names]
    ds = [film.dur(p) for p in paths]
    off = [0.0]
    for dd in ds[:-1]:
        off.append(off[-1] + dd - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(names, off))
    for s, o, dd in zip(names, off, ds):
        print(f"  {s:12s} {o:7.2f} + {dd:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video5.mp4")
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
              (g("sequences"), 0.5), (g("results"), 0.8), (g("real"), 0.5), (g("outro"), 0.4), (total, 0.0)]
    sc = SC.Score(total, sorted(energy), [(g("curve"), 0.6), (g("results"), 0.8)],
                  [(g("ai_title") - 2.0, g("ai_title")), (g("results") - 2.0, g("results"))],
                  transpose_at=g("ai_title"), final_at=g("outro") + 0.5, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.7
    mix = mix + film.sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score5.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "ai_optometry_v3.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], dd] for s, dd in zip(names, ds)}, "total": total},
              open(os.path.join(OUT, "timeline5.json"), "w"), indent=1)
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
        Image.fromarray(S[sys.argv[2]].frame(float(sys.argv[3]))).save(os.path.join(OUT, f"frame5_{sys.argv[2]}.png"))
    elif cmd == "all":
        render_all()
        assemble()
    elif cmd == "assemble":
        assemble()
