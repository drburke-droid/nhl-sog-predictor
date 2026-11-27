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
             ("rx", "PRESCRIBE", "Done. Whatever is in the phoropter becomes the glasses.")]

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
    ANSWERS = [("“One, much clearer.”  “One… a little.”  “They look the same.”", CU.SPHERE),
               ("“I can read down to 20/40.”", CU.CHART), ("“Stop! There.”  (spinning the axis)", CU.AXIS),
               ("“The top one.”  “The bottom one.”", CU.PRISM)]


class Learn5(Learn4):
    LAST = "And the patient smiles when the letters get clearer. That's all the help it gets."


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
        self.v0, self.v1 = -160, 80
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
    STEPS = [("0", "No autorefractor: find the sphere in 1.00 D steps, reading the chart"),
             ("1", "Best sphere: fog with plus, then take it off a quarter at a time, reading the chart"),
             ("2", "Astigmatism check: cross-cylinder at 180/90, then at 45/135"),
             ("3", "Cylinder axis: cross-cylinder either side of the axis, turn the way the patient prefers"),
             ("4", "Cylinder power: cross-cylinder along the axis, more or less cylinder"),
             ("5", "Best sphere again"),
             ("6", "The other eye"),
             ("7", "Both eyes: fog, split the chart with prism, make the two eyes equal"),
             ("8", "Clear the fog on both eyes together. Prescribe.")]
    dur = 34.0

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 40, "What the textbook says", 54, fade(t, 0.3, 999, 0.5, 0.1) * a, TB_COL)
        text(d, (W / 2, 112), "The standard sequence (Borish), as a script: same buttons, same patients.", 26, MUTED,
             "Medium", "ma", a)
        y = 190
        for i, (n, s) in enumerate(self.STEPS):
            al = fade(t, 1.2 + 2.0 * i, 999, 0.5, 0.1) * a
            text(d, (240, y), n, 34, TB_COL, "Bold", "ra", al)
            text(d, (280, y), s, 33, INK, "Medium", a=al)
            y += 92
        return np.asarray(im)


# ======================================================================== the centrepiece: the two sequences
class Sequences(Section):
    """The textbook's steps and the AI's, stage by stage, for the same patient."""

    def __init__(self, tb, ai, notes, dur=None):
        self.tb, self.ai = SQ.steps_of(tb), SQ.steps_of(ai)
        self.tb_tr, self.ai_tr = tb, ai
        self.notes = notes
        self.stage_t = [1.5, 12.0, 21.0]               # when each stage appears
        self.dur = dur or 48.0

    def col(self, d, x, w, steps, tr, title, colr, t, rows_y):
        a_all = fade(t, 0, self.dur, 0.5, 0.5)
        last = tr["steps"][-1]
        text(d, (x, 120), title, 32, colr, "Bold", a=a_all)
        text(d, (x + w, 128), f"{len(tr['steps'])} buttons  ·  {mmss(last['secs'] / 60)}", 22, MUTED, "Medium", "ra", a_all)
        for si in range(3):
            items = [s for s in steps if s["stage"] == si]
            y = rows_y[si]
            al = fade(t, self.stage_t[si], 999, 0.5, 0.1) * a_all
            for j, st in enumerate(items):
                aj = fade(t, self.stage_t[si] + 0.35 * j, 999, 0.4, 0.1) * a_all
                col = CU.GROUP_COL[SQ.KIND_GROUP[st["kind"]]]
                rr(d, [x, y, x + w, y + 46], 8, fill=CARD + (int(255 * aj),))
                d.rectangle([x, y, x + 8, y + 46], fill=tuple(col) + (int(255 * aj),))
                text(d, (x + 22, y + 23), SQ.KIND_NAME[st["kind"]], 21, INK, "SemiBold", "lm", aj)
                text(d, (x + w - 14, y + 23), SQ.detail(st).split("→")[-1].strip(), 20, INK2, "Medium", "rm", aj)
                y += 52

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 30, "The same patient: the textbook's steps and the AI's", 44, a)
        # rows: each stage gets the height of its longer column
        n_tb = [sum(1 for s in self.tb if s["stage"] == i) for i in range(3)]
        n_ai = [sum(1 for s in self.ai if s["stage"] == i) for i in range(3)]
        need = [max(a_, b_) for a_, b_ in zip(n_tb, n_ai)]
        rows_y, y = [], 220
        avail = 1020 - 220 - 3 * 44
        scale = min(1.0, avail / max(1, sum(need) * 52))
        for si in range(3):
            al = fade(t, self.stage_t[si], 999, 0.5, 0.1) * a
            text(d, (W / 2, y + 2), SQ.STAGES[si], 22, GOLD, "Bold", "ma", al)
            rows_y.append(y + 36)
            y += 36 + need[si] * 52 * scale + 10
        self.col(d, 60, 860, self.tb, self.tb_tr, "THE TEXTBOOK", TB_COL, t, rows_y)
        self.col(d, 1000, 860, self.ai, self.ai_tr, "THE AI", AI_COL, t, rows_y)
        # commentary, one at a time, over the bottom
        cur = [s for t0, s in self.notes if t >= t0]
        if cur:
            t0 = max(t0 for t0, s in self.notes if t >= t0)
            al = fade(t, t0, 999, 0.5, 0.1) * a
            rr(d, [60, 960, 1860, 1060], 16, fill=(10, 14, 24, int(240 * al)), outline=GOLD + (int(255 * al),), width=2)
            lines = wrap(d, cur[-1], 28, 1720, "SemiBold")
            yy = 1010 - 18 * len(lines)
            for ln in lines:
                text(d, (W / 2, yy), ln, 28, GOLD, "SemiBold", "ma", al)
                yy += 36
        return np.asarray(im)


# ======================================================================== the film
NOTES = {}


def sections():
    R = json.load(open(os.path.join(DATA, "results_h.json")))
    S = {}
    S["hook"] = Hook3()
    S["toolbox"] = Toolbox5()
    S["patient"] = PatientSays5()
    S["wants"] = Wants4()
    S["learn"] = Learn5()
    S["day1_title"] = Say([(0.5, "Day one", 72, INK, "Bold"),
                           (1.6, "Its first patient. It has never seen an eye.", 38, INK2, "Medium")], 6.0)
    d1 = trace("h_day1")
    S["day1_exam"] = Exam5(d1, "DAY ONE", "before any practice", slow=range(len(d1["steps"])),
                           notes={0: "It has no idea what any button does. It presses them at random."}, outro=6.0)
    st0 = R["stages"][0]
    S["day1_grid"] = F3.OutcomeGrid("1,000 PATIENTS ON DAY ONE", "Random buttons, then “Prescribe”. One square per "
                                    "patient.", [("Day one", st0["sampled_outcomes"], st0["sampled"]["net"],
                                                  st0["sampled"]["minutes"])], dur=11.0,
                                    note="Most people need real glasses. Random ones aren't glasses.")
    refs = [("Plano for everyone", R["plano"]["summary"]["net"], PL_COL), ("Textbook", R["textbook"]["summary"]["net"], TB_COL)]
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
    Rr = dict(R, ar=R["plano"])
    res = Results4(Rr, [
        ("Time in the chair", "per patient", lambda e: e["summary"]["minutes"], mmss, 0, 10),
        ("Glasses within 0.25 D", "of the true prescription, every eye counted",
         lambda e: e["summary"]["vec_within_025"], lambda v: f"{100 * v:.0f}%", 0, 1),
        ("Happy patients", "sees clearly, comes back, refers a friend", lambda e: e["summary"]["happy"],
         lambda v: f"{100 * v:.0f}%", 0, 1),
        ("Points per patient", "glasses minus chair time", lambda e: e["summary"]["net"], lambda v: f"{v:+.0f}", -120,
         80)])
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
