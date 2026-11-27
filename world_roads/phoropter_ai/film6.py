"""The clinic film, version 4 (env6): one AI, plano start, no mentor and no smile, a rougher spin and patients
who only now and then remark on what they see. Its sequence against the Borish method's; consistency on the
same patients with a reliability dial; where the points come from.

    python film6.py list | render <name> | frame <name> <t> | all | assemble     -> out/ai_optometry_v4.mp4
"""
import os, sys, json, math, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import film5 as F5                                   # (also installs the version-5 buttons in the shared scenes)
from film5 import (Toolbox5, PatientSays5, Wants5, Learn5, Practice5, TextbookSteps, Sequences, Results5, Hook3, Say,
                   F3, CU, SQ, E5, ui, clinic, mmss, fmt_n, AI_COL, TB_COL, PL_COL, XF, OUT)
from ui import W, H, BG, INK, INK2, MUTED, GOLD, GREEN, RED, CARD, LINE, PANEL, text, rr, fade, smooth
from scenes import Section, wrap, big_title
from scenes5 import Exam5

DATA = os.path.join(HERE, "data")
TR = os.path.join(DATA, "traces")
RUN = os.path.join(HERE, "runs", "j")
BORISH = "Borish method"
DIAL_NAMES = {"0.5": "RELIABLE", "1.0": "TYPICAL", "2.0": "UNRELIABLE"}


def trace(name):
    return json.load(open(os.path.join(TR, f"{name}.json")))


# ======================================================================== explainers
class PatientSays6(PatientSays5):
    ANSWERS = [("“One, much clearer.”  “Two… a little.”  “They look the same.”", CU.SPHERE),
               ("“I can read down to 20/40.”", CU.CHART),
               ("“Stop! …there, I think.”  (as the axis spins: a rough answer)", CU.AXIS),
               ("“My left eye is still blurry.”  (the final check)", CU.RXC),
               ("Now and then: “Oh, that’s clear!”  “Hmm, not as clear.”", INK2)]
    GOLD_Y = 930
    T_INTRO = 10.0


class Learn6(Learn5):
    LAST = "Now and then the patient says “Oh, that’s clear!”, and won't leave seeing blurry. That's all the help it gets."


class BorishSteps(TextbookSteps):
    TITLE = "The Borish method"
    SUB = "The standard sequence taught in optometry school. Written as a script: same buttons, same patients as the AI."
    STAGES = [("RIGHT EYE  ·  left eye covered", [
                  ("1", "sph_big", "Find the sphere: trial lenses, then 1.00 D steps, reading the chart"),
                  ("2", "fog", "Best sphere: fog with plus until the letters blur, then off a quarter at a time"),
                  ("3", "chk", "Astigmatism check: cross-cylinder at 180/90, then at 45/135"),
                  ("4", "axis", "Cylinder AXIS: flip either side of the axis, turn toward the clearer one"),
                  ("5", "power", "Cylinder POWER: flip along the axis, more or less  (small cylinder: power first)"),
                  ("6", "sph_fine", "Best sphere again, then “+0.25: 1 or 2?”  Keep the plus if it's no worse")]),
              ("LEFT EYE  ·  right eye covered", [("", "sph_fine", "Steps 1 to 6 again")]),
              ("BOTH EYES OPEN", [("7", "balance", "Fog both eyes, split the chart with prism, make the two eyes equal"),
                                  ("8", "ou_clear", "Clear the fog together, then “+0.25: 1 or 2?” once more"),
                                  ("", "rx_no", "Final check “still blurry”? Redo that eye, then both eyes again")])]


class Sequences6(Sequences):
    TB_NAME = "THE BORISH METHOD"
    TITLE = "The same patient: the Borish method's steps and the AI's"


class Results6(Results5):
    NAMES = ("Day one", BORISH, "AI")


# ======================================================================== where the points come from
class PointsSplit(Section):
    """Points per patient, split: what the glasses earn (happy, annoyed, re-checks, reviews, walkouts) and what
    the chair time costs (4 a minute). The net is the difference."""

    def __init__(self, R, dur=20.0):
        self.R, self.dur = R, dur
        self.rows = [("Day one", R["day1"], PL_COL), (BORISH, R["textbook"], TB_COL), ("AI", R["ai"], AI_COL)]

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        text(d, (W / 2, 40), f"{fmt_n(self.R['meta']['test_patients'])} NEW PATIENTS  ·  POINTS PER PATIENT", 24, MUTED,
             "Bold", "ma", a)
        big_title(d, W / 2, 100, "Where the points come from", 56, fade(t, 0.3, 999, 0.5, 0.1) * a)
        X0, X1, lo, hi = 700, 1500, -180, 100
        px = lambda v: X0 + (X1 - X0) * (v - lo) / (hi - lo)
        z = px(0)
        d.line([(z, 230), (z, 860)], fill=MUTED + (int(160 * a),), width=2)
        for j, (name, e, col) in enumerate(self.rows):
            y = 250 + j * 210
            al = fade(t, 1.0 + 2.5 * j, 999, 0.5, 0.1) * a
            s = e["summary"]
            glasses, time_ = s["points"], -clinic.POINTS_PER_MIN * s["minutes"]
            g1 = smooth((t - 1.4 - 2.5 * j) / 1.0)
            g2 = smooth((t - 2.4 - 2.5 * j) / 1.0)
            text(d, (X0 - 60, y + 70), name, 40, col, "Bold", "rm", al)
            # the glasses
            xg = px(glasses * g1)
            gcol = GREEN if glasses >= 0 else (176, 122, 122)          # walkouts and bad reviews: below zero
            d.rectangle([min(z, xg), y, max(z, xg), y + 56], fill=gcol + (int(220 * al),))
            lab = f"glasses {glasses:+.0f}"
            if glasses >= 0:
                text(d, (max(z, xg) + 14, y + 28), lab, 28, INK, "SemiBold", "lm", al * min(1, g1 * 2))
            else:
                text(d, (min(z, xg) - 14, y + 28), lab, 28, INK, "SemiBold", "rm", al * min(1, g1 * 2))
            # the chair time
            xt = px(time_ * g2)
            d.rectangle([xt, y + 70, z, y + 126], fill=RED + (int(220 * al),))
            text(d, (xt - 14, y + 98), f"{mmss(s['minutes'])} in the chair  {time_:+.0f}", 28, INK, "SemiBold", "rm",
                 al * min(1, g2 * 2))
            text(d, (1840, y + 63), f"= {s['net']:+.0f}", 56, INK, "Bold", "rm", al * min(1, g2 * 2))
        tb = self.R["textbook"]["summary"]
        tt = -clinic.POINTS_PER_MIN * tb["minutes"]
        if tb["net"] < 0:
            s_ = (f"The {BORISH}'s glasses earn {tb['points']:+.0f} a patient. Its {mmss(tb['minutes'])} in the chair "
                  f"costs {abs(tt):.0f}: that's why it ends up below zero.")
        else:
            s_ = (f"The {BORISH}'s glasses earn {tb['points']:+.0f} a patient, but {mmss(tb['minutes'])} in the chair "
                  f"costs {abs(tt):.0f} of that.")
        al = fade(t, 9.5, 999, 0.6, 0.1) * a
        for k, ln in enumerate(wrap(d, s_ + " Every minute costs 4 points.", 32, 1600, "SemiBold")):
            text(d, (W / 2, 900 + 44 * k), ln, 32, GOLD, "SemiBold", "ma", al)
        return np.asarray(im)


# ======================================================================== consistency: the same patient twice
class Consistency(Section):
    """Each patient examined twice (fresh answers, same eyes): how often the two prescriptions agree within
    0.25 D (spherical equivalent, every eye), with the patients made more or less reliable."""

    def __init__(self, R, dur=26.0, key="se_025"):
        self.R, self.dur, self.key = R, dur, key
        self.dials = [k for k in ("0.5", "1.0", "2.0") if k in R["ai"]["by_dial"]]

    @staticmethod
    def dial_icon(d, cx, cy, r, frac, al):
        d.arc([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=INK2 + (int(255 * al),), width=5)
        ang = math.pi * (1 - frac)
        d.line([(cx, cy), (cx + 0.85 * r * math.cos(ang), cy - 0.85 * r * math.sin(ang))],
               fill=GOLD + (int(255 * al),), width=6)
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=GOLD + (int(255 * al),))

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        big_title(d, W / 2, 34, "Same patient, tested twice", 56, fade(t, 0.3, 999, 0.5, 0.1) * a)
        text(d, (W / 2, 112), "Each patient examined twice, fresh answers. Do the two prescriptions agree within 0.25 D?",
             28, INK2, "Medium", "ma", fade(t, 0.8, 999, 0.5, 0.1) * a)
        base, top = 860, 380
        for j, dk in enumerate(self.dials):
            cx = 420 + 540 * j
            al = fade(t, 2.0 + 4.0 * j, 999, 0.5, 0.1) * a
            self.dial_icon(d, cx, 250, 50, {"0.5": 0.15, "1.0": 0.5, "2.0": 0.85}[dk], al)
            text(d, (cx, 280), DIAL_NAMES.get(dk, dk) + " PATIENTS", 26, INK, "Bold", "ma", al)
            for k, (who, col) in enumerate(((BORISH, TB_COL), ("AI", AI_COL))):
                ent = self.R["textbook" if k == 0 else "ai"]["by_dial"][dk]["agree"]
                v = ent[self.key]
                g = smooth((t - 2.4 - 4.0 * j - 0.4 * k) / 1.2)
                x0 = cx - 190 + k * 200
                hgt = (base - top) * v * g
                d.rectangle([x0, base - hgt, x0 + 180, base], fill=tuple(col) + (int(255 * al),))
                text(d, (x0 + 90, base - hgt - 14), f"{100 * v:.0f}%", 44, INK, "Bold", "md", al * min(1, g * 2))
                text(d, (x0 + 90, base + 14), who if k else "Borish", 26, col, "SemiBold", "ma", al)
        al = fade(t, 15.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 940, "Real optometrists, real patients: about 80% (published studies).", 32, al, INK2,
                  "Medium")
        al = fade(t, 18.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 996, "Consistent isn't the same as right (plano for everyone would agree every time): "
                                 "the happy patients say whether it's right.", 30, al, GOLD, "SemiBold")
        return np.asarray(im)


class Real6(Section):
    def __init__(self, R, dur=30.0):
        self.R, self.dur = R, dur

    def frame(self, t):
        im, d = ui.new_frame()
        a = fade(t, 0, self.dur, 0.5, 0.5)
        R = self.R
        big_title(d, W / 2, 46, "And real optometrists?", 54, fade(t, 0.3, 999, 0.5, 0.1) * a)
        items = [("Two refractions of the same eye agree within 0.25 D (typical patients)",
                  [("Real optometrists", "about 80%"), ("Borish script", f"{100 * R['textbook']['repeat']['within_025']:.0f}%"),
                   ("AI", f"{100 * R['ai']['repeat']['within_025']:.0f}%")]),
                 ("A subjective refraction takes",
                  [("Real optometrists", "about 5 min"), ("Borish script", mmss(R["textbook"]["summary"]["minutes"])),
                   ("AI", mmss(R["ai"]["summary"]["minutes"]))])]
        y = 170
        for i, (h_, vals) in enumerate(items):
            al = fade(t, 1.0 + 6.0 * i, 999, 0.6, 0.1) * a
            text(d, (W / 2, y), h_, 36, INK, "Bold", "ma", al)
            y += 70
            for j, (who, v) in enumerate(vals):
                x = 360 + j * 600
                col = INK2 if j == 0 else (TB_COL if j == 1 else AI_COL)
                text(d, (x, y), who, 28, col, "SemiBold", "ma", al)
                text(d, (x, y + 44), v, 54, INK, "Bold", "ma", al)
            y += 210
        al = fade(t, 13.0, 999, 0.6, 0.1) * a
        big_title(d, W / 2, 700, "Sources: Goss & Grosvenor 1996; Shah et al. 2009; Carracedo et al. 2018; Otero et al. 2019.",
                  22, al, MUTED, "Medium")
        al = fade(t, 16.0, 999, 0.6, 0.1) * a
        for k, ln in enumerate(wrap(d, "The script reads the chart after almost every click, so it is slower than an "
                                       "optometrist. Our patients are simulated and our scorecard is made up: compare "
                                       "the AI with the script, not with your own optometrist.", 32, 1560, "SemiBold")):
            text(d, (W / 2, 790 + 44 * k), ln, 32, GOLD, "SemiBold", "ma", al)
        return np.asarray(im)


# ======================================================================== captions
def make_notes(R, ai, tb):
    """Captions, placed by finding the moments in the traces and the practice checkpoints."""
    def first(tr, cond, lo=0):
        st = tr["steps"]
        return next((k for k in range(lo, len(st)) if cond(st[k])), None)

    def nth_next(tr, n):
        ks = [k for k, s in enumerate(tr["steps"]) if s["a"] == E5.NEXT]
        return ks[n] if len(ks) > n else None

    N = {}
    # ---- the Borish exam
    t_trial = first(tb, lambda s: s["a"] == E5.SM100)
    t_fog = first(tb, lambda s: s["a"] == E5.SP25)
    t_chk = first(tb, lambda s: s["a"] in (E5.CHK_180, E5.CHK_45))
    t_pw = first(tb, lambda s: s["a"] == E5.JCC_PW)
    t_ax = first(tb, lambda s: s["a"] == E5.JCC_AX)
    t_push = first(tb, lambda s: s["a"] == E5.CMP_P)
    t_l, t_b = nth_next(tb, 0), nth_next(tb, 1)
    t_bal = first(tb, lambda s: s["a"] == E5.BALANCE)
    t_clear = first(tb, lambda s: s["a"] == E5.OU_M, t_bal or 0)
    N["tb_exam"] = {k: v for k, v in {
        0: "Right eye first, left eye covered. “Read the smallest line you can.”",
        t_trial: "Trial lenses: −2.00, then +2.00. Then a whole dioptre at a time, reading the chart.",
        t_fog: "Best sphere: plus until the letters blur, then minus a quarter at a time, reading the chart each time.",
        t_chk: "Astigmatism check: the cross-cylinder at 180° or 90°?",
        t_pw: "Cylinder POWER: more or less? (Small cylinder: power first.)",
        t_ax: "Cylinder AXIS: which way? It turns toward the clearer side.",
        t_push: "“+0.25: 1 or 2?” If the extra plus is no worse, keep it: no more minus than the eye needs.",
        t_l: "The left eye, right eye covered: all of it again.",
        t_b: "Both eyes open: fog both, split the chart with prism, make the two eyes equal…",
        t_clear: "…then clear the fog together, and the final check."}.items() if k is not None}
    N["tb_slow"] = tuple(range(0, 4))
    N["ai_exam"], N["ai_slow"], N["sequences"] = {0: "Its first patient again."}, tuple(range(0, 6)), []
    N["curve"] = [(1000, "Day one: random buttons. Almost every patient walks out.")]
    return N


# ======================================================================== the film
def sections():
    R = json.load(open(os.path.join(DATA, "results_i.json")))
    S = {}
    S["hook"] = Hook3()
    S["toolbox"] = Toolbox5()
    S["patient"] = PatientSays6()
    S["wants"] = Wants5()
    S["learn"] = Learn6()
    S["day1_title"] = Say([(0.5, "Day one", 72, INK, "Bold"),
                           (1.6, "Its first patient. It has never seen an eye.", 38, INK2, "Medium")], 6.0)
    d1 = trace("i_day1")
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
    refs = [(BORISH, R["textbook"]["summary"]["net"], TB_COL)]
    ai, tb = trace("i_ai"), trace("i_borish")
    NOTES = make_notes(R, ai, tb)
    S["curve"] = Practice5(refs, NOTES.get("curve", []), max_exams=R["meta"]["exams"], run=RUN)
    S["ai_title"] = Say([(0.4, f"After {R['meta']['exams'] / 1e6:.1f} million patients", 60, INK, "Bold"),
                         (1.6, "The same first patient, again.", 36, INK2, "Medium")], 6.0)
    S["ai_exam"] = Exam5(ai, "THE AI", f"after {fmt_n(R['meta']['exams'])} practice patients",
                         notes=NOTES.get("ai_exam", {}), slow=NOTES.get("ai_slow", ()))
    S["tb_card"] = BorishSteps()
    S["tb_exam"] = Exam5(tb, "THE BORISH METHOD", "the same patient", who="BORISH", who_col=TB_COL,
                         notes=NOTES.get("tb_exam", {}), slow=NOTES.get("tb_slow", ()), fast_beat=0.2)
    S["sequences"] = Sequences6(tb, ai, NOTES.get("sequences", []))
    Rr = dict(R, day1=dict(summary=R["stages"][0]["sampled"]))
    S["results"] = Results6(Rr, [
        ("Time in the chair", "per patient", lambda e: e["summary"]["minutes"], mmss, 0, 16),
        ("Happy patients", "sees clearly, comes back, refers a friend", lambda e: e["summary"]["happy"],
         lambda v: f"{100 * v:.0f}%", 0, 1)])
    S["points"] = PointsSplit(Rr)
    S["consistency"] = Consistency(R)
    S["real"] = Real6(R)
    S["outro"] = F3.Card("The fine print", [
        (0.6, "Every patient is simulated. The optics are real; the answers are a model; the scorecard is made up.",
         30, INK2, "Medium"),
        (3.0, "The AI is reinforcement learning on a 4-core computer. It has never examined a real person. This is a "
              "demonstration, not a clinical tool.", 30, INK2, "Medium"),
        (5.6, "If your vision changes, see your optometrist.", 34, GOLD, "SemiBold")], 15.0, y0=230)
    return S


ORDER = ["hook", "toolbox", "patient", "wants", "learn", "day1_title", "day1_exam", "day1_grid", "curve", "ai_title",
         "ai_exam", "tb_card", "tb_exam", "sequences", "results", "points", "consistency", "real", "outro"]


def order():
    S = sections()
    return [n for n in ORDER if n in S], S


def render(name):
    _, S = order()
    S[name].render("i_" + name)


def render_all(names=None, par=4):
    names = names or order()[0]

    def one(nm):
        log = open(os.path.join(OUT, f"i_{nm}.log"), "w")
        r = subprocess.run([sys.executable, "film6.py", "render", nm], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
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
    paths = [os.path.join(OUT, f"i_{s}.mp4") for s in names]
    ds = [film.dur(p) for p in paths]
    off = [0.0]
    for dd in ds[:-1]:
        off.append(off[-1] + dd - XF)
    total = off[-1] + ds[-1]
    O = dict(zip(names, off))
    for s, o, dd in zip(names, off, ds):
        print(f"  {s:12s} {o:7.2f} + {dd:6.2f}")
    print(f"total {total:.2f} s ({int(total // 60)}:{total % 60:04.1f})", flush=True)
    vid = os.path.join(OUT, "_video6.mp4")
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
              (g("sequences"), 0.5), (g("results"), 0.8), (g("points"), 0.6), (g("consistency"), 0.6),
              (g("real"), 0.5), (g("outro"), 0.4), (total, 0.0)]
    sc = SC.Score(total, sorted(energy), [(g("curve"), 0.6), (g("results"), 0.8)],
                  [(g("ai_title") - 2.0, g("ai_title")), (g("results") - 2.0, g("results"))],
                  transpose_at=g("ai_title"), final_at=g("outro") + 0.5, intro_hit=False)
    sc.compose()
    mix = sc.mix() * 0.7
    mix = mix + film.sfx(mix.shape[1], SC.SR, events)
    wav = os.path.join(OUT, "score6.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "ai_optometry_v4.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    json.dump({"sections": {s: [O[s], dd] for s, dd in zip(names, ds)}, "total": total},
              open(os.path.join(OUT, "timeline6.json"), "w"), indent=1)
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
        Image.fromarray(S[sys.argv[2]].frame(float(sys.argv[3]))).save(os.path.join(OUT, f"frame6_{sys.argv[2]}.png"))
    elif cmd == "all":
        render_all()
        assemble()
    elif cmd == "assemble":
        assemble()
