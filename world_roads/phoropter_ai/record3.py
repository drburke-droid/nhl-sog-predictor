"""Record one clinic exam (env3.py) move by move, for the video: the button, the patient's answer and the
odds of every answer they could have given, the AI's odds for every button, both eyes' lenses, the stage,
the line on the projector, and the chair time.

    python record3.py <checkpoint.pt>          print the showcase exam
"""
import json
import numpy as np
from env3 import ClinicEnv, ACTS, RX, READ, READ_LINE, COMPARE, MASK
from optics import to_pv
from record import SHOWCASE2, fixed_pair
import clinic

# The showcase patient (chosen before any exam was filmed; the same 24-year-old as before, now with no
# autorefractor): OD -2.25 -1.00 x 170, OS -2.50 -0.75 x 010, thinks minus looks "crisper", some proximal
# accommodation behind the phoropter, right-eye dominant. The phoropter's axis dials were left at 90.
SHOWCASE3 = dict(SHOWCASE2, dial0=90.0, dial0_os=90.0)     # the axis dials left at 90 by the last patient


def line_cat(k, n):
    """all / most / a few / none, as in env3.ANSWERS['line']"""
    return 0 if k == n else (1 if 2 * k > n else (3 if k == 0 else 2))


def record3(agent=None, script=None, patient=SHOWCASE3, greedy=True, seed=3, max_steps=400, rand_dial=True):
    env = ClinicEnv(1, seed=seed, sampler=fixed_pair(**patient), record=True, rand_dial=rand_dial)
    obs = env.obs()
    gen = script(env, 0) if script else None
    a = next(gen) if gen else None
    steps = []
    for _ in range(max_steps):
        before = [(float(env.S[0, e]), float(env.C[0, e]), float(env.ax[0, e])) for e in (0, 1)]
        ph0, scr0, secs0 = int(env.phase[0]), float(env.line[0]), float(env.secs[0])
        probs = None
        if gen is None:
            probs = agent.probs(obs)[0]
            a = int(np.argmax(probs)) if greedy else int(np.random.default_rng(seed + len(steps)).choice(len(probs), p=probs))
        obs, r, done, info = env.step(np.array([a]))
        L = env.last
        st = dict(a=int(a), name=ACTS[a], code=int(L["code"][0]), va=float(L["va"][0]), k=int(L["k"][0]),
                  nl=int(L["nl"][0]), line=float(L["line"][0]), screen=float(L["screen"][0]), phase=int(L["phase"][0]),
                  phase_before=ph0, screen_before=scr0, secs_before=secs0, secs=float(L["secs"][0]), before=before,
                  ai_probs=None if probs is None else [round(float(p), 4) for p in probs], reward=float(r[0]))
        if "probs" in L:
            st["ans_probs"] = [float(x) for x in L["probs"][0]]
        if "right" in L:
            st["right"] = int(L["right"][0])
        if "read_dist" in L:
            st["read_dist"] = {f"{k_:.1f}": v for k_, v in L["read_dist"].items()}
        if a == READ_LINE:
            st["cat"] = line_cat(st["k"], st["nl"])
        if done[0]:
            st["after"] = [(float(info["S"][0]), float(info["C"][0]), float(info["ax"][0])),
                           (float(info["S_os"][0]), float(info["C_os"][0]), float(info["ax_os"][0]))]
            st["err"] = [float(info["err"][0]), float(info["err_os"][0])]
            st["e_pt"] = float(info["e_pt"][0])
            st["points"] = float(info["points"][0])
            st["outcome_p"] = [float(info[k][0]) for k in ("p_happy", "p_annoyed", "p_recheck", "p_review")]
            st["walked"] = bool(info["timeout"][0])
            steps.append(st)
            break
        st["after"] = [(float(env.S[0, e]), float(env.C[0, e]), float(env.ax[0, e])) for e in (0, 1)]
        steps.append(st)
        if gen is not None:
            try:
                a = gen.send((st["code"], st["va"], st["k"], st["nl"], st["line"]))
            except StopIteration:
                a = RX
    true = [[float(x) for x in to_pv(patient["S"], patient["C"], patient["axis"])],
            [float(x) for x in to_pv(patient["S_os"], patient["C_os"], patient["axis_os"])]]
    return dict(patient={k: float(v) for k, v in patient.items()}, true_pv=true, steps=steps, clinic=True)


def show(ex):
    for k, s in enumerate(ex["steps"]):
        ans = {-1: "", 0: "1", 1: "2", 2: "same", 3: f"reads {s['line']:.1f}", 4: f"{s['k']}/{s['nl']}"}[s["code"]]
        (S0, C0, a0), (S1, C1, a1) = s["after"]
        pr = "" if s["ai_probs"] is None else f" p={max(s['ai_probs']):.2f}"
        extra = ""
        if "ans_probs" in s:
            extra = " odds " + "/".join(f"{x:.2f}" for x in s["ans_probs"]) + (f" right={s['right']}" if "right" in s else "")
        print(f"{k + 1:3d} [{s['phase']}] {s['name']:30s} {ans:10s} OD {S0:+.2f} {C0:+.2f}x{a0:3.0f}  OS {S1:+.2f} {C1:+.2f}x{a1:3.0f}"
              f"  {s['secs'] / 60:4.1f}m{pr}{extra}")
    last = ex["steps"][-1]
    print("errors", [round(x, 3) for x in last["err"]], "points", round(last["points"], 1))


if __name__ == "__main__":
    import sys
    from policy import load
    if sys.argv[1] == "textbook":
        from clinicians3 import textbook3
        show(record3(script=textbook3))
    else:
        agent, meta = load(sys.argv[1])
        print(meta)
        show(record3(agent))
