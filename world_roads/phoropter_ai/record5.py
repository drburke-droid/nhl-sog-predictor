"""Record one clinic exam in the version-5 exam room (env5.py), move by move, for the video.

    python record5.py textbook | <checkpoint.pt>       print the showcase exam
"""
import sys, json
import numpy as np
import env5 as E5
from record import fixed_pair
from record3 import SHOWCASE3, line_cat
from optics import to_pv


def record5(agent=None, script=None, patient=SHOWCASE3, greedy=True, seed=3, max_steps=400):
    env = E5.ClinicEnv(1, seed=seed, sampler=fixed_pair(**patient), record=True)
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
        st = dict(a=int(a), name=E5.ACTS[a], code=int(L["code"][0]), strength=int(L["strength"][0]),
                  blurry=[bool(x) for x in L["blurry"][0]],
                  spin_ax=float(L["spin_ax"][0]), va=float(L["va"][0]), k=int(L["k"][0]), nl=int(L["nl"][0]),
                  line=float(L["line"][0]), screen=float(L["screen"][0]), phase=int(L["phase"][0]), phase_before=ph0,
                  screen_before=scr0, secs_before=secs0, secs=float(L["secs"][0]), before=before,
                  ai_probs=None if probs is None else [round(float(p), 4) for p in probs])
        if "probs" in L:
            st["ans_probs"] = [float(x) for x in L["probs"][0]]
        if "right" in L:
            st["right"] = int(L["right"][0])
        if a == E5.READ_LINE:
            st["cat"] = line_cat(st["k"], st["nl"])
        if done[0]:
            st["after"] = [(float(info["S"][0]), float(info["C"][0]), float(info["ax"][0])),
                           (float(info["S_os"][0]), float(info["C_os"][0]), float(info["ax_os"][0]))]
            st["err"] = [float(info["err"][0]), float(info["err_os"][0])]
            st["points"] = float(info["points"][0])
            st["walked"] = bool(info["timeout"][0])
            steps.append(st)
            break
        st["after"] = [(float(env.S[0, e]), float(env.C[0, e]), float(env.ax[0, e])) for e in (0, 1)]
        steps.append(st)
        if gen is not None:
            try:
                a = gen.send((st["code"], st["va"], st["k"], st["nl"], st["line"]))
            except StopIteration:
                a = E5.RX
    true = [[float(x) for x in to_pv(patient["S"], patient["C"], patient["axis"])],
            [float(x) for x in to_pv(patient["S_os"], patient["C_os"], patient["axis_os"])]]
    return dict(patient={k: float(v) for k, v in patient.items()}, true_pv=true, steps=steps, clinic=True)


def show(ex):
    for k, s in enumerate(ex["steps"]):
        ans = {-1: "", 0: "1", 1: "2", 2: "same", 3: f"reads {s['line']:.1f}", 4: f"{s['k']}/{s['nl']}",
               5: f"stop {s['spin_ax']:.0f}", 6: "no change", 7: "BLURRY right", 8: "BLURRY left",
               9: "BLURRY both"}[s["code"]]
        if s["code"] in (0, 1):
            ans += " (much)" if s["strength"] == 2 else " (a little)"
        (S0, C0, a0), (S1, C1, a1) = s["after"]
        print(f"{k + 1:3d} [{s['phase']}] {s['name']:38s} {ans:16s} OD {S0:+.2f} {C0:+.2f}x{a0:3.0f}  "
              f"OS {S1:+.2f} {C1:+.2f}x{a1:3.0f}  {s['secs'] / 60:4.1f}m")
    last = ex["steps"][-1]
    print("errors", [round(x, 3) for x in last["err"]], "points", round(last["points"], 1))


if __name__ == "__main__":
    if sys.argv[1] == "textbook":
        from clinicians5 import textbook3
        show(record5(script=textbook3))
    else:
        from policy import load
        ag, meta = load(sys.argv[1])
        show(record5(ag))
