"""Record one exam move by move (for the video): the AI's actions, the patient's answers, and every lens shown."""
import json
import numpy as np
from env import RefractionEnv, DONE, READ, ACT_NAME
from patients import fixed_patient
from optics import to_pv

# The showcase patient, chosen before any exam was filmed: a 24-year-old myope with astigmatism, whose
# autorefractor reading has the extra minus typical of young eyes, and who thinks minus looks "crisper".
SHOWCASE = dict(age=24, S=-2.25, C=-1.00, axis=170, S0=-2.75, C0=-0.75, ax0=4, mar0=10 ** -0.08, pupil=4.6,
                amp=11.0, sigma=0.025, tau=0.03, bias=0.15, lapse=0.0)


def record(agent, patient=SHOWCASE, greedy=True, seed=3, max_steps=60, t_max=60):
    env = RefractionEnv(1, seed=seed, sampler=fixed_patient(**patient), record=True,
                        flag=agent is not None and agent.flag, letters=agent is not None and agent.letters)
    env.t_max = t_max
    obs = env.obs()
    true = env.true_pv()
    steps = []
    for _ in range(max_steps):
        lens_before = (float(env.S[0]), float(env.C[0]), float(env.ax[0]))
        a = int(agent.act(obs, greedy=greedy)[0]) if agent is not None else DONE
        probs = agent.probs(obs)[0].tolist() if agent is not None else None
        obs, r, done, info = env.step(np.array([a]))
        last = env.last
        st = dict(a=a, name=ACT_NAME[a], code=int(last["code"][0]), va=float(last["va"][0]),
                  k=int(last["k"][0]) if "k" in last else 0, nl=int(last["nl"][0]) if "nl" in last else 0,
                  line=float(last["line"][0]) if "line" in last else 0.0,
                  screen=float(last["screen"][0]) if "screen" in last else 0.0, letters=env.letters,
                  before=lens_before, A=[float(x[0]) for x in last["A"]], B=[float(x[0]) for x in last["B"]],
                  L=[float(x[0]) for x in last["L"]], probs=probs)
        if done[0]:
            st["after"] = (float(info["S"][0]), float(info["C"][0]), float(info["ax"][0]))
            st["err"] = float(info["err"][0])
            steps.append(st)
            break
        st["after"] = (float(env.S[0]), float(env.C[0]), float(env.ax[0]))
        steps.append(st)
    return dict(patient={k: float(v) for k, v in patient.items()}, true_pv=[float(x[0]) for x in true], steps=steps)


if __name__ == "__main__":
    import sys
    from policy import load
    agent, meta = load(sys.argv[1])
    ex = record(agent, greedy=len(sys.argv) < 3 or sys.argv[2] != "sample")
    for k, s in enumerate(ex["steps"]):
        ans = {-1: "", 0: "1", 1: "2", 2: "same", 3: f"logMAR {s['va']:.1f}"}[s["code"]]
        S, C, ax = s["after"]
        print(f"{k + 1:2d} {s['name']:18s} {ans:12s} -> {S:+.2f} {C:+.2f} x {ax:.0f}" + (f"  err {s['err']:.3f}" if "err" in s else ""))


# ---------------------------------------------------------------------------- two eyes (attempt 4)
# The same 24-year-old, now with both eyes: the left eye slightly more myopic, its reading also too minus,
# a fair amount of proximal accommodation behind the phoropter, right-eye dominant.
SHOWCASE2 = dict(SHOWCASE, S_os=-2.50, C_os=-0.75, axis_os=10, S0_os=-3.00, C0_os=-0.50, ax0_os=2,
                 mar0_os=10 ** -0.08, prox=0.35, dom=1.0)


def fixed_pair(**kw):
    from patients import sample_pair

    def fn(rng, n):
        base = sample_pair(rng, n, age=kw.get("age"))
        for k, v in kw.items():
            base[k] = np.full(n, float(v))
        return base
    return fn


def record2(agent=None, script=None, patient=SHOWCASE2, greedy=True, seed=3, t_max=120):
    """Record a two-eye exam by an AI (agent) or a scripted examiner (script generator function)."""
    from env2 import BinocularEnv, ACTS, DONE as DONE2
    env = BinocularEnv(1, seed=seed, sampler=fixed_pair(**patient), record=True)
    env.t_max = t_max
    obs = env.obs()
    gen = script(env, 0) if script else None
    a = next(gen) if gen else None
    steps = []
    for _ in range(t_max):
        before = [(float(env.S[0, e]), float(env.C[0, e]), float(env.ax[0, e])) for e in (0, 1)]
        cur0, bin0, scr0 = int(env.cur[0]), float(env.bin[0]), float(env.line[0])
        if gen is None:
            a = int(agent.act(obs, greedy=greedy)[0])
        obs, r, done, info = env.step(np.array([a]))
        L = env.last
        st = dict(a=int(a), name=ACTS[a], code=int(L["code"][0]), va=float(L["va"][0]), k=int(L["k"][0]),
                  nl=int(L["nl"][0]), line=float(L["line"][0]), screen=float(L["screen"][0]), cur=int(L["cur"][0]),
                  bin=float(L["bin"][0]), cur_before=cur0, bin_before=bin0, screen_before=scr0, before=before,
                  letters=True, two_eyes=True)
        if done[0]:
            st["after"] = [(float(info["S"][0]), float(info["C"][0]), float(info["ax"][0])),
                           (float(info["S_os"][0]), float(info["C_os"][0]), float(info["ax_os"][0]))]
            st["err"] = [float(info["err"][0]), float(info["err_os"][0])]
            steps.append(st)
            break
        st["after"] = [(float(env.S[0, e]), float(env.C[0, e]), float(env.ax[0, e])) for e in (0, 1)]
        steps.append(st)
        if gen is not None:
            try:
                a = gen.send((st["code"], st["va"], st["k"], st["nl"], st["line"]))
            except StopIteration:
                a = DONE2
    true = [[float(x) for x in to_pv(patient["S"], patient["C"], patient["axis"])],
            [float(x) for x in to_pv(patient["S_os"], patient["C_os"], patient["axis_os"])]]
    return dict(patient={k: float(v) for k, v in patient.items()}, true_pv=true, steps=steps, two_eyes=True)
