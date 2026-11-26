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
                        flag=agent is not None and agent.flag)
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
