"""How far past the patient's prescription the version-5 AI goes in its first both-eyes visit, by age.

It starts every exam with both eyes open and adds minus until the chart stops improving; it takes the extra
minus off at the end. Measured on 1,500 new patients at the moment it first covers an eye.

    python overshoot5.py        -> data/overshoot_h.json
"""
import os, json
import numpy as np
import env5 as E5
from env5 import ClinicEnv, PlanoPairTestSet
from policy import load

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = ((0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 90))


def main(n_test=1500, n_par=500):
    ag, meta = load(os.path.join(HERE, "runs", "h", "ckpt", "final.pt"))
    test = PlanoPairTestSet(n_test, seed=99)
    env = ClinicEnv(n_par, seed=3, sampler=test.sampler, record=False, start_mode="plano", rand_dial=True)
    obs = env.obs()
    nexts, rows, done_n = np.zeros(n_par, int), [], 0
    while done_n < n_test:
        acts = ag.act(obs, greedy=True)
        pid = env.P["pid"].copy()
        for i in np.where(acts == E5.NEXT)[0]:
            nexts[i] += 1
            if nexts[i] == 3 and pid[i] >= 0 and env.phase[i] == 2:     # leaving the first both-eyes visit
                lm = np.mean([env.lens_pv(e, i)[0] for e in (0, 1)])
                tm = np.mean([env.true_pv(e, i)[0] for e in (0, 1)])
                rows.append((float(env.P["age"][i]), float(tm - lm)))
        obs, _, done, info = env.step(acts)
        nexts[done] = 0
        if info:
            done_n += int((info["pid"] >= 0).sum())
    r = np.array(rows)
    out = dict(n=len(r), exams=meta["exams"], bands=[])
    for lo, hi in BANDS:
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        out["bands"].append(dict(lo=lo, hi=hi, n=int(m.sum()), mean=float(r[m, 1].mean()),
                                 median=float(np.median(r[m, 1]))))
        print(f"age {lo}-{hi}: n {m.sum():4d}  extra minus {r[m, 1].mean():+.2f} D (median {np.median(r[m, 1]):+.2f})")
    json.dump(out, open(os.path.join(HERE, "data", "overshoot_h.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
