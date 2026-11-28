"""The learning montage: the same simulated patient (the showcase patient: 24, short-sighted, astigmatism)
examined by the AI at checkpoints through its practice, each exam written as a row of step numbers (steps7.py),
with the patient's most likely reaction to the glasses and the score; and, for each checkpoint, what happened to
2,000 test patients (out of 100).

Practice ran in two parts: 1,812,949 patients who smiled when the letters got clearer (runs/h, env5), then
585,322 more realistic patients (runs/m, env6: no smile, a rough spin, occasional remarks). Practice exams are
sampled (the AI still trying things); the last row is the finished AI (its most likely button every time).

    python montage7.py        -> data/montage7.json
"""
import os, json
import numpy as np
import clinic
from record3 import SHOWCASE3
from record5 import record5
from env5 import PlanoPairTestSet
from env6 import ClinicEnv6
from policy import load
from steps7 import collapse

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = 1812949
OUT_NAMES = ("happy", "annoyed", "recheck", "review", "walked")


def patient_b():
    """A 56-year-old with long sight (+1.50 -0.50 x 178, +1.25 -0.50 x 178) from the test patients."""
    p = PlanoPairTestSet(5000, seed=2026).pool
    return {k: float(v[112]) for k, v in p.items() if k not in ("pid", "has_reading")}


PATIENTS = [("A", "24, short-sighted, astigmatism", dict(SHOWCASE3))]


def stages():
    H = json.load(open(os.path.join(HERE, "data", "results_h.json")))
    M = json.load(open(os.path.join(HERE, "data", "results_m.json")))
    hs = {s["exams"]: s for s in H["stages"]}
    ms = {s["exams"]: s for s in M["stages"]}
    out = []
    for ex in (0, 2551, 10114, 20200, 751332, 1000094, 1812949):
        out.append(dict(total=ex, run="h", ckpt=f"runs/h/ckpt/exams_{ex}.pt", env=None, greedy=False,
                        outcomes=hs[ex]["sampled_outcomes"], net=hs[ex]["sampled"]["net"]))
    for ex in (0, 100195):
        out.append(dict(total=PHASE1 + ex, run="m", ckpt=f"runs/m/ckpt/exams_{ex}.pt", env="env6", greedy=False,
                        outcomes=ms[ex]["sampled_outcomes"], net=ms[ex]["sampled"]["net"]))
    out.append(dict(total=PHASE1 + M["meta"]["exams"], run="m", ckpt="runs/m/ckpt/final.pt", env="env6", greedy=True,
                    outcomes=M["ai"]["outcomes"], net=M["ai"]["summary"]["net"], final=True))
    return out


def main():
    res = []
    for st in stages():
        ag, meta = load(os.path.join(HERE, st["ckpt"]))
        kw = dict(env_cls=ClinicEnv6, dial=1.0) if st["env"] == "env6" else {}
        rows = []
        for j, (tag, desc, pat) in enumerate(PATIENTS):
            tr = record5(ag, patient=pat, greedy=st["greedy"], seed=11 + j, max_steps=1000, **kw)
            last = tr["steps"][-1]
            acts = [s["a"] for s in tr["steps"]]
            mins = last["secs"] / 60
            if last.get("walked"):
                out, pts, probs = 4, clinic.WALKOUT, [0.0, 0.0, 0.0, 0.0]
            else:                                   # the most likely reaction to these glasses
                ept = clinic.patient_error(np.array([last["err"][0]]), np.array([last["err"][1]]))
                probs = [float(x) for x in clinic.outcome_probs(ept)[0]]
                out = int(np.argmax(probs))
                pts = float(clinic.POINTS[out])
            rows.append(dict(patient=tag, desc=desc, steps=collapse(acts), buttons=len(acts), minutes=mins,
                             outcome=OUT_NAMES[out], score=pts - clinic.POINTS_PER_MIN * mins,
                             complaints=sum(s["code"] in (7, 8, 9) for s in tr["steps"]), probs=probs,
                             err=[float(x) for x in last["err"]], rx=last["after"]))
            print(f"{st['total']:>9,} {tag}: {rows[-1]['buttons']:4d} buttons {mins:5.1f} min {rows[-1]['outcome']:8s} "
                  f"{rows[-1]['score']:+7.1f}  {' '.join(rows[-1]['steps'][:40])}", flush=True)
        o = np.array(st["outcomes"], float)
        res.append(dict(st, rows=rows, per100=[int(x) for x in np.round(o / o.sum() * 100)]))
    json.dump(res, open(os.path.join(HERE, "data", "montage7.json"), "w"), indent=1)
    print("wrote data/montage7.json")


if __name__ == "__main__":
    main()
