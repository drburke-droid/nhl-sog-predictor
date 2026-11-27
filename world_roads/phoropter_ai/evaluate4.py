"""The autorefractor experiment: every exam starts at the patient's autorefractor reading.

On 5,000 new simulated patients: the autorefractor reading alone (prescribe it as it is), the textbook script
from the autorefractor (no trial-sphere search), and the AI after practice term 4. Also repeatability, habits,
and astigmatism by orientation. Term-1/term-2 facts for the story are copied from data/results_f.json.

    python evaluate4.py          -> data/results_g.json, data/seqs_g.json
"""
import os, json
from functools import partial
import numpy as np
import env3 as E3
from env3 import ARPairTestSet
from clinicians3 import run3, summary3, textbook3, plano
from evaluate3 import habits, se_stats, repeat, sampled_outcomes
from policy import load

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUN4 = os.path.join(HERE, "runs", "f4")
SEED = 2026


def by_orientation(res):
    tax = np.concatenate([res["tax"], res["tax_os"]]) % 180
    tC = np.concatenate([res["tC"], res["tC_os"]])
    err = np.concatenate([res["err"], res["err_os"]])
    wtr = ((tax < 30) | (tax >= 150)) & (tC <= -0.75)
    atr = (tax >= 60) & (tax < 120) & (tC <= -0.75)
    obl = ~((tax < 30) | (tax >= 150)) & ~((tax >= 60) & (tax < 120)) & (tC <= -0.75)
    return dict(wtr_within=float(np.mean(err[wtr] <= 0.25 + 1e-9)), atr_within=float(np.mean(err[atr] <= 0.25 + 1e-9)),
                obl_within=float(np.mean(err[obl] <= 0.25 + 1e-9)))


def main(n_test=5000):
    R = {"meta": {"test_patients": n_test, "seed": SEED, "start": "autorefractor"}}
    F = json.load(open(os.path.join(DATA, "results_f.json")))
    for k in ("term1", "term2"):
        if k in F:
            R[k] = F[k]
    R["meta"].update(term1_end=F["meta"]["term1_end"], term2_end=F["meta"].get("term2_end", 2_697_295))
    R["meta"]["term3_end"] = 3_098_047
    final = os.path.join(RUN4, "ckpt", "final.pt")
    agent, meta = load(final)
    R["meta"].update(exams=meta["exams"], steps=meta["steps"], ckpt="final.pt")
    tb = partial(textbook3, objective=True)
    seqs = {}
    for name, kw in (("ar", dict(make=plano)), ("textbook", dict(make=tb)), ("ai", dict(agent=agent))):
        res = run3(test=ARPairTestSet(n_test, seed=SEED), n_par=1000, start="objective", record_actions=name != "ar", **kw)
        s = summary3(res)
        s.update(se_stats(res))
        entry = dict(summary=s, outcomes=sampled_outcomes(res), orient=by_orientation(res))
        if name != "ar":
            entry["habits"] = habits(res["actions"])
            seqs[name] = res["actions"][:300]
            acts = res["actions"]
            entry["jcc_axis"] = float(np.mean([sum(a == E3.JCC_AX for a in q) for q in acts]))
            entry["jcc_power"] = float(np.mean([sum(a == E3.JCC_PW for a in q) for q in acts]))
            r1 = run3(test=ARPairTestSet(2000, seed=SEED), n_par=1000, seed=7, start="objective", **kw)
            r2 = run3(test=ARPairTestSet(2000, seed=SEED), n_par=1000, seed=8, start="objective", **kw)
            entry["repeat"] = repeat(r1, r2)
        R[name] = entry
        print(name, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, entry["orient"], flush=True)
    json.dump(R, open(os.path.join(DATA, "results_g.json"), "w"), indent=1)
    json.dump(seqs, open(os.path.join(DATA, "seqs_g.json"), "w"))
    print("wrote data/results_g.json")


if __name__ == "__main__":
    main()
