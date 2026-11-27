"""Version 6 (the fourth clinic video): the third video's AI after practice on env6 patients (a rougher spin,
occasional remarks, no smile, mixed reliability), against the same AI before that practice ("day one" in the new
clinic), the Borish script, and prescribing plano. Also the AI trained from nothing on env6 (runs/j), stopped.

* 5,000 new patients at typical reliability (dial 1): time, outcomes, points split into glasses and chair time.
* Consistency: the same 2,000 patients examined twice (fresh answers), at three settings of the reliability
  dial (0.5 reliable, 1 typical, 2 unreliable): how often the two prescriptions agree.
* Habits (which buttons, in what order), and each practice checkpoint on 2,000 patients.

    python evaluate6.py            -> data/results_m.json, data/seqs_m.json
"""
import os, json, glob, re
import numpy as np
from env6 import ClinicEnv6
from env5 import PlanoPairTestSet
from clinicians5 import run3, summary3, borish, plano
from evaluate3 import se_stats, sampled_outcomes
from evaluate4 import by_orientation
from evaluate5 import habits
from optics import to_pv
from policy import load
import clinic

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUN = os.path.join(HERE, "runs", "m")
SCRATCH = os.path.join(HERE, "runs", "j", "ckpt", "exams_904847.pt")
SEED = 2026
DIALS = (0.5, 1.0, 2.0)
EK = dict(env_cls=ClinicEnv6)


def ckpts():
    out = []
    for p in glob.glob(os.path.join(RUN, "ckpt", "exams_*.pt")):
        out.append((int(re.findall(r"exams_(\d+)", p)[0]), p))
    return sorted(out)


def final_ckpt():
    f = os.path.join(RUN, "ckpt", "final.pt")
    return f if os.path.exists(f) else ckpts()[-1][1]


def agree(res_a, res_b):
    """Two exams of the same patients (both eyes pooled): spherical equivalent, and the whole prescription
    (power-vector distance: sphere, cylinder and axis together)."""
    out = {}
    pa = [to_pv(res_a["S" + s], res_a["C" + s], res_a["ax" + s]) for s in ("", "_os")]
    pb = [to_pv(res_b["S" + s], res_b["C" + s], res_b["ax" + s]) for s in ("", "_os")]
    dM = np.concatenate([a[0] - b[0] for a, b in zip(pa, pb)])
    dv = np.concatenate([np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) for a, b in zip(pa, pb)])
    for tol in (0.25, 0.5):
        out[f"se_{int(tol * 100):03d}"] = float(np.mean(np.abs(dM) <= tol + 1e-9))
        out[f"rx_{int(tol * 100):03d}"] = float(np.mean(dv <= tol + 1e-9))
    out["se_sd"] = float(dM.std())
    return out


def axis_stats(res):
    """Eyes with at least 0.75 D of astigmatism: how far the prescribed axis is from the true one (no cylinder
    prescribed counts as 90° off)."""
    tC = np.abs(np.concatenate([res["tC"], res["tC_os"]]))
    C = np.abs(np.concatenate([res["C"], res["C_os"]]))
    d = np.abs(np.concatenate([res["ax"], res["ax_os"]]) - np.concatenate([res["tax"], res["tax_os"]])) % 180
    d = np.where(C > 0.01, np.minimum(d, 180 - d), 90.0)[tC >= 0.75]
    return dict(eyes=int(len(d)), median=float(np.median(d)), within_5=float(np.mean(d <= 5)),
                within_10=float(np.mean(d <= 10)), off_10=float(np.mean(d > 10)))


def entry_of(res, s):
    return dict(summary=s, outcomes=sampled_outcomes(res), orient=by_orientation(res), axis=axis_stats(res),
                complaints=float(res["complaints"].mean()), complaints_any=float((res["complaints"] > 0).mean()),
                walkouts=float(res["timeout"].mean()),
                split=dict(glasses=s["points"], time=-clinic.POINTS_PER_MIN * s["minutes"], net=s["net"]))


def main(n_test=5000, n_stage=2000, n_rep=2000):
    os.makedirs(DATA, exist_ok=True)
    R = {"meta": {"test_patients": n_test, "repeat_patients": n_rep, "seed": SEED, "start": "plano", "dials": DIALS}}
    seqs = {}
    final = final_ckpt()
    agent, meta = load(final)
    R["meta"].update(exams=meta["exams"], steps=meta["steps"], ckpt=os.path.basename(final))
    # "textbook" is the Borish script (the key keeps the shared scenes working)
    agent0, _ = load(os.path.join(RUN, "ckpt", "exams_0.pt"))          # the third video's AI, before practice here
    for name, kw in (("plano", dict(make=plano, complain=False)), ("textbook", dict(make=borish)),
                     ("day1", dict(agent=agent0)), ("ai", dict(agent=agent))):
        res = run3(test=PlanoPairTestSet(n_test, seed=SEED), n_par=1000, record_actions=name != "plano", dial=1.0,
                   **EK, **kw)
        s = summary3(res)
        s.update(se_stats(res))
        entry = entry_of(res, s)
        if name != "plano":
            entry["habits"] = habits(res["actions"])
            seqs[name] = res["actions"][:300]
        if name in ("textbook", "ai"):
            entry["by_dial"] = {}
            for dial in DIALS:
                r1 = run3(test=PlanoPairTestSet(n_rep, seed=SEED), n_par=1000, seed=7, dial=dial, **EK, **kw)
                r2 = run3(test=PlanoPairTestSet(n_rep, seed=SEED), n_par=1000, seed=8, dial=dial, **EK, **kw)
                s1 = summary3(r1)
                s1.update(se_stats(r1))
                entry["by_dial"][str(dial)] = dict(agree=agree(r1, r2), summary=s1,
                                                   walkouts=float(r1["timeout"].mean()))
                print(f"   {name} dial {dial}: agree {entry['by_dial'][str(dial)]['agree']}  net {s1['net']:+.1f} "
                      f"happy {s1['happy']:.2f} min {s1['minutes']:.2f}", flush=True)
            entry["repeat"] = dict(within_025=entry["by_dial"]["1.0"]["agree"]["se_025"],
                                   within_050=entry["by_dial"]["1.0"]["agree"]["se_050"])
        R[name] = entry
        print(name, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, entry["orient"], entry["axis"],
              flush=True)
        if name != "plano":
            h = entry["habits"]
            print("   used", {k: round(v, 2) for k, v in h["used_pct"].items()})
            print("   count", {k: round(v, 1) for k, v in h["mean_count"].items()})
            print("   minutes by stage", [round(x, 2) for x in h["stage_minutes"]], "first cyl step", h["first_cyl_step"])
            for p in range(3):
                print("   stage", p, h["top_patterns"][p][:4])
            print("   visit orders", h["visit_orders"][:5])
            for k, v in h["top_exams"][:4]:
                print(f"   {v:.2f}  {k}")
    ag_s, m_s = load(SCRATCH)                                           # from nothing, remarks only: stopped
    res = run3(agent=ag_s, test=PlanoPairTestSet(2000, seed=SEED), n_par=1000, record_actions=True, dial=1.0, **EK)
    s = summary3(res)
    s.update(se_stats(res))
    R["scratch"] = dict(entry_of(res, s), habits=habits(res["actions"]), exams=m_s["exams"])
    print("scratch", m_s["exams"], {k: round(v, 3) for k, v in s.items() if isinstance(v, float)},
          R["scratch"]["habits"]["visit_orders"][:3], flush=True)
    R["stages"] = []
    for ex_n, p in ckpts():
        ag, m = load(p)
        res = run3(agent=ag, test=PlanoPairTestSet(n_stage, seed=SEED), n_par=1000, greedy=False, record_actions=True,
                   dial=1.0, **EK)
        s = summary3(res)
        s.update(se_stats(res))
        h = habits(res["actions"])
        hab = dict(both_first=float(sum(v for k, v in h["visit_orders"] if k.startswith("B"))),
                   order=h["visit_orders"][0], used=h["used_pct"], count=h["mean_count"],
                   complaints=float(res["complaints"].mean()), walkouts=float(res["timeout"].mean()))
        R["stages"].append(dict(exams=ex_n, ckpt=os.path.basename(p), sampled=s, sampled_outcomes=sampled_outcomes(res),
                                habits=hab, axis=axis_stats(res)))
        print("stage", ex_n, "sampled net", round(s["net"], 1), "both first", round(hab["both_first"], 2),
              "spin", round(h["used_pct"]["spin"], 2), "jcc axis", round(h["used_pct"]["cyl axis (cross-cyl)"], 2),
              "cyl", round(h["used_pct"]["cylinder"], 2), "complaints", round(hab["complaints"], 2),
              "walkouts", round(hab["walkouts"], 2), "order", hab["order"], flush=True)
    json.dump(R, open(os.path.join(DATA, "results_m.json"), "w"), indent=1)
    json.dump(seqs, open(os.path.join(DATA, "seqs_m.json"), "w"))
    print("wrote data/results_m.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":            # a smoke test on a few patients
        main(n_test=200, n_stage=100, n_rep=200)
    else:
        main()
