"""Score the trained AI against the scripted examiners on 5,000 new simulated patients, and describe its strategy.

    python evaluate.py runs/a/ckpt/exams_1002464.pt a   -> data/results_a.json
    python evaluate.py ckpt/final.pt b                  -> data/results_b.json
"""
import os, sys, json, glob, re
from collections import Counter
import numpy as np
from patients import TestSet
from clinicians import run_scripted, textbook, textbook_letters, textbook_scratch, pleaser, summary
from policy import load, run_policy
from env import N_ACT, ACT_NAME, READ, CMP_P, CMP_M, JCC_AX, JCC_PW, DONE
from optics import to_pv

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 5000
SEED = 2026          # test patients are drawn independently of everything used in training


def by_age(res, edges=(8, 20, 40, 60, 81)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (res["age"] >= lo) & (res["age"] < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()), within=float(np.mean(res["err"][m] <= 0.25 + 1e-9)),
                        exact=float(np.mean(res["err"][m] < 0.13)), mean_dM=float(np.mean(res["dM"][m])),
                        over_minus=float(np.mean(res["dM"][m] <= -0.25 + 1e-9)),
                        over_plus=float(np.mean(res["dM"][m] >= 0.25 - 1e-9))))
    return out


def strategy(res, test):
    acts = res["actions"]
    L = max(len(a) for a in acts)
    NA = len(ACT_NAME)                                   # all 19 buttons (16 for attempts 1-2, never pressed above)
    heat = np.zeros((NA, L))
    for a in acts:
        for i, x in enumerate(a):
            heat[x, i] += 1
    alive = heat.sum(0)
    count = np.array([[a.count(k) for k in range(NA)] for a in acts])
    pool = test.pool
    young = pool["age"] < 40
    first = np.array([a[0] for a in acts])
    # net sphere the AI added to the autorefractor reading, by age (pushing plus = positive)
    dS = res["S"] - pool["S0"]
    return dict(
        heat=(heat / np.maximum(alive, 1)).round(4).tolist(), alive=alive.astype(int).tolist(),
        mean_count={ACT_NAME[k]: float(count[:, k].mean()) for k in range(NA)},
        used_pct={ACT_NAME[k]: float((count[:, k] > 0).mean()) for k in range(NA)},
        first_move={ACT_NAME[k]: float((first == k).mean()) for k in range(NA)},
        net_sphere_young=float(dS[young].mean()), net_sphere_old=float(dS[~young].mean()),
        autoref_dM_young=float(np.mean(pool["S0"][young] + pool["C0"][young] / 2
                                       - (pool["S"][young] + pool["C"][young] / 2))),
        questions=float(count[:, [10, 11, 12, 13, 14] + ([18] if NA > 18 else [])].sum(1).mean()),
        lens_moves=float(count[:, :10].sum(1).mean()),
        chart_moves=float(count[:, 16:18].sum(1).mean()) if NA > 17 else 0.0,
    )


def bias_split(res, test):
    b = test.pool["bias"]
    young = test.pool["age"] < 40
    hi, lo = young & (b > 0.13), young & (b < 0.07)
    return dict(over_minus_young_high_bias=float(np.mean(res["dM"][hi] <= -0.25 + 1e-9)),
                over_minus_young_low_bias=float(np.mean(res["dM"][lo] <= -0.25 + 1e-9)))


def err_vectors(res):
    """Per-patient error as a vector: (spherical-equivalent error, leftover cylinder as 2*dJ0, 2*dJ45), dioptres."""
    E = to_pv(res["tS"], res["tC"], res["tax"])
    L = to_pv(res["S"], res["C"], res["ax"])
    return np.stack([L[0] - E[0], 2 * (L[1] - E[1]), 2 * (L[2] - E[2])], 1).astype(np.float32)


def main(ck, name, start="autoref"):
    out = {"condition": start}
    t_max = 60 if start == "autoref" else 80
    agent, meta = load(ck)
    test = TestSet(N_TEST, seed=SEED, start=start)
    ai = run_policy(agent, test, n_par=1000, greedy=True, record_actions=True, t_max=t_max)
    vecs = {"ai": err_vectors(ai)}
    out["ai"] = dict(summary=summary(ai), summary_half=summary(ai, "err_half"), by_age=by_age(ai),
                     strategy=strategy(ai, test), meta=meta, bias=bias_split(ai, test))
    # scripts read the same kind of chart as the AI (whole lines for attempts 1-2, letters for attempt 3)
    tb = textbook_letters if agent.letters else textbook
    scripts = (("textbook", tb), ("pleaser", pleaser)) if start == "autoref" else (("textbook", textbook_scratch),)
    out["letters"] = bool(agent.letters)
    for nm, fn in scripts:
        t = TestSet(N_TEST, seed=SEED, start=start)
        r = run_scripted(fn, t, record_actions=True, t_max=t_max, letters=agent.letters)
        vecs[nm] = err_vectors(r)
        out[nm] = dict(summary=summary(r), summary_half=summary(r, "err_half"), by_age=by_age(r),
                       bias=bias_split(r, t), strategy=strategy(r, t))
        if nm == "textbook":
            tb_actions = r["actions"]
    # the autorefractor alone (prescribe the starting reading)
    from optics import pv_dist
    pool = test.pool
    E = to_pv(pool["S"], pool["C"], pool["axis"])
    L0 = to_pv(pool["S0"], pool["C0"], pool["ax0"])
    e0, e0h = ai["err0"], pv_dist(L0, E)
    dM0 = pool["S0"] + pool["C0"] / 2 - (pool["S"] + pool["C"] / 2)
    young = pool["age"] < 40
    auto = lambda e: dict(mean_err=float(e.mean()), within_025=float(np.mean(e <= 0.25 + 1e-9)),
                          exact=float(np.mean(e < 0.13)), steps=0.0,
                          over_minus_young=float(np.mean(dM0[young] <= -0.25 + 1e-9)),
                          over_plus=float(np.mean(dM0 >= 0.25 - 1e-9)))
    out["autoref" if start == "autoref" else "start"] = dict(summary=auto(e0), summary_half=auto(e0h))
    vecs["start"] = np.stack([L0[0] - E[0], 2 * (L0[1] - E[1]), 2 * (L0[2] - E[2])], 1).astype(np.float32)
    np.savez_compressed(os.path.join(HERE, "data", f"errors_{name}.npz"), **vecs)
    # every milestone checkpoint on the same patients (smaller set, for the stage cards)
    stages = []
    for p in sorted(glob.glob(os.path.join(os.path.dirname(ck), "exams_*.pt")), key=lambda s: int(re.findall(r"\d+", os.path.basename(s))[0])):
        a, m = load(p)
        r = run_policy(a, TestSet(2000, seed=SEED, start=start), n_par=1000, greedy=True, t_max=t_max)
        stages.append(dict(ckpt=os.path.basename(p), practice_exams=m.get("exams", 0), **summary(r),
                           half=summary(r, "err_half")))
        print(f"  {os.path.basename(p):22s} within {stages[-1]['within_025']:.3f} exact {stages[-1]['exact']:.3f} "
              f"steps {stages[-1]['steps']:.1f}", flush=True)
    out["stages"] = stages
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", f"results_{name}.json"), "w"), indent=1)
    seqs = {}
    for who, acts in (("ai", ai["actions"]), ("textbook", tb_actions)):
        c = Counter(tuple(x) for x in acts)
        seq, n = c.most_common(1)[0]
        seqs[who] = dict(seq=list(seq), n=n, distinct=len(c))
    json.dump(seqs, open(os.path.join(HERE, "data", f"seqs_{name}.json"), "w"))
    keys = [k for k in ("ai", "textbook", "pleaser", "autoref", "start") if k in out]
    for k in keys:
        h = out[k]["summary_half"]
        print(f"  [half-cyl score] {k:9s} within {h['within_025']:.3f} exact {h['exact']:.3f}")
    for k in keys:
        s = out[k]["summary"]
        print(f"{k:9s} within {s['within_025']:.3f} exact {s['exact']:.3f} mean err {s['mean_err']:.3f} "
              f"steps {s['steps']:.1f} over-minus(<40) {s['over_minus_young']:.3f} over-plus {s['over_plus']:.3f}")
    st = out["ai"]["strategy"]
    print("AI questions/exam", st["questions"], "lens moves", st["lens_moves"])
    print("used %:", {k: round(v, 2) for k, v in st["used_pct"].items()})
    print("first move:", {k: round(v, 2) for k, v in st["first_move"].items() if v > 0.01})
    print("net sphere young", st["net_sphere_young"], "old", st["net_sphere_old"], "autoref dM young", st["autoref_dM_young"])
    print("bias split AI", out["ai"]["bias"], "textbook", out["textbook"]["bias"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "autoref")
