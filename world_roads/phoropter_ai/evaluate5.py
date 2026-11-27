"""Version 5 (no mentor, plano start): the AI that learned from the patients' answers, their smiles and the clinic's
points, against the textbook script and against prescribing plano, on 5,000 new simulated patients.

Also: repeatability, habits (which buttons, which steps, in what order), astigmatism by orientation, and each
practice checkpoint on 2,000 patients (for day one and the practice curve).

    python evaluate5.py            -> data/results_h.json, data/seqs_h.json
"""
import os, json, glob, re, collections
import numpy as np
import env5 as E5
from env5 import PlanoPairTestSet
from clinicians5 import run3, summary3, textbook3, plano
from evaluate3 import se_stats, repeat, sampled_outcomes
from evaluate4 import by_orientation
from policy import load
import sequence5 as SQ

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUN = os.path.join(HERE, "runs", "h")
SEED = 2026

GROUPS = {"sphere lens": (E5.SP25, E5.SM25, E5.SP100, E5.SM100, E5.OU_P, E5.OU_M, E5.R_P, E5.L_P),
          "sphere 1-or-2": (E5.CMP_P, E5.CMP_M),
          "chart reading": (E5.READ, E5.READ_LINE),
          "line size": (E5.LINE_SM, E5.LINE_LG),
          "cylinder": (E5.CYL_MORE, E5.CYL_LESS),
          "axis turns": (E5.AXP2, E5.AXM2, E5.AXP5, E5.AXM5, E5.AXP15, E5.AXM15),
          "astigmatism check": (E5.CHK_180, E5.CHK_45),
          "check 180/90": (E5.CHK_180,),
          "check 45/135": (E5.CHK_45,),
          "spin": (E5.SPIN,),
          "cyl axis (cross-cyl)": (E5.JCC_AX,),
          "cyl power (cross-cyl)": (E5.JCC_PW,),
          "balance": (E5.BALANCE,),
          "next eye": (E5.NEXT,)}


def kinds(seq):
    """An action sequence as a list of (stage, step kind), consecutive repeats merged (no other merge rules)."""
    out, ph, v = [], 0, 0
    for a in seq:
        if a == E5.NEXT:
            ph = (ph + 1) % 3
            v += 1
            continue
        k = SQ.kind_of(a, ph)
        if k is None:
            continue
        if not out or out[-1][:2] != (ph, k) or out[-1][2] != v:
            out.append((ph, k, v))
    return out


def habits(actions):
    out = {g: [] for g in GROUPS}
    stage_buttons, stage_secs = [[], [], []], [[], [], []]
    first_cyl = collections.Counter()
    patterns = [collections.Counter(), collections.Counter(), collections.Counter()]
    orders, full = collections.Counter(), collections.Counter()
    for seq in actions:
        for g, acts in GROUPS.items():
            out[g].append(sum(a in acts for a in seq))
        ph, nb, ns = 0, [0, 0, 0], [0.0, 0.0, 0.0]
        for a in seq:
            nb[ph] += 1
            ns[ph] += E5.SECS[a]
            if a == E5.NEXT:
                ph = (ph + 1) % 3
        for p in range(3):
            stage_buttons[p].append(nb[p])
            stage_secs[p].append(ns[p])
        ks = kinds(seq)
        # the first cylinder step of the right eye: how does it look for astigmatism?
        cyl = [k for p, k, _ in ks if p == 0 and k in ("chk", "axis", "power", "spin", "turn", "cyl")]
        first_cyl[cyl[0] if cyl else "none"] += 1
        for p in range(3):
            # the order of step kinds in this stage, repeats collapsed to their first appearance
            seen = []
            for pp, k, _ in ks:
                if pp == p and k not in seen:
                    seen.append(k)
            patterns[p][" > ".join(seen)] += 1
        # the order of occluder visits that had any work in them (R, L, B), and the whole exam as kinds per visit
        vis = []
        for pp, k, v in ks:
            if not vis or vis[-1][0] != v:
                vis.append((v, "RLB"[pp], []))
            if k not in vis[-1][2]:
                vis[-1][2].append(k)
        orders[" ".join(x[1] for x in vis)] += 1
        full[" | ".join(f"{x[1]}: " + " > ".join(x[2]) for x in vis)] += 1
    n = len(actions)
    return dict(mean_count={g: float(np.mean(v)) for g, v in out.items()},
                used_pct={g: float(np.mean(np.array(v) > 0)) for g, v in out.items()},
                stage_buttons=[float(np.mean(v)) for v in stage_buttons],
                stage_minutes=[float(np.mean(v)) / 60 for v in stage_secs],
                first_cyl_step={k: v / n for k, v in first_cyl.most_common()},
                visit_orders=[(k, v / n) for k, v in orders.most_common(8)],
                top_exams=[(k, v / n) for k, v in full.most_common(8)],
                top_patterns=[[(k, v / n) for k, v in patterns[p].most_common(8)] for p in range(3)])


def ckpts():
    out = []
    for p in glob.glob(os.path.join(RUN, "ckpt", "exams_*.pt")):
        out.append((int(re.findall(r"exams_(\d+)", p)[0]), p))
    return sorted(out)


def main(n_test=5000, n_stage=2000):
    os.makedirs(DATA, exist_ok=True)
    R = {"meta": {"test_patients": n_test, "seed": SEED, "start": "plano"}}
    seqs = {}
    final = os.path.join(RUN, "ckpt", "final.pt")
    if not os.path.exists(final):
        final = ckpts()[-1][1]
    agent, meta = load(final)
    R["meta"].update(exams=meta["exams"], steps=meta["steps"], ckpt=os.path.basename(final))
    # plano: prescribing plano for everyone, with the final check switched off (otherwise blurry patients refuse it
    # and every such exam ends in a walkout): the do-nothing reference
    for name, kw in (("plano", dict(make=plano, complain=False)), ("textbook", dict(make=textbook3)),
                     ("ai", dict(agent=agent))):
        res = run3(test=PlanoPairTestSet(n_test, seed=SEED), n_par=1000, record_actions=name != "plano", **kw)
        s = summary3(res)
        s.update(se_stats(res))
        entry = dict(summary=s, outcomes=sampled_outcomes(res), orient=by_orientation(res),
                     complaints=float(res["complaints"].mean()), complaints_any=float((res["complaints"] > 0).mean()),
                     walkouts=float(res["timeout"].mean()))
        if name != "plano":
            entry["habits"] = habits(res["actions"])
            seqs[name] = res["actions"][:300]
            r1 = run3(test=PlanoPairTestSet(2000, seed=SEED), n_par=1000, seed=7, **kw)
            r2 = run3(test=PlanoPairTestSet(2000, seed=SEED), n_par=1000, seed=8, **kw)
            entry["repeat"] = repeat(r1, r2)
        R[name] = entry
        print(name, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, entry["orient"], flush=True)
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
    R["stages"] = []
    for ex_n, p in ckpts():
        ag, m = load(p)
        res = run3(agent=ag, test=PlanoPairTestSet(n_stage, seed=SEED), n_par=1000, greedy=False)
        s = summary3(res)
        s.update(se_stats(res))
        R["stages"].append(dict(exams=ex_n, ckpt=os.path.basename(p), sampled=s, sampled_outcomes=sampled_outcomes(res)))
        print("stage", ex_n, "sampled net", round(s["net"], 1), flush=True)
    json.dump(R, open(os.path.join(DATA, "results_h.json"), "w"), indent=1)
    json.dump(seqs, open(os.path.join(DATA, "seqs_h.json"), "w"))
    print("wrote data/results_h.json")


if __name__ == "__main__":
    main()
