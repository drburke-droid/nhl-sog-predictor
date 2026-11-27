"""Evaluate the clinic AI (runs/f) against the textbook script and against prescribing plano, on 5,000 new
simulated patients (every exam starts at plano). Also: each practice checkpoint on 2,000 of them, how
repeatable each examiner is (the same patients again, with fresh answer noise), and the AI's habits.

    python evaluate3.py            -> data/results_f.json, data/seqs_f.json
"""
import os, json, glob, re
import numpy as np
import env3 as E3
from env3 import PlanoPairTestSet
from clinicians3 import run3, summary3, textbook3, plano
from policy import load
from optics import to_pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUN = os.path.join(HERE, "runs", "f3")             # practice term 3 (the final AI)
RUN2 = os.path.join(HERE, "runs", "f2")            # practice term 2 (the dial where the last patient left it)
RUN1 = os.path.join(HERE, "runs", "f")             # practice term 1 (the axis dial always started at 180)
TERM1_END = 1_630_473
TERM2_END = 2_697_295
SEED = 2026

GROUPS = {"sphere lens": (E3.SP25, E3.SM25, E3.SP100, E3.SM100, E3.OU_P, E3.OU_M, E3.R_P, E3.L_P),
          "sphere 1-or-2": (E3.CMP_P, E3.CMP_M),
          "chart reading": (E3.READ, E3.READ_LINE),
          "line size": (E3.LINE_SM, E3.LINE_LG),
          "cylinder": (E3.CYL_MORE, E3.CYL_LESS),
          "axis": (E3.AXP2, E3.AXM2, E3.AXP5, E3.AXM5, E3.AXP15, E3.AXM15),
          "cross-cylinder": (E3.JCC_AX, E3.JCC_PW),
          "balance": (E3.BALANCE,),
          "next eye": (E3.NEXT,)}
SPH_WORK = set(GROUPS["sphere lens"] + GROUPS["sphere 1-or-2"] + GROUPS["chart reading"])
CYL_WORK = set(GROUPS["cylinder"] + GROUPS["axis"] + GROUPS["cross-cylinder"])


def habits(actions):
    """Per exam: uses of each group, buttons and minutes per stage, and how often sphere work and cylinder work
    alternate within one eye (the number of switches between them)."""
    out = {g: [] for g in GROUPS}
    stage_buttons = [[], [], []]
    stage_secs = [[], [], []]
    switches, reached = [], [[], [], []]
    for seq in actions:
        for g, acts in GROUPS.items():
            out[g].append(sum(a in acts for a in seq))
        ph, nb, ns = 0, [0, 0, 0], [0.0, 0.0, 0.0]
        sw, last = 0, None
        for a in seq:
            nb[ph] += 1
            ns[ph] += E3.SECS[a]
            if ph < 2:
                kind = "s" if a in SPH_WORK else ("c" if a in CYL_WORK else None)
                if kind is not None:
                    if last is not None and kind != last:
                        sw += 1
                    last = kind
            if a == E3.NEXT:
                ph = min(ph + 1, 2)
                last = None
        for p in range(3):
            stage_buttons[p].append(nb[p])
            stage_secs[p].append(ns[p])
            reached[p].append(ph >= p)
        switches.append(sw)
    return dict(mean_count={g: float(np.mean(v)) for g, v in out.items()},
                used_pct={g: float(np.mean(np.array(v) > 0)) for g, v in out.items()},
                stage_buttons=[float(np.mean(v)) for v in stage_buttons],
                stage_minutes=[float(np.mean(v)) / 60 for v in stage_secs],
                reached_stage=[float(np.mean(v)) for v in reached],
                sph_cyl_switches=float(np.mean(switches)),
                sph_cyl_switches_pct=[float(np.mean(np.array(switches) == k)) for k in range(8)])


def se_stats(res):
    dM = np.concatenate([res["dM"], res["dM_os"]])
    err = np.concatenate([res["err"], res["err_os"]])
    E = [to_pv(res["tS"], res["tC"], res["tax"]), to_pv(res["tS_os"], res["tC_os"], res["tax_os"])]
    F = [to_pv(res["S"], res["C"], res["ax"]), to_pv(res["S_os"], res["C_os"], res["ax_os"])]
    cyl = np.concatenate([2 * np.hypot(F[e][1] - E[e][1], F[e][2] - E[e][2]) for e in (0, 1)])
    return dict(se_within_025=float(np.mean(np.abs(dM) <= 0.25 + 1e-9)), se_within_050=float(np.mean(np.abs(dM) <= 0.5 + 1e-9)),
                cyl_within_025=float(np.mean(cyl <= 0.25 + 1e-9)), vec_within_025=float(np.mean(err <= 0.25 + 1e-9)),
                vec_within_050=float(np.mean(err <= 0.5 + 1e-9)), mean_dM=float(dM.mean()))


def repeat(res_a, res_b):
    """Agreement between two exams of the same patients (both eyes pooled), spherical equivalent."""
    Ma = np.concatenate([to_pv(res_a["S"], res_a["C"], res_a["ax"])[0], to_pv(res_a["S_os"], res_a["C_os"], res_a["ax_os"])[0]])
    Mb = np.concatenate([to_pv(res_b["S"], res_b["C"], res_b["ax"])[0], to_pv(res_b["S_os"], res_b["C_os"], res_b["ax_os"])[0]])
    d = Ma - Mb
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(within_025=float(np.mean(np.abs(d) <= 0.25 + 1e-9)), within_050=float(np.mean(np.abs(d) <= 0.5 + 1e-9)),
                sd=float(d.std()), p025=float(lo), p975=float(hi))


def sampled_outcomes(res, seed=11):
    """One sampled outcome per patient (for the scoreboard): happy / annoyed / recheck / review / walked out."""
    rng = np.random.default_rng(seed)
    P = np.stack([res["p_happy"], res["p_annoyed"], res["p_recheck"], res["p_review"]], 1)
    u = rng.uniform(size=len(P))[:, None]
    o = (u > np.cumsum(P, 1)).sum(1)
    o = np.where(res["timeout"] > 0.5, 4, o)
    return [int(x) for x in np.bincount(o, minlength=5)]


def ckpt_list():
    """Every practice checkpoint of both terms: (exams, path, rand_dial). Term-1 checkpoints are scored in the
    clinic they practised in (dial at 180)."""
    out = {}
    for run, later in ((RUN1, False), (RUN2, True), (RUN, True)):
        for p in glob.glob(os.path.join(run, "ckpt", "exams_*.pt")):
            n = int(re.findall(r"exams_(\d+)", p)[0])
            if n not in out:
                out[n] = (n, p, later and n > TERM1_END)
    return [out[k] for k in sorted(out)]


def cyl_breakdown(path, n=2000, rand_dial=True):
    """How a checkpoint handles astigmatism, on the normal clinic mix: cross-cylinder questions, axis turns, and
    eyes within 0.25 D by orientation (cylinder 0.75 D or more)."""
    ag, _ = load(path)
    res = run3(agent=ag, test=PlanoPairTestSet(n, seed=SEED), n_par=1000, record_actions=True, rand_dial=rand_dial)
    acts = res["actions"]
    cnt = lambda A: float(np.mean([sum(a in A for a in seq) for seq in acts]))
    tax = np.concatenate([res["tax"], res["tax_os"]]) % 180
    tC = np.concatenate([res["tC"], res["tC_os"]])
    err = np.concatenate([res["err"], res["err_os"]])
    wtr = ((tax < 30) | (tax >= 150)) & (tC <= -0.75)
    atr = (tax >= 60) & (tax < 120) & (tC <= -0.75)
    obl = ~((tax < 30) | (tax >= 150)) & ~((tax >= 60) & (tax < 120)) & (tC <= -0.75)
    s = summary3(res)
    s.update(se_stats(res))
    return dict(summary=s, jcc_axis=cnt((E3.JCC_AX,)), jcc_power=cnt((E3.JCC_PW,)),
                axis_turns=cnt((E3.AXP2, E3.AXM2, E3.AXP5, E3.AXM5, E3.AXP15, E3.AXM15)),
                wtr_within=float(np.mean(err[wtr] <= 0.25 + 1e-9)), atr_within=float(np.mean(err[atr] <= 0.25 + 1e-9)),
                obl_within=float(np.mean(err[obl] <= 0.25 + 1e-9)))


def term1_shortcut(n=2000):
    """The end of term 1: its habits in its own clinic (dial at 180), by astigmatism orientation, and what
    happened when the dial started anywhere."""
    ag, _ = load(os.path.join(RUN1, "ckpt", f"exams_{TERM1_END}.pt"))
    out = {}
    for name, rd in (("dial180", False), ("random_dial", True)):
        res = run3(agent=ag, test=PlanoPairTestSet(n, seed=SEED), n_par=1000, record_actions=True, rand_dial=rd)
        s = summary3(res)
        s.update(se_stats(res))
        h = habits(res["actions"])
        C = np.concatenate([res["C"], res["C_os"]])
        ax = np.concatenate([res["ax"], res["ax_os"]])
        tC = np.concatenate([res["tC"], res["tC_os"]])
        tax = np.concatenate([res["tax"], res["tax_os"]]) % 180
        err = np.concatenate([res["err"], res["err_os"]])
        wtr = ((tax < 30) | (tax >= 150)) & (tC <= -0.75)
        atr = (tax >= 60) & (tax < 120) & (tC <= -0.75)
        obl = ~((tax < 30) | (tax >= 150)) & ~((tax >= 60) & (tax < 120)) & (tC <= -0.75)
        out[name] = dict(summary=s, axis_turns=h["mean_count"]["axis"], cyl_given=float(np.mean(C < 0)),
                         cyl_at_180=float(np.mean(np.abs(((ax[C < 0] + 90) % 180) - 90) < 0.5)) if (C < 0).any() else 0.0,
                         wtr_within=float(np.mean(err[wtr] <= 0.25 + 1e-9)), atr_within=float(np.mean(err[atr] <= 0.25 + 1e-9)),
                         obl_within=float(np.mean(err[obl] <= 0.25 + 1e-9)),
                         n_wtr=int(wtr.sum()), n_atr=int(atr.sum()), n_obl=int(obl.sum()))
        print("term 1", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if k != "summary"},
              flush=True)
    return out


def textbook_cyl(n=2000):
    res = run3(make=textbook3, test=PlanoPairTestSet(n, seed=SEED), n_par=1000, record_actions=True)
    acts = res["actions"]
    cnt = lambda A: float(np.mean([sum(a in A for a in seq) for seq in acts]))
    tax = np.concatenate([res["tax"], res["tax_os"]]) % 180
    tC = np.concatenate([res["tC"], res["tC_os"]])
    err = np.concatenate([res["err"], res["err_os"]])
    wtr = ((tax < 30) | (tax >= 150)) & (tC <= -0.75)
    atr = (tax >= 60) & (tax < 120) & (tC <= -0.75)
    obl = ~((tax < 30) | (tax >= 150)) & ~((tax >= 60) & (tax < 120)) & (tC <= -0.75)
    return dict(jcc_axis=cnt((E3.JCC_AX,)), jcc_power=cnt((E3.JCC_PW,)),
                wtr_within=float(np.mean(err[wtr] <= 0.25 + 1e-9)), atr_within=float(np.mean(err[atr] <= 0.25 + 1e-9)),
                obl_within=float(np.mean(err[obl] <= 0.25 + 1e-9)))


def main(n_test=5000, n_stage=2000, tag="f"):
    os.makedirs(DATA, exist_ok=True)
    R = {"meta": {"test_patients": n_test, "seed": SEED}}
    seqs = {}
    final = os.path.join(RUN, "ckpt", "final.pt")
    if not os.path.exists(final):
        final = ckpt_list()[-1][1]
    R["meta"]["term1_end"] = TERM1_END
    R["meta"]["term2_end"] = TERM2_END
    agent, meta = load(final)
    R["meta"].update(exams=meta["exams"], steps=meta["steps"], ckpt=os.path.basename(final))
    for name, kw in (("plano", dict(make=plano)), ("textbook", dict(make=textbook3)), ("ai", dict(agent=agent))):
        res = run3(test=PlanoPairTestSet(n_test, seed=SEED), n_par=min(1000, n_test), record_actions=name != "plano", **kw)
        s = summary3(res)
        s.update(se_stats(res))
        entry = dict(summary=s, outcomes=sampled_outcomes(res))
        if name != "plano":
            entry["habits"] = habits(res["actions"])
            seqs[name] = res["actions"][:300]
            # repeatability: the same 2,000 patients again with fresh answer noise
            nr = min(2000, n_test)
            r1 = run3(test=PlanoPairTestSet(nr, seed=SEED), n_par=min(1000, nr), seed=7, **kw)
            r2 = run3(test=PlanoPairTestSet(nr, seed=SEED), n_par=min(1000, nr), seed=8, **kw)
            entry["repeat"] = repeat(r1, r2)
        R[name] = entry
        print(name, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, flush=True)
    R["term1"] = term1_shortcut(min(2000, n_test))
    R["term2"] = cyl_breakdown(os.path.join(RUN2, "ckpt", "final.pt"), min(2000, n_test))
    R["final_cyl"] = cyl_breakdown(final, min(2000, n_test))
    R["textbook_cyl"] = textbook_cyl(min(2000, n_test))
    print("term 2 end", {k: round(v, 3) for k, v in R["term2"].items() if k != "summary"})
    print("final", {k: round(v, 3) for k, v in R["final_cyl"].items() if k != "summary"})
    # practice checkpoints (sampled, as it practised; and greedy, its best guess), each in its own term's clinic
    R["stages"] = []
    for ex_n, p, rd in ckpt_list():
        ag, m = load(p)
        row = dict(exams=ex_n, ckpt=os.path.basename(p), term=2 if rd else 1)
        for mode, greedy in (("sampled", False), ("greedy", True)):
            res = run3(agent=ag, test=PlanoPairTestSet(n_stage, seed=SEED), n_par=min(1000, n_stage), greedy=greedy,
                       rand_dial=rd)
            s = summary3(res)
            s.update(se_stats(res))
            row[mode] = s
            row[mode + "_outcomes"] = sampled_outcomes(res)
        R["stages"].append(row)
        print("stage", ex_n, "sampled net", round(row["sampled"]["net"], 1), "greedy net", round(row["greedy"]["net"], 1),
              flush=True)
    json.dump(R, open(os.path.join(DATA, f"results_{tag}.json"), "w"), indent=1)
    json.dump(seqs, open(os.path.join(DATA, f"seqs_{tag}.json"), "w"))
    print(f"wrote data/results_{tag}.json")


def redo(name):
    """Re-run one examiner (textbook or plano) on the test set and patch it into data/results_f.json."""
    p = os.path.join(DATA, "results_f.json")
    R = json.load(open(p))
    seqs = json.load(open(os.path.join(DATA, "seqs_f.json")))
    kw = dict(make=textbook3) if name == "textbook" else dict(make=plano)
    n_test = R["meta"]["test_patients"]
    res = run3(test=PlanoPairTestSet(n_test, seed=SEED), n_par=1000, record_actions=name != "plano", **kw)
    s = summary3(res)
    s.update(se_stats(res))
    entry = dict(summary=s, outcomes=sampled_outcomes(res))
    if name != "plano":
        entry["habits"] = habits(res["actions"])
        seqs[name] = res["actions"][:300]
        r1 = run3(test=PlanoPairTestSet(2000, seed=SEED), n_par=1000, seed=7, **kw)
        r2 = run3(test=PlanoPairTestSet(2000, seed=SEED), n_par=1000, seed=8, **kw)
        entry["repeat"] = repeat(r1, r2)
    R[name] = entry
    json.dump(R, open(p, "w"), indent=1)
    json.dump(seqs, open(os.path.join(DATA, "seqs_f.json"), "w"))
    print(name, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "redo":
        redo(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        main(400, 200, tag="f_quick")
    else:
        main()
