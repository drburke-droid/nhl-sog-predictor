"""What the AI does at each practice checkpoint (sampled, as it practised), for the learning-curve captions.

    python stages3.py        -> data/stages_f.json
"""
import os, json, glob, re
import numpy as np
import env3 as E3
from env3 import PlanoPairTestSet
from clinicians3 import run3, summary3
from evaluate3 import habits, se_stats, ckpt_list, SEED
from policy import load

HERE = os.path.dirname(os.path.abspath(__file__))


def main(n=600):
    rows = []
    for ex_n, p in ckpt_list():
        ag, _ = load(p)
        res = run3(agent=ag, test=PlanoPairTestSet(n, seed=SEED), n_par=n, greedy=False, record_actions=True)
        s = summary3(res)
        s.update(se_stats(res))
        h = habits(res["actions"])
        lens_moved = np.array([any(a in (E3.SP25, E3.SM25, E3.SP100, E3.SM100, E3.CYL_MORE, E3.CYL_LESS, E3.OU_P,
                                         E3.OU_M, E3.R_P, E3.L_P) for a in seq) for seq in res["actions"]])
        cyl_given = np.mean(np.r_[res["C"], res["C_os"]] < 0)
        rows.append(dict(exams=ex_n, net=s["net"], minutes=s["minutes"], happy=s["happy"], se=s["se_within_025"],
                         vec=s["vec_within_025"], plano_rx=float(1 - lens_moved.mean()), cyl_given=float(cyl_given),
                         buttons=s["steps"], habits=h["mean_count"], used=h["used_pct"], reached=h["reached_stage"],
                         switches=h["sph_cyl_switches"]))
        r = rows[-1]
        print(f"{ex_n:>9,}  net {r['net']:+6.1f}  min {r['minutes']:4.1f}  happy {r['happy']:.2f}  SE {r['se']:.2f}  "
              f"vec {r['vec']:.2f}  plano {r['plano_rx']:.2f}  cyl {r['cyl_given']:.2f}  "
              f"jcc {r['habits']['cross-cylinder']:4.1f}  bal {r['habits']['balance']:3.1f}  "
              f"chart {r['habits']['chart reading']:4.1f}  1or2 {r['habits']['sphere 1-or-2']:4.1f}  "
              f"both-eyes {r['reached'][2]:.2f}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "data", "stages_f.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
