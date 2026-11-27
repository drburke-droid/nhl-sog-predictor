"""Quick check of cylinder refinement for a checkpoint (default: the running term's latest.pt), on the normal
clinic mix: cross-cylinder axis and power questions per exam, axis turns, and how often each kind of
astigmatism ends within 0.25 D.

    python check_cyl.py [checkpoint] [n]
"""
import sys
import numpy as np
import torch
from policy import Agent
from clinicians3 import run3, summary3
from env3 import PlanoPairTestSet, JCC_AX, JCC_PW, AXP2, AXM2, AXP5, AXM5, AXP15, AXM15


def load_any(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m = sd["meta"]
    ag = Agent(m["n_obs"], m["n_act"], m["flag"], True, "clinic")
    ag.load_state_dict({k: v.float() for k, v in sd["model"].items()})
    ag.eval()
    return ag, m


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/f3/ckpt/latest.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    ag, m = load_any(path)
    r = run3(agent=ag, test=PlanoPairTestSet(n, seed=2026), n_par=n, greedy=True, record_actions=True)
    s = summary3(r)
    acts = r["actions"]
    cnt = lambda A: np.mean([sum(a in A for a in seq) for seq in acts])
    tax = np.r_[r["tax"], r["tax_os"]] % 180
    tC = np.r_[r["tC"], r["tC_os"]]
    err = np.r_[r["err"], r["err_os"]]
    wtr = ((tax < 30) | (tax >= 150)) & (tC <= -0.75)
    atr = (tax >= 60) & (tax < 120) & (tC <= -0.75)
    obl = ~((tax < 30) | (tax >= 150)) & ~((tax >= 60) & (tax < 120)) & (tC <= -0.75)
    print(f"{m['exams']:,} patients | JCC axis q {cnt((JCC_AX,)):.2f}  power q {cnt((JCC_PW,)):.2f}  axis turns "
          f"{cnt((AXP2, AXM2, AXP5, AXM5, AXP15, AXM15)):.2f} | within 0.25 D: WTR {np.mean(err[wtr] <= 0.25):.2f} "
          f"ATR {np.mean(err[atr] <= 0.25):.2f} oblique {np.mean(err[obl] <= 0.25):.2f} all {np.mean(err <= 0.25):.2f} | "
          f"net {s['net']:+.1f} min {s['minutes']:.1f}")


if __name__ == "__main__":
    main()
