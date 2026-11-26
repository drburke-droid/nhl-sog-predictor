"""Two-eye comparison (attempt 4): the AI, the textbook script, the optometrist's routine, and the autorefractor
alone, on the same 5,000 simulated two-eyed patients.

    python evaluate2.py runs/d/ckpt/final.pt d   -> data/results_d.json, data/errors_d.npz, data/seqs_d.json
"""
import os, sys, json
from collections import Counter
import numpy as np
from patients import PairTestSet
from policy import load, run_policy
from clinicians2 import run2, textbook2, mine, summary2
from env2 import ACTS, N_ACT2
from optics import to_pv, rx_error

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST, SEED = 5000, 2026


def vectors(res):
    """Per-eye error vectors, both eyes pooled: (SE error, 2 dJ0, 2 dJ45)."""
    out = []
    for k in ("", "_os"):
        E = to_pv(res["tS" + k], res["tC" + k], res["tax" + k])
        L = to_pv(res["S" + k], res["C" + k], res["ax" + k])
        out.append(np.stack([L[0] - E[0], 2 * (L[1] - E[1]), 2 * (L[2] - E[2])], 1))
    return np.concatenate(out).astype(np.float32)


def habits(actions):
    count = np.array([[a.count(k) for k in range(N_ACT2)] for a in actions])
    c = Counter(tuple(a) for a in actions)
    seq, n = c.most_common(1)[0]
    return dict(mean_count={ACTS[k]: float(count[:, k].mean()) for k in range(N_ACT2)},
                used_pct={ACTS[k]: float((count[:, k] > 0).mean()) for k in range(N_ACT2)},
                common_seq=list(seq), common_n=n, distinct=len(c))


def main(ck, name):
    agent, meta = load(ck)
    runs = {"ai": run_policy(agent, PairTestSet(N_TEST, SEED), n_par=1000, record_actions=True, t_max=120)}
    runs["textbook"] = run2(textbook2, PairTestSet(N_TEST, SEED), record_actions=True)
    runs["mine"] = run2(mine, PairTestSet(N_TEST, SEED), record_actions=True)
    pool = PairTestSet(N_TEST, SEED).pool
    out, vecs = {"meta": meta}, {}
    for k, r in runs.items():
        out[k] = dict(summary=summary2(r), habits=habits(r["actions"]))
        vecs[k] = vectors(r)
    # the autorefractor alone
    e0 = np.concatenate([runs["ai"]["err0"], runs["ai"]["err0_os"]])
    dM0 = np.concatenate([pool["S0"] + pool["C0"] / 2 - (pool["S"] + pool["C"] / 2),
                          pool["S0_os"] + pool["C0_os"] / 2 - (pool["S_os"] + pool["C_os"] / 2)])
    young = np.concatenate([pool["age"], pool["age"]]) < 40
    out["autoref"] = dict(summary=dict(mean_err=float(e0.mean()), within_025=float(np.mean(e0 <= 0.25 + 1e-9)),
                                       exact=float(np.mean(e0 < 0.13)), steps=0.0,
                                       over_minus_young=float(np.mean(dM0[young] <= -0.25 + 1e-9)),
                                       over_plus=float(np.mean(dM0 >= 0.25 - 1e-9))))
    vr = []
    for k in ("", "_os"):
        E = to_pv(pool["S" + k], pool["C" + k], pool["axis" + k])
        L = to_pv(pool["S0" + k], pool["C0" + k], pool["ax0" + k])
        vr.append(np.stack([L[0] - E[0], 2 * (L[1] - E[1]), 2 * (L[2] - E[2])], 1))
    vecs["autoref"] = np.concatenate(vr).astype(np.float32)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", f"results_{name}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(HERE, "data", f"errors_{name}.npz"), **vecs)
    for k in ("autoref", "textbook", "mine", "ai"):
        s = out[k]["summary"]
        print(f"{k:9s} within {s['within_025']:.3f} exact {s['exact']:.3f} mean err {s['mean_err']:.3f} "
              f"moves {s['steps']:.1f} over-minus(<40) {s['over_minus_young']:.3f} over-plus {s['over_plus']:.3f}"
              + (f" both eyes within {s['both_within']:.3f}" if "both_within" in s else ""))
    h = out["ai"]["habits"]
    print("AI mean uses:", {k: round(v, 2) for k, v in h["mean_count"].items() if v > 0.05})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
