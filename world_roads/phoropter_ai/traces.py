"""Record every exam the video plays, on the showcase patient (record.SHOWCASE) -> data/traces/*.json

    python traces.py
"""
import os, sys, json
from policy import load
from record import record, SHOWCASE
from env import DONE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "traces")


def show(tr, tag):
    print(f"== {tag}: {len(tr['steps'])} moves, error {tr['steps'][-1].get('err', float('nan')):.3f} D")
    for k, s in enumerate(tr["steps"]):
        ans = {-1: "", 0: "1", 1: "2", 2: "same", 3: f"logMAR {s['va']:.1f}"}[s["code"]]
        S, C, ax = s["after"]
        print(f"  {k:2d} {s['name']:18s} {ans:12s} -> {S:+.2f} {C:+.2f} x {ax:.0f}")


def save(tr, name):
    json.dump(tr, open(os.path.join(OUT, f"{name}.json"), "w"))
    show(tr, name)


def main():
    os.makedirs(OUT, exist_ok=True)
    # the untrained network: sample its actions (its greedy choice just repeats one button); take the first
    # seed whose exam ends by itself in 10-20 moves, so the clip has a watchable length
    a0, m0 = load(os.path.join(HERE, "runs", "a", "ckpt", "exams_0.pt"))
    for seed in range(100):
        tr = record(a0, greedy=False, seed=seed)
        if 10 <= len(tr["steps"]) <= 20 and tr["steps"][-1]["a"] == DONE:
            break
    tr["meta"], tr["seed"] = m0, seed
    save(tr, "a_0")
    for ck in ("exams_107079", "exams_301936", "exams_1002464"):
        ag, m = load(os.path.join(HERE, "runs", "a", "ckpt", f"{ck}.pt"))
        tr = record(ag, greedy=True)
        tr["meta"] = m
        save(tr, f"a_{ck}")
    p = os.path.join(HERE, "runs", "c", "ckpt", "final.pt")
    if os.path.exists(p):
        ag, m = load(p)
        tr = record(ag, greedy=True)                                   # with the autorefractor reading
        tr["meta"] = m
        save(tr, "c_final")
        noauto = dict(SHOWCASE, S0=0.0, C0=0.0, ax0=180.0, has_reading=0.0)   # same patient, no reading
        tr = record(ag, patient=noauto, greedy=True, t_max=80, max_steps=80)
        tr["meta"] = m
        save(tr, "c_noauto")




def main_d():
    """The two-eye showcase: attempt 4, the textbook script and the optometrist's routine on the same patient."""
    from record import record2
    from clinicians2 import textbook2, mine
    for nm, kw in (("d_textbook", dict(script=textbook2)), ("d_mine", dict(script=mine))):
        tr = record2(**kw)
        json.dump(tr, open(os.path.join(OUT, f"{nm}.json"), "w"))
        print(f"== {nm}: {len(tr['steps'])} moves, errors {tr['steps'][-1]['err']}")
    p = os.path.join(HERE, "runs", "e", "ckpt", "final.pt")
    if os.path.exists(p):
        ag, m = load(p)
        tr = record2(agent=ag)
        tr["meta"] = m
        json.dump(tr, open(os.path.join(OUT, "e_ai.json"), "w"))
        print(f"== e_ai: {len(tr['steps'])} moves, errors {tr['steps'][-1]['err']}")


if __name__ == "__main__":
    if "d" in sys.argv[1:]:
        main_d()
    else:
        main()
