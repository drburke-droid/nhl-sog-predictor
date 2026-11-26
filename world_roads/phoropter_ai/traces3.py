"""Record the clinic film's exams (data/traces/f_*.json), all on the showcase patient.

    f_day1      the untrained AI (exams_0), sampling buttons as it does in practice
    f_ai        the trained AI (final.pt, or the latest milestone), its best guess at each step
    f_textbook  the textbook script
"""
import os, sys, json, glob, re
from record3 import record3, show
from clinicians3 import textbook3
from policy import load

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "data", "traces")
RUN = os.path.join(HERE, "runs", "f2", "ckpt")
RUN1 = os.path.join(HERE, "runs", "f", "ckpt")


def latest():
    p = os.path.join(RUN, "final.pt")
    if os.path.exists(p):
        return p
    return max(glob.glob(os.path.join(RUN, "exams_*.pt")), key=lambda q: int(re.findall(r"exams_(\d+)", q)[0]))


def save(name, ex):
    os.makedirs(TR, exist_ok=True)
    json.dump(ex, open(os.path.join(TR, f"{name}.json"), "w"))
    s = ex["steps"]
    print(f"{name}: {len(s)} buttons, {s[-1]['secs'] / 60:.1f} min, errors {[round(e, 3) for e in s[-1]['err']]}, "
          f"points {s[-1]['points']:+.1f}")


def main():
    ag0, _ = load(os.path.join(RUN1, "exams_0.pt"))
    save("f_day1", record3(ag0, greedy=False, seed=5, rand_dial=False))   # term 1: the dial started at 180
    p = latest()
    ag, meta = load(p)
    ex = record3(ag, greedy=True)
    ex["ckpt"], ex["exams"] = os.path.basename(p), meta["exams"]
    save("f_ai", ex)
    save("f_textbook", record3(script=textbook3))
    if len(sys.argv) > 1:
        show(ex)


if __name__ == "__main__":
    main()
