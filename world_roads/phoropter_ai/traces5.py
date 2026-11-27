"""Record the showcase patient's version-5 exams (plano start): day one, the final AI, the textbook.

    python traces5.py [show]      -> data/traces/h_day1.json, h_ai.json, h_textbook.json
"""
import os, sys, json
from record5 import record5, show
from clinicians5 import textbook3
from policy import load

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "data", "traces")
RUN = os.path.join(HERE, "runs", "h", "ckpt")


def save(name, ex):
    os.makedirs(TR, exist_ok=True)
    json.dump(ex, open(os.path.join(TR, f"{name}.json"), "w"))
    s = ex["steps"]
    print(f"{name}: {len(s)} buttons, {s[-1]['secs'] / 60:.1f} min, errors {[round(e, 3) for e in s[-1]['err']]}, "
          f"points {s[-1]['points']:+.1f}")


def main():
    ag0, m0 = load(os.path.join(RUN, "exams_0.pt"))
    d1 = record5(ag0, greedy=False, seed=5, max_steps=60)
    d1["exams"] = 0
    save("h_day1", d1)
    final = os.path.join(RUN, "final.pt")
    ag, meta = load(final)
    ex = record5(ag)
    ex["exams"] = meta["exams"]
    save("h_ai", ex)
    save("h_textbook", record5(script=textbook3))
    if len(sys.argv) > 1:
        show(ex)


if __name__ == "__main__":
    main()
