"""Record the autorefractor-start exams of the showcase patient (data/traces/g_*.json).

The showcase patient's autorefractor readings are the ones from the first video: OD -2.75 -0.75 x 004,
OS -3.00 -0.50 x 002 (the extra minus typical of young eyes).
"""
import os, sys, json
from functools import partial
from record3 import record3, show, SHOWCASE3
from clinicians3 import textbook3
from policy import load
import env3 as E3
from record import fixed_pair

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "data", "traces")


def record_ar(agent=None, script=None, seed=3):
    """record3 with the exam starting at the autorefractor reading."""
    orig = E3.ClinicEnv.__init__

    def init(self, n, seed=0, sampler=None, record=False, start_mode="plano", rand_dial=True):
        orig(self, n, seed=seed, sampler=sampler, record=record, start_mode="objective", rand_dial=rand_dial)
    E3.ClinicEnv.__init__ = init
    try:
        return record3(agent=agent, script=script, patient=SHOWCASE3, seed=seed)
    finally:
        E3.ClinicEnv.__init__ = orig


def save(name, ex):
    os.makedirs(TR, exist_ok=True)
    json.dump(ex, open(os.path.join(TR, f"{name}.json"), "w"))
    s = ex["steps"]
    print(f"{name}: {len(s)} buttons, {s[-1]['secs'] / 60:.1f} min, errors {[round(e, 3) for e in s[-1]['err']]}, "
          f"points {s[-1]['points']:+.1f}")


def main():
    p = os.path.join(HERE, "runs", "f4", "ckpt", "final.pt")
    ag, meta = load(p)
    ex = record_ar(agent=ag)
    ex["exams"] = meta["exams"]
    save("g_ai", ex)
    save("g_textbook", record_ar(script=partial(textbook3, objective=True)))
    if len(sys.argv) > 1:
        show(ex)


if __name__ == "__main__":
    main()
