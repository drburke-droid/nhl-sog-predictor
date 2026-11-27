"""Record the showcase patient's version-6 exams (plano start, typical reliability): day one, the final AI, the
Borish script.

    python traces6.py [show]      -> data/traces/i_day1.json, i_ai.json, i_borish.json
"""
import os, sys, json
from record5 import record5, show
from clinicians5 import borish
from env6 import ClinicEnv6
from policy import load
from evaluate6 import final_ckpt

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "data", "traces")
RUN = os.path.join(HERE, "runs", "j", "ckpt")
EK = dict(env_cls=ClinicEnv6, dial=1.0)


def save(name, ex):
    os.makedirs(TR, exist_ok=True)
    json.dump(ex, open(os.path.join(TR, f"{name}.json"), "w"))
    s = ex["steps"]
    print(f"{name}: {len(s)} buttons, {s[-1]['secs'] / 60:.1f} min, errors {[round(e, 3) for e in s[-1]['err']]}, "
          f"points {s[-1]['points']:+.1f}, remarks {sum(x.get('remark', -1) >= 0 for x in s)}")


def main():
    ag0, m0 = load(os.path.join(RUN, "exams_0.pt"))
    d1 = record5(ag0, greedy=False, seed=5, max_steps=400, **EK)
    d1["exams"] = 0
    save("i_day1", d1)
    ag, meta = load(final_ckpt())
    ex = record5(ag, **EK)
    ex["exams"] = meta["exams"]
    save("i_ai", ex)
    save("i_borish", record5(script=borish, **EK))
    if len(sys.argv) > 1:
        show(ex)


if __name__ == "__main__":
    main()
