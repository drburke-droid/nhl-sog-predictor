"""The third video's AI (runs/h, env5) as a starting point in the version-6 exam room (env6).

env6 adds seven inputs (the patient's last remark) just before the button mask. The AI's first layer gets seven
new input columns there, set to zero: at the start it ignores the remarks and behaves exactly as it did in the
third video; practice decides what to make of them.

    python init6.py            -> runs/m/init.pt
"""
import os, torch
import env5 as E5
from env6 import N_OBS6, N_EXTRA

HERE = os.path.dirname(os.path.abspath(__file__))


def widen(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=False)
    k = E5.N_ACT3
    n_old = sd["meta"]["n_obs"]
    assert n_old == E5.N_OBS3 and n_old + N_EXTRA == N_OBS6
    model = {}
    for name, w in sd["model"].items():
        if name.endswith("0.weight"):                   # first layer of the policy and of the value estimate
            w = torch.cat([w[:, :n_old - k], torch.zeros(w.shape[0], N_EXTRA, dtype=w.dtype), w[:, n_old - k:]], 1)
        model[name] = w
    meta = dict(sd["meta"], n_obs=N_OBS6, from_ckpt=os.path.relpath(src, HERE), from_exams=sd["meta"]["exams"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(dict(model=model, meta=meta), dst)
    return meta


if __name__ == "__main__":
    print(widen(os.path.join(HERE, "runs", "h", "ckpt", "final.pt"), os.path.join(HERE, "runs", "m", "init.pt")))
