"""The third video's AI (runs/h, env5) as a starting point in the version-6 exam room (env6).

env6 adds seven inputs (the patient's last remark) just before the button mask. The AI's first layer gets seven
new input columns there, set to zero: at the start it ignores the remarks and behaves exactly as it did in the
third video; practice decides what to make of them.

With --nudge: the third video's AI had stopped pressing the cross-cylinder AXIS button altogether (its chance:
under 1 in 100,000 wherever it is allowed), so practice could never find out what it does. The nudge gives that
button the same outputs as the cross-cylinder POWER button, which the AI uses all the time, with a quarter of
its chance. Nothing else changes; practice decides whether it keeps it.

    python init6.py [--nudge] [out]     -> runs/m/init.pt
"""
import os, sys, math, torch
import env5 as E5
from env6 import N_OBS6, N_EXTRA

HERE = os.path.dirname(os.path.abspath(__file__))


def widen(src, dst, nudge=False):
    sd = torch.load(src, map_location="cpu", weights_only=False)
    k = E5.N_ACT3
    n_old = sd["meta"]["n_obs"]
    assert n_old == E5.N_OBS3 and n_old + N_EXTRA == N_OBS6
    model = {}
    for name, w in sd["model"].items():
        if name.endswith("0.weight"):                   # first layer of the policy and of the value estimate
            w = torch.cat([w[:, :n_old - k], torch.zeros(w.shape[0], N_EXTRA, dtype=w.dtype), w[:, n_old - k:]], 1)
        model[name] = w
    if nudge:
        w, b = model["pi.4.weight"].clone(), model["pi.4.bias"].clone()
        w[E5.JCC_AX], b[E5.JCC_AX] = w[E5.JCC_PW], b[E5.JCC_PW] - math.log(4.0)
        model["pi.4.weight"], model["pi.4.bias"] = w, b
    meta = dict(sd["meta"], n_obs=N_OBS6, from_ckpt=os.path.relpath(src, HERE), from_exams=sd["meta"]["exams"],
                nudge=bool(nudge))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(dict(model=model, meta=meta), dst)
    return meta


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else os.path.join(HERE, "runs", "m", "init.pt")
    print(widen(os.path.join(HERE, "runs", "h", "ckpt", "final.pt"), out, nudge="--nudge" in sys.argv))
