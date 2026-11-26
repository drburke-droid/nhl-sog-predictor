"""Was the coach right never to pull?  Measure 6-on-5 scoring with the trained
skaters (team 0's goalie forced out all game) against normal 5-on-5 play, then
compute the chance a team down one goal ties it with t seconds left."""
import json
import sys

import numpy as np
import torch

from analyze import load_policy
from hockey_env import DT, HockeyVecEnv
from train import COACH_EVERY

torch.set_num_threads(4)
run = sys.argv[1] if len(sys.argv) > 1 else "runs/main"
pol, it, steps = load_policy(f"{run}/latest.pt")


def play(force_pull, n=64, games=2, seed=77):
    gf = ga = secs = 0.0
    for g in range(games):
        env = HockeyVecEnv(n, seed=seed + g)
        o = env.reset()
        want = np.zeros((n, 2), bool)
        for t in range(env.T_steps):
            move, puck, want, _ = pol.act(o, env.t % COACH_EVERY == 0, want)
            if force_pull:
                want[:, 0] = True
                want[:, 1] = False
            o, r, d, ev = env.step(move, puck, want)
        gf += env.score[:, 0].sum(); ga += env.score[:, 1].sum(); secs += n * env.T_steps * DT
    return gf / secs, ga / secs


f5, a5 = play(False)
f6, a6 = play(True)
lam5 = (f5 + a5) / 2
print(f"5-on-5 goals per team-minute {60*lam5:.3f}")
print(f"goalie pulled: goals for per minute {60*f6:.3f}, against {60*a6:.3f}")
out = dict(lam5=lam5, lam6_for=f6, lam6_against=a6, rows=[])
for t in (5, 10, 20, 30, 45, 60, 90):
    p_norm = 1 - np.exp(-lam5 * t)                       # we score before the end (our only way out of -1)
    p_pull = f6 / (f6 + a6) * (1 - np.exp(-(f6 + a6) * t))  # ... and do it before they score into the empty net
    out["rows"].append(dict(t=t, p_tie_5v5=p_norm, p_tie_pulled=p_pull))
    print(f"down 1, {t:3d} s left: chance to tie {p_norm:.3f} at 5-on-5 vs {p_pull:.3f} with the goalie pulled")
json.dump(out, open(f"{run}/coach_check.json", "w"), indent=1)
