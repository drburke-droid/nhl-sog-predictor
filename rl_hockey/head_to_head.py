"""Final policy vs an earlier checkpoint, playing on both ends."""
import json
import sys

import numpy as np
import torch

from analyze import load_policy
from hockey_env import HockeyVecEnv
from train import COACH_EVERY

torch.set_num_threads(4)
a_path, b_path = sys.argv[1], sys.argv[2]
A, B = load_policy(a_path)[0], load_policy(b_path)[0]


def games(pol0, pol1, n=64, seed=5):
    env = HockeyVecEnv(n, seed=seed)
    o = env.reset()
    w0 = np.zeros((n, 2), bool); w1 = np.zeros((n, 2), bool)
    for t in range(env.T_steps):
        turn = env.t % COACH_EVERY == 0
        m0, p0, w0, _ = pol0.act(o, turn, w0)
        m1, p1, w1, _ = pol1.act(o, turn, w1)
        move = np.concatenate([m0[:, :1], m1[:, 1:]], 1)
        puck = np.concatenate([p0[:, :1], p1[:, 1:]], 1)
        want = np.stack([w0[:, 0], w1[:, 1]], 1)
        o, r, d, ev = env.step(move, puck, want)
    return env.score.copy()


s1 = games(A, B)               # A is team 0
s2 = games(B, A, seed=6)[:, ::-1]  # A is team 1, flipped so column 0 is A
s = np.concatenate([s1, s2])
d = s[:, 0] - s[:, 1]
res = dict(games=int(len(d)), a_wins=int((d > 0).sum()), ties=int((d == 0).sum()), b_wins=int((d < 0).sum()),
           a_goals=float(s[:, 0].mean()), b_goals=float(s[:, 1].mean()))
print(json.dumps(res))
json.dump(res, open(sys.argv[3], "w"), indent=1) if len(sys.argv) > 3 else None
