"""The AI examiner: a small actor-critic network, and a runner that scores it on a fixed test set."""
import numpy as np
import torch
import torch.nn as nn
from env import RefractionEnv, N_OBS, N_ACT

HID = 256


def mlp(i, o, h=HID, gain=1.0):
    net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, gain)
    return net


class Agent(nn.Module):
    def __init__(self, n_obs=N_OBS, n_act=N_ACT, flag=None):
        super().__init__()
        self.n_obs, self.n_act = n_obs, n_act
        self.letters = n_act > N_ACT                       # attempt 3: letter chart and projector buttons
        self.flag = (n_obs == N_OBS + 1) if flag is None else flag   # sees "has an autorefractor reading"
        self.pi = mlp(n_obs, n_act, gain=0.01)
        self.v = mlp(n_obs, 1, gain=1.0)

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.pi(obs))

    @torch.no_grad()
    def act(self, obs, greedy=False):
        logits = self.pi(torch.as_tensor(obs))
        if greedy:
            return logits.argmax(-1).numpy()
        return torch.distributions.Categorical(logits=logits).sample().numpy()

    @torch.no_grad()
    def probs(self, obs):
        return torch.softmax(self.pi(torch.as_tensor(obs)), -1).numpy()


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    meta = sd.get("meta", {})
    a = Agent(meta.get("n_obs", N_OBS), meta.get("n_act", N_ACT), meta.get("flag"))
    a.load_state_dict({k: v.float() for k, v in sd["model"].items()})
    a.eval()
    return a, sd.get("meta", {})


def run_policy(agent, test, n_par=1000, seed=7, greedy=True, record_actions=False, t_max=60):
    """Score an agent on a TestSet; returns arrays indexed by patient id (same format as run_scripted)."""
    from clinicians import collect
    from env2 import BinocularEnv, N_ACT2
    if agent.n_act == N_ACT2:                                  # attempt 4: two eyes
        env = BinocularEnv(n_par, seed=seed, sampler=test.sampler)
        env.t_max = max(t_max, 120)
    else:
        env = RefractionEnv(n_par, seed=seed, sampler=test.sampler, flag=agent.flag, letters=agent.letters)
        env.t_max = t_max
    obs = env.obs()
    out, n_real = {}, 0
    acts_log = [[] for _ in range(n_par)]
    seqs = {}
    while n_real < test.size:
        a = agent.act(obs, greedy=greedy)
        pid_now = env.P["pid"].copy()
        obs, _, done, info = env.step(a)
        if record_actions:
            for i in range(n_par):
                acts_log[i].append(int(a[i]))
                if done[i]:
                    if pid_now[i] >= 0:
                        seqs[int(pid_now[i])] = acts_log[i]
                    acts_log[i] = []
        if info:
            for k, v in info.items():
                out.setdefault(k, []).extend(v.tolist())
            n_real += int((info["pid"] >= 0).sum())
    res = collect(out, test.size)
    if record_actions:
        res["actions"] = [seqs[i] for i in range(test.size)]
    return res
