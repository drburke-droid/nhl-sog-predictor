"""Attempt 5, stage 1: watch the textbook. Behaviour cloning of a scripted examiner in the two-eye exam room.

The scripted examiner runs on fresh simulated patients. At every move the network sees exactly what the AI
would see (env2 observations) and learns to predict the script's next move (cross-entropy). It never sees the
true prescription. Stage 2 (train.py --init) then practises from there with ordinary reinforcement learning.

    python imitate.py --teacher textbook2 --samples 6e6 --out runs/e
"""
import os, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from env2 import BinocularEnv, DONE
from patients import sample_pair, PairTestSet
from policy import Agent, run_policy
import clinicians2 as C2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="textbook2")
    ap.add_argument("--samples", type=float, default=6e6)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "e"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    teacher = getattr(C2, args.teacher)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    E, T = args.envs, args.rollout
    env = BinocularEnv(E, seed=args.seed, sampler=sample_pair, record=True)
    agent = Agent(env.n_obs, env.n_act, False)
    opt = torch.optim.Adam(agent.pi.parameters(), lr=args.lr)
    gens = [teacher(env, i) for i in range(E)]
    acts = np.array([next(g) for g in gens], np.int64)
    obs = env.obs()
    n_it = int(args.samples // (E * T))
    log = open(os.path.join(args.out, "logs", "imitate.csv"), "w")
    log.write("iter,samples,exams,loss,accuracy\n")
    exams, t0 = 0, time.time()
    for it in range(1, n_it + 1):
        bo = np.zeros((T, E, env.n_obs), np.float32)
        ba = np.zeros((T, E), np.int64)
        for t in range(T):
            bo[t], ba[t] = obs, acts
            obs, _, done, info = env.step(acts)
            last = env.last
            exams += int(done.sum())
            for i in range(E):
                if done[i]:
                    gens[i] = teacher(env, i)
                    acts[i] = next(gens[i])
                else:
                    try:
                        acts[i] = gens[i].send((int(last["code"][i]), float(last["va"][i]), int(last["k"][i]),
                                                int(last["nl"][i]), float(last["line"][i])))
                    except StopIteration:
                        acts[i] = DONE
        X = torch.as_tensor(bo.reshape(-1, env.n_obs))
        Y = torch.as_tensor(ba.reshape(-1))
        losses, accs = [], []
        for _ in range(args.epochs):
            perm = torch.randperm(len(Y))
            for s in range(0, len(Y), args.mb):
                i = perm[s: s + args.mb]
                logits = agent.pi(X[i])
                loss = F.cross_entropy(logits, Y[i])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                accs.append((logits.argmax(-1) == Y[i]).float().mean().item())
        for g in opt.param_groups:
            g["lr"] = args.lr * max(0.1, 1 - it / n_it)
        log.write(f"{it},{it * E * T},{exams},{np.mean(losses):.4f},{np.mean(accs):.4f}\n")
        log.flush()
        if it % 10 == 0 or it == n_it:
            print(f"iter {it}/{n_it} samples {it * E * T / 1e6:.1f}M exams watched {exams:,} loss {np.mean(losses):.3f} "
                  f"move-accuracy {np.mean(accs):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    meta = {"exams": 0, "watched": int(exams), "steps": 0, "n_obs": agent.n_obs, "n_act": agent.n_act, "flag": False,
            "teacher": args.teacher}
    torch.save({"model": {k: v.half() for k, v in agent.state_dict().items()}, "meta": meta},
               os.path.join(args.out, "ckpt", "watched.pt"))
    from clinicians2 import summary2
    r = run_policy(agent, PairTestSet(1000, seed=999), n_par=1000, t_max=120)
    s = summary2(r)
    print("cloned AI on 1,000 check patients:", {k: round(v, 3) for k, v in s.items() if isinstance(v, float)})
    json.dump({"meta": meta, "check": s}, open(os.path.join(args.out, "logs", "imitate_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
