"""Train the AI examiner with PPO (Schulman et al. 2017) on simulated patients.

    python train.py --steps 60e6            # ~1 h on 4 CPU cores

Writes ckpt/exams_<N>.pt at milestone exam counts (for the video's "after N exams" stages), and
logs/train.csv with a learning curve measured on a fixed 1,000-patient check set (greedy policy).
"""
import os, time, json, argparse, csv
import numpy as np
import torch
from env import RefractionEnv, N_OBS
from policy import Agent, run_policy
from clinicians import summary
from patients import TestSet

HERE = os.path.dirname(os.path.abspath(__file__))
MILESTONES = [0, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 2_000_000, 3_000_000,
              5_000_000, 8_000_000]
CLINIC_MILESTONES = [0, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000,
                     1_000_000, 1_500_000, 2_000_000, 3_000_000, 4_000_000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=60e6)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--cyl_weight", type=float, default=2.0,
                    help="score: 2 counts leftover cylinder in full dioptres; 1 = plain power vectors (attempt 1)")
    ap.add_argument("--start", default="autoref", help="autoref, plano (no autorefractor), or mix (half each)")
    ap.add_argument("--hard_frac", type=float, default=0.0,
                    help="fraction of training with harder practice patients (bigger cylinders, worse axes), fading")
    ap.add_argument("--flag", action="store_true", help="tell the AI whether there is an autorefractor reading")
    ap.add_argument("--letters", action="store_true", help="letter-by-letter projector chart with line buttons")
    ap.add_argument("--binocular", action="store_true", help="two eyes and binocular balance (env2.py)")
    ap.add_argument("--clinic", action="store_true", help="the clinic exam (env3.py): plano start, clinic points")
    ap.add_argument("--init", default=None, help="start from these weights (e.g. the watched-the-textbook network)")
    ap.add_argument("--critic_warmup", type=int, default=0,
                    help="first N updates train only the value estimate, so the starting habits are not disturbed")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from (latest.pt has the optimizer; an exams_*.pt restarts it)")
    ap.add_argument("--t_max", type=int, default=60, help="moves before an exam times out")
    ap.add_argument("--shape_frac", type=float, default=0.0,
                    help="fraction of training with the warmer/colder hint; its weight falls linearly 1 -> 0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    ck = os.path.join(args.out, "ckpt")
    lg = os.path.join(args.out, "logs")
    os.makedirs(ck, exist_ok=True)
    os.makedirs(lg, exist_ok=True)

    from functools import partial
    from patients import sample_patients
    state = {"hard": 0.0}
    if args.clinic:
        global MILESTONES
        MILESTONES = CLINIC_MILESTONES
        from env3 import ClinicEnv, plano_pair, PlanoPairTestSet
        from clinicians3 import run3, summary3
        env = ClinicEnv(args.envs, seed=args.seed + 1, sampler=lambda rng, n: plano_pair(rng, n, hard=state["hard"]))
    elif args.binocular:
        from env2 import BinocularEnv
        from patients import sample_pair, PairTestSet
        from clinicians2 import summary2
        env = BinocularEnv(args.envs, seed=args.seed + 1, sampler=lambda rng, n: sample_pair(rng, n, hard=state["hard"]))
    else:
        env = RefractionEnv(args.envs, seed=args.seed + 1, flag=args.flag, letters=args.letters,
                            sampler=lambda rng, n: sample_patients(rng, n, start=args.start, hard=state["hard"]))
    env.cyl_weight = args.cyl_weight
    if not args.clinic:
        env.t_max = args.t_max
    agent = Agent(env.n_obs, env.n_act, args.flag, masked=args.clinic, kind="clinic" if args.clinic else None)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    E, T = args.envs, args.rollout
    n_up = int(args.steps // (E * T))
    obs = torch.as_tensor(env.obs())
    exams, steps, t0 = 0, 0, time.time()
    ms_done = set()
    start_up = 1
    header = ["update", "steps", "exams", "hours", "shaping", "hard", "train_err", "train_within", "train_len",
              "entropy", "eval_err", "eval_within", "eval_exact", "eval_steps", "eval_over_minus_young", "eval_points",
              "eval_minutes", "train_points", "train_minutes"]
    if args.init:
        sd0 = torch.load(args.init, map_location="cpu", weights_only=False)
        agent.load_state_dict({k: v.float() for k, v in sd0["model"].items()})
        print(f"initialised from {args.init} {sd0.get('meta', {})}", flush=True)
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        agent.load_state_dict({k: v.float() for k, v in sd["model"].items()})
        if "opt" in sd:
            opt.load_state_dict(sd["opt"])
        exams, steps = sd["meta"]["exams"], sd["meta"]["steps"]
        start_up = steps // (E * T) + 1
        ms_done = {m for m in MILESTONES if m <= exams}
        # keep the log up to the resume point; anything after it belonged to the interrupted run
        rows = list(csv.reader(open(os.path.join(lg, "train.csv"))))
        rows = [rows[0]] + [r for r in rows[1:] if int(r[0]) < start_up]
        log = open(os.path.join(lg, "train.csv"), "w", newline="")
        w = csv.writer(log)
        w.writerows(rows)
        print(f"resumed from {args.resume}: update {start_up}, {steps:,} steps, {exams:,} exams", flush=True)
    else:
        log = open(os.path.join(lg, "train.csv"), "w", newline="")
        w = csv.writer(log)
        w.writerow(header)
    check = None

    def save(tag_exams):
        torch.save({"model": {k: v.half() for k, v in agent.state_dict().items()},
                    "meta": {"exams": int(tag_exams), "steps": int(steps), "n_obs": agent.n_obs, "n_act": agent.n_act,
                             "flag": agent.flag, "masked": agent.masked, "kind": agent.kind, "update": int(cur_up[0])}},
                   os.path.join(ck, f"exams_{int(tag_exams)}.pt"))

    def save_latest(up):
        """Every 20 updates: full state (weights + optimizer), so a machine restart loses at most a minute."""
        torch.save({"model": agent.state_dict(), "opt": opt.state_dict(),
                    "meta": {"exams": int(exams), "steps": int(steps), "n_obs": agent.n_obs, "n_act": agent.n_act,
                             "flag": agent.flag, "masked": agent.masked, "kind": agent.kind, "update": up}},
                   os.path.join(ck, "latest.tmp"))
        os.replace(os.path.join(ck, "latest.tmp"), os.path.join(ck, "latest.pt"))

    cur_up = [0]
    if not args.resume:
        save(0)
        ms_done.add(0)
    NO = agent.n_obs
    buf_o = torch.zeros(T, E, NO)
    buf_a = torch.zeros(T, E, dtype=torch.long)
    buf_lp, buf_r, buf_d, buf_v = (torch.zeros(T, E) for _ in range(4))
    for up in range(start_up, n_up + 1):
        frac = 1 - (up - 1) / n_up
        for g in opt.param_groups:
            g["lr"] = args.lr * max(frac, 0.1)
        ent_c = args.ent * (1.0 if frac > 0.5 else max(0.1, frac * 2))
        prog = (up - 1) / n_up
        env.shaping = max(0.0, 1.0 - prog / args.shape_frac) if args.shape_frac > 0 else 0.0
        state["hard"] = max(0.0, 1.0 - prog / args.hard_frac) if args.hard_frac > 0 else 0.0
        env.gamma = args.gamma
        errs, wins, lens, pts, secs = [], [], [], [], []
        cur_up[0] = up
        with torch.no_grad():
            for t in range(T):
                d = agent.dist(obs)
                a = d.sample()
                buf_o[t], buf_a[t], buf_lp[t] = obs, a, d.log_prob(a)
                buf_v[t] = agent.v(obs).squeeze(-1)
                o2, r, done, info = env.step(a.numpy())
                buf_r[t] = torch.as_tensor(r)
                buf_d[t] = torch.as_tensor(done, dtype=torch.float32)
                obs = torch.as_tensor(o2)
                if info:
                    errs += list(info["err"])
                    wins += list(info["err"] <= 0.25)
                    lens += list(info["steps"])
                    if "points" in info:
                        pts += list(info["points"])
                        secs += list(info["secs"])
            last_v = agent.v(obs).squeeze(-1)
        steps += E * T
        exams += len(errs)
        # GAE
        adv = torch.zeros(T, E)
        gae = torch.zeros(E)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nonterm = 1.0 - buf_d[t]
            delta = buf_r[t] + args.gamma * nv * nonterm - buf_v[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_v
        bo, ba, blp = buf_o.reshape(-1, NO), buf_a.reshape(-1), buf_lp.reshape(-1)
        badv, bret = adv.reshape(-1), ret.reshape(-1)
        n = bo.shape[0]
        ents = []
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, args.mb):
                i = perm[s: s + args.mb]
                d = agent.dist(bo[i])
                lp = d.log_prob(ba[i])
                ratio = (lp - blp[i]).exp()
                ad = badv[i]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(0.8, 1.2) * ad).mean()
                vl = 0.5 * (agent.v(bo[i]).squeeze(-1) - bret[i]).pow(2).mean()
                ent = d.entropy().mean()
                loss = pl + 0.5 * vl - ent_c * ent
                if up < start_up + args.critic_warmup:
                    loss = 0.5 * vl                          # warm-up: only the value estimate learns
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
                ents.append(ent.item())

        for m in MILESTONES:
            if m not in ms_done and exams >= m:
                save(exams if m > 0 else 0)
                ms_done.add(m)
        ev = {}
        if up % args.eval_every == 0 or up == 1 or up == n_up:
            if args.clinic:
                ev = summary3(run3(agent=agent, test=PlanoPairTestSet(1000, seed=999), n_par=1000))
            elif args.binocular:
                ev = summary2(run_policy(agent, PairTestSet(1000, seed=999), n_par=1000, t_max=args.t_max))
            else:
                check = TestSet(1000, seed=999, start=args.start if args.start != "mix" else "autoref")
                ev = summary(run_policy(agent, check, n_par=1000, t_max=args.t_max))
        hrs = (time.time() - t0) / 3600
        w.writerow([up, steps, exams, f"{hrs:.4f}", f"{env.shaping:.3f}", f"{state['hard']:.3f}", f"{np.mean(errs):.4f}", f"{np.mean(wins):.4f}",
                    f"{np.mean(lens):.2f}", f"{np.mean(ents):.4f}",
                    *(f"{ev[k]:.4f}" if ev else "" for k in ("mean_err", "within_025", "exact", "steps",
                                                             "over_minus_young")),
                    f"{ev['points']:.2f}" if "points" in ev else "", f"{ev['minutes']:.3f}" if "minutes" in ev else "",
                    f"{np.mean(pts):.2f}" if pts else "", f"{np.mean(secs) / 60:.3f}" if secs else ""])
        log.flush()
        if up % 20 == 0:
            save_latest(up)
        if up % 5 == 0 or ev:
            msg = (f"[shape {env.shaping:.2f}] up {up}/{n_up} steps {steps / 1e6:.1f}M exams {exams:,} {hrs * 60:.1f} min | train err "
                   f"{np.mean(errs):.3f} within {np.mean(wins):.2f} len {np.mean(lens):.1f} ent {np.mean(ents):.2f}")
            if pts:
                msg += f" pts {np.mean(pts):+.1f} min {np.mean(secs) / 60:.1f}"
            if ev:
                if "points" in ev:
                    msg += f" | EVAL pts {ev['points']:+.1f} min {ev['minutes']:.1f} happy {ev['happy']:.2f}"
                msg += f" | EVAL err {ev['mean_err']:.3f} within {ev['within_025']:.3f} exact {ev['exact']:.3f} " \
                       f"steps {ev['steps']:.1f} overminus_y {ev['over_minus_young']:.2f}"
            print(msg, flush=True)
    save(exams)
    torch.save({"model": {k: v.half() for k, v in agent.state_dict().items()},
                "meta": {"exams": int(exams), "steps": int(steps), "n_obs": agent.n_obs, "n_act": agent.n_act,
                         "flag": agent.flag, "masked": agent.masked, "kind": agent.kind}}, os.path.join(ck, "final.pt"))
    json.dump({"exams": exams, "steps": steps, "hours": (time.time() - t0) / 3600, "args": vars(args)},
              open(os.path.join(lg, "train_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
