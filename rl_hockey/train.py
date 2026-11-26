"""Self-play PPO for the hockey environment.

Three policies, all shared by both teams (self-play):
  * skater  - one network for all six skater slots; the slot id is an input, so
              slots *may* specialise but nothing tells them to
  * goalie  - scripted by default (see hockey_env); --learned_goalie trains one
              network for both goalies instead
  * coach   - decides once a second whether the goalie should be out; it is the
              only policy rewarded for winning the game (+1 win, 0 tie, -1 loss)

Skaters and goalies are rewarded +1 / -1 for every goal for / against.  For the
first part of training an extra team-level signal (puck possession and puck
progress up the ice) gets learning off the ground; it is annealed to zero, so
the final behaviour is driven by goals only.  The extra signal is shared by
the whole team, so it never says which player should do what.

Usage:  python train.py --run runs/main --hours 3
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from hockey_env import (COACH_DIM, DT, E, N_MOVE, N_PUCK, N_SLOTS, OBS_DIM, SIGN,
                        GOAL_X, HockeyVecEnv)

COACH_EVERY = 10   # decisions (1 s of game time)


# ---------------------------------------------------------------- networks
def _layer(i, o, std=np.sqrt(2)):
    m = nn.Linear(i, o)
    nn.init.orthogonal_(m.weight, std)
    nn.init.zeros_(m.bias)
    return m


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, heads, hidden):
        super().__init__()
        self.heads = list(heads)
        self.pi = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                _layer(hidden, sum(heads), 0.01))
        self.v = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                               _layer(hidden, 1, 1.0))

    def logits(self, x, masks=None):
        out = list(torch.split(self.pi(x), self.heads, -1))
        if masks is not None:
            for h, m in masks.items():
                out[h] = out[h].masked_fill(~m, -1e9)
        return out

    def value(self, x):
        return self.v(x).squeeze(-1)


def sample(logits_list, greedy=False):
    acts, logp = [], 0
    for lg in logits_list:
        lsm = torch.log_softmax(lg, -1)
        if greedy:
            a = lg.argmax(-1)
        else:
            g = -torch.log(-torch.log(torch.rand_like(lg).clamp_(1e-10, 1.0)))
            a = (lg + g).argmax(-1)
        acts.append(a)
        logp = logp + lsm.gather(-1, a[..., None]).squeeze(-1)
    return torch.stack(acts, -1), logp


def evaluate(logits_list, acts):
    """log-prob of the joint action and the entropy of each head."""
    logp, ents = 0, []
    for h, lg in enumerate(logits_list):
        lsm = torch.log_softmax(lg, -1)
        logp = logp + lsm.gather(-1, acts[:, h:h + 1]).squeeze(-1)
        p = lsm.exp()
        ents.append(-(p * lsm.clamp(min=-1e4)).sum(-1))
    return logp, ents


def make_nets(hidden=256):
    sk = ActorCritic(OBS_DIM, [N_MOVE, N_PUCK], hidden)
    gk = ActorCritic(OBS_DIM, [N_MOVE], hidden // 2)
    co = ActorCritic(COACH_DIM, [2], 64)
    with torch.no_grad():
        sk.pi[-1].bias[N_MOVE] = 2.5      # start out mostly keeping the puck
        co.pi[-1].bias[:] = torch.tensor([2.5, -2.5])   # start with goalie in
    return sk, gk, co


class Policy:
    """Acting wrapper used by training and analysis."""

    def __init__(self, sk, gk, co):
        self.sk, self.gk, self.co = sk, gk, co

    @torch.no_grad()
    def act(self, o, coach_turn, want, greedy=False):
        n = o["obs"].shape[0]
        x = torch.from_numpy(o["obs"])
        xs = x[:, :, :N_SLOTS].reshape(-1, OBS_DIM)
        pm = torch.from_numpy(o["puck_mask"]).reshape(-1, N_PUCK)
        a_s, lp_s = sample(self.sk.logits(xs, {1: pm}), greedy)
        v_s = self.sk.value(xs)
        xg = x[:, :, E - 1].reshape(-1, OBS_DIM)
        a_g, lp_g = sample(self.gk.logits(xg), greedy)
        v_g = self.gk.value(xg)
        out = dict(xs=xs, pm=pm, a_s=a_s, lp_s=lp_s, v_s=v_s, xg=xg, a_g=a_g, lp_g=lp_g, v_g=v_g)
        if coach_turn:
            xc = torch.from_numpy(o["coach"]).reshape(-1, o["coach"].shape[-1])
            a_c, lp_c = sample(self.co.logits(xc), greedy)
            out.update(xc=xc, a_c=a_c, lp_c=lp_c, v_c=self.co.value(xc))
            want = a_c[:, 0].numpy().reshape(n, 2).astype(bool)
        move = np.zeros((n, 2, E), np.int64)
        move[:, :, :N_SLOTS] = a_s[:, 0].numpy().reshape(n, 2, N_SLOTS)
        move[:, :, E - 1] = a_g[:, 0].numpy().reshape(n, 2)
        puck = a_s[:, 1].numpy().reshape(n, 2, N_SLOTS)
        return move, puck, want, out


# ---------------------------------------------------------------- PPO
def gae(rew, val, done, boot, gamma, lam):
    """rew/val: [T, M] ; done: [T] episode ended after step t ; boot: [M]"""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros_like(boot)
    for t in reversed(range(T)):
        nonterm = 0.0 if done[t] else 1.0
        nv = boot if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def ppo_update(net, opt, data, args, n_mb, ent_coef):
    keep = data["active"]
    if not keep.all():   # off-ice slots (extra attacker / pulled goalie) carry no learning signal
        data = {k: v[keep] for k, v in data.items()}
    obs, acts, old_lp, adv, ret, m = (data[k] for k in ("obs", "acts", "lp", "adv", "ret", "active"))
    masks = data.get("pmask")
    if m.sum() > 1:
        a_mean, a_std = adv[m].mean(), adv[m].std() + 1e-8
        adv = (adv - a_mean) / a_std
    n = obs.shape[0]
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0}
    cnt = 0
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for mb in perm.chunk(n_mb):
            mm = m[mb].float()
            denom = mm.sum().clamp(min=1.0)
            lg = net.logits(obs[mb], {1: masks[mb]} if masks is not None else None)
            lp, ents = evaluate(lg, acts[mb])
            ratio = (lp - old_lp[mb]).exp()
            a = adv[mb]
            pg = -(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a) * mm).sum() / denom
            v = net.value(obs[mb])
            vf = 0.5 * ((v - ret[mb]) ** 2).mean()
            # each head's entropy is averaged over the samples where it is a real choice
            # (the puck head only matters for the carrier, ~1 skater in 10)
            en = 0
            for h, e in enumerate(ents):
                if h == 1 and masks is not None:
                    real = mm * (masks[mb].sum(-1) > 1).float()
                    en = en + (e * real).sum() / real.sum().clamp(min=1.0)
                else:
                    en = en + (e * mm).sum() / denom
            loss = pg + args.vf_coef * vf - ent_coef * en
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                stats["ent"] += en.item()
                stats["kl"] += (((ratio - 1) - (lp - old_lp[mb])) * mm).sum().item() / denom.item()
                stats["clip"] += ((((ratio - 1).abs() > args.clip).float()) * mm).sum().item() / denom.item()
            cnt += 1
    return {k: v / cnt for k, v in stats.items()}


def best_perm_share(counts):
    """Share of frames where each slot plays 'its' role under the best
    slot->role assignment (1.0 = rigid positions, 0.2 = positionless)."""
    tot = counts.sum()
    if tot == 0:
        return 0.0, list(range(5))
    best, arg = -1, None
    for perm in itertools.permutations(range(5)):
        s = counts[range(5), perm].sum()
        if s > best:
            best, arg = s, perm
    return best / tot, list(arg)


def goal_distance(env):
    """[N,2] distance from the puck to the goal mouth each team attacks."""
    gx = np.array([GOAL_X, -GOAL_X])[None, :]
    return np.sqrt((env.ppos[:, None, 0] - gx) ** 2 + env.ppos[:, None, 1] ** 2)


def closest_skater(env):
    """[N,2] distance from each team's nearest skater to the puck (0 if it has the puck)."""
    d = np.sqrt(((env.pos - env.ppos[:, None, :]) ** 2).sum(-1))
    d = np.where(env.active & (np.arange(2 * E) % E < N_SLOTS)[None], d, 1e3).reshape(env.n, 2, E).min(-1)
    own = np.where(env.owner >= 0, env.owner // E, -1)
    return np.where(own[:, None] == np.arange(2)[None], 0.0, np.minimum(d, 200.0))


def shaping_weight(steps, args):
    if steps <= args.shape_full:
        return 1.0
    if steps >= args.shape_zero:
        return 0.0
    return 1.0 - (steps - args.shape_full) / (args.shape_zero - args.shape_full)


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--rollout", type=int, default=120)
    ap.add_argument("--game_seconds", type=int, default=180)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--max_steps", type=float, default=1e12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--ent_coach", type=float, default=0.003)
    ap.add_argument("--lr_coach", type=float, default=1e-4)
    ap.add_argument("--coach_start", type=float, default=12e6,
                    help="game steps before the coach starts learning (results are noise until the players can play)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--poss_reward", type=float, default=0.0002)
    ap.add_argument("--progress_reward", type=float, default=0.1,
                    help="per 89 ft the puck gets closer to the opponent's goal mouth")
    ap.add_argument("--chase_reward", type=float, default=0.5, help="per 100 ft the team's closest skater gains on a loose puck")
    ap.add_argument("--shape_full", type=float, default=4e6, help="game steps with full shaping")
    ap.add_argument("--shape_zero", type=float, default=12e6, help="game steps when shaping reaches 0")
    ap.add_argument("--no_offside", action="store_true")
    ap.add_argument("--no_icing", action="store_true")
    ap.add_argument("--learned_goalie", action="store_true", help="learn goalie movement instead of the scripted goalie")
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    assert args.rollout % COACH_EVERY == 0
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.run, exist_ok=True)

    sk, gk, co = make_nets(args.hidden)
    opt_s = torch.optim.Adam(sk.parameters(), lr=args.lr, eps=1e-5)
    opt_g = torch.optim.Adam(gk.parameters(), lr=args.lr, eps=1e-5)
    opt_c = torch.optim.Adam(co.parameters(), lr=args.lr_coach, eps=1e-5)
    it, steps = 0, 0
    ck = os.path.join(args.run, "latest.pt")
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        for net, key in ((sk, "sk"), (gk, "gk"), (co, "co")):
            net.load_state_dict(st[key])
        for opt, key in ((opt_s, "opt_s"), (opt_g, "opt_g"), (opt_c, "opt_c")):
            opt.load_state_dict(st[key])
        it, steps = st["it"], st["steps"]
        print(f"resumed {ck} at iteration {it}, {steps/1e6:.1f}M game steps")
    with open(os.path.join(args.run, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    policy = Policy(sk, gk, co)

    env = HockeyVecEnv(args.envs, args.game_seconds, seed=args.seed + 1000 * it,
                       offside=not args.no_offside, icing=not args.no_icing,
                       scripted_goalie=not args.learned_goalie)
    o = env.reset()
    N, T = args.envs, args.rollout
    want = np.zeros((N, 2), bool)
    t_start = time.time()
    log = open(os.path.join(args.run, "log.jsonl"), "a")
    game_stats = []
    pull_log = []

    while time.time() - t_start < args.hours * 3600 and steps < args.max_steps:
        t0 = time.time()
        w = shaping_weight(steps, args)
        B = {k: [] for k in ("xs", "pm", "a_s", "lp_s", "v_s", "xg", "a_g", "lp_g", "v_g",
                             "act", "rew", "done")}
        C = {k: [] for k in ("xc", "a_c", "lp_c", "v_c", "rew", "done")}
        role_counts = np.zeros((5, 5))
        depth = np.zeros((2, N_SLOTS + 1, 2))   # [phase(att/def), slot, (sum,count)]
        coach_rew = np.zeros((N, 2))
        coach_done = False
        for t in range(T):
            coach_turn = env.t % COACH_EVERY == 0
            if coach_turn and t > 0:
                C["rew"].append(coach_rew.reshape(-1).copy())
                C["done"].append(coach_done)
                coach_rew[:] = 0
                coach_done = False
            move, puck, want, out = policy.act(o, coach_turn, want)
            if coach_turn:
                for k in ("xc", "a_c", "lp_c", "v_c"):
                    C[k].append(out[k])
            act = o["active"]
            # ---- role / depth bookkeeping (5v5, live puck)
            live = env.setup == 0
            roles = env.roles()
            for T_ in (0, 1):
                ok = live & ~env.pulled[:, T_]
                if ok.any():
                    r = roles[ok, T_]
                    role_counts += np.bincount((np.arange(5)[None] * 5 + r).ravel(), minlength=25).reshape(5, 5)
            if live.any():
                tf = env.team_frame_positions()
                own = np.where(env.owner >= 0, env.owner // E, -1)
                for T_ in (0, 1):
                    rel = tf[live, T_, :, 0] - (env.ppos[live, 0] * SIGN[T_])[:, None]
                    a_ = env.active[live, T_ * E:(T_ + 1) * E]
                    att = (own[live] == T_)[:, None]
                    for ph, msk in ((0, att & a_), (1, (~att) & a_)):
                        depth[ph, :, 0] += (rel * msk).sum(0)[: N_SLOTS + 1]
                        depth[ph, :, 1] += msk.sum(0)[: N_SLOTS + 1]
            # ---- shaping terms (team level)
            pre_g = goal_distance(env)
            pre_d = closest_skater(env)
            o, r, done, ev = env.step(move, puck, want)
            own = np.where(env.owner >= 0, env.owner // E, -1)
            poss = np.stack([(own == 0).astype(float) - (own == 1), (own == 1).astype(float) - (own == 0)], -1)
            prog = np.where((ev == 1)[:, None], 0.0, (pre_g - goal_distance(env)) / GOAL_X)  # the goal itself is the reward
            chase = np.where((ev == 0)[:, None], (pre_d - closest_skater(env)) / 100.0, 0.0)
            shape = args.poss_reward * poss + args.progress_reward * prog + args.chase_reward * chase
            team_r = r + w * shape
            B["xs"].append(out["xs"]); B["pm"].append(out["pm"]); B["a_s"].append(out["a_s"])
            B["lp_s"].append(out["lp_s"]); B["v_s"].append(out["v_s"])
            B["xg"].append(out["xg"]); B["a_g"].append(out["a_g"]); B["lp_g"].append(out["lp_g"])
            B["v_g"].append(out["v_g"]); B["act"].append(act); B["rew"].append(team_r); B["done"].append(done)
            if done:
                d = env.score[:, 0] - env.score[:, 1]
                coach_rew[:, 0] += np.sign(d)
                coach_rew[:, 1] -= np.sign(d)
                coach_done = True
                gs = {k: v.sum(1).mean() / 2 for k, v in env.stats.items()}   # per team per game
                gs["score_abs_diff"] = float(np.abs(d).mean())
                gs["ties"] = float((d == 0).mean())
                game_stats.append(gs)
                pull_log.extend(env.pull_log)
                o = env.reset()
                want[:] = False
        C["rew"].append(coach_rew.reshape(-1).copy())
        C["done"].append(coach_done)
        steps += N * T
        t_roll = time.time() - t0

        # ---- bootstrap values
        with torch.no_grad():
            x = torch.from_numpy(o["obs"])
            boot_s = sk.value(x[:, :, :N_SLOTS].reshape(-1, OBS_DIM)).numpy()
            boot_g = gk.value(x[:, :, E - 1].reshape(-1, OBS_DIM)).numpy()
            boot_c = co.value(torch.from_numpy(o["coach"]).reshape(-1, COACH_DIM)).numpy()
        rew = np.stack(B["rew"])                         # [T,N,2]
        done_v = np.array(B["done"])
        # skaters: reward of the team, broadcast to its slots
        rs = np.repeat(rew[:, :, :, None], N_SLOTS, 3).reshape(T, -1)
        vs = torch.stack(B["v_s"]).numpy()
        adv_s, ret_s = gae(rs, vs, done_v, boot_s, args.gamma, args.lam)
        rg = rew.reshape(T, -1)
        vg = torch.stack(B["v_g"]).numpy()
        adv_g, ret_g = gae(rg, vg, done_v, boot_g, args.gamma, args.lam)
        rc = np.stack(C["rew"])
        vc = torch.stack(C["v_c"]).numpy()
        adv_c, ret_c = gae(rc, vc, np.array(C["done"]), boot_c, 1.0, args.lam)

        actv = np.stack(B["act"])                        # [T,N,2,7]
        data_s = dict(obs=torch.cat(B["xs"]), acts=torch.cat(B["a_s"]), lp=torch.cat(B["lp_s"]),
                      pmask=torch.cat(B["pm"]), adv=torch.from_numpy(adv_s.reshape(-1)).float(),
                      ret=torch.from_numpy(ret_s.reshape(-1)).float(),
                      active=torch.from_numpy(actv[:, :, :, :N_SLOTS].reshape(-1)))
        data_g = dict(obs=torch.cat(B["xg"]), acts=torch.cat(B["a_g"]), lp=torch.cat(B["lp_g"]),
                      adv=torch.from_numpy(adv_g.reshape(-1)).float(),
                      ret=torch.from_numpy(ret_g.reshape(-1)).float(),
                      active=torch.from_numpy(actv[:, :, :, E - 1].reshape(-1)))
        data_c = dict(obs=torch.cat(C["xc"]), acts=torch.cat(C["a_c"]), lp=torch.cat(C["lp_c"]),
                      adv=torch.from_numpy(adv_c.reshape(-1)).float(),
                      ret=torch.from_numpy(ret_c.reshape(-1)).float(),
                      active=torch.ones(rc.size, dtype=torch.bool))
        t1 = time.time()
        st_s = ppo_update(sk, opt_s, data_s, args, n_mb=8, ent_coef=args.ent)
        st_g = (ppo_update(gk, opt_g, data_g, args, n_mb=8, ent_coef=args.ent) if args.learned_goalie
                else {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0})
        st_c = (ppo_update(co, opt_c, data_c, args, n_mb=2, ent_coef=args.ent_coach)
                if steps >= args.coach_start else {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0})
        t_upd = time.time() - t1
        it += 1

        cons, perm = best_perm_share(role_counts)
        dfc = role_counts.copy()
        df = np.stack([dfc[:, :2].sum(1), dfc[:, 2:].sum(1)], 1)       # D vs F per slot
        best_df = max((df[list(c), 0].sum() + df[[i for i in range(5) if i not in c], 1].sum())
                      for c in itertools.combinations(range(5), 2)) / max(df.sum(), 1)
        rec = dict(it=it, steps=steps, hours=(time.time() - t_start) / 3600, fps=N * T / (t_roll + t_upd),
                   t_roll=t_roll, t_upd=t_upd, shaping=w,
                   role_consistency=cons, role_perm=perm, df_consistency=best_df,
                   role_counts=role_counts.astype(int).tolist(),
                   depth_att=(depth[0, :, 0] / np.maximum(depth[0, :, 1], 1)).round(2).tolist(),
                   depth_def=(depth[1, :, 0] / np.maximum(depth[1, :, 1], 1)).round(2).tolist(),
                   sk=st_s, gk=st_g, co=st_c)
        if game_stats:
            rec["games"] = {k: float(np.mean([g[k] for g in game_stats])) for k in game_stats[0]}
            rec["pulls"] = pull_log[-2000:] if len(pull_log) else []
            game_stats, pull_log = [], []
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 5 == 0 or it == 1:
            g = rec.get("games", {})
            print(f"it {it:5d} | {steps/1e6:7.2f}M steps | {rec['fps']:6.0f} sps | shape {w:.2f} | "
                  f"roles {cons:.2f} D/F {best_df:.2f} | ent {st_s['ent']:.2f} kl {st_s['kl']:.4f} | "
                  f"goals/g {g.get('goals', float('nan')):.2f} shots {g.get('shots', float('nan')):.1f} "
                  f"pass {g.get('passes', float('nan')):.1f}", flush=True)
        if it % args.ckpt_every == 0:
            state = dict(sk=sk.state_dict(), gk=gk.state_dict(), co=co.state_dict(),
                         opt_s=opt_s.state_dict(), opt_g=opt_g.state_dict(), opt_c=opt_c.state_dict(),
                         it=it, steps=steps, args=vars(args))
            torch.save(state, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            torch.save(dict(sk=sk.state_dict(), gk=gk.state_dict(), co=co.state_dict(), it=it, steps=steps),
                       os.path.join(args.run, f"ckpt_{it:05d}.pt"))
    state = dict(sk=sk.state_dict(), gk=gk.state_dict(), co=co.state_dict(),
                 opt_s=opt_s.state_dict(), opt_g=opt_g.state_dict(), opt_c=opt_c.state_dict(),
                 it=it, steps=steps, args=vars(args))
    torch.save(state, ck)
    torch.save(dict(sk=sk.state_dict(), gk=gk.state_dict(), co=co.state_dict(), it=it, steps=steps),
               os.path.join(args.run, f"ckpt_{it:05d}.pt"))
    print("done", it, steps)


if __name__ == "__main__":
    main()
