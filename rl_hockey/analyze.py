"""Evaluate trained checkpoints: where does each slot play, and is it a position?

python analyze.py --run runs/main --out runs/main/analysis.json
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import torch

from hockey_env import DT, E, RINK_X, RINK_Y, SIGN, COACH_DIM, HockeyVecEnv, role_assign
from train import COACH_EVERY, Policy, best_perm_share, make_nets

BIN = 4.0
NX, NY = int(2 * RINK_X / BIN), int(np.ceil(2 * RINK_Y / BIN))


def load_policy(path, hidden=256):
    st = torch.load(path, weights_only=False)
    sk, gk, co = make_nets(hidden)
    sk.load_state_dict(st["sk"]); gk.load_state_dict(st["gk"]); co.load_state_dict(st["co"])
    for m in (sk, gk, co):
        m.eval()
    return Policy(sk, gk, co), st.get("it", 0), st.get("steps", 0)


def run_eval(policy, n_envs=64, games=1, seed=123, replay_games=0, max_steps=None):
    env = HockeyVecEnv(n_envs, seed=seed)
    heat = np.zeros((3, E, NX, NY))            # phase: all / attacking / defending
    role_counts = np.zeros((5, 5))
    depth_rank = np.zeros((5, 5))
    rel_sum = np.zeros((3, E, 2)); rel_n = np.zeros((3, E))
    abs_sum = np.zeros((3, E, 2))
    pair_gap = []                                # how far apart are the two deepest vs the forwards
    shape = np.zeros((2, 5)); shape_n = np.zeros(2)   # sorted depth rel. puck, by phase (identity-free)
    df_switch = np.zeros(5); df_frames = np.zeros(5)  # how long a slot keeps its D / F label
    stats, pulls, results, replays = [], [], [], []
    for g in range(games):
        o = env.reset()
        want = np.zeros((n_envs, 2), bool)
        prev_df = np.full((n_envs, 2, 5), -1)
        rec = [dict(frames=[]) for _ in range(replay_games if g == 0 else 0)]
        steps = env.T_steps if max_steps is None else min(env.T_steps, max_steps)
        for t in range(steps):
            coach_turn = env.t % COACH_EVERY == 0
            move, puck, want, _ = policy.act(o, coach_turn, want)
            live = env.setup == 0
            tf = env.team_frame_positions()                    # [N,2,7,2]
            own = np.where(env.owner >= 0, env.owner // E, -1)
            for T in (0, 1):
                act = env.active[:, T * E:(T + 1) * E]
                att = own == T
                dfd = (own >= 0) & ~att
                ptf = env.ppos * SIGN[T]
                for ph, m in enumerate((live, live & att, live & dfd)):
                    mm = m[:, None] & act
                    if not mm.any():
                        continue
                    xy = tf[:, T]
                    ix = np.clip(((xy[..., 0] + RINK_X) / BIN).astype(int), 0, NX - 1)
                    iy = np.clip(((xy[..., 1] + RINK_Y) / BIN).astype(int), 0, NY - 1)
                    sl = np.broadcast_to(np.arange(E), ix.shape)
                    np.add.at(heat[ph], (sl[mm], ix[mm], iy[mm]), 1)
                    rel = xy - ptf[:, None, :]
                    rel_sum[ph] += (rel * mm[..., None]).sum(0)
                    abs_sum[ph] += (xy * mm[..., None]).sum(0)
                    rel_n[ph] += mm.sum(0)
                ok = live & ~env.pulled[:, T]
                if ok.any():
                    rr_ = role_assign(tf[:, T, :5, 0], tf[:, T, :5, 1])
                    dfl = (rr_ < 2).astype(int)
                    sw = ok[:, None] & (prev_df[:, T] >= 0) & (prev_df[:, T] != dfl)
                    df_switch += sw.sum(0)
                    df_frames += np.where(ok[:, None] & (prev_df[:, T] >= 0), 1, 0).sum(0)
                    prev_df[:, T] = np.where(ok[:, None], dfl, -1)
                    for ph, pm in ((0, ok & att), (1, ok & dfd)):
                        if pm.any():
                            relx = np.sort(tf[pm, T, :5, 0] - (env.ppos[pm, 0] * SIGN[T])[:, None], -1)
                            shape[ph] += relx.sum(0)
                            shape_n[ph] += pm.sum()
                    x5, y5 = tf[ok, T, :5, 0], tf[ok, T, :5, 1]
                    r = role_assign(x5, y5)
                    role_counts += np.bincount((np.arange(5)[None] * 5 + r).ravel(), minlength=25).reshape(5, 5)
                    rk = np.argsort(np.argsort(x5, -1), -1)
                    depth_rank += np.bincount((np.arange(5)[None] * 5 + rk).ravel(), minlength=25).reshape(5, 5)
                    xs = np.sort(x5, -1)
                    pair_gap.append(np.stack([xs[:, 1] - xs[:, 0], xs[:, 2] - xs[:, 1],
                                              xs[:, 3] - xs[:, 2], xs[:, 4] - xs[:, 3]], -1).mean(0))
            for i, rr in enumerate(rec):
                rr["frames"].append(dict(
                    p=np.round(env.pos[i] * 10).astype(int).tolist(),
                    a=env.active[i].astype(int).tolist(),
                    k=np.round(env.ppos[i] * 10).astype(int).tolist(),
                    o=int(env.owner[i]), s=env.score[i].tolist(), su=int(env.setup[i] > 0)))
            o, r, done, ev = env.step(move, puck, want)
            for i, rr in enumerate(rec):
                if ev[i]:
                    rr["frames"][-1]["e"] = int(ev[i])
        for i, rr in enumerate(rec):
            rr["final"] = env.score[i].tolist()
            replays.append(rr)
        stats.append({k: v.copy() for k, v in env.stats.items()})
        pulls.extend(env.pull_log)
        results.append(env.score.copy())
    tot = {k: np.concatenate([s[k] for s in stats]) for k in stats[0]}
    per_team = {k: float(v.mean()) for k, v in tot.items()}
    sc = np.concatenate(results)
    game_seconds = env.T_steps * DT
    per_team["save_pct"] = float(tot["saves"].sum() / max(tot["sog"].sum(), 1))
    per_team["shooting_pct"] = float(tot["goals"].sum() / max(tot["shots"].sum(), 1))
    per_team["pass_completion"] = float(tot["pass_ok"].sum() / max(tot["passes"].sum(), 1))
    per_team["games"] = int(sc.shape[0])
    per_team["game_seconds"] = game_seconds
    per_team["tie_rate"] = float((sc[:, 0] == sc[:, 1]).mean())
    cons, perm = best_perm_share(role_counts)
    df = np.stack([role_counts[:, :2].sum(1), role_counts[:, 2:].sum(1)], 1)
    best_df, d_slots = -1, None
    for c in itertools.combinations(range(5), 2):
        v = df[list(c), 0].sum() + df[[i for i in range(5) if i not in c], 1].sum()
        if v > best_df:
            best_df, d_slots = v, c
    out = dict(
        heat=[[heat[ph, s].round().astype(int).tolist() for s in range(E)] for ph in range(3)],
        role_counts=role_counts.astype(int).tolist(),
        role_consistency=cons, role_perm=perm,
        df_consistency=float(best_df / max(df.sum(), 1)), d_slots=list(d_slots),
        depth_rank=depth_rank.astype(int).tolist(),
        rel_puck=(rel_sum / np.maximum(rel_n, 1)[..., None]).round(2).tolist(),
        mean_pos=(abs_sum / np.maximum(rel_n, 1)[..., None]).round(2).tolist(),
        gaps=np.mean(pair_gap, 0).round(2).tolist() if pair_gap else [],
        shape=(shape / np.maximum(shape_n, 1)[:, None]).round(2).tolist(),
        df_run_seconds=(DT / np.maximum(df_switch / np.maximum(df_frames, 1), 1e-9)).round(2).tolist(),
        stats=per_team, pulls=[[round(a, 4), b] for a, b in pulls], replays=replays,
    )
    return out


@torch.no_grad()
def coach_grid(policy, game_seconds=180):
    """P(goalie out) by score difference and time left, own team in possession
    in the offensive zone, play live, goalie currently in."""
    diffs = list(range(-3, 4))
    tls = np.linspace(0, 1, 37)
    rows = []
    for d in diffs:
        x = np.zeros((len(tls), COACH_DIM), np.float32)
        x[:, d + 3] = 1
        x[:, 7] = tls
        x[:, 8] = tls ** 2
        x[:, 12] = 1        # possession: us
        x[:, 14] = 0.5      # puck at the top of the offensive zone
        p = torch.softmax(policy.co.pi(torch.from_numpy(x)), -1)[:, 1].numpy()
        rows.append(p.round(4).tolist())
    return dict(diffs=diffs, seconds_left=(tls * game_seconds).round(1).tolist(), p_pull=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default=None)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--snapshots", type=int, default=6, help="checkpoints sampled for the evolution view")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = {}
    policy, it, steps = load_policy(os.path.join(args.run, "latest.pt"))
    print(f"final policy: iteration {it}, {steps/1e6:.1f}M game steps")
    out["final"] = run_eval(policy, args.envs, args.games, replay_games=args.replays)
    out["final"]["it"], out["final"]["steps"] = it, steps
    out["coach"] = coach_grid(policy)
    print("role consistency", round(out["final"]["role_consistency"], 3), "D/F", round(out["final"]["df_consistency"], 3))
    print("shape (sorted depth rel. puck) attacking", out["final"]["shape"][0], "defending", out["final"]["shape"][1])
    print("sorted-depth gaps", out["final"]["gaps"], " D/F label run (s) per slot", out["final"]["df_run_seconds"])
    print("stats", json.dumps(out["final"]["stats"], indent=None))

    cks = sorted(glob.glob(os.path.join(args.run, "ckpt_*.pt")))
    if cks:
        pick = sorted(set(np.linspace(0, len(cks) - 1, args.snapshots).round().astype(int)))
        out["snapshots"] = []
        for i in pick:
            pol, it_i, st_i = load_policy(cks[i])
            r = run_eval(pol, 32, 1, seed=7, max_steps=900)
            out["snapshots"].append(dict(it=it_i, steps=st_i, heat=r["heat"][0], role_counts=r["role_counts"],
                                         role_consistency=r["role_consistency"], df_consistency=r["df_consistency"],
                                         rel_puck=r["rel_puck"], mean_pos=r["mean_pos"], shape=r["shape"],
                                         df_run_seconds=r["df_run_seconds"], gaps=r["gaps"], stats=r["stats"]))
            print(f"  snapshot it {it_i}: roles {r['role_consistency']:.3f}  D/F {r['df_consistency']:.3f}  gaps {r['gaps']}  runs {r['df_run_seconds']}")
    log = [json.loads(l) for l in open(os.path.join(args.run, "log.jsonl"))]
    out["log"] = [dict(it=r["it"], steps=r["steps"], shaping=r["shaping"], roles=round(r["role_consistency"], 4),
                       df=round(r["df_consistency"], 4), ent=round(r["sk"]["ent"], 3),
                       goals=round(r["games"]["goals"], 3) if "games" in r else None,
                       shots=round(r["games"]["shots"], 2) if "games" in r else None,
                       passes=round(r["games"]["passes"], 2) if "games" in r else None,
                       pulls=round(r["games"]["pulls"], 3) if "games" in r else None)
                  for r in log]
    path = args.out or os.path.join(args.run, "analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


if __name__ == "__main__":
    main()
