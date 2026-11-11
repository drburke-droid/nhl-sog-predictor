"""
Round-bowl mixing simulation: six powders loaded as pie wedges in a glass
bowl, stirred by one kitchen tool (see tools.py for the tools and how each is
moved).  Same physics as simulate.py.

    python bowl_sim.py --tool spoon
    python bowl_sim.py --tool paddle_large --seconds 20 --grain 1.5
    python bowl_sim.py --tool whisk --speed 2 --no-cache

Writes <out>/particles.pc2 (unless --no-cache), meta.npz and meta.json.
"""
import argparse
import json
import math
import os
import struct
import time

import numpy as np

from pbd import substep
from tools import BOWL, TOOLS, Motion, build_tool

G = 9810.0
HERE = os.path.dirname(os.path.abspath(__file__))


def mix_index(pos, ingr, R, n_ingr, cell=7.5):
    """Same index as simulate.py: spread of composition between full-depth
    sample columns, 0 = as loaded, 1 = indistinguishable from a random mix."""
    nc = int(math.ceil(2 * R / cell))
    ix = np.clip(((pos[:, 0] + R) / cell).astype(np.int64), 0, nc - 1)
    iy = np.clip(((pos[:, 1] + R) / cell).astype(np.int64), 0, nc - 1)
    c = ix * nc + iy
    tot = np.bincount(c, minlength=nc * nc).astype(np.float64)
    ok = tot > 0.35 * tot.max()                    # whole cells only, not rim slivers
    M = []
    for k in range(n_ingr):
        p = (ingr == k).mean()
        if p == 0:                                  # a powder this bowl does not use
            continue
        cnt = np.bincount(c[ingr == k], minlength=nc * nc)
        frac = cnt[ok] / tot[ok]
        s = np.sqrt(np.average((frac - p) ** 2, weights=tot[ok]))
        s0 = np.sqrt(p * (1 - p))
        sr = np.sqrt(p * (1 - p) / tot[ok].mean())
        M.append((s0 - s) / (s0 - sr))
    return float(np.mean(M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=sorted(TOOLS), default="spoon")
    ap.add_argument("--grain", type=float, default=1.8, help="grain size in mm")
    ap.add_argument("--seconds", type=float, default=20.0, help="how long the tool works")
    ap.add_argument("--speed", type=float, default=1.0, help="multiplies the tool's orbit and spin rates")
    ap.add_argument("--load", default="wedges",
                    help="how the powders start out: wedges, stripes, rings, or art:<name> (see patterns.py)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--ingredients", type=int, default=6)
    ap.add_argument("--max-dt", type=float, default=1.0, help="longest solver time step in ms")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "cache", "bowl", args.tool)
    os.makedirs(out, exist_ok=True)

    bowl, d, fps = dict(BOWL), args.grain, args.fps
    R, rf = bowl["R"], bowl["rf"]
    rng = np.random.default_rng(args.seed)
    tool = build_tool(args.tool, d, bowl)
    motion = Motion(tool, args.seconds, args.speed)
    K = sum(len(p["pts"]) for p in tool["parts"])

    # ---- fill ---------------------------------------------------------------
    vol = math.pi * R * R * bowl["bed"] - 0.5 * (2 * math.pi * R) * (rf * rf * (1 - math.pi / 4))
    n_target = int(vol / d ** 3 * 1.02)
    sp = 1.3 * d
    g1 = np.arange(-R + sp / 2, R, sp)
    X, Y = np.meshgrid(g1, g1, indexing="ij")
    keep = (X ** 2 + Y ** 2) < (R - rf * 0.6) ** 2
    X, Y = X[keep], Y[keep]
    layers = int(np.ceil(n_target / len(X)))
    pos = np.concatenate([np.stack([X, Y, np.full_like(X, rf * 0.5 + d + sp * k)], 1) for k in range(layers)])
    pos += rng.uniform(-0.12, 0.12, pos.shape) * d
    N = len(pos)
    n_all = N + K
    P = np.zeros((n_all, 3)); Pprev = np.zeros((n_all, 3)); V = np.zeros((N, 3))
    P[:N] = pos
    P[N:] = 1e6

    h = d * 1.02
    inv_h = 1.0 / h
    ox = oy = -R - 2 * h
    oz = -h
    nx = ny = int(2 * R / h) + 5
    tool_top = max(p["geom"]["H"] for p in tool["parts"])
    nz = int((max(pos[:, 2].max(), bowl["lift"] + tool_top) + 6 * d) / h) + 3
    cell_of = np.zeros(n_all, np.int64)
    start = np.zeros(nx * ny * nz + 1, np.int64)
    order = np.zeros(n_all, np.int64)

    # ---- collider kinematics --------------------------------------------------
    def place(t, sl_out):
        phi, e, angles, pl = motion.at(t)
        cx, cy = math.cos(phi), math.sin(phi)
        zb = bowl["clear"] + bowl["lift"] * (1 - pl)
        k0 = N
        for p, ang in zip(tool["parts"], angles):
            n = len(p["pts"])
            ax, ay = (e + p["du"]) * cx, (e + p["du"]) * cy
            ca, sa = math.cos(ang), math.sin(ang)
            sl_out[k0:k0 + n, 0] = ax + p["pts"][:, 0] * ca - p["pts"][:, 1] * sa
            sl_out[k0:k0 + n, 1] = ay + p["pts"][:, 0] * sa + p["pts"][:, 1] * ca
            sl_out[k0:k0 + n, 2] = zb + p["pts"][:, 2]
            k0 += n
        return pl

    # substeps: short enough that nothing moves more than 0.4 grain, and <= max-dt
    probe = np.zeros((n_all, 3)); probe2 = np.zeros((n_all, 3))
    vmax_tool = 0.0
    for t in np.linspace(1.0, min(args.seconds - 1.0, 8.0), 60):
        place(t, probe); place(t + 1e-3, probe2)
        vmax_tool = max(vmax_tool, float(np.sqrt(((probe2[N:] - probe[N:]) ** 2).sum(1)).max()) / 1e-3)
    nsub = max(int(np.ceil(vmax_tool / fps / (0.4 * d))), int(np.ceil(1000.0 / (fps * args.max_dt))))
    dt = 1.0 / fps / nsub
    print(f"{tool['label']}: grains {N}, collider spheres {K}, rotor radius {tool['e']:.1f} mm, "
          f"fastest point {vmax_tool:.0f} mm/s, substeps/frame {nsub} (dt {dt*1e3:.2f} ms)", flush=True)

    def step(n_active):
        substep(P, Pprev, V, N, n_active, dt, G, d, args.mu, 0.4, 2,
                1, R, rf, 0.0, 0.0, inv_h, ox, oy, oz, nx, ny, nz,
                cell_of, start, order, 4.0, 0.45 * d / dt)

    for f in range(int(1.2 * fps)):                 # let the fill settle
        for s in range(nsub):
            step(N)
    n_ing = args.ingredients
    if args.load.startswith("art:"):                # pixel art laid over the bowl; up to eight powders
        from patterns import LETTERS, paint
        ingr = paint(args.load[4:], P[:N, 0], P[:N, 1], R)
        n_ing = len(LETTERS)
    elif args.load == "wedges":
        ingr = (np.floor((np.arctan2(P[:N, 1], P[:N, 0]) + math.pi) / (2 * math.pi) * n_ing).astype(np.int32)) % n_ing
    elif args.load == "stripes":
        ingr = np.clip(((P[:N, 1] + R) / (2 * R) * n_ing).astype(np.int32), 0, n_ing - 1)
    else:                                           # equal-area rings
        ingr = np.clip((np.sqrt(P[:N, 0] ** 2 + P[:N, 1] ** 2) / R) ** 2 * n_ing, 0, n_ing - 1e-6).astype(np.int32)
    bed_top = float(np.percentile(P[:N, 2], 99) + d / 2)

    # ---- run ------------------------------------------------------------------
    t_settle = 0.6
    n_frames = int(round((args.seconds + t_settle) * fps)) + 1
    pc2 = os.devnull if args.no_cache else os.path.join(out, "particles.pc2")
    fpc = open(pc2, "wb")
    fpc.write(b"POINTCACHE2\0")
    fpc.write(struct.pack("<iiffi", 1, N, 0.0, 1.0, n_frames))
    ch = {k: np.zeros(n_frames) for k in ("t", "mix", "count", "phi", "e", "plunge")}
    ang = np.zeros((n_frames, len(tool["parts"])))
    snap_every = int(round(2.5 * fps))              # grain positions every 2.5 s, for top-down pictures
    snaps = []
    t0 = time.time()
    active = False
    for f in range(n_frames):
        t = f / fps
        if f > 0:
            for s in range(nsub):
                ts = t - 1.0 / fps + (s + 1) * dt
                Pprev[N:] = P[N:]
                pl = place(ts, P)
                if pl > 1e-4:
                    if not active:
                        Pprev[N:] = P[N:]
                        active = True
                    step(n_all)
                else:
                    active = False
                    step(N)
        fpc.write(P[:N].astype("<f4").tobytes())
        if f % snap_every == 0 or f == n_frames - 1:
            snaps.append(P[:N].astype(np.float32).copy())
        phi, e, angles, pl = motion.at(t)
        ch["t"][f], ch["phi"][f], ch["e"][f], ch["plunge"][f] = t, phi, e, pl
        ch["count"][f] = motion.count(t)
        ch["mix"][f] = mix_index(P[:N], ingr, R, n_ing)
        ang[f] = angles
        if f % (5 * fps) == 0 and f:
            print(f"  t={t:5.1f}s  {tool['unit']} {ch['count'][f]:5.1f}  mixed {100*ch['mix'][f]:5.1f}%  "
                  f"max z {P[:N,2].max():.0f}  [{time.time()-t0:.0f}s]", flush=True)
    fpc.close()
    ch["mix_raw"] = ch["mix"].copy()               # against perfectly separated powders
    ch["mix"] = (ch["mix"] - ch["mix"][0]) / (1.0 - ch["mix"][0])   # 0 = the bowl as loaded

    def reach(level):
        i = np.where(ch["mix"] >= level)[0]
        return (float(ch["t"][i[0]]), float(ch["count"][i[0]])) if len(i) else (None, None)

    summary = {f"{int(l*100)}": reach(l) for l in (0.5, 0.8, 0.9, 0.95)}
    np.savez_compressed(os.path.join(out, "meta.npz"), ingredient=ingr, rand=rng.random((N, 4)).astype(np.float32),
                        angles=ang, snaps=np.array(snaps), **ch)
    meta = dict(tool=args.tool, label=tool["label"], unit=tool["unit"], bowl=bowl, grain=d, fps=fps,
                seconds=args.seconds, speed=args.speed, load=args.load, n_grains=int(N), n_frames=n_frames,
                ingredients=n_ing, rotor_radius=tool["e"], spiral=tool["spiral"], orbit=tool["orbit"],
                part_du=[p["du"] for p in tool["parts"]], bed_top=bed_top, dt_ms=dt * 1e3,
                final_mix=float(ch["mix"][-1]), reach=summary)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"{tool['label']}: {100*ch['mix'][-1]:.1f}% after {args.seconds:.0f}s; "
          + "  ".join(f"{k}% at {v[0]:.1f}s" if v[0] is not None else f"{k}% not reached" for k, v in summary.items()))


if __name__ == "__main__":
    main()
