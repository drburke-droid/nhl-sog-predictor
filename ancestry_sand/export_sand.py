"""Export the simulation's sand at given video times for a full Blender render (blender_scene.py --sand).

    python export_sand.py 46 135 165      # -> data/sand_<t>.npz

Every grain becomes a small sphere: resting grains at their place in the pile (map px; height px), flying
grains at their 3D position with a short trail behind them (a motion streak). Colours are linear RGB."""
import os, sys
import numpy as np
import story as S
import sand as Sd
from film import Painter, FPS, COL

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def export(sim, s, path):
    rng = np.random.default_rng(1)
    occ = np.where(sim.h > 0)[0]
    hh = sim.h[occ]
    cell = np.repeat(occ, hh)
    lev = np.concatenate([np.arange(h) for h in hh])
    col = sim.stack[cell, lev].astype(np.int64)
    n = len(cell)
    # stable jitter per grain (cell, level), so piles look the same from frame to frame
    r2 = np.random.default_rng(cell * 31 + lev)
    jit = np.random.default_rng(7).normal(0, 0.55, (Sd.NCELL * Sd.HMAX + 1, 2))[cell * Sd.HMAX + lev]
    xy = Sd.CELL_XY[cell] + jit
    rad = 1.35 + 0.35 * np.random.default_rng(8).random(Sd.NCELL * Sd.HMAX)[cell * Sd.HMAX + lev]
    z = rad + lev * (0.55 if Sd.TOWERS else 1.9)
    tint = (0.86 + 0.24 * np.random.default_rng(9).random(Sd.NCELL))[cell]
    rgb = srgb_to_linear(np.clip(COL[col] * tint[:, None], 0, 1))
    # flying grains, with three trailing points each
    fx, fy, fz, fr, fc = [], [], [], [], []
    for p in sim.particles:
        u = (s - p["t0"]) / p["dur"]
        idx = np.where((u > 0) & (u < 1))[0]
        if not len(idx):
            continue
        for back, shrink in ((0.0, 1.0), (0.012, 0.8), (0.024, 0.6), (0.036, 0.45)):
            uu = np.clip(u[idx] - back, 0, 1)
            x, y, zz, ok, pc = Painter.flight_pos(p, idx, uu)
            ok &= uu > 0.04
            fx.append(x[ok]); fy.append(y[ok]); fz.append(zz[ok] + 2)
            fr.append(np.full(ok.sum(), 1.3 * shrink)); fc.append(COL[p["col"][idx][ok]])
    if fx:
        fx, fy, fz, fr = map(np.concatenate, (fx, fy, fz, fr))
        fc = srgb_to_linear(np.concatenate(fc))
    else:
        fx = fy = fz = fr = np.zeros(0); fc = np.zeros((0, 3))
    np.savez_compressed(path, x=np.concatenate([xy[:, 0], fx]), y=np.concatenate([xy[:, 1], fy]),
                        z=np.concatenate([z, fz]), r=np.concatenate([rad, fr]),
                        c=np.concatenate([rgb, fc]).astype(np.float32), s=s, ybp=Sd.ybp_at(s))
    print(f"{path}: {n} resting grains, {len(fx)} flying points")


def main(times):
    sim = Sd.Sand()
    want = sorted(times)
    k = 0
    for f in range(1, int(S.DURATION * FPS) + 1):
        s = f / FPS
        sim.step(s)
        while k < len(want) and want[k] <= s + 1e-9:
            export(sim, s, os.path.join(DATA, f"sand_{want[k]:g}.npz"))
            k += 1
        if k >= len(want):
            break


if __name__ == "__main__":
    main([float(a) for a in sys.argv[1:]])
