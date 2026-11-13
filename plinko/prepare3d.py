"""Animation data and the shot list for the Blender edition of the film.

    python3 prepare3d.py   -> out/anim3d/paths.npz, out/anim3d/shots.json

Every chip path comes from physics.py at the tuned bounciness, sampled at 60 Hz so Blender can key subframes
and compute real motion blur. The shot list says, for each shot: camera, length, the 2D crop of the plate
(scale s = fraction of the plate width shown, centre cx, cy), and which chips fly when.
"""
import json
import os

import numpy as np

import board as B
import physics as P

OUT = "out/anim3d"
FPS = 30
HZ = 60
E = json.load(open("results/strategy.json"))["calibrated_e"]
PRM = P.params(e=E)
W, H = B.SLOT_PITCH, B.ROW_SPACING


def path(x0, seed, w0=0.0, prm=PRM, want=None, tries=400, rng=None):
    """A physics path from (x0, 0) sampled at 60 Hz: columns t, x, y, angle."""
    rng = rng or np.random.default_rng(seed)
    for i in range(tries):
        xx = x0 + (rng.normal(0, 0.004) if i else 0.0)
        ww = w0 + (rng.normal(0, 0.5) if i else 0.0)
        slot, rec = P.trajectory(xx, 0.0, 0.0, 0.0, ww, seed=seed + i, prm=prm, fps=HZ)
        if slot > 0 and (want is None or slot in want) and 2.6 < rec[-1, 0] < 5.5:
            return slot, rec[:, :4].astype(np.float64)
    raise RuntimeError("no path")


def coin_path(flips, step=0.42):
    """The textbook chip: half a slot left or right at each of 12 pins, at a steady pace."""
    pts = [(5 * W, 0.0)]
    u = 10
    for r, f in enumerate(flips):
        u += 1 if f else -1
        pts.append((u / 2 * W, (r + 1) * H - 0.12 * H))
    pts.append((pts[-1][0], B.last_row_y() + B.FLOOR_BELOW_LAST_ROW - B.CHIP_DIAMETER / 2))
    rows = []
    t = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(step * HZ)
        for k in range(n):
            f = k / n
            e = f * f * (3 - 2 * f)
            x = a[0] + (b[0] - a[0]) * e
            y = a[1] + (b[1] - a[1]) * (f ** 1.4)
            rows.append((t, x, y, 0.0))
            t += 1 / HZ
    rows.append((t, pts[-1][0], pts[-1][1], 0.0))
    return np.array(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    arrays = {}
    # cold open: lands in a $0 beside the $10,000; outro: lands in the $10,000
    s, arrays["open"] = path(5 * W, 3, want={4, 6})
    s, arrays["outro"] = path(5 * W, 8, want={5})
    # rain: 420 chips from the middle, as in the 2D film's simulation
    rng = np.random.default_rng(42)
    init = P.releases(5, 420, rng)
    rain, slots = [], []
    for i in range(len(init)):
        sl, rec = P.trajectory(*init[i], seed=int(rng.integers(1, 2**31)), prm=PRM, fps=HZ)
        if sl > 0:
            rain.append(rec[:, :4])
            slots.append(sl)
    for i, r in enumerate(rain):
        arrays[f"rain{i:03d}"] = r
    # twins: perfect board, released 0.001 mm apart; pick a pair that shares a long stretch then splits far
    prm0 = P.params(e=E, jitter_deg=0.0, e_jitter=0.0)
    rng = np.random.default_rng(5)
    best = None
    for i in range(1500):
        x0 = B.drop_x(5) + rng.normal(0, 0.004)
        w0 = rng.normal(0, 0.5)
        sa, ra = P.trajectory(x0, 0.0, 0, 0, w0, seed=7, prm=prm0, fps=HZ)
        sb, rb = P.trajectory(x0 + 1e-6, 0.0, 0, 0, w0, seed=7, prm=prm0, fps=HZ)
        if sa == 0 or sb == 0:
            continue
        m = min(len(ra), len(rb))
        sep = np.hypot(ra[:m, 1] - rb[:m, 1], ra[:m, 2] - rb[:m, 2])
        k = int(np.argmax(sep > 0.01))
        frac = ra[k, 2] / B.last_row_y()
        score = abs(sa - sb) * 10 + frac * 5
        if 5 in (sa, sb) and abs(sa - sb) >= 3 and 0.30 < frac < 0.70 and (best is None or score > best[0]):
            best = (score, ra[:, :4], rb[:, :4], sa, sb, k / HZ)
    arrays["twin_a"], arrays["twin_b"] = best[1], best[2]
    twin_split_t = best[5]
    # textbook coin-flip path (the same flips as the 2D film)
    a2 = np.load("out/anim.npz")
    arrays["coin"] = coin_path(a2["flips"])
    np.savez_compressed(f"{OUT}/paths.npz", **arrays)

    rain_ids = [f"rain{i:03d}" for i in range(len(rain))]
    shots = [
        dict(name="open", cam="hero", dur=11.0, crop=[[0, 1.0, 0.50, 0.50], [11.0, 0.80, 0.40, 0.47]],
             chips=[dict(path="open", t0=2.0, appear=0.6, hold=True, linger=99, face="chip_face")]),
        dict(name="measure", cam="front", dur=18.0, crop=[[0, 0.88, 0.50, 0.50]],
             chips=[dict(path=None, still=[5 * W, -0.02 * W, 0.4], appear=11.5, face="chip_face")]),
        dict(name="textbook", cam="front", dur=20.0, crop=[[0, 0.88, 0.50, 0.50]],
             chips=[dict(path="coin", t0=1.5, appear=1.0, hold=True, linger=99, face="chip_face")]),
        dict(name="reality", cam="front", dur=16.0, crop=[[0, 0.88, 0.50, 0.50]], chips=[]),
        dict(name="rain", cam="front", dur=30.0, crop=[[0, 0.88, 0.50, 0.50]],
             chips=[dict(path=pid, t0=3.0 + 0.052 * i, appear=None, linger=0.35, face="chip_face")
                    for i, pid in enumerate(rain_ids)]),
        dict(name="question", cam="top", dur=8.0, crop=[[0, 1.0, 0.5, 0.5], [8.0, 0.86, 0.5, 0.52]],
             chips=[dict(path="twin_a", t0=5.2, appear=0.4, hold=True, linger=99, face="chip_face", depth_offset=0.0004),
                    dict(path="twin_b", t0=5.2, appear=0.4, hold=True, linger=99, face="chip_blue_face")]),
        dict(name="twins", cam="front", dur=14.0, crop=[[0, 0.88, 0.50, 0.50]], slowmo=0.5,
             chips=[dict(path="twin_a", t0=0.6, appear=0.0, hold=True, linger=99, face="chip_face", depth_offset=0.0004),
                    dict(path="twin_b", t0=0.6, appear=0.0, hold=True, linger=99, face="chip_blue_face")]),
        dict(name="zoom", cam="top", dur=24.0, crop=[[0, 0.86, 0.5, 0.52]],
             chips=[dict(path=None, still=[5 * W, 0.0, 0.0], appear=0.0, face="chip_face")]),
        dict(name="strategy", cam="front", dur=24.0, crop=[[0, 0.88, 0.50, 0.50]], chips=[]),
        dict(name="outro", cam="hero", dur=10.0, crop=[[0, 0.80, 0.40, 0.47], [10.0, 1.0, 0.50, 0.50]],
             chips=[dict(path="outro", t0=1.2, appear=0.5, hold=True, linger=99, face="chip_face")]),
    ]
    meta = dict(fps=FPS, hz=HZ, e=E, shots=shots, rain_slots=slots, twin_split_t=twin_split_t,
                twin_slots=[int(best[3]), int(best[4])])
    json.dump(meta, open(f"{OUT}/shots.json", "w"), indent=1)
    print("shots", [(s["name"], s["dur"]) for s in shots], "total", sum(s["dur"] for s in shots))
    print("rain chips", len(rain), "twin slots", best[3], best[4], "split at", round(twin_split_t, 2), "s")


if __name__ == "__main__":
    main()
